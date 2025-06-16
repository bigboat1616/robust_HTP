import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from dataset_jrdb import batch_process_coords, create_dataset, collate_batch
from model_jrdb_traj import create_model
from model_jrdb_original import create_model
from utils.utils import create_logger
from torch.utils.data import DataLoader

def load_model(ckpt_path, device="cpu"):
    """モデルをロード"""
    logger = create_logger('')
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']
    config["DEVICE"] = device
    
    # モデルの種類を判断（configの内容やチェックポイントのパスから判断）
    if "traj" in ckpt_path:
        from model_jrdb_traj import create_model  # 軌跡のみモデル
    else:
        from model_jrdb_original import create_model  # オリジナルモデル（2DBBあり）
    
    model = create_model(config, logger)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, config

def get_predictions(model, config, joints, masks, padding_mask):
    """モデルの予測を取得"""
    padding_mask = padding_mask.to(config["DEVICE"])
    in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(
        joints, masks, padding_mask, config, modality_selection='traj'
    )
    
    with torch.no_grad():
        pred_joints = model(in_joints, padding_mask)
        pred_joints = pred_joints[:,-config['TRAIN']['output_track_size']:]
    
    return in_joints.cpu(), out_joints.cpu(), pred_joints.cpu()

def plot_trajectories(gt_xy, pred_xys, obs_xy, person_id, model_names, save_path="trajectory_plots"):
    """複数モデルの予測を同時プロット"""
    plt.figure(figsize=(10, 10))
    
    # スケール設定
    all_trajectories = [obs_xy, gt_xy] + pred_xys
    all_x = np.concatenate([traj[:,0] for traj in all_trajectories])
    all_y = np.concatenate([traj[:,1] for traj in all_trajectories])
    margin = 0.5
    x_min, x_max = np.min(all_x) - margin, np.max(all_x) + margin
    y_min, y_max = np.min(all_y) - margin, np.max(all_y) + margin
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # 観測軌跡（過去）
    plt.plot(obs_xy[:,0], obs_xy[:,1], 'b.-', label='Observed', alpha=0.5)
    
    # Ground Truth（未来）
    gt_with_start = np.vstack([obs_xy[-1:], gt_xy])
    plt.plot(gt_with_start[:,0], gt_with_start[:,1], 'g.-', label='Ground Truth', alpha=0.5)
    
    # 各モデルの予測
    colors = ['r', 'm', 'c', 'y']  # 必要に応じて色を追加
    for i, pred_xy in enumerate(pred_xys):
        pred_with_start = np.vstack([obs_xy[-1:], pred_xy])
        plt.plot(pred_with_start[:,0], pred_with_start[:,1], 
                f'{colors[i]}.-', label=f'{model_names[i]}', alpha=0.5)
        # 終点をマーク
        plt.plot(pred_xy[-1,0], pred_xy[-1,1], f'{colors[i]}*', 
                label=f'{model_names[i]} End')
    
    
    gt_with_start = np.vstack([obs_xy[-1:], gt_xy])

    
    # 始点と終点
    plt.plot(obs_xy[0,0], obs_xy[0,1], 'ko', label='Start')
    plt.plot(gt_xy[-1,0], gt_xy[-1,1], 'k*', label='GT End')
    
    plt.title(f'Trajectory for Person {person_id}')
    plt.xlabel('X coordinate [m]')
    plt.ylabel('Y coordinate [m]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/trajectory_person_{person_id}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs='+', required=True, help="checkpoint paths")
    parser.add_argument("--model_names", nargs='+', help="names for each model")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="trajectory_plots")
    parser.add_argument("--num_samples", type=int, default=10, help="number of samples to plot")
    args = parser.parse_args()
    
    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # モデル名の設定
    if not args.model_names:
        args.model_names = [f"Model_{i}" for i in range(len(args.ckpts))]
    assert len(args.ckpts) == len(args.model_names), "Number of checkpoints and model names must match"
    
    # モデルのロード
    models_and_configs = [load_model(ckpt, device) for ckpt in args.ckpts]
    models, configs = zip(*models_and_configs)
    
    # データセットの作成
    logger = create_logger('')
    dataset = create_dataset(
        configs[0]['DATA']['train_datasets'][0],
        logger,
        split=args.split,
        track_size=(configs[0]['TRAIN']['input_track_size'] + configs[0]['TRAIN']['output_track_size']),
        track_cutoff=configs[0]['TRAIN']['input_track_size']
    )
    
    # データローダーの設定
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 1サンプルずつ処理
        shuffle=False,
        collate_fn=collate_batch
    )
    
    sample_count = 0
    for batch in dataloader:
        if sample_count >= args.num_samples:
            break
            
        joints, masks, padding_mask = batch
        
        # 各モデルの予測を取得
        all_predictions = []
        for model, config in zip(models, configs):
            in_joints, out_joints, pred = get_predictions(model, config, joints, masks, padding_mask)
            if len(all_predictions) == 0:  # 最初のモデルの時
                obs_xy = in_joints[0,:,0,:2]  # 観測データ
                gt_xy = out_joints[0,:,0,:2]  # Ground Truth
            pred_xy = pred[0].reshape(out_joints.size(1), 1, 2)[:,0,:2]
            all_predictions.append(pred_xy)
        
        # パディングチェック
        if padding_mask[0].sum() == padding_mask[0].shape[0]:
            continue
            
        # プロット
        plot_trajectories(
            gt_xy.numpy(),
            [p.numpy() for p in all_predictions],
            obs_xy.numpy(),
            person_id=sample_count,
            model_names=args.model_names,
            save_path=args.output_dir
        )
        
        sample_count += 1

if __name__ == "__main__":
    main()