import argparse
import torch
import random
import numpy as np
from progress.bar import Bar
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
from collections import defaultdict

from dataset_jrdb import batch_process_coords, create_dataset, collate_batch
from model_jrdb_original import create_model
from utils.utils import create_logger

def analyze_trajectory_frequency(trajectory):
    """
    軌跡の周波数成分を分析
    """
    fft_result = np.fft.fft(trajectory, axis=0)
    power = np.abs(fft_result)**2
    power_sum = power.sum(axis=1)
    
    freq_powers = np.zeros(5)  # DC, 1, 2, 3, 4次
    freq_powers[0] = power_sum[0]
    for i in range(1, 5):
        freq_powers[i] = power_sum[i] + power_sum[-i]
    
    total_power = np.sum(freq_powers)
    freq_ratios = freq_powers / total_power
    
    return {
        'dominant_freq': np.argmax(freq_ratios[1:]) + 1,
        'freq_ratios': freq_ratios,
        'raw_powers': freq_powers
    }

def inference(model, config, input_joints, padding_mask, out_len=14):
    model.eval()
    with torch.no_grad():
        pred_joints = model(input_joints, padding_mask)
    output_joints = pred_joints[:, -out_len:]
    return output_joints

def evaluate_ade_fde_by_frequency(model, dataloader, bs, config, logger):
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    bar = Bar(f"EVAL ADE_FDE", fill="#", max=len(dataloader))

    # 周波数ごとの評価結果を保存
    freq_metrics = defaultdict(lambda: {'ade_sum': 0, 'fde_sum': 0, 'count': 0, 'trajectories': []})
    
    for i, batch in enumerate(dataloader):
        joints, masks, padding_mask = batch
        padding_mask = padding_mask.to(config["DEVICE"])
   
        in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(
            joints, masks, padding_mask, config, modality_selection='traj'
        )

        # モデルの予測
        pred_joints = inference(model, config, in_joints, padding_mask, out_len=out_F)

        # 評価用にCPUに移動
        out_joints = out_joints.cpu() 
        pred_joints = pred_joints.cpu().reshape(out_joints.size(0), 12, 1, 2)
        in_joints = in_joints.cpu()

        for k in range(len(out_joints)):
            if padding_mask[k].sum() == padding_mask[k].shape[0]:
                continue

            # 入力軌跡の周波数分析
            input_traj = in_joints[k, :, 0, :2].numpy()
            freq_analysis = analyze_trajectory_frequency(input_traj)
            dominant_freq = freq_analysis['dominant_freq']

            person_out_joints = out_joints[k, :, 0:1]
            person_pred_joints = pred_joints[k, :, 0:1]

            gt_xy = person_out_joints[:, 0, :2]
            pred_xy = person_pred_joints[:, 0, :2]

            if np.any(np.isnan(gt_xy.numpy())) or np.any(np.isnan(pred_xy.numpy())):
                print(f"Warning: NaN values detected for person {k + i*bs}")
                continue
            
            # ADE計算
            sum_ade = 0
            for t in range(12):
                d1 = (gt_xy[t, 0].detach().cpu().numpy() - pred_xy[t, 0].detach().cpu().numpy())
                d2 = (gt_xy[t, 1].detach().cpu().numpy() - pred_xy[t, 1].detach().cpu().numpy())
                dist_ade = [d1, d2]
                sum_ade += np.linalg.norm(dist_ade)
            sum_ade /= 12

            # FDE計算
            d3 = (gt_xy[-1, 0].detach().cpu().numpy() - pred_xy[-1, 0].detach().cpu().numpy())
            d4 = (gt_xy[-1, 1].detach().cpu().numpy() - pred_xy[-1, 1].detach().cpu().numpy())
            dist_fde = [d3, d4]
            scene_fde = np.linalg.norm(dist_fde)

            # 周波数ごとに結果を集計
            freq_metrics[dominant_freq]['ade_sum'] += sum_ade
            freq_metrics[dominant_freq]['fde_sum'] += scene_fde
            freq_metrics[dominant_freq]['count'] += 1
            freq_metrics[dominant_freq]['trajectories'].append((gt_xy, pred_xy))

        bar.next()

    bar.finish()

    # 結果の集計と表示
    results = {}
    for freq, metrics in freq_metrics.items():
        if metrics['count'] > 0:
            results[freq] = {
                'ade': metrics['ade_sum'] / metrics['count'],
                'fde': metrics['fde_sum'] / metrics['count'],
                'count': metrics['count'],
                'trajectories': metrics['trajectories']
            }

    return results

def plot_trajectories(trajectories, freq_name):
    plt.figure(figsize=(10, 6))
    for gt_xy, pred_xy in trajectories:
        plt.plot(gt_xy[:, 0], gt_xy[:, 1], 'g-o', label='Ground Truth', alpha=0.5)
        plt.plot(pred_xy[:, 0], pred_xy[:, 1], 'r-o', label='Prediction', alpha=0.5)
    plt.title(f"Trajectories for {freq_name}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint path")
    parser.add_argument("--split", type=str, default="test", help="Split to use")
    args = parser.parse_args()

    # 設定の読み込み
    logger = create_logger('')
    logger.info(f'Loading checkpoint from {args.ckpt}') 
    ckpt = torch.load(args.ckpt, map_location=torch.device('cpu'))
    config = ckpt['config']
    
    if torch.cuda.is_available():
        config["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
        torch.cuda.manual_seed(0)
    else:
        config["DEVICE"] = "cpu"

    # モデルの初期化
    model = create_model(config, logger)
    model.load_state_dict(ckpt['model'])

    # データのロード
    dataset = create_dataset(
        config['DATA']['train_datasets'][0], 
        logger, 
        split=args.split,
        track_size=(config['TRAIN']['input_track_size'] + config['TRAIN']['output_track_size']),
        track_cutoff=config['TRAIN']['input_track_size']
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['TRAIN']['batch_size'],
        num_workers=config['TRAIN']['num_workers'],
        shuffle=False,
        collate_fn=collate_batch
    )

    # 周波数ごとの評価
    results = evaluate_ade_fde_by_frequency(model, dataloader, config['TRAIN']['batch_size'], config, logger)

    # 結果の表示
    print("\n=== 周波数別評価結果 ===")
    freq_names = {
        0: "DC成分",
        1: "1次周波数",
        2: "2次周波数",
        3: "3次周波数",
        4: "4次周波数"
    }
    
    for freq, metrics in results.items():
        print(f"\n{freq_names[freq]}:")
        print(f"  サンプル数: {metrics['count']}")
        print(f"  ADE: {metrics['ade']:.4f}")
        print(f"  FDE: {metrics['fde']:.4f}")

    # 3次周波数の軌跡をプロット
        if 3 in freq_names:
            plot_trajectories(results[3]['trajectories'], freq_names[3])

    # 結果の保存
    # save_path = f"frequency_analysis_{args.split}.json"
    # with open(save_path, 'w') as f:
    #     json.dump(results, f, indent=2)
    # print(f"\n結果を {save_path} に保存しました")

if __name__ == "__main__":
    main()