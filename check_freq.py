import torch
import numpy as np
import os
import json
from torch.utils.data import DataLoader
from dataset_jrdb import create_dataset, collate_batch, batch_process_coords
from utils.utils import create_logger
from datetime import datetime
from progress.bar import Bar

def analyze_trajectory_frequency(trajectory):
    """
    軌跡の周波数成分を詳細に分析
    """
    fft_result = np.fft.fft(trajectory, axis=0)  # [9, 2]
    power = np.abs(fft_result)**2                # [9, 2]
    power_sum = power.sum(axis=1)                # [9]
    
    # 各周波数成分のパワーを計算
    freq_powers = np.zeros(5)  # DC, 1, 2, 3, 4次
    freq_powers[0] = power_sum[0]  # DC成分
    for i in range(1, 5):
        freq_powers[i] = power_sum[i] + power_sum[-i]
    
    # パワーの合計
    total_power = np.sum(freq_powers)
    
    # 各周波数成分の割合を計算
    freq_ratios = freq_powers / total_power
    
    return {
        'dominant_freq': np.argmax(freq_ratios[1:]) + 1,
        'freq_ratios': freq_ratios,
        'raw_powers': freq_powers
    }

def main():
    # ---- 設定 ----
    logger = create_logger('')
    config = {
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "TRAIN": {
            "input_track_size": 9,
            "output_track_size": 12,
            "batch_size": 1,
            "num_workers": 0
        },
        "DATA": {
            "train_datasets": ["jrdb_2dbox"]
        }
    }

    # ---- 保存用ディレクトリ作成 ----
    save_dir = f"freq_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    # ---- データロード ----
    dataset_name = config['DATA']['train_datasets'][0]
    dataset = create_dataset(
        dataset_name, logger, split="train",
        track_size=(config['TRAIN']['input_track_size'] + config['TRAIN']['output_track_size']),
        track_cutoff=config['TRAIN']['input_track_size']
    )
    dataloader = DataLoader(
        dataset, batch_size=config['TRAIN']['batch_size'],
        num_workers=config['TRAIN']['num_workers'],
        shuffle=False, collate_fn=collate_batch
    )

    # ---- 分割用バケツ作成 ----
    freq_buckets = {i: [] for i in range(5)}  # DC, 1, 2, 3, 4次
    freq_stats = {i: {'count': 0, 'avg_power_ratio': 0} for i in range(5)}

    # ---- データ分割 ----
    bar = Bar('Processing', max=len(dataloader))
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        joints, masks, padding_mask = batch
        in_joints, _, _, _, _ = batch_process_coords(
            joints, masks, padding_mask, config, modality_selection="traj"
        )
        
        traj = in_joints[0, :, 0, :2].cpu().numpy()
        analysis = analyze_trajectory_frequency(traj)
        
        # メタデータと共に保存
        traj_data = {
            "id": total_samples,
            "batch_idx": batch_idx,
            "dominant_freq": int(analysis['dominant_freq']),
            "freq_ratios": analysis['freq_ratios'].tolist(),
            "raw_powers": analysis['raw_powers'].tolist(),
            "trajectory": [
                {"x": float(x), "y": float(y)} 
                for x, y in traj
            ]
        }
        
        freq_buckets[analysis['dominant_freq']].append(traj_data)
        freq_stats[analysis['dominant_freq']]['count'] += 1
        freq_stats[analysis['dominant_freq']]['avg_power_ratio'] += analysis['freq_ratios'][analysis['dominant_freq']]
        
        total_samples += 1
        bar.next()
    
    bar.finish()

    # ---- 結果の表示と保存 ----
    print("\n=== 周波数別サンプル数 ===")
    print(f"総サンプル数: {total_samples}")
    
    freq_names = {
        0: "DC成分",
        1: "1次周波数",
        2: "2次周波数",
        3: "3次周波数",
        4: "4次周波数"
    }
    
    with open(os.path.join(save_dir, 'analysis_results.txt'), 'w') as f:
        f.write(f"総サンプル数: {total_samples}\n\n")
        f.write("=== 周波数別サンプル数 ===\n")
        
        for freq, stats in freq_stats.items():
            if stats['count'] == 0:
                continue
                
            percentage = (stats['count'] / total_samples) * 100
            avg_power = stats['avg_power_ratio'] / stats['count']
            
            result_str = (
                f"{freq_names[freq]}:\n"
                f"  サンプル数: {stats['count']} ({percentage:.1f}%)\n"
                f"  平均パワー比: {avg_power:.3f}\n"
            )
            print(result_str)
            f.write(result_str + '\n')
            
            # NDJSONとして保存
            with open(os.path.join(save_dir, f'freq_{freq}_trajs.ndjson'), 'w') as ndjson_file:
                for traj_data in freq_buckets[freq]:
                    ndjson_file.write(json.dumps(traj_data) + '\n')

    print(f"\n分析結果とデータを {save_dir} に保存しました")

if __name__ == "__main__":
    main()