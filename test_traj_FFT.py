import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_jrdb import batch_process_coords, create_dataset, collate_batch
from model_jrdb import create_model
from utils.utils import create_logger
import os

# def apply_fft_ifft(trajectory):
#     """ FFT適用後、IFFTで戻す """
#     print("Original trajectory shape:", trajectory.shape)
#     freq_domain = torch.fft.fft(trajectory, dim=1)
#     reconstructed = torch.fft.ifft(freq_domain, dim=1).real  # 実部を取り出す
#     return reconstructed

# def apply_fft_filter_ifft(trajectory, num_freqs=1):
#     """ 周波数成分を制限して軌跡を再構成
#     Args:
#         trajectory: 入力軌跡 [9, 2]
#         num_freqs: 使用する周波数成分の数（1～5）
#                   1: DC成分のみ（平均位置）
#                   2: DC + 最も低い周波数1つ
#                   3: DC + 低周波2つ
#                   4: DC + 低周波3つ
#                   5: 全ての周波数（原軌跡と同じ）
#     """
#     # FFTを適用
#     freq_domain = torch.fft.fft(trajectory, dim=0)
    
#     # 周波数成分を制限
#     mask = torch.zeros_like(freq_domain)
#     if num_freqs > 0:
#         # DC成分は必ず使用
#         mask[0] = 1
#         # 低周波から順に追加（共役対称性を保持）
#         for i in range(1, num_freqs):
#             mask[i] = 1
#             mask[-i] = 1
    
#     # マスクを適用して逆変換
#     freq_filtered = freq_domain * mask
#     reconstructed = torch.fft.ifft(freq_filtered, dim=0).real
    
#     return reconstructed

# def plot_trajectories(gt_xy, pred_xy_fft_filtered_dict, batch_idx, data_num):
#     """ 軌跡の可視化（複数のkeep_fraction） """
#     plt.figure(figsize=(10, 8))
    
#     # Ground Truth
#     plt.plot(gt_xy[:, 0], gt_xy[:, 1], 'k-o', label="Ground Truth", alpha=0.7, linewidth=2)
#     # GTの始点と終点を強調
#     plt.plot(gt_xy[0, 0], gt_xy[0, 1], 'k*', markersize=15, label="Start")
#     plt.plot(gt_xy[-1, 0], gt_xy[-1, 1], 'k^', markersize=15, label="GT Goal")
    
#     # 異なるkeep_fractionの結果
#     colors = ['r', 'g', 'b', 'm']  # 各keep_fractionに対する色
#     for i, (keep_frac, pred_xy) in enumerate(pred_xy_fft_filtered_dict.items()):
#         # 軌跡
#         plt.plot(pred_xy[:, 0], pred_xy[:, 1], 
#                 f'{colors[i]}--o', 
#                 label=f"FFT (keep={keep_frac})", 
#                 alpha=0.7)
#         # 各フィルタ結果の終点を強調
#         plt.plot(pred_xy[-1, 0], pred_xy[-1, 1], 
#                 f'{colors[i]}^', markersize=12,
#                 label=f"FFT Goal (keep={keep_frac})")


#     plt.xlabel("X Coordinate [m]")
#     plt.ylabel("Y Coordinate [m]")
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.title("Trajectory Comparison with Different FFT Filters")
#     plt.grid(True)
#     plt.gca().set_aspect('equal', adjustable='box')  # アスペクト比を1:1に
    
#     os.makedirs("data/fft_plots_compare", exist_ok=True)
#     plt.savefig(f"data/fft_plots_compare/trajectory_{batch_idx}_{data_num}.png", 
#                 dpi=300, bbox_inches='tight')
#     plt.close()

# # ----------------
# # 設定の読み込み
# # ----------------
# logger = create_logger('')
# config = {
#     "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
#     "TRAIN": {
#         "input_track_size": 9,
#         "output_track_size": 12,
#         "batch_size": 1,  # バッチサイズ
#         "num_workers": 0
#     },
#     "DATA": {
#         "train_datasets": ["jrdb_2dbox"]
#     }
# }

# # ----------------
# # データのロード
# # ----------------
# dataset_name = config['DATA']['train_datasets'][0]
# dataset = create_dataset(dataset_name, logger, split="test",
#                          track_size=(config['TRAIN']['input_track_size'] + config['TRAIN']['output_track_size']),
#                          track_cutoff=config['TRAIN']['input_track_size'])

# dataloader = DataLoader(dataset, batch_size=config['TRAIN']['batch_size'], 
#                         num_workers=config['TRAIN']['num_workers'], 
#                         shuffle=False, collate_fn=collate_batch)

# # ----------------
# # サンプルデータの取得
# # ----------------
# max_samples = 100
# data_num = 0
# keep_freqs = [1, 2, 3, 4]  # 使用する周波数成分の数

# for i, batch in enumerate(dataloader):
#     if data_num >= max_samples:
#         break
        
#     joints, masks, padding_mask = batch
#     padding_mask = padding_mask.to(config["DEVICE"])
   
#     in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(
#         joints, masks, padding_mask, config, modality_selection="traj+2dbox"
#     )

#     B, in_F, J, K = in_joints.shape
#     if J % 2 == 0:
#         traj_out_joints = in_joints[:, :, ::2, :2]
#     else:
#         raise ValueError("J must be even to separate traj and 2dbox.")

#     gt_xy = traj_out_joints[0, :, 0, :].cpu().numpy()
    
#     # 異なるkeep_fractionでの結果を計算
#     pred_xy_dict = {}
#     for keep_freq in keep_freqs:
#         pred_xy_filtered = apply_fft_filter_ifft(
#             torch.tensor(gt_xy), 
#             num_freqs=keep_freq
#         )
#         pred_xy_dict[keep_freq] = pred_xy_filtered

#     # プロット
#     plot_trajectories(gt_xy, pred_xy_dict, i, data_num)
#     data_num += 1

# print(f"Plotted {data_num} samples")

def visualize_trajectory_fft(trajectory, title="Trajectory and FFT Analysis"):
    """軌跡データとそのFFT結果の可視化
    Args:
        trajectory: shape [T, 2] のx-y座標データ
    """
    # FFT計算
    fft_result = np.fft.fft(trajectory, axis=0)
    
    # プロット準備
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 元の軌跡
    ax1 = plt.subplot(231)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-o')
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'g*', label='Start', markersize=15)
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', label='End', markersize=15)
    ax1.set_title('Original Trajectory')
    ax1.grid(True)
    ax1.legend()
    
    # 2. 時系列データ
    ax2 = plt.subplot(232)
    ax2.plot(trajectory[:, 0], 'r-', label='x')
    ax2.plot(trajectory[:, 1], 'b-', label='y')
    ax2.set_title('Time Series Data')
    ax2.grid(True)
    ax2.legend()
    
    # 3-6. FFTの実部・虚部
    # X方向
    ax3 = plt.subplot(234)
    ax3.stem(np.real(fft_result[:, 0]))
    ax3.set_title("Real part of FFT (x)")
    ax3.grid(True)
    
    ax4 = plt.subplot(235)
    ax4.stem(np.imag(fft_result[:, 0]))
    ax4.set_title("Imag part of FFT (x)")
    ax4.grid(True)
    
    # Y方向
    ax5 = plt.subplot(233)
    ax5.stem(np.real(fft_result[:, 1]))
    ax5.set_title("Real part of FFT (y)")
    ax5.grid(True)
    
    ax6 = plt.subplot(236)
    ax6.stem(np.imag(fft_result[:, 1]))
    ax6.set_title("Imag part of FFT (y)")
    ax6.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def reconstruct_with_single_frequency(trajectory, freq_idx):
    """特定の周波数成分だけで軌跡を再構成
    Args:
        trajectory: 入力軌跡 [T, 2]
        freq_idx: 使用する周波数成分のインデックス（0: DC成分, 1: 最も低い周波数, ...）
    Returns:
        再構成された軌跡
    """
    # FFTを適用
    fft_result = np.fft.fft(trajectory, axis=0)
    
    # マスクを作成（指定された周波数成分のみ使用）
    mask = np.zeros_like(fft_result, dtype=bool)
    mask[freq_idx] = True
    # 共役対称性を保持するため、対応する負の周波数も使用
    if freq_idx > 0:
        mask[-freq_idx] = True
    
    # マスクを適用して逆変換
    fft_filtered = fft_result.copy()
    fft_filtered[~mask] = 0
    reconstructed = np.fft.ifft(fft_filtered, axis=0).real
    
    return reconstructed

def reconstruct_without_high_frequencies_improved(trajectory, num_removed):
    """高周波成分を1つずつ削除して軌跡を再構成（改善版）
    Args:
        trajectory: 入力軌跡 [9, 2]
        num_removed: 削除する高周波成分の数
    Returns:
        再構成された軌跡
    """
    # FFTを適用
    fft_result = np.fft.fft(trajectory, axis=0)
    n = len(trajectory)  # n = 9
    half_n = (n - 1) // 2  # (9-1)/2 = 4
    
    # マスクを作成（高周波成分を削除）
    mask = np.ones_like(fft_result, dtype=bool)
    if num_removed > 0:
        for i in range(num_removed):
            # 正の周波数側の高周波を削除
            pos_idx = half_n - i  # 4→3→2→1（高周波から低周波へ）
            # 負の周波数側の高周波を削除
            neg_idx = half_n + 1 + i  # 5→6→7→8（高周波から低周波へ）
            
            if pos_idx > 0:  # DC成分は保持
                mask[pos_idx] = False
                mask[neg_idx] = False
    
    # マスクを適用して逆変換
    fft_filtered = fft_result.copy()
    fft_filtered[~mask] = 0
    reconstructed = np.fft.ifft(fft_filtered, axis=0).real
    
    return reconstructed

def visualize_frequency_components(trajectory, title="Frequency Component Analysis"):
    """各周波数成分の影響を可視化
    Args:
        trajectory: 入力軌跡 [T, 2]
    """
    # プロット準備
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 元の軌跡
    ax1 = plt.subplot(231)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'k-o', label='Original', linewidth=2)
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'g*', label='Start', markersize=15)
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', label='End', markersize=15)
    ax1.set_title('Original Trajectory')
    ax1.grid(True)
    ax1.legend()
    
    # 2-5. 各周波数成分の影響
    colors = ['r', 'g', 'b', 'm']
    for i in range(4):
        ax = plt.subplot(2, 3, i+3)
        # 特定の周波数成分だけで再構成
        reconstructed = reconstruct_with_single_frequency(trajectory, i)
        ax.plot(reconstructed[:, 0], reconstructed[:, 1], f'{colors[i]}--o', 
                label=f'Freq {i}', alpha=0.7)
        ax.plot(reconstructed[0, 0], reconstructed[0, 1], f'{colors[i]}*', 
                markersize=10, label='Start')
        ax.plot(reconstructed[-1, 0], reconstructed[-1, 1], f'{colors[i]}^', 
                markersize=10, label='End')
        ax.set_title(f'Reconstruction with Freq {i}')
        ax.grid(True)
        ax.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def visualize_high_frequency_removal(trajectory, title="High Frequency Removal Analysis"):
    """高周波成分を1つずつ削除した影響を可視化
    Args:
        trajectory: 入力軌跡 [T, 2]
    """
    # プロット準備
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 元の軌跡
    ax1 = plt.subplot(231)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'k-o', label='Original', linewidth=2)
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'g*', label='Start', markersize=15)
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', label='End', markersize=15)
    ax1.set_title('Original Trajectory')
    ax1.grid(True)
    ax1.legend()
    
    # 2-5. 高周波成分を1つずつ削除した影響
    colors = ['r', 'g', 'b', 'm']
    for i in range(4):
        ax = plt.subplot(2, 3, i+3)
        # 高周波成分を削除して再構成
        reconstructed = reconstruct_without_high_frequencies_improved(trajectory, i)
        ax.plot(reconstructed[:, 0], reconstructed[:, 1], f'{colors[i]}--o', 
                label=f'Remove {i} high freqs', alpha=0.7)
        ax.plot(reconstructed[0, 0], reconstructed[0, 1], f'{colors[i]}*', 
                markersize=10, label='Start')
        ax.plot(reconstructed[-1, 0], reconstructed[-1, 1], f'{colors[i]}^', 
                markersize=10, label='End')
        ax.set_title(f'Remove {i} High Frequencies')
        ax.grid(True)
        ax.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def analyze_batch_trajectories(dataloader, config, num_samples=5):
    """バッチからサンプルを取り出して分析"""
    for i, batch in enumerate(dataloader):
        if i >= 10:  # 1バッチのみ処理
            break
            
        joints, masks, padding_mask = batch
        in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(
            joints, masks, padding_mask, config
        )
        
        # バッチから複数の軌跡を分析
        for j in range(min(num_samples, in_joints.shape[0])):  # out_joints → in_joints
            if in_joints.shape[2] % 2 == 0:  # 偶数の場合は2dboxデータも含まれている
                trajectory = in_joints[j, :, ::2, :2].cpu().numpy()  # 軌跡データのみ抽出
            else:
                trajectory = in_joints[j, :, :, :2].cpu().numpy()
            
            trajectory = trajectory[:, 0, :]  # [9, 1, 2] → [9, 2]
            print("Input trajectory shape:", trajectory.shape)  # デバッグ用
            
            # 元のFFT分析
            visualize_trajectory_fft(trajectory, f"Trajectory Analysis - Sample {j+1}")
            
            # 各周波数成分の影響を可視化
            visualize_frequency_components(trajectory, f"Frequency Components - Sample {j+1}")
            
            # 高周波成分を1つずつ削除した影響を可視化
            visualize_high_frequency_removal(trajectory, f"High Frequency Removal - Sample {j+1}")

# 使用例
# 設定の読み込み
logger = create_logger('')
config = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "TRAIN": {
        "input_track_size": 9,
        "output_track_size": 12,
        "batch_size": 1,  # バッチサイズ
        "num_workers": 0
    },
    "DATA": {
        "train_datasets": ["jrdb_2dbox"]
    }
}

# データのロード
dataset_name = config['DATA']['train_datasets'][0]
dataset = create_dataset(dataset_name, logger, split="val",
                         track_size=(config['TRAIN']['input_track_size'] + config['TRAIN']['output_track_size']),
                         track_cutoff=config['TRAIN']['input_track_size'])

val_dataloader = DataLoader(dataset, batch_size=config['TRAIN']['batch_size'], 
                        num_workers=config['TRAIN']['num_workers'], 
                        shuffle=False, collate_fn=collate_batch)

# 分析実行
analyze_batch_trajectories(val_dataloader, config)