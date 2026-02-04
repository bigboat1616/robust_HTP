import argparse
import json
import os
import torch
import random
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from progress.bar import Bar
from torch.utils.data import DataLoader

from dataset_jta import batch_process_coords, create_dataset, collate_batch
from model_jta_3dp_finetune_coord2 import create_model
from utils.utils import create_logger


def evaluate_ade_fde(
    model,
    modality_selection,
    dataloader,
    bs,
    config,
    logger,
    return_all=False,
    bar_prefix="",
    per_joint=False,
    show_avg=False,
    warmup_batches=10,   # 追加: ウォームアップ（計測から除外）
    pred_save_path=None,
    viz_samples=0,
    viz_dir=None,
    viz_scale_margin=0.1,
):
    """
    - forward時間だけを測りやすい形に整理
    - model.eval()/no_grad はループ外へ
    - slicing(pred[:, -out_F:]) は計測外へ
    - GPU計測は synchronize で境界を切る（forwardの実時間に近い）
    - viz_samples/viz_dir を指定すると traj 画像を保存
    """
    in_F, out_F = config["TRAIN"]["input_track_size"], config["TRAIN"]["output_track_size"]
    bar = Bar(f"{bar_prefix}EVAL ADE_FDE", fill="#", max=len(dataloader))

    device_str = str(config.get("DEVICE", "cpu"))
    use_cuda = device_str.startswith("cuda")

    model.eval()
    total_forward_time = 0.0
    total_samples = 0
    min_per_sample = float("inf")
    max_per_sample = 0.0

    batch_size = bs
    batch_id = 0
    ade_batch = torch.zeros((), device=config["DEVICE"])
    fde_batch = torch.zeros((), device=config["DEVICE"])

    fde_max = 0.0
    viz_records = []
    if viz_samples > 0 and viz_dir:
        os.makedirs(viz_dir, exist_ok=True)

    pred_trajectories = [] if pred_save_path else None

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            bar.next()
            joints, masks, padding_mask = batch
            padding_mask = padding_mask.to(config["DEVICE"])

            # 前処理（計測外）
            in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(
                joints, masks, padding_mask, config, modality_selection
            )

            # --- forward-only timing ---
            if use_cuda:
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            pred_all = model(in_joints, padding_mask)  # ← forward本体

            if use_cuda:
                torch.cuda.synchronize()
            forward_time = time.perf_counter() - start_time
            # --------------------------

            # ウォームアップは集計しない（最初は遅く出がち）
            # out_joints は GPU 上に保持し、距離計算も GPU でまとめて行う
            batch_samples = out_joints.size(0)

            if i >= warmup_batches:
                total_forward_time += forward_time
                total_samples += batch_samples
                if batch_samples > 0:
                    per_sample = forward_time / batch_samples
                    min_per_sample = min(min_per_sample, per_sample)
                    max_per_sample = max(max_per_sample, per_sample)

            # 以降は計測外（スライス・CPU化・評価）
            pred_future = pred_all[:, -out_F:, 0, :2]
            gt_future = out_joints[:, :, 0, :2]
            diff = pred_future - gt_future
            distances = diff.norm(dim=-1)
            ade_batch += distances.mean(dim=1).sum()
            fde_batch += distances[:, -1].sum()

            if pred_trajectories is not None:
                pred_trajectories.extend(pred_future.detach().cpu().tolist())

            if viz_samples > 0 and viz_dir and len(viz_records) < viz_samples:
                for sample_idx in range(batch_samples):
                    if len(viz_records) >= viz_samples:
                        break
                    viz_records.append(
                        {
                            "pred": pred_future[sample_idx].detach().cpu().numpy(),
                            "gt": gt_future[sample_idx].detach().cpu().numpy(),
                            "batch": i,
                            "sample_idx": sample_idx,
                        }
                    )

            batch_id += 1

    bar.finish()

    denom = (batch_id - 1) * batch_size + len(out_joints)
    denom = float(denom) if denom else 1.0

    # ADE/FDE 集計（元コードの割り方を踏襲）
    ade = (ade_batch / denom).item()
    fde = (fde_batch / denom).item()

    # 時間表示（ウォームアップ除外後の平均）
    if total_samples > 0:
        avg_per_sample = total_forward_time / total_samples
        if min_per_sample == float("inf"):
            min_per_sample = 0.0
        time_message = (
            f"Forward time (sec) per sample -> avg: {avg_per_sample:.6f}, "
            f"min: {min_per_sample:.6f}, max: {max_per_sample:.6f} "
            f"(warmup_batches={warmup_batches})"
        )
        logger.info(time_message)
        print(time_message)

    if viz_records:
        all_points = np.concatenate(
            [np.concatenate([rec["pred"], rec["gt"]], axis=0) for rec in viz_records],
            axis=0,
        )
        min_xy = all_points.min(axis=0)
        max_xy = all_points.max(axis=0)
        span = max(max_xy - min_xy)
        if span == 0:
            span = 1.0
        half_span = span * 0.5 * (1.0 + viz_scale_margin)
        center = (max_xy + min_xy) * 0.5
        fixed_bounds = (
            (center[0] - half_span, center[0] + half_span),
            (center[1] - half_span, center[1] + half_span),
        )

        for idx, rec in enumerate(viz_records):
            filename = os.path.join(viz_dir, f"traj_{idx:05d}.png")
            title = f"Sample {idx} (batch {rec['batch']}, idx {rec['sample_idx']})"
            visualize_future_trajectory(
                rec["pred"],
                rec["gt"],
                sample_idx=0,
                title=title,
                output_path=filename,
                fixed_bounds=fixed_bounds,
                scale_margin=viz_scale_margin,
            )
            logger.info(f"Saved trajectory visualization to {filename}")

        logger.info(f"Saved {len(viz_records)} trajectory visualizations under {viz_dir}")

    if pred_trajectories is not None:
        pred_dir = os.path.dirname(pred_save_path)
        if pred_dir:
            os.makedirs(pred_dir, exist_ok=True)
        with open(pred_save_path, "w", encoding="utf-8") as f:
            json.dump(pred_trajectories, f)
        logger.info(f"Saved {len(pred_trajectories)} predicted trajectories to {pred_save_path}")

    return ade, fde


def visualize_future_trajectory(
    pred_future,
    gt_future,
    sample_idx=0,
    title=None,
    output_path=None,
    show=False,
    figsize=(6, 6),
    ax=None,
    lock_scale=True,
    scale_margin=0.1,
    fixed_bounds=None,
    lock_threshold=5.0,
):
    """
    Plot predicted vs. ground-truth XY trajectories for a single sample.

    lock_scale:
        - True  : always enforce equal limits based on combined span
        - False : leave Matplotlib defaults
        - "auto": enforce only when combined span >= lock_threshold
    fixed_bounds:
        ((xmin, xmax), (ymin, ymax)) to force identical limits across plots.
    """

    def _to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    pred_arr = _to_numpy(pred_future)
    gt_arr = _to_numpy(gt_future)

    if pred_arr.ndim == 3:
        if pred_arr.shape[0] <= sample_idx:
            raise IndexError(f"sample_idx {sample_idx} is out of range for batch size {pred_arr.shape[0]}.")
        pred_xy = pred_arr[sample_idx]
        gt_xy = gt_arr[sample_idx]
    elif pred_arr.ndim == 2:
        pred_xy = pred_arr
        gt_xy = gt_arr
    else:
        raise ValueError("Expected (batch, frames, 2) or (frames, 2) for pred_future.")

    if pred_xy.shape[-1] != 2 or gt_xy.shape[-1] != 2:
        raise ValueError("Last dimension must be 2 (x, y).")

    created_axes = ax is None
    if created_axes:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "o-", label="GT", color="#1f77b4")
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], "s--", label="Prediction", color="#d62728")
    ax.scatter(gt_xy[0, 0], gt_xy[0, 1], c="#2ca02c", marker="^", label="Start")
    ax.scatter(gt_xy[-1, 0], gt_xy[-1, 1], c="#ff7f0e", marker="x", label="GT End")
    ax.scatter(pred_xy[-1, 0], pred_xy[-1, 1], c="#9467bd", marker="x", label="Pred End")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)

    if fixed_bounds is not None:
        (xmin, xmax), (ymin, ymax) = fixed_bounds
        width = xmax - xmin
        height = ymax - ymin
        span = max(width, height)
        if span == 0:
            span = 1.0
        half_span = span * 0.5 * (1.0 + scale_margin)
        center = np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5])
        ax.set_xlim(center[0] - half_span, center[0] + half_span)
        ax.set_ylim(center[1] - half_span, center[1] + half_span)
    else:
        if isinstance(lock_scale, str):
            if lock_scale not in {"auto"}:
                raise ValueError("lock_scale string must be 'auto'.")
            stacked = np.vstack([pred_xy, gt_xy])
            span = float(np.max(np.max(stacked, axis=0) - np.min(stacked, axis=0)))
            do_lock = span >= lock_threshold
        else:
            do_lock = bool(lock_scale)

        if do_lock:
            stacked = np.vstack([pred_xy, gt_xy])
            min_xy = stacked.min(axis=0)
            max_xy = stacked.max(axis=0)
            span = max(max_xy - min_xy)
            if span == 0:
                span = 1.0
            half_span = span * 0.5 * (1.0 + scale_margin)
            center = (max_xy + min_xy) * 0.5
            ax.set_xlim(center[0] - half_span, center[0] + half_span)
            ax.set_ylim(center[1] - half_span, center[1] + half_span)

    frames = np.arange(gt_xy.shape[0])
    for f_idx, (x_gt, y_gt) in enumerate(gt_xy):
        ax.text(x_gt, y_gt, str(frames[f_idx]), fontsize=8, color="#1f77b4", alpha=0.7)

    if title is None:
        title = f"Sample {sample_idx}"
    ax.set_title(title)
    ax.legend()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    elif created_axes:
        plt.close(fig)

    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint path")
    parser.add_argument("--split", type=str, default="test", help="Split to use. one of [train, test, valid]")
    parser.add_argument("--metric", type=str, default="vim", help="Evaluation metric. One of (vim, mpjpe)")
    parser.add_argument(
        "--modality",
        type=str,
        default="traj+all",
        help="available modality combination from['traj','traj+3dpose','traj+all']",
    )
    parser.add_argument("--warmup_batches", type=int, default=10, help="Warmup batches excluded from timing")
    parser.add_argument("--viz_samples", type=int, default=0, help="Number of samples to visualize (0 disables)")
    parser.add_argument("--viz_output_dir", type=str, default="traj", help="Base directory for trajectory plots")
    parser.add_argument(
        "--viz_scale_margin",
        type=float,
        default=0.1,
        help="Relative margin applied when fixing trajectory plot bounds",
    )
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ################################
    # Load checkpoint
    ################################
    logger = create_logger("")
    logger.info(f"Loading checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=torch.device("cpu"))
    config = ckpt["config"]

    if torch.cuda.is_available():
        config["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
        torch.cuda.manual_seed(0)
    else:
        config["DEVICE"] = "cpu"

    logger.info("Initializing with config:")
    logger.info(config)

    ################################
    # Initialize model
    ################################
    model = create_model(config, logger)
    model.load_state_dict(ckpt["model"])
    model = model.to(config["DEVICE"])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters (total/trainable): {total_params:,}/{trainable_params:,}")
    print(f"Model parameters (total/trainable): {total_params:,}/{trainable_params:,}")

    ################################
    # Load data
    ################################
    in_F, out_F = config["TRAIN"]["input_track_size"], config["TRAIN"]["output_track_size"]
    assert in_F == 9
    assert out_F == 12

    name = config["DATA"]["train_datasets"]
    dataset = create_dataset(name[0], logger, split=args.split, track_size=(in_F + out_F), track_cutoff=in_F)

    bs = config["TRAIN"]["batch_size"]
    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        num_workers=config["TRAIN"]["num_workers"],
        shuffle=False,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    ckpt_stem = os.path.splitext(os.path.basename(args.ckpt))[0]

    viz_dir = None
    if args.viz_samples > 0:
        viz_dir = os.path.join(args.viz_output_dir, ckpt_stem)
        os.makedirs(viz_dir, exist_ok=True)
        logger.info(f"Saving trajectory visualizations under {viz_dir}")

    pred_save_dir = os.path.join("future_traj", ckpt_stem)
    os.makedirs(pred_save_dir, exist_ok=True)
    pred_save_path = os.path.join(pred_save_dir, f"{args.split}_pred_future.json")
    logger.info(f"Saving predicted trajectories under {pred_save_path}")

    ade, fde = evaluate_ade_fde(
        model,
        args.modality,
        dataloader,
        bs,
        config,
        logger,
        return_all=True,
        warmup_batches=args.warmup_batches,
        pred_save_path=pred_save_path,
        viz_samples=args.viz_samples,
        viz_dir=viz_dir,
        viz_scale_margin=args.viz_scale_margin,
    )

    print("ADE: ", ade)
    print("FDE: ", fde)