#!/usr/bin/env bash

set -euo pipefail

GPU_ID="${1:-7}"
CKPT="experiments/jta_3dp_not_normalization_finetune_final_sampling_mask0_true/checkpoints/checkpoint.pth.tar"
RECON_CKPT="skeleton_mae/ckpt/0.5_mask0/checkpoints/best_model_final.pth"

# mask joints を0から21まで1刻みで実行（関節15は内部で除外）
# for mask_joints in $(seq 0 1 21); do
#
#   echo "=== Evaluating mask joints: ${mask_joints} ==="
#   CUDA_VISIBLE_DEVICES="${GPU_ID}" python evaluate_jta_3dp_cleaning.py \
#     --ckpt "${CKPT}" \
#     --recon_ckpt "${RECON_CKPT}" \
#     --mask_joints "${mask_joints}"
#
# done

# mask rateを0から1まで0.1刻みで実行
for mask_rate in $(seq 0.0 0.1 1.0); do

  echo "=== Evaluating mask rate: ${mask_rate} ==="
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python evaluate_jta_3dp_cleaning.py \
    --ckpt "${CKPT}" \
    --recon_ckpt "${RECON_CKPT}" \
    --mask_rate "${mask_rate}"

done

