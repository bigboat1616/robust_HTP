#!/usr/bin/env bash

# Simple helper script to run evaluate_jta_3dp.py across random occlusion splits.
# Usage: ./eval_random_occlusion.sh [GPU_ID]
# GPU_ID defaults to 4 if not provided.

set -euo pipefail
GPU_ID="${1:-7}"
CKPT="experiments/jta_3dp_not_normalization_2/checkpoints/best_val_checkpoint.pth.tar"

for joints in $(seq 1 21); do
  if [[ "${joints}" -eq 1 ]]; then
    joint_label="joint"
  else
    joint_label="joints"
  fi

if [[ "${joints}" -eq 1 ]]; then
  split="test_occlusion/random_${joints}joint_mask/"
else
  split="test_occlusion/random_${joints}joints_mask/"
fi

  echo "=== Evaluating split: ${split} ==="
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python evaluate_jta_3dp_cleaning.py \
    --ckpt "${CKPT}" \
    --cleaning_ckpt ./finetune/jta_3dp_stgcn/checkpoints/cleaning.pth\
    --split "${split}"
done

