
#!/usr/bin/env bash

set -euo pipefail

GPU_ID="${1:-6}"
CKPT="experiments/jta_3dp_not_normalization_2/checkpoints/checkpoint.pth.tar"
# Run mask joints from 0 to 21 in steps of 1 (joint 15 is excluded internally)
# for mask_joints in $(seq 0 1 21); do
#
#   echo "=== Evaluating mask joints: ${mask_joints} ==="
#   CUDA_VISIBLE_DEVICES="${GPU_ID}" python evaluate_jta_3dp.py \
#     --ckpt "${CKPT}" \
#     --mask_joints "${mask_joints}"
#
# done

# Run mask rate from 0 to 1 in steps of 0.1
for mask_rate in $(seq 0.0 0.1 1.0); do

  echo "=== Evaluating mask rate: ${mask_rate} ==="
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python evaluate_jta_3dp.py \
    --ckpt "${CKPT}" \
    --mask_rate "${mask_rate}"

done

