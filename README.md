<div align="center">
<h1>Robust Human Trajectory Prediction via Self-Supervised Skeleton Representation Learning</h1>
<h3>Taishu Arashima*, Hiroshi Kera*, Kazuhiko Kawamoto</h3>
</div>

This repository provides code for our trajectory prediction framework that pretrains a skeleton encoder with masked joint reconstruction and then fine-tunes a trajectory predictor with robust skeletal representations. See the paper: [Robust Human Trajectory Prediction via Self-Supervised Skeleton Representation Learning](file://_arXiv_.pdf).

<div align="center">
<img src="docs/figure1.pdf" alt="Overview" width="90%">
</div>

# Setup

Install the requirements:
```
pip install -r requirements.txt
```

# Data

Place JTA-3DP data under `data/jta_3dp` (train/val/test subfolders). For other datasets, adjust paths in the YAML configs.

# Stage 1: Skeleton Pretraining (Self-Supervised)

Pretrain the skeleton encoder:
```
python skeleton_mae/main_skeleton_coord.py --cfg skeleton_mae/configs_skeleton.yml
```
Checkpoints are saved under the YAML-defined `TRAIN.save_dir`, with:
- `model_final.pth` (final epoch)
- `best_model_final.pth` (best validation loss during training)

# Stage 2: Trajectory Fine-Tuning

Use the pretrained encoder checkpoint via `MODEL.backbone_ckpt` in `configs/jta_3dp.yaml`, then train:
```
python train_jta_3dp.py --cfg configs/jta_3dp.yaml
```

# Evaluation

Evaluate the trajectory predictor:
```
python evaluate_jta_3dp.py --ckpt <checkpoint> --metric ade_fde --modality traj+3dpose
```

# Notes

- Skeleton encoder weights are loaded and frozen in `model_jta_3dp_finetune_coord2.py`.
- If you change checkpoint naming or directories, update `TRAIN.save_dir` in `skeleton_mae/configs_skeleton.yml` and `MODEL.backbone_ckpt` in `configs/jta_3dp.yaml`.