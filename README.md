<div align="center">
<h1> Robust_Social-Transmotion </h1>
<h3>Taishu Arashima*, Hiroshi Kera*, Kazuhiko Kawamoto</h3>

</div>

<div align="center"> <h3> Abstract </h3>  </div>
<div align="justify">


# Getting Started

Install the requirements using `pip`:
```
pip install -r requirements.txt
```

We have conveniently added the preprocessed data to the release section of the repository (for license details, please refer to the original papers).
Place the data subdirectory of JTA under `data/jta_all_visual_cues` and the data subdirectory of JRDB under `data/jrdb_2dbox` of the repository.

# Training and Testing

## JTA dataset
You can train the Social-Transmotion model on this dataset using the following command:
```
python train_jta.py --cfg configs/jta_all_visual_cues.yaml --exp_name jta
```


To evaluate the trained model, use the following command:
```
python evaluate_jta.py --ckpt ./experiments/jta/checkpoints/checkpoint.pth.tar --metric ade_fde --modality traj+all
```
Please note that the evaluation modality can be any of `[traj, traj+2dbox, traj+3dpose, traj+2dpose, traj+3dpose+3dbox, traj+all]`.
For the ease of use, we have also provided the trained model in the release section of this repo. In order to use that, you should pass the address of the saved checkpoint via `--ckpt`.

## JRDB dataset
You can train the Social-Transmotion model on this dataset using the following command:
```
python train_jrdb.py --cfg configs/jrdb_2dbox.yaml --exp_name jrdb
```

To evaluate the trained model, use the following command:
```
python evaluate_jrdb.py --ckpt ./experiments/jrdb/checkpoints/checkpoint.pth.tar --metric ade_fde --modality traj+2dbox
```
Please note that the evaluation modality can be one any of `[traj, traj+2dbox]`.
For the ease of use, we have also provided the trained model in the release section of this repo. In order to use that, you should pass the address of the saved checkpoint via `--ckpt`.

# Work in Progress

This repository is work-in-progress and will continue to get updated and improved over the coming months.

