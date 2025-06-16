import argparse
from datetime import datetime
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from progress.bar import Bar
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from dataset_jrdb import collate_batch, batch_process_coords, get_datasets, create_dataset
from model_jrdb_traj_FFT_schedule import create_model
from utils.utils import create_logger, load_default_config, load_config, AverageMeter
from utils.metrics import MSE_LOSS
import wandb
from scipy.fft import fft


def evaluate_loss(model, dataloader, config):
    bar = Bar(f"EVAL", fill="#", max=len(dataloader))
    loss_avg = AverageMeter()
    dataiter = iter(dataloader)
    
    model.eval()
    with torch.no_grad():
        for i in range(len(dataloader)):
            try:
                joints, masks, padding_mask = next(dataiter)
            except StopIteration:
                break

            in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(joints, masks, padding_mask, config)
            padding_mask = padding_mask.to(config["DEVICE"])

            loss, _ = compute_loss(model, config, in_joints, out_joints, in_masks, out_masks, padding_mask)
            loss_avg.update(loss.item(), len(in_joints))
            
            summary = [
                f"({i + 1}/{len(dataloader)})",
                f"LOSS: {loss_avg.avg:.4f}",
                f"T-TOT: {bar.elapsed_td}",
                f"T-ETA: {bar.eta_td:}"
            ]

            bar.suffix = " | ".join(summary)
            bar.next()

        bar.finish()

    return loss_avg.avg

def get_num_freqs(epoch, config):
    """エポックに応じて使用する周波数成分数を決定
    Args:
        epoch: 現在のエポック
        config: 設定
    Returns:
        num_freqs: 使用する周波数成分数（Noneの場合は全周波数使用）
    """
    total_epochs = config["TRAIN"]["epochs"]
    seq_len = config["TRAIN"]["input_track_size"]  # 9
    half_n = (seq_len - 1) // 2  # (9-1)/2 = 4
    max_freqs = half_n + 1  # DC + 4対の周波数成分 = 5
    
    # スケジューリング戦略
    if epoch < total_epochs * 0.2:    # 10-30%
        return 2                        # DC + 最低周波数対
    elif epoch < total_epochs * 0.5:    # 30-50%
        return 3                        # DC + 低周波2対
    elif epoch < total_epochs * 0.7:    # 50-70%
        return 4                        # DC + 低周波3対
    return 5    

def compute_loss(model, config, in_joints, out_joints, in_masks, out_masks, padding_mask, epoch=None, mode='val', loss_last=True, optimizer=None):
    _, in_F, _, _ = in_joints.shape
    metamask = (mode == 'train')
    
    # 周波数成分数の決定
    num_freqs = get_num_freqs(epoch, config) if mode == 'train' else None
    
    # モデル予測（周波数スケジューリングを適用）
    pred_joints = model(in_joints, padding_mask, metamask=metamask, num_freqs=num_freqs)
    
    if mode == 'train' and num_freqs is not None:
        # GTに対しても同じ周波数スケジューリングを適用
        B, F, N, D = out_joints.shape
        gt_freq = torch.fft.fft(out_joints[..., :2], dim=1)  # FFT適用
        
        # 周波数マスクの生成
        mask = torch.zeros_like(gt_freq, device=gt_freq.device)
        mask[:, 0] = 1.0  # DC成分
        if num_freqs > 1:
            for i in range(1, min(num_freqs, gt_freq.size(1)//2 + 1)):
                mask[:, 5-i] = 1.0
                mask[:, -5+i] = 1.0

        # マスクを適用してIFFTで戻す
        gt_freq_filtered = gt_freq * mask
        out_joints_filtered = torch.fft.ifft(gt_freq_filtered, dim=1).real
        
        # フィルタリングしたGTで損失を計算
        loss = MSE_LOSS(pred_joints[:,in_F:], out_joints_filtered, out_masks)
    else:
        # 評価時は通常通りの損失計算
        loss = MSE_LOSS(pred_joints[:,in_F:], out_joints, out_masks)
    
    return loss, pred_joints

def compute_frequency_loss(pred, target, masks):
    """
    12の周波数帯域ごとの誤差を計算
    pred, target: (batch, sequence_length, dim)
    """
    # 勾配計算を切り離してからnumpy配列に変換
    pred = pred[:,:,0,:2].detach().cpu().numpy()
    target = target[:,:,0,:2].detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()
    
    # 時系列方向にFFTを適用
    pred_fft = fft(pred, axis=1)
    target_fft = fft(target, axis=1)
    
    # 周波数ごとの誤差を計算
    freq_loss = np.abs(pred_fft - target_fft)
    
    # 12の周波数帯域に分割（低周波から高周波まで）
    freq_losses = {}
    
    for i in range(7):
        band_loss = np.mean(freq_loss[:,i,0])+np.mean(freq_loss[:,i,1])
        # 低周波から順番に番号付け
        freq_losses[f'freq_band_{i:02d}'] = band_loss
    return freq_losses


def adjust_learning_rate(optimizer, epoch, config):
    """
    From: https://github.com/microsoft/MeshTransformer/
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs*2/3 = 100
    """
    lr = config['TRAIN']['lr'] * (config['TRAIN']['lr_decay'] ** epoch)
    if 'lr_drop' in config['TRAIN'] and config['TRAIN']['lr_drop']:
        lr = lr * (0.1 ** (epoch // (config['TRAIN']['epochs']*4./5.)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    print('lr: ',lr)
        
def save_checkpoint(model, optimizer, epoch, config, filename, logger):
    logger.info(f'Saving checkpoint to {filename}.')
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'config': config
    }
    torch.save(ckpt, os.path.join(config['OUTPUT']['ckpt_dir'], filename))

def dataloader_for(dataset, config, **kwargs):
    g = torch.Generator()
    g.manual_seed(config['SEED'])
    
    return DataLoader(dataset,
                      batch_size=config['TRAIN']['batch_size'],
                      num_workers=config['TRAIN']['num_workers'],
                      collate_fn=collate_batch,
                      worker_init_fn=seed_worker,
                      generator=g,
                      **kwargs)

def dataloader_for_val(dataset, config, **kwargs):
    g = torch.Generator()
    g.manual_seed(config['SEED'])
    
    return DataLoader(dataset,
                      batch_size=1,
                      num_workers=0,
                      collate_fn=collate_batch,
                      worker_init_fn=seed_worker,
                      generator=g,
                      **kwargs)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(config, logger, experiment_name="", dataset_name=""):
    ################################
    # Load data
    ################################
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    dataset_train = ConcatDataset(get_datasets(config['DATA']['train_datasets'], config, logger))
    dataloader_train = dataloader_for(dataset_train, config, shuffle=True, pin_memory=True)
    logger.info(f"Training on a total of {len(dataset_train)} annotations.")

    dataset_val = create_dataset(config['DATA']['train_datasets'][0], logger, split="val", track_size=(in_F+out_F), track_cutoff=in_F)
    dataloader_val = dataloader_for_val(dataset_val, config, shuffle=False, pin_memory=True)

    writer_name = experiment_name + "_" + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    writer_train = SummaryWriter(os.path.join(config["OUTPUT"]["runs_dir"], f"{writer_name}_TRAIN"))
    writer_valid = SummaryWriter(os.path.join(config["OUTPUT"]["runs_dir"], f"{writer_name}_VALID"))
    
    ################################
    # Create model, loss, optimizer
    ################################
    model = create_model(config, logger)

    if config["MODEL"]["checkpoint"] != "":
        logger.info(f"Loading checkpoint from {config['MODEL']['checkpoint']}")
        checkpoint = torch.load(os.path.join(config['OUTPUT']['ckpt_dir'], config["MODEL"]["checkpoint"]))
        model.load_state_dict(checkpoint["model"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['TRAIN']['lr'])

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_parameters} parameters.")
    
    ################################
    # Begin Training 
    ################################
    global_step = 0
    min_val_loss = 1e4

    wandb.init(
        project="jrdb_trajectory",
        name=writer_name,
        config=config
    )
    # 周波数バンドの可視化設定
    wandb.define_metric("step")
    wandb.define_metric("freq/*", step_metric="step")  # freq/で始まるメトリックをグループ化
    
    for epoch in range(config["TRAIN"]["epochs"]):
        # 現在の周波数成分数をログに記録
        num_freqs = get_num_freqs(epoch, config)
        logger.info(f"Epoch {epoch}: Using {num_freqs if num_freqs is not None else 'all'} frequency components")
        
        start_time = time.time()
        dataiter = iter(dataloader_train)

        timer = {"DATA": 0, "FORWARD": 0, "BACKWARD": 0}
        loss_avg = AverageMeter()

        if config["TRAIN"]["optimizer"] == "adam":
            adjust_learning_rate(optimizer, epoch, config)

        train_steps = len(dataloader_train)
        bar = Bar(f"TRAIN {epoch}/{config['TRAIN']['epochs'] - 1} (freq={num_freqs if num_freqs is not None else 'all'})", 
                 fill="#", max=train_steps)

        # 周波数帯域ごとの平均を記録するメーター
        freq_meters = {f'freq_band_{i:02d}': AverageMeter() for i in range(7)}

        for i in range(train_steps): 
            model.train()
            optimizer.zero_grad()

            ################################
            # Load a batch of data
            ################################
            start = time.time()
            try:
                joints, masks, padding_mask = next(dataiter)
            except StopIteration:
                dataiter = iter(dataloader_train)
                joints, masks, padding_mask = next(dataiter)

            in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(
                joints, masks, padding_mask, config, training=True, modality_selection='traj'
            )
            padding_mask = padding_mask.to(config["DEVICE"])
            
            timer["DATA"] = time.time() - start

            ################################
            # Forward Pass with frequency scheduling
            ################################
            start = time.time()
            loss, pred_joints = compute_loss(
                model, config, in_joints, out_joints, in_masks, out_masks, 
                padding_mask, epoch=epoch, mode='train', optimizer=None
            )
            timer["FORWARD"] = time.time() - start

            ################################
            # Backward Pass + Optimization
            ################################
            start = time.time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["TRAIN"]["max_grad_norm"])
            optimizer.step()
            timer["BACKWARD"] = time.time() - start

            ################################
            # Logging 
            ################################
            loss_avg.update(loss.item(), len(joints))

            # 周波数損失の計算と更新（これは必要）
            freq_losses = compute_frequency_loss(pred_joints[:,in_F:], out_joints, out_masks)
            if freq_losses is not None:
                # 周波数帯域ごとの移動平均を更新
                for k, v in freq_losses.items():
                    freq_meters[k].update(v, len(joints))
            
            summary = [
                f"{str(epoch).zfill(3)} ({i + 1}/{train_steps})",
                f"LOSS: {loss_avg.avg:.4f}",
                f"LOW-F: {freq_meters['freq_band_01'].avg:.4f}",
                f"MID-F: {freq_meters['freq_band_03'].avg:.4f}",
                f"HIGH-F: {freq_meters['freq_band_05'].avg:.4f}",
                f"T-TOT: {bar.elapsed_td}",
                f"T-ETA: {bar.eta_td:}"
            ]

            for key, val in timer.items():
                summary.append(f"{key}: {val:.2f}")

            bar.suffix = " | ".join(summary)
            bar.next()

            if config.get('dry_run', False):
                break
            
        bar.finish()

        ################################
        # Tensorboard logs
        ################################
        global_step += train_steps
        writer_train.add_scalar("num_frequencies", num_freqs if num_freqs is not None else config["TRAIN"]["input_track_size"], epoch)
        writer_train.add_scalar("loss", loss_avg.avg, epoch)
        
        val_loss = evaluate_loss(model, dataloader_val, config)
        writer_valid.add_scalar("loss", val_loss, epoch)

        # エポック終了時にまとめてWandBにログを記録
        log_dict = {
            "step": epoch,
            "train_loss": loss_avg.avg,
            "val_loss": val_loss,
        }
        
        # 周波数帯域のロスを追加（freq/プレフィックスを使用）
        for i in range(7):
            log_dict[f"freq/band_{i:02d}"] = freq_meters[f'freq_band_{i:02d}'].avg

        wandb.log(log_dict)


        
        val_ade = val_loss/100
        if val_ade < min_val_loss:
            min_val_loss = val_ade
            print('------------------------------BEST MODEL UPDATED------------------------------')
            print('Best ADE: ', val_ade)
            save_checkpoint(model, optimizer, epoch, config, 'best_val'+'_checkpoint.pth.tar', logger)

        if config.get('dry_run', False):
            break
            
        # エポック終了時のサマリー表示
        print('------------------------------EPOCH SUMMARY------------------------------')
        print(f'Epoch {epoch} finished!')
        print(f'Training Loss: {loss_avg.avg:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print('Frequency Band Losses:')
        for i in range(7):
            print(f'  Band {i:02d}: {freq_meters[f"freq_band_{i:02d}"].avg:.4f}')
        print(f'Time for training: {time.time()-start_time:.2f}s')
        print('----------------------------------------------------------------------')

    if not config.get('dry_run', False):
        save_checkpoint(model, optimizer, epoch, config, 'checkpoint.pth.tar', logger)
    logger.info("All done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name. Otherwise will use timestamp")
    parser.add_argument("--cfg", type=str, default="", help="Config name. Otherwise will use default config")
    parser.add_argument('--dry-run', action='store_true', help="Run just one iteration")
    args = parser.parse_args()

    if args.cfg != "":
        cfg = load_config(args.cfg, exp_name=args.exp_name)
    else:
        cfg = load_default_config()

    cfg['dry_run'] = args.dry_run
    
    # 決定的アルゴリズムの使用を設定
    torch.use_deterministic_algorithms(True)
    
    # 既存のseed設定
    random.seed(cfg['SEED'])
    torch.manual_seed(cfg['SEED'])
    np.random.seed(cfg['SEED'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg['SEED'])
        torch.cuda.manual_seed_all(cfg['SEED'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    if torch.cuda.is_available():
        cfg["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
    else:
        cfg["DEVICE"] = "cpu"

    # Force trajectory-only mode
    if 'TRAIN' not in cfg:
        cfg['TRAIN'] = {}
    cfg['TRAIN']['modality'] = 'traj'

    dataset = cfg["DATA"]["train_datasets"]

    logger = create_logger(cfg["OUTPUT"]["log_dir"])
    logger.info("Initializing trajectory-only training with config:")
    logger.info(cfg)

    train(cfg, logger, experiment_name=args.exp_name, dataset_name=dataset) 