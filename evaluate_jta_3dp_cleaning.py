import argparse
import time
import json
import torch
import torch.nn as nn
import random
import numpy as np
from progress.bar import Bar
from torch.utils.data import DataLoader

from dataset_jta import batch_process_coords, create_dataset, collate_batch
from model_jta_3dp_cleaning import create_model
from utils.utils import create_logger


def _extract_state_dict(payload):
    if not isinstance(payload, dict):
        return payload
    for key in ("model", "model_state_dict", "state_dict", "encoder_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate
    return payload


def _strip_module_prefix(key):
    return key[7:] if key.startswith("module.") else key


def load_model_with_checkpoints(model, main_ckpt_payload, recon_ckpt_path, device, logger):
    current_state = model.state_dict()
    updated_state = dict(current_state)

    def _assign(target_key, tensor, source):
        if target_key not in updated_state:
            logger.debug(f"[ckpt:{source}] skip missing key {target_key}")
            return False
        if updated_state[target_key].shape != tensor.shape:
            logger.debug(
                f"[ckpt:{source}] shape mismatch for {target_key}: "
                f"{tensor.shape} -> {updated_state[target_key].shape}"
            )
            return False
        if tensor.dtype != updated_state[target_key].dtype:
            tensor = tensor.to(updated_state[target_key].dtype)
        updated_state[target_key] = tensor
        return True

    recon_loaded = 0
    if recon_ckpt_path:
        recon_payload = torch.load(recon_ckpt_path, map_location=device)
        recon_sd = _extract_state_dict(recon_payload)
        for raw_key, tensor in recon_sd.items():
            key = _strip_module_prefix(raw_key)
            if key.startswith("encoder."):
                key = "recon_stgcn." + key[len("encoder."):]
            elif key.startswith("stgcn."):
                key = "recon_stgcn." + key[len("stgcn."):]
            elif key.startswith("coord_decoder."):
                key = "recon_coord_decoder." + key[len("coord_decoder."):]
            if key.startswith("recon_stgcn.") or key.startswith("recon_coord_decoder."):
                if _assign(key, tensor, "recon"):
                    recon_loaded += 1
        logger.info(f"[ckpt] reconstructed modules loaded: {recon_loaded}")

    main_sd = _extract_state_dict(main_ckpt_payload)
    main_loaded = 0
    for raw_key, tensor in main_sd.items():
        key = _strip_module_prefix(raw_key)
        if key.startswith(("recon_stgcn.", "recon_coord_decoder.")):
            continue
        if _assign(key, tensor, "main"):
            main_loaded += 1
    logger.info(f"[ckpt] main modules loaded: {main_loaded}")

    missing, unexpected = model.load_state_dict(updated_state, strict=False)
    if missing:
        logger.warning(f"[ckpt] missing keys after load: {missing}")
    if unexpected:
        logger.warning(f"[ckpt] unexpected keys after load: {unexpected}")


def _freeze_module(module, name, logger, report_limit=5):
    module.eval()
    for param in module.parameters():
        param.requires_grad = False

    reported = 0
    for child_name, sub_module in module.named_modules():
        if isinstance(sub_module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            sub_module.eval()
            sub_module.track_running_stats = True
            if reported < report_limit:
                mean_norm = sub_module.running_mean.norm().item() if sub_module.running_mean is not None else 0.0
                var_norm = sub_module.running_var.norm().item() if sub_module.running_var is not None else 0.0
                weight_req = sub_module.weight.requires_grad if sub_module.affine else None
                bias_req = sub_module.bias.requires_grad if sub_module.affine else None
                logger.info(
                    f"[{name}_bn] "
                    f"name={child_name} "
                    f"mean_norm={mean_norm:.6f} "
                    f"var_norm={var_norm:.6f} "
                    f"weight_req={weight_req} "
                    f"bias_req={bias_req}"
                )
                reported += 1


def freeze_backbones(model, logger):
    if hasattr(model, "stgcn"):
        _freeze_module(model.stgcn, "stgcn", logger)
    else:
        logger.info("stgcn not found in model")
    if hasattr(model, "recon_stgcn"):
        _freeze_module(model.recon_stgcn, "recon_stgcn", logger)
    if hasattr(model, "recon_coord_decoder"):
        model.recon_coord_decoder.eval()
        for param in model.recon_coord_decoder.parameters():
            param.requires_grad = False

def inference(model, config, input_joints, padding_mask, out_len=14):
    model.eval()
    
    with torch.no_grad():
        pred_joints = model(input_joints, padding_mask)

    output_joints = pred_joints[:,-out_len:]

    return output_joints


def evaluate_ade_fde(model, dataloader, bs, config, logger, return_all=False, bar_prefix="", per_joint=False, show_avg=False, save_predictions=False):
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    bar = Bar(f"EVAL ADE_FDE", fill="#", max=len(dataloader))

    batch_size = bs
    batch_id = 0
    ade = 0
    fde = 0
    ade_batch = 0 
    fde_batch = 0
    all_predictions = [] if save_predictions else None
    for i, batch in enumerate(dataloader):
        bar.next()
        joints, masks, padding_mask = batch
        padding_mask = padding_mask.to(config["DEVICE"])
   
        in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(joints, masks, padding_mask, config)
        pred_joints = inference(model, config, in_joints, padding_mask, out_len=out_F)

        out_joints = out_joints.cpu() 

        pred_joints = pred_joints.cpu().reshape(out_joints.size(0), 12, 1, 2)    
        
        for k in range(len(out_joints)):

            person_out_joints = out_joints[k,:,0:1]
            person_pred_joints = pred_joints[k,:,0:1]

            gt_xy = person_out_joints[:,0,:2]
            pred_xy = person_pred_joints[:,0,:2]
            
            # 予測結果を保存
            if save_predictions:
                pred_list = [[float(pred_xy[t, 0].item()), float(pred_xy[t, 1].item())] for t in range(12)]
                all_predictions.append(pred_list)
            
            sum_ade = 0
                
            for t in range(12):
                d1 = (gt_xy[t,0].detach().cpu().numpy() - pred_xy[t,0].detach().cpu().numpy())
                d2 = (gt_xy[t,1].detach().cpu().numpy() - pred_xy[t,1].detach().cpu().numpy())
             
                dist_ade = [d1,d2]
                sum_ade += np.linalg.norm(dist_ade)
            sum_ade /= 12
            ade_batch += sum_ade
            d3 = (gt_xy[-1,0].detach().cpu().numpy() - pred_xy[-1,0].detach().cpu().numpy())
            d4 = (gt_xy[-1,1].detach().cpu().numpy() - pred_xy[-1,1].detach().cpu().numpy())
            dist_fde = [d3,d4]
            scene_fde = np.linalg.norm(dist_fde)

            fde_batch += scene_fde
        batch_id+=1
    bar.finish()

    ade = ade_batch/((batch_id-1)*batch_size+len(out_joints))
    fde = fde_batch/((batch_id-1)*batch_size+len(out_joints))
    return ade, fde, (batch_id - 1) * batch_size + len(out_joints), batch_id, all_predictions


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="experiments/jta_3dp_not_normalization_finetune_final_sampling_mask0_true/checkpoints/checkpoint.pth.tar",
        help="checkpoint path for main model weights",
    )
    parser.add_argument(
        "--recon_ckpt",
        type=str,
        default="skeleton_mae/ckpt/0.5_mask0/checkpoints/best_model_final.pth",
        help="checkpoint path for stgcn/decoder weights",
    )
    parser.add_argument("--split", type=str, default="test", help="Split to use. one of [train, test, valid]")
    parser.add_argument("--metric", type=str, default="vim", help="Evaluation metric. One of (vim, mpjpe)")
    parser.add_argument("--mask_rate", type=float, default=0.0, help="Override MODEL.mask_rate during evaluation")
    parser.add_argument("--mask_joints", type=int, default=None, help="Override MODEL.mask_joints during evaluation")
    parser.add_argument("--output", type=str, default=None, help="Path to save predictions as JSON")

    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
        
    ################################
    # Load checkpoint
    ################################

    logger = create_logger('')
    logger.info(f'Loading checkpoint from {args.ckpt}') 
    ckpt = torch.load(args.ckpt, map_location = torch.device('cpu'))
    config = ckpt['config']
    if args.mask_rate is not None:
        config.setdefault("MODEL", {})
        config["MODEL"]["mask_rate"] = args.mask_rate
    if args.mask_joints is not None:
        config.setdefault("MODEL", {})
        config["MODEL"]["mask_joints"] = args.mask_joints
    
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
    load_model_with_checkpoints(
        model,
        ckpt,
        args.recon_ckpt,
        device=config["DEVICE"],
        logger=logger,
    )
    freeze_backbones(model, logger)
    total_params, trainable_params = count_parameters(model)
    ################################
    # Load data
    ################################

    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    assert in_F == 9
    assert out_F == 12

    name = config['DATA']['train_datasets']
    dataset = create_dataset(name[0], logger, split=args.split, track_size=(in_F+out_F), track_cutoff=in_F)

    
 
    bs = config['TRAIN']['batch_size']
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=config['TRAIN']['num_workers'], shuffle=False, collate_fn=collate_batch)
    start_time = time.perf_counter()
    ade, fde, sample_num, batch_id, all_predictions = evaluate_ade_fde(
        model, dataloader, bs, config, logger, return_all=True, save_predictions=(args.output is not None)
    )
    elapsed_sec = time.perf_counter() - start_time
    per_batch_sec = elapsed_sec / batch_id if batch_id > 0 else 0.0
    per_sample_sec = elapsed_sec / sample_num if sample_num > 0 else 0.0


    print('ADE: ', ade)
    print('FDE: ', fde)
    print('Eval time (s): ', elapsed_sec)
    print('Time per batch (s): ', per_batch_sec)
    print('Time per sample (s): ', per_sample_sec)
    print('Params (total): ', total_params)
    print('Params (trainable): ', trainable_params)
    
    # 予測結果をJSONで保存
    if args.output is not None and all_predictions is not None:
        with open(args.output, 'w') as f:
            json.dump(all_predictions, f)
        print(f'Predictions saved to: {args.output}')
    

