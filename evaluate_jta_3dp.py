import argparse
import time
import json
import torch
import random
import numpy as np
from progress.bar import Bar
from torch.utils.data import DataLoader
import os

from dataset_jta import batch_process_coords, create_dataset, collate_batch
from model_jta_3dp_finetune import create_model
from utils.utils import create_logger
from skeleton_mae.utils import (
    to_numpy_array, to_camera_coords, _render_tile_image, 
    _draw_original_view, _draw_mask_view, compute_coord_axis_bounds, 
    _crop_image_lists_uniform, _save_stitched_image, _mask_observed_coords
)

def inference(model, config, input_joints, padding_mask, out_len=14):
    model.eval()
    
    with torch.no_grad():
        pred_joints = model(input_joints, padding_mask)

    output_joints = pred_joints[:,-out_len:]

    return output_joints


def evaluate_ade_fde(model, dataloader, bs, config, logger, return_all=False, bar_prefix="", per_joint=False, show_avg=False, save_predictions=False, visualize_input=False, vis_dir=None, vis_sample_indices=None):
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    bar = Bar(f"EVAL ADE_FDE", fill="#", max=len(dataloader))

    batch_size = bs
    batch_id = 0
    ade = 0
    fde = 0
    ade_batch = 0 
    fde_batch = 0
    all_predictions = [] if save_predictions else None
    
    # Build a set of sample indices to visualize
    vis_sample_set = set(vis_sample_indices) if vis_sample_indices is not None else set()
    global_sample_idx = 0  # Global sample index across batches (starts at 0)
    
    for i, batch in enumerate(dataloader):
        bar.next()
        joints, masks, padding_mask = batch
        padding_mask = padding_mask.to(config["DEVICE"])
   
        in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(joints, masks, padding_mask, config)
        
        # Visualize input skeletons (only requested samples)
        if visualize_input and vis_dir is not None and len(vis_sample_set) > 0:
            for local_sample_idx in range(in_joints.size(0)):
                current_global_idx = global_sample_idx + local_sample_idx
                if current_global_idx in vis_sample_set:
                    vis_path = os.path.join(vis_dir, f"input_skeleton_sample_{current_global_idx}.png")
                    visualize_input_skeleton(
                        in_joints, vis_path, sample_idx=local_sample_idx, 
                        config=config, device=config["DEVICE"]
                    )
                    logger.info(f"Visualized sample {current_global_idx} -> {vis_path}")
        
        pred_joints = inference(model, config, in_joints, padding_mask, out_len=out_F)

        out_joints = out_joints.cpu() 

        pred_joints = pred_joints.cpu().reshape(out_joints.size(0), 12, 1, 2)    
        
        for k in range(len(out_joints)):
            # Use the same global index for saving predictions and visualization
            current_global_idx = global_sample_idx + k

            person_out_joints = out_joints[k,:,0:1]
            person_pred_joints = pred_joints[k,:,0:1]

            gt_xy = person_out_joints[:,0,:2]
            pred_xy = person_pred_joints[:,0,:2]
            
            # Save predictions in order starting from index 0
            if save_predictions:
                pred_list = [[float(pred_xy[t, 0].item()), float(pred_xy[t, 1].item())] for t in range(12)]
                all_predictions.append(pred_list)
                # Accessible via all_predictions[current_global_idx] (0-based)
            
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
        
        # Update global sample index after each batch
        global_sample_idx += in_joints.size(0)
        batch_id+=1
    bar.finish()

    ade = ade_batch/((batch_id-1)*batch_size+len(out_joints))
    fde = fde_batch/((batch_id-1)*batch_size+len(out_joints))
    return ade, fde, (batch_id - 1) * batch_size + len(out_joints), batch_id, all_predictions


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def visualize_input_skeleton(in_joints, save_path, sample_idx=0, config=None, device='cuda:0'):
    """
    Visualize input skeletons (both before and after masking).
    
    Args:
        in_joints: Tensor shaped (B, in_F, N*J, 4)
                   - index 0: trajectory data (traj)
                   - index 1-22: 3D pose data for 22 joints
        save_path: output file path
        sample_idx: sample index to visualize
        config: config dict (contains mask_ratio, mask_joints)
        device: device
    """
    # Take the first person (person_idx=0): (in_F, 22, 3)
    # Indices 1:23 correspond to the first person's 22 joints
    # Each person has 23 tokens (1 traj + 22 joints)
    # Matches model input: tgt_3dpose = tgt[:,:,:,1:,:3]
    person_idx = 0  # Always visualize the first person
    start_idx = person_idx * 23 + 1  # Skip traj token (index 0)
    end_idx = start_idx + 22  # 22 joints (indices 1-22)
    in_joints_3dpose = in_joints[sample_idx, :, start_idx:end_idx, :3]  # (in_F, 22, 3)
    # This visualizes the same 3D pose data (22 joints) used by the model
    
    in_F = in_joints_3dpose.shape[0]
    joints_pose = 22
    
    # Apply masking (same logic as the model)
    mask_ratio = config.get("MODEL", {}).get("mask_rate", 0.0) if config else 0.0
    mask_joints = config.get("MODEL", {}).get("mask_joints", None) if config else None
    
    # Keep a copy before masking
    orig_joints = in_joints_3dpose.clone()
    
    # Masking logic (same as the model)
    allowed_joints = torch.tensor(
        [j for j in range(joints_pose) if j != 15],
        device=device,
    )
    
    masked_indices_per_frame = []
    if mask_joints is None:
        num_mask = int(round(mask_ratio * allowed_joints.numel()))
    else:
        num_mask = int(mask_joints)
    num_mask = max(0, min(num_mask, allowed_joints.numel()))
    
    # Apply masking per frame
    for frame_idx in range(in_F):
        if num_mask > 0:
            # Keep reproducibility via a fixed seed (sample + frame)
            torch.manual_seed(sample_idx * 1000 + frame_idx)
            scores = torch.rand((allowed_joints.numel(),), device=device)
            _, idx = scores.topk(num_mask, dim=-1)
            selected = allowed_joints[idx]
            
            masked_indices = selected.cpu().numpy()
            masked_indices_per_frame.append(masked_indices)
            
            # Replace with mask token
            mask_token = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=in_joints_3dpose.dtype)
            in_joints_3dpose[frame_idx, selected] = mask_token
        else:
            masked_indices_per_frame.append(np.array([], dtype=int))
    
    # Convert to numpy
    orig_joints_np = to_numpy_array(orig_joints)  # (in_F, 22, 3)
    masked_joints_np = to_numpy_array(in_joints_3dpose)  # (in_F, 22, 3)
    
    # Convert coordinates to camera space per frame
    orig_coords_list = []
    mask_coords_list = []
    mask_coords_raw_list = []
    joint_idx = np.arange(joints_pose)
    
    for frame_idx in range(in_F):
        orig_coords = tuple(to_camera_coords(orig_joints_np[frame_idx]))
        mask_coords_raw = tuple(to_camera_coords(masked_joints_np[frame_idx]))
        
        masked_ids = np.asarray(masked_indices_per_frame[frame_idx], dtype=int)
        mask_coords = _mask_observed_coords(mask_coords_raw, orig_coords, masked_ids)
        
        orig_coords_list.append(orig_coords)
        mask_coords_list.append(mask_coords)
        mask_coords_raw_list.append(mask_coords_raw)
    
    # Compute global axis bounds
    global_bounds = compute_coord_axis_bounds(
        orig_coords_list + mask_coords_list + mask_coords_raw_list,
        margin=0.08,
    )
    
    # Render each frame (two rows: before and after masking)
    # Frame order: frame_idx=0 (F1) leftmost, frame_idx=in_F-1 (F9) rightmost
    orig_images = []
    mask_images = []
    
    for frame_idx in range(in_F):
        masked_ids = np.asarray(masked_indices_per_frame[frame_idx], dtype=int)
        unmasked_ids = np.setdiff1d(joint_idx, masked_ids)
        
        # Before masking (original skeleton)
        # frame_idx=0 -> F1 (first frame), frame_idx=in_F-1 -> F9 (last observed)
        orig_images.append(
            _render_tile_image(
                lambda ax, idx=frame_idx: _draw_original_view(
                    ax,
                    orig_coords_list[idx],
                    title=f"Original F{idx + 1}",
                    point_size=55,
                    edge_width=1.1,
                    alpha=0.9,
                    axis_bounds=global_bounds,
                ),
                figsize=(4, 4),
            )
        )
        
        # After masking (masked skeleton)
        mask_images.append(
            _render_tile_image(
                lambda ax, idx=frame_idx: _draw_mask_view(
                    ax,
                    orig_coords_list[idx],
                    mask_coords_list[idx],
                    mask_coords_raw_list[idx],
                    masked_ids,
                    unmasked_ids,
                    title=f"Masked F{idx + 1}",
                    visible_size=45,
                    masked_size=58,
                    token_size=32,
                    visible_edge_width=1.5,
                    masked_edge_width=0.9,
                    axis_bounds=global_bounds,
                ),
                figsize=(4, 4),
            )
        )
    
    # Crop images to a common size
    orig_images, mask_images = _crop_image_lists_uniform([orig_images, mask_images])
    
    # Stack into two rows (row1: original, row2: masked)
    row1 = np.concatenate(orig_images, axis=1)
    row2 = np.concatenate(mask_images, axis=1)
    final = np.concatenate([row1, row2], axis=0)
    
    # Save
    import matplotlib.pyplot as plt
    plt.imsave(save_path, final)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,  help="checkpoint path")
    parser.add_argument("--split", type=str, default="test", help="Split to use. one of [train, test, valid]")
    parser.add_argument("--metric", type=str, default="vim", help="Evaluation metric. One of (vim, mpjpe)")

    parser.add_argument("--mask_joints", type=int, default=None, help="Override MODEL.mask_joints during evaluation")
    parser.add_argument("--mask_rate", type=float, default=None, help="Override MODEL.mask_rate during evaluation")
    parser.add_argument("--output", type=str, default=None, help="Path to save predictions as JSON")
    parser.add_argument("--vis_dir", type=str, default=None, help="Directory to save input skeleton visualizations")
    parser.add_argument("--vis_sample_idx", type=int, nargs="+", default=None, help="Sample indices to visualize (e.g., --vis_sample_idx 0 5 10)")
    
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
    model.load_state_dict(ckpt['model'])
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
    
    # Create visualization output directory
    vis_dir = None
    if args.vis_dir is not None:
        vis_dir = args.vis_dir
        os.makedirs(vis_dir, exist_ok=True)
        logger.info(f"Input skeleton visualizations will be saved to: {vis_dir}")
    
    start_time = time.perf_counter()
    ade, fde, sample_num, batch_id, all_predictions = evaluate_ade_fde(
        model, dataloader, bs, config, logger, return_all=True, 
        save_predictions=(args.output is not None),
        visualize_input=(args.vis_dir is not None),
        vis_dir=vis_dir,
        vis_sample_indices=args.vis_sample_idx
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
    
    # Save predictions to JSON
    if args.output is not None and all_predictions is not None:
        with open(args.output, 'w') as f:
            json.dump(all_predictions, f)
        print(f'Predictions saved to: {args.output}')
    

