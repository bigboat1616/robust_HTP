import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from dataset_jrdb import batch_process_coords, create_dataset, collate_batch
from model_jrdb_traj import create_model
from model_jrdb_original import create_model
from utils.utils import create_logger
from torch.utils.data import DataLoader

def load_model(ckpt_path, device="cpu"):
    """Load a model."""
    logger = create_logger('')
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']
    config["DEVICE"] = device
    
    # Determine model type (from config or checkpoint path)
    if "traj" in ckpt_path:
        from model_jrdb_traj import create_model  # Trajectory-only model
    else:
        from model_jrdb_original import create_model  # Original model (with 2D BB)
    
    model = create_model(config, logger)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, config

def get_predictions(model, config, joints, masks, padding_mask):
    """Get model predictions."""
    padding_mask = padding_mask.to(config["DEVICE"])
    in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(
        joints, masks, padding_mask, config, modality_selection='traj'
    )
    
    with torch.no_grad():
        pred_joints = model(in_joints, padding_mask)
        pred_joints = pred_joints[:,-config['TRAIN']['output_track_size']:]
    
    return in_joints.cpu(), out_joints.cpu(), pred_joints.cpu()

def plot_trajectories(gt_xy, pred_xys, obs_xy, person_id, model_names, save_path="trajectory_plots"):
    """Plot predictions from multiple models."""
    plt.figure(figsize=(10, 10))
    
    # Scale settings
    all_trajectories = [obs_xy, gt_xy] + pred_xys
    all_x = np.concatenate([traj[:,0] for traj in all_trajectories])
    all_y = np.concatenate([traj[:,1] for traj in all_trajectories])
    margin = 0.5
    x_min, x_max = np.min(all_x) - margin, np.max(all_x) + margin
    y_min, y_max = np.min(all_y) - margin, np.max(all_y) + margin
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Observed trajectory (past)
    plt.plot(obs_xy[:,0], obs_xy[:,1], 'b.-', label='Observed', alpha=0.5)
    
    # Ground truth (future)
    gt_with_start = np.vstack([obs_xy[-1:], gt_xy])
    plt.plot(gt_with_start[:,0], gt_with_start[:,1], 'g.-', label='Ground Truth', alpha=0.5)
    
    # Predictions per model
    colors = ['r', 'm', 'c', 'y']  # Add more colors if needed
    for i, pred_xy in enumerate(pred_xys):
        pred_with_start = np.vstack([obs_xy[-1:], pred_xy])
        plt.plot(pred_with_start[:,0], pred_with_start[:,1], 
                f'{colors[i]}.-', label=f'{model_names[i]}', alpha=0.5)
        # Mark end point
        plt.plot(pred_xy[-1,0], pred_xy[-1,1], f'{colors[i]}*', 
                label=f'{model_names[i]} End')
    
    
    gt_with_start = np.vstack([obs_xy[-1:], gt_xy])

    
    # Start and end points
    plt.plot(obs_xy[0,0], obs_xy[0,1], 'ko', label='Start')
    plt.plot(gt_xy[-1,0], gt_xy[-1,1], 'k*', label='GT End')
    
    plt.title(f'Trajectory for Person {person_id}')
    plt.xlabel('X coordinate [m]')
    plt.ylabel('Y coordinate [m]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/trajectory_person_{person_id}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs='+', required=True, help="checkpoint paths")
    parser.add_argument("--model_names", nargs='+', help="names for each model")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="trajectory_plots")
    parser.add_argument("--num_samples", type=int, default=10, help="number of samples to plot")
    args = parser.parse_args()
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model name settings
    if not args.model_names:
        args.model_names = [f"Model_{i}" for i in range(len(args.ckpts))]
    assert len(args.ckpts) == len(args.model_names), "Number of checkpoints and model names must match"
    
    # Load models
    models_and_configs = [load_model(ckpt, device) for ckpt in args.ckpts]
    models, configs = zip(*models_and_configs)
    
    # Create dataset
    logger = create_logger('')
    dataset = create_dataset(
        configs[0]['DATA']['train_datasets'][0],
        logger,
        split=args.split,
        track_size=(configs[0]['TRAIN']['input_track_size'] + configs[0]['TRAIN']['output_track_size']),
        track_cutoff=configs[0]['TRAIN']['input_track_size']
    )
    
    # DataLoader setup
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one sample at a time
        shuffle=False,
        collate_fn=collate_batch
    )
    
    sample_count = 0
    for batch in dataloader:
        if sample_count >= args.num_samples:
            break
            
        joints, masks, padding_mask = batch
        
        # Get predictions per model
        all_predictions = []
        for model, config in zip(models, configs):
            in_joints, out_joints, pred = get_predictions(model, config, joints, masks, padding_mask)
            if len(all_predictions) == 0:  # First model only
                obs_xy = in_joints[0,:,0,:2]  # Observed data
                gt_xy = out_joints[0,:,0,:2]  # Ground Truth
            pred_xy = pred[0].reshape(out_joints.size(1), 1, 2)[:,0,:2]
            all_predictions.append(pred_xy)
        
        # Padding check
        if padding_mask[0].sum() == padding_mask[0].shape[0]:
            continue
            
        # Plot
        plot_trajectories(
            gt_xy.numpy(),
            [p.numpy() for p in all_predictions],
            obs_xy.numpy(),
            person_id=sample_count,
            model_names=args.model_names,
            save_path=args.output_dir
        )
        
        sample_count += 1

if __name__ == "__main__":
    main()