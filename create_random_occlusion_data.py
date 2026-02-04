#!/usr/bin/env python3
"""
Random occlusion simulation script for JTA 3D pose data.
This script creates occlusion scenarios by randomly removing 1-21 joints.
"""

import json
import os
import glob
import random
from pathlib import Path

# Body part definitions (0-21 joints)
BODY_PARTS = {
    0: "head_top",
    1: "head_center", 
    2: "neck",
    3: "right_clavicle",
    4: "right_shoulder",
    5: "right_elbow",
    6: "right_wrist",
    7: "left_clavicle",
    8: "left_shoulder",
    9: "left_elbow",
    10: "left_wrist",
    11: "spine0",
    12: "spine1",
    13: "spine2",
    14: "spine3",
    15: "spine4",
    16: "right_hip",
    17: "right_knee",
    18: "right_ankle",
    19: "left_hip",
    20: "left_knee",
    21: "left_ankle"
}

MASK_TOKEN = {
    "x": -0.0278,
    "y":  0.0141,
    "z":  0.0044
}

def apply_random_occlusion_to_track(
    track_data,
    num_joints_to_remove,
    seed=None,
    fill_mode="mask",
):
    """
    Apply random occlusion to track data by setting randomly selected joints to a fill mode.
    
    Args:
        track_data (dict): Track data containing x0-x21, y0-y21, z0-z21 coordinates
        num_joints_to_remove (int): Number of joints to randomly remove (1-21)
        seed (int): Random seed for reproducibility
        fill_mode (str): How to fill occluded joints, either "mask" or "zero"
    
    Returns:
        dict: Modified track data with randomly occluded joints set to selected fill mode
    """
    if seed is not None:
        random.seed(seed)
    
    modified_track = track_data.copy()
    
    # Randomly select joints to occlude
    all_joints = list(BODY_PARTS.keys())  # 0-21 joints
    candidate_joints = [j for j in all_joints if j != 15]
    occluded_joints = random.sample(candidate_joints, num_joints_to_remove)
    for joint_idx in occluded_joints:
        if fill_mode == "mask":
            modified_track[f"x{joint_idx}"] = MASK_TOKEN["x"]
            modified_track[f"y{joint_idx}"] = MASK_TOKEN["y"]
            modified_track[f"z{joint_idx}"] = MASK_TOKEN["z"]
        elif fill_mode == "zero":
            modified_track[f"x{joint_idx}"] = 0.0
            modified_track[f"y{joint_idx}"] = 0.0
            modified_track[f"z{joint_idx}"] = 0.0
        else:
            raise ValueError(f"Unknown fill_mode '{fill_mode}'. Expected 'mask' or 'zero'.")
    
    return modified_track, occluded_joints

def process_file_random_occlusion(
    input_file,
    output_dir,
    occlusion_scenario_name,
    num_joints_to_remove,
    seed=None,
    fill_mode="mask",
):
    """
    Process a single input file and create randomly occluded version.
    
    Args:
        input_file (str): Path to input file
        output_dir (str): Output directory path
        occlusion_scenario_name (str): Name of the occlusion scenario
        num_joints_to_remove (int): Number of joints to randomly remove
        seed (int): Random seed for reproducibility
        fill_mode (str): How to fill occluded joints, either "mask" or "zero"
    """
    # Create output filename
    input_filename = Path(input_file).name
    output_filename = f"{input_filename.replace('.ndjson', '')}_{occlusion_scenario_name}.ndjson"
    output_file = os.path.join(output_dir, output_filename)
    
    print(f"Processing {input_file} -> {output_file}")
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line_num, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # Check if this is a track entry
                if "track" in data:
                    # Apply random occlusion to track data
                    # Use line number as part of seed for consistency across runs
                    track_seed = seed + line_num if seed is not None else None
                    data["track"], occluded_joints = apply_random_occlusion_to_track(
                        data["track"],
                        num_joints_to_remove,
                        track_seed,
                        fill_mode=fill_mode,
                    )
                
                # Write the modified data
                f_out.write(json.dumps(data) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error at line {line_num + 1}: {e}")
                # Write the original line if there's an error
                f_out.write(line + '\n')

def main():
    """Main function to process all test_occlusion files with random occlusion."""
    input_dir = "data/jta_3dp/test"
    base_output_dir = "data/jta_3dp/test_occlusion"
    
    # Create output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Get all input files
    input_files = glob.glob(os.path.join(input_dir, "coords_traj_seq_*.ndjson"))
    
    if not input_files:
        print(f"No input files found in {input_dir}")
        return
    
    print(f"Found {len(input_files)} input files")
    
    # Process each random occlusion scenario for 1-21 joints
    for num_joints in range(1, 22):
        joint_label = "joint" if num_joints == 1 else "joints"
        base_scenario_name = f"random_{num_joints}{joint_label}"
        scenario_seed = hash(base_scenario_name) % 10000
        
        print(f"\n=== Processing random occlusion scenario: {base_scenario_name} ===")
        print(f"Randomly removing {num_joints} joint(s)")
        
        for fill_mode, suffix in (("mask", "mask"), ("zero", "0")):
            scenario_name = f"{base_scenario_name}_{suffix}"
            scenario_output_dir = os.path.join(base_output_dir, scenario_name)
            os.makedirs(scenario_output_dir, exist_ok=True)
            
            print(f"  -> Saving to {scenario_output_dir} with fill mode '{fill_mode}'")
            
            for input_file in input_files:
                process_file_random_occlusion(
                    input_file,
                    scenario_output_dir,
                    scenario_name,
                    num_joints,
                    seed=scenario_seed,
                    fill_mode=fill_mode,
                )
    
    print(f"\n=== Processing complete ===")
    print(f"Output files saved to: {base_output_dir}")
    print("Created scenarios for 1-22 joints with '_mask' and '_0' variants.")

if __name__ == "__main__":
    main() 