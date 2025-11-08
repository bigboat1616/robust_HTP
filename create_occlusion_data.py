#!/usr/bin/env python3
"""
Occlusion simulation script for JTA 3D pose data.
This script creates occlusion scenarios by setting specific body parts to zero.
"""

import json
import os
import glob
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

# Occlusion scenarios - 5 body parts
OCCLUSION_SCENARIOS = {
    "head": [0, 1, 2],  # head_top, head_center, neck
    "right_arm": [3, 4, 5, 6],  # right_clavicle, right_shoulder, right_elbow, right_wrist
    "left_arm": [7, 8, 9, 10],  # left_clavicle, left_shoulder, left_elbow, left_wrist
    "torso": [11, 12, 13, 14, 15],  # spine0-4
    "right_leg": [16, 17, 18],  # right_hip, right_knee, right_ankle
    "left_leg": [19, 20, 21]  # left_hip, left_knee, left_ankle
}

def apply_occlusion_to_track(track_data, occluded_parts):
    """
    Apply occlusion to specific body parts by setting their coordinates to zero.
    
    Args:
        track_data (dict): Track data containing x0-x21, y0-y21, z0-z21 coordinates
        occluded_parts (list): List of body part indices to occlude
    
    Returns:
        dict: Modified track data with occluded parts set to zero
    """
    modified_track = track_data.copy()
    
    for part_idx in occluded_parts:
        # Set x, y, z coordinates to 0 for the occluded part
        modified_track[f"x{part_idx}"] = 0.0
        modified_track[f"y{part_idx}"] = 0.0
        modified_track[f"z{part_idx}"] = 0.0
    
    return modified_track

def process_file(input_file, output_dir, occlusion_scenario_name, occluded_parts):
    """
    Process a single input file and create occluded version.
    
    Args:
        input_file (str): Path to input file
        output_dir (str): Output directory path
        occlusion_scenario_name (str): Name of the occlusion scenario
        occluded_parts (list): List of body part indices to occlude
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
                    # Apply occlusion to track data
                    data["track"] = apply_occlusion_to_track(data["track"], occluded_parts)
                
                # Write the modified data
                f_out.write(json.dumps(data) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error at line {line_num + 1}: {e}")
                # Write the original line if there's an error
                f_out.write(line + '\n')

def main():
    """Main function to process all test_occlusion files."""
    input_dir = "data/jta_3dp/test_occlusion"
    base_output_dir = "data/jta_3dp/test_occlusion_modified"
    
    # Create output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Get all input files
    input_files = glob.glob(os.path.join(input_dir, "coords_traj_seq_*.ndjson"))
    
    if not input_files:
        print(f"No input files found in {input_dir}")
        return
    
    print(f"Found {len(input_files)} input files")
    
    # Process each occlusion scenario
    for scenario_name, occluded_parts in OCCLUSION_SCENARIOS.items():
        print(f"\n=== Processing occlusion scenario: {scenario_name} ===")
        print(f"Occluding parts: {[BODY_PARTS[i] for i in occluded_parts]}")
        
        # Create scenario-specific output directory
        scenario_output_dir = os.path.join(base_output_dir, scenario_name)
        os.makedirs(scenario_output_dir, exist_ok=True)
        
        # Process all files for this scenario
        for input_file in input_files:
            process_file(input_file, scenario_output_dir, scenario_name, occluded_parts)
    
    print(f"\n=== Processing complete ===")
    print(f"Output files saved to: {base_output_dir}")
    print(f"Created scenarios: {list(OCCLUSION_SCENARIOS.keys())}")

if __name__ == "__main__":
    main() 