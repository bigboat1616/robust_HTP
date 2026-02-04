import argparse
import json
import numpy as np

def load_last_points(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a list at the top level.")
    last_points = []
    for idx, sample in enumerate(data):
        if not sample:
            raise ValueError(f"Sample {idx} in {path} is empty.")
        last_points.append(sample[-1])
    return np.asarray(last_points, dtype=float)

def main(file_a, file_b, file_gt, threshold):
    last_a = load_last_points(file_a)
    last_b = load_last_points(file_b)
    last_gt = load_last_points(file_gt)

    if not (last_a.shape == last_b.shape == last_gt.shape):
        raise ValueError(
            f"Shape mismatch: A {last_a.shape}, B {last_b.shape}, GT {last_gt.shape}"
        )

    diff_a = np.linalg.norm(last_a - last_gt, axis=1)
    diff_b = np.linalg.norm(last_b - last_gt, axis=1)

    mask = diff_a <= threshold
    if not np.any(mask):
        print(f"No samples with A-GT L2 <= {threshold}")
        return

    candidate_indices = np.where(mask)[0]
    best_idx = candidate_indices[np.argmax(diff_b[mask])]
    print(f"Selected index: {best_idx}")
    print(f"A-GT L2: {diff_a[best_idx]:.6f}")
    print(f"B-GT L2: {diff_b[best_idx]:.6f}")
    print(f"last_a: {last_a[best_idx]}")
    print(f"last_b: {last_b[best_idx]}")
    print(f"last_gt: {last_gt[best_idx]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_a")
    parser.add_argument("file_b")
    parser.add_argument("file_gt")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args.file_a, args.file_b, args.file_gt, args.threshold)