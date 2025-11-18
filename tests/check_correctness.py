import os
import torch
import argparse

"""
BASICALLY THIS GOES THROUGH ALL OF THE RANK FILES IN LOGS/PROBLEM_1/PROBLEMS OR SOLUTIONS AND COMPARES: THE TENSORS OR .TXT FILES NEED TO BE THE SAME
"""

# Hardcoded directories
txt_dir = "/home/willychan/other_projs/ParallelKernelBench/logs/problem_1/solution"
pt_dir = "/home/willychan/other_projs/ParallelKernelBench/logs/problem_1/reference"

# Find all rank_*.txt files
txt_files = [f for f in os.listdir(txt_dir) if f.startswith("rank_") and f.endswith(".txt")]


def main(problem_id):
    # Compare each rank
    for txt_file in txt_files:
        rank = txt_file.split(".")[0]  # e.g., "rank_0"
        pt_file = f"{rank}.pt"
        txt_path = os.path.join(txt_dir, txt_file)
        pt_path = os.path.join(pt_dir, pt_file)
        if not os.path.exists(pt_path):
            print(f"[ERROR] {pt_file} not found for {txt_file}")
            continue
        # Read all ints from .txt file (one per line, or space-separated)
        with open(txt_path, "r") as f:
            txt_values = []
            for line in f:
                # Support both space-separated and newline-separated
                txt_values.extend([int(x) for x in line.strip().split() if x])
        # Load tensor from .pt file
        tensor = torch.load(pt_path)
        tensor_flat = tensor.flatten().cpu().numpy()
        pt_values = tensor_flat.tolist()

        print(f"First 10 vals for {rank}: {pt_values[:10]}")
        # Compare lengths first
        if len(txt_values) != len(pt_values):
            print(f"[MISMATCH] {rank}: txt has {len(txt_values)} values, pt has {len(pt_values)} values")
            continue
        # Compare elementwise
        mismatches = []
        for i, (txt_v, pt_v) in enumerate(zip(txt_values, pt_values)):
            if txt_v != pt_v:
                mismatches.append((i, txt_v, pt_v))
        if not mismatches:
            print(f"[OK] {rank}: all {len(txt_values)} values match")
        else:
            print(f"[MISMATCH] {rank}: {len(mismatches)} mismatches")
            for idx, txt_v, pt_v in mismatches[:10]:  # Show up to 10 mismatches
                print(f"  idx {idx}: txt={txt_v} != pt={pt_v}")
            if len(mismatches) > 10:
                print(f"  ... and {len(mismatches)-10} more mismatches")
    print("Comparison complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checks the correctness for a specific problem_id in the logs folder.")
    parser.add_argument("--problem_id", type=int, default=1, help="The name of the problem to check.")

    args = parser.parse_args()

    main(args.problem_id)