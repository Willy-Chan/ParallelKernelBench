#!/usr/bin/env python3

import os
import argparse
import tempfile
import importlib.util
import glob
import hashlib
import shutil
import subprocess
import torch
import torch.distributed as dist


def init_dist() -> None:
    """Initialize torch.distributed with NCCL and bind CUDA device to LOCAL_RANK."""
    assert torch.cuda.is_available(), "CUDA is required"
    backend = "nccl"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", os.environ.get("SLURM_PROCID", "0"))
    os.environ.setdefault("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
    os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Robust init across torch versions
    rank = int(os.environ["RANK"]); world_size = int(os.environ["WORLD_SIZE"])
    try:
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size, device_id=local_rank)
    except TypeError:
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)


def finalize_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def load_python_solution(problem_py_path: str):
    """Dynamically load the Python solution function named 'solution' from a file."""
    problem_py_path = os.path.abspath(problem_py_path)
    assert os.path.isfile(problem_py_path), f"Problem .py not found: {problem_py_path}"
    spec = importlib.util.spec_from_file_location("problem_solution_mod", problem_py_path)
    assert spec and spec.loader, f"Failed to load spec for {problem_py_path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    assert hasattr(mod, "solution"), f"'solution' not found in {problem_py_path}"
    return getattr(mod, "solution")


def save_tensor_to_logs(t: torch.Tensor, logs_dir: str, prefix: str) -> str:
    os.makedirs(logs_dir, exist_ok=True)
    rank = dist.get_rank()
    path = os.path.join(logs_dir, f"{prefix}_rank{rank}.pt")
    torch.save(t.detach().cpu(), path)
    return path


def run_python_solution_and_save(py_fn, shape, dtype, logs_dir: str) -> str:
    """Run Python solution on per-rank input and save output to logs."""
    dev = torch.device("cuda", torch.cuda.current_device())
    rank = dist.get_rank()
    val = float(rank + 1)  # deterministic per-rank input
    x = torch.full(shape, val, dtype=dtype, device=dev)
    y = py_fn(x)
    torch.cuda.synchronize()
    dist.barrier()
    return save_tensor_to_logs(y, logs_dir, "py")


def main():
    parser = argparse.ArgumentParser(description="Run Python solution per-rank and write outputs to logs.")
    # Derive project root to set absolute defaults
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)
    default_logs = os.path.join(project_root, "logs")

    parser.add_argument("--level", type=int, default=1, help="Problem level (e.g., 1 for problems/level1)")
    parser.add_argument("--problem_id", type=int, default=1, help="Problem ID within the level (e.g., 1 for 1_allreduce.py)")
    parser.add_argument("--logs_dir", type=str, default=default_logs, help="Directory to write per-rank outputs")
    parser.add_argument("--m", type=int, default=1024, help="Tensor rows")
    parser.add_argument("--n", type=int, default=1024, help="Tensor cols (use 1 for 1D)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16", "float64"], help="Tensor dtype")
    args = parser.parse_args()

    # Resolve problem .py path from level and problem_id: problems/level{level}/{problem_id}_*.py
    problem_dir = os.path.join(project_root, "problems", f"level{args.level}")
    matches = glob.glob(os.path.join(problem_dir, f"{args.problem_id}_*.py"))
    problem_path = matches[0]

    init_dist()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"I am {rank + 1} of {world_size}")

    # Derive structured logs directory: logs/problem_<id>/reference and ensure companion 'solution' exists
    problem_basename = os.path.splitext(os.path.basename(problem_path))[0]
    problem_label = f"problem_{problem_basename}"
    logs_problem_root = os.path.join(args.logs_dir, problem_label)
    logs_reference_dir = os.path.join(logs_problem_root, "reference")
    logs_solution_dir = os.path.join(logs_problem_root, "solution")
    os.makedirs(logs_reference_dir, exist_ok=True)
    os.makedirs(logs_solution_dir, exist_ok=True)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
    }
    dtype = dtype_map[args.dtype]
    shape = (args.m, args.n)

    py_solution_fn = load_python_solution(problem_path)
    _ = run_python_solution_and_save(py_solution_fn, shape, dtype, logs_reference_dir)
    if rank == 0:
        print(f"Wrote per-rank outputs to: {logs_reference_dir}")

    dist.barrier()
    # if rank == 0:
    #     files = sorted(f for f in os.listdir(logs_solution_dir) if f.startswith("cu_rank") and f.endswith(".pt"))
    #     assert len(files) == world_size, f"Expected {world_size} files, found {len(files)} in {logs_solution_dir}"
    #     ref = torch.load(os.path.join(logs_solution_dir, files[0]))
    #     all_ranks_correct = True
    #     for fname in files[1:]:
    #         t = torch.load(os.path.join(logs_solution_dir, fname))
    #         if not torch.equal(ref, t):
    #             print(f"Mismatch: {files[0]} != {fname}")
    #             all_ranks_correct = False
    #     if all_ranks_correct:
    #         print("All rank outputs are identical.")

    finalize_dist()


if __name__ == "__main__":
    main()