#!/usr/bin/env python3
"""
Generate and evaluate CUDA kernel solutions using OpenAI API.

Usage:
    python gen_and_eval.py --problem_id 1 --topology path/to/topology.json [--model gpt-4o] [--api_key YOUR_KEY]

    python gen_and_eval.py --problem_id 1 --topology /Users/willychan/Desktop/projects/parallel_KernelBench/topo_dump_parsed.json
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess
import shutil
import time
import re
from typing import Optional, Tuple

# Import construct_prompt from utils directory
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / "utils"))
from construct_prompt import construct_prompt


def call_openai_api(prompt: str, model: str = "gpt-4o", api_key: str = None) -> str:
    """Call OpenAI API to generate CUDA kernel code."""
    try:
        from openai import OpenAI
    except ImportError as e:
        print(f"Python executable: {sys.executable}", file=sys.stderr)
        print(f"Python version: {sys.version}", file=sys.stderr)
        print(f"Import error details: {e}", file=sys.stderr)
        raise ImportError(
            f"openai library not found. Install with: pip install openai\n"
            f"Current Python: {sys.executable}\n"
            f"Error: {e}"
        )
    
    # Get API key from argument, environment variable, or error
    if api_key:
        client = OpenAI(api_key=api_key)
    elif "OPENAI_API_KEY" in os.environ:
        client = OpenAI()
    else:
        raise ValueError(
            "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
            "or use --api_key argument."
        )
    
    print(f"Calling OpenAI API with model: {model}...", file=sys.stderr)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert CUDA and NVSHMEM programmer. Return ONLY the .cu source code, no markdown formatting or explanations."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,  # Lower temperature for more deterministic code generation
    )
    
    generated_code = response.choices[0].message.content.strip()
    
    # Remove markdown code fences if present
    if generated_code.startswith("```"):
        lines = generated_code.split("\n")
        # Remove first line (```cuda or ```cpp)
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        generated_code = "\n".join(lines)
    
    return generated_code






####################################################################################
# NEW EVALUATION STUFF: START
####################################################################################
def _is_exe_available(name: str) -> bool:
    return shutil.which(name) is not None


def _write_sbatch_script(
    root_dir: Path,
    job_name: str,
    nodes: int,
    gpus: int,
    time_limit: str,
    partition: Optional[str],
    m: int,
    n: int,
    dtype: str,
    iters: int,
    warmup: int,
) -> Tuple[Path, Path, str]:
    scripts_dir = root_dir / "scripts"
    logs_dir = root_dir / "logs"
    scripts_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    output_template = logs_dir / f"{job_name}_%j.out"
    sbatch_file = scripts_dir / f"{job_name}.sbatch"

    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -N {nodes}",
        f"#SBATCH --ntasks-per-node={gpus}",
        f"#SBATCH --gres=gpu:{gpus}",
        f"#SBATCH -t {time_limit}",
        f"#SBATCH -o {output_template}",
        f"#SBATCH -D {root_dir}",
        "",
        "set -euo pipefail",
        "export OMP_NUM_THREADS=1",
        "",
        (
            f"srun --ntasks-per-node={gpus} --ntasks={gpus*nodes} --gpus-per-task=1 "
            f"python -u tests/run_1.py --m {m} --n {n} --dtype {dtype} --iters {iters} --warmup {warmup}"
        ),
        "",
    ]
    if partition:
        lines.insert(6, f"#SBATCH -p {partition}")

    sbatch_file.write_text("\n".join(lines))
    return sbatch_file, output_template, job_name


def _submit_and_wait_sbatch(sbatch_path: Path, output_template: Path) -> Tuple[str, Path]:
    proc = subprocess.run(["sbatch", str(sbatch_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed: {proc.stderr.strip()}")
    m = re.search(r"Submitted batch job (\d+)", proc.stdout)
    if not m:
        raise RuntimeError(f"Unable to parse job id from sbatch output: {proc.stdout.strip()}")
    job_id = m.group(1)
    out_file = Path(str(output_template).replace("%j", job_id))

    while True:
        q = subprocess.run(["squeue", "-h", "-j", job_id], capture_output=True, text=True)
        if q.returncode != 0:
            break
        if not q.stdout.strip():
            break
        time.sleep(3)

    for _ in range(30):
        if out_file.exists() and out_file.stat().st_size > 0:
            break
        time.sleep(1)
    return job_id, out_file


def _parse_run1_output(text: str) -> dict:
    metrics = {}
    m = re.search(r"Correctness:\s+(PASS|FAIL)", text)
    if m:
        metrics["correctness"] = m.group(1)
    m = re.search(r"Target avg latency:\s+([0-9.]+) ms", text)
    if m:
        metrics["target_ms"] = float(m.group(1))
    m = re.search(r"Baseline avg latency:\s+([0-9.]+) ms", text)
    if m:
        metrics["baseline_ms"] = float(m.group(1))
    m = re.search(r"Speedup \(baseline/target\):\s+([0-9.]+)x", text)
    if m:
        metrics["speedup"] = float(m.group(1))
    return metrics


def run_eval(
    root_dir: Path,
    use_sbatch: bool,
    nodes: int,
    gpus: int,
    m: int,
    n: int,
    dtype: str,
    iters: int,
    warmup: int,
    partition: Optional[str],
    time_limit: str,
) -> dict:
    if use_sbatch and _is_exe_available("sbatch"):
        job_name = "eval_run_1"
        sbatch_file, out_tmpl, _ = _write_sbatch_script(
            root_dir=root_dir,
            job_name=job_name,
            nodes=nodes,
            gpus=gpus,
            time_limit=time_limit,
            partition=partition,
            m=m,
            n=n,
            dtype=dtype,
            iters=iters,
            warmup=warmup,
        )
        print(f"Submitting sbatch: {sbatch_file}", file=sys.stderr)
        job_id, out_file = _submit_and_wait_sbatch(sbatch_file, out_tmpl)
        print(f"\u2713 Job {job_id} finished. Output: {out_file}", file=sys.stderr)
        try:
            content = out_file.read_text()
        except Exception as e:
            raise RuntimeError(f"Failed to read job output {out_file}: {e}")
        return _parse_run1_output(content)

    if not _is_exe_available("torchrun"):
        raise RuntimeError("Neither sbatch nor torchrun is available to execute the benchmark.")

    cmd = [
        "torchrun",
        f"--nproc_per_node={gpus}",
        "tests/run_1.py",
        "--m", str(m),
        "--n", str(n),
        "--dtype", dtype,
        "--iters", str(iters),
        "--warmup", str(warmup),
    ]
    print(f"Running locally: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, cwd=str(root_dir), capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Local run failed with code {proc.returncode}: {proc.stdout}\n{proc.stderr}")
    return _parse_run1_output(proc.stdout)




####################################################################################
# NEW EVALUATION STUFF: END
####################################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Generate CUDA kernel solution using OpenAI API"
    )
    parser.add_argument(
        "--problem_id",
        required=True,
        help="Problem ID (e.g., '1' for problems/1.py)"
    )
    parser.add_argument(
        "--topology",
        required=True,
        help="Path to topology JSON file"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--solutions_dir",
        default="solutions",
        help="Directory to save generated .cu files (default: solutions/)"
    )
    parser.add_argument("--eval", action="store_true", help="Run tests/run_1.py after generation")
    parser.add_argument("--nodes", type=int, default=1, help="Nodes for evaluation (default: 1)")
    parser.add_argument("--gpus", type=int, default=8, help="GPUs per node for evaluation (default: 8)")
    parser.add_argument("--m", type=int, default=1024, help="Tensor rows for benchmark")
    parser.add_argument("--n", type=int, default=1024, help="Tensor cols for benchmark")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16", "float64"], help="Tensor dtype for benchmark")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--use_sbatch", action="store_true", help="Prefer sbatch for evaluation")
    parser.add_argument("--partition", type=str, default=None, help="Slurm partition for sbatch")
    parser.add_argument("--time_limit", type=str, default="00:15:00", help="Slurm time limit (e.g., 00:15:00)")
    args = parser.parse_args()
    
    # Construct prompt
    prompt = construct_prompt(problem_id=args.problem_id, topology=args.topology)
    
    print(f"Generated prompt ({len(prompt)} chars)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    
    # Call OpenAI API
    try:
        generated_code = call_openai_api(prompt, model=args.model, api_key=args.api_key)
    except Exception as e:
        print(f"Error calling OpenAI API: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Save to solutions directory (relative to script directory)
    solutions_dir = script_dir / args.solutions_dir
    solutions_dir.mkdir(exist_ok=True)
    
    output_file = solutions_dir / f"{args.problem_id}.cu"
    
    with open(output_file, "w") as f:
        f.write(generated_code)
    
    print(f"✓ Saved generated kernel to: {output_file}", file=sys.stderr)
    print(f"✓ File size: {len(generated_code)} bytes", file=sys.stderr)

    # if args.eval:
    #     try:
    #         metrics = run_eval(
    #             root_dir=script_dir,
    #             use_sbatch=args.use_sbatch,
    #             nodes=args.nodes,
    #             gpus=args.gpus,
    #             m=args.m,
    #             n=args.n,
    #             dtype=args.dtype,
    #             iters=args.iters,
    #             warmup=args.warmup,
    #             partition=args.partition,
    #             time_limit=args.time_limit,
    #         )
    #     except Exception as e:
    #         print(f"Error during evaluation: {e}", file=sys.stderr)
    #         sys.exit(2)

    #     print("=== Evaluation Summary ===")
    #     if "correctness" in metrics:
    #         print(f"Correctness: {metrics['correctness']}")
    #     if "target_ms" in metrics:
    #         print(f"Target avg latency: {metrics['target_ms']:.3f} ms")
    #     if "baseline_ms" in metrics:
    #         print(f"Baseline avg latency: {metrics['baseline_ms']:.3f} ms")
    #     if "speedup" in metrics:
    #         print(f"Speedup (baseline/target): {metrics['speedup']:.3f}x")


if __name__ == "__main__":
    main()

