#!/bin/bash
#SBATCH -J eval_run_1
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH -t 00:15:00
#SBATCH -o /home/willychan/other_projs/ParallelKernelBench/logs/eval_run_1_%j.out
#SBATCH -D /home/willychan/other_projs/ParallelKernelBench

set -euo pipefail
export OMP_NUM_THREADS=1

srun --ntasks-per-node=8 --ntasks=8 --gpus-per-task=1 python -u tests/run_1.py --m 1024 --n 1024 --dtype float32 --iters 50 --warmup 10
