#!/bin/bash 
#SBATCH --job-name=pytorch_distributed 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00 
#SBATCH --output=/home/ubuntu/nathanjp/ParallelKernelBench/logs/output_%j.log 
#SBATCH --error=/home/ubuntu/nathanjp/ParallelKernelBench/logs/error_%j.log 

########################################################################
# Parse command-line arguments
########################################################################
# Usage: sbatch run_reference.sh [PROBLEM_ID] [M] [N] [DTYPE] [LEVEL]
# Example: sbatch run_reference.sh 2 2048 2048 float32 1
# Defaults: PROBLEM_ID=1, M=1024, N=1024, DTYPE=float32, LEVEL=1

PROBLEM_ID=${1:-1}
M=${2:-1024}
N=${3:-1024}
DTYPE=${4:-float32}
LEVEL=${5:-1}

########################################################################
# Part 0: Set up environment variables as needed
########################################################################

# Set rendezvous for NCCL env:// init used by tests/run_reference.py
source ~/activate_conda_aarch.sh 

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((15000 + ($SLURM_JOB_ID % 20000)))

export CUDA_HOME=$HOME/cuda-12.9                                                            
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export NVSHMEM_HOME=/home/willychan/other_projs/libnvshmem-linux-x86_64-3.4.5_cuda12-archive

echo "=========================================="
echo "Running Problem ID: $PROBLEM_ID"
echo "Level: $LEVEL"
echo "Tensor shape: [$M, $N]"
echo "Data type: $DTYPE"
echo "=========================================="

# Submit reference pytorch function with arguments
srun --export=ALL bash -lc "
  CONDA_ROOT=\"/home/ubuntu/aarch_miniforge3/\"
  ENV_NAME=\"\${ENV_NAME:-pkb}\"
  if [ -n \"\$CONDA_ROOT\" ]; then
    \"\$CONDA_ROOT/bin/conda\" run -n \"\$ENV_NAME\" python /home/ubuntu/nathanjp/ParallelKernelBench/tests/run_reference.py --level $LEVEL --problem_id $PROBLEM_ID --m $M --n $N --dtype $DTYPE
  else
    echo \"Warning: conda not found on node \$(hostname); using system python\" >&2
    python3 /home/ubuntu/nathanjp/ParallelKernelBench/tests/run_reference.py --level $LEVEL --problem_id $PROBLEM_ID --m $M --n $N --dtype $DTYPE
  fi
"