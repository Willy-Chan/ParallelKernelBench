#!/bin/bash 
#SBATCH --job-name=pytorch_distributed 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00 
#SBATCH --output=/home/willychan/other_projs/ParallelKernelBench/logs/output_%j.log 
#SBATCH --error=/home/willychan/other_projs/ParallelKernelBench/logs/error_%j.log 

########################################################################
# Part 0: Set up environment variables as needed
########################################################################

# Set rendezvous for NCCL env:// init used by tests/run_1.py
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((15000 + ($SLURM_JOB_ID % 20000)))

export CUDA_HOME=$HOME/cuda-12.9                                                            
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export NVSHMEM_HOME=/home/willychan/other_projs/libnvshmem-linux-x86_64-3.4.5_cuda12-archive

# Submit reference pytorch function
srun --export=ALL bash -lc '
  ENV_NAME="${ENV_NAME:-pkb}"
  CONDA_ROOT="$HOME/miniconda3"

  if [ -n "$CONDA_ROOT" ]; then
    "$CONDA_ROOT/bin/conda" run -n "$ENV_NAME" python /home/willychan/other_projs/ParallelKernelBench/tests/run_reference.py
  else
    echo "Warning: conda not found on node $(hostname); using system python" >&2
    python3 /home/willychan/other_projs/ParallelKernelBench/tests/run_reference.py
  fi
'