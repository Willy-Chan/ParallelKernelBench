#!/bin/bash 
#SBATCH --job-name=triton_distributed 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00 
#SBATCH --output=logs/output_triton_%j.log 
#SBATCH --error=logs/error_triton_%j.log 

########################################################################
# Part 0: Set up environment variables as needed
########################################################################

# Set rendezvous for NCCL env:// init
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((15000 + ($SLURM_JOB_ID % 20000)))

# Example CUDA setup (adjust paths as needed)
export CUDA_HOME=$HOME/cuda-12.9                                                            
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Submit Triton solution
srun --export=ALL bash -lc '
  ENV_NAME="${ENV_NAME:-pkb}"
  CONDA_ROOT="$HOME/miniconda3"
  
  # Determine project root assuming script is submitted from project root
  PROJECT_ROOT=$(pwd)
  
  if [ -n "$CONDA_ROOT" ]; then
    "$CONDA_ROOT/bin/conda" run -n "$ENV_NAME" python "$PROJECT_ROOT/tests/run_reference.py" --problem_py "$PROJECT_ROOT/solutions/1_tritondist.py" --log_subdir triton
  else
    python3 "$PROJECT_ROOT/tests/run_reference.py" --problem_py "$PROJECT_ROOT/solutions/1_tritondist.py" --log_subdir triton
  fi
'

