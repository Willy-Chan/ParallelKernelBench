
#!/bin/bash
#SBATCH --job-name=nvshmem_test
#SBATCH --output=nvshmem_test.out
#SBATCH --error=nvshmem_test.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --time=00:10:00

# Load modules or set environment variables as needed
export CUDA_HOME=$HOME/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export NVSHMEM_HOME=/home/willychan/other_projs/libnvshmem-linux-x86_64-3.4.5_cuda12-archive

# Set python path if needed, or assume standard python
# Ensure pytorch and nvshmem4py are installed
# We can reuse the conda logic from run_reference.sh if desired, 
# but simpler is to assume the user runs this in the correct env.

# Path to the python script
# Assuming this script is in tests/ and run from project root or tests/
# We use absolute path for clarity
SCRIPT_PATH=$(realpath $(dirname $0)/run_nvshmem.py)

# Run with nvshmrun
# -np 8 matches ntasks
echo "Running Python NVSHMEM script: $SCRIPT_PATH"
${NVSHMEM_HOME}/bin/nvshmrun -np 8 python "$SCRIPT_PATH"
