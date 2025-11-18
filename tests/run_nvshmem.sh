
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

# export NVSHMEM_BOOTSTRAP=MPI

# # Compile
nvcc -std=c++20 -arch=sm_80 -rdc=true \
    -I/opt/amazon/openmpi5/include \
    -I${NVSHMEM_HOME}/include \
    -L${NVSHMEM_HOME}/lib -lnvshmem \
    -o /home/willychan/other_projs/ParallelKernelBench/solutions/my_nvshmem_app /home/willychan/other_projs/ParallelKernelBench/tests/run_nvshmem.cu

# Run with nvshmrun
${NVSHMEM_HOME}/bin/nvshmrun -np 8 /home/willychan/other_projs/ParallelKernelBench/solutions/my_nvshmem_app

# nvcc -std=c++20 -arch=sm_80 -rdc=true \
#     -I/opt/amazon/openmpi5/include \
#     -I${NVSHMEM_HOME}/include \
#     -L${NVSHMEM_HOME}/lib -lnvshmem \
#     -o /home/willychan/other_projs/ParallelKernelBench/solutions/my_nvshmem_app /home/willychan/other_projs/ParallelKernelBench/tests/run_nvshmem.cu && ${NVSHMEM_HOME}/bin/nvshmrun -np 8 /home/willychan/other_projs/ParallelKernelBench/solutions/my_nvshmem_app

