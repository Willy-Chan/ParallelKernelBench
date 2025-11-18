
import os
import sys
import torch
import torch.utils.cpp_extension
import nvshmem

# Ensure we can find the solution file
# Assuming script is run from project root or tests/
# We'll locate solutions/1_nvshmem.cu relative to this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SOLUTION_FILE = os.path.join(PROJECT_ROOT, "solutions", "1_nvshmem.cu")

if not os.path.exists(SOLUTION_FILE):
    raise FileNotFoundError(f"Could not find solution file at {SOLUTION_FILE}")

NVSHMEM_HOME = os.environ.get("NVSHMEM_HOME")
if not NVSHMEM_HOME:
    # Try to guess or error
    # Fallback to the path seen in run_nvshmem.sh if not set
    NVSHMEM_HOME = "/home/willychan/other_projs/libnvshmem-linux-x86_64-3.4.5_cuda12-archive"
    print(f"NVSHMEM_HOME not set, using default: {NVSHMEM_HOME}")

# Read the kernel source
with open(SOLUTION_FILE, "r") as f:
    kernel_source = f.read()

# Define the extension source code
cpp_source = f"""
#include <torch/extension.h>
#include <stdio.h>

// nvshmem headers should be available in include path
#include "nvshmem.h"
#include "nvshmemx.h"

// --- Kernel Code Start ---
{kernel_source}
// --- Kernel Code End ---

void launch_solution(torch::Tensor tensor) {{
    // Ensure tensor is int32
    if (tensor.dtype() != torch::kInt32) {{
        throw std::runtime_error("Tensor must be int32");
    }}
    
    int* d_ptr = tensor.data_ptr<int>();
    void *args[] = {{&d_ptr}};
    
    dim3 dimBlock(1);
    dim3 dimGrid(1);
    
    // Launch the kernel using nvshmem collective launch
    // solution is the kernel name from 1_nvshmem.cu
    nvshmemx_collective_launch((const void *)solution, dimGrid, dimBlock, args, 0, 0);
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("launch_solution", &launch_solution, "Launch solution kernel");
}}
"""

# Compile and load the extension
# We use load_inline which builds it in a temp dir
print("Compiling NVSHMEM extension...")
nvshmem_ext = torch.utils.cpp_extension.load_inline(
    name="nvshmem_solution_ext",
    cpp_sources=[cpp_source],
    extra_cuda_cflags=["-rdc=true", "-arch=sm_80"],
    extra_ldflags=["-lnvshmem", f"-L{NVSHMEM_HOME}/lib"],
    include_dirs=[
        f"{NVSHMEM_HOME}/include",
        "/opt/amazon/openmpi5/include"
    ],
    library_dirs=[f"{NVSHMEM_HOME}/lib"],
    with_cuda=True,
    verbose=True
)

def main():
    # Initialize NVSHMEM
    # nvshmrun should have set up the environment
    nvshmem.init()
    
    mype = nvshmem.my_pe()
    npes = nvshmem.n_pes()
    
    # Determine local rank for device selection
    # nvshmem.team_my_pe(nvshmem.TEAM_NODE) might be available
    # If not, use environment variables
    try:
        # Try to access TEAM_NODE constant
        team_node = nvshmem.TEAM_NODE
        local_rank = nvshmem.team_my_pe(team_node)
    except AttributeError:
        # Fallback: assume SLURM or MPI env vars or simple mod
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
        elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        else:
            # Fallback to mod if we assume uniform distribution
            # e.g. 8 gpus per node
            local_rank = mype % 8 

    torch.cuda.set_device(local_rank)
    
    print(f"[Rank {mype}] Initialized on device {local_rank}")
    
    # Allocate symmetric memory
    # We need a tensor that resides in symmetric memory so the kernel can access it remotely
    # nvshmem.full/empty/zeros should return such a tensor
    
    tensor = None
    # Using rank as input value, similar to run_nvshmem.cu: int rank_input = mype;
    
    # Try different ways to allocate symmetric memory via nvshmem4py
    try:
        # Option 1: Top level functions
        if hasattr(nvshmem, 'full'):
            tensor = nvshmem.full((1,), mype, dtype=torch.int32, device="cuda")
        # Option 2: Interop module
        elif hasattr(nvshmem, 'interop') and hasattr(nvshmem.interop, 'torch') and hasattr(nvshmem.interop.torch, 'full'):
             tensor = nvshmem.interop.torch.full((1,), mype, dtype=torch.int32, device="cuda")
        # Option 3: Manual allocation and wrap (fallback)
        else:
            # Basic allocation if available
            # This is a guess at API if 'full' is missing
             pass
             
    except Exception as e:
        print(f"Attempt to allocate failed: {e}")

    if tensor is None:
         print("nvshmem.full or equivalent not found. Trying nvshmem.interop.torch...")
         try:
             import nvshmem.interop.torch
             tensor = nvshmem.interop.torch.full((1,), mype, dtype=torch.int32, device="cuda")
         except ImportError:
             # Try just creating a tensor on device and hoping it works? 
             # NO, it must be symmetric for nvshmem_int_p.
             # If we can't allocate symmetric memory, we can't run this kernel as is.
             print("Error: Could not find nvshmem symmetric allocation function.")
             sys.exit(1)
         except Exception as e:
             print(f"Error allocating symmetric memory: {e}")
             sys.exit(1)

    # Run the kernel
    print(f"[Rank {mype}] Running kernel...")
    nvshmem_ext.launch_solution(tensor)
    
    # Sync CUDA to ensure kernel finished
    torch.cuda.synchronize()
    
    # Barrier to ensure all ranks are done
    nvshmem.barrier_all()
    
    print(f"[Rank {mype}] Kernel finished. Result: {tensor.item()}")
    
    # Save output
    # Construct output path
    # /home/willychan/other_projs/ParallelKernelBench/logs/problem_1/solution/rank_X.pt
    output_dir = os.path.join(PROJECT_ROOT, "logs", "problem_1", "solution")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"rank_{mype}.pt")
    
    # Save as CPU tensor
    torch.save(tensor.cpu(), output_path)
    print(f"[Rank {mype}] Saved to {output_path}")

    nvshmem.finalize()

if __name__ == "__main__":
    main()

