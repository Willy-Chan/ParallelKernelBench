# run with mpirun -np 8 --oversubscribe python tests/run_numba.py

# tests/run_numba.py
import sys
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import mpi4py.MPI as MPI
import nvshmem.core
from cuda.core.experimental import Device, system
from numba import cuda as nb_cuda
import os
import torch


def _load_kernel():
    # Load solutions/1.py as a module and return a kernel callable
    repo_root = Path(__file__).resolve().parent.parent
    kernel_path = repo_root / "solutions" / "1.py"
    spec = spec_from_file_location("kernel_module", str(kernel_path))
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    # Prefer 'solution' if present, otherwise fall back to 'simple_shift_kernel'

    return getattr(mod, "simple_shift_kernel")


def solution():
    # Import kernel entry
    my_kernel = _load_kernel()

    # MPI/NVSHMEM setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    local_rank_per_node = rank % system.num_devices
    dev = Device(local_rank_per_node)
    dev.set_current()
    stream = dev.create_stream()

    nvshmem.core.init(device=dev, mpi_comm=comm, initializer_method="mpi")

    # Ring shift using NVSHMEM peer view
    my_pe = nvshmem.core.my_pe()
    npes = nvshmem.core.n_pes()
    dst_pe = (my_pe + 1) % npes
    left_pe = (my_pe - 1 + npes) % npes

    # Symmetric buffer and peer view
    array = nvshmem.core.array((1,), dtype="int32")
    peer_view = nvshmem.core.get_peer_array(array, dst_pe)

    # Launch kernel: write my_pe into dst_pe's array[0]
    # HERE'S THE ISSUE: You can't do device-side nvshmem operations in pythonic DSLs yet,  you can only allocate symmetric memory using nvshmem and pass in data pointers to symmetric memory, like you're doing in this example. With that in mind can you modify run_numba and 1.py to account for this and implement an allreduce using symmetric memory?
    my_kernel[1, 1](peer_view, my_pe)
    nb_cuda.synchronize()

    # Global barrier
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream)

    # Prepare per-rank tensor value:
    # - Input example value per rank (like run_reference): rank + 1
    # - Output we save: observed value in our local symmetric array after kernel
    input_val = float(rank + 1)
    output_val = int(array[0])
    y = torch.tensor([output_val], dtype=torch.int32)  # save result per-rank

    # Save to logs/problem_1/solution as rank files
    project_root = Path(__file__).resolve().parent.parent
    logs_dir = project_root / "logs" / "problem_1" / "solution"
    os.makedirs(logs_dir, exist_ok=True)
    out_path = logs_dir / f"cu_rank{rank}.pt"
    torch.save(y, str(out_path))

    # Cleanup
    nvshmem.core.free_array(array)
    nvshmem.core.finalize()


if __name__ == "__main__":
    solution()