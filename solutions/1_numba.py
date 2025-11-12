"""
HERE'S THE ISSUE: You can't do device-side nvshmem operations in pythonic DSLs yet,  you can only allocate symmetric memory using nvshmem and pass in data pointers to symmetric memory, like you're doing in this example. With that in mind can you modify run_numba and 1.py to account for this and implement an allreduce using symmetric memory?
"""

import nvshmem.core
from cuda.core.experimental import Device
from numba import cuda as nb_cuda


@nb_cuda.jit
def solution(peer_arr, value):
    """Write 'value' into peer_arr[0] on the destination PE."""
    if nb_cuda.threadIdx.x == 0 and nb_cuda.blockIdx.x == 0:
        peer_arr[0] = value

