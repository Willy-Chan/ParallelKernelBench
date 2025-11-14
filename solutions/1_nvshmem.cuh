// reduce_ring_kernel.cuh

#pragma once
#include "nvshmem.h"
#include "nvshmemx.h"

__global__ void reduce_ring(int *target, int mype, int npes) {
    int peer = (mype + 1) % npes;
    int lvalue = mype;

    for (int i = 1; i < npes; i++) {
        nvshmem_int_p(target, lvalue, peer);
        nvshmem_barrier_all();
        lvalue = *target + mype;
        nvshmem_barrier_all();
    }
}
