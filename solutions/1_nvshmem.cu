// TODO: WHY IS THIS SOLUTION KERNEL NOT OUTPUTTING THE CORRECT THING?

#include "nvshmem.h"
#include "nvshmemx.h"

__global__ void solution(int *target) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    int sum = *target;
    for (int i = 1; i < npes; ++i) {
        int send_peer = (mype + 1) % npes;
        int recv_peer = (mype - 1 + npes) % npes;

        nvshmem_int_p(target, sum, send_peer);
        nvshmem_quiet();
        nvshmem_barrier_all();

        int recv = *target;
        sum += recv;
        nvshmem_barrier_all();
    }
    *target = sum;
}
