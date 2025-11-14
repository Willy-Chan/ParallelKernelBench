// main_nvshmem_app.cu

#include <stdio.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "/home/willychan/other_projs/ParallelKernelBench/solutions/1_nvshmem.cuh"

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

#define NVSHMEM_CHECK(stmt)                                                                \
    do {                                                                                   \
        int result = (stmt);                                                               \
        if (NVSHMEMX_SUCCESS != result) {                                                  \
            fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n", __FILE__, __LINE__, \
                    result);                                                               \
            exit(-1);                                                                      \
        }                                                                                  \
    } while (0)

int main(int c, char *v[]) {
    int mype, npes, mype_node;

    nvshmem_init();

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaSetDevice(mype_node));
    int *u = (int *)nvshmem_calloc(1, sizeof(int));
    int *h = (int *)calloc(1, sizeof(int));

    void *args[] = {&u, &mype, &npes};
    dim3 dimBlock(1);
    dim3 dimGrid(1);

    NVSHMEM_CHECK(
        nvshmemx_collective_launch((const void *)reduce_ring, dimGrid, dimBlock, args, 0, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(h, u,  sizeof(int), cudaMemcpyDeviceToHost);
    printf("results on device [%d] is %d \n",mype, h[0]);

    nvshmem_free(u);
    free(h);
    nvshmem_finalize();

    return 0;
}