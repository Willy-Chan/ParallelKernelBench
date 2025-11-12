// // NVCC COMPILING INSTRUCTIONS::::::
// // export CUDA_HOME=$HOME/cuda-12.9                                                                      - sets CUDA tools to version 12.9 which is what's needed by nvshmem 3.4.5
// // export PATH=$CUDA_HOME/bin:$PATH
// // export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
// // export NVSHMEM_HOME=/home/willychan/other_projs/libnvshmem-linux-x86_64-3.4.5_cuda12-archive
// // nvcc -std=c++20 -arch=sm_80 -rdc=true \
// //   -I"$CUDA_HOME/include" \
// //   -I"$CUDA_HOME/targets/x86_64-linux/include" \
// //   -I"$NVSHMEM_HOME/include" \
// //   -c solutions/1.cu \
// //   -o solutions/1.o
// //   -L"$NVSHMEM_HOME/lib" -L"$NVSHMEM_HOME/lib64" -L"$CUDA_HOME/lib64" \
// //   -lnvshmem_host -lnvshmem_device -lcudadevrt


// // srun --partition=learn --nodes=1 --ntasks=1 --gres=gpu:8 --pty bash


// // nvcc -std=c++20 -arch=sm_80 -rdc=true \
// //   -I"$CUDA_HOME/include" \
// //   -I"$CUDA_HOME/targets/x86_64-linux/include" \
// //   -I"$NVSHMEM_HOME/include" \
// //   -c solutions/1.cu \
// //   -o solutions/1.o



// // NVSHMEM device and host includes
// #include <cuda_runtime.h>
// #include <nvshmem.h>
// #include <nvshmemx.h>
// #include <stdio.h>
// #include <vector>
// #include <cstdlib>

// // Device kernel: sum elementwise across all PEs using symmetric buffer 'sym'
// __global__ void nvshmem_allreduce_sum_kernel(float* sym, size_t numel) {
//     const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     const int me = nvshmem_my_pe();
//     const int np = nvshmem_n_pes();
//     if (idx == 0 && threadIdx.x == 0 && blockIdx.x == 0) {
//         printf("[PE %d] In kernel: world_size(np)=%d\n", me, np);
//     }
//     if (idx < numel && idx < 4 && threadIdx.x == 0 && blockIdx.x < 4) {
//         printf("[PE %d] Before allreduce, sym[%zu]=%f\n", me, idx, sym[idx]);
//     }
//     if (idx >= numel) return;
//     float sum = sym[idx]; // local element
//     // Accumulate remote elements from the same symmetric address on other PEs
//     for (int pe = 0; pe < np; ++pe) {
//         if (pe == me) continue;
//         sum += nvshmem_float_g(sym + idx, pe);
//     }
//     sym[idx] = sum;
//     if (idx < numel && idx < 4 && threadIdx.x == 0 && blockIdx.x < 4) {
//         printf("[PE %d] After allreduce, sym[%zu]=%f\n", me, idx, sym[idx]);
//     }
// }

// extern "C" void solution(float* data, size_t numel, cudaStream_t stream) {
//     // Initialize NVSHMEM on first use (once per process), using two-stage init.
//     static bool nvshmem_initialized = false;
//     if (!nvshmem_initialized) {
//         // Stage 1: initialize NVSHMEM (bootstrap provided via env, e.g., PMIX/PMI2)
//         nvshmem_init();
//         // Select device based on PE id (two-stage device selection)
//         int me = nvshmem_my_pe();
//         int ndev = 0;
//         cudaGetDeviceCount(&ndev);
//         if (ndev > 0) {
//             cudaSetDevice(me % ndev);
//         }
//         nvshmem_barrier_all();
//         nvshmem_initialized = true;
//         // Debug: report world size from host after init
//         int np = nvshmem_n_pes();
//         printf("[Host] NVSHMEM init complete: me=%d np=%d (ndev=%d)\\n", me, np, ndev);
//     }
//     // Allocate/reuse symmetric buffer to hold data across PEs
//     static float* sym = nullptr;
//     static size_t sym_capacity = 0;
//     if (numel > sym_capacity) {
//         if (sym) {
//             nvshmem_free(sym);
//             sym = nullptr;
//         }
//         sym = (float*)nvshmem_malloc(numel * sizeof(float));
//         sym_capacity = numel;
//     }
//     // Copy input into symmetric buffer on provided stream
//     cudaMemcpyAsync(sym, data, numel * sizeof(float), cudaMemcpyDeviceToDevice, stream);
//     // Ensure all copies complete before reduction
//     cudaStreamSynchronize(stream);
//     nvshmem_barrier_all();

//     // Launch device-side NVSHMEM reduction using symmetric buffer
//     const int threads = 256;
//     const int blocks = static_cast<int>((numel + threads - 1) / threads);
//     nvshmem_allreduce_sum_kernel<<<blocks, threads, 0, stream>>>(sym, numel);
//     cudaStreamSynchronize(stream);
//     nvshmem_barrier_all();

//     // Copy result back to user buffer
//     cudaMemcpyAsync(data, sym, numel * sizeof(float), cudaMemcpyDeviceToDevice, stream);
// }

// // Utility: fill buffer with a constant
// __global__ void fill_value(float* p, size_t n, float v) {
//     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n) p[i] = v;
// }

// // Standalone entrypoint so this TU can be run with nvshmrun
// int main(int argc, char** argv) {
//     // Read matrix shape and output dir from environment (optional)
//     const char* m_env = std::getenv("PKB_M");
//     const char* n_env = std::getenv("PKB_N");
//     size_t M = m_env ? static_cast<size_t>(std::strtoull(m_env, nullptr, 10)) : 1024ull;
//     size_t N = n_env ? static_cast<size_t>(std::strtoull(n_env, nullptr, 10)) : 1024ull;
//     size_t NUMEL = M * N;
//     const char* out_dir = std::getenv("PKB_OUT_DIR");
//     if (!out_dir) out_dir = ".";

//     // Two-stage NVSHMEM init (bootstrap controlled by env, e.g., PMIX/MPI)
//     nvshmem_init();
//     int me = nvshmem_my_pe();
//     int np = nvshmem_n_pes();
//     int ndev = 0;
//     cudaGetDeviceCount(&ndev);
//     if (ndev > 0) cudaSetDevice(me % ndev);
//     nvshmem_barrier_all();
//     printf("[Host] Started PE %d/%d (ndev=%d) M=%zu N=%zu\n", me, np, ndev, M, N);
//     fflush(stdout);

//     // Allocate device tensor and fill with rank+1
//     float* d_data = nullptr;
//     cudaMalloc(&d_data, NUMEL * sizeof(float));
//     cudaStream_t stream{};
//     cudaStreamCreate(&stream);
//     dim3 block(256);
//     dim3 grid((NUMEL + block.x - 1) / block.x);
//     fill_value<<<grid, block, 0, stream>>>(d_data, NUMEL, static_cast<float>(me + 1));
//     cudaStreamSynchronize(stream);

//     // Run the NVSHMEM solution
//     solution(d_data, NUMEL, stream);
//     cudaStreamSynchronize(stream);
//     nvshmem_barrier_all();

//     // Copy back and write a simple raw binary per rank (float32)
//     std::vector<float> h(NUMEL);
//     cudaMemcpy(h.data(), d_data, NUMEL * sizeof(float), cudaMemcpyDeviceToHost);
//     char path[4096];
//     std::snprintf(path, sizeof(path), "%s/cu_rank%d.bin", out_dir, me);
//     FILE* f = std::fopen(path, "wb");
//     if (!f) {
//         std::fprintf(stderr, "[%d] Failed to open %s for write\n", me, path);
//     } else {
//         std::fwrite(h.data(), sizeof(float), NUMEL, f);
//         std::fclose(f);
//         std::printf("[%d/%d] Wrote %s\n", me, np, path);
//         std::fflush(stdout);
//     }

//     // Cleanup
//     cudaFree(d_data);
//     cudaStreamDestroy(stream);
//     nvshmem_barrier_all();
//     nvshmem_finalize();
//     return 0;
// }

// // TO COMPILE:
// nvcc -std=c++20 -arch=sm_80 -rdc=true \
//   -I"$CUDA_HOME/include" -I"$CUDA_HOME/targets/x86_64-linux/include" \
//   -I"$NVSHMEM_HOME/include" -I/opt/amazon/openmpi5/include \
//   /home/willychan/other_projs/ParallelKernelBench/solutions/1.cu \
//   -o /home/willychan/other_projs/ParallelKernelBench/solutions/one_out \
//   -L/opt/amazon/openmpi5/lib -L/opt/amazon/openmpi5/lib64 -lmpi \
//   -L"$NVSHMEM_HOME/lib" -L"$NVSHMEM_HOME/lib64" -L"$CUDA_HOME/lib64" \
//   -lnvshmem_host -lnvshmem_device -lcudadevrt

// nvcc  \
//   solutions/1.o -o solutions/one_out \

// nvcc -std=c++20 -arch=sm_80 -rdc=true -I/opt/amazon/openmpi5/include -I${NVSHMEM_HOME}/include -L${NVSHMEM_HOME}/lib -lnvshmem -o solutions/my_nvshmem_app solutions/1_nvshmem.cu
// USE NVSHMEM WITHOUT MPI IS POSSIBLE, YOU JUST NEED TO USE NVSHMRUN!!!
//      # like so: $NVSHMEM_HOME/bin/nvshmrun solutions/my_nvshmem_app

#include <stdio.h>
#include "nvshmem.h"
#include "nvshmemx.h"



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

int main(int c, char *v[]) {
    int mype, npes, mype_node;




    nvshmem_init();


    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    // application picks the device each PE will use
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