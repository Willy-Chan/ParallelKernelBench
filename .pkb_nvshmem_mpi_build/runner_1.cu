#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

extern "C" void solution(float* data, size_t numel, cudaStream_t stream);

__global__ void fill_value(float* p, size_t n, float v) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

int main(int argc, char** argv) {
    // Parse env for size and output dir
    const char* m_env = std::getenv("PKB_M");
    const char* n_env = std::getenv("PKB_N");
    size_t M = m_env ? static_cast<size_t>(std::strtoull(m_env, nullptr, 10)) : 1024ull;
    size_t N = n_env ? static_cast<size_t>(std::strtoull(n_env, nullptr, 10)) : 1024ull;
    size_t NUMEL = M * N;
    const char* out_dir = std::getenv("PKB_OUT_DIR");
    if (!out_dir) out_dir = ".";

    // Stage 0: MPI init (required for NVSHMEM_BOOTSTRAP=MPI)
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    // Stage 1: NVSHMEM init (two-stage)
    nvshmem_init();
    int me = nvshmem_my_pe();
    int np = nvshmem_n_pes();

    // Select CUDA device based on PE
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    if (ndev > 0) cudaSetDevice(me % ndev);

    cudaStream_t stream{};
    cudaStreamCreate(&stream);

    // Allocate device buffer and fill with rank+1
    float* d_data = nullptr;
    cudaMalloc(&d_data, NUMEL * sizeof(float));
    dim3 block(256);
    dim3 grid((NUMEL + block.x - 1) / block.x);
    fill_value<<<grid, block, 0, stream>>>(d_data, NUMEL, static_cast<float>(me + 1));
    cudaStreamSynchronize(stream);

    // Call solution kernel wrapper from solutions/1.cu
    solution(d_data, NUMEL, stream);
    cudaStreamSynchronize(stream);
    nvshmem_barrier_all();

    // Copy back to host and write raw f32
    std::vector<float> h(NUMEL);
    cudaMemcpy(h.data(), d_data, NUMEL * sizeof(float), cudaMemcpyDeviceToHost);
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/tmp_rank%d.bin", out_dir, me);
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "[%d] Failed to open %s for write\\n", me, path);
    } else {
        std::fwrite(h.data(), sizeof(float), NUMEL, f);
        std::fclose(f);
        std::printf("[%d/%d] Wrote %s\\n", me, np, path);
        std::fflush(stdout);
    }

    cudaFree(d_data);
    cudaStreamDestroy(stream);
    nvshmem_barrier_all();
    nvshmem_finalize();
    MPI_Finalize();
    return 0;
}
