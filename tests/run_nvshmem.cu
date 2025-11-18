#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include "nvshmem.h"
#include "nvshmemx.h"

#include <iostream>

// User must supply the correct kernel header path for the problem
#include "/home/willychan/other_projs/ParallelKernelBench/solutions/1_nvshmem.cu" // Change as needed

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

// Utility to create a directory if it doesn't exist
void ensure_dir(const std::string& path) {
    struct stat st = {0};
    if (stat(path.c_str(), &st) == -1) {
        mkdir(path.c_str(), 0755);
    }
}

// Save tensor to file
void save_tensor(const std::string& filename, int value) {
    std::ofstream ofs(filename);
    ofs << value << std::endl;
    ofs.close();
}

// Save log to file
void save_log(const std::string& filename, const std::string& log) {
    std::ofstream ofs(filename);
    ofs << log << std::endl;
    ofs.close();
}

int main(int argc, char *argv[]) {
    // Default problem id and output directory
    std::string problem_id = "1";
    std::string output_dir = "/home/willychan/other_projs/ParallelKernelBench/logs/problem_1";

    // Parse command line arguments
    if (argc > 1) problem_id = argv[1];
    if (argc > 2) output_dir = argv[2];

    // Ensure output directories exist
    ensure_dir(output_dir);
    std::string solution_dir = output_dir + "/solution";
    ensure_dir(solution_dir);

    nvshmem_init();

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);


    CUDA_CHECK(cudaSetDevice(mype_node));

    // Rank-dependent input: for example, each rank gets its own value
    int rank_input = mype; // Example: customize as needed
    std::cout << "I am " << mype << " of " << npes << " on node " << mype_node << " and I hold value " << rank_input << std::endl;

    int *u = (int *)nvshmem_calloc(1, sizeof(int));
    int *h = (int *)calloc(1, sizeof(int));
    CUDA_CHECK(cudaMemcpy(u, &rank_input, sizeof(int), cudaMemcpyHostToDevice));

    // Prepare kernel args
    void *args[] = {&u};
    dim3 dimBlock(1);
    dim3 dimGrid(1);

    std::cout <<  "[Rank " << mype << "]"<< " Running the kernel!" << std::endl;


    // Run the imported kernel called "solution"
    NVSHMEM_CHECK(
        nvshmemx_collective_launch((const void *)solution, dimGrid, dimBlock, args, 0, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(h, u, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "[Rank " << mype << "]"<< " Finished running the kernel!" << std::endl;

    // Save log and tensor
    std::ostringstream log_stream;
    // log_stream << "Rank " << mype << " input: " << rank_input << ", output: " << h[0];
    std::string log_filename = output_dir + "/rank_" + std::to_string(mype) + ".log";
    std::string tensor_filename = solution_dir + "/rank_" + std::to_string(mype) + ".txt";
    // save_log(log_filename, log_stream.str());
    save_tensor(tensor_filename, h[0]);

    std::cout << "Finished saving the tensor for rank " << mype << "!" << std::endl;

    nvshmem_free(u);
    free(h);
    nvshmem_finalize();

    return 0;
}