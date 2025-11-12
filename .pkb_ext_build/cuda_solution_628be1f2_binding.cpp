
    #include <torch/extension.h>
    #include <c10/cuda/CUDAStream.h>
    #include <nvshmem.h>
    #include <mutex>

    // Declare the CUDA solution without CUDA headers here
    extern "C" void solution(float* data, size_t numel, void* stream);

    namespace {
    std::once_flag nvshmem_once;
    void ensure_nvshmem_init() {
        std::call_once(nvshmem_once, [](){
            nvshmem_init();
            nvshmem_barrier_all();
        });
    }
    } // namespace

    torch::Tensor solution_binding(torch::Tensor input) {
        TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
        TORCH_CHECK(input.dtype() == torch::kFloat32, "only float32 is supported by CUDA solution");
        auto out = input.contiguous().clone();
        ensure_nvshmem_init();
        auto stream = c10::cuda::getCurrentCUDAStream();
        solution(out.data_ptr<float>(), static_cast<size_t>(out.numel()), reinterpret_cast<void*>(stream.stream()));
        return out;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("solution", &solution_binding, "CUDA solution(binding)");
    }
    