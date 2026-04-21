#pragma once

#include <mutex>

#include "vgpu/cuda_abi.h"

namespace vgpu {

class CudaRuntimeLoader {
public:
    struct Api {
        cudaError_t (*cudaMalloc)(void**, std::size_t) = nullptr;
        cudaError_t (*cudaFree)(void*) = nullptr;
        cudaError_t (*cudaMemcpy)(void*, const void*, std::size_t, cudaMemcpyKind) = nullptr;
        cudaError_t (*cudaMemcpyAsync)(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t) = nullptr;
        cudaError_t (*cudaSetDevice)(int) = nullptr;
        cudaError_t (*cudaSetDeviceFlags)(unsigned int) = nullptr;
        cudaError_t (*cudaGetDeviceFlags)(unsigned int*) = nullptr;
        cudaError_t (*cudaGetDevice)(int*) = nullptr;
        cudaError_t (*cudaGetDeviceCount)(int*) = nullptr;
        cudaError_t (*cudaDriverGetVersion)(int*) = nullptr;
        cudaError_t (*cudaRuntimeGetVersion)(int*) = nullptr;
        cudaError_t (*cudaDeviceGetAttribute)(int*, int, int) = nullptr;
        cudaError_t (*cudaMemGetInfo)(std::size_t*, std::size_t*) = nullptr;
        cudaError_t (*cudaDeviceSynchronize)() = nullptr;
        cudaError_t (*cudaDeviceReset)() = nullptr;
        cudaError_t (*cudaStreamCreate)(cudaStream_t*) = nullptr;
        cudaError_t (*cudaStreamDestroy)(cudaStream_t) = nullptr;
        cudaError_t (*cudaStreamSynchronize)(cudaStream_t) = nullptr;
        cudaError_t (*cudaStreamQuery)(cudaStream_t) = nullptr;
        cudaError_t (*cudaStreamWaitEvent)(cudaStream_t, cudaEvent_t, unsigned int) = nullptr;
        cudaError_t (*cudaEventCreateWithFlags)(cudaEvent_t*, unsigned int) = nullptr;
        cudaError_t (*cudaEventDestroy)(cudaEvent_t) = nullptr;
        cudaError_t (*cudaEventRecord)(cudaEvent_t, cudaStream_t) = nullptr;
        cudaError_t (*cudaEventSynchronize)(cudaEvent_t) = nullptr;
        cudaError_t (*cudaEventQuery)(cudaEvent_t) = nullptr;
        cudaError_t (*cudaMemset)(void*, int, std::size_t) = nullptr;
        cudaError_t (*cudaMemsetAsync)(void*, int, std::size_t, cudaStream_t) = nullptr;
        cudaError_t (*cudaLaunchKernel)(const void*, dim3, dim3, void**, std::size_t, cudaStream_t) = nullptr;
    };

    CudaRuntimeLoader() = default;

    bool ensureLoaded();
    const Api& api() const { return api_; }

private:
    bool resolveSymbol(const char* name, void** fn_ptr_out);

    std::once_flag once_;
    void* handle_ = nullptr;
    bool ok_ = false;
    Api api_;
};

}  // namespace vgpu
