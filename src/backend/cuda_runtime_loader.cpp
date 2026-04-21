#include "vgpu/cuda_runtime_loader.h"

#include <dlfcn.h>

namespace vgpu {

bool CudaRuntimeLoader::resolveSymbol(const char* name, void** fn_ptr_out) {
    *fn_ptr_out = dlsym(handle_, name);
    return *fn_ptr_out != nullptr;
}

bool CudaRuntimeLoader::ensureLoaded() {
    std::call_once(once_, [this]() {
        const char* candidates[] = {
            "libcudart.so",
            "libcudart.so.12",
            "libcudart.so.11.0"
        };

        for (const char* lib : candidates) {
            handle_ = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
            if (handle_ != nullptr) {
                break;
            }
        }
        if (handle_ == nullptr) {
            ok_ = false;
            return;
        }

        bool required_ok = resolveSymbol("cudaMalloc", reinterpret_cast<void**>(&api_.cudaMalloc)) &&
                           resolveSymbol("cudaFree", reinterpret_cast<void**>(&api_.cudaFree)) &&
                           resolveSymbol("cudaMemcpy", reinterpret_cast<void**>(&api_.cudaMemcpy)) &&
                           resolveSymbol("cudaSetDevice", reinterpret_cast<void**>(&api_.cudaSetDevice)) &&
                           resolveSymbol("cudaGetDevice", reinterpret_cast<void**>(&api_.cudaGetDevice)) &&
                           resolveSymbol("cudaDeviceSynchronize", reinterpret_cast<void**>(&api_.cudaDeviceSynchronize)) &&
                           resolveSymbol("cudaStreamCreate", reinterpret_cast<void**>(&api_.cudaStreamCreate)) &&
                           resolveSymbol("cudaStreamDestroy", reinterpret_cast<void**>(&api_.cudaStreamDestroy));

        if (!required_ok) {
            ok_ = false;
            return;
        }

        // Optional symbols: missing entries remain nullptr and are treated as not supported at runtime.
        resolveSymbol("cudaMemcpyAsync", reinterpret_cast<void**>(&api_.cudaMemcpyAsync));
        resolveSymbol("cudaSetDeviceFlags", reinterpret_cast<void**>(&api_.cudaSetDeviceFlags));
        resolveSymbol("cudaGetDeviceFlags", reinterpret_cast<void**>(&api_.cudaGetDeviceFlags));
        resolveSymbol("cudaGetDeviceCount", reinterpret_cast<void**>(&api_.cudaGetDeviceCount));
        resolveSymbol("cudaDriverGetVersion", reinterpret_cast<void**>(&api_.cudaDriverGetVersion));
        resolveSymbol("cudaRuntimeGetVersion", reinterpret_cast<void**>(&api_.cudaRuntimeGetVersion));
        resolveSymbol("cudaDeviceGetAttribute", reinterpret_cast<void**>(&api_.cudaDeviceGetAttribute));
        resolveSymbol("cudaMemGetInfo", reinterpret_cast<void**>(&api_.cudaMemGetInfo));
        resolveSymbol("cudaDeviceReset", reinterpret_cast<void**>(&api_.cudaDeviceReset));
        resolveSymbol("cudaStreamSynchronize", reinterpret_cast<void**>(&api_.cudaStreamSynchronize));
        resolveSymbol("cudaStreamQuery", reinterpret_cast<void**>(&api_.cudaStreamQuery));
        resolveSymbol("cudaStreamWaitEvent", reinterpret_cast<void**>(&api_.cudaStreamWaitEvent));
        resolveSymbol("cudaEventCreateWithFlags", reinterpret_cast<void**>(&api_.cudaEventCreateWithFlags));
        resolveSymbol("cudaEventDestroy", reinterpret_cast<void**>(&api_.cudaEventDestroy));
        resolveSymbol("cudaEventRecord", reinterpret_cast<void**>(&api_.cudaEventRecord));
        resolveSymbol("cudaEventSynchronize", reinterpret_cast<void**>(&api_.cudaEventSynchronize));
        resolveSymbol("cudaEventQuery", reinterpret_cast<void**>(&api_.cudaEventQuery));
        resolveSymbol("cudaMemset", reinterpret_cast<void**>(&api_.cudaMemset));
        resolveSymbol("cudaMemsetAsync", reinterpret_cast<void**>(&api_.cudaMemsetAsync));
        resolveSymbol("cudaLaunchKernel", reinterpret_cast<void**>(&api_.cudaLaunchKernel));

        ok_ = true;
    });

    return ok_;
}

}  // namespace vgpu
