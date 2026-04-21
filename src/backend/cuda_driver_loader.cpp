#include "vgpu/backend/cuda_driver_loader.h"

#include <cstdio>
#include <dlfcn.h>

namespace vgpu {

namespace {
// Candidate library names for libcuda
static const char* kDriverCandidates[] = {
    "libcuda.so.1",
    "libcuda.so",
    nullptr,
};

template <typename T>
bool resolveOptional(void* handle, const char* name, T*& out) {
    out = reinterpret_cast<T*>(dlsym(handle, name));
    return out != nullptr;
}

template <typename T>
bool resolveRequired(void* handle, const char* name, T*& out) {
    out = reinterpret_cast<T*>(dlsym(handle, name));
    if (!out) {
        std::fprintf(stderr, "[vgpu-driver-loader] missing required symbol: %s\n", name);
        return false;
    }
    return true;
}
}  // namespace

bool CudaDriverLoader::ensureLoaded() {
    if (loaded_) return true;

    for (int i = 0; kDriverCandidates[i]; ++i) {
        handle_ = dlopen(kDriverCandidates[i], RTLD_LAZY | RTLD_GLOBAL);
        if (handle_) break;
    }
    if (!handle_) {
        std::fprintf(stderr, "[vgpu-driver-loader] cannot open libcuda.so: %s\n", dlerror());
        return false;
    }

    bool ok = true;
    // Required
    ok &= resolveRequired(handle_, "cuInit",           api_.cuInit);
    ok &= resolveRequired(handle_, "cuModuleLoadData", api_.cuModuleLoadData);
    ok &= resolveRequired(handle_, "cuModuleGetFunction", api_.cuModuleGetFunction);
    ok &= resolveRequired(handle_, "cuLaunchKernel",   api_.cuLaunchKernel);

    // Optional but important
    resolveOptional(handle_, "cuDeviceGetCount",         api_.cuDeviceGetCount);
    resolveOptional(handle_, "cuDeviceGet",              api_.cuDeviceGet);
    resolveOptional(handle_, "cuDevicePrimaryCtxRetain", api_.cuDevicePrimaryCtxRetain);
    resolveOptional(handle_, "cuCtxSetCurrent",          api_.cuCtxSetCurrent);
    resolveOptional(handle_, "cuCtxGetCurrent",          api_.cuCtxGetCurrent);
    resolveOptional(handle_, "cuModuleUnload",           api_.cuModuleUnload);
    resolveOptional(handle_, "cuMemAlloc_v2",            api_.cuMemAlloc_v2);
    resolveOptional(handle_, "cuMemFree_v2",             api_.cuMemFree_v2);
    resolveOptional(handle_, "cuMemcpyHtoD_v2",          api_.cuMemcpyHtoD_v2);
    resolveOptional(handle_, "cuMemcpyDtoH_v2",          api_.cuMemcpyDtoH_v2);
    resolveOptional(handle_, "cuMemcpyDtoD_v2",          api_.cuMemcpyDtoD_v2);
    resolveOptional(handle_, "cuStreamSynchronize",      api_.cuStreamSynchronize);
    resolveOptional(handle_, "cuCtxSynchronize",         api_.cuCtxSynchronize);
    resolveOptional(handle_, "cuFuncGetAttribute",       api_.cuFuncGetAttribute);

    if (!ok) {
        dlclose(handle_);
        handle_ = nullptr;
        return false;
    }

    loaded_ = true;
    return true;
}

}  // namespace vgpu
