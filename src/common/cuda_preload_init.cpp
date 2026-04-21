// cuda_preload_init.cpp
//
// Initialization library that ensures our CUDA shims are loaded FIRST
// before any other CUDA-related code runs.
//
// This library should be preloaded FIRST in LD_PRELOAD to guarantee
// that our shims take priority over any other CUDA libraries:
//
//   LD_PRELOAD=/path/to/libcuda_preload_init.so:/path/to/libcuda.so:/path/to/libcudart.so
//
// or via a single convenience library:
//
//   LD_PRELOAD=/path/to/libvgpu_preload.so

#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>

namespace vgpu {
namespace {

// Attribute to run this code at library load time (highest priority)
__attribute__((constructor(101)))
void initVgpuPreload() {
    // Only print if debug env var is set
    const char* debug = std::getenv("VGPU_DEBUG");
    
    if (debug) {
        std::fprintf(stderr, "[vGPU] Preload initializer running (priority 101)\n");
    }
    
    // The presence of this library in LD_PRELOAD ensures that:
    // 1. Our dlopen hook (if compiled in) is available
    // 2. libcuda.so and libcudart.so symbols are registered
    // 3. PyTorch's dlopen calls will be intercepted by dlopen_hook
    
    // We don't need to do much here - the key is that this initializer runs
    // BEFORE any CUDA-related code in the main application
    
    if (debug) {
        std::fprintf(stderr, "[vGPU] Preload ready (app_pid=%d)\n", (int)getpid());
    }
}

}  // namespace
}  // namespace vgpu
