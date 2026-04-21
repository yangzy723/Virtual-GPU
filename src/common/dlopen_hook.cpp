// dlopen_hook.cpp
//
// Intercept dlopen/dlsym to ensure that PyTorch (and other CUDA apps)
// always load our shim libraries instead of the real NVIDIA CUDA libraries.
//
// This solves the problem where PyTorch uses dlopen("libcuda.so") internally,
// which would bypass LD_PRELOAD and load the system's real CUDA library.

#include <dlfcn.h>
#include <cstring>
#include <cstdio>
#include <atomic>

namespace vgpu {
namespace {

// Flag to prevent recursive dlopen during our own initialization
std::atomic<bool> g_in_dlopen_hook{false};

// Original dlopen and dlsym function pointers
using DlopenFn = void* (*)(const char*, int);
using DlsymFn = void* (*)(void*, const char*);

DlopenFn g_original_dlopen = nullptr;
DlsymFn g_original_dlsym = nullptr;

// Initialize the original function pointers on first use
void initOriginalFunctions() {
    if (g_original_dlopen == nullptr) {
        // Get the original dlopen/dlsym from libc
        // We use RTLD_NEXT to bypass our own hook
        g_original_dlopen = reinterpret_cast<DlopenFn>(
            dlsym(RTLD_NEXT, "dlopen"));
        g_original_dlsym = reinterpret_cast<DlsymFn>(
            dlsym(RTLD_NEXT, "dlsym"));
    }
}

// Helper to check if a library name is a CUDA library we want to intercept
bool isCudaLibrary(const char* filename) {
    if (!filename) return false;
    
    // Check for various CUDA library naming patterns
    const char* cuda_libs[] = {
        "libcuda.so",      // Driver API
        "libcudart.so",    // Runtime API
        "libcublas.so",    // BLAS
        "libcublasLt.so",  // BLAS Lite
        "libcurand.so",    // Random
        "libcudnn.so",     // cuDNN
        "libcusparse.so",  // Sparse
        "libcusolver.so",  // Solver
        "libcufft.so",     // FFT
        "libnvrtc.so",     // NVRTC
        nullptr
    };
    
    for (int i = 0; cuda_libs[i]; ++i) {
        const char* lib = cuda_libs[i];
        // Check if filename contains the library name
        if (strstr(filename, lib) != nullptr) {
            return true;
        }
    }
    return false;
}

// Helper to resolve our shim library path
const char* getShimLibraryPath(const char* original_filename) {
    // Map real library names to our shim names
    if (strstr(original_filename, "libcuda.so")) {
        // Return our libcuda.so shim (in the current directory or LD_LIBRARY_PATH)
        return "libcuda.so";
    }
    if (strstr(original_filename, "libcudart.so")) {
        return "libcudart.so";
    }
    // For other libraries, just return as-is for now
    // (could be extended to provide stubs for cuBLAS, cuDNN, etc.)
    return original_filename;
}

}  // namespace

// Hooked dlopen function
extern "C" void* dlopen(const char* filename, int flags) {
    initOriginalFunctions();
    
    // Prevent recursive calls
    if (g_in_dlopen_hook.exchange(true)) {
        return g_original_dlopen(filename, flags);
    }
    
    void* result = nullptr;
    
    try {
        if (isCudaLibrary(filename)) {
            const char* shim_name = getShimLibraryPath(filename);
            
            // Log the interception for debugging
            std::fprintf(stderr, "[vGPU] dlopen interception: %s -> %s\n",
                        filename, shim_name);
            
            // Load our shim instead of the real library
            // Use RTLD_GLOBAL to make symbols globally visible
            // This helps avoid symbol conflicts in complex scenarios
            result = g_original_dlopen(shim_name, flags | RTLD_GLOBAL);
            
            if (result == nullptr) {
                std::fprintf(stderr, "[vGPU] Failed to load shim %s: %s\n",
                            shim_name, dlerror());
                // Fall back to loading the original if our shim fails
                result = g_original_dlopen(filename, flags);
            }
        } else {
            // Not a CUDA library, load normally
            result = g_original_dlopen(filename, flags);
        }
    } catch (...) {
        // In case of any exception, restore flag and rethrow
        g_in_dlopen_hook = false;
        throw;
    }
    
    g_in_dlopen_hook = false;
    return result;
}

// Hooked dlsym function (usually not needed but provided for completeness)
extern "C" void* dlsym(void* handle, const char* symbol) {
    initOriginalFunctions();
    
    // Prevent recursive calls
    if (g_in_dlopen_hook.exchange(true)) {
        return g_original_dlsym(handle, symbol);
    }
    
    void* result = nullptr;
    
    try {
        // Just forward to the original dlsym
        // The symbol should already be resolved correctly via dlopen hook
        result = g_original_dlsym(handle, symbol);
    } catch (...) {
        g_in_dlopen_hook = false;
        throw;
    }
    
    g_in_dlopen_hook = false;
    return result;
}

}  // namespace vgpu
