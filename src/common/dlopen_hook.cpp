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

namespace vgpu {
namespace {

// Flag to prevent recursive dlopen during our own initialization
thread_local bool g_in_dlopen_hook = false;

// Original dlopen and dlsym function pointers
using DlopenFn = void* (*)(const char*, int);
using DlsymFn = void* (*)(void*, const char*);

DlopenFn g_original_dlopen = nullptr;
DlsymFn g_original_dlsym = nullptr;

const char* kGlibcSymbolVersion = "GLIBC_2.2.5";

struct HookGuard {
    HookGuard() : reentrant(g_in_dlopen_hook) {
        if (!reentrant) {
            g_in_dlopen_hook = true;
        }
    }
    ~HookGuard() {
        if (!reentrant) {
            g_in_dlopen_hook = false;
        }
    }
    bool reentrant;
};

void* rawDlvsym(void* handle, const char* symbol) {
    return dlvsym(handle, symbol, kGlibcSymbolVersion);
}

// Initialize the original function pointers on first use
void initOriginalFunctions() {
    if (g_original_dlopen == nullptr || g_original_dlsym == nullptr) {
        // Use dlvsym to avoid recursively calling our dlsym hook.
        g_original_dlopen = reinterpret_cast<DlopenFn>(rawDlvsym(RTLD_NEXT, "dlopen"));
        g_original_dlsym = reinterpret_cast<DlsymFn>(rawDlvsym(RTLD_NEXT, "dlsym"));

        // Fallback for environments where versioned lookup is unavailable.
        if (g_original_dlopen == nullptr) {
            g_original_dlopen = reinterpret_cast<DlopenFn>(rawDlvsym(RTLD_DEFAULT, "dlopen"));
        }
        if (g_original_dlsym == nullptr) {
            g_original_dlsym = reinterpret_cast<DlsymFn>(rawDlvsym(RTLD_DEFAULT, "dlsym"));
        }
    }
}

const char* findShimLibrary(const char* filename) {
    if (!filename) {
        return nullptr;
    }
    if (std::strstr(filename, "libcuda.so") != nullptr) {
        return "libcuda.so";
    }
    if (std::strstr(filename, "libcudart.so") != nullptr) {
        return "libcudart.so";
    }
    return nullptr;
}

}  // namespace

// Hooked dlopen function
extern "C" void* dlopen(const char* filename, int flags) {
    initOriginalFunctions();

    if (g_original_dlopen == nullptr) {
        return nullptr;
    }
    
    // Prevent recursive calls
    HookGuard guard;
    if (guard.reentrant) {
        return g_original_dlopen(filename, flags);
    }
    
    void* result = nullptr;
    
    const char* shim_name = findShimLibrary(filename);
    if (shim_name != nullptr) {

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

    return result;
}

// Hooked dlsym function (usually not needed but provided for completeness)
extern "C" void* dlsym(void* handle, const char* symbol) {
    initOriginalFunctions();

    if (g_original_dlsym == nullptr) {
        return nullptr;
    }
    
    // Prevent recursive calls
    HookGuard guard;
    if (guard.reentrant) {
        return g_original_dlsym(handle, symbol);
    }

    // Just forward to the original dlsym.
    return g_original_dlsym(handle, symbol);
}

}  // namespace vgpu
