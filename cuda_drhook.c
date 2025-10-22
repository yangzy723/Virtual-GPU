#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <stdint.h>
#include <pthread.h>
#include <cuda.h>

static unsigned long long g_kernel_count = 0;
static pthread_mutex_t g_kernel_count_lock = PTHREAD_MUTEX_INITIALIZER;

/* ---- 定义 HOOK 宏 ---- */
#define HOOK_DECLARE(ret_type, name, args_decl, args_call, logfmt, ...) \
    typedef ret_type (*name##_t) args_decl; \
    ret_type name args_decl { \
        static name##_t real = NULL; \
        if (!real) real = (name##_t)get_sym(#name); \
        fprintf(stderr, "[HOOK] " #name " " logfmt "\n", ##__VA_ARGS__); \
        if (!real) return (ret_type)(CUDA_ERROR_UNKNOWN); \
        return real args_call; \
    }

/* ---- Hook 常用函数 ---- */
HOOK_DECLARE(CUresult, cuMemAlloc,
             (CUdeviceptr *dptr, size_t bytesize),
             (dptr, bytesize),
             "bytes=%zu", bytesize)

HOOK_DECLARE(CUresult, cuMemFree,
             (CUdeviceptr dptr),
             (dptr),
             "dptr=0x%llx", (unsigned long long)dptr)

HOOK_DECLARE(CUresult, cuMemcpyHtoD,
             (CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount),
             (dstDevice, srcHost, ByteCount),
             "dst=0x%llx bytes=%zu", (unsigned long long)dstDevice, ByteCount)

HOOK_DECLARE(CUresult, cuMemcpyDtoH,
             (void *dstHost, CUdeviceptr srcDevice, size_t ByteCount),
             (dstHost, srcDevice, ByteCount),
             "src=0x%llx bytes=%zu", (unsigned long long)srcDevice, ByteCount)

// 自定义 Hook: cuLaunchKernel
typedef CUresult (*cuLaunchKernel_t)(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
CUresult cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {

    static cuLaunchKernel_t real = NULL;
    if (!real) real = (cuLaunchKernel_t)get_sym("cuLaunchKernel");
    if (!real) return CUDA_ERROR_UNKNOWN;

    pthread_mutex_lock(&g_kernel_count_lock);
    g_kernel_count++;
    pthread_mutex_unlock(&g_kernel_count_lock);

    fprintf(stderr,
            "[HOOK] cuLaunchKernel f=%p grid=(%u,%u,%u) block=(%u,%u,%u) shared=%u stream=%p\n",
            (void*)f,
            gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes, (void*)hStream);

    return real(f, gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                sharedMemBytes, hStream, kernelParams, extra);
}

/* ---- 构造/析构 ---- */
__attribute__((constructor)) static void hook_init(void) {
    g_kernel_count = 0;
    fprintf(stderr, "[HOOK] Driver API hook init\n");
}

__attribute__((destructor)) static void hook_fini(void) {
    fprintf(stderr, "[HOOK] Driver API: total kernels launched = %llu\n", g_kernel_count);
}
