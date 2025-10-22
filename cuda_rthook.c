#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <stdint.h>
#include <pthread.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "resource-mg.h"

static unsigned long long g_kernel_count = 0;
static pthread_mutex_t g_kernel_count_lock = PTHREAD_MUTEX_INITIALIZER;

/* ---- 定义 HOOK 宏 ---- */
#define HOOK_DECLARE(ret_type, name, args_decl, args_call) \
    typedef ret_type (*name##_t) args_decl; \
    ret_type name args_decl { \
        static name##_t real = NULL; \
        if (!real) real = (name##_t)get_sym(#name); \
        if (!real) return (ret_type)(cudaErrorUnknown); \
        return real args_call; \
    }

/* ---- Hook 常用函数 ---- */
HOOK_DECLARE(cudaError_t, cudaMalloc,
             (void **devPtr, size_t size),
             (devPtr, size))

HOOK_DECLARE(cudaError_t, cudaFree,
             (void *devPtr),
             (devPtr))

HOOK_DECLARE(cudaError_t, cudaMemcpy,
             (void *dst, const void *src, size_t count, enum cudaMemcpyKind kind),
             (dst, src, count, kind))

HOOK_DECLARE(cudaError_t, cudaMemcpyAsync,
             (void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream),
             (dst, src, count, kind, stream))

// 自定义 Hook: cudaLaunchKernel
typedef cudaError_t (*cudaLaunchKernel_t)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
typedef CUresult (*cuLaunchKernel_t)(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem, cudaStream_t stream) {
    static cudaLaunchKernel_t real_cuda = NULL;
    if (!real_cuda) real_cuda = (cudaLaunchKernel_t)get_sym("cudaLaunchKernel");
    if (!real_cuda) return cudaErrorUnknown;

    static cuLaunchKernel_t real_cu = NULL;
    if (!real_cu) real_cu = (cuLaunchKernel_t)get_sym("cuLaunchKernel");
    if (!real_cu) return cudaErrorUnknown;

    pthread_mutex_lock(&g_kernel_count_lock);
    g_kernel_count++;
    pthread_mutex_unlock(&g_kernel_count_lock);

    fprintf(stderr,
            "[HOOK] cudaLaunchKernel: func=%p gridDim=(%u,%u,%u) blockDim=(%u,%u,%u) "
            "sharedMem=%zu stream=%p\n",
            (CUfunction)resource_mg_get(&rm_functions, (void *)func),
            gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z,
            sharedMem, (void*)stream);

    return real_cu(
        (CUfunction)resource_mg_get(&rm_functions, (void *)func),
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        sharedMem, (CUstream)stream, args, NULL);
}

/* ---- 构造/析构 ---- */
__attribute__((constructor)) static void hook_init(void) {
    g_kernel_count = 0;
}

__attribute__((destructor)) static void hook_fini(void) {
    fprintf(stderr, "[HOOK] Total kernels launched: %llu\n", g_kernel_count);
}
