#include <cstdio>
#include <vector>

#include "vgpu/cuda_abi.h"

extern "C" cudaError_t cudaGetDeviceCount(int* count);
extern "C" cudaError_t cudaRuntimeGetVersion(int* runtimeVersion);
extern "C" cudaError_t cudaDriverGetVersion(int* driverVersion);
extern "C" cudaError_t cudaSetDeviceFlags(unsigned int flags);
extern "C" cudaError_t cudaGetDeviceFlags(unsigned int* flags);
extern "C" cudaError_t cudaDeviceGetAttribute(int* value, int attr, int device);
extern "C" cudaError_t cudaMalloc(void** devPtr, std::size_t size);
extern "C" cudaError_t cudaMemset(void* dst, int value, std::size_t count);
extern "C" cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind);
extern "C" cudaError_t cudaFree(void* devPtr);
extern "C" const char* cudaGetErrorString(cudaError_t error);

int main() {
    int count = 0;
    cudaError_t st = cudaGetDeviceCount(&count);
    if (st != cudaSuccess || count <= 0) {
        std::fprintf(stderr, "cudaGetDeviceCount failed: %d (%s), count=%d\n", st, cudaGetErrorString(st), count);
        return 1;
    }

    int rt_ver = 0;
    int drv_ver = 0;
    st = cudaRuntimeGetVersion(&rt_ver);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaRuntimeGetVersion failed: %d (%s)\n", st, cudaGetErrorString(st));
        return 7;
    }

    st = cudaDriverGetVersion(&drv_ver);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaDriverGetVersion failed: %d (%s)\n", st, cudaGetErrorString(st));
        return 8;
    }

    st = cudaSetDeviceFlags(0);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDeviceFlags failed: %d (%s)\n", st, cudaGetErrorString(st));
        return 9;
    }

    unsigned int flags = 0;
    st = cudaGetDeviceFlags(&flags);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceFlags failed: %d (%s)\n", st, cudaGetErrorString(st));
        return 10;
    }

    int mp_count = 0;
    // 16 is cudaDevAttrMultiProcessorCount in CUDA runtime enum.
    st = cudaDeviceGetAttribute(&mp_count, 16, 0);
    if (st != cudaSuccess || mp_count <= 0) {
        std::fprintf(stderr, "cudaDeviceGetAttribute failed: %d (%s), value=%d\n", st, cudaGetErrorString(st), mp_count);
        return 11;
    }

    constexpr int n = 1024;
    std::vector<unsigned char> h(n, 0);
    void* d = nullptr;

    st = cudaMalloc(&d, n);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed: %d (%s)\n", st, cudaGetErrorString(st));
        return 2;
    }

    st = cudaMemset(d, 0x2a, n);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaMemset failed: %d (%s)\n", st, cudaGetErrorString(st));
        cudaFree(d);
        return 3;
    }

    st = cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy D2H failed: %d (%s)\n", st, cudaGetErrorString(st));
        cudaFree(d);
        return 4;
    }

    for (int i = 0; i < n; ++i) {
        if (h[i] != 0x2a) {
            std::fprintf(stderr, "verify failed at %d: %u\n", i, static_cast<unsigned int>(h[i]));
            cudaFree(d);
            return 5;
        }
    }

    st = cudaFree(d);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaFree failed: %d (%s)\n", st, cudaGetErrorString(st));
        return 6;
    }

    std::printf("runtime_shim_smoke_test passed (rt=%d drv=%d mp=%d flags=%u)\n", rt_ver, drv_ver, mp_count, flags);
    return 0;
}
