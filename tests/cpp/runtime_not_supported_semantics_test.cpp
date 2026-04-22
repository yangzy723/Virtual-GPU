#include <cstdio>
#include <cstdint>

#include "vgpu/common/cuda_abi.h"

extern "C" cudaError_t cudaPeekAtLastError(void);
extern "C" cudaError_t cudaIpcGetMemHandle(void* handle, void* devPtr);
extern "C" cudaError_t cudaIpcOpenMemHandle(void** devPtr, void* handle, unsigned int flags);
extern "C" cudaError_t cudaIpcCloseMemHandle(void* devPtr);
extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t stream, int mode);
extern "C" cudaError_t cudaStreamIsCapturing(cudaStream_t stream, int* captureStatus);
extern "C" cudaError_t cudaDeviceGetDefaultMemPool(void** memPool, int device);
extern "C" cudaError_t cudaMemPoolTrimTo(void* memPool, std::size_t minBytesToKeep);

namespace {

bool expectNotSupported(const char* api, cudaError_t st) {
    if (st != cudaErrorNotSupported) {
        std::fprintf(stderr, "%s should return cudaErrorNotSupported(801), got %d\n", api, static_cast<int>(st));
        return false;
    }
    return true;
}

bool expectLastErrorUpdated(const char* api) {
    cudaError_t last = cudaPeekAtLastError();
    if (last != cudaErrorNotSupported) {
        std::fprintf(stderr, "%s should update last error to cudaErrorNotSupported(801), got %d\n", api, static_cast<int>(last));
        return false;
    }
    return true;
}

}  // namespace

int main() {
    alignas(16) unsigned char ipc_handle[64] = {0};
    void* dev_ptr = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x12345678));
    void* opened_ptr = nullptr;

    if (!expectNotSupported("cudaIpcGetMemHandle", cudaIpcGetMemHandle(ipc_handle, dev_ptr)) ||
        !expectLastErrorUpdated("cudaIpcGetMemHandle")) {
        return 1;
    }

    if (!expectNotSupported("cudaIpcOpenMemHandle", cudaIpcOpenMemHandle(&opened_ptr, ipc_handle, 0)) ||
        !expectLastErrorUpdated("cudaIpcOpenMemHandle")) {
        return 2;
    }

    if (!expectNotSupported("cudaIpcCloseMemHandle", cudaIpcCloseMemHandle(opened_ptr)) ||
        !expectLastErrorUpdated("cudaIpcCloseMemHandle")) {
        return 3;
    }

    int capture_status = 0;
    if (!expectNotSupported("cudaStreamBeginCapture", cudaStreamBeginCapture(nullptr, 0)) ||
        !expectLastErrorUpdated("cudaStreamBeginCapture")) {
        return 4;
    }

    if (!expectNotSupported("cudaStreamIsCapturing", cudaStreamIsCapturing(nullptr, &capture_status)) ||
        !expectLastErrorUpdated("cudaStreamIsCapturing")) {
        return 5;
    }

    void* mem_pool = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x1));
    if (!expectNotSupported("cudaMemPoolTrimTo", cudaMemPoolTrimTo(mem_pool, 0)) ||
        !expectLastErrorUpdated("cudaMemPoolTrimTo")) {
        return 6;
    }

    void* default_pool = mem_pool;
    if (!expectNotSupported("cudaDeviceGetDefaultMemPool", cudaDeviceGetDefaultMemPool(&default_pool, 0)) ||
        !expectLastErrorUpdated("cudaDeviceGetDefaultMemPool")) {
        return 7;
    }

    if (default_pool != nullptr) {
        std::fprintf(stderr, "cudaDeviceGetDefaultMemPool should leave output as nullptr on unsupported path\n");
        return 8;
    }

    std::printf("runtime_not_supported_semantics_test passed\n");
    return 0;
}
