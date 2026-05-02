#include <cstdio>
#include <cstddef>

#include "vgpu/common/cuda_abi.h"

extern "C" cudaError_t cudaPeekAtLastError(void);
extern "C" cudaError_t cudaStreamGetCaptureInfo_v2(
    cudaStream_t stream,
    int* captureStatus,
    unsigned long long* id,
    void** graph,
    const void*** dependencies,
    std::size_t* numDependencies);
extern "C" cudaError_t cudaStreamEndCapture(cudaStream_t stream, void** pGraph);
extern "C" cudaError_t cudaGetDriverEntryPoint(
    const char* symbol,
    void** funcPtr,
    unsigned long long flags,
    void* driverStatus);

namespace {

bool expectSuccess(const char* api, cudaError_t st) {
    if (st != cudaSuccess) {
        std::fprintf(stderr, "%s should return cudaSuccess(0), got %d\n", api, static_cast<int>(st));
        return false;
    }
    return true;
}

bool expectNotSupported(const char* api, cudaError_t st) {
    if (st != cudaErrorNotSupported) {
        std::fprintf(stderr, "%s should return cudaErrorNotSupported(801), got %d\n", api, static_cast<int>(st));
        return false;
    }
    return true;
}

bool expectLastError(const char* api, cudaError_t expected) {
    cudaError_t last = cudaPeekAtLastError();
    if (last != expected) {
        std::fprintf(stderr,
                     "%s should update last error to %d, got %d\n",
                     api,
                     static_cast<int>(expected),
                     static_cast<int>(last));
        return false;
    }
    return true;
}

}  // namespace

int main() {
    int capture_status = -1;
    unsigned long long capture_id = 123;
    void* graph = reinterpret_cast<void*>(0x1);
    const void** dependencies = reinterpret_cast<const void**>(0x1);
    std::size_t dependency_count = 99;

    if (!expectSuccess(
            "cudaStreamGetCaptureInfo_v2",
            cudaStreamGetCaptureInfo_v2(nullptr,
                                        &capture_status,
                                        &capture_id,
                                        &graph,
                                        &dependencies,
                                        &dependency_count)) ||
        !expectLastError("cudaStreamGetCaptureInfo_v2", cudaSuccess)) {
        return 1;
    }

    if (capture_status != 0 || capture_id != 0 || graph != nullptr ||
        dependencies != nullptr || dependency_count != 0) {
        std::fprintf(stderr,
                     "cudaStreamGetCaptureInfo_v2 should report empty non-capturing state, got status=%d id=%llu graph=%p deps=%p dep_count=%zu\n",
                     capture_status,
                     capture_id,
                     graph,
                     static_cast<const void*>(dependencies),
                     dependency_count);
        return 2;
    }

    void* cuda_malloc_fn = nullptr;
    int symbol_status = -1;
    if (!expectSuccess(
            "cudaGetDriverEntryPoint(cudaMalloc)",
            cudaGetDriverEntryPoint("cudaMalloc", &cuda_malloc_fn, 0, &symbol_status)) ||
        !expectLastError("cudaGetDriverEntryPoint(cudaMalloc)", cudaSuccess)) {
        return 3;
    }

    if (cuda_malloc_fn == nullptr || symbol_status != 0) {
        std::fprintf(stderr,
                     "cudaGetDriverEntryPoint(cudaMalloc) should resolve a symbol, got fn=%p status=%d\n",
                     cuda_malloc_fn,
                     symbol_status);
        return 4;
    }

    void* missing_fn = reinterpret_cast<void*>(0x1);
    symbol_status = -1;
    if (!expectSuccess(
            "cudaGetDriverEntryPoint(missing)",
            cudaGetDriverEntryPoint("cudaDefinitelyMissingSymbol", &missing_fn, 0, &symbol_status)) ||
        !expectLastError("cudaGetDriverEntryPoint(missing)", cudaSuccess)) {
        return 5;
    }

    if (missing_fn != nullptr || symbol_status != 1) {
        std::fprintf(stderr,
                     "cudaGetDriverEntryPoint(missing) should report symbol miss, got fn=%p status=%d\n",
                     missing_fn,
                     symbol_status);
        return 6;
    }

    void* captured_graph = reinterpret_cast<void*>(0x1);
    if (!expectNotSupported("cudaStreamEndCapture", cudaStreamEndCapture(nullptr, &captured_graph)) ||
        !expectLastError("cudaStreamEndCapture", cudaErrorNotSupported)) {
        return 7;
    }

    if (captured_graph != nullptr) {
        std::fprintf(stderr, "cudaStreamEndCapture should null the graph output on unsupported path\n");
        return 8;
    }

    std::printf("runtime_capability_boundary_test passed\n");
    return 0;
}