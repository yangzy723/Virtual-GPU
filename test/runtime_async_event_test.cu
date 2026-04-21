#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

int main() {
    constexpr int n = 4096;
    const size_t bytes = static_cast<size_t>(n) * sizeof(unsigned char);

    int device_count = 0;
    cudaError_t st = cudaGetDeviceCount(&device_count);
    if (st != cudaSuccess || device_count <= 0) {
        std::fprintf(stderr, "cudaGetDeviceCount failed: %d, count=%d\n", static_cast<int>(st), device_count);
        return 1;
    }

    std::vector<unsigned char> h_out(n, 0);

    unsigned char* d_buf = nullptr;
    cudaStream_t stream = nullptr;
    cudaStream_t stream2 = nullptr;
    cudaEvent_t event = nullptr;

    st = cudaStreamCreate(&stream);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaStreamCreate(stream) failed: %d\n", static_cast<int>(st));
        return 2;
    }

    st = cudaStreamCreate(&stream2);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaStreamCreate(stream2) failed: %d\n", static_cast<int>(st));
        cudaStreamDestroy(stream);
        return 3;
    }

    st = cudaEventCreate(&event);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaEventCreate failed: %d\n", static_cast<int>(st));
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream);
        return 4;
    }

    size_t free_before = 0;
    size_t total_before = 0;
    st = cudaMemGetInfo(&free_before, &total_before);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaMemGetInfo(before) failed: %d\n", static_cast<int>(st));
        cudaEventDestroy(event);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream);
        return 5;
    }

    st = cudaMalloc(reinterpret_cast<void**>(&d_buf), bytes);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed: %d\n", static_cast<int>(st));
        cudaEventDestroy(event);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream);
        return 6;
    }

    st = cudaMemsetAsync(d_buf, 0x5a, bytes, stream);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaMemsetAsync failed: %d\n", static_cast<int>(st));
        cudaFree(d_buf);
        cudaEventDestroy(event);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream);
        return 7;
    }

    st = cudaEventRecord(event, stream);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaEventRecord failed: %d\n", static_cast<int>(st));
        cudaFree(d_buf);
        cudaEventDestroy(event);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream);
        return 8;
    }

    st = cudaEventQuery(event);
    if (st != cudaSuccess && st != cudaErrorNotReady) {
        std::fprintf(stderr, "cudaEventQuery unexpected status: %d\n", static_cast<int>(st));
        cudaFree(d_buf);
        cudaEventDestroy(event);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream);
        return 9;
    }

    st = cudaEventSynchronize(event);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaEventSynchronize failed: %d\n", static_cast<int>(st));
        cudaFree(d_buf);
        cudaEventDestroy(event);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream);
        return 10;
    }

    st = cudaStreamWaitEvent(stream2, event, 0);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaStreamWaitEvent failed: %d\n", static_cast<int>(st));
        cudaFree(d_buf);
        cudaEventDestroy(event);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream);
        return 11;
    }

    st = cudaMemcpyAsync(h_out.data(), d_buf, bytes, cudaMemcpyDeviceToHost, stream2);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpyAsync D2H failed: %d\n", static_cast<int>(st));
        cudaFree(d_buf);
        cudaEventDestroy(event);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream);
        return 12;
    }

    st = cudaStreamQuery(stream2);
    if (st != cudaSuccess && st != cudaErrorNotReady) {
        std::fprintf(stderr, "cudaStreamQuery unexpected status: %d\n", static_cast<int>(st));
        cudaFree(d_buf);
        cudaEventDestroy(event);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream);
        return 13;
    }

    st = cudaStreamSynchronize(stream2);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaStreamSynchronize(stream2) failed: %d\n", static_cast<int>(st));
        cudaFree(d_buf);
        cudaEventDestroy(event);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream);
        return 14;
    }

    for (int i = 0; i < n; ++i) {
        if (h_out[i] != 0x5a) {
            std::fprintf(stderr, "verification failed at %d: got %u\n", i, static_cast<unsigned int>(h_out[i]));
            cudaFree(d_buf);
            cudaEventDestroy(event);
            cudaStreamDestroy(stream2);
            cudaStreamDestroy(stream);
            return 15;
        }
    }

    st = cudaMemset(d_buf, 0x11, bytes);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaMemset failed: %d\n", static_cast<int>(st));
        cudaFree(d_buf);
        cudaEventDestroy(event);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream);
        return 16;
    }

    size_t free_after = 0;
    size_t total_after = 0;
    st = cudaMemGetInfo(&free_after, &total_after);
    if (st != cudaSuccess || total_after != total_before) {
        std::fprintf(stderr, "cudaMemGetInfo(after) failed: %d, total_before=%zu total_after=%zu\n", static_cast<int>(st), total_before, total_after);
        cudaFree(d_buf);
        cudaEventDestroy(event);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream);
        return 17;
    }

    st = cudaFree(d_buf);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaFree failed: %d\n", static_cast<int>(st));
        cudaEventDestroy(event);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream);
        return 18;
    }

    st = cudaEventDestroy(event);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaEventDestroy failed: %d\n", static_cast<int>(st));
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream);
        return 19;
    }

    st = cudaStreamDestroy(stream2);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaStreamDestroy(stream2) failed: %d\n", static_cast<int>(st));
        cudaStreamDestroy(stream);
        return 20;
    }

    st = cudaStreamDestroy(stream);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaStreamDestroy failed: %d\n", static_cast<int>(st));
        return 21;
    }

    std::printf("runtime_async_event_test passed\n");
    return 0;
}
