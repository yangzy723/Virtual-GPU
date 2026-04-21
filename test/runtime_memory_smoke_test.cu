#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

int main() {
    constexpr int n = 1024;
    const size_t bytes = static_cast<size_t>(n) * sizeof(int);

    std::vector<int> h_in(n, 7);
    std::vector<int> h_out(n, 0);

    int* d_buf = nullptr;

    cudaError_t st = cudaMalloc(reinterpret_cast<void**>(&d_buf), bytes);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed: %d\n", static_cast<int>(st));
        return 1;
    }

    st = cudaMemcpy(d_buf, h_in.data(), bytes, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy H2D failed: %d\n", static_cast<int>(st));
        cudaFree(d_buf);
        return 2;
    }

    st = cudaMemcpy(h_out.data(), d_buf, bytes, cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy D2H failed: %d\n", static_cast<int>(st));
        cudaFree(d_buf);
        return 3;
    }

    st = cudaFree(d_buf);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "cudaFree failed: %d\n", static_cast<int>(st));
        return 4;
    }

    for (int i = 0; i < n; ++i) {
        if (h_out[i] != 7) {
            std::fprintf(stderr, "verification failed at %d: got %d\n", i, h_out[i]);
            return 5;
        }
    }

    std::printf("runtime_memory_smoke_test passed\n");
    return 0;
}
