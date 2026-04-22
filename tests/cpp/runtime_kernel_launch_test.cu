// runtime_kernel_launch_test.cu
// Real CUDA kernel test – exercises the full fatbin registration + launch path.
// Compiled with NVCC, linked against libcudart.so (our shim), so ALL fatbin
// registration callbacks go through our interceptor and are forwarded to the
// vgpu server for execution.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

// ── Simple kernels ─────────────────────────────────────────────────────────

// Add two float arrays element-wise
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Scale an array in-place
__global__ void scaleInPlace(float* a, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= scale;
    }
}

// Dot-product reduction (single-block, for small arrays)
__global__ void dotProduct(const float* a, const float* b, float* result, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < n) ? a[idx] * b[idx] : 0.0f;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *result = sdata[0];
}

// ── Host helpers ───────────────────────────────────────────────────────────

#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %d at %s:%d: %s\n",            \
                         (int)_e, __FILE__, __LINE__,                       \
                         cudaGetErrorString(_e));                            \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

int main() {
    constexpr int N = 256;
    constexpr int BLOCK = 256;

    int dev_count = 0;
    CHECK(cudaGetDeviceCount(&dev_count));
    std::printf("[kernel-app] device count = %d\n", dev_count);

    // ── Allocate host buffers ──────────────────────────────────────────────
    float h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(N - i);
    }

    // ── Allocate device buffers ────────────────────────────────────────────
    float *d_a, *d_b, *d_c, *d_dot;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), N * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), N * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), N * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dot), sizeof(float)));

    // ── Upload ────────────────────────────────────────────────────────────
    CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    // ── Launch vectorAdd ──────────────────────────────────────────────────
    vectorAdd<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_a, d_b, d_c, N);
    CHECK(cudaDeviceSynchronize());
    std::printf("[kernel-app] vectorAdd launched\n");

    // Verify: c[i] = i + (N - i) = N for all i
    CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != static_cast<float>(N)) {
            std::fprintf(stderr,
                         "vectorAdd mismatch at %d: got %f expected %f\n",
                         i, h_c[i], static_cast<float>(N));
            return 1;
        }
    }
    std::printf("[kernel-app] vectorAdd verified OK\n");

    // ── Launch scaleInPlace (c[i] *= 2) ──────────────────────────────────
    scaleInPlace<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_c, 2.0f, N);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != static_cast<float>(2 * N)) {
            std::fprintf(stderr, "scaleInPlace mismatch at %d\n", i);
            return 1;
        }
    }
    std::printf("[kernel-app] scaleInPlace verified OK\n");

    // ── Launch dotProduct (a · b) ─────────────────────────────────────────
    dotProduct<<<1, BLOCK>>>(d_a, d_b, d_dot, N);
    CHECK(cudaDeviceSynchronize());

    float h_dot = 0.0f;
    CHECK(cudaMemcpy(&h_dot, d_dot, sizeof(float), cudaMemcpyDeviceToHost));

    // Expected: sum_{i=0}^{N-1} i*(N-i)
    float expected_dot = 0.0f;
    for (int i = 0; i < N; ++i) expected_dot += h_a[i] * h_b[i];

    float rel_err = std::abs(h_dot - expected_dot) / (std::abs(expected_dot) + 1e-6f);
    if (rel_err > 1e-4f) {
        std::fprintf(stderr, "dotProduct mismatch: got %f expected %f\n",
                     h_dot, expected_dot);
        return 1;
    }
    std::printf("[kernel-app] dotProduct = %f (expected %f) OK\n",
                h_dot, expected_dot);

    // ── Cleanup ───────────────────────────────────────────────────────────
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    CHECK(cudaFree(d_dot));

    std::printf("[kernel-app] all kernel tests passed\n");
    return 0;
}
