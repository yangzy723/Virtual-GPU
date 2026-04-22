#pragma once

#include <cstddef>

// Minimal CUDA Runtime ABI subset to compile interceptor without CUDA headers.
// Numeric values for enum entries match CUDA runtime API public constants.

typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

struct dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;

    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
};

static constexpr cudaError_t cudaSuccess = 0;
static constexpr cudaError_t cudaErrorInvalidValue = 1;
static constexpr cudaError_t cudaErrorNotSupported = 801;
static constexpr cudaError_t cudaErrorUnknown = 999;

static constexpr unsigned int cudaEventDefault = 0;
