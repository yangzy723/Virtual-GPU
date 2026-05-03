#pragma once

#include <cstddef>
#include <cstdint>

// CUDA Driver API 类型定义 — 不依赖 CUDA SDK 头文件

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef void* CUevent;

static constexpr CUresult CUDA_SUCCESS = 0;
static constexpr CUresult CUDA_ERROR_INVALID_VALUE = 1;
static constexpr CUresult CUDA_ERROR_OUT_OF_MEMORY = 2;
static constexpr CUresult CUDA_ERROR_NOT_INITIALIZED = 3;
static constexpr CUresult CUDA_ERROR_NO_DEVICE = 100;
static constexpr CUresult CUDA_ERROR_INVALID_CONTEXT = 201;
static constexpr CUresult CUDA_ERROR_NOT_FOUND = 500;
static constexpr CUresult CUDA_ERROR_LAUNCH_FAILED = 719;
static constexpr CUresult CUDA_ERROR_NOT_SUPPORTED = 801;
