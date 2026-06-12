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
typedef void* CUarray;
typedef void* CUstream;
typedef void* CUevent;
typedef struct CUlaunchAttribute_st CUlaunchAttribute;

static constexpr unsigned int CU_EVENT_DISABLE_TIMING = 0x2;

typedef enum CUmemorytype_enum {
	CU_MEMORYTYPE_HOST    = 0x01,
	CU_MEMORYTYPE_DEVICE  = 0x02,
	CU_MEMORYTYPE_ARRAY   = 0x03,
	CU_MEMORYTYPE_UNIFIED = 0x04,
} CUmemorytype;

typedef struct CUDA_MEMCPY2D_st {
	size_t srcXInBytes;
	size_t srcY;
	CUmemorytype srcMemoryType;
	const void* srcHost;
	CUdeviceptr srcDevice;
	CUarray srcArray;
	size_t srcPitch;
	size_t dstXInBytes;
	size_t dstY;
	CUmemorytype dstMemoryType;
	void* dstHost;
	CUdeviceptr dstDevice;
	CUarray dstArray;
	size_t dstPitch;
	size_t WidthInBytes;
	size_t Height;
} CUDA_MEMCPY2D_v2;
typedef CUDA_MEMCPY2D_v2 CUDA_MEMCPY2D;

typedef struct CUDA_MEMCPY3D_st {
	size_t srcXInBytes;
	size_t srcY;
	size_t srcZ;
	size_t srcLOD;
	CUmemorytype srcMemoryType;
	const void* srcHost;
	CUdeviceptr srcDevice;
	CUarray srcArray;
	void* reserved0;
	size_t srcPitch;
	size_t srcHeight;
	size_t dstXInBytes;
	size_t dstY;
	size_t dstZ;
	size_t dstLOD;
	CUmemorytype dstMemoryType;
	void* dstHost;
	CUdeviceptr dstDevice;
	CUarray dstArray;
	void* reserved1;
	size_t dstPitch;
	size_t dstHeight;
	size_t WidthInBytes;
	size_t Height;
	size_t Depth;
} CUDA_MEMCPY3D_v2;
typedef CUDA_MEMCPY3D_v2 CUDA_MEMCPY3D;

typedef struct CUDA_MEMCPY3D_PEER_st {
	size_t srcXInBytes;
	size_t srcY;
	size_t srcZ;
	size_t srcLOD;
	CUmemorytype srcMemoryType;
	const void* srcHost;
	CUdeviceptr srcDevice;
	CUarray srcArray;
	CUcontext srcContext;
	size_t srcPitch;
	size_t srcHeight;
	size_t dstXInBytes;
	size_t dstY;
	size_t dstZ;
	size_t dstLOD;
	CUmemorytype dstMemoryType;
	void* dstHost;
	CUdeviceptr dstDevice;
	CUarray dstArray;
	CUcontext dstContext;
	size_t dstPitch;
	size_t dstHeight;
	size_t WidthInBytes;
	size_t Height;
	size_t Depth;
} CUDA_MEMCPY3D_PEER_v1;
typedef CUDA_MEMCPY3D_PEER_v1 CUDA_MEMCPY3D_PEER;

typedef struct CUlaunchConfig_st {
	unsigned int gridDimX;
	unsigned int gridDimY;
	unsigned int gridDimZ;
	unsigned int blockDimX;
	unsigned int blockDimY;
	unsigned int blockDimZ;
	unsigned int sharedMemBytes;
	CUstream hStream;
	CUlaunchAttribute* attrs;
	unsigned int numAttrs;
} CUlaunchConfig;

typedef struct CUmemcpyAttributes_st CUmemcpyAttributes;

static constexpr CUresult CUDA_SUCCESS = 0;
static constexpr CUresult CUDA_ERROR_INVALID_VALUE = 1;
static constexpr CUresult CUDA_ERROR_OUT_OF_MEMORY = 2;
static constexpr CUresult CUDA_ERROR_NOT_INITIALIZED = 3;
static constexpr CUresult CUDA_ERROR_NO_DEVICE = 100;
static constexpr CUresult CUDA_ERROR_INVALID_CONTEXT = 201;
static constexpr CUresult CUDA_ERROR_NOT_FOUND = 500;
static constexpr CUresult CUDA_ERROR_NOT_READY = 600;
static constexpr CUresult CUDA_ERROR_LAUNCH_FAILED = 719;
static constexpr CUresult CUDA_ERROR_NOT_SUPPORTED = 801;
