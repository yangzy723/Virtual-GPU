#pragma once

#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// Minimal CUDA Driver API types (no cuda.h required)
// ---------------------------------------------------------------------------

typedef int          CUresult;
typedef unsigned int CUdevice;
typedef std::size_t  CUdeviceptr;

struct CUmod_st;
struct CUfunc_st;
struct CUstream_st;
struct CUctx_st;

typedef CUmod_st*    CUmodule;
typedef CUfunc_st*   CUfunction;
typedef CUstream_st* CUstream_drv;  // disambiguate from cudaStream_t
typedef CUctx_st*    CUcontext;

static constexpr CUresult CUDA_SUCCESS                = 0;
static constexpr CUresult CUDA_ERROR_INVALID_VALUE    = 1;
static constexpr CUresult CUDA_ERROR_OUT_OF_MEMORY    = 2;
static constexpr CUresult CUDA_ERROR_NOT_INITIALIZED  = 3;
static constexpr CUresult CUDA_ERROR_NOT_READY        = 600;
static constexpr CUresult CUDA_ERROR_NOT_FOUND        = 500;
static constexpr CUresult CUDA_ERROR_NOT_SUPPORTED    = 801;
static constexpr CUresult CUDA_ERROR_UNKNOWN          = 999;

// cuFuncGetAttribute enum value we need
static constexpr int CU_FUNC_ATTR_PARAM_SIZE_BYTES = 11;

// cuLaunchKernel extra[] sentinels
// Use inline functions to avoid constexpr reinterpret_cast restriction
inline void* cu_launch_param_end()            { return reinterpret_cast<void*>(0x00); }
inline void* cu_launch_param_buffer_pointer() { return reinterpret_cast<void*>(0x01); }
inline void* cu_launch_param_buffer_size()    { return reinterpret_cast<void*>(0x02); }
#define CU_LAUNCH_PARAM_END            cu_launch_param_end()
#define CU_LAUNCH_PARAM_BUFFER_POINTER cu_launch_param_buffer_pointer()
#define CU_LAUNCH_PARAM_BUFFER_SIZE    cu_launch_param_buffer_size()

namespace vgpu {

// All driver API symbols the server needs to execute kernel launches
struct CudaDriverApi {
    // Init / device
    CUresult (*cuInit)(unsigned int flags);
    CUresult (*cuDeviceGetCount)(int* count);
    CUresult (*cuDeviceGet)(CUdevice* device, int ordinal);

    // Context (needed for module loading)
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext* pctx, CUdevice dev);
    CUresult (*cuCtxSetCurrent)(CUcontext ctx);
    CUresult (*cuCtxGetCurrent)(CUcontext* pctx);

    // Module
    CUresult (*cuModuleLoadData)(CUmodule* module, const void* image);
    CUresult (*cuModuleGetFunction)(CUfunction* hfunc, CUmodule hmod, const char* name);
    CUresult (*cuModuleUnload)(CUmodule hmod);

    // Launch
    CUresult (*cuLaunchKernel)(
        CUfunction f,
        unsigned int gridX,  unsigned int gridY,  unsigned int gridZ,
        unsigned int blockX, unsigned int blockY, unsigned int blockZ,
        unsigned int sharedMem, CUstream_drv stream,
        void** kernelParams, void** extra);

    // Memory
    CUresult (*cuMemAlloc_v2)(CUdeviceptr* dptr, std::size_t size);
    CUresult (*cuMemFree_v2)(CUdeviceptr dptr);
    CUresult (*cuMemcpyHtoD_v2)(CUdeviceptr dst, const void* src, std::size_t count);
    CUresult (*cuMemcpyDtoH_v2)(void* dst, CUdeviceptr src, std::size_t count);
    CUresult (*cuMemcpyDtoD_v2)(CUdeviceptr dst, CUdeviceptr src, std::size_t count);

    // Sync
    CUresult (*cuStreamSynchronize)(CUstream_drv stream);
    CUresult (*cuCtxSynchronize)();

    // Introspection
    CUresult (*cuFuncGetAttribute)(int* pi, int attrib, CUfunction hfunc);
    CUresult (*cuFuncGetParamInfo)(CUfunction func, std::size_t paramIndex, std::size_t* paramOffset, std::size_t* paramSize);

    // Diagnostics
    CUresult (*cuGetErrorName)(CUresult error, const char** pStr);
    CUresult (*cuGetErrorString)(CUresult error, const char** pStr);
};

class CudaDriverLoader {
public:
    // dlopen libcuda.so.1 and resolve symbols.
    // Returns true if all required symbols were found.
    bool ensureLoaded();

    bool loaded() const { return loaded_; }
    const CudaDriverApi& api() const { return api_; }

private:
    bool         loaded_ = false;
    void*        handle_ = nullptr;
    CudaDriverApi api_{};
};

}  // namespace vgpu
