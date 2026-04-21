#pragma once

#include <cstdint>

namespace vgpu {

static constexpr std::uint32_t kRpcMagic = 0x56504755;  // VPGU
static constexpr std::uint32_t kRpcVersion = 1;

enum class RpcOp : std::uint32_t {
    kCudaMalloc = 1,
    kCudaFree = 2,
    kCudaMemcpy = 3,
    kCudaSetDevice = 4,
    kCudaGetDevice = 5,
    kCudaDeviceSynchronize = 6,
    kCudaStreamCreate = 7,
    kCudaStreamDestroy = 8,
    kCudaLaunchKernel = 9,
    kCudaMemcpyAsync = 10,
    kCudaStreamSynchronize = 11,
    kCudaEventCreate = 12,
    kCudaEventDestroy = 13,
    kCudaEventRecord = 14,
    kCudaEventSynchronize = 15,
    kCudaMemset = 16,
    kCudaMemsetAsync = 17,
    kCudaGetDeviceCount = 18,
    kCudaMemGetInfo = 19,
    kCudaStreamQuery = 20,
    kCudaEventQuery = 21,
    kCudaStreamWaitEvent = 22,
    kCudaSetDeviceFlags = 23,
    kCudaDeviceReset = 24,
    kCudaRuntimeGetVersion = 25,
    kCudaDriverGetVersion = 26,
    kCudaDeviceGetAttribute = 27,
    kCudaGetDeviceFlags = 28
};

// ── CUDA Driver API forwarded ops ──────────────────────────────────────────
// These are used by both the libcudart.so shim (for kernel launch) and the
// libcuda.so shim (for direct driver API users such as ML frameworks).
enum class RpcDrvOp : std::uint32_t {
    kCuModuleLoadData    = 29,  // fatbin bytes → server loads, returns module_id
    kCuModuleGetFunction = 30,  // (module_id, func_name) → func_id + param info
    kCuLaunchKernel      = 31,  // (func_id, grid, block, sharedMem, stream) + arg buf
    kCuModuleUnload      = 32,  // module_id
    kCuMemAlloc          = 33,  // size → dev_ptr
    kCuMemFree           = 34,  // dev_ptr
    kCuMemcpyHtoD        = 35,  // (dst, count) + host data
    kCuMemcpyDtoH        = 36,  // (src, count) → host data
    kCuMemcpyDtoD        = 37,  // (dst, src, count)
    kCuCtxSynchronize    = 38,  // no payload
};

// ── Driver-op payload structs ───────────────────────────────────────────────
// kCuModuleLoadData: payload = raw fatbin bytes (variable size)
// response: aux_u64 = module_id

// kCuModuleGetFunction
// request payload: RpcCuModuleGetFunctionReq followed by null-terminated func name
struct RpcCuModuleGetFunctionReq {
    std::uint64_t module_id;
    std::uint32_t name_len;   // length of func name including null terminator
    std::uint32_t reserved0;
};
// response: aux_u64 = func_id, payload = RpcCuModuleGetFunctionRsp
struct RpcCuModuleGetFunctionRsp {
    std::uint32_t param_count;
    std::uint32_t total_param_bytes;
    // followed by param_count * RpcParamInfo
};
struct RpcParamInfo {
    std::uint32_t size;
    std::uint32_t alignment;
};

// kCuLaunchKernel
// request payload: RpcCuLaunchKernelReq followed by arg_buf_size bytes of packed args
struct RpcCuLaunchKernelReq {
    std::uint64_t func_id;
    std::uint32_t grid_x, grid_y, grid_z;
    std::uint32_t block_x, block_y, block_z;
    std::uint32_t shared_mem_bytes;
    std::uint32_t arg_buf_size;
    std::uint64_t stream;   // server-side cudaStream_t handle (0 = default stream)
};
// response: aux_u64 = CUresult (0 = success)

// kCuModuleUnload: aux_u64 = module_id (no payload)

// kCuMemAlloc: reuse RpcMallocReq / aux_u64 = dev_ptr
// kCuMemFree:  reuse RpcFreeReq

// kCuMemcpyHtoD / DtoH / DtoD: reuse RpcMemcpyReq (kind ignored for these)

struct RpcRequestHeader {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t op;
    std::uint32_t reserved0;

    std::uint64_t app_id;
    std::uint64_t context_id;

    std::int32_t device;
    std::uint32_t reserved1;

    std::uint64_t payload_size;
};

struct RpcResponseHeader {
    std::uint32_t magic;
    std::uint32_t version;
    std::int32_t status;
    std::uint32_t reserved0;

    std::uint64_t aux_u64;
    std::uint64_t payload_size;
};

struct RpcMallocReq {
    std::uint64_t size;
};

struct RpcFreeReq {
    std::uint64_t dev_ptr;
};

struct RpcMemcpyReq {
    std::uint64_t dst;
    std::uint64_t src;
    std::uint64_t count;
    std::int32_t kind;
    std::uint32_t reserved0;
};

struct RpcMemcpyAsyncReq {
    std::uint64_t dst;
    std::uint64_t src;
    std::uint64_t count;
    std::int32_t kind;
    std::uint32_t reserved0;
    std::uint64_t stream;
};

struct RpcSetDeviceReq {
    std::int32_t device;
    std::uint32_t reserved0;
};

struct RpcStreamDestroyReq {
    std::uint64_t stream;
};

struct RpcStreamSyncReq {
    std::uint64_t stream;
};

struct RpcEventCreateReq {
    std::uint32_t flags;
    std::uint32_t reserved0;
};

struct RpcEventDestroyReq {
    std::uint64_t event;
};

struct RpcEventRecordReq {
    std::uint64_t event;
    std::uint64_t stream;
};

struct RpcEventSynchronizeReq {
    std::uint64_t event;
};

struct RpcMemsetReq {
    std::uint64_t dst;
    std::int32_t value;
    std::uint32_t reserved0;
    std::uint64_t count;
};

struct RpcMemsetAsyncReq {
    std::uint64_t dst;
    std::int32_t value;
    std::uint32_t reserved0;
    std::uint64_t count;
    std::uint64_t stream;
};

struct RpcMemGetInfoRsp {
    std::uint64_t free_bytes;
    std::uint64_t total_bytes;
};

struct RpcStreamQueryReq {
    std::uint64_t stream;
};

struct RpcEventQueryReq {
    std::uint64_t event;
};

struct RpcStreamWaitEventReq {
    std::uint64_t stream;
    std::uint64_t event;
    std::uint32_t flags;
    std::uint32_t reserved0;
};

struct RpcSetDeviceFlagsReq {
    std::uint32_t flags;
    std::uint32_t reserved0;
};

struct RpcDeviceGetAttributeReq {
    std::int32_t attr;
    std::int32_t device;
};

}  // namespace vgpu
