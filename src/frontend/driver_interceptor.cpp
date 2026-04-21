// driver_interceptor.cpp
//
// Drop-in libcuda.so replacement.  Intercepts the CUDA Driver API and
// forwards all work to the vgpu server over Unix-domain socket RPC.
//
// Key design points:
//
//  1. cuModuleLoadData / cuModuleLoadDataEx
//     - Parses the fatbin header to determine binary size.
//     - Extracts PTX via fatbin_parser to obtain per-kernel param info.
//     - Sends the raw fatbin bytes to the server (kCuModuleLoadData).
//     - Returns a fake CUmodule handle (heap-allocated sentinel).
//
//  2. cuModuleGetFunction
//     - Sends (server_module_id, func_name) to server (kCuModuleGetFunction).
//     - Server responds with func_id + param_count + param_sizes[].
//     - Returns a fake CUfunction handle; stores mapping in kernel_registry.
//
//  3. cuLaunchKernel
//     - If extra[] contains CU_LAUNCH_PARAM_BUFFER_POINTER/SIZE: the packed
//       arg buffer is already ready (this happens when real libcudart.so calls
//       us in LD_PRELOAD mode).
//     - Otherwise packs void** kernelParams using stored param info.
//     - Sends kCuLaunchKernel RPC with packed buffer.
//
//  4. Memory ops (cuMemAlloc_v2, cuMemFree_v2, cuMemcpy*)
//     - Forward to server using existing kCuMemAlloc / kCuMemFree / kCuMemcpy*
//       driver ops.
//
//  5. Context / device helpers
//     - cuInit → always returns CUDA_SUCCESS (no local GPU needed).
//     - cuDeviceGetCount / cuDeviceGet → forwarded via kCudaGetDeviceCount /
//       kCudaGetDevice so they share the same server-side path.
//     - cuCtxCreate / cuCtxDestroy → managed locally (no real context needed).

#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

#include "vgpu/common/context_registry.h"
#include "vgpu/common/cuda_abi.h"
#include "vgpu/backend/cuda_driver_loader.h"
#include "vgpu/common/fatbin_parser.h"
#include "vgpu/common/kernel_registry.h"
#include "vgpu/common/protocol.h"
#include "vgpu/common/rpc_client.h"

namespace vgpu {
namespace {

// ── RPC helpers ──────────────────────────────────────────────────────────────

RpcClient& drvClient() {
    static RpcClient c;
    return c;
}

ContextRegistry& drvContexts() {
    static ContextRegistry c;
    return c;
}

thread_local int g_drv_device = 0;

RpcResult drvCall(
    RpcOp op,
    const void* payload,
    std::size_t payload_size,
    const void* extra_payload = nullptr,
    std::size_t extra_size    = 0) {
    std::uint64_t app_id     = static_cast<std::uint64_t>(getpid());
    std::uint64_t context_id = drvContexts().acquireContextId(g_drv_device);
    return drvClient().call(op, app_id, context_id, g_drv_device,
                            payload, payload_size, extra_payload, extra_size);
}

RpcResult drvCallDrv(
    RpcDrvOp op,
    const void* payload,
    std::size_t payload_size,
    const void* extra_payload = nullptr,
    std::size_t extra_size    = 0) {
    return drvCall(static_cast<RpcOp>(op), payload, payload_size, extra_payload, extra_size);
}

// ── Fake handle counter ──────────────────────────────────────────────────────
std::atomic<std::uintptr_t> g_next_fake_handle{0x10000};

void* makeFakeHandle() {
    return reinterpret_cast<void*>(g_next_fake_handle.fetch_add(1));
}

// ── Fatbin image size detection ──────────────────────────────────────────────
// Returns total byte size of the raw fatbin image (magic 0xBA55ED50).
// If the pointer is to a wrapper struct (magic 0x466243B1), follows it first.
size_t fatbinImageSize(const void* image) {
    if (!image) return 0;
    static constexpr uint32_t kWrapMagic = 0x466243B1u;
    static constexpr uint32_t kFatMagic  = 0xBA55ED50u;

    const uint8_t* p = static_cast<const uint8_t*>(image);
    uint32_t magic = 0;
    std::memcpy(&magic, p, sizeof(magic));

    if (magic == kWrapMagic) {
        const void* data = nullptr;
        std::memcpy(&data, p + 8, sizeof(data));
        if (!data) return 0;
        p = static_cast<const uint8_t*>(data);
        std::memcpy(&magic, p, sizeof(magic));
    }
    if (magic != kFatMagic) return 0;

    uint32_t hdr_size = 0, data_size = 0;
    std::memcpy(&hdr_size,  p + 8,  sizeof(hdr_size));
    std::memcpy(&data_size, p + 12, sizeof(data_size));
    return static_cast<size_t>(hdr_size) + static_cast<size_t>(data_size);
}

// Follow wrapper → raw if needed; return pointer to raw fatbin
const void* resolveRawFatbin(const void* image) {
    static constexpr uint32_t kWrapMagic = 0x466243B1u;
    if (!image) return nullptr;
    const uint8_t* p = static_cast<const uint8_t*>(image);
    uint32_t magic = 0;
    std::memcpy(&magic, p, sizeof(magic));
    if (magic == kWrapMagic) {
        const void* data = nullptr;
        std::memcpy(&data, p + 8, sizeof(data));
        return data;
    }
    return image;
}

}  // namespace
}  // namespace vgpu

// ============================================================================
// Exported cu* symbols
// ============================================================================

extern "C" {

// ── Init / device ────────────────────────────────────────────────────────────

CUresult cuInit(unsigned int /*flags*/) {
    return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion(int* driverVersion) {
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaDriverGetVersion, nullptr, 0);
    if (r.status == cudaSuccess && driverVersion)
        *driverVersion = static_cast<int>(r.aux_u64);
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult cuDeviceGetCount(int* count) {
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaGetDeviceCount, nullptr, 0);
    if (r.status == cudaSuccess && count)
        *count = static_cast<int>(r.aux_u64);
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    if (device) *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char* name, int len, CUdevice /*dev*/) {
    if (name && len > 0) std::strncpy(name, "vGPU Remote Device", static_cast<size_t>(len - 1));
    return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice /*dev*/) {
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaMemGetInfo, nullptr, 0);
    if (r.status == cudaSuccess && bytes && r.payload.size() >= sizeof(vgpu::RpcMemGetInfoRsp)) {
        const auto* info = reinterpret_cast<const vgpu::RpcMemGetInfoRsp*>(r.payload.data());
        *bytes = static_cast<size_t>(info->total_bytes);
    }
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

// ── Context (stub – no real context on client) ───────────────────────────────

CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int /*flags*/, CUdevice dev) {
    vgpu::g_drv_device = static_cast<int>(dev);
    if (pctx) *pctx = reinterpret_cast<CUcontext>(vgpu::makeFakeHandle());
    return CUDA_SUCCESS;
}

CUresult cuCtxDestroy_v2(CUcontext /*ctx*/) { return CUDA_SUCCESS; }
CUresult cuCtxGetCurrent(CUcontext* pctx) {
    if (pctx) *pctx = reinterpret_cast<CUcontext>(static_cast<uintptr_t>(0xC7AC0001));
    return CUDA_SUCCESS;
}
CUresult cuCtxSetCurrent(CUcontext /*ctx*/) { return CUDA_SUCCESS; }
CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
    return cuCtxCreate_v2(pctx, 0, dev);
}
CUresult cuDevicePrimaryCtxRelease_v2(CUdevice /*dev*/) { return CUDA_SUCCESS; }
CUresult cuCtxSynchronize() {
    vgpu::RpcResult r = vgpu::drvCallDrv(vgpu::RpcDrvOp::kCuCtxSynchronize, nullptr, 0);
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

// ── Module loading ───────────────────────────────────────────────────────────

static CUresult doModuleLoad(CUmodule* pmod, const void* image) {
    if (!image || !pmod) return CUDA_ERROR_INVALID_VALUE;

    // Follow wrapper if needed
    const void* raw = vgpu::resolveRawFatbin(image);
    size_t raw_size = vgpu::fatbinImageSize(image);
    if (!raw || raw_size == 0) return CUDA_ERROR_INVALID_VALUE;

    // Parse PTX for param info, store in kernel registry
    auto kernel_infos = vgpu::parseFatbin(raw, raw_size);
    for (const auto& ki : kernel_infos) {
        vgpu::globalKernelRegistry().addParamInfo(ki.mangled_name, ki.params);
    }

    // Send raw fatbin to server
    vgpu::RpcResult r = vgpu::drvCallDrv(
        vgpu::RpcDrvOp::kCuModuleLoadData, nullptr, 0, raw, raw_size);

    if (r.status != cudaSuccess) return CUDA_ERROR_UNKNOWN;

    std::uint64_t module_id = r.aux_u64;
    void* fake_handle = vgpu::makeFakeHandle();
    vgpu::globalKernelRegistry().addDriverModule(fake_handle, module_id);
    *pmod = reinterpret_cast<CUmodule>(fake_handle);
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    return doModuleLoad(module, image);
}

CUresult cuModuleLoadDataEx(CUmodule* module, const void* image,
                            unsigned int /*numOptions*/,
                            int* /*options*/, void** /*optionValues*/) {
    return doModuleLoad(module, image);
}

CUresult cuModuleLoad(CUmodule* module, const char* fname) {
    // Read file and forward as cuModuleLoadData
    if (!fname || !module) return CUDA_ERROR_INVALID_VALUE;
    FILE* f = fopen(fname, "rb");
    if (!f) return CUDA_ERROR_NOT_FOUND;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    if (sz <= 0) { fclose(f); return CUDA_ERROR_INVALID_VALUE; }
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    size_t read_bytes = fread(buf.data(), 1, buf.size(), f);
    fclose(f);
    if (read_bytes != buf.size()) return CUDA_ERROR_UNKNOWN;
    return doModuleLoad(module, buf.data());
}

CUresult cuModuleUnload(CUmodule hmod) {
    std::uint64_t module_id =
        vgpu::globalKernelRegistry().getDriverModuleId(reinterpret_cast<void*>(hmod));
    if (module_id == 0) return CUDA_SUCCESS; // already gone
    std::uint64_t mid = module_id;
    vgpu::RpcResult r = vgpu::drvCallDrv(
        vgpu::RpcDrvOp::kCuModuleUnload, &mid, sizeof(mid));
    (void)r;
    return CUDA_SUCCESS;
}

// ── Function lookup ──────────────────────────────────────────────────────────

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    if (!hfunc || !name) return CUDA_ERROR_INVALID_VALUE;

    void* mod_handle = reinterpret_cast<void*>(hmod);
    std::uint64_t module_id = vgpu::globalKernelRegistry().getDriverModuleId(mod_handle);
    if (module_id == 0) return CUDA_ERROR_INVALID_VALUE;

    // Build request: header + null-terminated name
    size_t name_len = std::strlen(name) + 1;
    vgpu::RpcCuModuleGetFunctionReq req{};
    req.module_id = module_id;
    req.name_len  = static_cast<std::uint32_t>(name_len);

    std::vector<std::uint8_t> payload(sizeof(req) + name_len);
    std::memcpy(payload.data(), &req, sizeof(req));
    std::memcpy(payload.data() + sizeof(req), name, name_len);

    vgpu::RpcResult r = vgpu::drvCallDrv(
        vgpu::RpcDrvOp::kCuModuleGetFunction,
        payload.data(), payload.size());

    if (r.status != cudaSuccess) return CUDA_ERROR_NOT_FOUND;

    std::uint64_t func_id = r.aux_u64;

    // Decode response param info
    vgpu::KernelEntry ke{};
    ke.func_id = func_id;

    if (r.payload.size() >= sizeof(vgpu::RpcCuModuleGetFunctionRsp)) {
        const auto* rsp = reinterpret_cast<const vgpu::RpcCuModuleGetFunctionRsp*>(r.payload.data());
        ke.total_param_bytes = rsp->total_param_bytes;
        std::uint32_t pc = rsp->param_count;
        const vgpu::RpcParamInfo* pi = reinterpret_cast<const vgpu::RpcParamInfo*>(rsp + 1);
        if (r.payload.size() >= sizeof(vgpu::RpcCuModuleGetFunctionRsp) + pc * sizeof(*pi)) {
            for (std::uint32_t i = 0; i < pc; ++i) {
                vgpu::ParamInfo p{};
                p.size      = pi[i].size;
                p.alignment = pi[i].alignment;
                ke.params.push_back(p);
            }
        }
    }

    // If server didn't return param info, try local PTX cache
    if (ke.params.empty()) {
        const auto* cached = vgpu::globalKernelRegistry().findParamInfo(name);
        if (cached) {
            ke.params = *cached;
            ke.total_param_bytes = vgpu::computeParamBufSize(ke.params);
        }
    }

    void* fake_func = vgpu::makeFakeHandle();
    vgpu::globalKernelRegistry().addDriverFunc(fake_func, std::move(ke));
    *hfunc = reinterpret_cast<CUfunction>(fake_func);
    return CUDA_SUCCESS;
}

CUresult cuFuncGetAttribute(int* pi, int attrib, CUfunction hfunc) {
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    // CU_FUNC_ATTRIBUTE_PARAM_SIZE_BYTES = 11
    if (attrib == CU_FUNC_ATTR_PARAM_SIZE_BYTES) {
        const vgpu::KernelEntry* ke =
            vgpu::globalKernelRegistry().findDriverFunc(reinterpret_cast<void*>(hfunc));
        if (ke) { *pi = static_cast<int>(ke->total_param_bytes); return CUDA_SUCCESS; }
    }
    *pi = 0;
    return CUDA_SUCCESS;
}

// ── Kernel launch ─────────────────────────────────────────────────────────────

CUresult cuLaunchKernel(
    CUfunction f,
    unsigned int gridX,  unsigned int gridY,  unsigned int gridZ,
    unsigned int blockX, unsigned int blockY, unsigned int blockZ,
    unsigned int sharedMem,
    CUstream_drv stream,
    void** kernelParams,
    void** extra) {

    const vgpu::KernelEntry* ke =
        vgpu::globalKernelRegistry().findDriverFunc(reinterpret_cast<void*>(f));
    if (!ke) return CUDA_ERROR_INVALID_VALUE;

    // Determine packed arg buffer
    const void* arg_buf = nullptr;
    size_t arg_buf_size = 0;
    std::vector<std::uint8_t> packed;

    if (extra) {
        // Scan extra[] for CU_LAUNCH_PARAM_BUFFER_POINTER / _BUFFER_SIZE
        for (int i = 0; extra[i] != CU_LAUNCH_PARAM_END; i += 2) {
            if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
                arg_buf = extra[i + 1];
            } else if (extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE) {
                arg_buf_size = *reinterpret_cast<size_t*>(extra[i + 1]);
            }
        }
    }

    if (!arg_buf && kernelParams && !ke->params.empty()) {
        packed = vgpu::packArgs(ke->params, kernelParams);
        arg_buf      = packed.data();
        arg_buf_size = packed.size();
    }

    // Build RPC request
    vgpu::RpcCuLaunchKernelReq req{};
    req.func_id          = ke->func_id;
    req.grid_x           = gridX; req.grid_y = gridY; req.grid_z = gridZ;
    req.block_x          = blockX; req.block_y = blockY; req.block_z = blockZ;
    req.shared_mem_bytes = sharedMem;
    req.arg_buf_size     = static_cast<std::uint32_t>(arg_buf_size);
    req.stream           = reinterpret_cast<std::uint64_t>(stream);

    vgpu::RpcResult r = vgpu::drvCallDrv(
        vgpu::RpcDrvOp::kCuLaunchKernel,
        &req, sizeof(req),
        arg_buf, arg_buf_size);

    return r.status == cudaSuccess ? CUDA_SUCCESS : static_cast<CUresult>(r.status);
}

// ── Memory ───────────────────────────────────────────────────────────────────

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    vgpu::RpcMallocReq req{};
    req.size = static_cast<std::uint64_t>(bytesize);
    vgpu::RpcResult r = vgpu::drvCallDrv(vgpu::RpcDrvOp::kCuMemAlloc,
                                          &req, sizeof(req));
    if (r.status == cudaSuccess && dptr)
        *dptr = static_cast<CUdeviceptr>(r.aux_u64);
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    vgpu::RpcFreeReq req{};
    req.dev_ptr = static_cast<std::uint64_t>(dptr);
    vgpu::RpcResult r = vgpu::drvCallDrv(vgpu::RpcDrvOp::kCuMemFree,
                                          &req, sizeof(req));
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void* src, size_t count) {
    vgpu::RpcMemcpyReq req{};
    req.dst  = static_cast<std::uint64_t>(dst);
    req.src  = 0;
    req.count = static_cast<std::uint64_t>(count);
    req.kind  = static_cast<std::int32_t>(cudaMemcpyHostToDevice);
    vgpu::RpcResult r = vgpu::drvCallDrv(vgpu::RpcDrvOp::kCuMemcpyHtoD,
                                          &req, sizeof(req), src, count);
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult cuMemcpyDtoH_v2(void* dst, CUdeviceptr src, size_t count) {
    vgpu::RpcMemcpyReq req{};
    req.dst  = 0;
    req.src  = static_cast<std::uint64_t>(src);
    req.count = static_cast<std::uint64_t>(count);
    req.kind  = static_cast<std::int32_t>(cudaMemcpyDeviceToHost);
    vgpu::RpcResult r = vgpu::drvCallDrv(vgpu::RpcDrvOp::kCuMemcpyDtoH,
                                          &req, sizeof(req));
    if (r.status == cudaSuccess && dst) {
        if (r.payload.size() < count) return CUDA_ERROR_UNKNOWN;
        std::memcpy(dst, r.payload.data(), count);
    }
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t count) {
    vgpu::RpcMemcpyReq req{};
    req.dst  = static_cast<std::uint64_t>(dst);
    req.src  = static_cast<std::uint64_t>(src);
    req.count = static_cast<std::uint64_t>(count);
    req.kind  = static_cast<std::int32_t>(cudaMemcpyDeviceToDevice);
    vgpu::RpcResult r = vgpu::drvCallDrv(vgpu::RpcDrvOp::kCuMemcpyDtoD,
                                          &req, sizeof(req));
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

// Async variants – for simplicity forward synchronously
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void* src,
                               size_t count, CUstream_drv /*stream*/) {
    return cuMemcpyHtoD_v2(dst, src, count);
}
CUresult cuMemcpyDtoHAsync_v2(void* dst, CUdeviceptr src,
                               size_t count, CUstream_drv /*stream*/) {
    return cuMemcpyDtoH_v2(dst, src, count);
}
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src,
                               size_t count, CUstream_drv /*stream*/) {
    return cuMemcpyDtoD_v2(dst, src, count);
}

CUresult cuMemsetD8_v2(CUdeviceptr dst, unsigned char uc, size_t n) {
    vgpu::RpcMemsetReq req{};
    req.dst   = static_cast<std::uint64_t>(dst);
    req.value = static_cast<std::int32_t>(uc);
    req.count = static_cast<std::uint64_t>(n);
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaMemset, &req, sizeof(req));
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
CUresult cuMemsetD32_v2(CUdeviceptr dst, unsigned int ui, size_t n) {
    return cuMemsetD8_v2(dst, static_cast<unsigned char>(ui & 0xFF), n * 4);
}

// ── Stream / sync stubs ───────────────────────────────────────────────────────

CUresult cuStreamCreate(CUstream_drv* phStream, unsigned int /*flags*/) {
    cudaStream_t s = nullptr;
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaStreamCreate, nullptr, 0);
    if (r.status == cudaSuccess) {
        s = reinterpret_cast<cudaStream_t>(r.aux_u64);
    }
    if (phStream) *phStream = reinterpret_cast<CUstream_drv>(s);
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult cuStreamDestroy_v2(CUstream_drv hStream) {
    vgpu::RpcStreamDestroyReq req{};
    req.stream = reinterpret_cast<std::uint64_t>(hStream);
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaStreamDestroy, &req, sizeof(req));
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult cuStreamSynchronize(CUstream_drv hStream) {
    vgpu::RpcStreamSyncReq req{};
    req.stream = reinterpret_cast<std::uint64_t>(hStream);
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaStreamSynchronize, &req, sizeof(req));
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

// ── Error helpers ─────────────────────────────────────────────────────────────

const char* cuGetErrorName(CUresult error, const char** pStr) {
    static const char* unknown = "CUDA_ERROR_UNKNOWN";
    if (pStr) *pStr = (error == CUDA_SUCCESS) ? "CUDA_SUCCESS" : unknown;
    return (error == CUDA_SUCCESS) ? "CUDA_SUCCESS" : unknown;
}

CUresult cuGetErrorString(CUresult error, const char** pStr) {
    cuGetErrorName(error, pStr);
    return CUDA_SUCCESS;
}

}  // extern "C"
