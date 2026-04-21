#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>

#include "vgpu/context_registry.h"
#include "vgpu/cuda_abi.h"
#include "vgpu/protocol.h"
#include "vgpu/rpc_client.h"
#include "vgpu/fatbin_parser.h"
#include "vgpu/kernel_registry.h"

namespace vgpu {
namespace {

thread_local int g_current_device = -1;
thread_local cudaError_t g_last_error = cudaSuccess;

struct Meta {
    std::uint64_t app_id;
    std::uint64_t context_id;
    int device;
};

RpcClient& client() {
    static RpcClient c;
    return c;
}

ContextRegistry& contexts() {
    static ContextRegistry c;
    return c;
}

Meta buildMeta() {
    Meta m{};
    m.app_id = static_cast<std::uint64_t>(getpid());
    m.device = g_current_device;
    m.context_id = contexts().acquireContextId(m.device);
    return m;
}

cudaError_t unsupported() {
    static constexpr cudaError_t kCudaErrorNotSupported = 801;
    return kCudaErrorNotSupported;
}

cudaError_t remember(cudaError_t st) {
    g_last_error = st;
    return st;
}

RpcResult callServer(
    RpcOp op,
    const void* payload,
    std::size_t payload_size,
    const void* extra_payload,
    std::size_t extra_payload_size) {
    Meta m = buildMeta();
    return client().call(op, m.app_id, m.context_id, m.device, payload, payload_size, extra_payload, extra_payload_size);
}

cudaError_t copyBackIfNeeded(void* dst, std::size_t count, const RpcResult& r) {
    if (r.status != cudaSuccess) {
        return r.status;
    }
    if (r.payload.size() < count) {
        return cudaErrorUnknown;
    }
    std::memcpy(dst, r.payload.data(), count);
    return r.status;
}

cudaError_t doMemcpy(
    RpcOp op,
    void* dst,
    const void* src,
    std::size_t count,
    cudaMemcpyKind kind,
    cudaStream_t stream) {
    if (op == RpcOp::kCudaMemcpy) {
        RpcMemcpyReq req{};
        req.dst = reinterpret_cast<std::uint64_t>(dst);
        req.src = reinterpret_cast<std::uint64_t>(src);
        req.count = count;
        req.kind = static_cast<std::int32_t>(kind);

        if (kind == cudaMemcpyHostToDevice) {
            RpcResult r = callServer(op, &req, sizeof(req), src, count);
            return r.status;
        }
        if (kind == cudaMemcpyDeviceToHost) {
            RpcResult r = callServer(op, &req, sizeof(req), nullptr, 0);
            return copyBackIfNeeded(dst, count, r);
        }
        if (kind == cudaMemcpyDeviceToDevice) {
            RpcResult r = callServer(op, &req, sizeof(req), nullptr, 0);
            return r.status;
        }
        return unsupported();
    }

    RpcMemcpyAsyncReq req{};
    req.dst = reinterpret_cast<std::uint64_t>(dst);
    req.src = reinterpret_cast<std::uint64_t>(src);
    req.count = count;
    req.kind = static_cast<std::int32_t>(kind);
    req.stream = reinterpret_cast<std::uint64_t>(stream);

    if (kind == cudaMemcpyHostToDevice) {
        RpcResult r = callServer(op, &req, sizeof(req), src, count);
        return r.status;
    }
    if (kind == cudaMemcpyDeviceToHost) {
        RpcResult r = callServer(op, &req, sizeof(req), nullptr, 0);
        return copyBackIfNeeded(dst, count, r);
    }
    if (kind == cudaMemcpyDeviceToDevice) {
        RpcResult r = callServer(op, &req, sizeof(req), nullptr, 0);
        return r.status;
    }
    return unsupported();
}

}  // namespace
}  // namespace vgpu

extern "C" cudaError_t cudaMalloc(void** devPtr, std::size_t size) {
    vgpu::RpcMallocReq req{};
    req.size = size;
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaMalloc, &req, sizeof(req), nullptr, 0);
    if (r.status == cudaSuccess && devPtr != nullptr) {
        *devPtr = reinterpret_cast<void*>(r.aux_u64);
    }
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaFree(void* devPtr) {
    vgpu::RpcFreeReq req{};
    req.dev_ptr = reinterpret_cast<std::uint64_t>(devPtr);
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaFree, &req, sizeof(req), nullptr, 0);
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind) {
    return vgpu::remember(vgpu::doMemcpy(vgpu::RpcOp::kCudaMemcpy, dst, src, count, kind, nullptr));
}

extern "C" cudaError_t cudaMemcpyAsync(
    void* dst,
    const void* src,
    std::size_t count,
    cudaMemcpyKind kind,
    cudaStream_t stream) {
    return vgpu::remember(vgpu::doMemcpy(vgpu::RpcOp::kCudaMemcpyAsync, dst, src, count, kind, stream));
}

extern "C" cudaError_t cudaSetDevice(int device) {
    vgpu::RpcSetDeviceReq req{};
    req.device = device;
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaSetDevice, &req, sizeof(req), nullptr, 0);
    if (r.status == cudaSuccess) {
        vgpu::g_current_device = device;
    }
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaSetDeviceFlags(unsigned int flags) {
    vgpu::RpcSetDeviceFlagsReq req{};
    req.flags = flags;
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaSetDeviceFlags, &req, sizeof(req), nullptr, 0);
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaGetDeviceFlags(unsigned int* flags) {
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaGetDeviceFlags, nullptr, 0, nullptr, 0);
    if (r.status == cudaSuccess && flags != nullptr) {
        *flags = static_cast<unsigned int>(r.aux_u64);
    }
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaGetDevice(int* device) {
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaGetDevice, nullptr, 0, nullptr, 0);
    if (r.status == cudaSuccess && device != nullptr) {
        *device = static_cast<int>(r.aux_u64);
        vgpu::g_current_device = *device;
    }
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaGetDeviceCount(int* count) {
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaGetDeviceCount, nullptr, 0, nullptr, 0);
    if (r.status == cudaSuccess && count != nullptr) {
        *count = static_cast<int>(r.aux_u64);
    }
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaDriverGetVersion(int* driverVersion) {
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaDriverGetVersion, nullptr, 0, nullptr, 0);
    if (r.status == cudaSuccess && driverVersion != nullptr) {
        *driverVersion = static_cast<int>(r.aux_u64);
    }
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) {
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaRuntimeGetVersion, nullptr, 0, nullptr, 0);
    if (r.status == cudaSuccess && runtimeVersion != nullptr) {
        *runtimeVersion = static_cast<int>(r.aux_u64);
    }
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaDeviceGetAttribute(int* value, int attr, int device) {
    vgpu::RpcDeviceGetAttributeReq req{};
    req.attr = attr;
    req.device = device;
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaDeviceGetAttribute, &req, sizeof(req), nullptr, 0);
    if (r.status == cudaSuccess && value != nullptr) {
        *value = static_cast<int>(r.aux_u64);
    }
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes) {
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaMemGetInfo, nullptr, 0, nullptr, 0);
    if (r.status == cudaSuccess) {
        if (r.payload.size() < sizeof(vgpu::RpcMemGetInfoRsp)) {
            return cudaErrorUnknown;
        }
        const auto* info = reinterpret_cast<const vgpu::RpcMemGetInfoRsp*>(r.payload.data());
        if (free_bytes != nullptr) {
            *free_bytes = static_cast<std::size_t>(info->free_bytes);
        }
        if (total_bytes != nullptr) {
            *total_bytes = static_cast<std::size_t>(info->total_bytes);
        }
    }
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaDeviceReset(void) {
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaDeviceReset, nullptr, 0, nullptr, 0);
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaDeviceSynchronize(void) {
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaDeviceSynchronize, nullptr, 0, nullptr, 0);
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaStreamCreate, nullptr, 0, nullptr, 0);
    if (r.status == cudaSuccess && pStream != nullptr) {
        *pStream = reinterpret_cast<cudaStream_t>(r.aux_u64);
    }
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    vgpu::RpcStreamDestroyReq req{};
    req.stream = reinterpret_cast<std::uint64_t>(stream);
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaStreamDestroy, &req, sizeof(req), nullptr, 0);
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    vgpu::RpcStreamSyncReq req{};
    req.stream = reinterpret_cast<std::uint64_t>(stream);
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaStreamSynchronize, &req, sizeof(req), nullptr, 0);
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaStreamQuery(cudaStream_t stream) {
    vgpu::RpcStreamQueryReq req{};
    req.stream = reinterpret_cast<std::uint64_t>(stream);
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaStreamQuery, &req, sizeof(req), nullptr, 0);
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    vgpu::RpcStreamWaitEventReq req{};
    req.stream = reinterpret_cast<std::uint64_t>(stream);
    req.event = reinterpret_cast<std::uint64_t>(event);
    req.flags = flags;
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaStreamWaitEvent, &req, sizeof(req), nullptr, 0);
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaEventCreate(cudaEvent_t* event) {
    vgpu::RpcEventCreateReq req{};
    req.flags = cudaEventDefault;
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaEventCreate, &req, sizeof(req), nullptr, 0);
    if (r.status == cudaSuccess && event != nullptr) {
        *event = reinterpret_cast<cudaEvent_t>(r.aux_u64);
    }
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    vgpu::RpcEventCreateReq req{};
    req.flags = flags;
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaEventCreate, &req, sizeof(req), nullptr, 0);
    if (r.status == cudaSuccess && event != nullptr) {
        *event = reinterpret_cast<cudaEvent_t>(r.aux_u64);
    }
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaEventDestroy(cudaEvent_t event) {
    vgpu::RpcEventDestroyReq req{};
    req.event = reinterpret_cast<std::uint64_t>(event);
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaEventDestroy, &req, sizeof(req), nullptr, 0);
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    vgpu::RpcEventRecordReq req{};
    req.event = reinterpret_cast<std::uint64_t>(event);
    req.stream = reinterpret_cast<std::uint64_t>(stream);
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaEventRecord, &req, sizeof(req), nullptr, 0);
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    vgpu::RpcEventSynchronizeReq req{};
    req.event = reinterpret_cast<std::uint64_t>(event);
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaEventSynchronize, &req, sizeof(req), nullptr, 0);
    return r.status;
}

extern "C" cudaError_t cudaEventQuery(cudaEvent_t event) {
    vgpu::RpcEventQueryReq req{};
    req.event = reinterpret_cast<std::uint64_t>(event);
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaEventQuery, &req, sizeof(req), nullptr, 0);
    return r.status;
}

extern "C" cudaError_t cudaMemset(void* dst, int value, std::size_t count) {
    vgpu::RpcMemsetReq req{};
    req.dst = reinterpret_cast<std::uint64_t>(dst);
    req.value = value;
    req.count = count;
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaMemset, &req, sizeof(req), nullptr, 0);
    return r.status;
}

extern "C" cudaError_t cudaMemsetAsync(void* dst, int value, std::size_t count, cudaStream_t stream) {
    vgpu::RpcMemsetAsyncReq req{};
    req.dst = reinterpret_cast<std::uint64_t>(dst);
    req.value = value;
    req.count = count;
    req.stream = reinterpret_cast<std::uint64_t>(stream);
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaMemsetAsync, &req, sizeof(req), nullptr, 0);
    return r.status;
}

extern "C" cudaError_t cudaLaunchKernel(
    const void* func,
    dim3 gridDim,
    dim3 blockDim,
    void** args,
    std::size_t sharedMem,
    cudaStream_t stream) {
    const vgpu::KernelEntry* ke = vgpu::globalKernelRegistry().findKernel(func);
    if (!ke) {
        return vgpu::remember(vgpu::unsupported());
    }

    // Pack void** args into a flat buffer using stored param info
    std::vector<std::uint8_t> arg_buf;
    if (!ke->params.empty() && args) {
        arg_buf = vgpu::packArgs(ke->params, args);
    }

    vgpu::RpcCuLaunchKernelReq req{};
    req.func_id          = ke->func_id;
    req.grid_x           = gridDim.x; req.grid_y = gridDim.y; req.grid_z = gridDim.z;
    req.block_x          = blockDim.x; req.block_y = blockDim.y; req.block_z = blockDim.z;
    req.shared_mem_bytes = static_cast<std::uint32_t>(sharedMem);
    req.arg_buf_size     = static_cast<std::uint32_t>(arg_buf.size());
    req.stream           = reinterpret_cast<std::uint64_t>(stream);

    vgpu::RpcResult r = vgpu::callServer(
        static_cast<vgpu::RpcOp>(vgpu::RpcDrvOp::kCuLaunchKernel),
        &req, sizeof(req),
        arg_buf.data(), arg_buf.size());
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaGetLastError(void) {
    cudaError_t st = vgpu::g_last_error;
    vgpu::g_last_error = cudaSuccess;
    return st;
}

extern "C" cudaError_t cudaPeekAtLastError(void) {
    return vgpu::g_last_error;
}

extern "C" const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return "cudaSuccess";
        case 34:
            return "cudaErrorStubLibrary";
        case 35:
            return "cudaErrorInsufficientDriver";
        case 801:
            return "cudaErrorNotSupported";
        case cudaErrorUnknown:
            return "cudaErrorUnknown";
        default:
            return "cudaError";
    }
}

extern "C" const char* cudaGetErrorName(cudaError_t error) {
    return cudaGetErrorString(error);
}

extern "C" void** __cudaRegisterFatBinary(void* fatCubin) {
    // Allocate a persistent handle (freed when module is unregistered)
    void** handle = new void*(nullptr);

    // Determine raw fatbin pointer and size
    static constexpr std::uint32_t kWrapMagic = 0x466243B1u;
    static constexpr std::uint32_t kFatMagic  = 0xBA55ED50u;
    const std::uint8_t* p = static_cast<const std::uint8_t*>(fatCubin);
    std::uint32_t magic = 0;
    std::memcpy(&magic, p, sizeof(magic));
    const void* raw_img = fatCubin;
    if (magic == kWrapMagic) {
        const void* data = nullptr;
        std::memcpy(&data, p + 8, sizeof(data));
        raw_img = data;
        if (raw_img) {
            std::memcpy(&magic, raw_img, sizeof(magic));
        }
    }
    std::size_t raw_size = 0;
    if (magic == kFatMagic && raw_img) {
        std::uint32_t hdr_sz = 0, data_sz = 0;
        std::memcpy(&hdr_sz,  static_cast<const std::uint8_t*>(raw_img) + 8,  sizeof(hdr_sz));
        std::memcpy(&data_sz, static_cast<const std::uint8_t*>(raw_img) + 12, sizeof(data_sz));
        raw_size = static_cast<std::size_t>(hdr_sz) + static_cast<std::size_t>(data_sz);
    }

    // Parse PTX to extract per-kernel param info (stored by mangled name)
    if (raw_img && raw_size > 0) {
        auto ki_list = vgpu::parseFatbin(raw_img, raw_size);
        for (const auto& ki : ki_list) {
            vgpu::globalKernelRegistry().addParamInfo(ki.mangled_name, ki.params);
        }
    }

    // Upload fatbin to server; get back a module_id
    if (raw_img && raw_size > 0) {
        vgpu::RpcResult r = vgpu::callServer(
            static_cast<vgpu::RpcOp>(vgpu::RpcDrvOp::kCuModuleLoadData),
            nullptr, 0, raw_img, raw_size);
        if (r.status == cudaSuccess) {
            vgpu::globalKernelRegistry().addModule(handle, r.aux_u64);
        }
    }

    *handle = fatCubin; // keep original pointer for reference
    return handle;
}

extern "C" void __cudaRegisterFatBinaryEnd(void** fatCubinHandle) {
    (void)fatCubinHandle;
}

extern "C" void __cudaUnregisterFatBinary(void** fatCubinHandle) {
    if (fatCubinHandle) {
        std::uint64_t mod_id = vgpu::globalKernelRegistry().getModuleId(fatCubinHandle);
        if (mod_id != 0) {
            vgpu::callServer(static_cast<vgpu::RpcOp>(vgpu::RpcDrvOp::kCuModuleUnload),
                             &mod_id, sizeof(mod_id), nullptr, 0);
        }
        delete fatCubinHandle;
    }
}

extern "C" void __cudaRegisterFunction(
    void** fatCubinHandle,
    const char* hostFun,
    char* deviceFun,
    const char* deviceName,
    int thread_limit,
    void* tid,
    void* bid,
    void* bDim,
    void* gDim,
    int* wSize) {
    (void)thread_limit; (void)tid; (void)bid; (void)bDim; (void)gDim; (void)wSize;

    std::uint64_t module_id = vgpu::globalKernelRegistry().getModuleId(fatCubinHandle);
    if (module_id == 0) return;  // module upload failed earlier

    // Use the mangled device function name for the lookup
    const char* lookup_name = (deviceFun && deviceFun[0]) ? deviceFun : deviceName;
    if (!lookup_name) return;

    // Try param info from local PTX parse cache first
    const std::vector<vgpu::ParamInfo>* cached_params =
        vgpu::globalKernelRegistry().findParamInfo(lookup_name);

    // Ask server: kCuModuleGetFunction
    std::size_t name_len = std::strlen(lookup_name) + 1;
    vgpu::RpcCuModuleGetFunctionReq req{};
    req.module_id = module_id;
    req.name_len  = static_cast<std::uint32_t>(name_len);

    std::vector<std::uint8_t> payload(sizeof(req) + name_len);
    std::memcpy(payload.data(), &req, sizeof(req));
    std::memcpy(payload.data() + sizeof(req), lookup_name, name_len);

    vgpu::RpcResult r = vgpu::callServer(
        static_cast<vgpu::RpcOp>(vgpu::RpcDrvOp::kCuModuleGetFunction),
        payload.data(), payload.size(), nullptr, 0);
    if (r.status != cudaSuccess) return;

    vgpu::KernelEntry ke{};
    ke.func_id = r.aux_u64;

    // Decode param info from server response
    if (r.payload.size() >= sizeof(vgpu::RpcCuModuleGetFunctionRsp)) {
        const auto* rsp =
            reinterpret_cast<const vgpu::RpcCuModuleGetFunctionRsp*>(r.payload.data());
        ke.total_param_bytes = rsp->total_param_bytes;
        std::uint32_t pc = rsp->param_count;
        const vgpu::RpcParamInfo* pi =
            reinterpret_cast<const vgpu::RpcParamInfo*>(rsp + 1);
        if (r.payload.size() >=
            sizeof(vgpu::RpcCuModuleGetFunctionRsp) + pc * sizeof(*pi)) {
            for (std::uint32_t i = 0; i < pc; ++i) {
                vgpu::ParamInfo p{};
                p.size      = pi[i].size;
                p.alignment = pi[i].alignment;
                ke.params.push_back(p);
            }
        }
    }

    // Fall back to local PTX cache if server didn't send param info
    if (ke.params.empty() && cached_params) {
        ke.params            = *cached_params;
        ke.total_param_bytes = vgpu::computeParamBufSize(ke.params);
    }

    vgpu::globalKernelRegistry().addKernel(hostFun, std::move(ke));
}

extern "C" cudaError_t __cudaPushCallConfiguration(
    dim3 gridDim,
    dim3 blockDim,
    std::size_t sharedMem,
    cudaStream_t stream) {
    (void)gridDim;
    (void)blockDim;
    (void)sharedMem;
    (void)stream;
    return cudaSuccess;
}

extern "C" cudaError_t __cudaPopCallConfiguration(
    dim3* gridDim,
    dim3* blockDim,
    std::size_t* sharedMem,
    cudaStream_t* stream) {
    if (gridDim != nullptr) {
        *gridDim = dim3{};
    }
    if (blockDim != nullptr) {
        *blockDim = dim3{};
    }
    if (sharedMem != nullptr) {
        *sharedMem = 0;
    }
    if (stream != nullptr) {
        *stream = nullptr;
    }
    return cudaSuccess;
}
