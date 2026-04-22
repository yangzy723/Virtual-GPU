#include <sys/types.h>
#include <unistd.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>

#include "vgpu/common/context_registry.h"
#include "vgpu/common/cuda_abi.h"
#include "vgpu/frontend/shim_utils.h"
#include "vgpu/common/protocol.h"
#include "vgpu/common/rpc_client.h"
#include "vgpu/common/fatbin_parser.h"
#include "vgpu/common/kernel_registry.h"

namespace vgpu {
namespace {

thread_local int g_current_device = -1;
thread_local cudaError_t g_last_error = cudaSuccess;

struct DevicePropCompat {
    char name[256];
    unsigned char uuid[16];
    char luid[8];
    unsigned int luidDeviceNodeMask;
    std::size_t totalGlobalMem;
    std::size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    std::size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    std::size_t totalConstMem;
    int major;
    int minor;
    std::size_t textureAlignment;
    std::size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DMipmap;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int maxTexture2DMipmap[2];
    int maxTexture2DLinear[3];
    int maxTexture2DGather[2];
    int maxTexture3D[3];
    int maxTexture3DAlt[3];
    int maxTextureCubemap;
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    int maxTextureCubemapLayered[2];
    int maxSurface1D;
    int maxSurface2D[2];
    int maxSurface3D[3];
    int maxSurface1DLayered[2];
    int maxSurface2DLayered[3];
    int maxSurfaceCubemap;
    int maxSurfaceCubemapLayered[2];
    std::size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int persistingL2CacheMaxSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    std::size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int hostNativeAtomicSupported;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;
    int canUseHostPointerForRegisteredMem;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    std::size_t sharedMemPerBlockOptin;
    int pageableMemoryAccessUsesHostPageTables;
    int directManagedMemAccessFromHost;
    int maxBlocksPerMultiProcessor;
    int accessPolicyMaxWindowSize;
    std::size_t reservedSharedMemPerBlock;
    int hostRegisterSupported;
    int sparseCudaArraySupported;
    int hostRegisterReadOnlySupported;
    int timelineSemaphoreInteropSupported;
    int memoryPoolsSupported;
    int gpuDirectRDMASupported;
    unsigned int gpuDirectRDMAFlushWritesOptions;
    int gpuDirectRDMAWritesOrdering;
    unsigned int memoryPoolSupportedHandleTypes;
    int deferredMappingCudaArraySupported;
    int ipcEventSupported;
    int clusterLaunch;
    int unifiedFunctionPointers;
    int reserved[63];
};

struct Meta {
    std::uint64_t app_id;
    std::uint64_t context_id;
    int device;
};

struct PendingLaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t shared_mem;
    cudaStream_t stream;
    bool valid;
};

struct FatbinImage {
    const void* raw = nullptr;
    std::size_t size = 0;
};

RpcClient& client() {
    static RpcClient c;
    return c;
}

ContextRegistry& contexts() {
    static ContextRegistry c;
    return c;
}

thread_local PendingLaunchConfig g_pending_launch_cfg{dim3{}, dim3{}, 0, nullptr, false};

Meta buildMeta() {
    Meta m{};
    m.app_id = static_cast<std::uint64_t>(getpid());
    m.device = g_current_device;
    m.context_id = contexts().acquireContextId(m.device);
    return m;
}

cudaError_t remember(cudaError_t st) {
    g_last_error = st;
    return st;
}

FatbinImage resolveFatbinImage(const void* fat_cubin) {
    static constexpr std::uint32_t kWrapMagic = 0x466243B1u;
    static constexpr std::uint32_t kFatMagic = 0xBA55ED50u;

    auto read32 = [](const void* ptr, std::size_t off, std::uint32_t* out) -> bool {
        if (ptr == nullptr || out == nullptr) {
            return false;
        }
        std::memcpy(out, static_cast<const std::uint8_t*>(ptr) + off, sizeof(*out));
        return true;
    };

    auto read64 = [](const void* ptr, std::size_t off, std::uint64_t* out) -> bool {
        if (ptr == nullptr || out == nullptr) {
            return false;
        }
        std::memcpy(out, static_cast<const std::uint8_t*>(ptr) + off, sizeof(*out));
        return true;
    };

    auto inferSize = [&](const void* raw) -> std::size_t {
        if (raw == nullptr) {
            return 0;
        }

        std::uint32_t magic = 0;
        read32(raw, 0, &magic);
        if (magic != kFatMagic) {
            return 0;
        }

        // CUDA 12+ fatbin header variant:
        //   u32 magic, u32 version, u64 data_size, u32 unknown, u32 header_size
        // Total image bytes are header_size + data_size.
        std::uint32_t version = 0;
        std::uint64_t data64 = 0;
        std::uint32_t header_v2 = 0;
        read32(raw, 4, &version);
        read64(raw, 8, &data64);
        read32(raw, 20, &header_v2);

        auto sane = [](std::size_t total, std::size_t header) -> bool {
            static constexpr std::size_t kMinHeader = 16;
            static constexpr std::size_t kMaxHeader = 1u << 20;   // 1 MiB
            static constexpr std::size_t kMaxImage = 1ull << 33;  // 8 GiB safety cap
            return header >= kMinHeader && header <= kMaxHeader && total >= header && total <= kMaxImage;
        };

        if (version == 0x00100001u) {
            const std::size_t s_v2 = static_cast<std::size_t>(data64 + static_cast<std::uint64_t>(header_v2));
            if (sane(s_v2, static_cast<std::size_t>(header_v2))) {
                return s_v2;
            }
        }

        // Legacy layout: u32 header_size @8, u32 data_size @12.
        std::uint32_t h32 = 0;
        std::uint32_t d32 = 0;
        read32(raw, 8, &h32);
        read32(raw, 12, &d32);
        std::size_t s32 = static_cast<std::size_t>(h32) + static_cast<std::size_t>(d32);

        if (sane(s32, static_cast<std::size_t>(h32))) {
            return s32;
        }

        // Some toolchains store 64-bit header/data sizes.
        std::uint64_t h64 = 0;
        std::uint64_t d64 = 0;
        read64(raw, 8, &h64);
        read64(raw, 16, &d64);
        std::size_t s64 = static_cast<std::size_t>(h64 + d64);
        if (sane(s64, static_cast<std::size_t>(h64))) {
            return s64;
        }

        return 0;
    };

    auto looksLikeFatRaw = [&](const void* ptr) -> bool {
        if (ptr == nullptr) {
            return false;
        }
        std::uint32_t magic = 0;
        read32(ptr, 0, &magic);
        return magic == kFatMagic;
    };

    if (fat_cubin == nullptr) {
        return {};
    }

    const std::uint8_t* p = static_cast<const std::uint8_t*>(fat_cubin);
    std::uint32_t magic = 0;
    std::memcpy(&magic, p, sizeof(magic));

    const void* raw = fat_cubin;
    if (magic == kWrapMagic) {
        // Wrapper layouts vary by toolchain/version. Probe common pointer slots.
        const std::size_t candidate_offsets[] = {8, 16, 24, 32};
        raw = nullptr;
        for (std::size_t off : candidate_offsets) {
            const void* data = nullptr;
            std::memcpy(&data, p + off, sizeof(data));
            if (looksLikeFatRaw(data)) {
                raw = data;
                break;
            }
        }
        if (raw == nullptr) {
            return {};
        }
        std::memcpy(&magic, raw, sizeof(magic));
    }

    if (magic != kFatMagic) {
        return {};
    }

    FatbinImage out{};
    out.raw = raw;
    out.size = inferSize(raw);
    if (out.size == 0) {
        return {};
    }
    return out;
}

void registerKnownModuleAliases(void** fat_cubin_handle, const void* alias, std::uint64_t module_id) {
    if (module_id == 0) {
        return;
    }

    if (fat_cubin_handle != nullptr) {
        globalKernelRegistry().addModule(fat_cubin_handle, module_id);
        if (*fat_cubin_handle != nullptr) {
            globalKernelRegistry().addModule(*fat_cubin_handle, module_id);
        }
    }

    if (alias != nullptr) {
        globalKernelRegistry().addModule(const_cast<void*>(alias), module_id);
    }
}

void cacheParamInfoForModule(std::uint64_t module_id, const FatbinImage& img) {
    if (module_id == 0 || img.raw == nullptr || img.size == 0) {
        return;
    }

    auto ki_list = parseFatbin(img.raw, img.size);
    for (const auto& ki : ki_list) {
        globalKernelRegistry().addParamInfo(module_id, ki.mangled_name, ki.params);
    }
}

RpcResult callServer(
    RpcOp op,
    const void* payload,
    std::size_t payload_size,
    const void* extra_payload,
    std::size_t extra_payload_size);

bool queryDeviceAttributeFromServer(int attr, int device, int* out) {
    if (out == nullptr) {
        return false;
    }

    RpcDeviceGetAttributeReq req{};
    req.attr = attr;
    req.device = device;
    RpcResult r = callServer(RpcOp::kCudaDeviceGetAttribute, &req, sizeof(req), nullptr, 0);
    if (r.status != cudaSuccess) {
        return false;
    }

    *out = static_cast<int>(r.aux_u64);
    return true;
}

bool queryTotalGlobalMemFromServer(std::size_t* total_bytes) {
    if (total_bytes == nullptr) {
        return false;
    }

    RpcResult r = callServer(RpcOp::kCudaMemGetInfo, nullptr, 0, nullptr, 0);
    if (r.status != cudaSuccess || r.payload.size() < sizeof(RpcMemGetInfoRsp)) {
        return false;
    }

    const auto* info = reinterpret_cast<const RpcMemGetInfoRsp*>(r.payload.data());
    *total_bytes = static_cast<std::size_t>(info->total_bytes);
    return true;
}

std::vector<std::uint8_t> makePackedArgsFromTotalBytes(void** args, std::uint32_t total_param_bytes) {
    std::vector<std::uint8_t> out;
    if (args == nullptr || total_param_bytes == 0) {
        return out;
    }

    out.assign(total_param_bytes, 0);

    // Heuristic fallback when PTX-derived per-argument metadata is unavailable.
    const std::size_t slot = sizeof(std::uint64_t);
    std::size_t max_args = (static_cast<std::size_t>(total_param_bytes) + slot - 1) / slot;
    if (max_args > 64) {
        max_args = 64;
    }

    for (std::size_t i = 0; i < max_args; ++i) {
        void* arg_ptr = args[i];
        if (arg_ptr == nullptr && i > 0) {
            break;
        }
        if (arg_ptr == nullptr) {
            continue;
        }

        const std::size_t off = i * slot;
        if (off >= out.size()) {
            break;
        }
        const std::size_t n = std::min(slot, out.size() - off);
        std::memcpy(out.data() + off, arg_ptr, n);
    }

    return out;
}

std::vector<std::uint8_t> makeArgumentSlotsUnknown(void** args, std::size_t max_slots = 12) {
    std::vector<std::uint8_t> out;
    if (args == nullptr || max_slots == 0) {
        return out;
    }

    const std::size_t slot = sizeof(std::uint64_t);
    out.assign(max_slots * slot, 0);

    std::size_t used = 0;
    for (std::size_t i = 0; i < max_slots; ++i) {
        void* arg_ptr = args[i];
        if (arg_ptr == nullptr && i > 0) {
            break;
        }
        if (arg_ptr == nullptr) {
            used = i + 1;
            continue;
        }
        if (!processRangeHasAccess(reinterpret_cast<std::uintptr_t>(arg_ptr), slot, ProcMapAccess::Read)) {
            break;
        }
        std::memcpy(out.data() + i * slot, arg_ptr, slot);
        used = i + 1;
    }

    out.resize(used * slot);
    return out;
}

int queryAttrOrDefault(int attr, int device, int fallback) {
    int value = 0;
    if (queryDeviceAttributeFromServer(attr, device, &value)) {
        return value;
    }
    return fallback;
}

std::size_t querySizeAttrOrDefault(int attr, int device, std::size_t fallback) {
    int value = 0;
    if (!queryDeviceAttributeFromServer(attr, device, &value) || value < 0) {
        return fallback;
    }
    return static_cast<std::size_t>(value);
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
    if (count > 0 && dst == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (r.payload.size() < count) {
        return cudaErrorUnknown;
    }
    std::memcpy(dst, r.payload.data(), count);
    return r.status;
}

template <typename Req>
cudaError_t doMemcpyRequest(RpcOp op, Req* req, void* dst, const void* src, std::size_t count, cudaMemcpyKind kind) {
    req->dst = reinterpret_cast<std::uint64_t>(dst);
    req->src = reinterpret_cast<std::uint64_t>(src);
    req->count = count;
    req->kind = static_cast<std::int32_t>(kind);

    if (count == 0) {
        return cudaSuccess;
    }

    if (dst == nullptr || src == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (kind == cudaMemcpyHostToDevice) {
        RpcResult r = callServer(op, req, sizeof(*req), src, count);
        return r.status;
    }
    if (kind == cudaMemcpyDeviceToHost) {
        RpcResult r = callServer(op, req, sizeof(*req), nullptr, 0);
        return copyBackIfNeeded(dst, count, r);
    }
    if (kind == cudaMemcpyDeviceToDevice) {
        RpcResult r = callServer(op, req, sizeof(*req), nullptr, 0);
        return r.status;
    }
    return unsupported(op == RpcOp::kCudaMemcpy
        ? "cudaMemcpy(kind=default/unsupported)"
        : "cudaMemcpyAsync(kind=default/unsupported)");
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
        return doMemcpyRequest(op, &req, dst, src, count, kind);
    }

    if (op != RpcOp::kCudaMemcpyAsync) {
        return unsupported("doMemcpy(op unsupported)");
    }

    RpcMemcpyAsyncReq req{};
    req.stream = reinterpret_cast<std::uint64_t>(stream);
    return doMemcpyRequest(op, &req, dst, src, count, kind);
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
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaEventQuery(cudaEvent_t event) {
    vgpu::RpcEventQueryReq req{};
    req.event = reinterpret_cast<std::uint64_t>(event);
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaEventQuery, &req, sizeof(req), nullptr, 0);
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaMemset(void* dst, int value, std::size_t count) {
    vgpu::RpcMemsetReq req{};
    req.dst = reinterpret_cast<std::uint64_t>(dst);
    req.value = value;
    req.count = count;
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaMemset, &req, sizeof(req), nullptr, 0);
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaMemsetAsync(void* dst, int value, std::size_t count, cudaStream_t stream) {
    vgpu::RpcMemsetAsyncReq req{};
    req.dst = reinterpret_cast<std::uint64_t>(dst);
    req.value = value;
    req.count = count;
    req.stream = reinterpret_cast<std::uint64_t>(stream);
    vgpu::RpcResult r = vgpu::callServer(vgpu::RpcOp::kCudaMemsetAsync, &req, sizeof(req), nullptr, 0);
    return vgpu::remember(r.status);
}

extern "C" cudaError_t cudaLaunchKernel(
    const void* func,
    dim3 gridDim,
    dim3 blockDim,
    void** args,
    std::size_t sharedMem,
    cudaStream_t stream) {
    vgpu::KernelEntry ke{};
    bool found = vgpu::globalKernelRegistry().findKernel(func, &ke);
    if (!found) {
        found = vgpu::globalKernelRegistry().findDriverFunc(const_cast<void*>(func), &ke);
    }
    if (!found) {
        if (std::getenv("VGPU_DEBUG") != nullptr) {
            std::fprintf(stderr, "[vGPU][launch] miss func=%p\n", func);
        }
        const char* allow_noop_unregistered = std::getenv("VGPU_NOOP_UNREGISTERED_KERNEL");
        if (allow_noop_unregistered && allow_noop_unregistered[0] == '1') {
            if (std::getenv("VGPU_DEBUG") != nullptr) {
                std::fprintf(stderr, "[vGPU][launch] no-op unregistered func=%p\n", func);
            }
            return vgpu::remember(cudaSuccess);
        }
        return vgpu::remember(vgpu::unsupported("cudaLaunchKernel(unregistered kernel)"));
    }

    // Pack void** args into a flat buffer using stored param info
    std::vector<std::uint8_t> arg_buf;
    if (!ke.params.empty() && args) {
        arg_buf = vgpu::packArgs(ke.params, args);
    } else if (args && ke.total_param_bytes > 0) {
        arg_buf = vgpu::makePackedArgsFromTotalBytes(args, ke.total_param_bytes);
    } else if (args) {
        arg_buf = vgpu::makeArgumentSlotsUnknown(args);
    }

    vgpu::RpcCuLaunchKernelReq req{};
    req.func_id          = ke.func_id;
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

extern "C" cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int /*device*/, int /*peerDevice*/) {
    if (canAccessPeer) {
        *canAccessPeer = 0;
    }
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaDeviceEnablePeerAccess(int /*peerDevice*/, unsigned int /*flags*/) {
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int /*device*/) {
    if (pciBusId && len > 0) {
        std::strncpy(pciBusId, "0000:00:00.0", static_cast<std::size_t>(len - 1));
        pciBusId[len - 1] = '\0';
    }
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    if (leastPriority) {
        *leastPriority = 0;
    }
    if (greatestPriority) {
        *greatestPriority = 0;
    }
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t /*start*/, cudaEvent_t /*end*/) {
    if (ms) {
        *ms = 0.0f;
    }
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int /*flags*/) {
    return cudaEventRecord(event, stream);
}

extern "C" cudaError_t cudaHostAlloc(void** pHost, std::size_t size, unsigned int /*flags*/) {
    if (!pHost) {
        return vgpu::remember(cudaErrorUnknown);
    }
    *pHost = std::malloc(size);
    return vgpu::remember(*pHost ? cudaSuccess : cudaErrorUnknown);
}

extern "C" cudaError_t cudaFreeHost(void* ptr) {
    std::free(ptr);
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaHostRegister(void* /*ptr*/, std::size_t /*size*/, unsigned int /*flags*/) {
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaHostUnregister(void* /*ptr*/) {
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaFuncGetAttributes(void* attr, const void* /*func*/) {
    if (attr) {
        struct FuncAttrPrefix {
            std::size_t sharedSizeBytes;
            std::size_t constSizeBytes;
            std::size_t localSizeBytes;
            int maxThreadsPerBlock;
            int numRegs;
            int ptxVersion;
            int binaryVersion;
            int cacheModeCA;
            int maxDynamicSharedSizeBytes;
            int preferredShmemCarveout;
        };

        FuncAttrPrefix prefix{};
        prefix.sharedSizeBytes = 0;
        prefix.constSizeBytes = static_cast<std::size_t>(64) * 1024;
        prefix.localSizeBytes = 0;
        prefix.maxThreadsPerBlock = 1024;
        prefix.numRegs = 64;
        prefix.ptxVersion = 80;
        prefix.binaryVersion = 90;
        prefix.cacheModeCA = 0;
        prefix.maxDynamicSharedSizeBytes = static_cast<int>(96 * 1024);
        prefix.preferredShmemCarveout = 0;

        std::memset(attr, 0, 128);
        std::memcpy(attr, &prefix, sizeof(prefix));
    }
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaFuncSetAttribute(const void* /*func*/, int /*attr*/, int /*value*/) {
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaGetDeviceProperties_v2(void* prop, int device) {
    if (!prop) {
        return vgpu::remember(cudaErrorUnknown);
    }

    static constexpr int kAttrMaxThreadsPerBlock = 1;
    static constexpr int kAttrMaxBlockDimX = 2;
    static constexpr int kAttrMaxBlockDimY = 3;
    static constexpr int kAttrMaxBlockDimZ = 4;
    static constexpr int kAttrMaxGridDimX = 5;
    static constexpr int kAttrMaxGridDimY = 6;
    static constexpr int kAttrMaxGridDimZ = 7;
    static constexpr int kAttrSharedMemPerBlock = 8;
    static constexpr int kAttrTotalConstantMemory = 9;
    static constexpr int kAttrWarpSize = 10;
    static constexpr int kAttrMemPitch = 11;
    static constexpr int kAttrRegsPerBlock = 12;
    static constexpr int kAttrClockRate = 13;
    static constexpr int kAttrMultiProcessorCount = 16;
    static constexpr int kAttrMaxThreadsPerMultiprocessor = 39;
    static constexpr int kAttrComputeCapabilityMajor = 75;
    static constexpr int kAttrComputeCapabilityMinor = 76;

    const int query_device = (device >= 0) ? device : vgpu::g_current_device;

    vgpu::DevicePropCompat p{};
    std::strncpy(p.name, "vGPU Remote Device", sizeof(p.name) - 1);
    std::memset(p.uuid, 0, sizeof(p.uuid));
    std::memset(p.luid, 0, sizeof(p.luid));
    p.luidDeviceNodeMask = 0;
    p.totalGlobalMem = static_cast<std::size_t>(80) * 1024 * 1024 * 1024ULL;
    p.sharedMemPerBlock = static_cast<std::size_t>(48) * 1024;
    p.regsPerBlock = 65536;
    p.warpSize = 32;
    p.memPitch = static_cast<std::size_t>(1) << 31;
    p.maxThreadsPerBlock = 1024;
    p.maxThreadsDim[0] = 1024;
    p.maxThreadsDim[1] = 1024;
    p.maxThreadsDim[2] = 64;
    p.maxGridSize[0] = 2147483647;
    p.maxGridSize[1] = 65535;
    p.maxGridSize[2] = 65535;
    p.clockRate = 1400000;
    p.totalConstMem = static_cast<std::size_t>(64) * 1024;
    p.major = 9;
    p.minor = 0;
    p.textureAlignment = 512;
    p.texturePitchAlignment = 512;
    p.deviceOverlap = 1;
    p.multiProcessorCount = 80;
    p.kernelExecTimeoutEnabled = 0;
    p.integrated = 0;
    p.canMapHostMemory = 1;
    p.computeMode = 0;
    p.concurrentKernels = 1;
    p.asyncEngineCount = 2;
    p.unifiedAddressing = 1;
    p.maxThreadsPerMultiProcessor = 2048;
    p.streamPrioritiesSupported = 1;
    p.globalL1CacheSupported = 1;
    p.localL1CacheSupported = 1;
    p.sharedMemPerMultiprocessor = static_cast<std::size_t>(100) * 1024;
    p.regsPerMultiprocessor = 65536;
    p.managedMemory = 1;
    p.pageableMemoryAccess = 1;
    p.concurrentManagedAccess = 1;
    p.computePreemptionSupported = 1;
    p.canUseHostPointerForRegisteredMem = 1;
    p.cooperativeLaunch = 1;
    p.sharedMemPerBlockOptin = static_cast<std::size_t>(96) * 1024;
    p.maxBlocksPerMultiProcessor = 32;
    p.hostRegisterSupported = 1;
    p.ipcEventSupported = 0;
    p.memoryPoolsSupported = 0;

    std::size_t queried_total_mem = 0;
    if (vgpu::queryTotalGlobalMemFromServer(&queried_total_mem)) {
        p.totalGlobalMem = queried_total_mem;
    }

    p.sharedMemPerBlock = vgpu::querySizeAttrOrDefault(kAttrSharedMemPerBlock, query_device, p.sharedMemPerBlock);
    p.regsPerBlock = vgpu::queryAttrOrDefault(kAttrRegsPerBlock, query_device, p.regsPerBlock);
    p.warpSize = vgpu::queryAttrOrDefault(kAttrWarpSize, query_device, p.warpSize);
    p.memPitch = vgpu::querySizeAttrOrDefault(kAttrMemPitch, query_device, p.memPitch);
    p.maxThreadsPerBlock = vgpu::queryAttrOrDefault(kAttrMaxThreadsPerBlock, query_device, p.maxThreadsPerBlock);
    p.maxThreadsDim[0] = vgpu::queryAttrOrDefault(kAttrMaxBlockDimX, query_device, p.maxThreadsDim[0]);
    p.maxThreadsDim[1] = vgpu::queryAttrOrDefault(kAttrMaxBlockDimY, query_device, p.maxThreadsDim[1]);
    p.maxThreadsDim[2] = vgpu::queryAttrOrDefault(kAttrMaxBlockDimZ, query_device, p.maxThreadsDim[2]);
    p.maxGridSize[0] = vgpu::queryAttrOrDefault(kAttrMaxGridDimX, query_device, p.maxGridSize[0]);
    p.maxGridSize[1] = vgpu::queryAttrOrDefault(kAttrMaxGridDimY, query_device, p.maxGridSize[1]);
    p.maxGridSize[2] = vgpu::queryAttrOrDefault(kAttrMaxGridDimZ, query_device, p.maxGridSize[2]);
    p.clockRate = vgpu::queryAttrOrDefault(kAttrClockRate, query_device, p.clockRate);
    p.multiProcessorCount = vgpu::queryAttrOrDefault(kAttrMultiProcessorCount, query_device, p.multiProcessorCount);
    p.maxThreadsPerMultiProcessor =
        vgpu::queryAttrOrDefault(kAttrMaxThreadsPerMultiprocessor, query_device, p.maxThreadsPerMultiProcessor);
    p.totalConstMem = vgpu::querySizeAttrOrDefault(kAttrTotalConstantMemory, query_device, p.totalConstMem);
    p.major = vgpu::queryAttrOrDefault(kAttrComputeCapabilityMajor, query_device, p.major);
    p.minor = vgpu::queryAttrOrDefault(kAttrComputeCapabilityMinor, query_device, p.minor);

    std::memcpy(prop, &p, sizeof(p));
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaGetDeviceProperties(void* prop, int device) {
    return cudaGetDeviceProperties_v2(prop, device);
}

extern "C" cudaError_t cudaGetDriverEntryPoint(
    const char* symbol,
    void** funcPtr,
    unsigned long long /*flags*/,
    void* driverStatus) {
    if (!symbol || !funcPtr) {
        return vgpu::remember(cudaErrorUnknown);
    }

    const char* resolved_from = "none";

    static void* self_handle = []() -> void* {
        void* h = dlopen("libcuda.so", RTLD_NOW | RTLD_NOLOAD);
        if (!h) {
            h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        }
        return h;
    }();

    *funcPtr = nullptr;
    if (self_handle) {
        *funcPtr = dlsym(self_handle, symbol);
        if (*funcPtr) {
            resolved_from = "self";
        }
    }
    if (!*funcPtr) {
        *funcPtr = dlsym(RTLD_DEFAULT, symbol);
        if (*funcPtr) {
            resolved_from = "default";
        }
    }
    if (!*funcPtr) {
        *funcPtr = dlsym(RTLD_NEXT, symbol);
        if (*funcPtr) {
            resolved_from = "next";
        }
    }

    if (std::getenv("VGPU_LOG_DRIVER_ENTRYPOINT") != nullptr) {
        std::fprintf(stderr,
                     "[vgpu-cudaGetDriverEntryPoint] %s => %s (%s)\n",
                     symbol,
                     *funcPtr ? "FOUND" : "MISSING",
                     resolved_from);
    }

    if (driverStatus) {
        // cudaDriverEntryPointQueryResult:
        //   0 = Success, 1 = SymbolNotFound, 2 = VersionNotSufficent
        auto* s = static_cast<int*>(driverStatus);
        *s = *funcPtr ? 0 : 1;
    }

    // CUDA contract: symbol miss is reported via driverStatus + NULL funcPtr,
    // while return status remains cudaSuccess.
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaGetDriverEntryPointByVersion(
    const char* symbol,
    void** funcPtr,
    unsigned long long flags,
    unsigned int /*cudaVersion*/,
    void* driverStatus) {
    return cudaGetDriverEntryPoint(symbol, funcPtr, flags, driverStatus);
}

extern "C" cudaError_t cudaIpcGetEventHandle(void* handle, cudaEvent_t event) {
    (void)handle;
    (void)event;
    return vgpu::remember(vgpu::unsupported("cudaIpcGetEventHandle"));
}

extern "C" cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, void* handle) {
    (void)event;
    (void)handle;
    return vgpu::remember(vgpu::unsupported("cudaIpcOpenEventHandle"));
}

extern "C" cudaError_t cudaIpcGetMemHandle(void* handle, void* devPtr) {
    (void)handle;
    (void)devPtr;
    return vgpu::remember(vgpu::unsupported("cudaIpcGetMemHandle"));
}

extern "C" cudaError_t cudaIpcOpenMemHandle(void** devPtr, void* handle, unsigned int /*flags*/) {
    (void)devPtr;
    (void)handle;
    return vgpu::remember(vgpu::unsupported("cudaIpcOpenMemHandle"));
}

extern "C" cudaError_t cudaIpcCloseMemHandle(void* /*devPtr*/) {
    return vgpu::remember(vgpu::unsupported("cudaIpcCloseMemHandle"));
}

extern "C" cudaError_t cudaLaunchKernelExC(...) {
    return vgpu::remember(vgpu::unsupported("cudaLaunchKernelExC"));
}

extern "C" cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks,
    const void* /*func*/,
    int /*blockSize*/,
    std::size_t /*dynamicSMemSize*/,
    unsigned int /*flags*/) {
    if (numBlocks) {
        *numBlocks = 1;
    }
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaPointerGetAttributes(void* attributes, const void* /*ptr*/) {
    if (attributes) {
        std::memset(attributes, 0, 64);
    }
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaStreamAddCallback(
    cudaStream_t /*stream*/,
    void (*callback)(cudaStream_t, cudaError_t, void*),
    void* userData,
    unsigned int /*flags*/) {
    if (callback) {
        callback(nullptr, cudaSuccess, userData);
    }
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t /*stream*/, int /*mode*/) {
    return vgpu::remember(vgpu::unsupported("cudaStreamBeginCapture"));
}

extern "C" cudaError_t cudaStreamCreateWithPriority(
    cudaStream_t* pStream,
    unsigned int /*flags*/,
    int /*priority*/) {
    return cudaStreamCreate(pStream);
}

extern "C" cudaError_t cudaStreamGetPriority(cudaStream_t /*stream*/, int* priority) {
    if (priority) {
        *priority = 0;
    }
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaStreamEndCapture(cudaStream_t /*stream*/, void** pGraph) {
    if (pGraph) {
        *pGraph = nullptr;
    }
    return vgpu::remember(vgpu::unsupported("cudaStreamEndCapture"));
}

extern "C" cudaError_t cudaStreamGetCaptureInfo_v2(
    cudaStream_t /*stream*/,
    int* captureStatus,
    unsigned long long* id,
    void** graph,
    const void*** dependencies,
    std::size_t* numDependencies) {
    // Query-only path used by recent frameworks during init. Report
    // "not capturing" with empty dependency graph instead of hard-failing.
    if (captureStatus) {
        *captureStatus = 0;  // cudaStreamCaptureStatusNone
    }
    if (id) {
        *id = 0;
    }
    if (graph) {
        *graph = nullptr;
    }
    if (dependencies) {
        *dependencies = nullptr;
    }
    if (numDependencies) {
        *numDependencies = 0;
    }
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaStreamIsCapturing(cudaStream_t /*stream*/, int* captureStatus) {
    if (captureStatus) {
        *captureStatus = 0;  // cudaStreamCaptureStatusNone
    }
    return vgpu::remember(vgpu::unsupported("cudaStreamIsCapturing"));
}

extern "C" cudaError_t cudaThreadExchangeStreamCaptureMode(int* mode) {
    // Keep caller-provided mode unchanged and report success for compatibility.
    (void)mode;
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaGraphDebugDotPrint(...) {
    return vgpu::remember(vgpu::unsupported("cudaGraphDebugDotPrint"));
}

extern "C" cudaError_t cudaGraphDestroy(...) {
    return vgpu::remember(vgpu::unsupported("cudaGraphDestroy"));
}

extern "C" cudaError_t cudaGraphExecDestroy(...) {
    return vgpu::remember(vgpu::unsupported("cudaGraphExecDestroy"));
}

extern "C" cudaError_t cudaGraphGetNodes(...) {
    return vgpu::remember(vgpu::unsupported("cudaGraphGetNodes"));
}

extern "C" cudaError_t cudaGraphInstantiate(...) {
    return vgpu::remember(vgpu::unsupported("cudaGraphInstantiate"));
}

extern "C" cudaError_t cudaGraphInstantiateWithFlags(...) {
    return vgpu::remember(vgpu::unsupported("cudaGraphInstantiateWithFlags"));
}

extern "C" cudaError_t cudaGraphLaunch(...) {
    return vgpu::remember(vgpu::unsupported("cudaGraphLaunch"));
}

extern "C" cudaError_t cudaLaunchHostFunc(
    cudaStream_t /*stream*/,
    void (*fn)(void*),
    void* userData) {
    (void)fn;
    (void)userData;
    return vgpu::remember(vgpu::unsupported("cudaLaunchHostFunc"));
}

extern "C" cudaError_t cudaProfilerStart(void) {
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaProfilerStop(void) {
    return vgpu::remember(cudaSuccess);
}

extern "C" cudaError_t cudaDeviceGetDefaultMemPool(void** memPool, int /*device*/) {
    if (memPool) {
        *memPool = nullptr;
    }
    return vgpu::remember(vgpu::unsupported("cudaDeviceGetDefaultMemPool"));
}

extern "C" cudaError_t cudaMallocAsync(void** devPtr, std::size_t size, cudaStream_t /*stream*/) {
    return cudaMalloc(devPtr, size);
}

extern "C" cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t /*stream*/) {
    return cudaFree(devPtr);
}

extern "C" cudaError_t cudaMemcpyPeerAsync(
    void* dst,
    int /*dstDevice*/,
    const void* src,
    int /*srcDevice*/,
    std::size_t count,
    cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
}

extern "C" cudaError_t cudaMemPoolGetAttribute(void* /*memPool*/, int /*attr*/, void* value) {
    (void)value;
    return vgpu::remember(vgpu::unsupported("cudaMemPoolGetAttribute"));
}

extern "C" cudaError_t cudaMemPoolSetAttribute(void* /*memPool*/, int /*attr*/, void* /*value*/) {
    return vgpu::remember(vgpu::unsupported("cudaMemPoolSetAttribute"));
}

extern "C" cudaError_t cudaMemPoolSetAccess(void* /*memPool*/, const void* /*descList*/, std::size_t /*count*/) {
    return vgpu::remember(vgpu::unsupported("cudaMemPoolSetAccess"));
}

extern "C" cudaError_t cudaMemPoolTrimTo(void* /*memPool*/, std::size_t /*minBytesToKeep*/) {
    return vgpu::remember(vgpu::unsupported("cudaMemPoolTrimTo"));
}

extern "C" void __cudaRegisterVar(...) {
    // NVCC-generated static registration hook; no-op in remote shim mode.
}

extern "C" void __cudaRegisterManagedVar(...) {
    // Managed variable registration is intentionally ignored in shim mode.
}

extern "C" void** __cudaRegisterFatBinary(void* fatCubin) {
    if (std::getenv("VGPU_DEBUG") != nullptr) {
        std::fprintf(stderr, "[vGPU][reg] __cudaRegisterFatBinary fatCubin=%p\n", fatCubin);
    }
    // Allocate a persistent handle (freed when module is unregistered)
    void** handle = new void*(nullptr);

    vgpu::FatbinImage img = vgpu::resolveFatbinImage(fatCubin);

    // Upload fatbin to server; get back a module_id
    if (img.raw != nullptr && img.size > 0) {
        vgpu::RpcResult r = vgpu::callServer(
            static_cast<vgpu::RpcOp>(vgpu::RpcDrvOp::kCuModuleLoadData),
            nullptr, 0, img.raw, img.size);
        if (r.status == cudaSuccess) {
            vgpu::cacheParamInfoForModule(r.aux_u64, img);
            // Some runtimes feed __cudaRegisterFunction with alias pointers instead of
            // the wrapper pointer returned by __cudaRegisterFatBinary.
            vgpu::registerKnownModuleAliases(handle, fatCubin, r.aux_u64);
        } else if (std::getenv("VGPU_DEBUG") != nullptr) {
            std::fprintf(stderr,
                         "[vGPU][reg] __cudaRegisterFatBinary load failed status=%d size=%zu\n",
                         static_cast<int>(r.status),
                         img.size);
        }
    } else if (std::getenv("VGPU_DEBUG") != nullptr) {
        std::fprintf(stderr,
                     "[vGPU][reg] __cudaRegisterFatBinary unresolved image fatCubin=%p\n",
                     fatCubin);
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

    if (std::getenv("VGPU_DEBUG") != nullptr) {
        std::fprintf(
            stderr,
            "[vGPU][reg] __cudaRegisterFunction hostFun=%p deviceFun=%s deviceName=%s\n",
            static_cast<const void*>(hostFun),
            deviceFun ? deviceFun : "<null>",
            deviceName ? deviceName : "<null>");
    }

    std::uint64_t module_id = vgpu::globalKernelRegistry().getModuleId(fatCubinHandle);
    if (module_id == 0 && fatCubinHandle != nullptr && *fatCubinHandle != nullptr) {
        module_id = vgpu::globalKernelRegistry().getModuleId(*fatCubinHandle);
    }
    if (module_id == 0) {
        void* candidates[2] = {
            reinterpret_cast<void*>(fatCubinHandle),
            (fatCubinHandle != nullptr) ? *fatCubinHandle : nullptr,
        };

        for (void* cand : candidates) {
            if (!cand) {
                continue;
            }

            vgpu::FatbinImage img = vgpu::resolveFatbinImage(cand);
            if (img.raw == nullptr || img.size == 0) {
                continue;
            }

            vgpu::RpcResult load_r = vgpu::callServer(
                static_cast<vgpu::RpcOp>(vgpu::RpcDrvOp::kCuModuleLoadData),
                nullptr,
                0,
                img.raw,
                img.size);
            if (load_r.status != cudaSuccess) {
                if (std::getenv("VGPU_DEBUG") != nullptr) {
                    std::fprintf(stderr,
                                 "[vGPU][reg] fallback module load failed cand=%p status=%d size=%zu\n",
                                 cand,
                                 static_cast<int>(load_r.status),
                                 img.size);
                }
                continue;
            }

            module_id = load_r.aux_u64;
            vgpu::cacheParamInfoForModule(module_id, img);
            vgpu::registerKnownModuleAliases(fatCubinHandle, cand, module_id);
            break;
        }
    }
    if (module_id == 0) {
        if (std::getenv("VGPU_DEBUG") != nullptr) {
            std::fprintf(stderr, "[vGPU][reg] skip __cudaRegisterFunction: module_id=0\n");
        }
        return;  // module upload failed earlier
    }

    // Use the mangled device function name for the lookup
    const char* lookup_name = (deviceFun && deviceFun[0]) ? deviceFun : deviceName;
    if (!lookup_name) return;

    // Try param info from local PTX parse cache first
    std::vector<vgpu::ParamInfo> cached_params;
    const bool has_cached_params = vgpu::globalKernelRegistry().findParamInfo(module_id, lookup_name, &cached_params);

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
    if (r.status != cudaSuccess) {
        if (std::getenv("VGPU_DEBUG") != nullptr) {
            std::fprintf(stderr, "[vGPU][reg] kCuModuleGetFunction failed status=%d\n", static_cast<int>(r.status));
        }
        return;
    }

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
    if (ke.params.empty() && has_cached_params) {
        ke.params            = cached_params;
        ke.total_param_bytes = vgpu::computeParamBufSize(ke.params);
    }

    vgpu::globalKernelRegistry().addKernel(hostFun, std::move(ke));
    if (std::getenv("VGPU_DEBUG") != nullptr) {
        vgpu::KernelEntry added{};
        const bool found_kernel = vgpu::globalKernelRegistry().findKernel(hostFun, &added);
        std::fprintf(stderr, "[vGPU][reg] addKernel hostFun=%p func_id=%llu params=%zu total=%u\n",
                     static_cast<const void*>(hostFun),
                     found_kernel ? static_cast<unsigned long long>(added.func_id) : 0ULL,
                     found_kernel ? added.params.size() : 0U,
                     found_kernel ? added.total_param_bytes : 0U);
    }
}

extern "C" cudaError_t __cudaPushCallConfiguration(
    dim3 gridDim,
    dim3 blockDim,
    std::size_t sharedMem,
    cudaStream_t stream) {
    vgpu::g_pending_launch_cfg.grid = gridDim;
    vgpu::g_pending_launch_cfg.block = blockDim;
    vgpu::g_pending_launch_cfg.shared_mem = sharedMem;
    vgpu::g_pending_launch_cfg.stream = stream;
    vgpu::g_pending_launch_cfg.valid = true;
    return cudaSuccess;
}

extern "C" cudaError_t __cudaPopCallConfiguration(
    dim3* gridDim,
    dim3* blockDim,
    std::size_t* sharedMem,
    cudaStream_t* stream) {
    if (!vgpu::g_pending_launch_cfg.valid) {
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
        return cudaErrorUnknown;
    }

    if (gridDim != nullptr) {
        *gridDim = vgpu::g_pending_launch_cfg.grid;
    }
    if (blockDim != nullptr) {
        *blockDim = vgpu::g_pending_launch_cfg.block;
    }
    if (sharedMem != nullptr) {
        *sharedMem = vgpu::g_pending_launch_cfg.shared_mem;
    }
    if (stream != nullptr) {
        *stream = vgpu::g_pending_launch_cfg.stream;
    }
    vgpu::g_pending_launch_cfg.valid = false;
    return cudaSuccess;
}
