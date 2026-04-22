#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/common/cuda_abi.h"
#include "vgpu/backend/cuda_runtime_loader.h"
#include "vgpu/common/protocol.h"
#include "vgpu/backend/cuda_driver_loader.h"
#include "vgpu/common/fatbin_parser.h"

#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace vgpu {
namespace {

volatile sig_atomic_t g_stop = 0;

// ── Server-side driver kernel stores ────────────────────────────────────────
// Modules and functions are global (CUDA handles are process-global).
std::mutex              g_kernel_mutex;
std::atomic<uint64_t>   g_next_module_id{1};
std::atomic<uint64_t>   g_next_func_id{1};
std::unordered_map<uint64_t, CUmodule>   g_modules;
std::unordered_map<uint64_t, CUfunction> g_functions;
// module_id → { device_name → param_infos }
std::unordered_map<uint64_t,
    std::unordered_map<std::string, std::vector<ParamInfo>>> g_kernel_params;

std::mutex g_driver_ctx_mutex;
CUcontext g_driver_ctx = nullptr;
bool g_driver_ctx_ready = false;

// Lazy-loaded driver API (singleton)
CudaDriverLoader& drvLoader() {
    static CudaDriverLoader l;
    return l;
}

const char* cuResultName(CUresult cr) {
    const auto& api = drvLoader().api();
    if (api.cuGetErrorName != nullptr) {
        const char* name = nullptr;
        if (api.cuGetErrorName(cr, &name) == CUDA_SUCCESS && name != nullptr) {
            return name;
        }
    }
    return "CU_UNKNOWN";
}

const char* cuResultString(CUresult cr) {
    const auto& api = drvLoader().api();
    if (api.cuGetErrorString != nullptr) {
        const char* str = nullptr;
        if (api.cuGetErrorString(cr, &str) == CUDA_SUCCESS && str != nullptr) {
            return str;
        }
    }
    return "";
}

bool ensureDriverContextCurrent() {
    if (!drvLoader().ensureLoaded()) {
        return false;
    }

    std::lock_guard<std::mutex> lk(g_driver_ctx_mutex);
    const auto& api = drvLoader().api();

    if (g_driver_ctx_ready && g_driver_ctx != nullptr && api.cuCtxSetCurrent != nullptr) {
        return api.cuCtxSetCurrent(g_driver_ctx) == CUDA_SUCCESS;
    }

    if (api.cuInit == nullptr || api.cuDeviceGetCount == nullptr || api.cuDeviceGet == nullptr ||
        api.cuDevicePrimaryCtxRetain == nullptr || api.cuCtxSetCurrent == nullptr) {
        return false;
    }

    CUresult cr = api.cuInit(0);
    if (cr != CUDA_SUCCESS) {
        std::fprintf(stderr, "[vgpu-server][drv] cuInit failed cr=%d name=%s desc=%s\n",
                     static_cast<int>(cr), cuResultName(cr), cuResultString(cr));
        return false;
    }

    CUcontext cur = nullptr;
    if (api.cuCtxGetCurrent != nullptr && api.cuCtxGetCurrent(&cur) == CUDA_SUCCESS && cur != nullptr) {
        g_driver_ctx = cur;
        g_driver_ctx_ready = true;
        return true;
    }

    int count = 0;
    cr = api.cuDeviceGetCount(&count);
    if (cr != CUDA_SUCCESS || count <= 0) {
        std::fprintf(stderr, "[vgpu-server][drv] cuDeviceGetCount failed cr=%d count=%d name=%s desc=%s\n",
                     static_cast<int>(cr), count, cuResultName(cr), cuResultString(cr));
        return false;
    }

    CUdevice dev = 0;
    cr = api.cuDeviceGet(&dev, 0);
    if (cr != CUDA_SUCCESS) {
        std::fprintf(stderr, "[vgpu-server][drv] cuDeviceGet(0) failed cr=%d name=%s desc=%s\n",
                     static_cast<int>(cr), cuResultName(cr), cuResultString(cr));
        return false;
    }

    CUcontext ctx = nullptr;
    cr = api.cuDevicePrimaryCtxRetain(&ctx, dev);
    if (cr != CUDA_SUCCESS || ctx == nullptr) {
        std::fprintf(stderr, "[vgpu-server][drv] cuDevicePrimaryCtxRetain failed cr=%d name=%s desc=%s\n",
                     static_cast<int>(cr), cuResultName(cr), cuResultString(cr));
        return false;
    }

    cr = api.cuCtxSetCurrent(ctx);
    if (cr != CUDA_SUCCESS) {
        std::fprintf(stderr, "[vgpu-server][drv] cuCtxSetCurrent failed cr=%d name=%s desc=%s\n",
                     static_cast<int>(cr), cuResultName(cr), cuResultString(cr));
        return false;
    }

    g_driver_ctx = ctx;
    g_driver_ctx_ready = true;
    return true;
}

void onSignal(int) {
    g_stop = 1;
}

bool writeAll(int fd, const void* data, std::size_t len) {
    const std::uint8_t* p = static_cast<const std::uint8_t*>(data);
    std::size_t done = 0;
    while (done < len) {
        ssize_t n = write(fd, p + done, len - done);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }
        done += static_cast<std::size_t>(n);
    }
    return true;
}

bool readAll(int fd, void* data, std::size_t len) {
    std::uint8_t* p = static_cast<std::uint8_t*>(data);
    std::size_t done = 0;
    while (done < len) {
        ssize_t n = read(fd, p + done, len - done);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }
        done += static_cast<std::size_t>(n);
    }
    return true;
}

std::string socketPath() {
    const char* env = std::getenv("VGPU_SERVER_SOCK");
    if (env != nullptr && env[0] != '\0') {
        return std::string(env);
    }
    return "/tmp/vgpu_server.sock";
}

int envAsInt(const char* key, int def) {
    const char* v = std::getenv(key);
    if (v == nullptr) {
        return def;
    }
    return std::atoi(v);
}

std::size_t envAsSize(const char* key, std::size_t def) {
    const char* v = std::getenv(key);
    if (v == nullptr) {
        return def;
    }
    long long n = std::atoll(v);
    if (n <= 0) {
        return def;
    }
    return static_cast<std::size_t>(n);
}

struct Policy {
    int force_device = -1;
    int delay_us = 0;
    std::size_t memcpy_clamp_bytes = 0;
    bool verbose = false;
};

Policy loadPolicy() {
    Policy p;
    p.force_device = envAsInt("VGPU_FORCE_DEVICE", -1);
    p.delay_us = envAsInt("VGPU_DELAY_US", 0);
    if (p.delay_us < 0) {
        p.delay_us = 0;
    }
    p.memcpy_clamp_bytes = envAsSize("VGPU_MEMCPY_CLAMP_BYTES", 0);
    p.verbose = envAsInt("VGPU_VERBOSE", 0) != 0;
    return p;
}

void logReq(const Policy& p, const RpcRequestHeader& req, cudaError_t st) {
    if (!p.verbose) {
        return;
    }
    std::fprintf(
        stderr,
        "[vgpu-server] app=%llu ctx=%llu op=%u dev=%d st=%d\n",
        static_cast<unsigned long long>(req.app_id),
        static_cast<unsigned long long>(req.context_id),
        req.op,
        req.device,
        st);
}

cudaError_t notSupported() {
    static constexpr cudaError_t kCudaErrorNotSupported = 801;
    return kCudaErrorNotSupported;
}

struct DispatchResult {
    cudaError_t status = cudaErrorUnknown;
    std::uint64_t aux = 0;
    std::vector<std::uint8_t> payload;
};

std::size_t clampMemcpyCount(std::size_t count, const Policy& policy) {
    if (policy.memcpy_clamp_bytes > 0 && count > policy.memcpy_clamp_bytes) {
        return policy.memcpy_clamp_bytes;
    }
    return count;
}

bool handleRuntimeMemcpyOp(
    RpcOp op,
    const std::vector<std::uint8_t>& payload,
    const Policy& policy,
    const CudaRuntimeLoader::Api& api,
    DispatchResult* out) {
    if (op != RpcOp::kCudaMemcpy && op != RpcOp::kCudaMemcpyAsync) {
        return false;
    }

    if (op == RpcOp::kCudaMemcpy) {
        if (payload.size() < sizeof(RpcMemcpyReq)) {
            out->status = cudaErrorUnknown;
            return true;
        }
        const auto* p = reinterpret_cast<const RpcMemcpyReq*>(payload.data());
        const std::size_t requested_count = static_cast<std::size_t>(p->count);
        std::size_t count = clampMemcpyCount(requested_count, policy);
        if (count != requested_count) {
            out->status = cudaErrorInvalidValue;
            return true;
        }
        cudaMemcpyKind kind = static_cast<cudaMemcpyKind>(p->kind);

        if (kind == cudaMemcpyHostToDevice) {
            const std::size_t need = sizeof(RpcMemcpyReq) + count;
            out->status = (payload.size() < need)
                ? cudaErrorUnknown
                : api.cudaMemcpy(reinterpret_cast<void*>(p->dst), payload.data() + sizeof(RpcMemcpyReq), count, cudaMemcpyHostToDevice);
        } else if (kind == cudaMemcpyDeviceToHost) {
            out->payload.resize(count);
            out->status = api.cudaMemcpy(out->payload.data(), reinterpret_cast<const void*>(p->src), count, cudaMemcpyDeviceToHost);
        } else if (kind == cudaMemcpyDeviceToDevice) {
            out->status = api.cudaMemcpy(reinterpret_cast<void*>(p->dst), reinterpret_cast<const void*>(p->src), count, cudaMemcpyDeviceToDevice);
        } else {
            out->status = notSupported();
        }
        return true;
    }

    if (api.cudaMemcpyAsync == nullptr || payload.size() < sizeof(RpcMemcpyAsyncReq)) {
        out->status = (api.cudaMemcpyAsync == nullptr) ? notSupported() : cudaErrorUnknown;
        return true;
    }

    const auto* p = reinterpret_cast<const RpcMemcpyAsyncReq*>(payload.data());
    const std::size_t requested_count = static_cast<std::size_t>(p->count);
    std::size_t count = clampMemcpyCount(requested_count, policy);
    if (count != requested_count) {
        out->status = cudaErrorInvalidValue;
        return true;
    }
    cudaMemcpyKind kind = static_cast<cudaMemcpyKind>(p->kind);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(p->stream);

    if (kind == cudaMemcpyHostToDevice) {
        const std::size_t need = sizeof(RpcMemcpyAsyncReq) + count;
        out->status = (payload.size() < need)
            ? cudaErrorUnknown
            : api.cudaMemcpyAsync(reinterpret_cast<void*>(p->dst), payload.data() + sizeof(RpcMemcpyAsyncReq), count, cudaMemcpyHostToDevice, stream);
    } else if (kind == cudaMemcpyDeviceToHost) {
        out->payload.resize(count);
        out->status = api.cudaMemcpyAsync(out->payload.data(), reinterpret_cast<const void*>(p->src), count, cudaMemcpyDeviceToHost, stream);
        if (out->status == cudaSuccess && api.cudaStreamSynchronize != nullptr) {
            out->status = api.cudaStreamSynchronize(stream);
        }
    } else if (kind == cudaMemcpyDeviceToDevice) {
        out->status = api.cudaMemcpyAsync(reinterpret_cast<void*>(p->dst), reinterpret_cast<const void*>(p->src), count, cudaMemcpyDeviceToDevice, stream);
    } else {
        out->status = notSupported();
    }
    return true;
}

bool handleRuntimeMemoryOp(
    RpcOp op,
    const std::vector<std::uint8_t>& payload,
    const CudaRuntimeLoader::Api& api,
    DispatchResult* out) {
    if (op == RpcOp::kCudaMalloc) {
        if (payload.size() != sizeof(RpcMallocReq)) {
            out->status = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcMallocReq*>(payload.data());
            void* dev_ptr = nullptr;
            out->status = api.cudaMalloc(&dev_ptr, static_cast<std::size_t>(p->size));
            out->aux = reinterpret_cast<std::uint64_t>(dev_ptr);
        }
        return true;
    }
    if (op == RpcOp::kCudaFree) {
        if (payload.size() != sizeof(RpcFreeReq)) {
            out->status = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcFreeReq*>(payload.data());
            out->status = api.cudaFree(reinterpret_cast<void*>(p->dev_ptr));
        }
        return true;
    }
    if (op == RpcOp::kCudaMemset || op == RpcOp::kCudaMemsetAsync) {
        if (op == RpcOp::kCudaMemset) {
            if (api.cudaMemset == nullptr || payload.size() != sizeof(RpcMemsetReq)) {
                out->status = (api.cudaMemset == nullptr) ? notSupported() : cudaErrorUnknown;
            } else {
                const auto* p = reinterpret_cast<const RpcMemsetReq*>(payload.data());
                out->status = api.cudaMemset(reinterpret_cast<void*>(p->dst), p->value, static_cast<std::size_t>(p->count));
            }
        } else {
            if (api.cudaMemsetAsync == nullptr || payload.size() != sizeof(RpcMemsetAsyncReq)) {
                out->status = (api.cudaMemsetAsync == nullptr) ? notSupported() : cudaErrorUnknown;
            } else {
                const auto* p = reinterpret_cast<const RpcMemsetAsyncReq*>(payload.data());
                out->status = api.cudaMemsetAsync(reinterpret_cast<void*>(p->dst), p->value, static_cast<std::size_t>(p->count), reinterpret_cast<cudaStream_t>(p->stream));
            }
        }
        return true;
    }
    if (op == RpcOp::kCudaMemGetInfo) {
        if (api.cudaMemGetInfo == nullptr) {
            out->status = notSupported();
        } else {
            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            out->status = api.cudaMemGetInfo(&free_bytes, &total_bytes);
            if (out->status == cudaSuccess) {
                RpcMemGetInfoRsp info{};
                info.free_bytes = static_cast<std::uint64_t>(free_bytes);
                info.total_bytes = static_cast<std::uint64_t>(total_bytes);
                out->payload.resize(sizeof(info));
                std::memcpy(out->payload.data(), &info, sizeof(info));
            }
        }
        return true;
    }
    return false;
}

bool handleRuntimeStreamOp(
    RpcOp op,
    const std::vector<std::uint8_t>& payload,
    const CudaRuntimeLoader::Api& api,
    DispatchResult* out) {
    if (op == RpcOp::kCudaDeviceSynchronize) {
        out->status = api.cudaDeviceSynchronize();
        return true;
    }
    if (op == RpcOp::kCudaStreamCreate) {
        cudaStream_t stream = nullptr;
        out->status = api.cudaStreamCreate(&stream);
        out->aux = reinterpret_cast<std::uint64_t>(stream);
        return true;
    }
    if (op == RpcOp::kCudaStreamDestroy) {
        if (payload.size() != sizeof(RpcStreamDestroyReq)) {
            out->status = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcStreamDestroyReq*>(payload.data());
            out->status = api.cudaStreamDestroy(reinterpret_cast<cudaStream_t>(p->stream));
        }
        return true;
    }
    if (op == RpcOp::kCudaStreamSynchronize || op == RpcOp::kCudaStreamQuery || op == RpcOp::kCudaStreamWaitEvent) {
        if (op == RpcOp::kCudaStreamSynchronize) {
            if (api.cudaStreamSynchronize == nullptr || payload.size() != sizeof(RpcStreamSyncReq)) {
                out->status = (api.cudaStreamSynchronize == nullptr) ? notSupported() : cudaErrorUnknown;
            } else {
                const auto* p = reinterpret_cast<const RpcStreamSyncReq*>(payload.data());
                out->status = api.cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(p->stream));
            }
        } else if (op == RpcOp::kCudaStreamQuery) {
            if (api.cudaStreamQuery == nullptr || payload.size() != sizeof(RpcStreamQueryReq)) {
                out->status = (api.cudaStreamQuery == nullptr) ? notSupported() : cudaErrorUnknown;
            } else {
                const auto* p = reinterpret_cast<const RpcStreamQueryReq*>(payload.data());
                out->status = api.cudaStreamQuery(reinterpret_cast<cudaStream_t>(p->stream));
            }
        } else {
            if (api.cudaStreamWaitEvent == nullptr || payload.size() != sizeof(RpcStreamWaitEventReq)) {
                out->status = (api.cudaStreamWaitEvent == nullptr) ? notSupported() : cudaErrorUnknown;
            } else {
                const auto* p = reinterpret_cast<const RpcStreamWaitEventReq*>(payload.data());
                out->status = api.cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(p->stream), reinterpret_cast<cudaEvent_t>(p->event), p->flags);
            }
        }
        return true;
    }
    return false;
}

bool handleRuntimeEventOp(
    RpcOp op,
    const std::vector<std::uint8_t>& payload,
    const CudaRuntimeLoader::Api& api,
    DispatchResult* out) {
    if (op != RpcOp::kCudaEventCreate && op != RpcOp::kCudaEventDestroy &&
        op != RpcOp::kCudaEventRecord && op != RpcOp::kCudaEventSynchronize &&
        op != RpcOp::kCudaEventQuery) {
        return false;
    }

    if (op == RpcOp::kCudaEventCreate) {
        if (api.cudaEventCreateWithFlags == nullptr || payload.size() != sizeof(RpcEventCreateReq)) {
            out->status = (api.cudaEventCreateWithFlags == nullptr) ? notSupported() : cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcEventCreateReq*>(payload.data());
            cudaEvent_t event = nullptr;
            out->status = api.cudaEventCreateWithFlags(&event, p->flags);
            out->aux = reinterpret_cast<std::uint64_t>(event);
        }
    } else if (op == RpcOp::kCudaEventDestroy) {
        if (api.cudaEventDestroy == nullptr || payload.size() != sizeof(RpcEventDestroyReq)) {
            out->status = (api.cudaEventDestroy == nullptr) ? notSupported() : cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcEventDestroyReq*>(payload.data());
            out->status = api.cudaEventDestroy(reinterpret_cast<cudaEvent_t>(p->event));
        }
    } else if (op == RpcOp::kCudaEventRecord) {
        if (api.cudaEventRecord == nullptr || payload.size() != sizeof(RpcEventRecordReq)) {
            out->status = (api.cudaEventRecord == nullptr) ? notSupported() : cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcEventRecordReq*>(payload.data());
            out->status = api.cudaEventRecord(reinterpret_cast<cudaEvent_t>(p->event), reinterpret_cast<cudaStream_t>(p->stream));
        }
    } else if (op == RpcOp::kCudaEventSynchronize) {
        if (api.cudaEventSynchronize == nullptr || payload.size() != sizeof(RpcEventSynchronizeReq)) {
            out->status = (api.cudaEventSynchronize == nullptr) ? notSupported() : cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcEventSynchronizeReq*>(payload.data());
            out->status = api.cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(p->event));
        }
    } else {
        if (api.cudaEventQuery == nullptr || payload.size() != sizeof(RpcEventQueryReq)) {
            out->status = (api.cudaEventQuery == nullptr) ? notSupported() : cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcEventQueryReq*>(payload.data());
            out->status = api.cudaEventQuery(reinterpret_cast<cudaEvent_t>(p->event));
        }
    }
    return true;
}

bool handleRuntimeDeviceOp(
    RpcOp op,
    const std::vector<std::uint8_t>& payload,
    const Policy& policy,
    const CudaRuntimeLoader::Api& api,
    DispatchResult* out) {
    if (op == RpcOp::kCudaSetDevice) {
        if (payload.size() != sizeof(RpcSetDeviceReq)) {
            out->status = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcSetDeviceReq*>(payload.data());
            int device = (policy.force_device >= 0) ? policy.force_device : p->device;
            out->status = api.cudaSetDevice(device);
        }
        return true;
    }
    if (op == RpcOp::kCudaGetDevice || op == RpcOp::kCudaGetDeviceCount) {
        if (op == RpcOp::kCudaGetDevice) {
            int device = -1;
            out->status = api.cudaGetDevice(&device);
            out->aux = static_cast<std::uint64_t>(device);
        } else if (api.cudaGetDeviceCount == nullptr) {
            out->status = notSupported();
        } else {
            int count = 0;
            out->status = api.cudaGetDeviceCount(&count);
            out->aux = static_cast<std::uint64_t>(count);
        }
        return true;
    }
    if (op == RpcOp::kCudaSetDeviceFlags || op == RpcOp::kCudaGetDeviceFlags) {
        if (op == RpcOp::kCudaSetDeviceFlags) {
            if (api.cudaSetDeviceFlags == nullptr || payload.size() != sizeof(RpcSetDeviceFlagsReq)) {
                out->status = (api.cudaSetDeviceFlags == nullptr) ? notSupported() : cudaErrorUnknown;
            } else {
                const auto* p = reinterpret_cast<const RpcSetDeviceFlagsReq*>(payload.data());
                out->status = api.cudaSetDeviceFlags(p->flags);
            }
        } else if (api.cudaGetDeviceFlags == nullptr) {
            out->status = notSupported();
        } else {
            unsigned int flags = 0;
            out->status = api.cudaGetDeviceFlags(&flags);
            out->aux = static_cast<std::uint64_t>(flags);
        }
        return true;
    }
    if (op == RpcOp::kCudaDeviceReset || op == RpcOp::kCudaRuntimeGetVersion ||
        op == RpcOp::kCudaDriverGetVersion || op == RpcOp::kCudaDeviceGetAttribute ||
        op == RpcOp::kCudaLaunchKernel) {
        if (op == RpcOp::kCudaDeviceReset) {
            out->status = (api.cudaDeviceReset == nullptr) ? notSupported() : api.cudaDeviceReset();
        } else if (op == RpcOp::kCudaRuntimeGetVersion) {
            if (api.cudaRuntimeGetVersion == nullptr) {
                out->status = notSupported();
            } else {
                int version = 0;
                out->status = api.cudaRuntimeGetVersion(&version);
                out->aux = static_cast<std::uint64_t>(version);
            }
        } else if (op == RpcOp::kCudaDriverGetVersion) {
            if (api.cudaDriverGetVersion == nullptr) {
                out->status = notSupported();
            } else {
                int version = 0;
                out->status = api.cudaDriverGetVersion(&version);
                out->aux = static_cast<std::uint64_t>(version);
            }
        } else if (op == RpcOp::kCudaDeviceGetAttribute) {
            if (api.cudaDeviceGetAttribute == nullptr || payload.size() != sizeof(RpcDeviceGetAttributeReq)) {
                out->status = (api.cudaDeviceGetAttribute == nullptr) ? notSupported() : cudaErrorUnknown;
            } else {
                const auto* p = reinterpret_cast<const RpcDeviceGetAttributeReq*>(payload.data());
                int value = 0;
                out->status = api.cudaDeviceGetAttribute(&value, p->attr, p->device);
                out->aux = static_cast<std::uint64_t>(value);
            }
        } else {
            out->status = notSupported();
        }
        return true;
    }
    return false;
}

bool handleRuntimeOp(
    RpcOp op,
    const std::vector<std::uint8_t>& payload,
    const Policy& policy,
    const CudaRuntimeLoader::Api& api,
    DispatchResult* out) {
    if (out == nullptr) {
        return false;
    }
    return handleRuntimeMemcpyOp(op, payload, policy, api, out) ||
           handleRuntimeMemoryOp(op, payload, api, out) ||
           handleRuntimeStreamOp(op, payload, api, out) ||
           handleRuntimeEventOp(op, payload, api, out) ||
           handleRuntimeDeviceOp(op, payload, policy, api, out);
}

bool handleDriverModuleLoadDataOp(
    const std::vector<std::uint8_t>& payload,
    DispatchResult* out) {
    if (!ensureDriverContextCurrent()) {
        out->status = notSupported();
        return true;
    }
    if (payload.empty()) {
        out->status = cudaErrorUnknown;
        return true;
    }

    CUmodule mod = nullptr;
    CUresult cr = drvLoader().api().cuModuleLoadData(&mod, payload.data());
    if (cr != CUDA_SUCCESS) {
        std::uint32_t magic = 0;
        std::uint32_t w1 = 0;
        std::uint32_t w2 = 0;
        std::uint32_t w3 = 0;
        std::uint32_t w4 = 0;
        std::uint32_t w5 = 0;
        if (payload.size() >= sizeof(magic)) {
            std::memcpy(&magic, payload.data(), sizeof(magic));
        }
        if (payload.size() >= 24) {
            std::memcpy(&w1, payload.data() + 4, sizeof(w1));
            std::memcpy(&w2, payload.data() + 8, sizeof(w2));
            std::memcpy(&w3, payload.data() + 12, sizeof(w3));
            std::memcpy(&w4, payload.data() + 16, sizeof(w4));
            std::memcpy(&w5, payload.data() + 20, sizeof(w5));
        }
        std::fprintf(stderr,
                     "[vgpu-server][drv] cuModuleLoadData failed cr=%d name=%s desc=%s payload_size=%zu magic=0x%08x hdr=[0x%08x 0x%08x 0x%08x 0x%08x 0x%08x]\n",
                     static_cast<int>(cr),
                     cuResultName(cr),
                     cuResultString(cr),
                     payload.size(),
                     magic,
                     w1,
                     w2,
                     w3,
                     w4,
                     w5);
        out->status = cudaErrorUnknown;
        return true;
    }

    uint64_t mid = g_next_module_id.fetch_add(1);
    {
        std::lock_guard<std::mutex> lk(g_kernel_mutex);
        g_modules[mid] = mod;
        auto ki_list = parseFatbin(payload.data(), payload.size());
        for (auto& ki : ki_list) {
            g_kernel_params[mid][ki.mangled_name] = std::move(ki.params);
        }
    }
    out->aux = mid;
    out->status = cudaSuccess;
    return true;
}

bool handleDriverModuleGetFunctionOp(
    const std::vector<std::uint8_t>& payload,
    DispatchResult* out) {
    if (payload.size() < sizeof(RpcCuModuleGetFunctionReq)) {
        out->status = cudaErrorUnknown;
        return true;
    }
    if (!ensureDriverContextCurrent()) {
        out->status = notSupported();
        return true;
    }

    const auto* req2 = reinterpret_cast<const RpcCuModuleGetFunctionReq*>(payload.data());
    const char* fname = reinterpret_cast<const char*>(req2 + 1);
    CUmodule mod = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_kernel_mutex);
        auto it = g_modules.find(req2->module_id);
        if (it != g_modules.end()) mod = it->second;
    }
    if (!mod) {
        out->status = cudaErrorUnknown;
        return true;
    }

    CUfunction func = nullptr;
    CUresult cr = drvLoader().api().cuModuleGetFunction(&func, mod, fname);
    if (cr != CUDA_SUCCESS || !func) {
        std::fprintf(stderr,
                     "[vgpu-server][drv] cuModuleGetFunction failed cr=%d name=%s desc=%s module_id=%llu func=%s\n",
                     static_cast<int>(cr),
                     cuResultName(cr),
                     cuResultString(cr),
                     static_cast<unsigned long long>(req2->module_id),
                     fname);
        out->status = cudaErrorUnknown;
        return true;
    }

    uint64_t fid = g_next_func_id.fetch_add(1);
    {
        std::lock_guard<std::mutex> lk(g_kernel_mutex);
        g_functions[fid] = func;
    }
    out->aux = fid;
    out->status = cudaSuccess;

    std::vector<RpcParamInfo> param_wire;
    uint32_t total_param_bytes = 0;
    {
        std::lock_guard<std::mutex> lk(g_kernel_mutex);
        auto mit = g_kernel_params.find(req2->module_id);
        if (mit != g_kernel_params.end()) {
            auto fit = mit->second.find(fname);
            if (fit != mit->second.end()) {
                for (const auto& p : fit->second) {
                    RpcParamInfo rp{};
                    rp.size = p.size;
                    rp.alignment = p.alignment;
                    param_wire.push_back(rp);
                }
                total_param_bytes = computeParamBufSize(fit->second);
            }
        }
    }

    if (param_wire.empty() && drvLoader().api().cuFuncGetAttribute != nullptr) {
        int psz = 0;
        drvLoader().api().cuFuncGetAttribute(&psz, CU_FUNC_ATTR_PARAM_SIZE_BYTES, func);
        total_param_bytes = static_cast<uint32_t>(psz);
    }

    if (param_wire.empty() && drvLoader().api().cuFuncGetParamInfo != nullptr) {
        struct ParamOffSize {
            uint32_t off;
            uint32_t size;
        };
        std::vector<ParamOffSize> raw_params;

        for (std::size_t i = 0; i < 128; ++i) {
            std::size_t off = 0;
            std::size_t sz = 0;
            CUresult pr = drvLoader().api().cuFuncGetParamInfo(func, i, &off, &sz);
            if (pr == CUDA_SUCCESS) {
                if (sz == 0 || off > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) ||
                    sz > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
                    break;
                }
                ParamOffSize p{};
                p.off = static_cast<uint32_t>(off);
                p.size = static_cast<uint32_t>(sz);
                raw_params.push_back(p);
                continue;
            }
            if (pr == CUDA_ERROR_INVALID_VALUE || pr == CUDA_ERROR_NOT_FOUND) {
                break;
            }
            break;
        }

        uint32_t cursor = 0;
        for (const auto& p : raw_params) {
            auto alignUp = [](uint32_t v, uint32_t a) -> uint32_t {
                return (v + a - 1u) & ~(a - 1u);
            };

            uint32_t align = 1;
            const uint32_t cands[] = {32, 16, 8, 4, 2, 1};
            for (uint32_t a : cands) {
                if (alignUp(cursor, a) == p.off) {
                    align = a;
                    break;
                }
            }

            RpcParamInfo rp{};
            rp.size = p.size;
            rp.alignment = align;
            param_wire.push_back(rp);
            cursor = p.off + p.size;
            if (cursor > total_param_bytes) {
                total_param_bytes = cursor;
            }
        }
    }

    RpcCuModuleGetFunctionRsp rsp2{};
    rsp2.param_count = static_cast<uint32_t>(param_wire.size());
    rsp2.total_param_bytes = total_param_bytes;
    out->payload.resize(sizeof(rsp2) + param_wire.size() * sizeof(RpcParamInfo));
    std::memcpy(out->payload.data(), &rsp2, sizeof(rsp2));
    std::memcpy(out->payload.data() + sizeof(rsp2), param_wire.data(), param_wire.size() * sizeof(RpcParamInfo));
    return true;
}

bool handleDriverLaunchKernelOp(
    const std::vector<std::uint8_t>& payload,
    const CudaRuntimeLoader::Api& api,
    DispatchResult* out) {
    if (payload.size() < sizeof(RpcCuLaunchKernelReq)) {
        out->status = cudaErrorUnknown;
        return true;
    }
    if (!ensureDriverContextCurrent()) {
        out->status = notSupported();
        return true;
    }

    const auto* lreq = reinterpret_cast<const RpcCuLaunchKernelReq*>(payload.data());
    CUfunction func = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_kernel_mutex);
        auto it = g_functions.find(lreq->func_id);
        if (it != g_functions.end()) func = it->second;
    }
    if (!func || payload.size() < sizeof(RpcCuLaunchKernelReq) + static_cast<size_t>(lreq->arg_buf_size)) {
        out->status = cudaErrorUnknown;
        return true;
    }

    const uint8_t* arg_data = payload.data() + sizeof(RpcCuLaunchKernelReq);
    size_t arg_size = static_cast<size_t>(lreq->arg_buf_size);
    void* buf_ptr = const_cast<uint8_t*>(arg_data);
    void* extra[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER, buf_ptr,
        CU_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        CU_LAUNCH_PARAM_END
    };

    CUresult cr = drvLoader().api().cuLaunchKernel(
        func,
        lreq->grid_x, lreq->grid_y, lreq->grid_z,
        lreq->block_x, lreq->block_y, lreq->block_z,
        lreq->shared_mem_bytes,
        reinterpret_cast<CUstream_drv>(lreq->stream),
        nullptr,
        extra);

    if (cr != CUDA_SUCCESS && arg_size > 0 && (arg_size % sizeof(std::uint64_t) == 0)) {
        // Fallback path for kernels without decoded param metadata:
        // treat arg_data as contiguous 8-byte argument slots and launch
        // via kernelParams[] mode.
        const std::size_t slot_count = arg_size / sizeof(std::uint64_t);
        std::vector<std::uint64_t> arg_slots(slot_count, 0);
        std::memcpy(arg_slots.data(), arg_data, arg_size);
        std::vector<void*> kernel_params(slot_count, nullptr);
        for (std::size_t i = 0; i < slot_count; ++i) {
            kernel_params[i] = &arg_slots[i];
        }

        cr = drvLoader().api().cuLaunchKernel(
            func,
            lreq->grid_x, lreq->grid_y, lreq->grid_z,
            lreq->block_x, lreq->block_y, lreq->block_z,
            lreq->shared_mem_bytes,
            reinterpret_cast<CUstream_drv>(lreq->stream),
            kernel_params.empty() ? nullptr : kernel_params.data(),
            nullptr);
    }

    out->status = (cr == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
    if (cr != CUDA_SUCCESS) {
        std::fprintf(stderr,
                     "[vgpu-server][drv] cuLaunchKernel failed cr=%d name=%s desc=%s func_id=%llu\n",
                     static_cast<int>(cr),
                     cuResultName(cr),
                     cuResultString(cr),
                     static_cast<unsigned long long>(lreq->func_id));
    }
    if (cr == CUDA_SUCCESS && api.cudaDeviceSynchronize) {
        api.cudaDeviceSynchronize();
    }
    return true;
}

bool handleDriverModuleUnloadOp(
    const std::vector<std::uint8_t>& payload,
    DispatchResult* out) {
    if (payload.size() < sizeof(uint64_t)) {
        return true;
    }
    uint64_t mid = 0;
    std::memcpy(&mid, payload.data(), sizeof(mid));
    CUmodule mod = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_kernel_mutex);
        auto it = g_modules.find(mid);
        if (it != g_modules.end()) {
            mod = it->second;
            g_modules.erase(it);
            g_kernel_params.erase(mid);
        }
    }
    if (mod && drvLoader().ensureLoaded() && drvLoader().api().cuModuleUnload) {
        drvLoader().api().cuModuleUnload(mod);
    }
    out->status = cudaSuccess;
    return true;
}

bool handleDriverModuleOp(
    RpcDrvOp op,
    const std::vector<std::uint8_t>& payload,
    const CudaRuntimeLoader::Api& api,
    DispatchResult* out) {
    if (op == RpcDrvOp::kCuModuleLoadData) {
        return handleDriverModuleLoadDataOp(payload, out);
    }
    if (op == RpcDrvOp::kCuModuleGetFunction) {
        return handleDriverModuleGetFunctionOp(payload, out);
    }
    if (op == RpcDrvOp::kCuLaunchKernel) {
        return handleDriverLaunchKernelOp(payload, api, out);
    }
    if (op == RpcDrvOp::kCuModuleUnload) {
        return handleDriverModuleUnloadOp(payload, out);
    }
    return false;
}

bool handleDriverMemoryOp(
    RpcDrvOp op,
    const std::vector<std::uint8_t>& payload,
    const CudaRuntimeLoader::Api& api,
    DispatchResult* out) {
    if (op == RpcDrvOp::kCuMemAlloc) {
        if (payload.size() == sizeof(RpcMallocReq)) {
            const auto* p = reinterpret_cast<const RpcMallocReq*>(payload.data());
            void* dev_ptr = nullptr;
            out->status = api.cudaMalloc(&dev_ptr, static_cast<std::size_t>(p->size));
            out->aux = reinterpret_cast<std::uint64_t>(dev_ptr);
        }
        return true;
    }
    if (op == RpcDrvOp::kCuMemFree) {
        if (payload.size() == sizeof(RpcFreeReq)) {
            const auto* p = reinterpret_cast<const RpcFreeReq*>(payload.data());
            out->status = api.cudaFree(reinterpret_cast<void*>(p->dev_ptr));
        }
        return true;
    }
    if (op == RpcDrvOp::kCuMemcpyHtoD || op == RpcDrvOp::kCuMemcpyDtoH || op == RpcDrvOp::kCuMemcpyDtoD) {
        if (payload.size() < sizeof(RpcMemcpyReq)) {
            out->status = cudaErrorUnknown;
            return true;
        }
        const auto* p = reinterpret_cast<const RpcMemcpyReq*>(payload.data());
        std::size_t count = static_cast<std::size_t>(p->count);
        if (op == RpcDrvOp::kCuMemcpyHtoD) {
            const std::size_t need = sizeof(RpcMemcpyReq) + count;
            out->status = (payload.size() < need)
                ? cudaErrorUnknown
                : api.cudaMemcpy(reinterpret_cast<void*>(p->dst), payload.data() + sizeof(RpcMemcpyReq), count, cudaMemcpyHostToDevice);
        } else if (op == RpcDrvOp::kCuMemcpyDtoH) {
            out->payload.resize(count);
            out->status = api.cudaMemcpy(out->payload.data(), reinterpret_cast<const void*>(p->src), count, cudaMemcpyDeviceToHost);
        } else {
            out->status = api.cudaMemcpy(reinterpret_cast<void*>(p->dst), reinterpret_cast<const void*>(p->src), count, cudaMemcpyDeviceToDevice);
        }
        return true;
    }
    return false;
}

bool handleDriverOp(
    RpcDrvOp op,
    const std::vector<std::uint8_t>& payload,
    const CudaRuntimeLoader::Api& api,
    DispatchResult* out) {
    if (out == nullptr) {
        return false;
    }
    if (handleDriverModuleOp(op, payload, api, out)) {
        return true;
    }
    if (handleDriverMemoryOp(op, payload, api, out)) {
        return true;
    }
    if (op == RpcDrvOp::kCuCtxSynchronize) {
        out->status = api.cudaDeviceSynchronize();
        return true;
    }
    return false;
}

bool handleClient(int fd, CudaRuntimeLoader& rt, const Policy& policy) {
    RpcRequestHeader req{};
    if (!readAll(fd, &req, sizeof(req))) {
        return false;
    }
    if (req.magic != kRpcMagic || req.version != kRpcVersion) {
        return false;
    }

    std::vector<std::uint8_t> payload;
    if (req.payload_size > 0) {
        payload.resize(static_cast<std::size_t>(req.payload_size));
        if (!readAll(fd, payload.data(), payload.size())) {
            return false;
        }
    }

    if (!rt.ensureLoaded()) {
        RpcResponseHeader rsp{};
        rsp.magic = kRpcMagic;
        rsp.version = kRpcVersion;
        rsp.status = cudaErrorUnknown;
        return writeAll(fd, &rsp, sizeof(rsp));
    }

    const auto& api = rt.api();
    if (policy.delay_us > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(policy.delay_us));
    }

    DispatchResult dispatch{};

    RpcOp op = static_cast<RpcOp>(req.op);
    if (policy.force_device >= 0 && api.cudaSetDevice != nullptr && op != RpcOp::kCudaSetDevice) {
        api.cudaSetDevice(policy.force_device);
    } else if (req.device >= 0 && api.cudaSetDevice != nullptr && op != RpcOp::kCudaSetDevice) {
        api.cudaSetDevice(req.device);
    }

    const bool runtime_handled = handleRuntimeOp(op, payload, policy, api, &dispatch);
    if (!runtime_handled) {
        (void)handleDriverOp(static_cast<RpcDrvOp>(req.op), payload, api, &dispatch);
    }

    RpcResponseHeader rsp{};
    rsp.magic = kRpcMagic;
    rsp.version = kRpcVersion;
    rsp.status = dispatch.status;
    rsp.aux_u64 = dispatch.aux;
    rsp.payload_size = dispatch.payload.size();

    logReq(policy, req, dispatch.status);

    if (!writeAll(fd, &rsp, sizeof(rsp))) {
        return false;
    }
    if (!dispatch.payload.empty()) {
        if (!writeAll(fd, dispatch.payload.data(), dispatch.payload.size())) {
            return false;
        }
    }
    return true;
}

}  // namespace
}  // namespace vgpu

int main() {
    using namespace vgpu;

    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    const std::string sock = socketPath();
    unlink(sock.c_str());

    int listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        std::fprintf(stderr, "failed to create socket\n");
        return 1;
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (sock.size() >= sizeof(addr.sun_path)) {
        std::fprintf(stderr, "socket path too long: %s\n", sock.c_str());
        close(listen_fd);
        return 2;
    }
    std::strncpy(addr.sun_path, sock.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::fprintf(stderr, "bind failed: %s\n", std::strerror(errno));
        close(listen_fd);
        return 3;
    }

    if (listen(listen_fd, 128) != 0) {
        std::fprintf(stderr, "listen failed: %s\n", std::strerror(errno));
        close(listen_fd);
        unlink(sock.c_str());
        return 4;
    }

    CudaRuntimeLoader rt;
    Policy policy = loadPolicy();

    if (policy.verbose) {
        std::fprintf(stderr, "[vgpu-server] listening on %s\n", sock.c_str());
    }

    while (!g_stop) {
        int fd = accept(listen_fd, nullptr, nullptr);
        if (fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            std::fprintf(stderr, "accept failed: %s\n", std::strerror(errno));
            break;
        }

        std::thread([fd, &rt, &policy]() {
            handleClient(fd, rt, policy);
            close(fd);
        }).detach();
    }

    close(listen_fd);
    unlink(sock.c_str());
    return 0;
}
