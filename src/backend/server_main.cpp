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

// Lazy-loaded driver API (singleton)
CudaDriverLoader& drvLoader() {
    static CudaDriverLoader l;
    return l;
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

    cudaError_t st = cudaErrorUnknown;
    std::uint64_t aux = 0;
    std::vector<std::uint8_t> rsp_payload;

    RpcOp op = static_cast<RpcOp>(req.op);
    if (policy.force_device >= 0 && api.cudaSetDevice != nullptr && op != RpcOp::kCudaSetDevice) {
        api.cudaSetDevice(policy.force_device);
    } else if (req.device >= 0 && api.cudaSetDevice != nullptr && op != RpcOp::kCudaSetDevice) {
        api.cudaSetDevice(req.device);
    }

    if (op == RpcOp::kCudaMalloc) {
        if (payload.size() != sizeof(RpcMallocReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcMallocReq*>(payload.data());
            void* dev_ptr = nullptr;
            st = api.cudaMalloc(&dev_ptr, static_cast<std::size_t>(p->size));
            aux = reinterpret_cast<std::uint64_t>(dev_ptr);
        }
    } else if (op == RpcOp::kCudaFree) {
        if (payload.size() != sizeof(RpcFreeReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcFreeReq*>(payload.data());
            st = api.cudaFree(reinterpret_cast<void*>(p->dev_ptr));
        }
    } else if (op == RpcOp::kCudaMemcpy) {
        if (payload.size() < sizeof(RpcMemcpyReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcMemcpyReq*>(payload.data());
            std::size_t count = static_cast<std::size_t>(p->count);
            if (policy.memcpy_clamp_bytes > 0 && count > policy.memcpy_clamp_bytes) {
                count = policy.memcpy_clamp_bytes;
            }
            cudaMemcpyKind kind = static_cast<cudaMemcpyKind>(p->kind);
            if (kind == cudaMemcpyHostToDevice) {
                const std::size_t need = sizeof(RpcMemcpyReq) + count;
                if (payload.size() < need) {
                    st = cudaErrorUnknown;
                } else {
                    const void* host_data = payload.data() + sizeof(RpcMemcpyReq);
                    st = api.cudaMemcpy(reinterpret_cast<void*>(p->dst), host_data, count, cudaMemcpyHostToDevice);
                }
            } else if (kind == cudaMemcpyDeviceToHost) {
                rsp_payload.resize(count);
                st = api.cudaMemcpy(rsp_payload.data(), reinterpret_cast<const void*>(p->src), count, cudaMemcpyDeviceToHost);
            } else if (kind == cudaMemcpyDeviceToDevice) {
                st = api.cudaMemcpy(
                    reinterpret_cast<void*>(p->dst), reinterpret_cast<const void*>(p->src), count, cudaMemcpyDeviceToDevice);
            } else {
                st = notSupported();
            }
        }
    } else if (op == RpcOp::kCudaSetDevice) {
        if (payload.size() != sizeof(RpcSetDeviceReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcSetDeviceReq*>(payload.data());
            int device = (policy.force_device >= 0) ? policy.force_device : p->device;
            st = api.cudaSetDevice(device);
        }
    } else if (op == RpcOp::kCudaGetDevice) {
        int device = -1;
        st = api.cudaGetDevice(&device);
        aux = static_cast<std::uint64_t>(device);
    } else if (op == RpcOp::kCudaDeviceSynchronize) {
        st = api.cudaDeviceSynchronize();
    } else if (op == RpcOp::kCudaStreamCreate) {
        cudaStream_t stream = nullptr;
        st = api.cudaStreamCreate(&stream);
        aux = reinterpret_cast<std::uint64_t>(stream);
    } else if (op == RpcOp::kCudaStreamDestroy) {
        if (payload.size() != sizeof(RpcStreamDestroyReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcStreamDestroyReq*>(payload.data());
            st = api.cudaStreamDestroy(reinterpret_cast<cudaStream_t>(p->stream));
        }
    } else if (op == RpcOp::kCudaMemcpyAsync) {
        if (api.cudaMemcpyAsync == nullptr) {
            st = notSupported();
        } else if (payload.size() < sizeof(RpcMemcpyAsyncReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcMemcpyAsyncReq*>(payload.data());
            std::size_t count = static_cast<std::size_t>(p->count);
            if (policy.memcpy_clamp_bytes > 0 && count > policy.memcpy_clamp_bytes) {
                count = policy.memcpy_clamp_bytes;
            }
            cudaMemcpyKind kind = static_cast<cudaMemcpyKind>(p->kind);
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(p->stream);

            if (kind == cudaMemcpyHostToDevice) {
                const std::size_t need = sizeof(RpcMemcpyAsyncReq) + count;
                if (payload.size() < need) {
                    st = cudaErrorUnknown;
                } else {
                    const void* host_data = payload.data() + sizeof(RpcMemcpyAsyncReq);
                    st = api.cudaMemcpyAsync(
                        reinterpret_cast<void*>(p->dst), host_data, count, cudaMemcpyHostToDevice, stream);
                }
            } else if (kind == cudaMemcpyDeviceToHost) {
                rsp_payload.resize(count);
                st = api.cudaMemcpyAsync(
                    rsp_payload.data(), reinterpret_cast<const void*>(p->src), count, cudaMemcpyDeviceToHost, stream);
                if (st == cudaSuccess && api.cudaStreamSynchronize != nullptr) {
                    st = api.cudaStreamSynchronize(stream);
                }
            } else if (kind == cudaMemcpyDeviceToDevice) {
                st = api.cudaMemcpyAsync(
                    reinterpret_cast<void*>(p->dst), reinterpret_cast<const void*>(p->src), count, cudaMemcpyDeviceToDevice, stream);
            } else {
                st = notSupported();
            }
        }
    } else if (op == RpcOp::kCudaStreamSynchronize) {
        if (api.cudaStreamSynchronize == nullptr) {
            st = notSupported();
        } else if (payload.size() != sizeof(RpcStreamSyncReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcStreamSyncReq*>(payload.data());
            st = api.cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(p->stream));
        }
    } else if (op == RpcOp::kCudaEventCreate) {
        if (api.cudaEventCreateWithFlags == nullptr) {
            st = notSupported();
        } else if (payload.size() != sizeof(RpcEventCreateReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcEventCreateReq*>(payload.data());
            cudaEvent_t event = nullptr;
            st = api.cudaEventCreateWithFlags(&event, p->flags);
            aux = reinterpret_cast<std::uint64_t>(event);
        }
    } else if (op == RpcOp::kCudaEventDestroy) {
        if (api.cudaEventDestroy == nullptr) {
            st = notSupported();
        } else if (payload.size() != sizeof(RpcEventDestroyReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcEventDestroyReq*>(payload.data());
            st = api.cudaEventDestroy(reinterpret_cast<cudaEvent_t>(p->event));
        }
    } else if (op == RpcOp::kCudaEventRecord) {
        if (api.cudaEventRecord == nullptr) {
            st = notSupported();
        } else if (payload.size() != sizeof(RpcEventRecordReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcEventRecordReq*>(payload.data());
            st = api.cudaEventRecord(reinterpret_cast<cudaEvent_t>(p->event), reinterpret_cast<cudaStream_t>(p->stream));
        }
    } else if (op == RpcOp::kCudaEventSynchronize) {
        if (api.cudaEventSynchronize == nullptr) {
            st = notSupported();
        } else if (payload.size() != sizeof(RpcEventSynchronizeReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcEventSynchronizeReq*>(payload.data());
            st = api.cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(p->event));
        }
    } else if (op == RpcOp::kCudaMemset) {
        if (api.cudaMemset == nullptr) {
            st = notSupported();
        } else if (payload.size() != sizeof(RpcMemsetReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcMemsetReq*>(payload.data());
            st = api.cudaMemset(reinterpret_cast<void*>(p->dst), p->value, static_cast<std::size_t>(p->count));
        }
    } else if (op == RpcOp::kCudaMemsetAsync) {
        if (api.cudaMemsetAsync == nullptr) {
            st = notSupported();
        } else if (payload.size() != sizeof(RpcMemsetAsyncReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcMemsetAsyncReq*>(payload.data());
            st = api.cudaMemsetAsync(
                reinterpret_cast<void*>(p->dst), p->value, static_cast<std::size_t>(p->count), reinterpret_cast<cudaStream_t>(p->stream));
        }
    } else if (op == RpcOp::kCudaGetDeviceCount) {
        if (api.cudaGetDeviceCount == nullptr) {
            st = notSupported();
        } else {
            int count = 0;
            st = api.cudaGetDeviceCount(&count);
            aux = static_cast<std::uint64_t>(count);
        }
    } else if (op == RpcOp::kCudaMemGetInfo) {
        if (api.cudaMemGetInfo == nullptr) {
            st = notSupported();
        } else {
            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            st = api.cudaMemGetInfo(&free_bytes, &total_bytes);
            if (st == cudaSuccess) {
                RpcMemGetInfoRsp info{};
                info.free_bytes = static_cast<std::uint64_t>(free_bytes);
                info.total_bytes = static_cast<std::uint64_t>(total_bytes);
                rsp_payload.resize(sizeof(info));
                std::memcpy(rsp_payload.data(), &info, sizeof(info));
            }
        }
    } else if (op == RpcOp::kCudaStreamQuery) {
        if (api.cudaStreamQuery == nullptr) {
            st = notSupported();
        } else if (payload.size() != sizeof(RpcStreamQueryReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcStreamQueryReq*>(payload.data());
            st = api.cudaStreamQuery(reinterpret_cast<cudaStream_t>(p->stream));
        }
    } else if (op == RpcOp::kCudaEventQuery) {
        if (api.cudaEventQuery == nullptr) {
            st = notSupported();
        } else if (payload.size() != sizeof(RpcEventQueryReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcEventQueryReq*>(payload.data());
            st = api.cudaEventQuery(reinterpret_cast<cudaEvent_t>(p->event));
        }
    } else if (op == RpcOp::kCudaStreamWaitEvent) {
        if (api.cudaStreamWaitEvent == nullptr) {
            st = notSupported();
        } else if (payload.size() != sizeof(RpcStreamWaitEventReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcStreamWaitEventReq*>(payload.data());
            st = api.cudaStreamWaitEvent(
                reinterpret_cast<cudaStream_t>(p->stream), reinterpret_cast<cudaEvent_t>(p->event), p->flags);
        }
    } else if (op == RpcOp::kCudaSetDeviceFlags) {
        if (api.cudaSetDeviceFlags == nullptr) {
            st = notSupported();
        } else if (payload.size() != sizeof(RpcSetDeviceFlagsReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcSetDeviceFlagsReq*>(payload.data());
            st = api.cudaSetDeviceFlags(p->flags);
        }
    } else if (op == RpcOp::kCudaDeviceReset) {
        if (api.cudaDeviceReset == nullptr) {
            st = notSupported();
        } else {
            st = api.cudaDeviceReset();
        }
    } else if (op == RpcOp::kCudaRuntimeGetVersion) {
        if (api.cudaRuntimeGetVersion == nullptr) {
            st = notSupported();
        } else {
            int version = 0;
            st = api.cudaRuntimeGetVersion(&version);
            aux = static_cast<std::uint64_t>(version);
        }
    } else if (op == RpcOp::kCudaDriverGetVersion) {
        if (api.cudaDriverGetVersion == nullptr) {
            st = notSupported();
        } else {
            int version = 0;
            st = api.cudaDriverGetVersion(&version);
            aux = static_cast<std::uint64_t>(version);
        }
    } else if (op == RpcOp::kCudaDeviceGetAttribute) {
        if (api.cudaDeviceGetAttribute == nullptr) {
            st = notSupported();
        } else if (payload.size() != sizeof(RpcDeviceGetAttributeReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* p = reinterpret_cast<const RpcDeviceGetAttributeReq*>(payload.data());
            int value = 0;
            st = api.cudaDeviceGetAttribute(&value, p->attr, p->device);
            aux = static_cast<std::uint64_t>(value);
        }
    } else if (op == RpcOp::kCudaGetDeviceFlags) {
        if (api.cudaGetDeviceFlags == nullptr) {
            st = notSupported();
        } else {
            unsigned int flags = 0;
            st = api.cudaGetDeviceFlags(&flags);
            aux = static_cast<std::uint64_t>(flags);
        }
    } else if (op == RpcOp::kCudaLaunchKernel) {
        st = notSupported();
    } else {
        st = cudaErrorUnknown;  // runtime ops done
    }

    // ── CUDA Driver API ops ────────────────────────────────────────────────
    RpcDrvOp dop = static_cast<RpcDrvOp>(req.op);

    if (dop == RpcDrvOp::kCuModuleLoadData) {
        // payload = raw fatbin bytes
        if (!drvLoader().ensureLoaded()) {
            st = notSupported();
        } else if (payload.empty()) {
            st = cudaErrorUnknown;
        } else {
            CUmodule mod = nullptr;
            CUresult cr = drvLoader().api().cuModuleLoadData(&mod, payload.data());
            if (cr == CUDA_SUCCESS) {
                uint64_t mid = g_next_module_id.fetch_add(1);
                {
                    std::lock_guard<std::mutex> lk(g_kernel_mutex);
                    g_modules[mid] = mod;
                    // Parse PTX param info for all kernels in this module
                    auto ki_list = parseFatbin(payload.data(), payload.size());
                    for (auto& ki : ki_list) {
                        g_kernel_params[mid][ki.mangled_name] =
                            std::move(ki.params);
                    }
                }
                aux = mid;
                st = cudaSuccess;
            } else {
                st = cudaErrorUnknown;
            }
        }
    } else if (dop == RpcDrvOp::kCuModuleGetFunction) {
        if (!drvLoader().ensureLoaded()) {
            st = notSupported();
        } else if (payload.size() < sizeof(RpcCuModuleGetFunctionReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* req2 =
                reinterpret_cast<const RpcCuModuleGetFunctionReq*>(payload.data());
            const char* fname = reinterpret_cast<const char*>(req2 + 1);
            uint64_t mid = req2->module_id;
            CUmodule mod = nullptr;
            {
                std::lock_guard<std::mutex> lk(g_kernel_mutex);
                auto it = g_modules.find(mid);
                if (it != g_modules.end()) mod = it->second;
            }
            if (!mod) {
                st = cudaErrorUnknown;
            } else {
                CUfunction func = nullptr;
                CUresult cr = drvLoader().api().cuModuleGetFunction(&func, mod, fname);
                if (cr == CUDA_SUCCESS && func) {
                    uint64_t fid = g_next_func_id.fetch_add(1);
                    {
                        std::lock_guard<std::mutex> lk(g_kernel_mutex);
                        g_functions[fid] = func;
                    }
                    aux = fid;
                    st = cudaSuccess;

                    // Build param info response
                    std::vector<RpcParamInfo> param_wire;
                    uint32_t total_param_bytes = 0;
                    {
                        std::lock_guard<std::mutex> lk(g_kernel_mutex);
                        auto mit = g_kernel_params.find(mid);
                        if (mit != g_kernel_params.end()) {
                            auto fit = mit->second.find(fname);
                            if (fit != mit->second.end()) {
                                for (const auto& p : fit->second) {
                                    RpcParamInfo rp{};
                                    rp.size      = p.size;
                                    rp.alignment = p.alignment;
                                    param_wire.push_back(rp);
                                }
                                total_param_bytes =
                                    computeParamBufSize(fit->second);
                            }
                        }
                    }
                    // Also query total param size from driver if available
                    if (param_wire.empty() &&
                        drvLoader().api().cuFuncGetAttribute != nullptr) {
                        int psz = 0;
                        drvLoader().api().cuFuncGetAttribute(
                            &psz, CU_FUNC_ATTR_PARAM_SIZE_BYTES, func);
                        total_param_bytes = static_cast<uint32_t>(psz);
                    }

                    RpcCuModuleGetFunctionRsp rsp2{};
                    rsp2.param_count       = static_cast<uint32_t>(param_wire.size());
                    rsp2.total_param_bytes = total_param_bytes;
                    rsp_payload.resize(sizeof(rsp2) +
                                       param_wire.size() * sizeof(RpcParamInfo));
                    std::memcpy(rsp_payload.data(), &rsp2, sizeof(rsp2));
                    std::memcpy(rsp_payload.data() + sizeof(rsp2),
                                param_wire.data(),
                                param_wire.size() * sizeof(RpcParamInfo));
                } else {
                    st = cudaErrorUnknown;
                }
            }
        }
    } else if (dop == RpcDrvOp::kCuLaunchKernel) {
        if (!drvLoader().ensureLoaded()) {
            st = notSupported();
        } else if (payload.size() < sizeof(RpcCuLaunchKernelReq)) {
            st = cudaErrorUnknown;
        } else {
            const auto* lreq =
                reinterpret_cast<const RpcCuLaunchKernelReq*>(payload.data());
            CUfunction func = nullptr;
            {
                std::lock_guard<std::mutex> lk(g_kernel_mutex);
                auto it = g_functions.find(lreq->func_id);
                if (it != g_functions.end()) func = it->second;
            }
            if (!func) {
                st = cudaErrorUnknown;
            } else {
                const uint8_t* arg_data =
                    payload.data() + sizeof(RpcCuLaunchKernelReq);
                size_t arg_size = static_cast<size_t>(lreq->arg_buf_size);
                // Verify we have the arg data
                if (payload.size() < sizeof(RpcCuLaunchKernelReq) + arg_size) {
                    st = cudaErrorUnknown;
                } else {
                    // Use CU_LAUNCH_PARAM_BUFFER_POINTER so we can pass the
                    // pre-packed flat buffer directly.
                    void* buf_ptr = const_cast<uint8_t*>(arg_data);
                    void* extra[] = {
                        CU_LAUNCH_PARAM_BUFFER_POINTER, buf_ptr,
                        CU_LAUNCH_PARAM_BUFFER_SIZE,    &arg_size,
                        CU_LAUNCH_PARAM_END
                    };
                    CUstream_drv cu_stream =
                        reinterpret_cast<CUstream_drv>(lreq->stream);
                    CUresult cr = drvLoader().api().cuLaunchKernel(
                        func,
                        lreq->grid_x,  lreq->grid_y,  lreq->grid_z,
                        lreq->block_x, lreq->block_y, lreq->block_z,
                        lreq->shared_mem_bytes, cu_stream,
                        nullptr, extra);
                    st = (cr == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
                    if (cr == CUDA_SUCCESS && api.cudaDeviceSynchronize) {
                        api.cudaDeviceSynchronize();
                    }
                }
            }
        }
    } else if (dop == RpcDrvOp::kCuModuleUnload) {
        if (payload.size() >= sizeof(uint64_t)) {
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
            if (mod && drvLoader().ensureLoaded() &&
                drvLoader().api().cuModuleUnload) {
                drvLoader().api().cuModuleUnload(mod);
            }
            st = cudaSuccess;
        }
    } else if (dop == RpcDrvOp::kCuMemAlloc) {
        // Re-use cudaMalloc path
        if (payload.size() == sizeof(RpcMallocReq)) {
            const auto* p = reinterpret_cast<const RpcMallocReq*>(payload.data());
            void* dev_ptr = nullptr;
            st = api.cudaMalloc(&dev_ptr, static_cast<std::size_t>(p->size));
            aux = reinterpret_cast<std::uint64_t>(dev_ptr);
        }
    } else if (dop == RpcDrvOp::kCuMemFree) {
        if (payload.size() == sizeof(RpcFreeReq)) {
            const auto* p = reinterpret_cast<const RpcFreeReq*>(payload.data());
            st = api.cudaFree(reinterpret_cast<void*>(p->dev_ptr));
        }
    } else if (dop == RpcDrvOp::kCuMemcpyHtoD) {
        if (payload.size() >= sizeof(RpcMemcpyReq)) {
            const auto* p = reinterpret_cast<const RpcMemcpyReq*>(payload.data());
            std::size_t count = static_cast<std::size_t>(p->count);
            const std::size_t need = sizeof(RpcMemcpyReq) + count;
            if (payload.size() >= need) {
                st = api.cudaMemcpy(
                    reinterpret_cast<void*>(p->dst),
                    payload.data() + sizeof(RpcMemcpyReq),
                    count, cudaMemcpyHostToDevice);
            } else { st = cudaErrorUnknown; }
        }
    } else if (dop == RpcDrvOp::kCuMemcpyDtoH) {
        if (payload.size() >= sizeof(RpcMemcpyReq)) {
            const auto* p = reinterpret_cast<const RpcMemcpyReq*>(payload.data());
            std::size_t count = static_cast<std::size_t>(p->count);
            rsp_payload.resize(count);
            st = api.cudaMemcpy(
                rsp_payload.data(),
                reinterpret_cast<const void*>(p->src),
                count, cudaMemcpyDeviceToHost);
        }
    } else if (dop == RpcDrvOp::kCuMemcpyDtoD) {
        if (payload.size() >= sizeof(RpcMemcpyReq)) {
            const auto* p = reinterpret_cast<const RpcMemcpyReq*>(payload.data());
            st = api.cudaMemcpy(
                reinterpret_cast<void*>(p->dst),
                reinterpret_cast<const void*>(p->src),
                static_cast<std::size_t>(p->count), cudaMemcpyDeviceToDevice);
        }
    } else if (dop == RpcDrvOp::kCuCtxSynchronize) {
        st = api.cudaDeviceSynchronize();
    }

    RpcResponseHeader rsp{};
    rsp.magic = kRpcMagic;
    rsp.version = kRpcVersion;
    rsp.status = st;
    rsp.aux_u64 = aux;
    rsp.payload_size = rsp_payload.size();

    logReq(policy, req, st);

    if (!writeAll(fd, &rsp, sizeof(rsp))) {
        return false;
    }
    if (!rsp_payload.empty()) {
        if (!writeAll(fd, rsp_payload.data(), rsp_payload.size())) {
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
