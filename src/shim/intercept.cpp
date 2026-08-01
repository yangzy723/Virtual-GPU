// intercept.cpp — libcuda.so shim (LD_PRELOAD)
//
// Intercepts CUDA Driver API calls and delegates admission control to either
// the in-process scheduler or the daemon transport. Light ops pass through.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vgpu/cuda_abi.h"
#include "vgpu/config.h"
#include "vgpu/protocol.h"
#include "scheduler_backend.h"

// Trace logging — controlled by VGPU_TRACE env var (default: on)
static bool g_trace = [] {
    return vgpu::config::getBool("VGPU_TRACE", true);
}();

#define TRACE(fmt, ...) do { \
    if (g_trace) std::fprintf(stderr, "[vGPU trace] " fmt "\n", ##__VA_ARGS__); \
} while(0)

namespace vgpu {
namespace {

// ── Real CUDA driver loader ──────────────────────────────────────────────

void* g_real_cuda = nullptr;
std::mutex g_cuda_mu;
thread_local bool g_in_hook = false;

using DlsymFn = void*(*)(void*, const char*);
DlsymFn g_real_dlsym = nullptr;

static DlsymFn resolveRealDlsym() {
    auto fn = reinterpret_cast<DlsymFn>(dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34"));
    if (!fn) fn = reinterpret_cast<DlsymFn>(dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.17"));
    if (!fn) fn = reinterpret_cast<DlsymFn>(dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5"));
    return fn;
}

__attribute__((constructor))
static void initDlsymHook() {
    g_real_dlsym = resolveRealDlsym();
}

void* realCudaHandle() {
    if (g_real_cuda) return g_real_cuda;

    std::lock_guard<std::mutex> lock(g_cuda_mu);
    if (g_real_cuda) return g_real_cuda;

    static const char* paths[] = {
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcuda.so.1",
        "/usr/local/cuda/lib64/libcuda.so.1",
        nullptr
    };
    for (const char** p = paths; *p; ++p) {
        g_real_cuda = dlopen(*p, RTLD_NOW | RTLD_LOCAL);
        if (g_real_cuda) break;
    }

    return g_real_cuda;
}

template<typename T>
T realSym(const char* name) {
    void* h = realCudaHandle();
    if (!h) return nullptr;
    if (!g_real_dlsym) g_real_dlsym = resolveRealDlsym();
    if (!g_real_dlsym) return nullptr;

    g_in_hook = true;
    void* sym = g_real_dlsym(h, name);
    g_in_hook = false;
    return reinterpret_cast<T>(sym);
}

// ── Shim process state ───────────────────────────────────────────────────

std::mutex g_alloc_mu;
std::unordered_map<uint64_t, size_t> g_alloc_sizes;

struct PendingCompletion {
    CUevent event = nullptr;
    SchedOp op = SchedOp::NONE;
    uint64_t value = 0;
    int device = 0;
};

std::mutex g_completion_mu;
std::vector<PendingCompletion> g_pending_completions;
std::atomic<bool> g_completion_stop{false};
bool g_completion_worker_started = false;

uint64_t checkedMemcpyBytes(size_t width, size_t height, size_t depth) {
    const auto limit = std::numeric_limits<uint64_t>::max();
    uint64_t total = static_cast<uint64_t>(width);

    if (height != 0 && total > limit / static_cast<uint64_t>(height)) {
        return limit;
    }
    total *= static_cast<uint64_t>(height);

    if (depth != 0 && total > limit / static_cast<uint64_t>(depth)) {
        return limit;
    }
    total *= static_cast<uint64_t>(depth);

    return total;
}

uint64_t memcpy2DBytes(const CUDA_MEMCPY2D* copy) {
    if (!copy) return 0;
    return checkedMemcpyBytes(copy->WidthInBytes, copy->Height, 1);
}

uint64_t memcpy3DBytes(const CUDA_MEMCPY3D* copy) {
    if (!copy) return 0;
    return checkedMemcpyBytes(copy->WidthInBytes, copy->Height, copy->Depth);
}

uint64_t memcpy3DPeerBytes(const CUDA_MEMCPY3D_PEER* copy) {
    if (!copy) return 0;
    return checkedMemcpyBytes(copy->WidthInBytes, copy->Height, copy->Depth);
}

uint64_t memcpyBatchBytes(const size_t* sizes, size_t count) {
    if (!sizes) return 0;
    uint64_t total = 0;
    const auto limit = std::numeric_limits<uint64_t>::max();
    for (size_t index = 0; index < count; ++index) {
        uint64_t size = static_cast<uint64_t>(sizes[index]);
        if (total > limit - size) return limit;
        total += size;
    }
    return total;
}

uint64_t checkedElementBytes(size_t elements, size_t element_size) {
    const auto limit = std::numeric_limits<uint64_t>::max();
    if (element_size != 0 && elements > limit / element_size) return limit;
    return static_cast<uint64_t>(elements * element_size);
}

static int completionPollUs() {
    static const int poll_us = vgpu::config::getInt(
        "GPU_SCHEDULER_COMPLETION_POLL_US", 100, 1, 1000000);
    return poll_us;
}

// ── Scheduling helpers ───────────────────────────────────────────────────

thread_local int g_current_device = 0;

bool schedRequest(SchedOp op, uint64_t value, int device) {
    return schedulerRequest(op, value, device);
}

void schedReport(SchedOp op, uint64_t value, int device) {
    schedulerReport(op, value, device);
}

void finishAllocationAdmission(CUresult result, CUdeviceptr* dptr,
                               size_t bytesize, int device) {
    if (result == CUDA_SUCCESS && dptr) {
        std::lock_guard<std::mutex> lock(g_alloc_mu);
        g_alloc_sizes[*dptr] = bytesize;
        return;
    }
    // Admission is accounted before the real driver call. Roll it back when
    // the driver rejects the allocation so quotas cannot leak permanently.
    schedReport(SchedOp::FREE, bytesize, device);
}

void completionWorker() {
    const auto query = realSym<CUresult(*)(CUevent)>("cuEventQuery");
    auto destroy = realSym<CUresult(*)(CUevent)>("cuEventDestroy_v2");
    if (!destroy) destroy = realSym<CUresult(*)(CUevent)>("cuEventDestroy");

    while (!g_completion_stop.load(std::memory_order_acquire)) {
        std::vector<PendingCompletion> completed;

        {
            std::lock_guard<std::mutex> lock(g_completion_mu);
            if (query) {
                size_t write_index = 0;
                for (size_t read_index = 0; read_index < g_pending_completions.size(); ++read_index) {
                    auto item = g_pending_completions[read_index];
                    CUresult rc = query(item.event);
                    if (rc == CUDA_ERROR_NOT_READY) {
                        g_pending_completions[write_index++] = item;
                    } else {
                        completed.push_back(item);
                    }
                }
                g_pending_completions.resize(write_index);
            }
        }

        for (const auto& item : completed) {
            if (destroy) destroy(item.event);
            schedReport(item.op, item.value, item.device);
        }

        std::this_thread::sleep_for(std::chrono::microseconds(completionPollUs()));
    }
}

void startCompletionWorkerIfNeeded() {
    std::lock_guard<std::mutex> lock(g_completion_mu);
    if (g_completion_worker_started) return;
    g_completion_worker_started = true;
    std::thread(completionWorker).detach();
}

bool enqueueStreamCompletion(SchedOp op, uint64_t value, int device, CUstream stream) {
    auto create = realSym<CUresult(*)(CUevent*, unsigned int)>("cuEventCreate");
    auto record = realSym<CUresult(*)(CUevent, CUstream)>("cuEventRecord");
    auto destroy = realSym<CUresult(*)(CUevent)>("cuEventDestroy_v2");
    if (!destroy) destroy = realSym<CUresult(*)(CUevent)>("cuEventDestroy");
    if (!create || !record || !destroy) return false;

    CUevent event = nullptr;
    if (create(&event, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS || !event) {
        return false;
    }
    if (record(event, stream) != CUDA_SUCCESS) {
        destroy(event);
        return false;
    }

    startCompletionWorkerIfNeeded();
    {
        std::lock_guard<std::mutex> lock(g_completion_mu);
        g_pending_completions.push_back({event, op, value, device});
    }
    return true;
}

void reportAfterStreamCompletionOrNow(SchedOp op, uint64_t value, int device, CUstream stream) {
    if (!enqueueStreamCompletion(op, value, device, stream)) {
        schedReport(op, value, device);
    }
}

__attribute__((destructor))
static void stopCompletionWorker() {
    g_completion_stop.store(true, std::memory_order_release);
}

template <typename Callable>
CUresult runScheduledCall(SchedOp request_op,
                         uint64_t request_value,
                         SchedOp complete_op,
                         uint64_t complete_value,
                         int device,
                         CUresult reject_code,
                         Callable&& callable) {
    if (!schedRequest(request_op, request_value, device)) {
        return reject_code;
    }

    CUresult err = std::forward<Callable>(callable)();
    schedReport(complete_op, complete_value, device);
    return err;
}

template <typename Callable>
CUresult runScheduledAsyncCall(SchedOp request_op,
                              uint64_t request_value,
                              SchedOp complete_op,
                              uint64_t complete_value,
                              int device,
                              CUstream stream,
                              CUresult reject_code,
                              Callable&& callable) {
    if (!schedRequest(request_op, request_value, device)) {
        return reject_code;
    }

    CUresult err = std::forward<Callable>(callable)();
    if (err == CUDA_SUCCESS) {
        reportAfterStreamCompletionOrNow(complete_op, complete_value, device, stream);
    } else {
        schedReport(complete_op, complete_value, device);
    }
    return err;
}

template <typename ValueT>
CUresult runMemset1D(const char* real_v2, const char* real_legacy,
                    CUdeviceptr dst, ValueT value, size_t count,
                    size_t element_size) {
    using Fn = CUresult(*)(CUdeviceptr, ValueT, size_t);
    auto real = realSym<Fn>(real_v2);
    if (!real) real = realSym<Fn>(real_legacy);
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    const uint64_t bytesize = checkedElementBytes(count, element_size);
    return runScheduledCall(SchedOp::MEMCPY_REQUEST, bytesize,
                            SchedOp::MEMCPY_COMPLETE, bytesize,
                            g_current_device,
                            CUDA_ERROR_NOT_SUPPORTED,
                            [&] { return real(dst, value, count); });
}

template <typename ValueT>
CUresult runMemset1DAsync(const char* real_name,
                         CUdeviceptr dst, ValueT value, size_t count,
                         size_t element_size, CUstream stream) {
    using Fn = CUresult(*)(CUdeviceptr, ValueT, size_t, CUstream);
    auto real = realSym<Fn>(real_name);
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    const uint64_t bytesize = checkedElementBytes(count, element_size);
    return runScheduledAsyncCall(SchedOp::MEMCPY_REQUEST, bytesize,
                                 SchedOp::MEMCPY_COMPLETE, bytesize,
                                 g_current_device, stream,
                                 CUDA_ERROR_NOT_SUPPORTED,
                                 [&] { return real(dst, value, count, stream); });
}

template <typename ValueT>
CUresult runMemset2D(const char* real_v2, const char* real_legacy,
                    CUdeviceptr dst, size_t pitch, ValueT value,
                    size_t width, size_t height, size_t element_size) {
    using Fn = CUresult(*)(CUdeviceptr, size_t, ValueT, size_t, size_t);
    auto real = realSym<Fn>(real_v2);
    if (!real) real = realSym<Fn>(real_legacy);
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    const uint64_t bytesize = checkedMemcpyBytes(width, height, element_size);
    return runScheduledCall(SchedOp::MEMCPY_REQUEST, bytesize,
                            SchedOp::MEMCPY_COMPLETE, bytesize,
                            g_current_device,
                            CUDA_ERROR_NOT_SUPPORTED,
                            [&] { return real(dst, pitch, value, width, height); });
}

template <typename ValueT>
CUresult runMemset2DAsync(const char* real_name,
                         CUdeviceptr dst, size_t pitch, ValueT value,
                         size_t width, size_t height, size_t element_size,
                         CUstream stream) {
    using Fn = CUresult(*)(CUdeviceptr, size_t, ValueT, size_t, size_t, CUstream);
    auto real = realSym<Fn>(real_name);
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    const uint64_t bytesize = checkedMemcpyBytes(width, height, element_size);
    return runScheduledAsyncCall(SchedOp::MEMCPY_REQUEST, bytesize,
                                 SchedOp::MEMCPY_COMPLETE, bytesize,
                                 g_current_device, stream,
                                 CUDA_ERROR_NOT_SUPPORTED,
                                 [&] { return real(dst, pitch, value, width, height, stream); });
}

// ── Symbol classification ────────────────────────────────────────────────

bool isInterceptedSymbol(const char* symbol) {
    using namespace std::literals;
    static const std::unordered_set<std::string_view> kIntercepted = {
        "cuMemAlloc"sv, "cuMemAlloc_v2"sv,
        "cuMemAllocAsync"sv, "cuMemAllocFromPoolAsync"sv,
        "cuMemFree"sv, "cuMemFree_v2"sv,
        "cuMemFreeAsync"sv,
        "cuMemcpy"sv, "cuMemcpyAsync"sv,
        "cuMemcpyPeer"sv, "cuMemcpyPeerAsync"sv,
        "cuMemcpyHtoD"sv, "cuMemcpyHtoD_v2"sv,
        "cuMemcpyDtoH"sv, "cuMemcpyDtoH_v2"sv,
        "cuMemcpyDtoD"sv, "cuMemcpyDtoD_v2"sv,
        "cuMemcpyHtoDAsync"sv, "cuMemcpyHtoDAsync_v2"sv,
        "cuMemcpyDtoHAsync"sv, "cuMemcpyDtoHAsync_v2"sv,
        "cuMemcpyDtoDAsync"sv, "cuMemcpyDtoDAsync_v2"sv,
        "cuMemcpy2D"sv, "cuMemcpy2D_v2"sv,
        "cuMemcpy2DUnaligned"sv, "cuMemcpy2DUnaligned_v2"sv,
        "cuMemcpy2DAsync"sv, "cuMemcpy2DAsync_v2"sv,
        "cuMemcpy3D"sv, "cuMemcpy3D_v2"sv,
        "cuMemcpy3DAsync"sv, "cuMemcpy3DAsync_v2"sv,
        "cuMemcpy3DPeer"sv, "cuMemcpy3DPeerAsync"sv,
        "cuMemcpyBatchAsync_v2"sv,
        "cuMemsetD8"sv, "cuMemsetD8_v2"sv,
        "cuMemsetD16"sv, "cuMemsetD16_v2"sv,
        "cuMemsetD32"sv, "cuMemsetD32_v2"sv,
        "cuMemsetD2D8"sv, "cuMemsetD2D8_v2"sv,
        "cuMemsetD2D16"sv, "cuMemsetD2D16_v2"sv,
        "cuMemsetD2D32"sv, "cuMemsetD2D32_v2"sv,
        "cuMemsetD8Async"sv, "cuMemsetD16Async"sv, "cuMemsetD32Async"sv,
        "cuMemsetD2D8Async"sv, "cuMemsetD2D16Async"sv, "cuMemsetD2D32Async"sv,
        "cuLaunchKernel"sv,
        "cuLaunchKernelEx"sv, "cuLaunchKernelExC"sv,
        "cuStreamCreate"sv, "cuStreamDestroy"sv,
        "cuStreamSynchronize"sv, "cuStreamQuery"sv,
        "cuGetProcAddress"sv, "cuGetProcAddress_v2"sv,
    };
    return kIntercepted.count(symbol) > 0;
}

bool shouldRouteProcAddressToShim(const char* symbol, int cudaVersion) {
    if (!isInterceptedSymbol(symbol)) return false;

    using namespace std::literals;
    static const std::unordered_set<std::string_view> kLegacyStructMemcpy = {
        "cuMemcpy2D"sv,
        "cuMemcpy2DUnaligned"sv,
        "cuMemcpy2DAsync"sv,
        "cuMemcpy3D"sv,
        "cuMemcpy3DAsync"sv,
    };
    static const std::unordered_set<std::string_view> kLegacyV1PointerSymbols = {
        "cuMemsetD8"sv,
        "cuMemsetD16"sv,
        "cuMemsetD32"sv,
        "cuMemsetD2D8"sv,
        "cuMemsetD2D16"sv,
        "cuMemsetD2D32"sv,
    };

    if (cudaVersion > 0 && cudaVersion < 7000 && kLegacyStructMemcpy.count(symbol) > 0) {
        return false;
    }
    if (cudaVersion > 0 && cudaVersion < 3020 && kLegacyV1PointerSymbols.count(symbol) > 0) {
        return false;
    }
    return true;
}

static bool envFlag(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] == '1';
}

}  // namespace
}  // namespace vgpu

extern "C"
void* dlsym(void* handle, const char* symbol) {
    if (!vgpu::g_real_dlsym) vgpu::g_real_dlsym = vgpu::resolveRealDlsym();
    if (!vgpu::g_real_dlsym) return nullptr;

    if (vgpu::g_in_hook) {
        return vgpu::g_real_dlsym(handle, symbol);
    }

    if (std::strcmp(symbol, "cuGetProcAddress") == 0 ||
                   std::strcmp(symbol, "cuGetProcAddress_v2") == 0) {
        void* fn = vgpu::g_real_dlsym(RTLD_DEFAULT, symbol);
        if (fn) return fn;
    }

    return vgpu::g_real_dlsym(handle, symbol);
}

// ── Driver API exports ───────────────────────────────────────────────────

extern "C" {

CUresult cuGetProcAddress(const char* symbol, void** pfn,
                          int cudaVersion, unsigned long long flags);
CUresult cuGetProcAddress_v2(const char* symbol, void** pfn,
                             int cudaVersion, unsigned long long flags,
                             unsigned long long* symbolStatus);

CUresult cuInit(unsigned int flags) {
    TRACE("cuInit(flags=%u)", flags);
    auto real = vgpu::realSym<CUresult(*)(unsigned int)>("cuInit");
    if (!real) return CUDA_SUCCESS;
    return real(flags);
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    TRACE("cuDeviceGet(ordinal=%d)", ordinal);
    auto real = vgpu::realSym<CUresult(*)(CUdevice*, int)>("cuDeviceGet");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(device, ordinal);
}

CUresult cuDeviceGetCount(int* count) {
    TRACE("cuDeviceGetCount()");
    auto real = vgpu::realSym<CUresult(*)(int*)>("cuDeviceGetCount");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(count);
}

CUresult cuDeviceGetAttribute(int* value, int attrib, CUdevice dev) {
    TRACE("cuDeviceGetAttribute(attrib=%d, dev=%d)", attrib, dev);
    auto real = vgpu::realSym<CUresult(*)(int*, int, CUdevice)>("cuDeviceGetAttribute");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(value, attrib, dev);
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
    TRACE("cuDeviceGetName(dev=%d)", dev);
    auto real = vgpu::realSym<CUresult(*)(char*, int, CUdevice)>("cuDeviceGetName");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(name, len, dev);
}

CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
    TRACE("cuDeviceTotalMem(dev=%d)", dev);
    auto real = vgpu::realSym<CUresult(*)(size_t*, CUdevice)>("cuDeviceTotalMem_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(size_t*, CUdevice)>("cuDeviceTotalMem");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(bytes, dev);
}

CUresult cuDeviceGetUuid(void* uuid, CUdevice dev) {
    TRACE("cuDeviceGetUuid(dev=%d)", dev);
    auto real = vgpu::realSym<CUresult(*)(void*, CUdevice)>("cuDeviceGetUuid");
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;
    return real(uuid, dev);
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) {
    TRACE("cuDevicePrimaryCtxGetState(dev=%d)", dev);
    auto real = vgpu::realSym<CUresult(*)(CUdevice, unsigned int*, int*)>("cuDevicePrimaryCtxGetState");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(dev, flags, active);
}

CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
    TRACE("cuDevicePrimaryCtxRetain(dev=%d)", dev);
    auto real = vgpu::realSym<CUresult(*)(CUcontext*, CUdevice)>("cuDevicePrimaryCtxRetain");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(pctx, dev);
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    TRACE("cuDevicePrimaryCtxRelease(dev=%d)", dev);
    auto real = vgpu::realSym<CUresult(*)(CUdevice)>("cuDevicePrimaryCtxRelease_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(CUdevice)>("cuDevicePrimaryCtxRelease");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(dev);
}

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    TRACE("cuCtxCreate(flags=%u, dev=%d)", flags, dev);
    auto real = vgpu::realSym<CUresult(*)(CUcontext*, unsigned int, CUdevice)>("cuCtxCreate_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(CUcontext*, unsigned int, CUdevice)>("cuCtxCreate");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(pctx, flags, dev);
}

CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    return cuCtxCreate(pctx, flags, dev);
}

CUresult cuCtxDestroy(CUcontext ctx) {
    TRACE("cuCtxDestroy(ctx=%p)", ctx);
    auto real = vgpu::realSym<CUresult(*)(CUcontext)>("cuCtxDestroy_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(CUcontext)>("cuCtxDestroy");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(ctx);
}

CUresult cuCtxDestroy_v2(CUcontext ctx) { return cuCtxDestroy(ctx); }

CUresult cuCtxGetCurrent(CUcontext* pctx) {
    TRACE("cuCtxGetCurrent()");
    auto real = vgpu::realSym<CUresult(*)(CUcontext*)>("cuCtxGetCurrent");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(pctx);
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    TRACE("cuCtxSetCurrent(ctx=%p)", ctx);
    auto real = vgpu::realSym<CUresult(*)(CUcontext)>("cuCtxSetCurrent");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    CUresult err = real(ctx);
    if (err == CUDA_SUCCESS) {
        CUdevice dev = 0;
        auto get_dev = vgpu::realSym<CUresult(*)(CUdevice*)>("cuCtxGetDevice");
        if (get_dev && get_dev(&dev) == CUDA_SUCCESS) {
            vgpu::g_current_device = dev;
        }
    }
    return err;
}

CUresult cuCtxSynchronize() {
    TRACE("cuCtxSynchronize()");
    auto real = vgpu::realSym<CUresult(*)()>("cuCtxSynchronize");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real();
}

CUresult cuCtxGetDevice(CUdevice* device) {
    TRACE("cuCtxGetDevice()");
    auto real = vgpu::realSym<CUresult(*)(CUdevice*)>("cuCtxGetDevice");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    CUresult err = real(device);
    if (err == CUDA_SUCCESS && device) {
        vgpu::g_current_device = *device;
    }
    return err;
}

CUresult cuModuleLoad(CUmodule* module, const char* fname) {
    TRACE("cuModuleLoad(fname=%s)", fname ? fname : "null");
    auto real = vgpu::realSym<CUresult(*)(CUmodule*, const char*)>("cuModuleLoad");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(module, fname);
}

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    TRACE("cuModuleLoadData(image=%p)", image);
    auto real = vgpu::realSym<CUresult(*)(CUmodule*, const void*)>("cuModuleLoadData");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(module, image);
}

CUresult cuModuleLoadDataEx(CUmodule* module, const void* image,
                            unsigned int numOptions, void* options,
                            void* optionValues) {
    TRACE("cuModuleLoadDataEx(image=%p, numOptions=%u)", image, numOptions);
    auto real = vgpu::realSym<CUresult(*)(CUmodule*, const void*, unsigned int, void*, void*)>(
        "cuModuleLoadDataEx");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(module, image, numOptions, options, optionValues);
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    TRACE("cuModuleGetFunction(hmod=%p, name=%s)", hmod, name ? name : "null");
    auto real = vgpu::realSym<CUresult(*)(CUfunction*, CUmodule, const char*)>(
        "cuModuleGetFunction");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(hfunc, hmod, name);
}

CUresult cuModuleUnload(CUmodule hmod) {
    TRACE("cuModuleUnload(hmod=%p)", hmod);
    auto real = vgpu::realSym<CUresult(*)(CUmodule)>("cuModuleUnload");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(hmod);
}

// Memory — scheduled through the selected backend

CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    TRACE("cuMemAlloc(bytesize=%zu)", bytesize);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr*, size_t)>("cuMemAlloc_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(CUdeviceptr*, size_t)>("cuMemAlloc");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    if (!vgpu::schedRequest(vgpu::SchedOp::ALLOC_REQUEST, bytesize, vgpu::g_current_device)) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    CUresult err = real(dptr, bytesize);
    vgpu::finishAllocationAdmission(
        err, dptr, bytesize, vgpu::g_current_device);
    return err;
}

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    return cuMemAlloc(dptr, bytesize);
}

CUresult cuMemFree(CUdeviceptr dptr) {
    TRACE("cuMemFree(dptr=0x%llx)", (unsigned long long)dptr);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr)>("cuMemFree_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(CUdeviceptr)>("cuMemFree");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    size_t freed_size = 0;
    bool has_size = false;
    {
        std::lock_guard<std::mutex> lock(vgpu::g_alloc_mu);
        auto it = vgpu::g_alloc_sizes.find(static_cast<uint64_t>(dptr));
        if (it != vgpu::g_alloc_sizes.end()) {
            freed_size = it->second;
            has_size = true;
        }
    }

    CUresult err = real(dptr);
    if (err == CUDA_SUCCESS && has_size && freed_size > 0) {
        {
            std::lock_guard<std::mutex> lock(vgpu::g_alloc_mu);
            vgpu::g_alloc_sizes.erase(static_cast<uint64_t>(dptr));
        }
        vgpu::schedReport(vgpu::SchedOp::FREE, freed_size, vgpu::g_current_device);
    }
    return err;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) { return cuMemFree(dptr); }

CUresult cuMemGetInfo(size_t* free, size_t* total) {
    TRACE("cuMemGetInfo()");
    auto real = vgpu::realSym<CUresult(*)(size_t*, size_t*)>("cuMemGetInfo_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(size_t*, size_t*)>("cuMemGetInfo");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(free, total);
}

CUresult cuMemGetInfo_v2(size_t* free, size_t* total) { return cuMemGetInfo(free, total); }

CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) {
    TRACE("cuMemAllocAsync(bytesize=%zu, stream=%p)", bytesize, (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr*, size_t, CUstream)>("cuMemAllocAsync");
    if (!real) {
        // Driver too old: fall back to synchronous allocation path.
        return cuMemAlloc(dptr, bytesize);
    }

    if (!vgpu::schedRequest(vgpu::SchedOp::ALLOC_REQUEST, bytesize, vgpu::g_current_device)) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    CUresult err = real(dptr, bytesize, hStream);
    vgpu::finishAllocationAdmission(
        err, dptr, bytesize, vgpu::g_current_device);
    return err;
}

CUresult cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize,
                                 void* pool, CUstream hStream) {
    TRACE("cuMemAllocFromPoolAsync(bytesize=%zu, pool=%p, stream=%p)",
          bytesize, pool, (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr*, size_t, void*, CUstream)>(
        "cuMemAllocFromPoolAsync");
    if (!real) {
        return cuMemAllocAsync(dptr, bytesize, hStream);
    }

    if (!vgpu::schedRequest(vgpu::SchedOp::ALLOC_REQUEST, bytesize, vgpu::g_current_device)) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    CUresult err = real(dptr, bytesize, pool, hStream);
    vgpu::finishAllocationAdmission(
        err, dptr, bytesize, vgpu::g_current_device);
    return err;
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    TRACE("cuMemFreeAsync(dptr=0x%llx, stream=%p)", (unsigned long long)dptr, (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr, CUstream)>("cuMemFreeAsync");
    if (!real) {
        return cuMemFree(dptr);
    }

    size_t freed_size = 0;
    bool has_size = false;
    {
        std::lock_guard<std::mutex> lock(vgpu::g_alloc_mu);
        auto it = vgpu::g_alloc_sizes.find(static_cast<uint64_t>(dptr));
        if (it != vgpu::g_alloc_sizes.end()) {
            freed_size = it->second;
            has_size = true;
        }
    }

    CUresult err = real(dptr, hStream);
    if (err == CUDA_SUCCESS && has_size && freed_size > 0) {
        {
            std::lock_guard<std::mutex> lock(vgpu::g_alloc_mu);
            vgpu::g_alloc_sizes.erase(static_cast<uint64_t>(dptr));
        }
        vgpu::reportAfterStreamCompletionOrNow(vgpu::SchedOp::FREE, freed_size,
                               vgpu::g_current_device, hStream);
    }
    return err;
}

// Memcpy — scheduled through the selected backend

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t bytesize) {
    TRACE("cuMemcpy(dst=0x%llx, src=0x%llx, bytesize=%zu)",
          (unsigned long long)dst, (unsigned long long)src, bytesize);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr, CUdeviceptr, size_t)>("cuMemcpy");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    return vgpu::runScheduledCall(vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                  vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                  vgpu::g_current_device,
                                  CUDA_ERROR_NOT_SUPPORTED,
                                  [&] { return real(dst, src, bytesize); });
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t bytesize, CUstream hStream) {
    TRACE("cuMemcpyAsync(dst=0x%llx, src=0x%llx, bytesize=%zu, stream=%p)",
                    (unsigned long long)dst, (unsigned long long)src, bytesize, (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr, CUdeviceptr, size_t, CUstream)>(
                "cuMemcpyAsync");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    return vgpu::runScheduledAsyncCall(vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                       vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                       vgpu::g_current_device, hStream,
                                       CUDA_ERROR_NOT_SUPPORTED,
                                       [&] { return real(dst, src, bytesize, hStream); });
}

CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                                            CUdeviceptr srcDevice, CUcontext srcContext, size_t bytesize) {
    TRACE("cuMemcpyPeer(dst=0x%llx, dst_ctx=%p, src=0x%llx, src_ctx=%p, bytesize=%zu)",
                    (unsigned long long)dstDevice, dstContext,
                    (unsigned long long)srcDevice, srcContext, bytesize);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t)>(
                "cuMemcpyPeer");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    return vgpu::runScheduledCall(vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                  vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                  vgpu::g_current_device,
                                  CUDA_ERROR_NOT_SUPPORTED,
                                  [&] { return real(dstDevice, dstContext, srcDevice, srcContext, bytesize); });
}

CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                           CUdeviceptr srcDevice, CUcontext srcContext,
                           size_t bytesize, CUstream hStream) {
    TRACE("cuMemcpyPeerAsync(dst=0x%llx, dst_ctx=%p, src=0x%llx, src_ctx=%p, bytesize=%zu, stream=%p)",
          (unsigned long long)dstDevice, dstContext,
          (unsigned long long)srcDevice, srcContext, bytesize, (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext,
                                          size_t, CUstream)>("cuMemcpyPeerAsync");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    return vgpu::runScheduledAsyncCall(vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                       vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                       vgpu::g_current_device, hStream,
                                       CUDA_ERROR_NOT_SUPPORTED,
                                       [&] { return real(dstDevice, dstContext, srcDevice, srcContext,
                                                         bytesize, hStream); });
}

CUresult cuMemcpyHtoD(CUdeviceptr dst, const void* src, size_t bytesize) {
    TRACE("cuMemcpyHtoD(dst=0x%llx, bytesize=%zu)", (unsigned long long)dst, bytesize);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr, const void*, size_t)>("cuMemcpyHtoD_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(CUdeviceptr, const void*, size_t)>("cuMemcpyHtoD");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    return vgpu::runScheduledCall(vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                  vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                  vgpu::g_current_device,
                                  CUDA_ERROR_NOT_SUPPORTED,
                                  [&] { return real(dst, src, bytesize); });
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void* src, size_t bytesize) {
    return cuMemcpyHtoD(dst, src, bytesize);
}

CUresult cuMemcpyDtoH(void* dst, CUdeviceptr src, size_t bytesize) {
    TRACE("cuMemcpyDtoH(src=0x%llx, bytesize=%zu)", (unsigned long long)src, bytesize);
    auto real = vgpu::realSym<CUresult(*)(void*, CUdeviceptr, size_t)>("cuMemcpyDtoH_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(void*, CUdeviceptr, size_t)>("cuMemcpyDtoH");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return vgpu::runScheduledCall(vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                  vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                  vgpu::g_current_device,
                                  CUDA_ERROR_NOT_SUPPORTED,
                                  [&] { return real(dst, src, bytesize); });
}

CUresult cuMemcpyDtoH_v2(void* dst, CUdeviceptr src, size_t bytesize) {
    return cuMemcpyDtoH(dst, src, bytesize);
}

CUresult cuMemcpyDtoD(CUdeviceptr dst, CUdeviceptr src, size_t bytesize) {
    TRACE("cuMemcpyDtoD(dst=0x%llx, src=0x%llx, bytesize=%zu)",
          (unsigned long long)dst, (unsigned long long)src, bytesize);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr, CUdeviceptr, size_t)>("cuMemcpyDtoD_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(CUdeviceptr, CUdeviceptr, size_t)>("cuMemcpyDtoD");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return vgpu::runScheduledCall(vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                  vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                  vgpu::g_current_device,
                                  CUDA_ERROR_NOT_SUPPORTED,
                                  [&] { return real(dst, src, bytesize); });
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t bytesize) {
    return cuMemcpyDtoD(dst, src, bytesize);
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dst, const void* src, size_t bytesize, CUstream hStream) {
    TRACE("cuMemcpyHtoDAsync(dst=0x%llx, bytesize=%zu, stream=%p)",
          (unsigned long long)dst, bytesize, (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr, const void*, size_t, CUstream)>(
        "cuMemcpyHtoDAsync_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(CUdeviceptr, const void*, size_t, CUstream)>(
        "cuMemcpyHtoDAsync");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    return vgpu::runScheduledAsyncCall(vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                       vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                       vgpu::g_current_device, hStream,
                                       CUDA_ERROR_NOT_SUPPORTED,
                                       [&] { return real(dst, src, bytesize, hStream); });
}

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void* src, size_t bytesize,
                              CUstream hStream) {
    return cuMemcpyHtoDAsync(dst, src, bytesize, hStream);
}

CUresult cuMemcpyDtoHAsync(void* dst, CUdeviceptr src, size_t bytesize, CUstream hStream) {
    TRACE("cuMemcpyDtoHAsync(src=0x%llx, bytesize=%zu, stream=%p)",
          (unsigned long long)src, bytesize, (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(void*, CUdeviceptr, size_t, CUstream)>(
        "cuMemcpyDtoHAsync_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(void*, CUdeviceptr, size_t, CUstream)>(
        "cuMemcpyDtoHAsync");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    return vgpu::runScheduledAsyncCall(vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                       vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                       vgpu::g_current_device, hStream,
                                       CUDA_ERROR_NOT_SUPPORTED,
                                       [&] { return real(dst, src, bytesize, hStream); });
}

CUresult cuMemcpyDtoHAsync_v2(void* dst, CUdeviceptr src, size_t bytesize,
                              CUstream hStream) {
    return cuMemcpyDtoHAsync(dst, src, bytesize, hStream);
}

CUresult cuMemcpyDtoDAsync(CUdeviceptr dst, CUdeviceptr src, size_t bytesize, CUstream hStream) {
    TRACE("cuMemcpyDtoDAsync(dst=0x%llx, src=0x%llx, bytesize=%zu, stream=%p)",
          (unsigned long long)dst, (unsigned long long)src, bytesize, (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr, CUdeviceptr, size_t, CUstream)>(
        "cuMemcpyDtoDAsync_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(CUdeviceptr, CUdeviceptr, size_t, CUstream)>(
        "cuMemcpyDtoDAsync");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    return vgpu::runScheduledAsyncCall(vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                       vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                       vgpu::g_current_device, hStream,
                                       CUDA_ERROR_NOT_SUPPORTED,
                                       [&] { return real(dst, src, bytesize, hStream); });
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src, size_t bytesize,
                              CUstream hStream) {
    return cuMemcpyDtoDAsync(dst, src, bytesize, hStream);
}

CUresult cuMemcpy2D(const CUDA_MEMCPY2D* pCopy) {
    const uint64_t bytesize = vgpu::memcpy2DBytes(pCopy);
    TRACE("cuMemcpy2D(pCopy=%p, bytesize=%llu)",
          (const void*)pCopy, (unsigned long long)bytesize);
    auto real = vgpu::realSym<CUresult(*)(const CUDA_MEMCPY2D*)>("cuMemcpy2D_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(const CUDA_MEMCPY2D*)>("cuMemcpy2D");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    if (!pCopy) return real(pCopy);

    return vgpu::runScheduledCall(vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                  vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                  vgpu::g_current_device,
                                  CUDA_ERROR_NOT_SUPPORTED,
                                  [&] { return real(pCopy); });
}

CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D* pCopy) {
    return cuMemcpy2D(pCopy);
}

CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D* pCopy) {
    const uint64_t bytesize = vgpu::memcpy2DBytes(pCopy);
    TRACE("cuMemcpy2DUnaligned(pCopy=%p, bytesize=%llu)",
          (const void*)pCopy, (unsigned long long)bytesize);
    auto real = vgpu::realSym<CUresult(*)(const CUDA_MEMCPY2D*)>("cuMemcpy2DUnaligned_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(const CUDA_MEMCPY2D*)>("cuMemcpy2DUnaligned");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    if (!pCopy) return real(pCopy);

    return vgpu::runScheduledCall(vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                  vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                  vgpu::g_current_device,
                                  CUDA_ERROR_NOT_SUPPORTED,
                                  [&] { return real(pCopy); });
}

CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D* pCopy) {
    return cuMemcpy2DUnaligned(pCopy);
}

CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D* pCopy, CUstream hStream) {
    const uint64_t bytesize = vgpu::memcpy2DBytes(pCopy);
    TRACE("cuMemcpy2DAsync(pCopy=%p, bytesize=%llu, stream=%p)",
          (const void*)pCopy, (unsigned long long)bytesize, (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(const CUDA_MEMCPY2D*, CUstream)>(
        "cuMemcpy2DAsync_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(const CUDA_MEMCPY2D*, CUstream)>(
        "cuMemcpy2DAsync");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    if (!pCopy) return real(pCopy, hStream);

    return vgpu::runScheduledAsyncCall(vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                       vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                       vgpu::g_current_device, hStream,
                                       CUDA_ERROR_NOT_SUPPORTED,
                                       [&] { return real(pCopy, hStream); });
}

CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D* pCopy, CUstream hStream) {
    return cuMemcpy2DAsync(pCopy, hStream);
}

CUresult cuMemcpy3D(const CUDA_MEMCPY3D* pCopy) {
    const uint64_t bytesize = vgpu::memcpy3DBytes(pCopy);
    TRACE("cuMemcpy3D(pCopy=%p, bytesize=%llu)",
          (const void*)pCopy, (unsigned long long)bytesize);
    auto real = vgpu::realSym<CUresult(*)(const CUDA_MEMCPY3D*)>("cuMemcpy3D_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(const CUDA_MEMCPY3D*)>("cuMemcpy3D");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    if (!pCopy) return real(pCopy);

    return vgpu::runScheduledCall(vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                  vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                  vgpu::g_current_device,
                                  CUDA_ERROR_NOT_SUPPORTED,
                                  [&] { return real(pCopy); });
}

CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D* pCopy) {
    return cuMemcpy3D(pCopy);
}

CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D* pCopy, CUstream hStream) {
    const uint64_t bytesize = vgpu::memcpy3DBytes(pCopy);
    TRACE("cuMemcpy3DAsync(pCopy=%p, bytesize=%llu, stream=%p)",
          (const void*)pCopy, (unsigned long long)bytesize, (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(const CUDA_MEMCPY3D*, CUstream)>(
        "cuMemcpy3DAsync_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(const CUDA_MEMCPY3D*, CUstream)>(
        "cuMemcpy3DAsync");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    if (!pCopy) return real(pCopy, hStream);

    return vgpu::runScheduledAsyncCall(vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                       vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                       vgpu::g_current_device, hStream,
                                       CUDA_ERROR_NOT_SUPPORTED,
                                       [&] { return real(pCopy, hStream); });
}

CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D* pCopy, CUstream hStream) {
    return cuMemcpy3DAsync(pCopy, hStream);
}

CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER* pCopy) {
    const uint64_t bytesize = vgpu::memcpy3DPeerBytes(pCopy);
    TRACE("cuMemcpy3DPeer(pCopy=%p, bytesize=%llu)",
          (const void*)pCopy, (unsigned long long)bytesize);
    auto real = vgpu::realSym<CUresult(*)(const CUDA_MEMCPY3D_PEER*)>("cuMemcpy3DPeer");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    if (!pCopy) return real(pCopy);

    return vgpu::runScheduledCall(vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                  vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                  vgpu::g_current_device,
                                  CUDA_ERROR_NOT_SUPPORTED,
                                  [&] { return real(pCopy); });
}

CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream) {
    const uint64_t bytesize = vgpu::memcpy3DPeerBytes(pCopy);
    TRACE("cuMemcpy3DPeerAsync(pCopy=%p, bytesize=%llu, stream=%p)",
          (const void*)pCopy, (unsigned long long)bytesize, (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(const CUDA_MEMCPY3D_PEER*, CUstream)>(
        "cuMemcpy3DPeerAsync");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    if (!pCopy) return real(pCopy, hStream);

    return vgpu::runScheduledAsyncCall(vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                       vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                       vgpu::g_current_device, hStream,
                                       CUDA_ERROR_NOT_SUPPORTED,
                                       [&] { return real(pCopy, hStream); });
}

CUresult cuMemcpyBatchAsync_v2(CUdeviceptr* dsts, CUdeviceptr* srcs, size_t* sizes,
                               size_t count, CUmemcpyAttributes* attrs,
                               size_t* attrsIdxs, size_t numAttrs, CUstream hStream) {
    const uint64_t bytesize = vgpu::memcpyBatchBytes(sizes, count);
    TRACE("cuMemcpyBatchAsync_v2(count=%zu, bytesize=%llu, stream=%p)",
          count, (unsigned long long)bytesize, (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr*, CUdeviceptr*, size_t*, size_t,
                                          CUmemcpyAttributes*, size_t*, size_t, CUstream)>(
        "cuMemcpyBatchAsync_v2");
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;
    if (!dsts || !srcs || !sizes) {
        return real(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, hStream);
    }

    return vgpu::runScheduledAsyncCall(vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                       vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                       vgpu::g_current_device, hStream,
                                       CUDA_ERROR_NOT_SUPPORTED,
                                       [&] { return real(dsts, srcs, sizes, count,
                                                         attrs, attrsIdxs, numAttrs, hStream); });
}

// Memset — scheduled through the selected backend as memory-write work

CUresult cuMemsetD8(CUdeviceptr dst, unsigned char value, size_t count) {
    TRACE("cuMemsetD8(dst=0x%llx, value=%u, count=%zu)", (unsigned long long)dst, value, count);
    return vgpu::runMemset1D("cuMemsetD8_v2", "cuMemsetD8", dst, value, count, 1);
}

CUresult cuMemsetD8_v2(CUdeviceptr dst, unsigned char value, size_t count) {
    return cuMemsetD8(dst, value, count);
}

CUresult cuMemsetD16(CUdeviceptr dst, unsigned short value, size_t count) {
    TRACE("cuMemsetD16(dst=0x%llx, value=%u, count=%zu)", (unsigned long long)dst, value, count);
    return vgpu::runMemset1D("cuMemsetD16_v2", "cuMemsetD16", dst, value, count, 2);
}

CUresult cuMemsetD16_v2(CUdeviceptr dst, unsigned short value, size_t count) {
    return cuMemsetD16(dst, value, count);
}

CUresult cuMemsetD32(CUdeviceptr dst, unsigned int value, size_t count) {
    TRACE("cuMemsetD32(dst=0x%llx, value=%u, count=%zu)", (unsigned long long)dst, value, count);
    return vgpu::runMemset1D("cuMemsetD32_v2", "cuMemsetD32", dst, value, count, 4);
}

CUresult cuMemsetD32_v2(CUdeviceptr dst, unsigned int value, size_t count) {
    return cuMemsetD32(dst, value, count);
}

CUresult cuMemsetD2D8(CUdeviceptr dst, size_t dstPitch, unsigned char value,
                      size_t width, size_t height) {
    TRACE("cuMemsetD2D8(dst=0x%llx, pitch=%zu, value=%u, width=%zu, height=%zu)",
          (unsigned long long)dst, dstPitch, value, width, height);
    return vgpu::runMemset2D("cuMemsetD2D8_v2", "cuMemsetD2D8",
                             dst, dstPitch, value, width, height, 1);
}

CUresult cuMemsetD2D8_v2(CUdeviceptr dst, size_t dstPitch, unsigned char value,
                         size_t width, size_t height) {
    return cuMemsetD2D8(dst, dstPitch, value, width, height);
}

CUresult cuMemsetD2D16(CUdeviceptr dst, size_t dstPitch, unsigned short value,
                       size_t width, size_t height) {
    TRACE("cuMemsetD2D16(dst=0x%llx, pitch=%zu, value=%u, width=%zu, height=%zu)",
          (unsigned long long)dst, dstPitch, value, width, height);
    return vgpu::runMemset2D("cuMemsetD2D16_v2", "cuMemsetD2D16",
                             dst, dstPitch, value, width, height, 2);
}

CUresult cuMemsetD2D16_v2(CUdeviceptr dst, size_t dstPitch, unsigned short value,
                          size_t width, size_t height) {
    return cuMemsetD2D16(dst, dstPitch, value, width, height);
}

CUresult cuMemsetD2D32(CUdeviceptr dst, size_t dstPitch, unsigned int value,
                       size_t width, size_t height) {
    TRACE("cuMemsetD2D32(dst=0x%llx, pitch=%zu, value=%u, width=%zu, height=%zu)",
          (unsigned long long)dst, dstPitch, value, width, height);
    return vgpu::runMemset2D("cuMemsetD2D32_v2", "cuMemsetD2D32",
                             dst, dstPitch, value, width, height, 4);
}

CUresult cuMemsetD2D32_v2(CUdeviceptr dst, size_t dstPitch, unsigned int value,
                          size_t width, size_t height) {
    return cuMemsetD2D32(dst, dstPitch, value, width, height);
}

CUresult cuMemsetD8Async(CUdeviceptr dst, unsigned char value, size_t count, CUstream hStream) {
    TRACE("cuMemsetD8Async(dst=0x%llx, value=%u, count=%zu, stream=%p)",
          (unsigned long long)dst, value, count, (void*)hStream);
    return vgpu::runMemset1DAsync("cuMemsetD8Async", dst, value, count, 1, hStream);
}

CUresult cuMemsetD16Async(CUdeviceptr dst, unsigned short value, size_t count, CUstream hStream) {
    TRACE("cuMemsetD16Async(dst=0x%llx, value=%u, count=%zu, stream=%p)",
          (unsigned long long)dst, value, count, (void*)hStream);
    return vgpu::runMemset1DAsync("cuMemsetD16Async", dst, value, count, 2, hStream);
}

CUresult cuMemsetD32Async(CUdeviceptr dst, unsigned int value, size_t count, CUstream hStream) {
    TRACE("cuMemsetD32Async(dst=0x%llx, value=%u, count=%zu, stream=%p)",
          (unsigned long long)dst, value, count, (void*)hStream);
    return vgpu::runMemset1DAsync("cuMemsetD32Async", dst, value, count, 4, hStream);
}

CUresult cuMemsetD2D8Async(CUdeviceptr dst, size_t dstPitch, unsigned char value,
                           size_t width, size_t height, CUstream hStream) {
    TRACE("cuMemsetD2D8Async(dst=0x%llx, pitch=%zu, value=%u, width=%zu, height=%zu, stream=%p)",
          (unsigned long long)dst, dstPitch, value, width, height, (void*)hStream);
    return vgpu::runMemset2DAsync("cuMemsetD2D8Async",
                                  dst, dstPitch, value, width, height, 1, hStream);
}

CUresult cuMemsetD2D16Async(CUdeviceptr dst, size_t dstPitch, unsigned short value,
                            size_t width, size_t height, CUstream hStream) {
    TRACE("cuMemsetD2D16Async(dst=0x%llx, pitch=%zu, value=%u, width=%zu, height=%zu, stream=%p)",
          (unsigned long long)dst, dstPitch, value, width, height, (void*)hStream);
    return vgpu::runMemset2DAsync("cuMemsetD2D16Async",
                                  dst, dstPitch, value, width, height, 2, hStream);
}

CUresult cuMemsetD2D32Async(CUdeviceptr dst, size_t dstPitch, unsigned int value,
                            size_t width, size_t height, CUstream hStream) {
    TRACE("cuMemsetD2D32Async(dst=0x%llx, pitch=%zu, value=%u, width=%zu, height=%zu, stream=%p)",
          (unsigned long long)dst, dstPitch, value, width, height, (void*)hStream);
    return vgpu::runMemset2DAsync("cuMemsetD2D32Async",
                                  dst, dstPitch, value, width, height, 4, hStream);
}

// Kernel launch — scheduled through the selected backend

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void** kernelParams, void** extra) {
    TRACE("cuLaunchKernel(f=%p, grid=(%u,%u,%u), block=(%u,%u,%u), sharedMem=%u, stream=%p)",
          (void*)f, gridDimX, gridDimY, gridDimZ,
          blockDimX, blockDimY, blockDimZ, sharedMemBytes, (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(
        CUfunction, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int,
        unsigned int, CUstream, void**, void**)>("cuLaunchKernel");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    return vgpu::runScheduledAsyncCall(vgpu::SchedOp::KERNEL_REQUEST, 0,
                                       vgpu::SchedOp::KERNEL_COMPLETE, 0,
                                       vgpu::g_current_device, hStream,
                                       CUDA_ERROR_LAUNCH_FAILED,
                                       [&] {
                                           return real(f, gridDimX, gridDimY, gridDimZ,
                                                       blockDimX, blockDimY, blockDimZ,
                                                       sharedMemBytes, hStream, kernelParams, extra);
                                       });
}

CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f,
                         void** kernelParams, void** extra) {
    TRACE("cuLaunchKernelEx(config=%p, f=%p)", (const void*)config, (void*)f);
    auto real = vgpu::realSym<CUresult(*)(const CUlaunchConfig*, CUfunction, void**, void**)>(
        "cuLaunchKernelEx");
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;

    CUstream stream = config ? config->hStream : nullptr;
    return vgpu::runScheduledAsyncCall(vgpu::SchedOp::KERNEL_REQUEST, 0,
                                       vgpu::SchedOp::KERNEL_COMPLETE, 0,
                                       vgpu::g_current_device, stream,
                                       CUDA_ERROR_LAUNCH_FAILED,
                                       [&] { return real(config, f, kernelParams, extra); });
}

CUresult cuLaunchKernelExC(const void* config, CUfunction f,
                          void** kernelParams, void** extra) {
    TRACE("cuLaunchKernelExC(config=%p, f=%p)", config, (void*)f);
    auto real = vgpu::realSym<CUresult(*)(const void*, CUfunction, void**, void**)>(
        "cuLaunchKernelExC");
    if (!real) {
        return cuLaunchKernelEx(static_cast<const CUlaunchConfig*>(config), f, kernelParams, extra);
    }

    return vgpu::runScheduledCall(vgpu::SchedOp::KERNEL_REQUEST, 0,
                                  vgpu::SchedOp::KERNEL_COMPLETE, 0,
                                  vgpu::g_current_device,
                                  CUDA_ERROR_LAUNCH_FAILED,
                                  [&] { return real(config, f, kernelParams, extra); });
}

// Stream management

CUresult cuStreamCreate(CUstream* phStream, unsigned int flags) {
    TRACE("cuStreamCreate(flags=%u)", flags);
    auto real = vgpu::realSym<CUresult(*)(CUstream*, unsigned int)>("cuStreamCreate");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(phStream, flags);
}

CUresult cuStreamDestroy(CUstream hStream) {
    TRACE("cuStreamDestroy(stream=%p)", (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(CUstream)>("cuStreamDestroy_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(CUstream)>("cuStreamDestroy");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(hStream);
}

CUresult cuStreamDestroy_v2(CUstream hStream) { return cuStreamDestroy(hStream); }

CUresult cuStreamSynchronize(CUstream hStream) {
    TRACE("cuStreamSynchronize(stream=%p)", (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(CUstream)>("cuStreamSynchronize");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(hStream);
}

CUresult cuStreamQuery(CUstream hStream) {
    TRACE("cuStreamQuery(stream=%p)", (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(CUstream)>("cuStreamQuery");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(hStream);
}

// Event management

CUresult cuEventCreate(CUevent* phEvent, unsigned int flags) {
    TRACE("cuEventCreate(flags=%u)", flags);
    auto real = vgpu::realSym<CUresult(*)(CUevent*, unsigned int)>("cuEventCreate");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(phEvent, flags);
}

CUresult cuEventDestroy(CUevent hEvent) {
    TRACE("cuEventDestroy(event=%p)", (void*)hEvent);
    auto real = vgpu::realSym<CUresult(*)(CUevent)>("cuEventDestroy_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(CUevent)>("cuEventDestroy");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(hEvent);
}

CUresult cuEventDestroy_v2(CUevent hEvent) { return cuEventDestroy(hEvent); }

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    TRACE("cuEventRecord(event=%p, stream=%p)", (void*)hEvent, (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(CUevent, CUstream)>("cuEventRecord");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(hEvent, hStream);
}

CUresult cuEventSynchronize(CUevent hEvent) {
    TRACE("cuEventSynchronize(event=%p)", (void*)hEvent);
    auto real = vgpu::realSym<CUresult(*)(CUevent)>("cuEventSynchronize");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(hEvent);
}

CUresult cuEventQuery(CUevent hEvent) {
    TRACE("cuEventQuery(event=%p)", (void*)hEvent);
    auto real = vgpu::realSym<CUresult(*)(CUevent)>("cuEventQuery");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(hEvent);
}

static bool resolveShimProcAddress(const char* symbol, void** function,
                                   int cuda_version,
                                   unsigned long long* status) {
    void* resolved = nullptr;
    if (std::strcmp(symbol, "cuGetProcAddress") == 0) {
        resolved = reinterpret_cast<void*>(cuGetProcAddress);
    } else if (std::strcmp(symbol, "cuGetProcAddress_v2") == 0) {
        resolved = reinterpret_cast<void*>(cuGetProcAddress_v2);
    } else if (vgpu::shouldRouteProcAddressToShim(symbol, cuda_version)) {
        resolved = dlsym(RTLD_DEFAULT, symbol);
    }

    if (!resolved) return false;
    *function = resolved;
    if (status) *status = 0;
    return true;
}

// cuGetProcAddress — dynamic symbol resolution routing

CUresult cuGetProcAddress(const char* symbol, void** pfn,
                          int cudaVersion, unsigned long long flags) {
    TRACE("cuGetProcAddress(symbol=%s, version=%d)", symbol ? symbol : "null", cudaVersion);
    if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;

    if (resolveShimProcAddress(symbol, pfn, cudaVersion, nullptr)) {
        return CUDA_SUCCESS;
    }

    using RealGetProcAddressFn = CUresult(*)(const char*, void**, int, unsigned long long);
    auto real_get_proc = vgpu::realSym<RealGetProcAddressFn>("cuGetProcAddress");
    if (real_get_proc) {
        CUresult rc = real_get_proc(symbol, pfn, cudaVersion, flags);
        if (rc == CUDA_SUCCESS && pfn && *pfn) return rc;
    }

    *pfn = nullptr;
    return CUDA_ERROR_NOT_FOUND;
}

CUresult cuGetProcAddress_v2(const char* symbol, void** pfn,
                             int cudaVersion, unsigned long long flags,
                             unsigned long long* symbolStatus) {
    TRACE("cuGetProcAddress_v2(symbol=%s, version=%d)", symbol ? symbol : "null", cudaVersion);
    if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;

    if (resolveShimProcAddress(symbol, pfn, cudaVersion, symbolStatus)) {
        return CUDA_SUCCESS;
    }

    using RealGetProcAddressV2Fn = CUresult(*)(const char*, void**, int, unsigned long long,
                                               unsigned long long*);
    auto real_get_proc_v2 = vgpu::realSym<RealGetProcAddressV2Fn>("cuGetProcAddress_v2");
    if (real_get_proc_v2) {
        CUresult rc = real_get_proc_v2(symbol, pfn, cudaVersion, flags, symbolStatus);
        if (rc == CUDA_SUCCESS && pfn && *pfn) return rc;
    }

    using RealGetProcAddressFn = CUresult(*)(const char*, void**, int, unsigned long long);
    auto real_get_proc = vgpu::realSym<RealGetProcAddressFn>("cuGetProcAddress");
    if (real_get_proc) {
        CUresult rc = real_get_proc(symbol, pfn, cudaVersion, flags);
        if (symbolStatus) *symbolStatus = (rc == CUDA_SUCCESS) ? 0ull : 1ull;
        if (rc == CUDA_SUCCESS && pfn && *pfn) return rc;
    }

    *pfn = nullptr;
    if (symbolStatus) *symbolStatus = 1ull;
    return CUDA_ERROR_NOT_FOUND;
}

// cuGetExportTable — needed for cuBLAS/cuDNN

CUresult cuGetExportTable(const void** ppExportTable, const void* pExportTableId) {
    TRACE("cuGetExportTable()");
    if (!ppExportTable) return CUDA_ERROR_INVALID_VALUE;

    if (vgpu::envFlag("VGPU_CU_EXPORT_TABLE_SUCCESS_NULL")) {
        *ppExportTable = nullptr;
        return CUDA_SUCCESS;
    }

    void* real = vgpu::realCudaHandle();
    if (real) {
        using Fn = CUresult(*)(const void**, const void*);
        auto real_fn = reinterpret_cast<Fn>(dlsym(real, "cuGetExportTable"));
        if (real_fn) return real_fn(ppExportTable, pExportTableId);
    }

    *ppExportTable = nullptr;
    return CUDA_ERROR_NOT_SUPPORTED;
}

}  // extern "C"
