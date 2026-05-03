// intercept.cpp — libcuda.so shim (LD_PRELOAD)
//
// Intercepts CUDA Driver API calls for scheduling coordination.
// Heavy ops (memory alloc, kernel launch) go through the daemon via shared
// memory with atomic spin-wait. Light ops pass through to the real driver.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <unordered_map>
#include <unordered_set>

#include "vgpu/cuda_abi.h"
#include "vgpu/config.h"
#include "vgpu/protocol.h"

// Trace logging — controlled by VGPU_TRACE env var (default: on)
static bool g_trace = [] {
    return vgpu::config::getBool("VGPU_TRACE", true);
}();

#define TRACE(fmt, ...) do { \
    if (g_trace) std::fprintf(stderr, "[vGPU trace] " fmt "\n", ##__VA_ARGS__); \
} while(0)

// Portable spin-wait hint
#if defined(__x86_64__) || defined(_M_X64)
#define SPIN_PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__) || defined(_M_ARM64)
#define SPIN_PAUSE() asm volatile("yield" ::: "memory")
#else
#define SPIN_PAUSE() ((void)0)
#endif

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

// ── Shared memory channel ────────────────────────────────────────────────

ShmChannel* g_channel = nullptr;
std::mutex g_connect_mu;
std::mutex g_sched_mu;

std::mutex g_alloc_mu;
std::unordered_map<uint64_t, size_t> g_alloc_sizes;

bool memcpySchedulingEnabled() {
    std::string v = vgpu::config::getEnvOrConfig("GPU_SCHEDULER_CONTROL_MEMCPY");
    if (v.empty()) {
        v = vgpu::config::getEnvOrConfig("VGPU_CONTROL_MEMCPY");
    }
    if (v.empty()) return false;
    return v == "1" || v == "true" || v == "TRUE";
}

std::string daemonSocketPath() {
    return vgpu::config::getEnvOrConfig("GPU_SCHEDULER_SOCKET", defaultSocketPath());
}

bool connectToDaemon() {
    if (g_channel) return true;
    std::lock_guard<std::mutex> lock(g_connect_mu);
    if (g_channel) return true;

    std::string path = daemonSocketPath();

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return false;

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        close(fd);
        return false;
    }
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close(fd);
        return false;
    }

    HandshakeRequest req{};
    req.op = static_cast<uint32_t>(HandshakeOp::HELLO);
    req.client_id = static_cast<uint64_t>(getpid());

    auto writeAll = [](int out_fd, const void* buf, size_t len) {
        const auto* p = static_cast<const uint8_t*>(buf);
        size_t done = 0;
        while (done < len) {
            ssize_t n = write(out_fd, p + done, len - done);
            if (n < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            if (n == 0) return false;
            done += static_cast<size_t>(n);
        }
        return true;
    };

    auto readAll = [](int in_fd, void* buf, size_t len) {
        auto* p = static_cast<uint8_t*>(buf);
        size_t done = 0;
        while (done < len) {
            ssize_t n = read(in_fd, p + done, len - done);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) continue;
                return false;
            }
            done += static_cast<size_t>(n);
        }
        return true;
    };

    if (!writeAll(fd, &req, sizeof(req))) {
        close(fd);
        return false;
    }

    HandshakeResponse rsp{};
    if (!readAll(fd, &rsp, sizeof(rsp)) || rsp.status != 0) {
        close(fd);
        return false;
    }

    int shm_fd = shm_open(rsp.shm_name, O_RDWR, 0666);
    if (shm_fd < 0) {
        close(fd);
        return false;
    }

    g_channel = static_cast<ShmChannel*>(
        mmap(nullptr, sizeof(ShmChannel), PROT_READ | PROT_WRITE,
             MAP_SHARED, shm_fd, 0));
    close(shm_fd);

    if (g_channel == MAP_FAILED) {
        g_channel = nullptr;
        close(fd);
        return false;
    }

    return true;
}

bool ensureConnected() {
    if (g_channel) return true;
    return connectToDaemon();
}

// ── Scheduling helpers ───────────────────────────────────────────────────

thread_local int g_current_device = 0;

static int waitIterations() {
    static const int kDefaultIters = 200000;
    static const int kIters = [] {
        std::string raw = vgpu::config::getEnvOrConfig("GPU_SCHEDULER_WAIT_ITERS");
        if (raw.empty()) return kDefaultIters;

        char* end = nullptr;
        long parsed = std::strtol(raw.c_str(), &end, 10);
        if (end == raw.c_str() || parsed <= 0) return kDefaultIters;
        if (parsed > 5000000) return 5000000;
        return static_cast<int>(parsed);
    }();
    return kIters;
}

static SchedState waitForDecision(int max_iters) {
    if (!g_channel) return SchedState::REJECTED;
    for (int i = 0; i < max_iters; i++) {
        auto state = static_cast<SchedState>(
            g_channel->state.load(std::memory_order_acquire));
        if (state == SchedState::APPROVED || state == SchedState::REJECTED) {
            return state;
        }
        if (i < 1000) SPIN_PAUSE();
        else std::this_thread::yield();
    }
    return SchedState::PENDING;
}

static bool waitForIdle(int max_iters) {
    if (!g_channel) return false;
    for (int i = 0; i < max_iters; i++) {
        auto state = static_cast<SchedState>(
            g_channel->state.load(std::memory_order_acquire));
        if (state == SchedState::IDLE) return true;
        if (i < 1000) SPIN_PAUSE();
        else std::this_thread::yield();
    }
    return false;
}

bool schedRequest(SchedOp op, uint64_t value, int device) {
    if (!ensureConnected()) return true;
    std::lock_guard<std::mutex> lock(g_sched_mu);
    if (!waitForIdle(waitIterations())) return true;

    TRACE("daemon request op=%u value=%llu device=%d",
          static_cast<unsigned int>(op),
          static_cast<unsigned long long>(value), device);

    g_channel->device = static_cast<uint32_t>(device);
    g_channel->client_id = static_cast<uint64_t>(getpid());
    g_channel->value.store(value, std::memory_order_relaxed);
    g_channel->op.store(static_cast<uint32_t>(op), std::memory_order_relaxed);
    g_channel->state.store(static_cast<uint32_t>(SchedState::PENDING),
                           std::memory_order_release);

    auto decision = waitForDecision(waitIterations());
    if (decision == SchedState::APPROVED) {
        g_channel->state.store(static_cast<uint32_t>(SchedState::IDLE),
                               std::memory_order_release);
        TRACE("daemon approved op=%u", static_cast<unsigned int>(op));
        return true;
    }
    if (decision == SchedState::REJECTED) {
        g_channel->state.store(static_cast<uint32_t>(SchedState::IDLE),
                               std::memory_order_release);
        TRACE("daemon rejected op=%u", static_cast<unsigned int>(op));
        return false;
    }

    TRACE("daemon request timeout, bypass op=%u", static_cast<unsigned int>(op));
    return true;
}

void schedReport(SchedOp op, uint64_t value, int device) {
    if (!ensureConnected()) return;
    std::lock_guard<std::mutex> lock(g_sched_mu);
    if (!waitForIdle(waitIterations())) return;

    TRACE("daemon report op=%u value=%llu device=%d",
          static_cast<unsigned int>(op),
          static_cast<unsigned long long>(value), device);

    g_channel->device = static_cast<uint32_t>(device);
    g_channel->client_id = static_cast<uint64_t>(getpid());
    g_channel->value.store(value, std::memory_order_relaxed);
    g_channel->op.store(static_cast<uint32_t>(op), std::memory_order_relaxed);
    g_channel->state.store(static_cast<uint32_t>(SchedState::PENDING),
                           std::memory_order_release);

    if (waitForIdle(waitIterations())) {
        TRACE("daemon report done op=%u", static_cast<unsigned int>(op));
        return;
    }

    TRACE("daemon report timeout op=%u", static_cast<unsigned int>(op));
}

template <typename Callable>
CUresult runScheduledCall(bool enabled,
                         SchedOp request_op,
                         uint64_t request_value,
                         SchedOp complete_op,
                         uint64_t complete_value,
                         int device,
                         CUresult reject_code,
                         Callable&& callable) {
    if (!enabled) {
        return std::forward<Callable>(callable)();
    }

    if (!schedRequest(request_op, request_value, device)) {
        return reject_code;
    }

    CUresult err = std::forward<Callable>(callable)();
    schedReport(complete_op, complete_value, device);
    return err;
}

// ── Symbol classification ────────────────────────────────────────────────

bool isInterceptedSymbol(const char* symbol) {
    using namespace std::literals;
    static const std::unordered_set<std::string_view> kIntercepted = {
        "cuMemAlloc"sv, "cuMemAlloc_v2"sv,
        "cuMemAllocAsync"sv, "cuMemAllocFromPoolAsync"sv,
        "cuMemFree"sv, "cuMemFree_v2"sv,
        "cuMemFreeAsync"sv,
        "cuMemcpyHtoD"sv, "cuMemcpyHtoD_v2"sv,
        "cuMemcpyDtoH"sv, "cuMemcpyDtoH_v2"sv,
        "cuMemcpyDtoD"sv, "cuMemcpyDtoD_v2"sv,
        "cuMemcpyHtoDAsync"sv, "cuMemcpyDtoHAsync"sv, "cuMemcpyDtoDAsync"sv,
        "cuLaunchKernel"sv,
        "cuLaunchKernelEx"sv, "cuLaunchKernelExC"sv,
        "cuStreamCreate"sv, "cuStreamDestroy"sv,
        "cuStreamSynchronize"sv, "cuStreamQuery"sv,
        "cuGetProcAddress"sv, "cuGetProcAddress_v2"sv,
    };
    return kIntercepted.count(symbol) > 0;
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

    if (symbol && (std::strcmp(symbol, "cuGetProcAddress") == 0 ||
                   std::strcmp(symbol, "cuGetProcAddress_v2") == 0)) {
        void* fn = vgpu::g_real_dlsym(RTLD_DEFAULT, symbol);
        if (fn) return fn;
    }

    return vgpu::g_real_dlsym(handle, symbol);
}

// ── Driver API exports ───────────────────────────────────────────────────

extern "C" {

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

// Memory — scheduled through daemon

CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    TRACE("cuMemAlloc(bytesize=%zu)", bytesize);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr*, size_t)>("cuMemAlloc_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(CUdeviceptr*, size_t)>("cuMemAlloc");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    if (!vgpu::schedRequest(vgpu::SchedOp::ALLOC_REQUEST, bytesize, vgpu::g_current_device)) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    CUresult err = real(dptr, bytesize);
    if (err == CUDA_SUCCESS && dptr) {
        std::lock_guard<std::mutex> lock(vgpu::g_alloc_mu);
        vgpu::g_alloc_sizes[*dptr] = bytesize;
    }
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
    if (err == CUDA_SUCCESS && dptr) {
        std::lock_guard<std::mutex> lock(vgpu::g_alloc_mu);
        vgpu::g_alloc_sizes[*dptr] = bytesize;
    }
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
    if (err == CUDA_SUCCESS && dptr) {
        std::lock_guard<std::mutex> lock(vgpu::g_alloc_mu);
        vgpu::g_alloc_sizes[*dptr] = bytesize;
    }
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
        vgpu::schedReport(vgpu::SchedOp::FREE, freed_size, vgpu::g_current_device);
    }
    return err;
}

// Memcpy/memset — direct to real driver

CUresult cuMemcpyHtoD(CUdeviceptr dst, const void* src, size_t bytesize) {
    TRACE("cuMemcpyHtoD(dst=0x%llx, bytesize=%zu)", (unsigned long long)dst, bytesize);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr, const void*, size_t)>("cuMemcpyHtoD_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(CUdeviceptr, const void*, size_t)>("cuMemcpyHtoD");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    return vgpu::runScheduledCall(vgpu::memcpySchedulingEnabled(),
                                  vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
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
    return vgpu::runScheduledCall(vgpu::memcpySchedulingEnabled(),
                                  vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
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
    return vgpu::runScheduledCall(vgpu::memcpySchedulingEnabled(),
                                  vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
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

    return vgpu::runScheduledCall(vgpu::memcpySchedulingEnabled(),
                                  vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                  vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                  vgpu::g_current_device,
                                  CUDA_ERROR_NOT_SUPPORTED,
                                  [&] { return real(dst, src, bytesize, hStream); });
}

CUresult cuMemcpyDtoHAsync(void* dst, CUdeviceptr src, size_t bytesize, CUstream hStream) {
    TRACE("cuMemcpyDtoHAsync(src=0x%llx, bytesize=%zu, stream=%p)",
          (unsigned long long)src, bytesize, (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(void*, CUdeviceptr, size_t, CUstream)>(
        "cuMemcpyDtoHAsync_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(void*, CUdeviceptr, size_t, CUstream)>(
        "cuMemcpyDtoHAsync");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    return vgpu::runScheduledCall(vgpu::memcpySchedulingEnabled(),
                                  vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                  vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                  vgpu::g_current_device,
                                  CUDA_ERROR_NOT_SUPPORTED,
                                  [&] { return real(dst, src, bytesize, hStream); });
}

CUresult cuMemcpyDtoDAsync(CUdeviceptr dst, CUdeviceptr src, size_t bytesize, CUstream hStream) {
    TRACE("cuMemcpyDtoDAsync(dst=0x%llx, src=0x%llx, bytesize=%zu, stream=%p)",
          (unsigned long long)dst, (unsigned long long)src, bytesize, (void*)hStream);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr, CUdeviceptr, size_t, CUstream)>(
        "cuMemcpyDtoDAsync_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(CUdeviceptr, CUdeviceptr, size_t, CUstream)>(
        "cuMemcpyDtoDAsync");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;

    return vgpu::runScheduledCall(vgpu::memcpySchedulingEnabled(),
                                  vgpu::SchedOp::MEMCPY_REQUEST, bytesize,
                                  vgpu::SchedOp::MEMCPY_COMPLETE, bytesize,
                                  vgpu::g_current_device,
                                  CUDA_ERROR_NOT_SUPPORTED,
                                  [&] { return real(dst, src, bytesize, hStream); });
}

CUresult cuMemsetD8(CUdeviceptr dst, unsigned char value, size_t count) {
    TRACE("cuMemsetD8(dst=0x%llx, value=%u, count=%zu)", (unsigned long long)dst, value, count);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr, unsigned char, size_t)>("cuMemsetD8_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(CUdeviceptr, unsigned char, size_t)>("cuMemsetD8");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(dst, value, count);
}

CUresult cuMemsetD8_v2(CUdeviceptr dst, unsigned char value, size_t count) {
    return cuMemsetD8(dst, value, count);
}

CUresult cuMemsetD32(CUdeviceptr dst, unsigned int value, size_t count) {
    TRACE("cuMemsetD32(dst=0x%llx, value=%u, count=%zu)", (unsigned long long)dst, value, count);
    auto real = vgpu::realSym<CUresult(*)(CUdeviceptr, unsigned int, size_t)>("cuMemsetD32_v2");
    if (!real) real = vgpu::realSym<CUresult(*)(CUdeviceptr, unsigned int, size_t)>("cuMemsetD32");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    return real(dst, value, count);
}

CUresult cuMemsetD32_v2(CUdeviceptr dst, unsigned int value, size_t count) {
    return cuMemsetD32(dst, value, count);
}

// Kernel launch — scheduled through daemon

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

    return vgpu::runScheduledCall(true,
                                  vgpu::SchedOp::KERNEL_REQUEST, 0,
                                  vgpu::SchedOp::KERNEL_COMPLETE, 0,
                                  vgpu::g_current_device,
                                  CUDA_ERROR_LAUNCH_FAILED,
                                  [&] {
                                      return real(f, gridDimX, gridDimY, gridDimZ,
                                                  blockDimX, blockDimY, blockDimZ,
                                                  sharedMemBytes, hStream, kernelParams, extra);
                                  });
}

CUresult cuLaunchKernelEx(const void* config, CUfunction f,
                         void** kernelParams, void** extra) {
    TRACE("cuLaunchKernelEx(config=%p, f=%p)", config, (void*)f);
    auto real = vgpu::realSym<CUresult(*)(const void*, CUfunction, void**, void**)>(
        "cuLaunchKernelEx");
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;

    return vgpu::runScheduledCall(true,
                                  vgpu::SchedOp::KERNEL_REQUEST, 0,
                                  vgpu::SchedOp::KERNEL_COMPLETE, 0,
                                  vgpu::g_current_device,
                                  CUDA_ERROR_LAUNCH_FAILED,
                                  [&] { return real(config, f, kernelParams, extra); });
}

CUresult cuLaunchKernelExC(const void* config, CUfunction f,
                          void** kernelParams, void** extra) {
    TRACE("cuLaunchKernelExC(config=%p, f=%p)", config, (void*)f);
    auto real = vgpu::realSym<CUresult(*)(const void*, CUfunction, void**, void**)>(
        "cuLaunchKernelExC");
    if (!real) {
        return cuLaunchKernelEx(config, f, kernelParams, extra);
    }

    return vgpu::runScheduledCall(true,
                                  vgpu::SchedOp::KERNEL_REQUEST, 0,
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

// cuGetProcAddress — dynamic symbol resolution routing

CUresult cuGetProcAddress(const char* symbol, void** pfn,
                          int cudaVersion, unsigned long long flags) {
    TRACE("cuGetProcAddress(symbol=%s, version=%d)", symbol ? symbol : "null", cudaVersion);
    if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;

    if (std::strcmp(symbol, "cuGetProcAddress") == 0) {
        *pfn = reinterpret_cast<void*>(cuGetProcAddress);
        return CUDA_SUCCESS;
    }
    if (std::strcmp(symbol, "cuGetProcAddress_v2") == 0) {
        *pfn = reinterpret_cast<void*>(cuGetProcAddress_v2);
        return CUDA_SUCCESS;
    }

    if (vgpu::isInterceptedSymbol(symbol)) {
        void* fn = dlsym(RTLD_DEFAULT, symbol);
        if (fn) { *pfn = fn; return CUDA_SUCCESS; }
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

    if (std::strcmp(symbol, "cuGetProcAddress") == 0) {
        *pfn = reinterpret_cast<void*>(cuGetProcAddress);
        if (symbolStatus) *symbolStatus = 0ull;
        return CUDA_SUCCESS;
    }
    if (std::strcmp(symbol, "cuGetProcAddress_v2") == 0) {
        *pfn = reinterpret_cast<void*>(cuGetProcAddress_v2);
        if (symbolStatus) *symbolStatus = 0ull;
        return CUDA_SUCCESS;
    }

    if (vgpu::isInterceptedSymbol(symbol)) {
        void* fn = dlsym(RTLD_DEFAULT, symbol);
        if (fn) {
            *pfn = fn;
            if (symbolStatus) *symbolStatus = 0ull;
            return CUDA_SUCCESS;
        }
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
