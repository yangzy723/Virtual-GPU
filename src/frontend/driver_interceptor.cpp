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
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

#include "vgpu/common/context_registry.h"
#include "vgpu/common/cuda_abi.h"
#include "vgpu/backend/cuda_driver_loader.h"
#include "vgpu/frontend/shim_utils.h"
#include "vgpu/common/fatbin_parser.h"
#include "vgpu/common/kernel_registry.h"
#include <fstream>
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

bool logCtxOps();

bool queryDeviceAttributeFromServer(int attr, int device, int* out) {
    if (out == nullptr) {
        return false;
    }

    RpcDeviceGetAttributeReq req{};
    req.attr = attr;
    req.device = device;
    RpcResult r = drvCall(RpcOp::kCudaDeviceGetAttribute, &req, sizeof(req));
    if (r.status != cudaSuccess) {
        if (logCtxOps()) {
            std::fprintf(stderr,
                         "[vgpu-ctx] cuDeviceGetAttribute rpc attr=%d dev=%d status=%d\n",
                         attr,
                         device,
                         static_cast<int>(r.status));
        }
        return false;
    }

    *out = static_cast<int>(r.aux_u64);
    if (logCtxOps()) {
        std::fprintf(stderr,
                     "[vgpu-ctx] cuDeviceGetAttribute rpc attr=%d dev=%d val=%d\n",
                     attr,
                     device,
                     *out);
    }
    return true;
}

bool logCtxOps() {
    return envFlagEnabled("VGPU_LOG_CTX");
}

bool logExportTableCalls() {
    return envFlagEnabled("VGPU_LOG_CU_EXPORT_TABLE_CALLS");
}

bool allowFakeExportTableWriteback() {
    return envFlagEnabled("VGPU_FAKE_EXPORT_TABLE_WRITEBACK");
}

bool tokenInCsv(const char* csv, const char* token) {
    if (csv == nullptr || token == nullptr || token[0] == '\0') {
        return false;
    }
    const std::size_t token_len = std::strlen(token);
    const char* p = csv;
    while ((p = std::strstr(p, token)) != nullptr) {
        const bool left_ok = (p == csv) || (p[-1] == ',') || (p[-1] == ' ');
        const char right = p[token_len];
        const bool right_ok = (right == '\0') || (right == ',') || (right == ' ');
        if (left_ok && right_ok) {
            return true;
        }
        p += token_len;
    }
    return false;
}

bool shouldWriteExportArg(const char* uuid_tag, const char* arg_tag) {
    if (!allowFakeExportTableWriteback()) {
        return false;
    }
    const char* mask = std::getenv("VGPU_FAKE_EXPORT_TABLE_WRITEBACK_MASK");
    if (mask == nullptr || mask[0] == '\0') {
        // Empirically, a094:a0 writeback corrupts cuBLAS teardown state and
        // leads to SIGSEGV at process exit. Keep writeback enabled by default
        // for other slots, but avoid this one unless explicitly opted in.
        if (std::strcmp(uuid_tag, "a0") == 0 && std::strcmp(arg_tag, "a0") == 0) {
            return false;
        }
        return true;
    }

    char token[32];
    std::snprintf(token, sizeof(token), "%s:%s", uuid_tag, arg_tag);
    return tokenInCsv(mask, token);
}

void maybeWritePtrFor(const char* uuid_tag,
                      const char* arg_tag,
                      std::uintptr_t dst,
                      std::uintptr_t value) {
    if (!shouldWriteExportArg(uuid_tag, arg_tag)) {
        return;
    }
    if (!processRangeHasAccess(dst, sizeof(std::uintptr_t), ProcMapAccess::Write)) {
        return;
    }
    *reinterpret_cast<std::uintptr_t*>(dst) = value;
}

std::uintptr_t nextFakeExportHandle() {
    static std::atomic<std::uintptr_t> g_next{0xEE0000};
    return g_next.fetch_add(1);
}

CUresult cuExportTable69Stub(std::uintptr_t a0,
                             std::uintptr_t a1,
                             std::uintptr_t a2,
                             std::uintptr_t a3,
                             std::uintptr_t a4,
                             std::uintptr_t a5) {
    if (logExportTableCalls()) {
        std::fprintf(stderr,
                     "[vgpu-cuGetExportTable] call uuid=69a28e9f a0=%p a1=%p a2=%p a3=%p a4=%p a5=%p rc=%d\n",
                     reinterpret_cast<void*>(a0),
                     reinterpret_cast<void*>(a1),
                     reinterpret_cast<void*>(a2),
                     reinterpret_cast<void*>(a3),
                     reinterpret_cast<void*>(a4),
                     reinterpret_cast<void*>(a5),
                     static_cast<int>(CUDA_ERROR_NOT_SUPPORTED));
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuExportTable6bStub(std::uintptr_t a0,
                             std::uintptr_t a1,
                             std::uintptr_t a2,
                             std::uintptr_t a3,
                             std::uintptr_t a4,
                             std::uintptr_t a5) {
    // Best-effort: these entry points appear to return opaque handles via out-pointers.
    maybeWritePtrFor("6b", "a0", a0, nextFakeExportHandle());
    maybeWritePtrFor("6b", "a5", a5, nextFakeExportHandle());

    if (logExportTableCalls()) {
        std::fprintf(stderr,
                     "[vgpu-cuGetExportTable] call uuid=6bd5fb6c a0=%p a1=%p a2=%p a3=%p a4=%p a5=%p rc=%d\n",
                     reinterpret_cast<void*>(a0),
                     reinterpret_cast<void*>(a1),
                     reinterpret_cast<void*>(a2),
                     reinterpret_cast<void*>(a3),
                     reinterpret_cast<void*>(a4),
                     reinterpret_cast<void*>(a5),
                     static_cast<int>(CUDA_SUCCESS));
    }
    return CUDA_SUCCESS;
}

CUresult cuExportTablea0Stub(std::uintptr_t a0,
                             std::uintptr_t a1,
                             std::uintptr_t a2,
                             std::uintptr_t a3,
                             std::uintptr_t a4,
                             std::uintptr_t a5) {
    maybeWritePtrFor("a0", "a0", a0, nextFakeExportHandle());
    maybeWritePtrFor("a0", "a1", a1, nextFakeExportHandle());

    if (logExportTableCalls()) {
        std::fprintf(stderr,
                     "[vgpu-cuGetExportTable] call uuid=a094798c a0=%p a1=%p a2=%p a3=%p a4=%p a5=%p rc=%d\n",
                     reinterpret_cast<void*>(a0),
                     reinterpret_cast<void*>(a1),
                     reinterpret_cast<void*>(a2),
                     reinterpret_cast<void*>(a3),
                     reinterpret_cast<void*>(a4),
                     reinterpret_cast<void*>(a5),
                     static_cast<int>(CUDA_SUCCESS));
    }
    return CUDA_SUCCESS;
}

CUresult cuExportTable42Stub(std::uintptr_t a0,
                             std::uintptr_t a1,
                             std::uintptr_t a2,
                             std::uintptr_t a3,
                             std::uintptr_t a4,
                             std::uintptr_t a5) {
    if (logExportTableCalls()) {
        std::fprintf(stderr,
                     "[vgpu-cuGetExportTable] call uuid=42d85a81 a0=%p a1=%p a2=%p a3=%p a4=%p a5=%p rc=%d\n",
                     reinterpret_cast<void*>(a0),
                     reinterpret_cast<void*>(a1),
                     reinterpret_cast<void*>(a2),
                     reinterpret_cast<void*>(a3),
                     reinterpret_cast<void*>(a4),
                     reinterpret_cast<void*>(a5),
                     static_cast<int>(CUDA_SUCCESS));
    }
    return CUDA_SUCCESS;
}

const void* fakeExportTableForUuid(const void* pExportTableId) {
    if (pExportTableId == nullptr) {
        return nullptr;
    }

    static constexpr unsigned char kUuid69[16] = {
        0x69, 0xa2, 0x8e, 0x9f, 0x24, 0x49, 0x8b, 0x47,
        0xb0, 0x30, 0xc5, 0xda, 0x33, 0x0e, 0xa4, 0xe9,
    };
    static constexpr unsigned char kUuid6b[16] = {
        0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a,
        0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9,
    };
    static constexpr unsigned char kUuida0[16] = {
        0xa0, 0x94, 0x79, 0x8c, 0x2e, 0x74, 0x2e, 0x74,
        0x93, 0xf2, 0x08, 0x00, 0x20, 0x0c, 0x0a, 0x66,
    };
    static constexpr unsigned char kUuid42[16] = {
        0x42, 0xd8, 0x5a, 0x81, 0x23, 0xf6, 0xcb, 0x47,
        0x82, 0x98, 0xf6, 0xe7, 0x8a, 0x3a, 0xec, 0xdc,
    };
    static constexpr unsigned char kUuidc693[16] = {
        0xc6, 0x93, 0x33, 0x6e, 0x11, 0x21, 0xdf, 0x11,
        0xa8, 0xc3, 0x68, 0xf3, 0x55, 0xd8, 0x95, 0x93,
    };

    static std::uintptr_t table69[34] = {};
    static std::uintptr_t table6b[13] = {};
    static std::uintptr_t tablea0[7] = {};
    static std::uintptr_t table42[3] = {};
    static std::uintptr_t tablec693[10] = {};
    static bool initialized = false;
    if (!initialized) {
        table69[0] = static_cast<std::uintptr_t>(0x110);
        for (size_t i = 1; i < 34; ++i) {
            table69[i] = reinterpret_cast<std::uintptr_t>(&cuExportTable69Stub);
        }

        table6b[0] = static_cast<std::uintptr_t>(0x68);
        for (size_t i = 1; i < 13; ++i) {
            table6b[i] = reinterpret_cast<std::uintptr_t>(&cuExportTable6bStub);
        }

        tablea0[0] = static_cast<std::uintptr_t>(0x38);
        for (size_t i = 1; i < 7; ++i) {
            tablea0[i] = reinterpret_cast<std::uintptr_t>(&cuExportTablea0Stub);
        }

        table42[0] = static_cast<std::uintptr_t>(0x18);
        for (size_t i = 1; i < 3; ++i) {
            table42[i] = reinterpret_cast<std::uintptr_t>(&cuExportTable42Stub);
        }

        for (size_t i = 0; i < 10; ++i) {
            tablec693[i] = reinterpret_cast<std::uintptr_t>(&cuExportTable6bStub);
        }
        initialized = true;
    }

    const auto* bytes = reinterpret_cast<const unsigned char*>(pExportTableId);
    if (std::memcmp(bytes, kUuid69, 16) == 0) {
        return table69;
    }
    if (std::memcmp(bytes, kUuid6b, 16) == 0) {
        return table6b;
    }
    if (std::memcmp(bytes, kUuida0, 16) == 0) {
        return tablea0;
    }
    if (std::memcmp(bytes, kUuid42, 16) == 0) {
        return table42;
    }
    if (std::memcmp(bytes, kUuidc693, 16) == 0) {
        return tablec693;
    }
    return nullptr;
}

// ── Fake handle counter ──────────────────────────────────────────────────────
std::atomic<std::uintptr_t> g_next_fake_handle{0x10000};

void* makeFakeHandle() {
    return reinterpret_cast<void*>(g_next_fake_handle.fetch_add(1));
}

void*& cachedRealCudaHandle() {
    static void* handle = nullptr;
    return handle;
}

void closeRealCudaHandle() {
    void*& handle = cachedRealCudaHandle();
    if (handle != nullptr) {
        dlclose(handle);
        handle = nullptr;
    }
}

void* realCudaHandle() {
    void*& handle = cachedRealCudaHandle();
    if (handle) return handle;

    using DlopenFn = void* (*)(const char*, int);
    static DlopenFn original_dlopen = []() -> DlopenFn {
        constexpr const char* kGlibcSymbolVersion = "GLIBC_2.2.5";
        void* sym = dlvsym(RTLD_NEXT, "dlopen", kGlibcSymbolVersion);
        if (!sym) {
            sym = dlvsym(RTLD_DEFAULT, "dlopen", kGlibcSymbolVersion);
        }
        return reinterpret_cast<DlopenFn>(sym);
    }();

    const char* candidates[] = {
        "/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "libcuda.so.1",
        nullptr,
    };
    for (int i = 0; candidates[i] != nullptr && handle == nullptr; ++i) {
        if (original_dlopen) {
            handle = original_dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        }
    }

    if (handle != nullptr) {
        static bool registered_cleanup = []() {
            std::atexit(closeRealCudaHandle);
            return true;
        }();
        (void)registered_cleanup;
    }

    return handle;
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

    // CUDA 12+ fatbin header variant:
    //   u32 magic, u32 version, u64 data_size, u32 unknown, u32 header_size
    uint32_t version = 0;
    uint64_t data_size64 = 0;
    uint32_t header_size_v2 = 0;
    std::memcpy(&version,        p + 4,  sizeof(version));
    std::memcpy(&data_size64,    p + 8,  sizeof(data_size64));
    std::memcpy(&header_size_v2, p + 20, sizeof(header_size_v2));
    if (version == 0x00100001u && header_size_v2 >= 16 && header_size_v2 <= (1u << 20)) {
        size_t total_v2 = static_cast<size_t>(data_size64 + static_cast<uint64_t>(header_size_v2));
        if (total_v2 >= header_size_v2) {
            return total_v2;
        }
    }

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

std::vector<std::uint8_t> makePackedArgsFromTotalBytes(void** kernelParams, uint32_t total_param_bytes) {
    std::vector<std::uint8_t> out;
    if (kernelParams == nullptr || total_param_bytes == 0) {
        return out;
    }

    out.assign(total_param_bytes, 0);
    const size_t slot = sizeof(std::uint64_t);
    size_t max_args = (static_cast<size_t>(total_param_bytes) + slot - 1) / slot;
    if (max_args > 64) {
        max_args = 64;
    }

    for (size_t i = 0; i < max_args; ++i) {
        void* arg_ptr = kernelParams[i];
        if (arg_ptr == nullptr && i > 0) {
            break;
        }
        if (arg_ptr == nullptr) {
            continue;
        }

        const size_t off = i * slot;
        if (off >= out.size()) {
            break;
        }
        const size_t n = std::min(slot, out.size() - off);
        std::memcpy(out.data() + off, arg_ptr, n);
    }

    return out;
}

std::vector<std::uint8_t> makeArgumentSlotsUnknown(void** kernelParams, size_t max_slots = 12) {
    auto isReadable = [](const void* p, size_t len) -> bool {
        if (p == nullptr || len == 0) {
            return false;
        }

        struct Range { std::uintptr_t lo; std::uintptr_t hi; };
        static thread_local std::vector<Range> ranges;
        static thread_local bool loaded = false;
        if (!loaded) {
            loaded = true;
            std::ifstream maps("/proc/self/maps");
            std::string line;
            while (std::getline(maps, line)) {
                if (line.size() < 20) {
                    continue;
                }
                size_t dash = line.find('-');
                size_t sp = line.find(' ');
                if (dash == std::string::npos || sp == std::string::npos || dash >= sp) {
                    continue;
                }
                if (sp + 1 >= line.size() || line[sp + 1] != 'r') {
                    continue;
                }
                std::uintptr_t lo = static_cast<std::uintptr_t>(std::strtoull(line.substr(0, dash).c_str(), nullptr, 16));
                std::uintptr_t hi = static_cast<std::uintptr_t>(std::strtoull(line.substr(dash + 1, sp - dash - 1).c_str(), nullptr, 16));
                if (lo < hi) {
                    ranges.push_back({lo, hi});
                }
            }
        }

        const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(p);
        const std::uintptr_t end = addr + len;
        if (end < addr) {
            return false;
        }
        for (const auto& r : ranges) {
            if (addr >= r.lo && end <= r.hi) {
                return true;
            }
        }
        return false;
    };

    std::vector<std::uint8_t> out;
    if (kernelParams == nullptr || max_slots == 0) {
        return out;
    }

    const size_t slot = sizeof(std::uint64_t);
    out.assign(max_slots * slot, 0);

    size_t used = 0;
    for (size_t i = 0; i < max_slots; ++i) {
        void* arg_ptr = kernelParams[i];
        if (arg_ptr == nullptr && i > 0) {
            break;
        }
        if (arg_ptr == nullptr) {
            used = i + 1;
            continue;
        }
        if (!isReadable(arg_ptr, slot)) {
            break;
        }
        std::memcpy(out.data() + i * slot, arg_ptr, slot);
        used = i + 1;
    }

    out.resize(used * slot);
    return out;
}

}  // namespace
}  // namespace vgpu

// ============================================================================
// Exported cu* symbols
// ============================================================================

extern "C" {

// ── Init / device ────────────────────────────────────────────────────────────

CUresult cuInit(unsigned int /*flags*/) {
    if (vgpu::logCtxOps()) {
        std::fprintf(stderr, "[vgpu-ctx] cuInit -> CUDA_SUCCESS\n");
    }
    return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion(int* driverVersion) {
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaDriverGetVersion, nullptr, 0);
    if (r.status == cudaSuccess && driverVersion) {
        int version = static_cast<int>(r.aux_u64);
        const char* forced = std::getenv("VGPU_FAKE_DRIVER_VERSION");
        if (forced && forced[0] != '\0') {
            version = std::atoi(forced);
        } else if (version > 12080) {
            // Keep compatibility with many CUDA 12.x user-space stacks.
            version = 12080;
        }
        *driverVersion = version;
    }
    CUresult rc = r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    if (vgpu::logCtxOps()) {
        std::fprintf(stderr,
                     "[vgpu-ctx] cuDriverGetVersion rc=%d ver=%d\n",
                     static_cast<int>(rc),
                     driverVersion ? *driverVersion : -1);
    }
    return rc;
}

CUresult cuDeviceGetCount(int* count) {
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaGetDeviceCount, nullptr, 0);
    if (r.status == cudaSuccess && count)
        *count = static_cast<int>(r.aux_u64);
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    if (device) *device = ordinal;
    if (vgpu::logCtxOps()) {
        std::fprintf(stderr,
                     "[vgpu-ctx] cuDeviceGet ordinal=%d -> dev=%d\n",
                     ordinal,
                     device ? static_cast<int>(*device) : -1);
    }
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char* name, int len, CUdevice /*dev*/) {
    if (name && len > 0) std::strncpy(name, "vGPU Remote Device", static_cast<size_t>(len - 1));
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int* pi, int attrib, CUdevice dev) {
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    if (vgpu::queryDeviceAttributeFromServer(attrib, static_cast<int>(dev), pi)) {
        return CUDA_SUCCESS;
    }

    switch (attrib) {
        case 1:   // CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
            *pi = 1024;
            break;
        case 13:  // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
            *pi = 80;
            break;
        case 75:  // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
            *pi = 9;
            break;
        case 76:  // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
            *pi = 0;
            break;
        default:
            *pi = 1;
            break;
    }
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetP2PAttribute(int* value, int /*attrib*/, CUdevice /*src*/, CUdevice /*dst*/) {
    if (value) *value = 0;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetByPCIBusId(CUdevice* dev, const char* /*pciBusId*/) {
    if (dev) *dev = 0;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice /*dev*/) {
    if (pciBusId && len > 0) {
        std::strncpy(pciBusId, "0000:00:00.0", static_cast<size_t>(len - 1));
    }
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetUuid(void* uuid, CUdevice /*dev*/) {
    if (uuid) {
        std::memset(uuid, 0, 16);
    }
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t* maxWidth,
                                            int /*format*/,
                                            unsigned int /*numChannels*/,
                                            CUdevice /*dev*/) {
    if (maxWidth) *maxWidth = static_cast<size_t>(1) << 27;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetDefaultMemPool(void** pool, CUdevice /*dev*/) {
    if (pool) *pool = vgpu::makeFakeHandle();
    return CUDA_SUCCESS;
}

CUresult cuDeviceSetMemPool(CUdevice /*dev*/, void* /*pool*/) {
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetMemPool(void** pool, CUdevice /*dev*/) {
    if (pool) *pool = vgpu::makeFakeHandle();
    return CUDA_SUCCESS;
}

CUresult cuFlushGPUDirectRDMAWrites(int /*target*/, int /*scope*/) {
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

CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
    return cuDeviceTotalMem_v2(bytes, dev);
}

// ── Context (stub – no real context on client) ───────────────────────────────

CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int /*flags*/, CUdevice dev) {
    vgpu::g_drv_device = static_cast<int>(dev);
    using CuInitFn = CUresult (*)(unsigned int);
    using CuDeviceGetFn = CUresult (*)(CUdevice*, int);
    using CuDevicePrimaryCtxRetainFn = CUresult (*)(CUcontext*, CUdevice);
    using CuCtxSetCurrentFn = CUresult (*)(CUcontext);

    void* real = vgpu::realCudaHandle();
    if (real != nullptr) {
        auto* cuInitFn = reinterpret_cast<CuInitFn>(dlsym(real, "cuInit"));
        auto* cuDeviceGetFn = reinterpret_cast<CuDeviceGetFn>(dlsym(real, "cuDeviceGet"));
        auto* cuDevicePrimaryCtxRetainFn =
            reinterpret_cast<CuDevicePrimaryCtxRetainFn>(dlsym(real, "cuDevicePrimaryCtxRetain"));
        auto* cuCtxSetCurrentFn = reinterpret_cast<CuCtxSetCurrentFn>(dlsym(real, "cuCtxSetCurrent"));
        if (cuInitFn && cuDeviceGetFn && cuDevicePrimaryCtxRetainFn && cuCtxSetCurrentFn) {
            if (cuInitFn(0) == CUDA_SUCCESS) {
                CUdevice real_dev = 0;
                if (cuDeviceGetFn(&real_dev, static_cast<int>(dev)) == CUDA_SUCCESS) {
                    CUcontext real_ctx = nullptr;
                    if (cuDevicePrimaryCtxRetainFn(&real_ctx, real_dev) == CUDA_SUCCESS && real_ctx != nullptr) {
                        cuCtxSetCurrentFn(real_ctx);
                        if (pctx) {
                            *pctx = real_ctx;
                        }
                        if (vgpu::logCtxOps()) {
                            std::fprintf(stderr,
                                         "[vgpu-ctx] cuCtxCreate_v2 dev=%d -> real_ctx=%p\n",
                                         static_cast<int>(dev),
                                         real_ctx);
                        }
                        return CUDA_SUCCESS;
                    }
                }
            }
        }
    }

    if (pctx) *pctx = reinterpret_cast<CUcontext>(vgpu::makeFakeHandle());
    if (vgpu::logCtxOps()) {
        std::fprintf(stderr,
                     "[vgpu-ctx] cuCtxCreate_v2 dev=%d -> fake_ctx=%p\n",
                     static_cast<int>(dev),
                     pctx ? *pctx : nullptr);
    }
    return CUDA_SUCCESS;
}

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    return cuCtxCreate_v2(pctx, flags, dev);
}

CUresult cuCtxDestroy_v2(CUcontext /*ctx*/) { return CUDA_SUCCESS; }
CUresult cuCtxDestroy(CUcontext ctx) { return cuCtxDestroy_v2(ctx); }
CUresult cuCtxGetCurrent(CUcontext* pctx) {
    using CuCtxGetCurrentFn = CUresult (*)(CUcontext*);
    using CuInitFn = CUresult (*)(unsigned int);
    using CuDeviceGetFn = CUresult (*)(CUdevice*, int);
    using CuDevicePrimaryCtxRetainFn = CUresult (*)(CUcontext*, CUdevice);
    using CuCtxSetCurrentFn = CUresult (*)(CUcontext);

    void* real = vgpu::realCudaHandle();
    if (real != nullptr) {
        auto* cuCtxGetCurrentFn = reinterpret_cast<CuCtxGetCurrentFn>(dlsym(real, "cuCtxGetCurrent"));
        if (cuCtxGetCurrentFn) {
            CUcontext cur = nullptr;
            if (cuCtxGetCurrentFn(&cur) == CUDA_SUCCESS && cur != nullptr) {
                if (pctx) *pctx = cur;
                if (vgpu::logCtxOps()) {
                    std::fprintf(stderr, "[vgpu-ctx] cuCtxGetCurrent -> real current=%p\n", cur);
                }
                return CUDA_SUCCESS;
            }
        }

        auto* cuInitFn = reinterpret_cast<CuInitFn>(dlsym(real, "cuInit"));
        auto* cuDeviceGetFn = reinterpret_cast<CuDeviceGetFn>(dlsym(real, "cuDeviceGet"));
        auto* cuDevicePrimaryCtxRetainFn =
            reinterpret_cast<CuDevicePrimaryCtxRetainFn>(dlsym(real, "cuDevicePrimaryCtxRetain"));
        auto* cuCtxSetCurrentFn = reinterpret_cast<CuCtxSetCurrentFn>(dlsym(real, "cuCtxSetCurrent"));
        if (cuInitFn && cuDeviceGetFn && cuDevicePrimaryCtxRetainFn && cuCtxSetCurrentFn) {
            if (cuInitFn(0) == CUDA_SUCCESS) {
                CUdevice real_dev = 0;
                if (cuDeviceGetFn(&real_dev, (vgpu::g_drv_device >= 0) ? vgpu::g_drv_device : 0) == CUDA_SUCCESS) {
                    CUcontext real_ctx = nullptr;
                    if (cuDevicePrimaryCtxRetainFn(&real_ctx, real_dev) == CUDA_SUCCESS && real_ctx != nullptr) {
                        cuCtxSetCurrentFn(real_ctx);
                        if (pctx) *pctx = real_ctx;
                        if (vgpu::logCtxOps()) {
                            std::fprintf(stderr,
                                         "[vgpu-ctx] cuCtxGetCurrent -> retained real_ctx=%p\n",
                                         real_ctx);
                        }
                        return CUDA_SUCCESS;
                    }
                }
            }
        }
    }

    if (pctx) *pctx = reinterpret_cast<CUcontext>(static_cast<uintptr_t>(0xC7AC0001));
    if (vgpu::logCtxOps()) {
        std::fprintf(stderr,
                     "[vgpu-ctx] cuCtxGetCurrent -> synthetic=%p\n",
                     pctx ? *pctx : nullptr);
    }
    return CUDA_SUCCESS;
}
CUresult cuCtxSetCurrent(CUcontext ctx) {
    using CuCtxSetCurrentFn = CUresult (*)(CUcontext);
    void* real = vgpu::realCudaHandle();
    if (real != nullptr) {
        auto* fn = reinterpret_cast<CuCtxSetCurrentFn>(dlsym(real, "cuCtxSetCurrent"));
        if (fn != nullptr) {
            CUresult rc = fn(ctx);
            if (rc == CUDA_SUCCESS) {
                if (vgpu::logCtxOps()) {
                    std::fprintf(stderr,
                                 "[vgpu-ctx] cuCtxSetCurrent real ctx=%p rc=0\n",
                                 ctx);
                }
                return CUDA_SUCCESS;
            }
        }
    }
    (void)ctx;
    if (vgpu::logCtxOps()) {
        std::fprintf(stderr,
                     "[vgpu-ctx] cuCtxSetCurrent shim ctx=%p rc=0\n",
                     ctx);
    }
    return CUDA_SUCCESS;
}
CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
    return cuCtxCreate_v2(pctx, 0, dev);
}
CUresult cuDevicePrimaryCtxSetFlags(CUdevice /*dev*/, unsigned int /*flags*/) {
    return CUDA_SUCCESS;
}
CUresult cuDevicePrimaryCtxGetState(CUdevice /*dev*/, unsigned int* flags, int* active) {
    if (flags) *flags = 0;
    if (active) *active = 1;
    return CUDA_SUCCESS;
}
CUresult cuDevicePrimaryCtxReset(CUdevice /*dev*/) { return CUDA_SUCCESS; }
CUresult cuDevicePrimaryCtxRelease_v2(CUdevice /*dev*/) { return CUDA_SUCCESS; }
CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    return cuDevicePrimaryCtxRelease_v2(dev);
}
CUresult cuCtxGetFlags(unsigned int* flags) {
    if (flags) *flags = 0;
    return CUDA_SUCCESS;
}
CUresult cuCtxDetach(CUcontext ctx) { return cuCtxDestroy_v2(ctx); }
CUresult cuCtxGetApiVersion(CUcontext /*ctx*/, unsigned int* version) {
    if (version) *version = 12080;
    return CUDA_SUCCESS;
}
CUresult cuCtxGetDevice(CUdevice* dev) {
    using CuCtxGetDeviceFn = CUresult (*)(CUdevice*);
    void* real = vgpu::realCudaHandle();
    if (real != nullptr) {
        auto* fn = reinterpret_cast<CuCtxGetDeviceFn>(dlsym(real, "cuCtxGetDevice"));
        if (fn != nullptr) {
            CUresult rc = fn(dev);
            if (rc == CUDA_SUCCESS) {
                if (vgpu::logCtxOps()) {
                    std::fprintf(stderr,
                                 "[vgpu-ctx] cuCtxGetDevice real dev=%d\n",
                                 dev ? static_cast<int>(*dev) : -1);
                }
                return CUDA_SUCCESS;
            }
        }
    }
    if (dev) *dev = static_cast<CUdevice>(vgpu::g_drv_device);
    if (vgpu::logCtxOps()) {
        std::fprintf(stderr,
                     "[vgpu-ctx] cuCtxGetDevice shim dev=%d\n",
                     dev ? static_cast<int>(*dev) : -1);
    }
    return CUDA_SUCCESS;
}
CUresult cuCtxGetLimit(size_t* pvalue, int /*limit*/) {
    if (pvalue) *pvalue = 0;
    return CUDA_SUCCESS;
}
CUresult cuCtxSetLimit(int /*limit*/, size_t /*value*/) { return CUDA_SUCCESS; }
CUresult cuCtxGetCacheConfig(int* pconfig) {
    if (pconfig) *pconfig = 0;
    return CUDA_SUCCESS;
}
CUresult cuCtxSetCacheConfig(int /*config*/) { return CUDA_SUCCESS; }
CUresult cuCtxGetSharedMemConfig(int* pconfig) {
    if (pconfig) *pconfig = 0;
    return CUDA_SUCCESS;
}
CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    if (leastPriority) *leastPriority = 0;
    if (greatestPriority) *greatestPriority = 0;
    return CUDA_SUCCESS;
}
CUresult cuCtxSetSharedMemConfig(int /*config*/) { return CUDA_SUCCESS; }
CUresult cuCtxResetPersistingL2Cache() { return CUDA_SUCCESS; }
CUresult cuCtxPopCurrent(CUcontext* pctx) {
    if (pctx) *pctx = reinterpret_cast<CUcontext>(static_cast<uintptr_t>(0xC7AC0001));
    return CUDA_SUCCESS;
}
CUresult cuCtxPushCurrent(CUcontext /*ctx*/) { return CUDA_SUCCESS; }
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

    // Send raw fatbin to server
    vgpu::RpcResult r = vgpu::drvCallDrv(
        vgpu::RpcDrvOp::kCuModuleLoadData, nullptr, 0, raw, raw_size);

    if (r.status != cudaSuccess) return CUDA_ERROR_UNKNOWN;

    std::uint64_t module_id = r.aux_u64;
    auto kernel_infos = vgpu::parseFatbin(raw, raw_size);
    for (const auto& ki : kernel_infos) {
        vgpu::globalKernelRegistry().addParamInfo(module_id, ki.mangled_name, ki.params);
    }

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

CUresult cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) {
    return doModuleLoad(module, fatCubin);
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

CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes,
                           CUmodule /*hmod*/, const char* /*name*/) {
    if (dptr) *dptr = 0;
    if (bytes) *bytes = 0;
    return CUDA_SUCCESS;
}

CUresult cuModuleGetTexRef(void** pTexRef, CUmodule /*hmod*/, const char* /*name*/) {
    if (pTexRef) *pTexRef = nullptr;
    return CUDA_SUCCESS;
}

CUresult cuModuleGetSurfRef(void** pSurfRef, CUmodule /*hmod*/, const char* /*name*/) {
    if (pSurfRef) *pSurfRef = nullptr;
    return CUDA_SUCCESS;
}

CUresult cuModuleGetLoadingMode(int* mode) {
    if (mode) *mode = 0;
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
        std::vector<vgpu::ParamInfo> cached;
        if (vgpu::globalKernelRegistry().findParamInfo(module_id, name, &cached)) {
            ke.params = std::move(cached);
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
    vgpu::KernelEntry ke{};
    const bool has_kernel =
        vgpu::globalKernelRegistry().findDriverFunc(reinterpret_cast<void*>(hfunc), &ke);

    // Keep attributes non-zero where frameworks use them for launch heuristics.
    switch (attrib) {
        case 0:  // CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
            *pi = 1024;
            return CUDA_SUCCESS;
        case 4:  // CU_FUNC_ATTRIBUTE_NUM_REGS
            *pi = 64;
            return CUDA_SUCCESS;
        case 5:  // CU_FUNC_ATTRIBUTE_PTX_VERSION
            *pi = 80;
            return CUDA_SUCCESS;
        case 6:  // CU_FUNC_ATTRIBUTE_BINARY_VERSION
            *pi = 90;
            return CUDA_SUCCESS;
        case 8:  // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            *pi = 96 * 1024;
            return CUDA_SUCCESS;
        default:
            break;
    }

    // CU_FUNC_ATTRIBUTE_PARAM_SIZE_BYTES
    if (attrib == CU_FUNC_ATTR_PARAM_SIZE_BYTES && has_kernel) {
        *pi = static_cast<int>(ke.total_param_bytes);
        return CUDA_SUCCESS;
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

    vgpu::KernelEntry ke{};
    if (!vgpu::globalKernelRegistry().findDriverFunc(reinterpret_cast<void*>(f), &ke)) {
        return CUDA_ERROR_INVALID_VALUE;
    }

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

    if (!arg_buf && kernelParams && !ke.params.empty()) {
        packed = vgpu::packArgs(ke.params, kernelParams);
        arg_buf      = packed.data();
        arg_buf_size = packed.size();
    } else if (!arg_buf && kernelParams && ke.total_param_bytes > 0) {
        packed = vgpu::makePackedArgsFromTotalBytes(kernelParams, ke.total_param_bytes);
        arg_buf      = packed.data();
        arg_buf_size = packed.size();
    } else if (!arg_buf && kernelParams) {
        packed = vgpu::makeArgumentSlotsUnknown(kernelParams);
        arg_buf      = packed.data();
        arg_buf_size = packed.size();
    }

    // Build RPC request
    vgpu::RpcCuLaunchKernelReq req{};
    req.func_id          = ke.func_id;
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

CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    return cuMemAlloc_v2(dptr, bytesize);
}

CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int /*flags*/) {
    return cuMemAlloc_v2(dptr, bytesize);
}

CUresult cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch,
                         size_t WidthInBytes, size_t Height,
                         unsigned int /*ElementSizeBytes*/) {
    if (pPitch) *pPitch = WidthInBytes;
    return cuMemAlloc_v2(dptr, WidthInBytes * Height);
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    vgpu::RpcFreeReq req{};
    req.dev_ptr = static_cast<std::uint64_t>(dptr);
    vgpu::RpcResult r = vgpu::drvCallDrv(vgpu::RpcDrvOp::kCuMemFree,
                                          &req, sizeof(req));
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult cuMemFree(CUdeviceptr dptr) {
    return cuMemFree_v2(dptr);
}

CUresult cuMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaMemGetInfo, nullptr, 0);
    if (r.status == cudaSuccess && r.payload.size() >= sizeof(vgpu::RpcMemGetInfoRsp)) {
        const auto* info = reinterpret_cast<const vgpu::RpcMemGetInfoRsp*>(r.payload.data());
        if (free_bytes) *free_bytes = static_cast<size_t>(info->free_bytes);
        if (total_bytes) *total_bytes = static_cast<size_t>(info->total_bytes);
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_UNKNOWN;
}

CUresult cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) {
    if (pbase) *pbase = dptr;
    if (psize) *psize = 0;
    return CUDA_SUCCESS;
}

CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int /*flags*/) {
    if (!pp) return CUDA_ERROR_INVALID_VALUE;
    *pp = std::malloc(bytesize);
    return *pp ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

CUresult cuMemFreeHost(void* p) {
    std::free(p);
    return CUDA_SUCCESS;
}

CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int /*flags*/) {
    if (!pdptr) return CUDA_ERROR_INVALID_VALUE;
    *pdptr = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(p));
    return CUDA_SUCCESS;
}

CUresult cuMemHostGetFlags(unsigned int* pFlags, void* /*p*/) {
    if (pFlags) *pFlags = 0;
    return CUDA_SUCCESS;
}

CUresult cuMemHostRegister(void* /*p*/, size_t /*bytesize*/, unsigned int /*flags*/) {
    return CUDA_SUCCESS;
}

CUresult cuMemHostUnregister(void* /*p*/) {
    return CUDA_SUCCESS;
}

CUresult cuPointerGetAttribute(void* data, int /*attribute*/, CUdeviceptr /*ptr*/) {
    if (data) {
        std::memset(data, 0, sizeof(std::uintptr_t));
    }
    return CUDA_SUCCESS;
}

CUresult cuPointerGetAttributes(unsigned int numAttributes,
                                int* /*attributes*/,
                                void** data,
                                CUdeviceptr /*ptr*/) {
    if (data) {
        for (unsigned int i = 0; i < numAttributes; ++i) {
            if (data[i]) {
                std::memset(data[i], 0, sizeof(std::uintptr_t));
            }
        }
    }
    return CUDA_SUCCESS;
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

CUresult cuMemcpyHtoD(CUdeviceptr dst, const void* src, size_t count) {
    return cuMemcpyHtoD_v2(dst, src, count);
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

CUresult cuMemcpyDtoH(void* dst, CUdeviceptr src, size_t count) {
    return cuMemcpyDtoH_v2(dst, src, count);
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

CUresult cuMemcpyDtoD(CUdeviceptr dst, CUdeviceptr src, size_t count) {
    return cuMemcpyDtoD_v2(dst, src, count);
}

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t count) {
    return cuMemcpyDtoD_v2(dst, src, count);
}

// Async variants – for simplicity forward synchronously
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void* src,
                               size_t count, CUstream_drv /*stream*/) {
    return cuMemcpyHtoD_v2(dst, src, count);
}
CUresult cuMemcpyHtoDAsync(CUdeviceptr dst, const void* src,
                           size_t count, CUstream_drv stream) {
    return cuMemcpyHtoDAsync_v2(dst, src, count, stream);
}
CUresult cuMemcpyDtoHAsync_v2(void* dst, CUdeviceptr src,
                               size_t count, CUstream_drv /*stream*/) {
    return cuMemcpyDtoH_v2(dst, src, count);
}
CUresult cuMemcpyDtoHAsync(void* dst, CUdeviceptr src,
                           size_t count, CUstream_drv stream) {
    return cuMemcpyDtoHAsync_v2(dst, src, count, stream);
}
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src,
                               size_t count, CUstream_drv /*stream*/) {
    return cuMemcpyDtoD_v2(dst, src, count);
}
CUresult cuMemcpyDtoDAsync(CUdeviceptr dst, CUdeviceptr src,
                           size_t count, CUstream_drv stream) {
    return cuMemcpyDtoDAsync_v2(dst, src, count, stream);
}

CUresult cuMemsetD8_v2(CUdeviceptr dst, unsigned char uc, size_t n) {
    vgpu::RpcMemsetReq req{};
    req.dst   = static_cast<std::uint64_t>(dst);
    req.value = static_cast<std::int32_t>(uc);
    req.count = static_cast<std::uint64_t>(n);
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaMemset, &req, sizeof(req));
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
CUresult cuMemsetD8(CUdeviceptr dst, unsigned char uc, size_t n) {
    return cuMemsetD8_v2(dst, uc, n);
}
CUresult cuMemsetD32_v2(CUdeviceptr dst, unsigned int ui, size_t n) {
    return cuMemsetD8_v2(dst, static_cast<unsigned char>(ui & 0xFF), n * 4);
}
CUresult cuMemsetD32(CUdeviceptr dst, unsigned int ui, size_t n) {
    return cuMemsetD32_v2(dst, ui, n);
}

CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream_drv /*hStream*/) {
    return cuMemAlloc_v2(dptr, bytesize);
}

CUresult cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, void* /*pool*/, CUstream_drv hStream) {
    return cuMemAllocAsync(dptr, bytesize, hStream);
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream_drv /*hStream*/) {
    return cuMemFree_v2(dptr);
}

CUresult cuMemPoolTrimTo(void* /*pool*/, size_t /*minBytesToKeep*/) {
    return CUDA_SUCCESS;
}

CUresult cuMemPoolSetAttribute(void* /*pool*/, int /*attr*/, void* /*value*/) {
    return CUDA_SUCCESS;
}

CUresult cuMemPoolGetAttribute(void* /*pool*/, int /*attr*/, void* value) {
    if (value) {
        std::memset(value, 0, sizeof(std::uint64_t));
    }
    return CUDA_SUCCESS;
}

CUresult cuMemPoolSetAccess(void* /*pool*/, const void* /*map*/, size_t /*count*/) {
    return CUDA_SUCCESS;
}

CUresult cuMemPoolGetAccess(unsigned long long* flags, void* /*pool*/, const void* /*location*/) {
    if (flags) {
        *flags = 0;
    }
    return CUDA_SUCCESS;
}

CUresult cuMemPoolCreate(void** pool, const void* /*poolProps*/) {
    if (pool) {
        *pool = vgpu::makeFakeHandle();
    }
    return CUDA_SUCCESS;
}

CUresult cuMemPoolDestroy(void* /*pool*/) {
    return CUDA_SUCCESS;
}

CUresult cuMemPoolExportToShareableHandle(void* /*shareableHandle*/, void* /*pool*/, int /*handleType*/, unsigned int /*flags*/) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuMemPoolImportFromShareableHandle(void** pool, void* /*shareableHandle*/, int /*handleType*/, unsigned int /*flags*/) {
    if (pool) {
        *pool = vgpu::makeFakeHandle();
    }
    return CUDA_SUCCESS;
}

CUresult cuMemPoolExportPointer(void* /*exportData*/, CUdeviceptr /*ptr*/) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuMemPoolImportPointer(CUdeviceptr* ptr, void* /*pool*/, void* /*exportData*/) {
    if (ptr) {
        *ptr = 0;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t count, CUstream_drv stream) {
    return cuMemcpyDtoDAsync_v2(dst, src, count, stream);
}

CUresult cuMemcpyPeer(CUdeviceptr dst, CUcontext /*dstContext*/, CUdeviceptr src, CUcontext /*srcContext*/, size_t count) {
    return cuMemcpyDtoD_v2(dst, src, count);
}

CUresult cuMemcpyPeerAsync(CUdeviceptr dst,
                           CUcontext /*dstContext*/,
                           CUdeviceptr src,
                           CUcontext /*srcContext*/,
                           size_t count,
                           CUstream_drv stream) {
    return cuMemcpyDtoDAsync_v2(dst, src, count, stream);
}

CUresult cuMemsetD8Async(CUdeviceptr dst, unsigned char uc, size_t n, CUstream_drv /*stream*/) {
    return cuMemsetD8_v2(dst, uc, n);
}

CUresult cuMemsetD2D8(CUdeviceptr dst, size_t pitch, unsigned char uc, size_t width, size_t height) {
    (void)pitch;
    return cuMemsetD8_v2(dst, uc, width * height);
}

CUresult cuMemsetD2D8Async(CUdeviceptr dst,
                           size_t pitch,
                           unsigned char uc,
                           size_t width,
                           size_t height,
                           CUstream_drv /*stream*/) {
    (void)pitch;
    return cuMemsetD8_v2(dst, uc, width * height);
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

CUresult cuStreamCreateWithPriority(CUstream_drv* phStream, unsigned int flags, int /*priority*/) {
    return cuStreamCreate(phStream, flags);
}

CUresult cuStreamDestroy_v2(CUstream_drv hStream) {
    vgpu::RpcStreamDestroyReq req{};
    req.stream = reinterpret_cast<std::uint64_t>(hStream);
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaStreamDestroy, &req, sizeof(req));
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult cuStreamDestroy(CUstream_drv hStream) {
    return cuStreamDestroy_v2(hStream);
}

CUresult cuStreamSynchronize(CUstream_drv hStream) {
    vgpu::RpcStreamSyncReq req{};
    req.stream = reinterpret_cast<std::uint64_t>(hStream);
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaStreamSynchronize, &req, sizeof(req));
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult cuStreamQuery(CUstream_drv hStream) {
    vgpu::RpcStreamQueryReq req{};
    req.stream = reinterpret_cast<std::uint64_t>(hStream);
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaStreamQuery, &req, sizeof(req));
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}

CUresult cuStreamGetPriority(CUstream_drv /*hStream*/, int* priority) {
    if (priority) *priority = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetFlags(CUstream_drv /*hStream*/, unsigned int* flags) {
    if (flags) *flags = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetCtx(CUstream_drv /*hStream*/, CUcontext* pctx) {
    if (pctx) {
        *pctx = reinterpret_cast<CUcontext>(static_cast<uintptr_t>(0xC7AC0001));
    }
    return CUDA_SUCCESS;
}

CUresult cuStreamGetDevice(CUstream_drv /*hStream*/, CUdevice* dev) {
    if (dev) {
        *dev = static_cast<CUdevice>(vgpu::g_drv_device);
    }
    return CUDA_SUCCESS;
}

CUresult cuStreamWaitEvent(CUstream_drv hStream, void* hEvent, unsigned int flags) {
    vgpu::RpcStreamWaitEventReq req{};
    req.stream = reinterpret_cast<std::uint64_t>(hStream);
    req.event = reinterpret_cast<std::uint64_t>(hEvent);
    req.flags = flags;
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaStreamWaitEvent, &req, sizeof(req));
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult cuEventCreate(void** phEvent, unsigned int flags) {
    vgpu::RpcEventCreateReq req{};
    req.flags = flags;
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaEventCreate, &req, sizeof(req));
    if (r.status == cudaSuccess && phEvent) {
        *phEvent = reinterpret_cast<void*>(r.aux_u64);
    }
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult cuEventDestroy_v2(void* hEvent) {
    vgpu::RpcEventDestroyReq req{};
    req.event = reinterpret_cast<std::uint64_t>(hEvent);
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaEventDestroy, &req, sizeof(req));
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult cuEventDestroy(void* hEvent) {
    return cuEventDestroy_v2(hEvent);
}

CUresult cuEventRecord(void* hEvent, CUstream_drv hStream) {
    vgpu::RpcEventRecordReq req{};
    req.event = reinterpret_cast<std::uint64_t>(hEvent);
    req.stream = reinterpret_cast<std::uint64_t>(hStream);
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaEventRecord, &req, sizeof(req));
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult cuEventSynchronize(void* hEvent) {
    vgpu::RpcEventSynchronizeReq req{};
    req.event = reinterpret_cast<std::uint64_t>(hEvent);
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaEventSynchronize, &req, sizeof(req));
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult cuEventQuery(void* hEvent) {
    vgpu::RpcEventQueryReq req{};
    req.event = reinterpret_cast<std::uint64_t>(hEvent);
    vgpu::RpcResult r = vgpu::drvCall(vgpu::RpcOp::kCudaEventQuery, &req, sizeof(req));
    return r.status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}

CUresult cuEventElapsedTime(float* pMilliseconds, void* /*hStart*/, void* /*hEnd*/) {
    if (pMilliseconds) {
        *pMilliseconds = 0.0f;
    }
    return CUDA_SUCCESS;
}

CUresult cuEventRecordWithFlags(void* hEvent, CUstream_drv hStream, unsigned int /*flags*/) {
    return cuEventRecord(hEvent, hStream);
}

CUresult cuThreadExchangeStreamCaptureMode(int* /*mode*/) {
    return CUDA_SUCCESS;
}

CUresult cuLaunchHostFunc(CUstream_drv /*hStream*/, void (*fn)(void*), void* userData) {
    if (fn) {
        fn(userData);
    }
    return CUDA_SUCCESS;
}

CUresult cuStreamGetId(CUstream_drv /*hStream*/, std::uint64_t* streamId) {
    if (streamId) {
        *streamId = 0;
    }
    return CUDA_SUCCESS;
}

CUresult cuStreamAddCallback(CUstream_drv hStream,
                             void (*callback)(CUstream_drv, CUresult, void*),
                             void* userData,
                             unsigned int /*flags*/) {
    if (callback) {
        callback(hStream, CUDA_SUCCESS, userData);
    }
    return CUDA_SUCCESS;
}

CUresult cuStreamAttachMemAsync(CUstream_drv /*hStream*/,
                                CUdeviceptr /*dptr*/,
                                size_t /*length*/,
                                unsigned int /*flags*/) {
    return CUDA_SUCCESS;
}

CUresult cuStreamCopyAttributes(CUstream_drv /*dst*/, CUstream_drv /*src*/) {
    return CUDA_SUCCESS;
}

CUresult cuStreamGetAttribute(CUstream_drv /*hStream*/, int /*attr*/, void* value) {
    if (value) {
        std::memset(value, 0, 64);
    }
    return CUDA_SUCCESS;
}

CUresult cuStreamSetAttribute(CUstream_drv /*hStream*/, int /*attr*/, const void* /*value*/) {
    return CUDA_SUCCESS;
}

CUresult cuStreamWaitValue32(CUstream_drv /*hStream*/, CUdeviceptr /*addr*/, unsigned int /*value*/, unsigned int /*flags*/) {
    return CUDA_SUCCESS;
}

CUresult cuStreamWriteValue32(CUstream_drv /*hStream*/, CUdeviceptr /*addr*/, unsigned int /*value*/, unsigned int /*flags*/) {
    return CUDA_SUCCESS;
}

CUresult cuStreamWaitValue64(CUstream_drv /*hStream*/, CUdeviceptr /*addr*/, std::uint64_t /*value*/, unsigned int /*flags*/) {
    return CUDA_SUCCESS;
}

CUresult cuStreamWriteValue64(CUstream_drv /*hStream*/, CUdeviceptr /*addr*/, std::uint64_t /*value*/, unsigned int /*flags*/) {
    return CUDA_SUCCESS;
}

CUresult cuStreamBatchMemOp(CUstream_drv /*hStream*/, unsigned int /*count*/, void* /*paramArray*/, unsigned int /*flags*/) {
    return CUDA_SUCCESS;
}

CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice /*dev*/, CUdevice /*peerDev*/) {
    if (canAccessPeer) {
        *canAccessPeer = 0;
    }
    return CUDA_SUCCESS;
}

CUresult cuCtxEnablePeerAccess(CUcontext /*peerContext*/, unsigned int /*Flags*/) {
    return CUDA_SUCCESS;
}

CUresult cuCtxDisablePeerAccess(CUcontext /*peerContext*/) {
    return CUDA_SUCCESS;
}

CUresult cuIpcGetEventHandle(void* /*pHandle*/, void* /*event*/) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuIpcOpenEventHandle(void** event, const void* /*handle*/) {
    if (event) {
        *event = nullptr;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuIpcGetMemHandle(void* /*pHandle*/, CUdeviceptr /*dptr*/) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuIpcOpenMemHandle(CUdeviceptr* pdptr, const void* /*handle*/, unsigned int /*Flags*/) {
    if (pdptr) {
        *pdptr = 0;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuIpcCloseMemHandle(CUdeviceptr /*dptr*/) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuSignalExternalSemaphoresAsync(const void* /*extSemArray*/,
                                         const void* /*paramsArray*/,
                                         unsigned int /*numExtSems*/,
                                         CUstream_drv /*stream*/) {
    return CUDA_SUCCESS;
}

CUresult cuWaitExternalSemaphoresAsync(const void* /*extSemArray*/,
                                       const void* /*paramsArray*/,
                                       unsigned int /*numExtSems*/,
                                       CUstream_drv /*stream*/) {
    return CUDA_SUCCESS;
}

CUresult cuMemAdvise(CUdeviceptr /*devPtr*/, size_t /*count*/, int /*advice*/, CUdevice /*device*/) {
    return CUDA_SUCCESS;
}

CUresult cuMemPrefetchAsync(CUdeviceptr /*devPtr*/, size_t /*count*/, CUdevice /*dstDevice*/, CUstream_drv /*hStream*/) {
    return CUDA_SUCCESS;
}

CUresult cuMemRangeGetAttribute(void* data,
                                size_t dataSize,
                                int /*attribute*/,
                                CUdeviceptr /*devPtr*/,
                                size_t /*count*/) {
    if (data && dataSize > 0) {
        std::memset(data, 0, dataSize);
    }
    return CUDA_SUCCESS;
}

CUresult cuMemRangeGetAttributes(void** data,
                                 size_t* dataSizes,
                                 int* /*attributes*/,
                                 size_t numAttributes,
                                 CUdeviceptr /*devPtr*/,
                                 size_t /*count*/) {
    if (data && dataSizes) {
        for (size_t i = 0; i < numAttributes; ++i) {
            if (data[i] && dataSizes[i] > 0) {
                std::memset(data[i], 0, dataSizes[i]);
            }
        }
    }
    return CUDA_SUCCESS;
}

CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize,
                                                 CUfunction /*func*/,
                                                 int /*numBlocks*/,
                                                 int /*blockSize*/) {
    if (dynamicSmemSize) {
        *dynamicSmemSize = 0;
    }
    return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxPotentialClusterSize(int* clusterSize,
                                            CUfunction /*func*/,
                                            const void* /*config*/) {
    if (clusterSize) {
        *clusterSize = 1;
    }
    return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxActiveClusters(int* numClusters,
                                      CUfunction /*func*/,
                                      const void* /*config*/) {
    if (numClusters) {
        *numClusters = 1;
    }
    return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks,
    CUfunction /*func*/,
    int /*blockSize*/,
    size_t /*dynamicSMemSize*/,
    unsigned int /*flags*/) {
    if (numBlocks) {
        *numBlocks = 1;
    }
    return CUDA_SUCCESS;
}

CUresult cuGetExportTable(const void** ppExportTable, const void* pExportTableId) {
    using CuGetExportTableFn = CUresult (*)(const void**, const void*);

    auto format_uuid = [&](char (&out)[33]) {
        out[0] = '\0';
        if (pExportTableId == nullptr) {
            std::strncpy(out, "null", sizeof(out) - 1);
            out[sizeof(out) - 1] = '\0';
            return;
        }
        const auto* bytes = reinterpret_cast<const unsigned char*>(pExportTableId);
        static const char* hex = "0123456789abcdef";
        for (int i = 0; i < 16; ++i) {
            out[2 * i] = hex[(bytes[i] >> 4) & 0xF];
            out[2 * i + 1] = hex[bytes[i] & 0xF];
        }
        out[32] = '\0';
    };
    char table_uuid[33];
    format_uuid(table_uuid);

    const char* use_fake_tables = std::getenv("VGPU_USE_FAKE_CU_EXPORT_TABLES");
    if (use_fake_tables != nullptr && use_fake_tables[0] == '1') {
        const void* fake = vgpu::fakeExportTableForUuid(pExportTableId);
        if (ppExportTable) {
            *ppExportTable = fake;
        }
        CUresult rc = (fake != nullptr) ? CUDA_SUCCESS : CUDA_ERROR_NOT_SUPPORTED;
        if (std::getenv("VGPU_LOG_CU_GETPROC") != nullptr) {
            std::fprintf(stderr,
                         "[vgpu-cuGetExportTable] fake rc=%d table=%p id=%p uuid=%s\n",
                         static_cast<int>(rc),
                         fake,
                         pExportTableId,
                         table_uuid);
        }
        return rc;
    }

    const char* force_success_null = std::getenv("VGPU_CU_EXPORT_TABLE_SUCCESS_NULL");
    if (force_success_null != nullptr && force_success_null[0] == '1') {
        if (ppExportTable) {
            *ppExportTable = nullptr;
        }
        if (std::getenv("VGPU_LOG_CU_GETPROC") != nullptr) {
            std::fprintf(stderr,
                         "[vgpu-cuGetExportTable] shim-success-null rc=%d table=null id=%p uuid=%s\n",
                         static_cast<int>(CUDA_SUCCESS),
                         pExportTableId,
                         table_uuid);
        }
        return CUDA_SUCCESS;
    }

    const char* allow_real = std::getenv("VGPU_ENABLE_REAL_CU_EXPORT_TABLE");
    const bool use_real = (allow_real != nullptr && allow_real[0] == '1');

    if (!use_real) {
        if (ppExportTable) {
            *ppExportTable = nullptr;
        }
        if (std::getenv("VGPU_LOG_CU_GETPROC") != nullptr) {
            std::fprintf(stderr,
                         "[vgpu-cuGetExportTable] shim rc=%d table=null id=%p uuid=%s\n",
                         static_cast<int>(CUDA_ERROR_NOT_SUPPORTED),
                         pExportTableId,
                         table_uuid);
        }
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    void* real = vgpu::realCudaHandle();
    if (real != nullptr) {
        auto* fn = reinterpret_cast<CuGetExportTableFn>(dlsym(real, "cuGetExportTable"));
        if (fn != nullptr) {
            CUresult rc = fn(ppExportTable, pExportTableId);
            if (std::getenv("VGPU_LOG_CU_GETPROC") != nullptr) {
                std::fprintf(stderr,
                             "[vgpu-cuGetExportTable] real rc=%d table=%p id=%p uuid=%s\n",
                             static_cast<int>(rc),
                             (ppExportTable != nullptr) ? *ppExportTable : nullptr,
                             pExportTableId,
                             table_uuid);
            }
            return rc;
        }
    }

    if (ppExportTable) {
        *ppExportTable = nullptr;
    }
    if (std::getenv("VGPU_LOG_CU_GETPROC") != nullptr) {
        std::fprintf(stderr,
                     "[vgpu-cuGetExportTable] no-real rc=%d table=null id=%p uuid=%s\n",
                     static_cast<int>(CUDA_ERROR_NOT_SUPPORTED),
                     pExportTableId,
                     table_uuid);
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

// ── Error helpers ─────────────────────────────────────────────────────────────

CUresult cuGetErrorName(CUresult error, const char** pStr) {
    static const char* unknown = "CUDA_ERROR_UNKNOWN";
    if (pStr) {
        *pStr = (error == CUDA_SUCCESS) ? "CUDA_SUCCESS" : unknown;
    }
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult error, const char** pStr) {
    return cuGetErrorName(error, pStr);
}

CUresult cuNoopStub() {
    return CUDA_SUCCESS;
}

CUresult cuNotSupportedStub() {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuGetProcAddress(const char* symbol,
                         void** pfn,
                         int /*cudaVersion*/,
                         std::uint64_t /*flags*/) {
    if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;
    const char* resolved_from = "none";

    if (std::strcmp(symbol, "cuGetProcAddress") == 0) {
        *pfn = reinterpret_cast<void*>(&cuGetProcAddress);
        resolved_from = "self-direct";
        return CUDA_SUCCESS;
    }

    // Prefer symbols exported by our own shim first.
    static void* self_handle = []() -> void* {
        void* h = dlopen("libcuda.so", RTLD_NOW | RTLD_NOLOAD);
        if (!h) {
            h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        }
        return h;
    }();

    *pfn = nullptr;
    if (self_handle) {
        *pfn = dlsym(self_handle, symbol);
        if (*pfn) resolved_from = "self";
    }

    if (!*pfn) {
        *pfn = dlsym(RTLD_DEFAULT, symbol);
        if (*pfn) resolved_from = "default";
    }

    if (!*pfn) {
        const bool optional_graph = std::strncmp(symbol, "cuGraph", 7) == 0;
        const bool optional_user_obj = std::strncmp(symbol, "cuUserObject", 12) == 0;
        const bool optional_stream_capture =
            std::strncmp(symbol, "cuStreamBeginCapture", 20) == 0 ||
            std::strncmp(symbol, "cuStreamEndCapture", 18) == 0 ||
            std::strncmp(symbol, "cuStreamIsCapturing", 19) == 0 ||
            std::strncmp(symbol, "cuStreamGetCaptureInfo", 22) == 0 ||
            std::strncmp(symbol, "cuStreamUpdateCaptureDependencies", 33) == 0 ||
            std::strcmp(symbol, "cuThreadExchangeStreamCaptureMode") == 0;
        const bool optional_misc =
            std::strcmp(symbol, "cuDeviceRegisterAsyncNotification") == 0 ||
            std::strcmp(symbol, "cuDeviceUnregisterAsyncNotification") == 0 ||
            std::strcmp(symbol, "cuOccupancyAvailableDynamicSMemPerBlock") == 0 ||
            std::strcmp(symbol, "cuOccupancyMaxPotentialClusterSize") == 0 ||
            std::strcmp(symbol, "cuOccupancyMaxActiveClusters") == 0 ||
            std::strcmp(symbol, "cuMemAdvise") == 0 ||
            std::strcmp(symbol, "cuMemPrefetchAsync") == 0 ||
            std::strcmp(symbol, "cuMemRangeGetAttribute") == 0 ||
            std::strcmp(symbol, "cuMemRangeGetAttributes") == 0;

        // Strict mode by default: don't advertise unsupported capabilities.
        // Some user-space stacks choose fast paths based on symbol presence
        // and later fail in confusing ways if the symbol only no-ops.
        const char* expose_optional = std::getenv("VGPU_EXPOSE_OPTIONAL_CUDA_SYMBOLS");
        const bool allow_optional = (expose_optional != nullptr && expose_optional[0] == '1');

        if ((optional_graph || optional_user_obj || optional_stream_capture || optional_misc) && allow_optional) {
            const char* optional_not_supported = std::getenv("VGPU_OPTIONAL_SYMBOLS_RETURN_NOT_SUPPORTED");
            const bool use_not_supported = (optional_not_supported != nullptr && optional_not_supported[0] == '1');
            *pfn = use_not_supported
                ? reinterpret_cast<void*>(&cuNotSupportedStub)
                : reinterpret_cast<void*>(&cuNoopStub);
            resolved_from = use_not_supported ? "builtin-not-supported" : "builtin-noop";
        }
    }

    // Keep real-libcuda fallback opt-in: for remote interception flows,
    // resolving unknown entries to a local real driver often causes init
    // failures on machines without matching local GPUs.
    const char* allow_real_fallback = std::getenv("VGPU_ENABLE_REAL_CUDA_FALLBACK");
    if (!*pfn && allow_real_fallback && allow_real_fallback[0] == '1') {
        void* real = vgpu::realCudaHandle();
        if (real) {
            *pfn = dlsym(real, symbol);
            if (*pfn) resolved_from = "real";
        }
    }

    const char* allow_noop = std::getenv("VGPU_NOOP_UNKNOWN_CUDA_SYMBOLS");
    if (!*pfn && allow_noop && allow_noop[0] == '1') {
        *pfn = reinterpret_cast<void*>(&cuNoopStub);
        resolved_from = "noop";
    }

    if (std::getenv("VGPU_LOG_CU_GETPROC") != nullptr) {
        std::fprintf(stderr, "[vgpu-cuGetProcAddress] %s => %s (%s)\n",
                     symbol,
                     *pfn ? "FOUND" : "MISSING",
                     resolved_from);
    }

    return *pfn ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
}

CUresult cuGetProcAddress_v2(const char* symbol,
                            void** pfn,
                            int cudaVersion,
                            std::uint64_t flags,
                            std::uint64_t* symbolStatus) {
    CUresult rc = cuGetProcAddress(symbol, pfn, cudaVersion, flags);
    if (symbolStatus) {
        // 0 = success, 1 = symbol not found (best-effort compatibility).
        *symbolStatus = (rc == CUDA_SUCCESS && pfn != nullptr && *pfn != nullptr) ? 0ull : 1ull;
    }
    return rc;
}

}  // extern "C"
