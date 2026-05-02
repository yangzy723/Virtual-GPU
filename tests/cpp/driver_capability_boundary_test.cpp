#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "vgpu/backend/cuda_driver_loader.h"

extern "C" CUresult cuGetProcAddress(const char* symbol,
                                      void** pfn,
                                      int cudaVersion,
                                      std::uint64_t flags);
extern "C" CUresult cuGetProcAddress_v2(const char* symbol,
                                         void** pfn,
                                         int cudaVersion,
                                         std::uint64_t flags,
                                         std::uint64_t* symbolStatus);
extern "C" CUresult cuGetExportTable(const void** ppExportTable, const void* pExportTableId);

namespace {

constexpr unsigned char kUuid6b[16] = {
    0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a,
    0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9,
};

bool expectResult(const char* api, CUresult got, CUresult expected) {
    if (got != expected) {
        std::fprintf(stderr,
                     "%s should return %d, got %d\n",
                     api,
                     static_cast<int>(expected),
                     static_cast<int>(got));
        return false;
    }
    return true;
}

struct ScopedEnvVar {
    explicit ScopedEnvVar(const char* key) : key_(key) {
        const char* value = std::getenv(key_);
        if (value != nullptr) {
            had_original_ = true;
            original_ = value;
        }
    }

    ~ScopedEnvVar() {
        if (had_original_) {
            setenv(key_, original_.c_str(), 1);
        } else {
            unsetenv(key_);
        }
    }

    void set(const char* value) {
        setenv(key_, value, 1);
    }

    void clear() {
        unsetenv(key_);
    }

private:
    const char* key_;
    bool had_original_ = false;
    std::string original_;
};

}  // namespace

int main() {
    ScopedEnvVar expose_optional("VGPU_EXPOSE_OPTIONAL_CUDA_SYMBOLS");
    ScopedEnvVar optional_not_supported("VGPU_OPTIONAL_SYMBOLS_RETURN_NOT_SUPPORTED");
    ScopedEnvVar export_success_null("VGPU_CU_EXPORT_TABLE_SUCCESS_NULL");
    ScopedEnvVar fake_export_tables("VGPU_USE_FAKE_CU_EXPORT_TABLES");
    ScopedEnvVar real_export_tables("VGPU_ENABLE_REAL_CU_EXPORT_TABLE");
    expose_optional.clear();
    optional_not_supported.clear();
    export_success_null.clear();
    fake_export_tables.clear();
    real_export_tables.clear();

    void* fn = nullptr;
    if (!expectResult("cuGetProcAddress(cuGetProcAddress)",
                      cuGetProcAddress("cuGetProcAddress", &fn, 0, 0),
                      CUDA_SUCCESS)) {
        return 1;
    }
    if (fn == nullptr) {
        std::fprintf(stderr, "cuGetProcAddress should resolve itself\n");
        return 2;
    }

    fn = reinterpret_cast<void*>(0x1);
    if (!expectResult("cuGetProcAddress(missing)",
                      cuGetProcAddress("cuDefinitelyMissingSymbol", &fn, 0, 0),
                      CUDA_ERROR_NOT_FOUND)) {
        return 3;
    }
    if (fn != nullptr) {
        std::fprintf(stderr, "missing symbol should clear the output pointer\n");
        return 4;
    }

    fn = reinterpret_cast<void*>(0x1);
    if (!expectResult("cuGetProcAddress(optional strict)",
                      cuGetProcAddress("cuGraphInstantiate", &fn, 0, 0),
                      CUDA_ERROR_NOT_FOUND)) {
        return 5;
    }
    if (fn != nullptr) {
        std::fprintf(stderr, "optional symbol should stay hidden in strict mode\n");
        return 6;
    }

    expose_optional.set("1");
    optional_not_supported.set("1");
    fn = nullptr;
    if (!expectResult("cuGetProcAddress(optional exposed)",
                      cuGetProcAddress("cuGraphInstantiate", &fn, 0, 0),
                      CUDA_SUCCESS)) {
        return 7;
    }
    if (fn == nullptr) {
        std::fprintf(stderr, "optional symbol should resolve when exposure is enabled\n");
        return 8;
    }

    expose_optional.clear();
    optional_not_supported.clear();

    fn = reinterpret_cast<void*>(0x1);
    std::uint64_t symbol_status = 99;
    if (!expectResult("cuGetProcAddress_v2(missing)",
                      cuGetProcAddress_v2("cuDefinitelyMissingSymbol", &fn, 0, 0, &symbol_status),
                      CUDA_ERROR_NOT_FOUND)) {
        return 9;
    }
    if (fn != nullptr || symbol_status != 1ull) {
        std::fprintf(stderr,
                     "cuGetProcAddress_v2 should report a missing symbol, got fn=%p status=%llu\n",
                     fn,
                     static_cast<unsigned long long>(symbol_status));
        return 10;
    }

    const void* export_table = reinterpret_cast<void*>(0x1);
    if (!expectResult("cuGetExportTable(default)",
                      cuGetExportTable(&export_table, kUuid6b),
                      CUDA_ERROR_NOT_SUPPORTED)) {
        return 11;
    }
    if (export_table != nullptr) {
        std::fprintf(stderr, "default export-table path should clear the table pointer\n");
        return 12;
    }

    export_success_null.set("1");
    export_table = reinterpret_cast<void*>(0x1);
    if (!expectResult("cuGetExportTable(success-null)",
                      cuGetExportTable(&export_table, kUuid6b),
                      CUDA_SUCCESS)) {
        return 13;
    }
    if (export_table != nullptr) {
        std::fprintf(stderr, "success-null mode should still return a null table\n");
        return 14;
    }

    export_success_null.clear();
    fake_export_tables.set("1");
    export_table = nullptr;
    if (!expectResult("cuGetExportTable(fake)",
                      cuGetExportTable(&export_table, kUuid6b),
                      CUDA_SUCCESS)) {
        return 15;
    }
    if (export_table == nullptr) {
        std::fprintf(stderr, "fake export-table mode should return a fake table for supported UUIDs\n");
        return 16;
    }

    std::printf("driver_capability_boundary_test passed\n");
    return 0;
}