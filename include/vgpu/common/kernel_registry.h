#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "vgpu/common/fatbin_parser.h"

namespace vgpu {

// Per-kernel entry stored on the client side
struct KernelEntry {
    uint64_t func_id;                // server-assigned function ID
    std::vector<ParamInfo> params;   // parameter info for arg packing
    uint32_t total_param_bytes;      // total packed buffer size
};

// Client-side kernel registry.
// Thread-safe.  One global instance shared between interceptor.cpp and
// driver_interceptor.cpp.
class KernelRegistry {
public:
    // fatbin handle → server module_id
    void addModule(void* handle, uint64_t module_id);
    uint64_t getModuleId(void* handle) const;

    // hostFun pointer → KernelEntry
    void addKernel(const void* host_fun, KernelEntry entry);
    bool findKernel(const void* host_fun, KernelEntry* out) const;

    // (module_id, mangled device function name) → param info
    // Populated when the fatbin is parsed at module load/registration time.
    void addParamInfo(uint64_t module_id, const std::string& mangled_name, std::vector<ParamInfo> params);
    bool findParamInfo(uint64_t module_id, const std::string& mangled_name, std::vector<ParamInfo>* out) const;

    // Driver-shim handle table (CUmodule / CUfunction fake handles → server IDs)
    // We reuse the same maps with different key spaces:
    //   driver CUmodule handle → server module_id
    void addDriverModule(void* cu_mod_handle, uint64_t module_id);
    uint64_t getDriverModuleId(void* cu_mod_handle) const;

    //   driver CUfunction handle → KernelEntry
    void addDriverFunc(void* cu_func_handle, KernelEntry entry);
    bool findDriverFunc(void* cu_func_handle, KernelEntry* out) const;

private:
    mutable std::mutex mutex_;

    std::unordered_map<void*, uint64_t>        handle_to_module_;
    std::unordered_map<const void*, KernelEntry> func_to_entry_;
    std::unordered_map<uint64_t, std::unordered_map<std::string, std::vector<ParamInfo>>> module_to_params_;

    std::unordered_map<void*, uint64_t>        drv_mod_to_id_;
    std::unordered_map<void*, KernelEntry>     drv_func_to_entry_;
};

// Process-global registry instance
KernelRegistry& globalKernelRegistry();

}  // namespace vgpu
