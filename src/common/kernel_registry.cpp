#include "vgpu/common/kernel_registry.h"

namespace vgpu {

void KernelRegistry::addModule(void* handle, uint64_t module_id) {
    std::lock_guard<std::mutex> lk(mutex_);
    handle_to_module_[handle] = module_id;
}

uint64_t KernelRegistry::getModuleId(void* handle) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = handle_to_module_.find(handle);
    return it != handle_to_module_.end() ? it->second : 0;
}

void KernelRegistry::addKernel(const void* host_fun, KernelEntry entry) {
    std::lock_guard<std::mutex> lk(mutex_);
    func_to_entry_[host_fun] = std::move(entry);
}

bool KernelRegistry::findKernel(const void* host_fun, KernelEntry* out) const {
    if (out == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = func_to_entry_.find(host_fun);
    if (it == func_to_entry_.end()) {
        return false;
    }
    *out = it->second;
    return true;
}

void KernelRegistry::addParamInfo(uint64_t module_id, const std::string& name, std::vector<ParamInfo> params) {
    std::lock_guard<std::mutex> lk(mutex_);
    module_to_params_[module_id][name] = std::move(params);
}

bool KernelRegistry::findParamInfo(uint64_t module_id, const std::string& name, std::vector<ParamInfo>* out) const {
    if (out == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lk(mutex_);
    auto mit = module_to_params_.find(module_id);
    if (mit == module_to_params_.end()) {
        return false;
    }
    auto it = mit->second.find(name);
    if (it == mit->second.end()) {
        return false;
    }
    *out = it->second;
    return true;
}

void KernelRegistry::addDriverModule(void* cu_mod_handle, uint64_t module_id) {
    std::lock_guard<std::mutex> lk(mutex_);
    drv_mod_to_id_[cu_mod_handle] = module_id;
}

uint64_t KernelRegistry::getDriverModuleId(void* cu_mod_handle) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = drv_mod_to_id_.find(cu_mod_handle);
    return it != drv_mod_to_id_.end() ? it->second : 0;
}

void KernelRegistry::addDriverFunc(void* cu_func_handle, KernelEntry entry) {
    std::lock_guard<std::mutex> lk(mutex_);
    drv_func_to_entry_[cu_func_handle] = std::move(entry);
}

bool KernelRegistry::findDriverFunc(void* cu_func_handle, KernelEntry* out) const {
    if (out == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = drv_func_to_entry_.find(cu_func_handle);
    if (it == drv_func_to_entry_.end()) {
        return false;
    }
    *out = it->second;
    return true;
}

KernelRegistry& globalKernelRegistry() {
    static KernelRegistry r;
    return r;
}

}  // namespace vgpu
