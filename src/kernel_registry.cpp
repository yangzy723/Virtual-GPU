#include "vgpu/kernel_registry.h"

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

const KernelEntry* KernelRegistry::findKernel(const void* host_fun) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = func_to_entry_.find(host_fun);
    return it != func_to_entry_.end() ? &it->second : nullptr;
}

void KernelRegistry::addParamInfo(const std::string& name, std::vector<ParamInfo> params) {
    std::lock_guard<std::mutex> lk(mutex_);
    name_to_params_[name] = std::move(params);
}

const std::vector<ParamInfo>* KernelRegistry::findParamInfo(const std::string& name) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = name_to_params_.find(name);
    return it != name_to_params_.end() ? &it->second : nullptr;
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

const KernelEntry* KernelRegistry::findDriverFunc(void* cu_func_handle) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = drv_func_to_entry_.find(cu_func_handle);
    return it != drv_func_to_entry_.end() ? &it->second : nullptr;
}

KernelRegistry& globalKernelRegistry() {
    static KernelRegistry r;
    return r;
}

}  // namespace vgpu
