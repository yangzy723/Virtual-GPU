#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include "vgpu/protocol.h"

namespace vgpu {

class Scheduler {
public:
    struct Config {
        size_t per_client_mem_limit = 0;  // 0 = unlimited
        size_t context_overhead = 800 * 1024 * 1024;  // 800MB pre-charge
        int max_concurrent_kernels = 1;
        int max_concurrent_memcpy = 0;  // 0 = unlimited
    };

    explicit Scheduler(Config cfg);

    void setVerbose(bool v);

    // Client lifecycle
    bool registerClient(uint64_t client_id, const std::string& shm_name);
    void unregisterClient(uint64_t client_id);

    // Scheduling decisions (called from poll loop)
    void processRequest(uint64_t client_id, ShmChannel* channel);

    // Config
    const Config& config() const { return cfg_; }

    // Stats
    size_t clientCount() const;

private:
    struct ClientState {
        size_t memory_used = 0;
        int active_kernels = 0;
        int active_memcpy = 0;
        std::string shm_name;
    };

    Config cfg_;
    mutable std::mutex mu_;
    std::unordered_map<uint64_t, ClientState> clients_;
};

}  // namespace vgpu
