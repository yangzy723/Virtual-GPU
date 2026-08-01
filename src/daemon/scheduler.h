#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "vgpu/protocol.h"
#include "vgpu/scheduler_config.h"

namespace vgpu {

class Scheduler {
public:
    explicit Scheduler(SchedulerConfig config, bool verbose = false);

    void registerClient(uint64_t client_id);
    void unregisterClient(uint64_t client_id);
    void processRequest(uint64_t client_id, ShmChannel* channel);

private:
    struct ClientState {
        uint64_t memory_used = 0;
        int active_kernels = 0;
        int active_memcpy = 0;
    };

    SchedulerConfig config_;
    bool verbose_;
    std::mutex mutex_;
    std::unordered_map<uint64_t, ClientState> clients_;
};

}  // namespace vgpu
