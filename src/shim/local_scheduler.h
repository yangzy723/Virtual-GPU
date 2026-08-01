#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>

#include "vgpu/protocol.h"
#include "vgpu/scheduler_config.h"

namespace vgpu {

// In-process admission control used by VGPU_SCHEDULER_MODE=local.
// It never creates a socket and only coordinates threads in the current process.
class LocalScheduler {
public:
    struct Snapshot {
        uint64_t memory_used_bytes = 0;
        int active_kernels = 0;
        int active_memcpy = 0;
    };

    explicit LocalScheduler(SchedulerConfig config);

    // Allocation requests may be rejected by the memory quota. Kernel and
    // memcpy requests wait for a local slot instead of surfacing transient
    // scheduling pressure as a CUDA error.
    bool request(SchedOp op, uint64_t value);
    void report(SchedOp op, uint64_t value);

    Snapshot snapshot() const;

private:
    SchedulerConfig config_;
    mutable std::mutex mutex_;
    std::condition_variable slot_available_;
    uint64_t allocated_bytes_ = 0;
    int active_kernels_ = 0;
    int active_memcpy_ = 0;
};

}  // namespace vgpu
