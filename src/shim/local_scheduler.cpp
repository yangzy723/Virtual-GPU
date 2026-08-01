#include "local_scheduler.h"

#include <limits>

namespace vgpu {

LocalScheduler::LocalScheduler(SchedulerConfig config) : config_(config) {
    if (config_.max_concurrent_kernels <= 0) {
        config_.max_concurrent_kernels = 1;
    }
    if (config_.max_concurrent_memcpy < 0) {
        config_.max_concurrent_memcpy = 0;
    }
}

bool LocalScheduler::request(SchedOp op, uint64_t value) {
    std::unique_lock<std::mutex> lock(mutex_);

    switch (op) {
        case SchedOp::ALLOC_REQUEST: {
            const uint64_t limit = config_.memory_limit_bytes;
            if (limit > 0) {
                if (config_.context_overhead_bytes > limit) return false;
                const uint64_t available = limit - config_.context_overhead_bytes;
                if (allocated_bytes_ > available || value > available - allocated_bytes_) {
                    return false;
                }
            }
            if (value > std::numeric_limits<uint64_t>::max() - allocated_bytes_) {
                return false;
            }
            allocated_bytes_ += value;
            return true;
        }
        case SchedOp::KERNEL_REQUEST:
            slot_available_.wait(lock, [this] {
                return active_kernels_ < config_.max_concurrent_kernels;
            });
            ++active_kernels_;
            return true;
        case SchedOp::MEMCPY_REQUEST:
            if (config_.max_concurrent_memcpy > 0) {
                slot_available_.wait(lock, [this] {
                    return active_memcpy_ < config_.max_concurrent_memcpy;
                });
            }
            ++active_memcpy_;
            return true;
        default:
            return false;
    }
}

void LocalScheduler::report(SchedOp op, uint64_t value) {
    std::lock_guard<std::mutex> lock(mutex_);

    switch (op) {
        case SchedOp::FREE:
            allocated_bytes_ = value >= allocated_bytes_ ? 0 : allocated_bytes_ - value;
            break;
        case SchedOp::KERNEL_COMPLETE:
            if (active_kernels_ > 0) --active_kernels_;
            slot_available_.notify_all();
            break;
        case SchedOp::MEMCPY_COMPLETE:
            if (active_memcpy_ > 0) --active_memcpy_;
            slot_available_.notify_all();
            break;
        default:
            break;
    }
}

LocalScheduler::Snapshot LocalScheduler::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {
        config_.context_overhead_bytes + allocated_bytes_,
        active_kernels_,
        active_memcpy_,
    };
}

}  // namespace vgpu
