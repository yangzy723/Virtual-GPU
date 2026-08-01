#pragma once

#include <cstdint>
#include <limits>

#include "vgpu/config.h"

namespace vgpu {

struct SchedulerConfig {
    uint64_t memory_limit_bytes = 0;  // 0 = unlimited
    uint64_t context_overhead_bytes = 800ULL * 1024 * 1024;
    int max_concurrent_kernels = 1;
    int max_concurrent_memcpy = 0;    // 0 = unlimited
};

inline SchedulerConfig loadSchedulerConfig() {
    constexpr uint64_t kMiB = 1024ULL * 1024;
    constexpr int kMax = std::numeric_limits<int>::max();

    SchedulerConfig config;
    config.memory_limit_bytes = static_cast<uint64_t>(config::getInt(
        "GPU_SCHEDULER_MEM_LIMIT_MB", 0, 0, kMax)) * kMiB;
    config.context_overhead_bytes = static_cast<uint64_t>(config::getInt(
        "GPU_SCHEDULER_CONTEXT_OVERHEAD_MB", 800, 0, kMax)) * kMiB;
    config.max_concurrent_kernels = config::getInt(
        "GPU_SCHEDULER_MAX_KERNELS", 1, 1, kMax);
    config.max_concurrent_memcpy = config::getInt(
        "GPU_SCHEDULER_MAX_MEMCPY", 0, 0, kMax);
    return config;
}

}  // namespace vgpu
