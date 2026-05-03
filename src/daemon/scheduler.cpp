#include "scheduler.h"

#include <cstdarg>
#include <cstdio>

namespace vgpu {

Scheduler::Scheduler(Config cfg) : cfg_(cfg) {}

static bool g_verbose = false;

void Scheduler::setVerbose(bool v) { g_verbose = v; }

static void schedLog(const char* fmt, ...) {
    if (!g_verbose) return;
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
}

bool Scheduler::registerClient(uint64_t client_id, const std::string& shm_name) {
    std::lock_guard<std::mutex> lock(mu_);
    auto& cs = clients_[client_id];
    cs.shm_name = shm_name;
    // Pre-charge context overhead
    cs.memory_used = cfg_.context_overhead;
    return true;
}

void Scheduler::unregisterClient(uint64_t client_id) {
    std::lock_guard<std::mutex> lock(mu_);
    clients_.erase(client_id);
}

void Scheduler::processRequest(uint64_t client_id, ShmChannel* channel) {
    auto op = static_cast<SchedOp>(channel->op.load(std::memory_order_acquire));
    uint64_t value = channel->value.load(std::memory_order_acquire);

    std::lock_guard<std::mutex> lock(mu_);
    auto it = clients_.find(client_id);
    if (it == clients_.end()) {
        channel->result.store(-1, std::memory_order_release);
        channel->state.store(static_cast<uint32_t>(SchedState::REJECTED),
                             std::memory_order_release);
        return;
    }

    auto& cs = it->second;

    const char* op_name = "UNKNOWN";
    switch (op) {
        case SchedOp::ALLOC_REQUEST: op_name = "ALLOC_REQUEST"; break;
        case SchedOp::FREE:          op_name = "FREE"; break;
        case SchedOp::KERNEL_REQUEST: op_name = "KERNEL_REQUEST"; break;
        case SchedOp::KERNEL_COMPLETE: op_name = "KERNEL_COMPLETE"; break;
        case SchedOp::MEMCPY_REQUEST: op_name = "MEMCPY_REQUEST"; break;
        case SchedOp::MEMCPY_COMPLETE: op_name = "MEMCPY_COMPLETE"; break;
        default: break;
    }

    schedLog("[scheduler] pid=%lu op=%s value=%lu mem_used=%zu\n",
             static_cast<unsigned long>(client_id), op_name,
             static_cast<unsigned long>(value), cs.memory_used);

    switch (op) {
        case SchedOp::ALLOC_REQUEST: {
            bool ok = true;
            if (cfg_.per_client_mem_limit > 0) {
                if (cs.memory_used > cfg_.per_client_mem_limit ||
                    value > (cfg_.per_client_mem_limit - cs.memory_used)) {
                    ok = false;
                }
            }
            if (ok) {
                cs.memory_used += value;
                channel->result.store(0, std::memory_order_release);
                channel->state.store(static_cast<uint32_t>(SchedState::APPROVED),
                                     std::memory_order_release);
                schedLog("[scheduler]   -> APPROVED (mem_used now %zu)\n", cs.memory_used);
            } else {
                channel->result.store(-1, std::memory_order_release);
                channel->state.store(static_cast<uint32_t>(SchedState::REJECTED),
                                     std::memory_order_release);
                schedLog("[scheduler]   -> REJECTED (quota exceeded)\n");
            }
            break;
        }
        case SchedOp::FREE: {
            if (value <= cs.memory_used) {
                cs.memory_used -= value;
            } else {
                cs.memory_used = 0;
            }
            // Must reset to IDLE so the shim's schedReport() unblocks.
            channel->result.store(0, std::memory_order_release);
            channel->state.store(static_cast<uint32_t>(SchedState::IDLE),
                                 std::memory_order_release);
            schedLog("[scheduler]   -> FREE complete (mem_used now %zu)\n", cs.memory_used);
            break;
        }
        case SchedOp::KERNEL_REQUEST: {
            int total_active = 0;
            for (const auto& [id, c] : clients_) {
                total_active += c.active_kernels;
            }
            if (total_active < cfg_.max_concurrent_kernels) {
                cs.active_kernels++;
                channel->result.store(0, std::memory_order_release);
                channel->state.store(static_cast<uint32_t>(SchedState::APPROVED),
                                     std::memory_order_release);
                schedLog("[scheduler]   -> APPROVED (active_kernels=%d)\n", cs.active_kernels);
            } else {
                channel->result.store(-1, std::memory_order_release);
                channel->state.store(static_cast<uint32_t>(SchedState::REJECTED),
                                     std::memory_order_release);
                schedLog("[scheduler]   -> REJECTED (max_kernels=%d reached)\n",
                         cfg_.max_concurrent_kernels);
            }
            break;
        }
        case SchedOp::KERNEL_COMPLETE: {
            if (cs.active_kernels > 0) cs.active_kernels--;
            // Must reset to IDLE so the shim's schedReport() unblocks.
            channel->result.store(0, std::memory_order_release);
            channel->state.store(static_cast<uint32_t>(SchedState::IDLE),
                                 std::memory_order_release);
            schedLog("[scheduler]   -> KERNEL_COMPLETE (active_kernels=%d)\n", cs.active_kernels);
            break;
        }
        case SchedOp::MEMCPY_REQUEST: {
            int total_active = 0;
            for (const auto& [id, c] : clients_) {
                total_active += c.active_memcpy;
            }
            if (cfg_.max_concurrent_memcpy <= 0 ||
                total_active < cfg_.max_concurrent_memcpy) {
                cs.active_memcpy++;
                channel->result.store(0, std::memory_order_release);
                channel->state.store(static_cast<uint32_t>(SchedState::APPROVED),
                                     std::memory_order_release);
                schedLog("[scheduler]   -> APPROVED (active_memcpy=%d)\n", cs.active_memcpy);
            } else {
                channel->result.store(-1, std::memory_order_release);
                channel->state.store(static_cast<uint32_t>(SchedState::REJECTED),
                                     std::memory_order_release);
                schedLog("[scheduler]   -> REJECTED (max_memcpy=%d reached)\n",
                         cfg_.max_concurrent_memcpy);
            }
            break;
        }
        case SchedOp::MEMCPY_COMPLETE: {
            if (cs.active_memcpy > 0) cs.active_memcpy--;
            channel->result.store(0, std::memory_order_release);
            channel->state.store(static_cast<uint32_t>(SchedState::IDLE),
                                 std::memory_order_release);
            schedLog("[scheduler]   -> MEMCPY_COMPLETE (active_memcpy=%d)\n", cs.active_memcpy);
            break;
        }
        default:
            channel->result.store(-1, std::memory_order_release);
            channel->state.store(static_cast<uint32_t>(SchedState::REJECTED),
                                 std::memory_order_release);
            schedLog("[scheduler]   -> REJECTED (unknown op)\n");
            break;
    }
}

size_t Scheduler::clientCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    return clients_.size();
}

}  // namespace vgpu
