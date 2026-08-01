#include "scheduler.h"

#include <cstdarg>
#include <cstdio>
#include <limits>

namespace vgpu {
namespace {

void schedLog(bool enabled, const char* format, ...) {
    if (!enabled) return;
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
}

const char* opName(SchedOp op) {
    switch (op) {
        case SchedOp::ALLOC_REQUEST: return "ALLOC_REQUEST";
        case SchedOp::FREE: return "FREE";
        case SchedOp::KERNEL_REQUEST: return "KERNEL_REQUEST";
        case SchedOp::KERNEL_COMPLETE: return "KERNEL_COMPLETE";
        case SchedOp::MEMCPY_REQUEST: return "MEMCPY_REQUEST";
        case SchedOp::MEMCPY_COMPLETE: return "MEMCPY_COMPLETE";
        default: return "UNKNOWN";
    }
}

void respond(ShmChannel* channel, SchedState state, int32_t result = 0) {
    channel->result.store(result, std::memory_order_release);
    channel->state.store(static_cast<uint32_t>(state), std::memory_order_release);
}

}  // namespace

Scheduler::Scheduler(SchedulerConfig config, bool verbose)
    : config_(config), verbose_(verbose) {}

void Scheduler::registerClient(uint64_t client_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    clients_[client_id].memory_used = config_.context_overhead_bytes;
}

void Scheduler::unregisterClient(uint64_t client_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    clients_.erase(client_id);
}

void Scheduler::processRequest(uint64_t client_id, ShmChannel* channel) {
    const auto op = static_cast<SchedOp>(
        channel->op.load(std::memory_order_acquire));
    const uint64_t value = channel->value.load(std::memory_order_acquire);

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = clients_.find(client_id);
    if (it == clients_.end()) {
        respond(channel, SchedState::REJECTED, -1);
        return;
    }

    auto& client = it->second;
    schedLog(verbose_, "[scheduler] pid=%llu op=%s value=%llu mem_used=%llu\n",
             static_cast<unsigned long long>(client_id), opName(op),
             static_cast<unsigned long long>(value),
             static_cast<unsigned long long>(client.memory_used));

    switch (op) {
        case SchedOp::ALLOC_REQUEST: {
            const bool no_overflow = value <=
                std::numeric_limits<uint64_t>::max() - client.memory_used;
            const bool within_limit = config_.memory_limit_bytes == 0 ||
                (client.memory_used <= config_.memory_limit_bytes &&
                 value <= config_.memory_limit_bytes - client.memory_used);
            if (!no_overflow || !within_limit) {
                respond(channel, SchedState::REJECTED, -1);
                schedLog(verbose_, "[scheduler]   -> REJECTED (quota exceeded)\n");
                break;
            }

            client.memory_used += value;
            respond(channel, SchedState::APPROVED);
            schedLog(verbose_, "[scheduler]   -> APPROVED (mem_used now %llu)\n",
                     static_cast<unsigned long long>(client.memory_used));
            break;
        }
        case SchedOp::FREE: {
            const uint64_t allocated =
                client.memory_used > config_.context_overhead_bytes
                ? client.memory_used - config_.context_overhead_bytes : 0;
            client.memory_used -= value > allocated ? allocated : value;
            respond(channel, SchedState::IDLE);
            schedLog(verbose_, "[scheduler]   -> FREE complete (mem_used now %llu)\n",
                     static_cast<unsigned long long>(client.memory_used));
            break;
        }
        case SchedOp::KERNEL_REQUEST: {
            int total_active = 0;
            for (const auto& entry : clients_) {
                total_active += entry.second.active_kernels;
            }
            if (total_active >= config_.max_concurrent_kernels) {
                respond(channel, SchedState::REJECTED, -1);
                schedLog(verbose_,
                         "[scheduler]   -> REJECTED (max_kernels=%d reached)\n",
                         config_.max_concurrent_kernels);
                break;
            }

            ++client.active_kernels;
            respond(channel, SchedState::APPROVED);
            schedLog(verbose_, "[scheduler]   -> APPROVED (active_kernels=%d)\n",
                     client.active_kernels);
            break;
        }
        case SchedOp::KERNEL_COMPLETE:
            if (client.active_kernels > 0) --client.active_kernels;
            respond(channel, SchedState::IDLE);
            schedLog(verbose_,
                     "[scheduler]   -> KERNEL_COMPLETE (active_kernels=%d)\n",
                     client.active_kernels);
            break;
        case SchedOp::MEMCPY_REQUEST: {
            int total_active = 0;
            for (const auto& entry : clients_) {
                total_active += entry.second.active_memcpy;
            }
            if (config_.max_concurrent_memcpy > 0 &&
                total_active >= config_.max_concurrent_memcpy) {
                respond(channel, SchedState::REJECTED, -1);
                schedLog(verbose_,
                         "[scheduler]   -> REJECTED (max_memcpy=%d reached)\n",
                         config_.max_concurrent_memcpy);
                break;
            }

            ++client.active_memcpy;
            respond(channel, SchedState::APPROVED);
            schedLog(verbose_, "[scheduler]   -> APPROVED (active_memcpy=%d)\n",
                     client.active_memcpy);
            break;
        }
        case SchedOp::MEMCPY_COMPLETE:
            if (client.active_memcpy > 0) --client.active_memcpy;
            respond(channel, SchedState::IDLE);
            schedLog(verbose_,
                     "[scheduler]   -> MEMCPY_COMPLETE (active_memcpy=%d)\n",
                     client.active_memcpy);
            break;
        default:
            respond(channel, SchedState::REJECTED, -1);
            schedLog(verbose_, "[scheduler]   -> REJECTED (unknown op)\n");
            break;
    }
}

}  // namespace vgpu
