#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include "local_scheduler.h"

using namespace std::chrono_literals;

namespace {

bool expect(bool condition, const char* message) {
    if (condition) return true;
    std::cerr << "local_scheduler_test: " << message << '\n';
    return false;
}

}  // namespace

int main() {
    vgpu::SchedulerConfig config;
    config.context_overhead_bytes = 16;
    config.memory_limit_bytes = 64;
    config.max_concurrent_kernels = 1;
    config.max_concurrent_memcpy = 1;
    vgpu::LocalScheduler scheduler(config);

    if (!expect(scheduler.request(vgpu::SchedOp::ALLOC_REQUEST, 32),
                "allocation inside quota was rejected") ||
        !expect(!scheduler.request(vgpu::SchedOp::ALLOC_REQUEST, 17),
                "allocation above quota was approved") ||
        !expect(scheduler.snapshot().memory_used_bytes == 48,
                "memory accounting after allocation is incorrect")) {
        return 1;
    }
    scheduler.report(vgpu::SchedOp::FREE, 32);
    if (!expect(scheduler.snapshot().memory_used_bytes == 16,
                "context overhead was not preserved after FREE")) {
        return 1;
    }

    if (!expect(scheduler.request(vgpu::SchedOp::KERNEL_REQUEST, 0),
                "first kernel request was rejected")) {
        return 1;
    }
    std::atomic<bool> second_started{false};
    std::atomic<bool> second_admitted{false};
    std::thread waiter([&] {
        second_started.store(true, std::memory_order_release);
        const bool admitted = scheduler.request(vgpu::SchedOp::KERNEL_REQUEST, 0);
        second_admitted.store(admitted, std::memory_order_release);
    });
    while (!second_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(20ms);
    if (!expect(!second_admitted.load(std::memory_order_acquire),
                "second kernel bypassed the concurrency limit")) {
        scheduler.report(vgpu::SchedOp::KERNEL_COMPLETE, 0);
        waiter.join();
        return 1;
    }

    scheduler.report(vgpu::SchedOp::KERNEL_COMPLETE, 0);
    waiter.join();
    if (!expect(second_admitted.load(std::memory_order_acquire),
                "waiting kernel was not admitted after completion")) {
        return 1;
    }
    scheduler.report(vgpu::SchedOp::KERNEL_COMPLETE, 0);

    if (!expect(scheduler.request(vgpu::SchedOp::MEMCPY_REQUEST, 4096),
                "memcpy request was rejected")) {
        return 1;
    }
    scheduler.report(vgpu::SchedOp::MEMCPY_COMPLETE, 4096);
    const auto final_state = scheduler.snapshot();
    if (!expect(final_state.active_kernels == 0, "kernel count leaked") ||
        !expect(final_state.active_memcpy == 0, "memcpy count leaked")) {
        return 1;
    }
    return 0;
}
