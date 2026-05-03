#pragma once

#include <atomic>
#include <cstdint>

namespace vgpu {

// ── 共享内存通道 ──────────────────────────────────────────────────────────
// 每个客户端一个，由 daemon 创建，双方 mmap

enum class SchedOp : uint32_t {
    NONE            = 0,
    ALLOC_REQUEST   = 1,
    FREE            = 2,
    KERNEL_REQUEST  = 3,
    KERNEL_COMPLETE = 4,
    MEMCPY_REQUEST  = 5,
    MEMCPY_COMPLETE = 6,
};

enum class SchedState : uint32_t {
    IDLE     = 0,
    PENDING  = 1,
    APPROVED = 2,
    REJECTED = 3,
};

struct alignas(64) ShmChannel {
    std::atomic<uint32_t> state;       // SchedState
    std::atomic<uint32_t> op;          // SchedOp
    std::atomic<uint64_t> value;       // ALLOC 字节数 / 0
    std::atomic<int32_t>  result;      // 0=ok, <0=error
    std::atomic<uint32_t> device;
    std::atomic<uint64_t> client_id;
    std::atomic<uint32_t> dying;       // daemon 侧：客户端断开标记
    uint8_t reserved[24];

    void reset() {
        state.store(0, std::memory_order_relaxed);
        op.store(0, std::memory_order_relaxed);
        value.store(0, std::memory_order_relaxed);
        result.store(0, std::memory_order_relaxed);
        dying.store(0, std::memory_order_relaxed);
    }
};

static_assert(sizeof(ShmChannel) <= 128);

// ── UDS 握手协议 ─────────────────────────────────────────────────────────
// 仅在连接建立时使用，之后全部走共享内存

enum class HandshakeOp : uint32_t {
    HELLO = 1,  // Client → Daemon
    FREE  = 3,  // Client → Daemon（优雅退出）
};

struct HandshakeRequest {
    uint32_t op;
    uint32_t padding;
    uint64_t client_id;  // PID
};

struct HandshakeResponse {
    int32_t  status;      // 0=ok, <0=error
    uint32_t padding;
    char     shm_name[64];
};

inline const char* defaultSocketPath() {
    return "/tmp/vgpu_control.sock";
}

}  // namespace vgpu
