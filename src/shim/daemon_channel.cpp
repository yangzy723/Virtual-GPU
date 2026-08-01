#include "daemon_channel.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <thread>

#include "vgpu/config.h"
#include "vgpu/io.h"

#if defined(__x86_64__) || defined(_M_X64)
#define VGPU_SPIN_PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__) || defined(_M_ARM64)
#define VGPU_SPIN_PAUSE() asm volatile("yield" ::: "memory")
#else
#define VGPU_SPIN_PAUSE() ((void)0)
#endif

namespace vgpu {

DaemonChannel::DaemonChannel()
    : wait_iterations_(config::getInt(
          "GPU_SCHEDULER_WAIT_ITERS", 200000, 1, 5000000)),
      trace_enabled_(config::getBool("VGPU_TRACE", true)) {}

DaemonChannel::~DaemonChannel() {
    if (channel_) munmap(channel_, sizeof(ShmChannel));
    if (daemon_fd_ >= 0) close(daemon_fd_);
}

void DaemonChannel::trace(const char* format, ...) const {
    if (!trace_enabled_) return;
    std::fprintf(stderr, "[vGPU trace] ");
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fprintf(stderr, "\n");
}

void DaemonChannel::warnFailOpenOnce(const char* reason, const std::string& detail) {
    if (fail_open_warned_.exchange(true, std::memory_order_acq_rel)) return;
    std::fprintf(stderr,
                 "[vGPU warning] scheduler unavailable, fail-open passthrough to real CUDA driver: %s",
                 reason);
    if (!detail.empty()) std::fprintf(stderr, " (%s)", detail.c_str());
    std::fprintf(stderr, "\n");
}

bool DaemonChannel::connectLocked() {
    if (channel_) return true;

    const std::string path = config::getEnvOrConfig(
        "GPU_SCHEDULER_SOCKET", defaultSocketPath());
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        warnFailOpenOnce("socket() failed", std::strerror(errno));
        return false;
    }

    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (path.size() >= sizeof(address.sun_path)) {
        close(fd);
        warnFailOpenOnce("daemon socket path is too long", path);
        return false;
    }
    std::strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1);
    if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        const std::string detail = path + ": " + std::strerror(errno);
        close(fd);
        warnFailOpenOnce("cannot connect to daemon", detail);
        return false;
    }

    HandshakeRequest request{};
    request.op = static_cast<uint32_t>(HandshakeOp::HELLO);
    request.client_id = static_cast<uint64_t>(getpid());
    HandshakeResponse response{};
    if (!io::writeAll(fd, &request, sizeof(request)) ||
        !io::readAll(fd, &response, sizeof(response)) || response.status != 0) {
        close(fd);
        warnFailOpenOnce("daemon handshake failed", path);
        return false;
    }

    const int shm_fd = shm_open(response.shm_name, O_RDWR, 0666);
    if (shm_fd < 0) {
        close(fd);
        warnFailOpenOnce("shared-memory open failed", response.shm_name);
        return false;
    }
    auto* mapped = static_cast<ShmChannel*>(
        mmap(nullptr, sizeof(ShmChannel), PROT_READ | PROT_WRITE,
             MAP_SHARED, shm_fd, 0));
    close(shm_fd);
    if (mapped == MAP_FAILED) {
        close(fd);
        warnFailOpenOnce("shared-memory mmap failed", response.shm_name);
        return false;
    }

    channel_ = mapped;
    daemon_fd_ = fd;  // retained so daemon can detect process lifetime
    return true;
}

SchedState DaemonChannel::waitForDecisionLocked() const {
    if (!channel_) return SchedState::REJECTED;
    for (int iteration = 0; iteration < wait_iterations_; ++iteration) {
        const auto state = static_cast<SchedState>(
            channel_->state.load(std::memory_order_acquire));
        if (state == SchedState::APPROVED || state == SchedState::REJECTED) {
            return state;
        }
        if (iteration < 1000) VGPU_SPIN_PAUSE();
        else std::this_thread::yield();
    }
    return SchedState::PENDING;
}

bool DaemonChannel::waitForIdleLocked() const {
    if (!channel_) return false;
    for (int iteration = 0; iteration < wait_iterations_; ++iteration) {
        const auto state = static_cast<SchedState>(
            channel_->state.load(std::memory_order_acquire));
        if (state == SchedState::IDLE) return true;
        if (iteration < 1000) VGPU_SPIN_PAUSE();
        else std::this_thread::yield();
    }
    return false;
}

bool DaemonChannel::request(SchedOp op, uint64_t value, int device) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!connectLocked()) return true;
    if (!waitForIdleLocked()) {
        warnFailOpenOnce("shared-memory channel did not become IDLE before request");
        return true;
    }

    trace("daemon request op=%u value=%llu device=%d",
          static_cast<unsigned int>(op),
          static_cast<unsigned long long>(value), device);
    channel_->device.store(static_cast<uint32_t>(device), std::memory_order_relaxed);
    channel_->client_id.store(static_cast<uint64_t>(getpid()), std::memory_order_relaxed);
    channel_->value.store(value, std::memory_order_relaxed);
    channel_->op.store(static_cast<uint32_t>(op), std::memory_order_relaxed);
    channel_->state.store(static_cast<uint32_t>(SchedState::PENDING),
                          std::memory_order_release);

    const auto decision = waitForDecisionLocked();
    if (decision == SchedState::APPROVED || decision == SchedState::REJECTED) {
        channel_->state.store(static_cast<uint32_t>(SchedState::IDLE),
                              std::memory_order_release);
        trace("daemon %s op=%u",
              decision == SchedState::APPROVED ? "approved" : "rejected",
              static_cast<unsigned int>(op));
        return decision == SchedState::APPROVED;
    }

    trace("daemon request timeout, bypass op=%u", static_cast<unsigned int>(op));
    warnFailOpenOnce("daemon decision timeout");
    return true;
}

void DaemonChannel::report(SchedOp op, uint64_t value, int device) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!connectLocked()) return;
    if (!waitForIdleLocked()) {
        warnFailOpenOnce("shared-memory channel did not become IDLE before report");
        return;
    }

    trace("daemon report op=%u value=%llu device=%d",
          static_cast<unsigned int>(op),
          static_cast<unsigned long long>(value), device);
    channel_->device.store(static_cast<uint32_t>(device), std::memory_order_relaxed);
    channel_->client_id.store(static_cast<uint64_t>(getpid()), std::memory_order_relaxed);
    channel_->value.store(value, std::memory_order_relaxed);
    channel_->op.store(static_cast<uint32_t>(op), std::memory_order_relaxed);
    channel_->state.store(static_cast<uint32_t>(SchedState::PENDING),
                          std::memory_order_release);
    if (waitForIdleLocked()) {
        trace("daemon report done op=%u", static_cast<unsigned int>(op));
        return;
    }
    trace("daemon report timeout op=%u", static_cast<unsigned int>(op));
    warnFailOpenOnce("daemon report timeout");
}

}  // namespace vgpu
