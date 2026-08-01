#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

#include "vgpu/protocol.h"

namespace vgpu {

// Client side of the daemon protocol. UDS is used only for registration and
// process-lifetime detection; scheduling messages use the per-client SHM slot.
class DaemonChannel {
public:
    DaemonChannel();
    ~DaemonChannel();

    DaemonChannel(const DaemonChannel&) = delete;
    DaemonChannel& operator=(const DaemonChannel&) = delete;

    // Returns true on approval. Transport failures deliberately fail open and
    // emit a single warning, preserving the project's availability contract.
    bool request(SchedOp op, uint64_t value, int device);
    void report(SchedOp op, uint64_t value, int device);

private:
    bool connectLocked();
    SchedState waitForDecisionLocked() const;
    bool waitForIdleLocked() const;
    void warnFailOpenOnce(const char* reason, const std::string& detail = "");
    void trace(const char* format, ...) const;

    std::mutex mutex_;
    ShmChannel* channel_ = nullptr;
    int daemon_fd_ = -1;
    int wait_iterations_ = 200000;
    bool trace_enabled_ = false;
    std::atomic<bool> fail_open_warned_{false};
};

}  // namespace vgpu
