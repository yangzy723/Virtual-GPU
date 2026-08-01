#include "scheduler_backend.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>

#include "daemon_channel.h"
#include "local_scheduler.h"
#include "vgpu/config.h"
#include "vgpu/scheduler_config.h"

namespace vgpu {
namespace {

enum class SchedulerMode { DAEMON, LOCAL };

SchedulerMode configuredMode() {
    std::string raw = config::getEnvOrConfig("VGPU_SCHEDULER_MODE");
    if (raw.empty() && config::getBool("VGPU_LOCAL_MODE", false)) {
        return SchedulerMode::LOCAL;
    }
    std::transform(raw.begin(), raw.end(), raw.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (raw.empty() || raw == "daemon") return SchedulerMode::DAEMON;
    if (raw == "local") return SchedulerMode::LOCAL;

    std::fprintf(stderr,
                 "[vGPU warning] invalid VGPU_SCHEDULER_MODE='%s'; using daemon\n",
                 raw.c_str());
    return SchedulerMode::DAEMON;
}

class Backend {
public:
    Backend() : mode(configuredMode()), local(loadSchedulerConfig()) {
        if (config::getBool("GPU_SCHEDULER_VERBOSE", false)) {
            std::fprintf(stderr, "[vGPU scheduler] mode=%s\n",
                         mode == SchedulerMode::LOCAL ? "local" : "daemon");
        }
    }

    SchedulerMode mode;
    LocalScheduler local;
    DaemonChannel daemon;
};

// Process-lifetime storage avoids teardown races with detached CUDA completion
// callbacks that may still report while shared objects are being unloaded.
Backend& backend() {
    static Backend* instance = new Backend();
    return *instance;
}

}  // namespace

bool schedulerRequest(SchedOp op, uint64_t value, int device) {
    auto& selected = backend();
    return selected.mode == SchedulerMode::LOCAL
        ? selected.local.request(op, value)
        : selected.daemon.request(op, value, device);
}

void schedulerReport(SchedOp op, uint64_t value, int device) {
    auto& selected = backend();
    if (selected.mode == SchedulerMode::LOCAL) {
        selected.local.report(op, value);
    } else {
        selected.daemon.report(op, value, device);
    }
}

}  // namespace vgpu
