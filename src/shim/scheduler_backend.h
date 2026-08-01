#pragma once

#include <cstdint>

#include "vgpu/protocol.h"

namespace vgpu {

bool schedulerRequest(SchedOp op, uint64_t value, int device);
void schedulerReport(SchedOp op, uint64_t value, int device);

}  // namespace vgpu
