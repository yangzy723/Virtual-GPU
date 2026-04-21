#pragma once

#include <cstdint>

namespace vgpu {

class ContextRegistry {
public:
    std::uint64_t acquireContextId(int device);
};

}  // namespace vgpu
