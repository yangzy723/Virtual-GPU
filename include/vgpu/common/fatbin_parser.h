#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vgpu {

// Per-parameter info extracted from PTX .param directive
struct ParamInfo {
    uint32_t size;       // size in bytes (1/2/4/8/16)
    uint32_t alignment;  // natural alignment (min 4)
};

// All parameter info for a single kernel
struct KernelParamInfo {
    std::string mangled_name;       // device function mangled name
    std::vector<ParamInfo> params;  // parameters in order
    uint32_t total_param_bytes;     // total packed buffer size (with alignment)
};

// Extract kernel parameter info from a fatbin blob.
// fat_data may point to:
//   (a) the __cudaRegisterFatBinary wrapper struct (magic 0x466243B1), OR
//   (b) the raw fatbin image (magic 0xBA55ED50), OR
//   (c) a raw PTX string
// fat_size: known byte length; pass 0 to auto-detect from header.
std::vector<KernelParamInfo> parseFatbin(const void* fat_data, size_t fat_size = 0);

// Compute total packed parameter buffer size (CUDA alignment rules).
uint32_t computeParamBufSize(const std::vector<ParamInfo>& params);

// Pack void** args into a flat buffer suitable for cuLaunchKernel's
// CU_LAUNCH_PARAM_BUFFER_POINTER mode.
// args[i] must be a valid pointer to the i-th argument value.
std::vector<uint8_t> packArgs(const std::vector<ParamInfo>& params, void** args);

}  // namespace vgpu
