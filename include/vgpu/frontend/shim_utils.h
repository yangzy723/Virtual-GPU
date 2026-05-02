#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "vgpu/common/cuda_abi.h"

namespace vgpu {

enum class ProcMapAccess {
    Read,
    Write,
};

inline bool envFlagEnabled(const char* key) {
    const char* value = std::getenv(key);
    return value != nullptr && value[0] == '1';
}

inline cudaError_t unsupported() {
    return cudaErrorNotSupported;
}

inline cudaError_t unsupported(const char* where) {
    if (std::getenv("VGPU_DEBUG") != nullptr && where != nullptr) {
        std::fprintf(stderr, "[vGPU][unsupported] %s -> cudaErrorNotSupported\n", where);
    }
    return unsupported();
}

inline bool processRangeHasAccess(std::uintptr_t addr, std::size_t len, ProcMapAccess access) {
    if (addr == 0 || len == 0) {
        return false;
    }

    const std::uintptr_t end = addr + len;
    if (end < addr) {
        return false;
    }

    struct Range {
        std::uintptr_t lo;
        std::uintptr_t hi;
    };

    const std::size_t perm_index = (access == ProcMapAccess::Read) ? 0 : 1;
    const char perm_char = (access == ProcMapAccess::Read) ? 'r' : 'w';

    static thread_local std::vector<Range> readable_ranges;
    static thread_local std::vector<Range> writable_ranges;
    static thread_local bool readable_loaded = false;
    static thread_local bool writable_loaded = false;

    std::vector<Range>& ranges = (access == ProcMapAccess::Read) ? readable_ranges : writable_ranges;
    bool& loaded = (access == ProcMapAccess::Read) ? readable_loaded : writable_loaded;

    if (!loaded) {
        loaded = true;
        std::ifstream maps("/proc/self/maps");
        std::string line;
        while (std::getline(maps, line)) {
            const std::size_t dash = line.find('-');
            const std::size_t space = line.find(' ');
            if (dash == std::string::npos || space == std::string::npos || dash >= space) {
                continue;
            }
            if (space < 4 || line[(space - 4) + perm_index] != perm_char) {
                continue;
            }

            const std::uintptr_t lo = static_cast<std::uintptr_t>(
                std::strtoull(line.substr(0, dash).c_str(), nullptr, 16));
            const std::uintptr_t hi = static_cast<std::uintptr_t>(
                std::strtoull(line.substr(dash + 1, space - dash - 1).c_str(), nullptr, 16));
            if (lo < hi) {
                ranges.push_back({lo, hi});
            }
        }
    }

    for (const auto& range : ranges) {
        if (addr >= range.lo && end <= range.hi) {
            return true;
        }
    }
    return false;
}

// ── Fatbin image helpers ──────────────────────────────────────────────────
// Shared between interceptor.cpp and driver_interceptor.cpp.

static constexpr std::uint32_t kFatWrapMagic = 0x466243B1u;
static constexpr std::uint32_t kFatBinMagic  = 0xBA55ED50u;

struct FatbinImage {
    const void* raw = nullptr;
    std::size_t size = 0;
};

inline const void* resolveRawFatbinPtr(const void* image) {
    if (!image) return nullptr;
    const auto* p = static_cast<const std::uint8_t*>(image);
    std::uint32_t magic = 0;
    std::memcpy(&magic, p, sizeof(magic));
    if (magic == kFatWrapMagic) {
        const void* data = nullptr;
        std::memcpy(&data, p + 8, sizeof(data));
        return data;
    }
    return image;
}

inline std::size_t fatbinImageSize(const void* image) {
    if (!image) return 0;

    const auto* p = static_cast<const std::uint8_t*>(image);
    std::uint32_t magic = 0;
    std::memcpy(&magic, p, sizeof(magic));

    if (magic == kFatWrapMagic) {
        const void* data = nullptr;
        std::memcpy(&data, p + 8, sizeof(data));
        if (!data) return 0;
        p = static_cast<const std::uint8_t*>(data);
        std::memcpy(&magic, p, sizeof(magic));
    }
    if (magic != kFatBinMagic) return 0;

    // CUDA 12+ fatbin header variant:
    //   u32 magic, u32 version, u64 data_size, u32 unknown, u32 header_size
    std::uint32_t version = 0;
    std::uint64_t data_size64 = 0;
    std::uint32_t header_size_v2 = 0;
    std::memcpy(&version,        p + 4,  sizeof(version));
    std::memcpy(&data_size64,    p + 8,  sizeof(data_size64));
    std::memcpy(&header_size_v2, p + 20, sizeof(header_size_v2));
    if (version == 0x00100001u && header_size_v2 >= 16 && header_size_v2 <= (1u << 20)) {
        std::size_t total_v2 = static_cast<std::size_t>(data_size64 + static_cast<std::uint64_t>(header_size_v2));
        if (total_v2 >= header_size_v2) {
            return total_v2;
        }
    }

    // Legacy layout: u32 header_size @8, u32 data_size @12
    std::uint32_t hdr_size = 0, data_size = 0;
    std::memcpy(&hdr_size,  p + 8,  sizeof(hdr_size));
    std::memcpy(&data_size, p + 12, sizeof(data_size));
    return static_cast<std::size_t>(hdr_size) + static_cast<std::size_t>(data_size);
}

// ── Argument packing helpers ──────────────────────────────────────────────

inline std::vector<std::uint8_t> makePackedArgsFromTotalBytes(void** args, std::uint32_t total_param_bytes) {
    std::vector<std::uint8_t> out;
    if (args == nullptr || total_param_bytes == 0) {
        return out;
    }

    out.assign(total_param_bytes, 0);
    const std::size_t slot = sizeof(std::uint64_t);
    std::size_t max_args = (static_cast<std::size_t>(total_param_bytes) + slot - 1) / slot;
    if (max_args > 64) {
        max_args = 64;
    }

    for (std::size_t i = 0; i < max_args; ++i) {
        void* arg_ptr = args[i];
        if (arg_ptr == nullptr && i > 0) {
            break;
        }
        if (arg_ptr == nullptr) {
            continue;
        }

        const std::size_t off = i * slot;
        if (off >= out.size()) {
            break;
        }
        const std::size_t n = std::min(slot, out.size() - off);
        std::memcpy(out.data() + off, arg_ptr, n);
    }

    return out;
}

inline std::vector<std::uint8_t> makeArgumentSlotsUnknown(void** args, std::size_t max_slots = 12) {
    std::vector<std::uint8_t> out;
    if (args == nullptr || max_slots == 0) {
        return out;
    }

    const std::size_t slot = sizeof(std::uint64_t);
    out.assign(max_slots * slot, 0);

    std::size_t used = 0;
    for (std::size_t i = 0; i < max_slots; ++i) {
        void* arg_ptr = args[i];
        if (arg_ptr == nullptr && i > 0) {
            break;
        }
        if (arg_ptr == nullptr) {
            used = i + 1;
            continue;
        }
        if (!processRangeHasAccess(reinterpret_cast<std::uintptr_t>(arg_ptr), slot, ProcMapAccess::Read)) {
            break;
        }
        std::memcpy(out.data() + i * slot, arg_ptr, slot);
        used = i + 1;
    }

    out.resize(used * slot);
    return out;
}

}  // namespace vgpu