#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
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

}  // namespace vgpu