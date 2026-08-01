#pragma once

#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>

namespace vgpu::io {

inline bool readAll(int fd, void* buffer, size_t length) {
    auto* bytes = static_cast<uint8_t*>(buffer);
    size_t done = 0;
    while (done < length) {
        const ssize_t count = read(fd, bytes + done, length - done);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        done += static_cast<size_t>(count);
    }
    return true;
}

inline bool writeAll(int fd, const void* buffer, size_t length) {
    const auto* bytes = static_cast<const uint8_t*>(buffer);
    size_t done = 0;
    while (done < length) {
        const ssize_t count = write(fd, bytes + done, length - done);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        done += static_cast<size_t>(count);
    }
    return true;
}

}  // namespace vgpu::io
