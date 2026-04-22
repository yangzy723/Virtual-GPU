#include "vgpu/common/rpc_client.h"

#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace vgpu {
namespace {

int rpcTimeoutMs() {
    static int timeout_ms = []() {
        const char* env = std::getenv("VGPU_RPC_TIMEOUT_MS");
        if (env == nullptr || env[0] == '\0') {
            return 30000;
        }

        char* end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        if (end == env || parsed <= 0) {
            return 30000;
        }

        if (parsed > 600000) {
            return 600000;
        }
        return static_cast<int>(parsed);
    }();
    return timeout_ms;
}

bool writeAll(int fd, const void* data, std::size_t len) {
    const std::uint8_t* p = static_cast<const std::uint8_t*>(data);
    std::size_t done = 0;
    while (done < len) {
        ssize_t n = write(fd, p + done, len - done);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }
        done += static_cast<std::size_t>(n);
    }
    return true;
}

bool readAll(int fd, void* data, std::size_t len) {
    std::uint8_t* p = static_cast<std::uint8_t*>(data);
    std::size_t done = 0;
    while (done < len) {
        ssize_t n = read(fd, p + done, len - done);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }
        done += static_cast<std::size_t>(n);
    }
    return true;
}

}  // namespace

RpcClient::RpcClient() = default;

const std::string& RpcClient::socketPath() {
    if (!socket_path_.empty()) {
        return socket_path_;
    }

    const char* env = std::getenv("VGPU_SERVER_SOCK");
    if (env != nullptr && env[0] != '\0') {
        socket_path_ = env;
    } else {
        socket_path_ = "/tmp/vgpu_server.sock";
    }
    return socket_path_;
}

int RpcClient::connectSocket() {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }

    const int timeout_ms = rpcTimeoutMs();
    timeval tv{};
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = static_cast<suseconds_t>((timeout_ms % 1000) * 1000);
    if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0 ||
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0) {
        close(fd);
        return -1;
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    const std::string& path = socketPath();
    if (path.size() >= sizeof(addr.sun_path)) {
        close(fd);
        errno = ENAMETOOLONG;
        return -1;
    }
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }

    return fd;
}

RpcResult RpcClient::call(
    RpcOp op,
    std::uint64_t app_id,
    std::uint64_t context_id,
    int device,
    const void* payload,
    std::size_t payload_size,
    const void* extra_payload,
    std::size_t extra_payload_size) {
    RpcResult out;
    int fd = connectSocket();
    if (fd < 0) {
        out.status = cudaErrorUnknown;
        return out;
    }

    RpcRequestHeader hdr{};
    hdr.magic = kRpcMagic;
    hdr.version = kRpcVersion;
    hdr.op = static_cast<std::uint32_t>(op);
    hdr.app_id = app_id;
    hdr.context_id = context_id;
    hdr.device = device;
    hdr.payload_size = payload_size + extra_payload_size;

    bool ok = writeAll(fd, &hdr, sizeof(hdr));
    if (ok && payload_size > 0) {
        ok = writeAll(fd, payload, payload_size);
    }
    if (ok && extra_payload_size > 0) {
        ok = writeAll(fd, extra_payload, extra_payload_size);
    }

    RpcResponseHeader rsp{};
    if (ok) {
        ok = readAll(fd, &rsp, sizeof(rsp));
    }

    if (ok && rsp.magic == kRpcMagic && rsp.version == kRpcVersion) {
        out.status = rsp.status;
        out.aux_u64 = rsp.aux_u64;
        if (rsp.payload_size > 0) {
            out.payload.resize(static_cast<std::size_t>(rsp.payload_size));
            ok = readAll(fd, out.payload.data(), out.payload.size());
            if (!ok) {
                out.status = cudaErrorUnknown;
            }
        }
    } else {
        out.status = cudaErrorUnknown;
    }

    close(fd);
    return out;
}

}  // namespace vgpu
