#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "vgpu/common/cuda_abi.h"
#include "vgpu/common/protocol.h"

namespace vgpu {

struct RpcResult {
    cudaError_t status = cudaErrorUnknown;
    std::uint64_t aux_u64 = 0;
    std::vector<std::uint8_t> payload;
};

class RpcClient {
public:
    RpcClient();

    RpcResult call(
        RpcOp op,
        std::uint64_t app_id,
        std::uint64_t context_id,
        int device,
        const void* payload,
        std::size_t payload_size,
        const void* extra_payload,
        std::size_t extra_payload_size);

private:
    int connectSocket();
    const std::string& socketPath();

    std::string socket_path_;
};

}  // namespace vgpu
