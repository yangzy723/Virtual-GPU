#include "vgpu/common/context_registry.h"

#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace vgpu {
namespace {

struct ContextKey {
    pid_t pid;
    std::size_t tid_hash;
    int device;

    bool operator==(const ContextKey& rhs) const {
        return pid == rhs.pid && tid_hash == rhs.tid_hash && device == rhs.device;
    }
};

struct ContextKeyHasher {
    std::size_t operator()(const ContextKey& key) const {
        std::size_t h1 = std::hash<int>()(key.pid);
        std::size_t h2 = std::hash<std::size_t>()(key.tid_hash);
        std::size_t h3 = std::hash<int>()(key.device);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

std::mutex g_ctx_mu;
std::unordered_map<ContextKey, std::uint64_t, ContextKeyHasher> g_ctx_ids;
std::atomic<std::uint64_t> g_next_ctx_id{1};

}  // namespace

std::uint64_t ContextRegistry::acquireContextId(int device) {
    ContextKey key{getpid(), std::hash<std::thread::id>()(std::this_thread::get_id()), device};

    std::lock_guard<std::mutex> lock(g_ctx_mu);
    auto it = g_ctx_ids.find(key);
    if (it != g_ctx_ids.end()) {
        return it->second;
    }

    std::uint64_t id = g_next_ctx_id.fetch_add(1);
    g_ctx_ids.emplace(key, id);
    return id;
}

}  // namespace vgpu
