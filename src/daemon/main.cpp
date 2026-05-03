// gpu_scheduler daemon: UDS handshake + shared-memory scheduling fast path.

#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdarg>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "scheduler.h"
#include "vgpu/config.h"
#include "vgpu/protocol.h"

static std::atomic<bool> g_stop{false};
static int g_poll_interval_us = 100;

static std::string socketPath() {
    return vgpu::config::getEnvOrConfig("GPU_SCHEDULER_SOCKET", vgpu::defaultSocketPath());
}

static bool g_verbose = false;

static void log(const char* fmt, ...) {
    if (!g_verbose) return;
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
}

struct ClientInfo {
    int fd = -1;
    uint64_t client_id = 0;
    std::string shm_name;
    vgpu::ShmChannel* channel = nullptr;
};

static std::unordered_map<int, ClientInfo> g_clients;
static std::mutex g_clients_mu;

static std::string shmNameForPid(uint64_t pid) {
    return "/vgpu_" + std::to_string(pid);
}

static vgpu::ShmChannel* createShm(uint64_t pid) {
    std::string name = shmNameForPid(pid);
    int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd < 0) return nullptr;

    if (ftruncate(fd, sizeof(vgpu::ShmChannel)) != 0) {
        close(fd);
        shm_unlink(name.c_str());
        return nullptr;
    }

    auto* ch = static_cast<vgpu::ShmChannel*>(
        mmap(nullptr, sizeof(vgpu::ShmChannel), PROT_READ | PROT_WRITE,
             MAP_SHARED, fd, 0));
    close(fd);

    if (ch == MAP_FAILED) {
        shm_unlink(name.c_str());
        return nullptr;
    }

    ch->reset();
    return ch;
}

static void destroyShm(uint64_t pid, vgpu::ShmChannel* channel) {
    if (channel && channel != MAP_FAILED) {
        munmap(channel, sizeof(vgpu::ShmChannel));
    }
    std::string name = shmNameForPid(pid);
    shm_unlink(name.c_str());
}

static bool readAll(int fd, void* buf, size_t len) {
    auto* p = static_cast<uint8_t*>(buf);
    size_t done = 0;
    while (done < len) {
        ssize_t n = read(fd, p + done, len - done);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

static bool writeAll(int fd, const void* buf, size_t len) {
    const auto* p = static_cast<const uint8_t*>(buf);
    size_t done = 0;
    while (done < len) {
        ssize_t n = write(fd, p + done, len - done);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        done += static_cast<size_t>(n);
    }
    return true;
}

static int createListener(const std::string& path) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    unlink(path.c_str());

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }

    if (listen(fd, 128) != 0) {
        close(fd);
        return -1;
    }

    chmod(path.c_str(), 0666);
    return fd;
}

static vgpu::Scheduler* g_scheduler = nullptr;

struct DeferredUnmap {
    uint64_t client_id;
    vgpu::ShmChannel* channel;
};

static void pollThread() {
    while (!g_stop) {
        struct PendingItem {
            uint64_t client_id;
            vgpu::ShmChannel* channel;
        };
        std::vector<PendingItem> pending;

        {
            std::lock_guard<std::mutex> lock(g_clients_mu);
            for (auto& [fd, info] : g_clients) {
                if (!info.channel) continue;
                if (info.channel->dying.load(std::memory_order_acquire)) {
                    info.channel = nullptr;
                    continue;
                }
                auto state = static_cast<vgpu::SchedState>(
                    info.channel->state.load(std::memory_order_acquire));
                if (state == vgpu::SchedState::PENDING) {
                    pending.push_back({info.client_id, info.channel});
                }
            }
        }

        for (auto& item : pending) {
            g_scheduler->processRequest(item.client_id, item.channel);
            log("[scheduler] processed request from pid %lu\n",
                static_cast<unsigned long>(item.client_id));
        }

        std::this_thread::sleep_for(std::chrono::microseconds(g_poll_interval_us));
    }
}

static void handleClientDeath(ClientInfo& info) {
    log("[scheduler] client pid %lu disconnected\n",
        static_cast<unsigned long>(info.client_id));

    g_scheduler->unregisterClient(info.client_id);

    if (info.channel) {
        info.channel->dying.store(1, std::memory_order_release);
        info.channel = nullptr;
    }
}

static void handleHandshake(int client_fd) {
    vgpu::HandshakeRequest req{};
    if (!readAll(client_fd, &req, sizeof(req))) {
        close(client_fd);
        return;
    }

    vgpu::HandshakeResponse rsp{};

    if (static_cast<vgpu::HandshakeOp>(req.op) == vgpu::HandshakeOp::HELLO) {
        auto* channel = createShm(req.client_id);
        if (!channel) {
            rsp.status = -1;
            writeAll(client_fd, &rsp, sizeof(rsp));
            close(client_fd);
            return;
        }

        std::string shm_name = shmNameForPid(req.client_id);

        g_scheduler->registerClient(req.client_id, shm_name);

        {
            std::lock_guard<std::mutex> lock(g_clients_mu);
            ClientInfo info;
            info.fd = client_fd;
            info.client_id = req.client_id;
            info.shm_name = shm_name;
            info.channel = channel;
            g_clients[client_fd] = info;
        }

        rsp.status = 0;
        std::strncpy(rsp.shm_name, shm_name.c_str(), sizeof(rsp.shm_name) - 1);
        writeAll(client_fd, &rsp, sizeof(rsp));

        log("[scheduler] registered client pid %lu, shm=%s\n",
            static_cast<unsigned long>(req.client_id), shm_name.c_str());

    } else if (static_cast<vgpu::HandshakeOp>(req.op) == vgpu::HandshakeOp::FREE) {
        {
            std::lock_guard<std::mutex> lock(g_clients_mu);
            auto it = g_clients.find(client_fd);
            if (it != g_clients.end()) {
                handleClientDeath(it->second);
                g_clients.erase(it);
            }
        }
        rsp.status = 0;
        writeAll(client_fd, &rsp, sizeof(rsp));
    }
}

int main() {
    vgpu::Scheduler::Config cfg;

    std::string mem_limit = vgpu::config::getEnvOrConfig("GPU_SCHEDULER_MEM_LIMIT_MB");
    if (!mem_limit.empty()) {
        long mb = std::strtol(mem_limit.c_str(), nullptr, 10);
        if (mb > 0) {
            cfg.per_client_mem_limit = static_cast<size_t>(mb) * 1024 * 1024;
        }
    }

    std::string overhead = vgpu::config::getEnvOrConfig("GPU_SCHEDULER_CONTEXT_OVERHEAD_MB");
    if (!overhead.empty()) {
        long mb = std::strtol(overhead.c_str(), nullptr, 10);
        if (mb > 0) {
            cfg.context_overhead = static_cast<size_t>(mb) * 1024 * 1024;
        }
    }

    std::string max_k = vgpu::config::getEnvOrConfig("GPU_SCHEDULER_MAX_KERNELS");
    if (!max_k.empty()) {
        int val = std::atoi(max_k.c_str());
        if (val > 0) cfg.max_concurrent_kernels = val;
    }

    std::string max_memcpy = vgpu::config::getEnvOrConfig("GPU_SCHEDULER_MAX_MEMCPY");
    if (!max_memcpy.empty()) {
        int val = std::atoi(max_memcpy.c_str());
        if (val >= 0) cfg.max_concurrent_memcpy = val;
    }

    std::string poll_us = vgpu::config::getEnvOrConfig("GPU_SCHEDULER_POLL_US");
    if (!poll_us.empty()) {
        long val = std::strtol(poll_us.c_str(), nullptr, 10);
        if (val > 0 && val <= 1000000) g_poll_interval_us = static_cast<int>(val);
    }

    g_verbose = vgpu::config::getBool("GPU_SCHEDULER_VERBOSE", false);

    vgpu::Scheduler sched(cfg);
    sched.setVerbose(g_verbose);
    g_scheduler = &sched;

    std::string path = socketPath();
    int listen_fd = createListener(path);
    if (listen_fd < 0) {
        std::perror("createListener");
        return 1;
    }

    int epoll_fd = epoll_create1(0);
    if (epoll_fd < 0) {
        std::perror("epoll_create1");
        return 1;
    }

    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.fd = listen_fd;
    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, listen_fd, &ev);

    std::thread poller(pollThread);

    std::fprintf(stderr,
                 "[scheduler] listening on %s (max_kernels=%d, max_memcpy=%d, context_overhead=%zuMB, poll_us=%d)\n",
                 path.c_str(), cfg.max_concurrent_kernels,
                 cfg.max_concurrent_memcpy,
                 cfg.context_overhead / (1024 * 1024), g_poll_interval_us);

    constexpr int MAX_EVENTS = 64;
    std::vector<epoll_event> events(MAX_EVENTS);

    while (!g_stop) {
        int n = epoll_wait(epoll_fd, events.data(), MAX_EVENTS, 100);
        if (n < 0) {
            if (errno == EINTR) continue;
            break;
        }

        std::vector<DeferredUnmap> deferred;

        for (int i = 0; i < n; i++) {
            int fd = events[i].data.fd;

            if (fd == listen_fd) {
                int client_fd = accept(listen_fd, nullptr, nullptr);
                if (client_fd >= 0) {
                    ev.events = EPOLLIN;
                    ev.data.fd = client_fd;
                    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &ev);
                }
            } else if (events[i].events & (EPOLLHUP | EPOLLERR)) {
                {
                    std::lock_guard<std::mutex> lock(g_clients_mu);
                    auto it = g_clients.find(fd);
                    if (it != g_clients.end()) {
                        auto& info = it->second;
                        vgpu::ShmChannel* ch = info.channel;
                        handleClientDeath(info);
                        if (ch) {
                            deferred.push_back({info.client_id, ch});
                        }
                        g_clients.erase(it);
                    }
                }
                epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, nullptr);
                close(fd);
            } else if (events[i].events & EPOLLIN) {
                handleHandshake(fd);
                epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, nullptr);
                ev.events = EPOLLHUP | EPOLLERR;
                ev.data.fd = fd;
                epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &ev);
            }
        }

        for (auto& du : deferred) {
            destroyShm(du.client_id, du.channel);
        }
    }

    {
        std::vector<DeferredUnmap> cleanup;
        {
            std::lock_guard<std::mutex> lock(g_clients_mu);
            for (auto& [fd, info] : g_clients) {
                epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, nullptr);
                g_scheduler->unregisterClient(info.client_id);
                if (info.channel) {
                    info.channel->dying.store(1, std::memory_order_release);
                    cleanup.push_back({info.client_id, info.channel});
                    info.channel = nullptr;
                }
                close(fd);
            }
            g_clients.clear();
        }
        for (auto& du : cleanup) {
            destroyShm(du.client_id, du.channel);
        }
    }

    poller.join();
    close(epoll_fd);
    close(listen_fd);
    unlink(path.c_str());
    return 0;
}
