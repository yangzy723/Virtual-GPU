# vGPU 设计文档

## 1. 目标与非目标

vGPU 在 CUDA Driver API 调用前增加一个小型 admission-control 层，支持两种部署：单进程本地调度，以及由 daemon 统一协调多进程。执行面始终是客户端与真实 `libcuda.so.1`。

本项目不试图实现：GPU 指令级抢占、显存地址空间隔离、MIG/MPS 替代品、容器安全边界，或未经覆盖的所有 CUDA API 的透明虚拟化。

## 2. 组件边界

```text
intercept.cpp
  ├─ real CUDA symbol loader / ABI wrappers
  ├─ allocation metadata
  └─ asynchronous completion observer
             │ request / report
             ▼
scheduler_backend.cpp
  ├─ LocalScheduler      (in-process)
  └─ DaemonChannel       (UDS registration + SHM protocol)
                              │
                              ▼
                    gpu_scheduler daemon
```

`intercept.cpp` 不再包含 socket 或 SHM 状态机。`DaemonChannel` 独占通信状态；`LocalScheduler` 不包含 CUDA 和 IPC 依赖，因此可以独立单元测试。

## 3. 后端选择

shim 第一次进入调度路径时读取配置并固定进程生命周期内的 mode：

1. `VGPU_SCHEDULER_MODE=local|daemon`。
2. 未设置时，兼容检查 `VGPU_LOCAL_MODE=1`。
3. 都未设置时使用 `daemon`。
4. 非法值打印 warning 并回退 `daemon`。

固定 mode 避免运行中切换导致已有 active 计数被另一个后端接管。

## 4. 本地调度

`LocalScheduler` 持有一个 mutex、condition variable 和三类状态：

- `allocated_bytes`：不含固定 context 预扣的动态显存记账。
- `active_kernels`：已批准且尚未完成的 kernel 数。
- `active_memcpy`：已批准且尚未完成的 memcpy/memset 数。

行为：

- allocation：在 `context_overhead + allocated + request` 不超过 limit 时批准；超限返回 OOM。
- kernel：达到上限时在 condition variable 等待，完成上报后唤醒。
- memcpy：上限为 0 时无限制，否则采用相同的等待语义。
- FREE/COMPLETE：计数饱和减法，保证状态不为负。

local 后端没有 socket、SHM 或跨进程共享状态。其目标是测量/使用拦截层与进程内策略的成本，而不是提供多进程公平性。

## 5. Daemon 通信与调度

### 5.1 注册慢路径

1. shim 连接 `GPU_SCHEDULER_SOCKET` 指定的 Unix domain socket。
2. 发送 `HELLO + PID`。
3. daemon 创建 `/vgpu_PID` POSIX SHM，注册客户端并返回名称。
4. shim mmap 通道并保留 UDS fd；daemon 通过 fd 的 HUP/断开检测客户端死亡。

### 5.2 请求快路径

每个客户端只有一个 `ShmChannel`，shim 内部 mutex 将并发请求串行化：

```text
IDLE → client writes op/value/device → PENDING
PENDING → daemon decision → APPROVED | REJECTED
APPROVED/REJECTED → client consumes → IDLE
```

FREE/KERNEL_COMPLETE/MEMCPY_COMPLETE 是报告：daemon 更新状态后直接把通道恢复为 IDLE。当前 daemon 在并发槽位满时返回 REJECTED，并未实现等待队列。

### 5.3 fail-open

socket、握手、mmap 或 SHM 等待失败时，`DaemonChannel` 只打印一次明确 warning，并允许 CUDA 调用继续执行。该行为是可用性策略，不是隔离保证。local 模式从不进入此路径。

## 6. 协议

`include/vgpu/protocol.h` 定义固定大小、cache-line 对齐的 `ShmChannel`：

- `state`：IDLE/PENDING/APPROVED/REJECTED。
- `op`：ALLOC/FREE/KERNEL/MEMCPY request 或 completion。
- `value`：分配或传输字节数。
- `result`、`device`、`client_id`、`dying`：结果和生命周期元数据。

发布请求和读取决策使用 release/acquire 顺序。协议目前是单槽，不支持同一客户端多个并行未决请求，因此客户端必须串行写通道。

## 7. CUDA 拦截与完成语义

### 7.1 符号路由

LD_PRELOAD 覆盖直接符号调用；shim 也接管 `dlsym` 和 `cuGetProcAddress*` 的目标符号路由。未调度的符号转给真实 Driver，以减小兼容面。

| 类别 | 代表入口 | admission | completion |
|---|---|---|---|
| 显存 | `cuMemAlloc*`, `cuMemFree*` | 配额 | FREE 成功后上报；async free 优先 event |
| Kernel | `cuLaunchKernel`, `cuLaunchKernelEx*` | kernel 槽位 | 常用 async 入口优先 event |
| Copy | 常用 `cuMemcpy*`, peer, 2D/3D, batch v2 | memcpy 槽位 | async 优先 event，同步按返回 |
| Set | 常用 `cuMemsetD*`, `cuMemsetD2D*` | memcpy 槽位 | async 优先 event，同步按返回 |
| 其他 | device/context/module/多数 stream/event | 无 | 透传 |

### 7.2 异步完成观察

成功入队后，shim 在目标 stream 记录 `CU_EVENT_DISABLE_TIMING` 私有 event。后台线程周期性调用真实 `cuEventQuery`；完成后销毁 event 并向当前后端 report。若 event API 不可用或记录失败，立即 report 以避免后端计数永久泄漏。

后端对象按进程生命周期保留，避免 ELF teardown 时完成线程上报到已析构对象。

## 8. 配额与一致性

context overhead 是调度记账值，不等同于一次真实 CUDA allocation。显存 allocation 的指针和大小由 shim 记录，成功 free 后用原大小上报。daemon 客户端断开时删除整份客户端状态；真实 GPU 资源由 Driver/OS 的进程清理负责。

关键不变量：

1. SHM 单槽在新请求前必须为 IDLE。
2. allocation 审批不能溢出或超过配置 limit。
3. active kernel/memcpy 计数不得为负。
4. local 模式不得创建 socket。
5. daemon 传输故障只能 warning 一次并保持 fail-open。
6. completion 路径失败时必须回退上报，避免槽位永久占用。

## 9. 配置语义

公共资源配置由两种后端复用：

- `GPU_SCHEDULER_MEM_LIMIT_MB`
- `GPU_SCHEDULER_CONTEXT_OVERHEAD_MB`
- `GPU_SCHEDULER_MAX_KERNELS`
- `GPU_SCHEDULER_MAX_MEMCPY`
- `GPU_SCHEDULER_VERBOSE`

通信专用：`GPU_SCHEDULER_SOCKET`、`GPU_SCHEDULER_WAIT_ITERS`、`GPU_SCHEDULER_POLL_US`。完成观察专用：`GPU_SCHEDULER_COMPLETION_POLL_US`。全部字段及默认值见 README。

## 10. 测试策略

- C++ 单元测试：local allocation 边界、阻塞/唤醒、completion 后计数归零。
- 无 Torch 集成测试：daemon 启停、动态符号路由、fail-open、local OOM、Driver memset、并发客户端。
- 可选 PyTorch 测试：daemon、local 和 fail-open 三种 GEMM 路径，以及 kernel/memcpy daemon 日志。
- 性能测试：同一 GEMM 参数下无 shim 与启用 shim 的端到端 P95 A/B。

## 11. 评估数据边界

GreenCtx lifecycle、池化、policy 遍历以及 MPS/GreenCtx resident-memory 都是独立 workload 实验，已迁移到配套 CoGPU 仓库的 `eval/overhead/`。历史四组数据包含这些 workload 分量和旧 `VGPU_LOCAL_MODE=1` passthrough，不能作为纯拦截开销。

本仓库的纯拦截评估只允许在相同 GEMM、相同软件栈和相同同步边界下做无 shim/有 shim A/B；当前 LocalScheduler 需要据此重新测量，不能由历史混合数据推导。

## 12. 后续工程方向

1. daemon 侧引入等待队列和明确的超时/取消协议，代替瞬时拒绝。
2. 按 device 分桶计数，并记录 allocation 所属 device/context。
3. 覆盖 CUDA Graph、更多 batch/prefetch 和 `cuLaunchKernelExC` 的 event 完成路径。
4. 增加 daemon 重启、客户端崩溃和长 kernel 的故障注入测试。
5. 若要 fail-closed，新增显式配置和可观测的拒绝指标，不能复用当前 fail-open 语义。
