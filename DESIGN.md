# 设计文档

## 1. 目标

让多个进程共享一块 GPU。Daemon 作为门卫（gatekeeper），追踪资源使用并决定每个进程何时可以执行。所有 GPU 操作在本地真实驱动上执行。

## 2. 核心洞察

CUDA Runtime API (`libcudart`) 是 Driver API (`libcuda`) 的薄封装。PyTorch 调用 `cudaMalloc` 时，libcudart 内部调用 `cuMemAlloc`；调用 `cudaLaunchKernel` 时，内部调用 `cuLaunchKernel`。

通过 `LD_PRELOAD` 拦截 `libcuda.so`，可以覆盖 CUDA Driver API 的主路径调用（包括 PyTorch、cuBLAS、cuDNN 常见路径）。这意味着：

- **无需拦截 libcudart** — 避免 SONAME 冲突和转发桩
- **通常无需 hook dlopen** — 真实 libcudart 正常加载
- **覆盖边界清晰** — 以 `cuGetProcAddress` 和显式导出符号为主，少数非常规解析路径可能需要额外 hook 策略

## 3. 系统架构

```
┌─────────────┐    ┌─────────────┐
│   进程 A     │    │   进程 B     │
│  (PyTorch)  │    │  (cuBLAS)   │
└──────┬──────┘    └──────┬──────┘
       │                  │
       ▼                  ▼
┌──────────────────────────────────┐
│  LD_PRELOAD: libcuda.so (shim)   │
│                                  │
│  重操作 → 共享内存 → daemon 审批  │
│  轻操作 → 直接调用真实驱动        │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────┐     ┌──────────────────┐
│  gpu_scheduler       │     │  真实 libcuda.so.1│
│  (daemon, 不碰 GPU)  │     │  (NVIDIA 驱动)    │
│                      │     │                   │
│  轮询共享内存         │     │  执行 GPU 操作     │
│  做审批决策           │     │                   │
└──────────────────────┘     └──────────────────┘
```

## 4. 调度模型

### 4.1 门卫模式

Daemon 从不执行 GPU 操作，只做调度决策：

1. **客户端请求** — shim 写入共享内存，设置 state 为 PENDING
2. **Daemon 决策** — 轮询共享内存，设置 state 为 APPROVED 或 REJECTED
3. **客户端执行** — shim 读取响应，调用真实 CUDA 函数
4. **完成报告** — shim 发送 FREE 或 KERNEL_COMPLETE，daemon 重置 state 为 IDLE

### 4.2 通信：共享内存

每个客户端有专属的共享内存通道（daemon 在握手时创建）。

```
ShmChannel (128 字节, cache-line 对齐):
  atomic<uint32_t> state      — IDLE/PENDING/APPROVED/REJECTED
  atomic<uint32_t> op         — ALLOC_REQUEST/FREE/KERNEL_REQUEST/KERNEL_COMPLETE
  atomic<uint64_t> value      — ALLOC 字节数
  atomic<int32_t>  result     — 0=成功, 负数=错误
  atomic<uint32_t> device     — GPU 设备号
  atomic<uint64_t> client_id  — 客户端 PID
  atomic<uint32_t> dying      — 客户端断开标记
```

**快路径**：shim 写请求 → 设置 PENDING → 自旋等待 → 读响应。延迟：微秒级。

**慢路径**：daemon 每 100μs 轮询所有通道。通过 epoll 检测 UDS 连接断开。

轮询间隔可通过 `GPU_SCHEDULER_POLL_US` 调整（默认 100μs）。

### 4.3 握手协议

1. Shim 通过 UDS 连接 daemon（`/tmp/vgpu_control.sock`）
2. 发送 HELLO + PID
3. Daemon 创建共享内存（`/dev/shm/vgpu_PID`），注册客户端
4. Daemon 返回共享内存名称
5. Shim mmap 共享内存
6. 后续所有通信走共享内存（不再有 UDS 请求）

### 4.4 死亡检测

- Daemon 保持每个客户端的 UDS 连接
- 通过 epoll 检测 EPOLLHUP（客户端进程死亡）
- 检测到死亡后：注销客户端、取消链接共享内存、释放配额
- GPU 显存由 OS 自动回收

### 4.5 延迟释放（Deferred Unmap）

多线程环境下，poll 线程和 epoll 线程可能同时访问同一通道。解决方案：

1. Epoll 线程检测到客户端死亡 → 设置 `dying` 标记 → 将通道指针置空
2. Poll 线程看到 `dying` 标记 → 停止使用该通道
3. Epoll 线程在释放锁后执行 munmap

这避免了 use-after-free，无需在持锁期间执行系统调用。

### 4.6 资源追踪

Daemon 追踪每客户端状态：

- **显存用量** — 注册时预扣 800MB context 开销
- **活跃 kernel 数** — KERNEL_REQUEST 时加一，KERNEL_COMPLETE 时减一

显存配额强制执行：如果客户端超限，daemon 拒绝请求，shim 返回 `CUDA_ERROR_OUT_OF_MEMORY`。

## 5. 符号拦截策略

`cuGetProcAddress` 是动态符号解析的入口。拦截策略：

**需要调度逻辑的符号**（从 shim 返回）：
- `cuMemAlloc` / `cuMemAlloc_v2` — 显存分配
- `cuMemAllocAsync` / `cuMemAllocFromPoolAsync` — 异步显存分配
- `cuMemFree` / `cuMemFree_v2` — 显存释放
- `cuMemFreeAsync` — 异步显存释放
- `cuLaunchKernel` — kernel 启动
- `cuLaunchKernelEx` / `cuLaunchKernelExC` — 新版 kernel 启动接口
- `cuStreamCreate` / `cuStreamDestroy` / `cuStreamSynchronize` / `cuStreamQuery` — stream 操作
- `cuGetProcAddress` / `cuGetProcAddress_v2` — 自身

**其余符号**（默认从真实驱动返回）：
- Context 操作（`cuCtxCreate`、`cuCtxSetCurrent` 等）
- Device 操作（`cuDeviceGet`、`cuDeviceGetAttribute` 等）
- Module 操作（`cuModuleLoad`、`cuModuleGetFunction` 等）
- Event 操作（`cuEventCreate`、`cuEventRecord` 等）
- Memcpy/Memset 操作

LD_PRELOAD 确保直接调用走 shim 转发，`cuGetProcAddress` 确保动态解析走正确路径。

### 5.1 接口覆盖矩阵

| 接口类别 | 代表 API | 处理方式 | 是否走 daemon | 备注 |
|---|---|---|---|---|
| 显存分配 | `cuMemAlloc*` | shim 拦截 | 是 | 受显存配额限制 |
| 显存释放 | `cuMemFree*` | shim 拦截 | 是（报告） | 释放后更新统计 |
| Kernel 启动 | `cuLaunchKernel*` | shim 拦截 | 是 | 当前完成语义为 launch-return |
| Memcpy | `cuMemcpy*` | shim 拦截 | 可选 | `GPU_SCHEDULER_CONTROL_MEMCPY=1` 启用 |
| Stream | `cuStream*` | shim 拦截 | 否 | 直接透传真实驱动 |
| Event | `cuEvent*` | shim 拦截 | 否 | 直接透传真实驱动 |
| 其余 Driver API | 如 `cuDevice*`、`cuModule*` | 转发真实驱动 | 否 | 维持兼容性优先 |

## 6. cuGetExportTable

默认转发到真实驱动。cuBLAS/cuDNN 需要真实的 export table 才能正常工作。

可通过环境变量 `VGPU_CU_EXPORT_TABLE_SUCCESS_NULL=1` 返回空（调试用）。

## 7. 优雅降级

如果 daemon socket 不可达：
- `connectToDaemon()` 返回 false
- `schedRequest()` 返回 true（批准）
- 所有操作直接在真实驱动上执行

系统在有/无 daemon 时行为完全一致——daemon 只在存在时增加调度。

## 8. 配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GPU_SCHEDULER_SOCKET` | `/tmp/vgpu_control.sock` | daemon socket 路径 |
| `GPU_SCHEDULER_MEM_LIMIT_MB` | 无限制 | 每客户端显存上限 |
| `GPU_SCHEDULER_CONTEXT_OVERHEAD_MB` | 800 | 预扣 context 开销 |
| `GPU_SCHEDULER_MAX_KERNELS` | 1 | 全局并发 kernel 上限 |
| `GPU_SCHEDULER_MAX_MEMCPY` | 0 | 全局并发 memcpy 上限（0=不限制） |
| `GPU_SCHEDULER_POLL_US` | 100 | daemon 轮询间隔（μs） |
| `GPU_SCHEDULER_VERBOSE` | 0 | 调度日志 |
| `GPU_SCHEDULER_CONTROL_MEMCPY` | 0 | 客户端是否让 memcpy 路径走 daemon |

## 9. 已知局限

### 9.1 无抢占式调度

Daemon 只能在 kernel 启动前阻塞客户端。Kernel 一旦开始执行，daemon 无法中断。

### 9.2 Memcpy 管控为可选接口

默认情况下，memcpy 仍可直接透传到真实驱动，以保持低开销。

当客户端设置 `GPU_SCHEDULER_CONTROL_MEMCPY=1` 后，`cuMemcpyHtoD/DtoH/DtoD`（含 Async）会经过 daemon 审批，并可通过 `GPU_SCHEDULER_MAX_MEMCPY` 限制全局并发拷贝数。

### 9.3 单 GPU

不支持多 GPU 调度或隔离。

## 10. 设计亮点

### 10.1 控制面与执行面分离

- **控制面（daemon）**：只做审批、计数和限流，不执行 GPU API。
- **执行面（shim + 真实驱动）**：真正发起 CUDA 调用，保证框架兼容性。

这使得调度策略和 GPU 执行机制解耦，便于后续迭代策略（配额、优先级、抢占提示）而不破坏执行路径。

### 10.2 双通路通信（握手慢路径 + 数据快路径）

- **慢路径**：UDS 负责连接建立、生命周期管理、故障检测。
- **快路径**：共享内存单生产者/单消费者协议，原子状态机完成审批交互。

该设计兼顾了可管理性（UDS）与低时延（SHM）。

### 10.3 失败优雅降级（Fail-Open）

daemon 不可达时 shim 透传真实驱动，业务仍可运行。该模式优先保障可用性，适合线上灰度与增量部署。

### 10.4 面向扩展的操作语义

`SchedOp` 抽象了调度事件（ALLOC/KERNEL/MEMCPY），后续可自然扩展到 memcpy 带宽、图执行（CUDA Graph）和多设备策略。

## 11. 正确性不变量

以下不变量用于约束实现和测试：

1. **状态机安全性**：请求完成后通道必须回到 `IDLE`，否则后续请求不得覆盖旧状态。
2. **内存配额单调性**：审批前不得突破配额上限；释放路径不得使统计出现负值。
3. **并发计数非负性**：`active_kernels`、`active_memcpy` 在任何时刻均应满足 $\ge 0$。
4. **客户端死亡可回收性**：检测到断连后，必须清理客户端状态与共享内存映射。
5. **降级可用性**：daemon 不可用时，核心 CUDA 路径仍应可运行。

建议后续将这些不变量映射到自动化测试命名中，形成“性质-用例”一一对应关系。

## 12. 评估方法（工程 + 学术）

建议将实验拆分为三个维度：

1. **性能开销**
- 空载开销：只加载 shim、不启调度的基线延迟。
- 调度开销：开启 daemon 后 `cuMemAlloc/cuLaunchKernel/cuMemcpy` 的新增时延。

2. **隔离与公平性**
- 多进程竞争下每个客户端的吞吐和尾延迟（P95/P99）。
- 不同配额策略下的资源占用公平度（如 Jain 指数）。

3. **鲁棒性**
- daemon 异常退出/重启后的任务行为。
- 客户端崩溃后的状态回收时延。

可将结果统一报告为：吞吐、时延分位、拒绝率、恢复时间。

## 13. 优化路线图

### 13.1 短期（可快速落地）

1. 将 kernel 完成语义从“launch 返回即完成”升级为“基于 event/stream 观测完成”。
2. 扩展 memcpy 管控到 2D/3D/Peer/Batch 系列接口。
3. 增加更细粒度日志开关（请求采样率、按 op 分类）。

### 13.2 中期（增强策略能力）

1. 引入 token-bucket 带宽调度，替代单纯并发数限制。
2. 支持按客户端优先级和权重进行加权公平调度。
3. 增加策略插件化接口，支持不同策略热切换。

### 13.3 长期（体系能力）

1. 多 GPU 拓扑感知调度（NUMA/NVLink 约束）。
2. 与容器/runtime 集成（cgroup + device plugin）实现租户级隔离。
3. 形成可复现实验基准套件，支撑论文与工程发布。
