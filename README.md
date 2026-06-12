# GPU Scheduler

单 GPU 多进程调度层。通过 `LD_PRELOAD` 拦截 CUDA Driver API，协调多个进程共享一块 GPU。

## 架构

```
进程 A ──拦截──┐
               ├──▶ 同一个 daemon（统一审批）
进程 B ──拦截──┘                 │
                                ├─批准──▶ 客户端调用真实 GPU
                                └─拒绝──▶ 客户端阻塞等待
```

**核心洞察**：CUDA Runtime API (`libcudart`) 内部调用 Driver API (`libcuda`)。优先拦截 `libcuda.so` 即可覆盖绝大多数框架主路径（含 PyTorch 常见路径）。对 cuBLAS/cuDNN 等库，未拦截符号保持透传真实驱动。

**调度模型**：daemon 只做审批决策，不执行任何 GPU 操作。所有 GPU 计算在本地真实驱动上执行。

**通信机制**：UDS 握手建立连接，之后请求走共享内存原子状态机。

**优雅降级**：daemon 未运行时，重操作会 fail-open 到真实驱动，保证可用性优先（仍保留 shim 自身的轻微拦截开销）。

## 版本

- 当前版本：`0.3.2`（见 `VERSION`）
- 变更记录：`CHANGELOG.md`

## 快速开始

### 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 启动 daemon

```bash
./build/gpu_scheduler &
```

### 运行带调度的程序

```bash
LD_PRELOAD=build/libcuda.so python your_script.py
```

### 两个进程共享一块 GPU

```bash
# 终端 1
LD_PRELOAD=build/libcuda.so python -c "
import torch
a = torch.randn(1000, 1000, device='cuda')
b = torch.matmul(a, a)
print('P1 ok', b.shape)"

# 终端 2（同时运行）
LD_PRELOAD=build/libcuda.so python -c "
import torch
a = torch.randn(1000, 1000, device='cuda')
b = torch.matmul(a, a)
print('P2 ok', b.shape)"
```

### 一键演示（自动启动 daemon + 跑 matmul + 验证拦截）

```bash
./scripts/run_matmul_demo.sh
```

脚本会输出两段关键信息：
1. 客户端侧 `daemon request/approved` 与 `matmul ok`
2. daemon 侧 `ALLOC_REQUEST/KERNEL_REQUEST/KERNEL_COMPLETE` 记录

如果日志里缺少 matmul 成功标记或 kernel 调度记录，脚本会非零退出。可用 `PYTHON_BIN=/path/to/python` 指定带 PyTorch 的解释器。

## 配置文件（推荐）

项目支持统一配置文件，减少大量环境变量配置。

默认查找顺序：
1. `GPU_SCHEDULER_CONFIG` 指定的路径
2. `./vgpu.conf`
3. `./config/vgpu.conf`
4. `/etc/vgpu.conf`

示例配置文件：`config/vgpu.conf.example`

```bash
cp config/vgpu.conf.example config/vgpu.conf
./build/gpu_scheduler &
LD_PRELOAD=build/libcuda.so python your_script.py
```

环境变量仍可用于临时覆盖配置文件中的同名项。

## 环境变量（可选覆盖）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GPU_SCHEDULER_SOCKET` | `/tmp/vgpu_control.sock` | daemon socket 路径 |
| `GPU_SCHEDULER_MEM_LIMIT_MB` | 无限制 | 每客户端显存上限（MB） |
| `GPU_SCHEDULER_CONTEXT_OVERHEAD_MB` | 800 | 每客户端预扣 context 开销（MB） |
| `GPU_SCHEDULER_MAX_KERNELS` | 1 | 全局最大并发 kernel 数 |
| `GPU_SCHEDULER_MAX_MEMCPY` | 0 | 全局最大并发 memcpy 数（0=不限制） |
| `GPU_SCHEDULER_POLL_US` | 100 | daemon 轮询共享内存间隔（微秒） |
| `GPU_SCHEDULER_COMPLETION_POLL_US` | 100 | shim 轮询私有 CUDA event 完成状态的间隔（微秒） |
| `GPU_SCHEDULER_VERBOSE` | 0 | 输出调度日志 |
| `GPU_SCHEDULER_WAIT_ITERS` | 200000 | shim 等待 daemon 响应的最大自旋迭代数 |
| `GPU_SCHEDULER_CONFIG` | 无 | 指定配置文件路径 |
| `VGPU_TRACE` | 1 | shim trace 输出开关 |

## 组件

| 组件 | 源码 | 说明 |
|---|---|---|
| `gpu_scheduler` | `src/daemon/` | 调度 daemon — 只做审批，不碰 GPU |
| `libcuda.so` | `src/shim/intercept.cpp` | Driver API shim（LD_PRELOAD） |
| `scripts/run_matmul_demo.sh` | `scripts/` | 一键演示脚本（构建+运行+日志摘要） |
| `config/vgpu.conf.example` | `config/` | 配置文件模板 |

## 目录结构

```text
vGPU/
    include/vgpu/         # 协议与最小 CUDA ABI 类型
    src/daemon/           # daemon 调度与生命周期管理
    src/shim/             # libcuda.so 拦截与转发
    scripts/              # 一键运行与调试脚本
    tests/                # unittest 回归
```

## 工作原理

1. Shim 库通过 `LD_PRELOAD` 优先接管 CUDA Driver API 符号，并覆写 `dlsym`/`cuGetProcAddress*` 以路由动态解析
2. **重操作**（`cuMemAlloc*`、`cuLaunchKernel*`、`cuMemcpy*`、`cuMemset*`）：shim 写入共享内存，原子自旋等待 daemon 决策，然后调用真实 CUDA 函数
3. **轻操作**（`cuDeviceGetAttribute`、`cuStreamCreate` 等）：shim 直接调用真实 CUDA 函数
4. **完成上报**：同步路径按 API-return 上报；`cuLaunchKernel`、`cuLaunchKernelEx`、Async memcpy/memset 与 `cuMemFreeAsync` 成功入队后，shim 记录私有 CUDA event，后台线程通过 `cuEventQuery` 观测到 stream 完成后再上报
5. **Daemon** 通过轮询共享内存追踪每客户端的显存用量、活跃 kernel 数和活跃 memcpy 数
6. Daemon 不可达时，shim **降级**为直接调用真实驱动，并向 stderr 打印一次明确的 fail-open warning

## 符号拦截策略

只拦截需要调度逻辑的符号：

- `cuMemAlloc` / `cuMemAllocAsync` / `cuMemAllocFromPoolAsync` / `cuMemFree` / `cuMemFreeAsync` — 显存分配/释放
- `cuLaunchKernel` / `cuLaunchKernelEx` / `cuLaunchKernelExC` — kernel 启动（`ExC` 当前完成上报仍按 API-return）
- `cuMemcpy*`、`cuMemcpyPeer*`、`cuMemcpy2D*`、`cuMemcpy3D*`（含 Async）— 主存/显存拷贝（始终走 daemon 调度）
- `cuMemcpyBatchAsync_v2` — CUDA 13 `_v2` pointer batch 拷贝（旧 `cuMemcpyBatchAsync` 签名和 `cuMemcpy3DBatchAsync*` 暂不拦截）
- `cuMemsetD*` / `cuMemsetD2D*`（含 Async）— 设备显存写入（复用 memcpy 并发预算）
- `cuStreamCreate` / `cuStreamDestroy` / `cuStreamSynchronize` / `cuStreamQuery` — stream 操作（当前仅透传，便于后续扩展）
- `cuGetProcAddress` — 动态符号解析路由
- `dlsym` — 动态符号查询入口（用于将 `cuGetProcAddress*` 查询导向 shim）

默认不 hook `dlopen`（兼容性优先）；当框架通过 `dlsym` + `cuGetProcAddress*` 解析 CUDA 符号时，仍可被当前路径覆盖。

其余符号（context、device、module、event 与未覆盖 Driver API）直接透传到真实驱动。shim 内部会创建私有 event 做完成观测；用户显式调用的 event API 仍是透传。

## 测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python -m unittest -v tests/test_vgpu.py
```

如需指定运行 PyTorch 客户端脚本的解释器，可设置 `VGPU_TEST_PYTHON=/path/to/python`。

测试项：
1. daemon 启停
2. `cuGetProcAddress` 对被调度符号的路由
3. fail-open warning
4. 带/不带 daemon 的 torch.matmul
5. kernel/memcpy/memset 调度路径
6. 多客户端并发

## 已知局限

- **无抢占式调度**：daemon 只能在 kernel 启动前阻塞客户端，无法中断正在执行的 kernel
- **Memcpy/Memset 管控粒度有限**：当前覆盖常见 `cuMemcpy*`、`cuMemcpyPeer*`、2D/3D 拷贝（含 Async）、CUDA 13 `cuMemcpyBatchAsync_v2` 与常见 `cuMemsetD*`/`cuMemsetD2D*`；旧 Batch 签名、3D Batch、prefetch/discard 等仍未纳入管控
- **Fail-open 会显式提示**：daemon 不可达、握手失败或调度等待超时时，shim 会在 stderr 打印一次 `[vGPU warning] scheduler unavailable, fail-open passthrough...`，随后透传真实驱动
- **完成语义仍有边界**：`cuLaunchKernel`、`cuLaunchKernelEx`、Async memcpy/memset 和 `cuMemFreeAsync` 优先基于私有 event/stream 完成上报；event 创建或记录失败时回退到 API-return。`cuLaunchKernelExC`、同步 memcpy/memset、同步显存释放等仍按 API-return/调用返回语义上报
- **单 GPU**：不支持多 GPU 调度或隔离
