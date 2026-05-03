# GPU Scheduler

单 GPU 多进程调度层。通过 `LD_PRELOAD` 拦截 CUDA Driver API，协调多个进程共享一块 GPU。

## 架构

```
进程 A ──拦截──▶ 询问 daemon ──批准──▶ 调用真实 GPU
                          │
                          └─拒绝──▶ 阻塞等待

进程 B ──拦截──▶ 询问 daemon ──批准──▶ 调用真实 GPU
                          │
                          └─拒绝──▶ 阻塞等待
```

**核心洞察**：CUDA Runtime API (`libcudart`) 内部调用 Driver API (`libcuda`)。只需拦截 `libcuda.so`，即可捕获所有 CUDA 操作（包括 PyTorch、cuBLAS、cuDNN），无需拦截 `libcudart`。

**调度模型**：daemon 只做审批决策，不执行任何 GPU 操作。所有 GPU 计算在本地真实驱动上执行。

**通信机制**：UDS 握手建立连接，之后请求走共享内存原子状态机。

**优雅降级**：daemon 未运行时，所有操作直接透传到真实驱动，行为完全一致。

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

### 一键演示（自动启动 daemon + 跑 matmul + 输出摘要）

```bash
./scripts/run_matmul_demo.sh
```

脚本会输出两段关键信息：
1. 客户端侧 `daemon request/approved` 与 `matmul ok`
2. daemon 侧 `ALLOC_REQUEST/KERNEL_REQUEST` 审批记录

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
| `GPU_SCHEDULER_VERBOSE` | 0 | 输出调度日志 |
| `GPU_SCHEDULER_CONTROL_MEMCPY` | 0 | 客户端是否让 memcpy 路径走 daemon 调度 |
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
    tests/                # pytest 回归
```

## 工作原理

1. Shim 库通过 `LD_PRELOAD` 导出 CUDA Driver API 符号
2. **重操作**（`cuMemAlloc*`、`cuLaunchKernel*`）：shim 写入共享内存，原子自旋等待 daemon 决策，然后调用真实 CUDA 函数
3. **轻操作**（`cuDeviceGetAttribute`、`cuStreamCreate` 等）：shim 直接调用真实 CUDA 函数
4. **Daemon** 通过轮询共享内存追踪每客户端的显存用量和活跃 kernel 数
5. Daemon 不可达时，shim **降级**为直接调用真实驱动

## 符号拦截策略

只拦截需要调度逻辑的符号：

- `cuMemAlloc` / `cuMemAllocAsync` / `cuMemAllocFromPoolAsync` / `cuMemFree` / `cuMemFreeAsync` — 显存分配/释放
- `cuLaunchKernel` / `cuLaunchKernelEx` / `cuLaunchKernelExC` — kernel 启动
- `cuMemcpyHtoD/DtoH/DtoD`（含 Async）— 当 `GPU_SCHEDULER_CONTROL_MEMCPY=1` 时走 daemon 调度
- `cuStreamCreate` / `cuStreamDestroy` / `cuStreamSynchronize` / `cuStreamQuery` — stream 操作（用于 kernel 依赖追踪）
- `cuGetProcAddress` — 动态符号解析路由

其余符号（context、device、module、event、memcpy 等）直接透传到真实驱动。

## 测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python -m pytest tests/test_vgpu.py -v
```

测试项：
1. daemon 启停
2. 带 daemon 的 torch.matmul
3. 无 daemon 的优雅降级
4. 多客户端并发

## 已知局限

- **无抢占式调度**：daemon 只能在 kernel 启动前阻塞客户端，无法中断正在执行的 kernel
- **Memcpy 未管控**：内存拷贝操作直接透传，大块传输可能造成竞争
- **单 GPU**：不支持多 GPU 调度或隔离
