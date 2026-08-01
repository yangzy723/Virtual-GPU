# vGPU CUDA Driver Scheduling Shim

一个面向单机单 GPU 实验的 CUDA Driver API 调度层。项目通过 `LD_PRELOAD` 接管选定的 `libcuda` 符号，在调用真实 NVIDIA Driver 前执行资源审批；GPU 工作始终由客户端进程提交，daemon 不执行 CUDA API。

> 当前状态：工程/研究原型。它提供调用级 admission control 和计数，不是硬件虚拟化、显存安全隔离或可抢占 GPU 调度器。

## 能力与模式

| 模式 | 配置 | 调度范围 | 通信 | 资源满时行为 |
|---|---|---|---|---|
| daemon | `VGPU_SCHEDULER_MODE=daemon`（默认） | 多进程统一计数与审批 | UDS 注册 + 每客户端 SHM | 当前协议返回拒绝；shim 映射为对应 CUDA 错误 |
| local | `VGPU_SCHEDULER_MODE=local` | 当前进程内各线程 | 无 socket/SHM | kernel/memcpy 等待本地槽位；显存超配额返回 OOM |
| daemon fail-open | daemon 不可达或响应超时 | 不再调度 | 连接尝试失败 | stderr 警告一次后透传真实驱动 |

`VGPU_LOCAL_MODE=1` 仅作为旧配置的兼容别名，等价于 `VGPU_SCHEDULER_MODE=local`。新部署应使用显式 mode。

## 架构

```text
CUDA application / PyTorch / cuBLAS
                 │ Driver API
                 ▼
       LD_PRELOAD: build/libcuda.so
                 │
       ┌─────────┴─────────┐
       │ scheduler backend │
       ├───────────────────┤
       │ local             │ 进程内 mutex/CV、配额与并发计数
       │ daemon            │ UDS 握手、SHM 原子状态机、fail-open
       └─────────┬─────────┘
                 │ approved
                 ▼
          real libcuda.so.1 ──▶ GPU
```

代码边界：

- `src/shim/intercept.cpp`：CUDA ABI 包装、真实符号转发、异步完成事件。
- `src/shim/scheduler_backend.*`：mode 解析与统一 request/report 入口。
- `src/shim/local_scheduler.*`：无通信的进程内 admission scheduler。
- `src/shim/daemon_channel.*`：UDS/SHM 客户端协议和 fail-open。
- `src/daemon/`：多客户端生命周期、配额和并发审批。

更完整的不变量与完成语义见 [DESIGN.md](DESIGN.md)。

## 构建

要求 Linux、CMake 3.18+、C++17，以及运行时可见的 NVIDIA Driver。构建 shim 本身不依赖 CUDA Toolkit 头文件。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

产物：

- `build/libcuda.so`：供 `LD_PRELOAD` 使用的 shim。
- `build/gpu_scheduler`：daemon 后端。
- `build/local_scheduler_test`：不依赖 GPU 的本地调度单元测试。

## 快速使用

### 多进程 daemon 模式

```bash
cp config/vgpu.conf.example config/vgpu.conf
./build/gpu_scheduler &
SCHEDULER_PID=$!

VGPU_SCHEDULER_MODE=daemon \
LD_PRELOAD="$PWD/build/libcuda.so" \
python your_workload.py

kill "$SCHEDULER_PID"
```

也可运行带日志校验的一键示例：

```bash
PYTHON_BIN=/path/to/python-with-torch ./scripts/run_matmul_demo.sh
```

### 无 socket 的本地模式

```bash
VGPU_SCHEDULER_MODE=local \
GPU_SCHEDULER_MAX_KERNELS=8 \
LD_PRELOAD="$PWD/build/libcuda.so" \
python your_workload.py
```

local 模式不会连接 daemon。它只协调一个进程内的调用。

## 配置

配置查找顺序为 `GPU_SCHEDULER_CONFIG` 指定路径、`./vgpu.conf`、`./config/vgpu.conf`、`/etc/vgpu.conf`。环境变量覆盖文件中的同名键。

| 变量 | 默认值 | 适用范围 | 说明 |
|---|---:|---|---|
| `VGPU_SCHEDULER_MODE` | `daemon` | shim | `daemon` 或 `local` |
| `GPU_SCHEDULER_SOCKET` | `/tmp/vgpu_control.sock` | daemon | UDS 路径 |
| `GPU_SCHEDULER_MEM_LIMIT_MB` | `0` | 两种后端 | 显存记账上限；0 表示无限制 |
| `GPU_SCHEDULER_CONTEXT_OVERHEAD_MB` | `800` | 两种后端 | 注册/初始化时的记账预扣，不会实际分配显存 |
| `GPU_SCHEDULER_MAX_KERNELS` | `1` | 两种后端 | 活跃 kernel 上限 |
| `GPU_SCHEDULER_MAX_MEMCPY` | `0` | 两种后端 | 活跃 memcpy/memset 上限；0 表示无限制 |
| `GPU_SCHEDULER_POLL_US` | `100` | daemon | daemon SHM 轮询间隔 |
| `GPU_SCHEDULER_COMPLETION_POLL_US` | `100` | shim | 私有 CUDA event 查询间隔 |
| `GPU_SCHEDULER_WAIT_ITERS` | `200000` | daemon client | SHM 状态等待迭代预算 |
| `GPU_SCHEDULER_VERBOSE` | `0` | 两种后端 | 后端级日志 |
| `VGPU_TRACE` | `1` | shim | API/传输 trace；性能测试建议设为 0 |

示例见 `config/vgpu.conf.example`。

## 拦截范围与语义

当前调度集合包括：

- `cuMemAlloc*` / `cuMemFree*`：配额记账和释放上报。
- `cuLaunchKernel*`：kernel admission；常用异步入口通过私有 CUDA event 上报完成。
- 常用 `cuMemcpy*`、peer、2D/3D、`cuMemcpyBatchAsync_v2`：复用 memcpy 并发预算。
- 常用 `cuMemsetD*` / `cuMemsetD2D*`：复用 memcpy 并发预算。

未列入集合的 Driver API 通常透传真实驱动。同步接口在 API 返回时上报；异步接口优先在对应 stream 上记录私有 event，并由后台线程在 event 完成后上报。event 创建或记录失败时回退为 API-return 上报。具体覆盖矩阵见设计文档。

## 测试

```bash
ctest --test-dir build --output-on-failure
python tests/test_vgpu.py -v
```

完整 CUDA/PyTorch 用例需要安装 Torch 的解释器：

```bash
VGPU_TEST_PYTHON=/path/to/python-with-torch python tests/test_vgpu.py -v
```

测试覆盖本地配额与槽位等待、daemon 生命周期、SHM 调度路径、fail-open warning、动态符号路由、多客户端以及可用时的 PyTorch GEMM。

## 基准与结果

- `scripts/benchmark_gemm_overhead.py`：纯 GEMM A/B harness；用完全相同的参数分别在无 shim 和 `LD_PRELOAD` 下运行，输出 wall/GPU 时间与 P95。

GreenCtx 创建/销毁、池化、policy 扫描以及 MPS/GreenCtx resident-memory 都是独立 workload 实验，已迁移到配套 CoGPU 仓库的 `eval/overhead/`。此前四组混合场景数据也随脚本迁移，本仓库不再把它们标记为纯拦截开销。

## 已知边界

- 只在单 GPU 环境验证；daemon 的并发计数当前不是按 device 分桶。
- 无运行中 kernel 抢占，也不提供 SM、带宽或显存地址空间的强隔离。
- daemon 达到并发上限时目前拒绝请求，不维护跨客户端等待队列。
- fail-open 优先可用性，不适用于必须 fail-closed 的强隔离场景。
- CUDA Graph、旧 batch/3D batch、prefetch/discard 等路径尚未完整纳入。
- `cuLaunchKernelExC` 和部分同步/兼容入口仍按 API-return 完成语义记账。
