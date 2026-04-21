# Virtual-GPU

**Transparent CUDA interception system** — Run CUDA programs without a local GPU by forwarding all CUDA operations to a remote backend server.

## Why Virtual-GPU?

| Scenario | Solution |
|----------|----------|
| CPU-only development machine | Run CUDA code with Virtual-GPU shims + shared backend GPU |
| PyTorch training without local GPU | `LD_PRELOAD` our dlopen hooks, all ops forwarded transparently |
| Multi-tenant GPU sharing | Server handles concurrent clients with resource isolation |
| Container without GPU support | Mount Virtual-GPU, no GPU drivers needed in container |

## Quick Start (5 minutes)

### 1️⃣ Compile

```bash
cd /path/to/Virtual-GPU
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 2️⃣ Start Backend Server

```bash
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
./build/vgpu_server &
```

### 3️⃣ Run Your CUDA Application

**C/C++ with Runtime API:**
```bash
export LD_LIBRARY_PATH=/path/to/build:$LD_LIBRARY_PATH
./my_cuda_app
```

**PyTorch (with dlopen interception):**
```bash
export LD_PRELOAD=/path/to/build/libvgpu_preload_init.so:\
/path/to/build/libcuda.so:/path/to/build/libcudart.so

python my_pytorch_script.py
```

**Output example:**
```
[vGPU] dlopen interception: /usr/local/cuda/lib64/libcuda.so -> libcuda.so
[pytorch] CUDA available: True
[pytorch] Matrix multiplication completed ✓
```

## Key Features

- ✅ **Transparent**: No code changes needed
- ✅ **PyTorch Support**: dlopen hook intercepts all dynamic CUDA loading
- ✅ **Multi-app**: Multiple clients can connect to one server
- ✅ **Thread-safe**: Concurrent operations supported
- ✅ **Lightweight**: Minimal RPC overhead (<1ms on Unix Socket)

## Supported Operations

### Runtime API (libcudart.so)
- `cudaMalloc` / `cudaFree` / `cudaMemcpy` / `cudaMemcpyAsync`
- `cudaStreamCreate/Destroy/Synchronize`
- `cudaEventCreate/Record/Synchronize`
- `cudaSetDevice` / `cudaGetDevice` / `cudaGetDeviceCount`
- [And 20+ more...]

### Driver API (libcuda.so)
- `cuModuleLoadData` — Load fatbin/cubin
- `cuModuleGetFunction` — Get kernel function
- `cuLaunchKernel` — Launch kernel with packed parameters
- `cuMemAlloc_v2` / `cuMemFree_v2` / `cuMemcpy*_v2`
- [And 30+ more...]

### AI Frameworks
- PyTorch: `torch.matmul()` (via cuBLAS)
- PyTorch: `torch.nn.Conv2d()` (via cuDNN)
- PyTorch: `torch.nn.Linear()` (via cuBLAS)
- All CUDA operations called internally by any framework

## Architecture Overview

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design:
- **3-tier architecture**: Frontend (client hooks) / RPC / Backend (execution)
- **dlopen Hook**: How PyTorch dynamic loading is intercepted
- **RPC Protocol**: Message format and communication layer
- **Parameter Packing**: Kernel parameter serialization algorithm
- **Multi-app Isolation**: Context management for concurrent clients

## Running Tests

**C++ smoke tests (end-to-end CUDA kernels):**
```bash
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
./build/vgpu_runtime_kernel_launch_test
./build/vgpu_runtime_memory_smoke_test
./build/vgpu_runtime_async_event_test
```

**Python tests (PyTorch integration):**
```bash
export LD_PRELOAD=/path/to/build/libvgpu_preload_init.so:\
/path/to/build/libcuda.so:/path/to/build/libcudart.so

python tests/python/test_pytorch_matmul.py
python tests/python/test_pytorch_conv.py
python tests/python/test_cuda_runtime.py
```

## Project Structure

```
├── ARCHITECTURE.md                  # Detailed system design (read this!)
├── IMPROVEMENTS.md                  # Future work & optimization ideas
├── README.md                        # This file
├── CMakeLists.txt
├── src/
│   ├── frontend/
│   │   ├── interceptor.cpp          # Runtime API hooks (cuda*)
│   │   └── driver_interceptor.cpp   # Driver API hooks (cu*)
│   ├── backend/
│   │   ├── server_main.cpp          # RPC server loop
│   │   ├── cuda_runtime_loader.cpp  # Dynamic load libcudart.so
│   │   └── cuda_driver_loader.cpp   # Dynamic load libcuda.so
│   └── common/
│       ├── dlopen_hook.cpp          # Intercept dynamic CUDA loading
│       ├── cuda_preload_init.cpp    # High-priority initializer
│       ├── rpc_client.cpp           # Unix Socket RPC client
│       ├── kernel_registry.cpp      # Virtual ↔ real handle mapping
│       ├── fatbin_parser.cpp        # Extract kernel param info
│       └── context_registry.cpp     # Isolate apps by PID
├── include/vgpu/
│   ├── common/                      # Protocol & shared types
│   └── backend/                     # Loader function pointers
├── test/                            # C++ smoke tests
├── tests/python/                    # Python test suite
└── examples/
    └── pytorch_matmul_example.py    # PyTorch matmul demo
```

## Environment Variables

```bash
# Server configuration
VGPU_SERVER_SOCK=/tmp/vgpu_server.sock   # Unix socket path (default)

# Client debugging
VGPU_DEBUG=1                              # Print dlopen interceptions
export LD_PRELOAD=...                     # IMPORTANT: Order matters!
                                          # 1. libvgpu_preload_init.so (hooks)
                                          # 2. libcuda.so (driver shim)
                                          # 3. libcudart.so (runtime shim)
```

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| dlopen interception | ~1ms | One-time cost at library load |
| RPC round-trip (Unix Socket) | <1ms | Local IPC overhead |
| Matrix multiply (1024×1024) | 5-10ms | Dominated by GPU execution |
| Memory copy overhead | <5% | Relative to transfer size |

For long-running kernels (milliseconds+), RPC overhead is negligible.

## Troubleshooting

| Issue | Diagnosis | Solution |
|-------|-----------|----------|
| `CUDA available: False` | dlopen hook not working | Check LD_PRELOAD order: init hook MUST be first |
| `symbol not found: cuLaunchKernel` | libcuda.so not loaded | Add `libcuda.so` to LD_PRELOAD |
| `Failed to connect to VGPU server` | Server not running | Run `ps aux \| grep vgpu_server` and start if needed |
| Segfault in dlopen | Incorrect hook initialization | Ensure `libvgpu_preload_init.so` is in LD_PRELOAD |

## Build & Development

### Compile with debug symbols
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

### Run with verbose output
```bash
export VGPU_DEBUG=1
export VGPU_VERBOSE=1  # Backend logging
./my_app
```

### Debugging with gdb
```bash
gdb -ex run ./build/vgpu_server
# or
strace -f -e trace=network ./my_app
```

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Deep system design
  - Complete data flow examples
  - RPC protocol specification
  - dlopen hook mechanism details
  - Parameter packing algorithm
  - Backend execution flow

- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** — What's next
  - Performance optimization opportunities
  - Known limitations (NCCL, CUDA Graphs)
  - Feature roadmap (TCP support, cuDNN stubs)

- **[tests/python/](tests/python/)** — Python test suite & examples
  - PyTorch integration tests
  - CUDA Runtime API tests
  - Performance benchmarks

## Contributing

Improvements welcome! Priority areas:
- **Performance**: Message batching, parameter caching
- **Libraries**: cuBLAS/cuDNN stub support for easier debugging
- **Networking**: TCP/IP transport for remote servers
- **Multi-GPU**: NCCL support for distributed training
- **Testing**: More comprehensive test coverage

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed roadmap.

## Related Documents

- **Old docs (preserved):**
  - [PYTORCH_INTEGRATION.md](PYTORCH_INTEGRATION.md) — PyTorch setup guide
  - [PYTORCH_ADVANCED_GUIDE.md](PYTORCH_ADVANCED_GUIDE.md) — Advanced PyTorch usage
  - [SOLUTION_SUMMARY.md](SOLUTION_SUMMARY.md) — Original design summary

## License

[Your license here]

---

**Questions?** Check [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design and data flows.
# Virtual-GPU

Virtual-GPU 是一个前后端分离的 CUDA 拦截系统：客户端只需加载本项目提供的 shim（`libcudart.so` 或 `libcuda.so`），即可把 CUDA Runtime/Driver 调用透明转发到后端服务进程执行。

目标是让没有本地 NVIDIA GPU 或本地 CUDA Runtime 的客户端，也能以最小改造成本运行 CUDA 程序。

## 一、系统架构

### 1. Frontend（客户端拦截层）

- Runtime 拦截：`libvgpu_intercept.so` / `libcudart.so`
- Driver 拦截：`libcuda.so`
- 作用：捕获 `cuda*` 和 `cu*` 调用，编码为 RPC 请求并发给后端

### 2. Backend（服务端执行层）

- 服务进程：`vgpu_server`
- 通过 Unix Domain Socket（默认 `/tmp/vgpu_server.sock`）接收请求
- 在服务端加载真实 CUDA Runtime/Driver 并执行请求

### 3. Common（共享基础层）

- 统一协议：RPC 头、操作码、请求/响应结构
- 统一组件：上下文映射、参数打包、fatbin/PTX 解析、客户端注册表

## 二、代码组织

### 1. 源码目录

- `src/frontend/`: 拦截器实现（runtime + driver）
- `src/backend/`: server 与 loader
- `src/common/`: RPC、解析、注册、上下文等公共实现

### 2. 头文件目录

- `include/vgpu/common/`: 协议与通用类型
- `include/vgpu/backend/`: 后端 loader 接口
- `include/vgpu/frontend/`: 预留（当前前端接口以导出符号为主）

### 3. 测试目录

- `test/runtime_shim_smoke_test.cpp`
- `test/runtime_memory_smoke_test.cu`
- `test/runtime_async_event_test.cu`
- `test/runtime_kernel_launch_test.cu`

## 三、实现原理

### 1. Runtime API 转发

`src/frontend/interceptor.cpp` 导出 `cudaMalloc/cudaMemcpy/cudaStream*` 等接口，构造 RPC 请求并发送给服务端。

### 2. Driver API 转发

`src/frontend/driver_interceptor.cpp` 导出 `cuModuleLoadData/cuModuleGetFunction/cuLaunchKernel` 等接口。

对于 kernel launch，支持两条路径：

- `extra[]` 已带打包参数（`CU_LAUNCH_PARAM_BUFFER_POINTER`）时直接透传
- 仅有 `kernelParams` 时，依据解析出的参数布局打包后再转发

### 3. Kernel 注册与参数解析

- `src/common/fatbin_parser.cpp` 从 fatbin/PTX 提取 `.entry/.param` 信息
- `src/common/kernel_registry.cpp` 维护 host 函数、模块、函数 ID 与参数布局映射

### 4. Backend 执行

- `src/backend/server_main.cpp` 解析请求并分发到 runtime/driver 执行分支
- `src/backend/cuda_runtime_loader.cpp` / `src/backend/cuda_driver_loader.cpp` 动态加载真实库符号

## 四、已支持能力

### 1. Runtime

- 内存：`cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemset`, `cudaMemsetAsync`
- 设备：`cudaGetDevice`, `cudaSetDevice`, `cudaGetDeviceCount`, `cudaSetDeviceFlags`, `cudaGetDeviceFlags`, `cudaDeviceReset`, `cudaDeviceSynchronize`
- 查询：`cudaRuntimeGetVersion`, `cudaDriverGetVersion`, `cudaDeviceGetAttribute`, `cudaMemGetInfo`
- Stream/Event：`cudaStreamCreate/Destroy/Synchronize/Query/WaitEvent`, `cudaEventCreate/Record/Synchronize/Destroy/Query`

### 2. Driver

- `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`
- `cuMemAlloc_v2`, `cuMemFree_v2`, `cuMemcpyHtoD_v2`, `cuMemcpyDtoH_v2`, `cuMemcpyDtoD_v2`

## 五、PyTorch 集成

Virtual-GPU 提供了 **dlopen 钩子机制** 来支持 PyTorch 的透明拦截。

PyTorch 会在运行时通过 `dlopen()` 动态加载 CUDA 库，而不是使用 LD_PRELOAD。Virtual-GPU 现在通过拦截 dlopen 调用来确保 PyTorch 加载我们的伪库。

**快速开始**（详见 [PYTORCH_INTEGRATION.md](PYTORCH_INTEGRATION.md)）：

```bash
# 编译
cmake -S . -B build && cmake --build build -j

# 启动服务
VGPU_SERVER_SOCK=/tmp/vgpu_server.sock ./build/vgpu_server &

# 运行 PyTorch（使用 dlopen 钩子）
export LD_PRELOAD=/path/to/build/libvgpu_preload_init.so:/path/to/build/libcuda.so:/path/to/build/libcudart.so
python my_pytorch_app.py
```

PyTorch 的矩阵乘、卷积等操作现在将被透明转发到后端服务执行。

## 六、如何使用

### 1. 构建

```bash
cd /home/yzy/Virtual-GPU
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 2. 启动后端服务

```bash
cd /home/yzy/Virtual-GPU
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
./build/vgpu_server
```

### 3. 运行测试（推荐）

#### 3.1 无本地 CUDA Runtime 客户端模式

```bash
cd /home/yzy/Virtual-GPU
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
export LD_LIBRARY_PATH=$PWD/build:${LD_LIBRARY_PATH}
./build/vgpu_runtime_shim_smoke_test
```

#### 3.2 Runtime 内存链路

```bash
cd /home/yzy/Virtual-GPU
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
export LD_PRELOAD=$PWD/build/libvgpu_intercept.so
./build/vgpu_runtime_memory_smoke_test
```

#### 3.3 Runtime 异步/事件链路

```bash
cd /home/yzy/Virtual-GPU
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
export LD_PRELOAD=$PWD/build/libvgpu_intercept.so
./build/vgpu_runtime_async_event_test
```

#### 3.4 Kernel launch 端到端链路

```bash
cd /home/yzy/Virtual-GPU
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
export LD_LIBRARY_PATH=$PWD/build:${LD_LIBRARY_PATH}
./build/vgpu_runtime_kernel_launch_test
```

## 七、运行策略开关（后端生效）

```bash
export VGPU_VERBOSE=1
export VGPU_DELAY_US=200
export VGPU_MEMCPY_CLAMP_BYTES=1048576
export VGPU_FORCE_DEVICE=0
```
