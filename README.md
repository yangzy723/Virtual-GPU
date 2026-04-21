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

## 五、如何使用

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

## 六、运行策略开关（后端生效）

```bash
export VGPU_VERBOSE=1
export VGPU_DELAY_US=200
export VGPU_MEMCPY_CLAMP_BYTES=1048576
export VGPU_FORCE_DEVICE=0
```
