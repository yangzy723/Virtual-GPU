# Virtual-GPU

Virtual-GPU 是一个前后端分离的 CUDA 透明拦截系统。客户端可以只加载本项目提供的 shim 库，不依赖本地 NVIDIA GPU 或本地 CUDA Runtime；服务端在远端宿主机上加载真实 CUDA Runtime / Driver API 执行请求。

## 目录结构

- `include/vgpu/`: 公共头文件与协议定义
- `src/`: shim、RPC、加载器、服务端实现
- `test/`: 端到端 smoke tests

## 核心组件

### Runtime shim

- `build/libvgpu_intercept.so`: 用于 `LD_PRELOAD` 的 Runtime 拦截库
- `build/libcudart.so`: 可替代本地 `libcudart.so` 的 shim

### Driver shim

- `build/libcuda.so`: 可替代本地 `libcuda.so` 的 Driver API shim
- 支持 `cuModuleLoadData` / `cuModuleGetFunction` / `cuLaunchKernel` 全链路转发

### Backend server

- `build/vgpu_server`: 后端服务进程
- 默认监听 Unix Domain Socket: `/tmp/vgpu_server.sock`

## 已实现能力

- Runtime API: `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemset`, `cudaMemsetAsync`
- 设备与上下文: `cudaGetDevice`, `cudaSetDevice`, `cudaGetDeviceCount`, `cudaSetDeviceFlags`, `cudaGetDeviceFlags`, `cudaDeviceReset`, `cudaDeviceSynchronize`
- 查询 API: `cudaRuntimeGetVersion`, `cudaDriverGetVersion`, `cudaDeviceGetAttribute`, `cudaMemGetInfo`
- Stream/Event: `cudaStreamCreate`, `cudaStreamDestroy`, `cudaStreamSynchronize`, `cudaStreamQuery`, `cudaStreamWaitEvent`, `cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventDestroy`, `cudaEventQuery`
- Driver API: `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`, `cuMemAlloc_v2`, `cuMemFree_v2`, `cuMemcpyHtoD_v2`, `cuMemcpyDtoH_v2`, `cuMemcpyDtoD_v2`
- Kernel launch: fatbin 注册、函数映射、参数打包、远端 launch

## 构建

```bash
cd /home/yzy/Virtual-GPU
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## 运行服务端

```bash
cd /home/yzy/Virtual-GPU
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
./build/vgpu_server
```

## 运行测试

### 1. 无本地 CUDA Runtime 客户端 smoke test

```bash
cd /home/yzy/Virtual-GPU
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
export LD_LIBRARY_PATH=$PWD/build:${LD_LIBRARY_PATH}
./build/vgpu_runtime_shim_smoke_test
```

### 2. Runtime 内存链路 smoke test

```bash
cd /home/yzy/Virtual-GPU
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
export LD_PRELOAD=$PWD/build/libvgpu_intercept.so
./build/vgpu_runtime_memory_smoke_test
```

### 3. Runtime 异步/事件 smoke test

```bash
cd /home/yzy/Virtual-GPU
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
export LD_PRELOAD=$PWD/build/libvgpu_intercept.so
./build/vgpu_runtime_async_event_test
```

### 4. Kernel launch 端到端测试

```bash
cd /home/yzy/Virtual-GPU
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
export LD_LIBRARY_PATH=$PWD/build:${LD_LIBRARY_PATH}
./build/vgpu_runtime_kernel_launch_test
```

## 可选策略开关

```bash
export VGPU_VERBOSE=1
export VGPU_DELAY_US=200
export VGPU_MEMCPY_CLAMP_BYTES=1048576
export VGPU_FORCE_DEVICE=0
```
