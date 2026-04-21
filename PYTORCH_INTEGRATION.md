# PyTorch 集成指南

本文档说明如何使用 Virtual-GPU 来透明拦截 PyTorch 程序的 CUDA 调用（包括矩阵乘等操作）。

## 问题背景

PyTorch 使用 cuBLAS、cuDNN 等 CUDA 库来加速矩阵乘法等操作。默认情况下：

```cpp
// PyTorch 内部在运行时会执行：
void* libcuda_handle = dlopen("/usr/local/cuda/lib64/libcuda.so", RTLD_LAZY);
auto cuLaunchKernel = dlsym(libcuda_handle, "cuLaunchKernel");
```

这会导致 PyTorch **绕过 LD_PRELOAD**，直接加载系统的真实 CUDA 库，导致拦截失败。

## 解决方案

Virtual-GPU 现在提供了 **dlopen 钩子机制** 来确保 PyTorch 加载我们的伪 CUDA 库。

### 核心改进

1. **dlopen 拦截** (`src/common/dlopen_hook.cpp`)
   - 拦截所有 `dlopen()` 调用
   - 检测 CUDA 库的 dlopen（libcuda.so、libcudart.so、libcublas.so 等）
   - 自动重定向到 Virtual-GPU 的伪库

2. **预加载初始化器** (`src/common/cuda_preload_init.cpp`)
   - 在进程启动时立即运行（最高优先级 constructor）
   - 确保 dlopen 钩子在任何 CUDA 代码之前就已激活

3. **便利库** (`libvgpu_preload_init.so`)
   - 将 dlopen 钩子和初始化器打包在一起
   - 简化了 PyTorch 集成流程

## 使用方法

### 方式 1：使用便利库（推荐）

编译完成后：

```bash
# 编译
cd /path/to/Virtual-GPU
mkdir -p build && cd build
cmake -S .. -B . && cmake --build . -j

# 启动服务（在后台）
VGPU_SERVER_SOCK=/tmp/vgpu_server.sock ./vgpu_server &

# 运行 PyTorch 程序
# 使用 libvgpu_preload_init.so 作为第一个预加载库
export LD_PRELOAD=/path/to/Virtual-GPU/build/libvgpu_preload_init.so:/path/to/Virtual-GPU/build/libcuda.so:/path/to/Virtual-GPU/build/libcudart.so
export VGPU_DEBUG=1  # 可选：启用调试输出

python my_pytorch_app.py
```

### 方式 2：完整的预加载链

如果不想编译整个项目，可以只编译便利库：

```bash
# 只编译 preload 库
cd /path/to/Virtual-GPU/build
cmake --build . --target vgpu_preload_init

# 或手动编译
g++ -fPIC -shared \
  src/common/dlopen_hook.cpp \
  src/common/cuda_preload_init.cpp \
  -ldl -o libvgpu_preload_init.so
```

### 方式 3：逐个库预加载

如果细粒度控制，可以按顺序加载：

```bash
export LD_PRELOAD=\
/path/to/Virtual-GPU/build/libvgpu_intercept.so:\
/path/to/Virtual-GPU/build/libcuda.so:\
/path/to/Virtual-GPU/build/libcudart.so

python my_pytorch_app.py
```

## 完整的 PyTorch 矩阵乘示例

### 1. Python 脚本 (`pytorch_matmul.py`)

```python
#!/usr/bin/env python3
import torch

print("[pytorch] PyTorch version:", torch.__version__)
print("[pytorch] CUDA available:", torch.cuda.is_available())

# Create matrices
A = torch.randn(512, 512).cuda()
B = torch.randn(512, 512).cuda()

print("[pytorch] Allocated matrices on CUDA device")

# Matrix multiplication (this will trigger cuBLAS -> cuLaunchKernel)
print("[pytorch] Starting matrix multiplication...")
C = torch.matmul(A, B)

print("[pytorch] Matrix multiplication completed")
print("[pytorch] Result shape:", C.shape)
print("[pytorch] Result sum:", C.sum().item())
```

### 2. 启动虚拟 GPU 服务

```bash
cd /path/to/Virtual-GPU/build
VGPU_SERVER_SOCK=/tmp/vgpu_server.sock ./vgpu_server &
```

### 3. 运行 PyTorch 程序

```bash
export LD_PRELOAD=/path/to/Virtual-GPU/build/libvgpu_preload_init.so:\
/path/to/Virtual-GPU/build/libcuda.so:\
/path/to/Virtual-GPU/build/libcudart.so
export VGPU_DEBUG=1

python pytorch_matmul.py

# 预期输出：
# [pytorch] PyTorch version: 2.x.x
# [pytorch] CUDA available: True
# [vGPU] dlopen interception: /usr/local/cuda/lib64/libcuda.so -> libcuda.so
# [vGPU] Preload ready (app_pid=12345)
# [pytorch] Allocated matrices on CUDA device
# [pytorch] Starting matrix multiplication...
# [pytorch] Matrix multiplication completed
# [pytorch] Result shape: torch.Size([512, 512])
# [pytorch] Result sum: 12345.6789
```

## 工作原理

### 执行流程

```
1. Python 进程启动
   ↓
2. LD_PRELOAD 加载 libvgpu_preload_init.so
   ↓
3. cuda_preload_init.cpp 的 constructor 运行 (priority 101)
   - 注册 dlopen/dlsym 钩子
   - 初始化 RPC 客户端
   ↓
4. PyTorch 初始化，调用 dlopen("libcuda.so", RTLD_LAZY)
   ↓
5. dlopen_hook.cpp 的 dlopen() 拦截该调用
   - 检测到 libcuda.so
   - 加载 Virtual-GPU 的 libcuda.so 代替
   ↓
6. PyTorch 获取 cuLaunchKernel 函数指针
   - dlsym 返回我们的伪实现
   ↓
7. PyTorch 调用 cuLaunchKernel
   - 我们的 driver_interceptor.cpp 拦截
   - 打包参数，发送 RPC 到服务器
   ↓
8. 服务器执行真实 CUDA 操作
   - 分配设备内存
   - 执行内核
   - 返回结果
```

### 关键代码片段

#### dlopen 拦截的核心逻辑

```cpp
// src/common/dlopen_hook.cpp
extern "C" void* dlopen(const char* filename, int flags) {
    if (isCudaLibrary(filename)) {
        const char* shim_name = getShimLibraryPath(filename);
        
        // 重定向到我们的库
        void* handle = original_dlopen(shim_name, flags | RTLD_GLOBAL);
        
        fprintf(stderr, "[vGPU] dlopen interception: %s -> %s\n",
                filename, shim_name);
        
        return handle;
    }
    return original_dlopen(filename, flags);
}
```

#### 预加载初始化器

```cpp
// src/common/cuda_preload_init.cpp
__attribute__((constructor(101)))
void initVgpuPreload() {
    // 这个函数在任何其他代码之前运行
    // 确保 dlopen 钩子已准备就绪
}
```

## 支持的 CUDA 库

dlopen 拦截器当前支持以下库的重定向：

- `libcuda.so` - Driver API 库 ✓
- `libcudart.so` - Runtime API 库 ✓
- `libcublas.so` - BLAS 库（检测到则传递，需要自行编译 stub）
- `libcublasLt.so` - BLAS Lite 库
- `libcurand.so` - Random 库
- `libcudnn.so` - cuDNN 库
- 其他库可通过修改 `isCudaLibrary()` 扩展

## 故障排查

### 问题 1：PyTorch 仍然加载真实 CUDA 库

**症状**：调用 dlopen，但 libvgpu_preload_init.so 未能拦截

**解决**：
```bash
# 确保正确的加载顺序
export LD_PRELOAD=/path/to/libvgpu_preload_init.so:/path/to/libcuda.so:/path/to/libcudart.so

# 或启用调试输出
export VGPU_DEBUG=1
```

### 问题 2：Symbol not found 错误

**症状**：`undefined symbol: cuLaunchKernel`

**解决**：确保 libcuda.so 在 LD_PRELOAD 中紧跟 libvgpu_preload_init.so 后面

```bash
export LD_PRELOAD=/path/to/libvgpu_preload_init.so:/path/to/libcuda.so:/path/to/libcudart.so
#                 ↑ 必须首先加载钩子库
```

### 问题 3：Server 连接失败

**症状**：`Failed to connect to VGPU server`

**解决**：
```bash
# 确保服务器已启动
ps aux | grep vgpu_server

# 如果未启动，手动启动
VGPU_SERVER_SOCK=/tmp/vgpu_server.sock /path/to/vgpu_server &

# 或检查 socket 文件
ls -la /tmp/vgpu_server.sock
```

## 性能考虑

1. **dlopen 拦截的开销**：最小（仅在库加载时执行一次）
2. **RPC 通信开销**：取决于网络延迟，对于本地 Unix Socket 通常 < 1ms
3. **参数打包开销**：对于小内核最小，对于大内核可能达到 1-5% 的开销

## 未来改进

1. 提供 cuBLAS、cuDNN 的伪库（当前仅支持 libcuda.so 和 libcudart.so）
2. 更细粒度的符号选择性拦截
3. 本地缓存以减少 RPC 往返
4. 性能计数器和跟踪支持

