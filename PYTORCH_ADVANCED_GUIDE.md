# PyTorch 实战指南

本指南说明如何使用 Virtual-GPU 来运行 PyTorch 程序，确保矩阵乘、卷积等操作被透明拦截。

## 一、问题诊断

### PyTorch 为何需要特殊支持？

PyTorch 在运行时会通过 `dlopen()` 动态加载 CUDA 库，而不是直接链接：

```cpp
// PyTorch/cuBLAS 内部会这样做
void* libcuda_handle = dlopen("/usr/local/cuda/lib64/libcuda.so", RTLD_LAZY);
auto cuLaunchKernel = (cuLaunchKernel_t)dlsym(libcuda_handle, "cuLaunchKernel");
```

**问题**：这绕过了 LD_PRELOAD 机制，导致：

1. PyTorch 加载系统的真实 CUDA 库，而非我们的伪库
2. CUDA 调用不经过 Virtual-GPU 拦截
3. 矩阵乘等操作无法在远程服务器上执行

### 解决方案：dlopen 钩子

Virtual-GPU 通过拦截 `dlopen()` 和 `dlsym()` 系统调用来解决：

```cpp
// Virtual-GPU 会拦截这个调用
void* dlopen(const char* filename, int flags) {
    if (strstr(filename, "libcuda.so")) {
        // 加载 Virtual-GPU 的伪库而非真实库
        return original_dlopen("libcuda.so", flags);
    }
    return original_dlopen(filename, flags);
}
```

## 二、环境设置

### 前置条件

- Virtual-GPU 已编译：`/path/to/Virtual-GPU/build/`
- PyTorch 已安装：`pip install torch`
- CUDA 12.x 在后端服务器上可用

### 编译（如尚未编译）

```bash
cd /path/to/Virtual-GPU
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

输出文件：
- `build/libvgpu_preload_init.so` - dlopen 钩子库（必须首先加载）
- `build/libcuda.so` - Driver API 伪库
- `build/libcudart.so` - Runtime API 伪库
- `build/vgpu_server` - 后端服务

## 三、快速启动

### 第一步：启动服务

```bash
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
/path/to/Virtual-GPU/build/vgpu_server &
```

验证服务运行：
```bash
ls -la /tmp/vgpu_server.sock
```

### 第二步：运行 PyTorch 程序

```bash
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
export LD_PRELOAD=/path/to/Virtual-GPU/build/libvgpu_preload_init.so:\
/path/to/Virtual-GPU/build/libcuda.so:\
/path/to/Virtual-GPU/build/libcudart.so

# 可选：启用调试输出
export VGPU_DEBUG=1

python my_pytorch_app.py
```

**关键点**：`libvgpu_preload_init.so` 必须首先加载。

### 第三步：验证

如果看到类似输出，说明拦截成功：

```
[vGPU] dlopen interception: /usr/local/cuda/lib64/libcuda.so -> libcuda.so
[vGPU] Preload ready (app_pid=12345)
[pytorch] CUDA available: True
```

## 四、使用示例

### 示例 1：简单矩阵乘

```python
import torch

print("CUDA available:", torch.cuda.is_available())

# 创建矩阵
A = torch.randn(1024, 1024).cuda()
B = torch.randn(1024, 1024).cuda()

# 矩阵乘（将被拦截）
C = torch.matmul(A, B)

print("Result shape:", C.shape)
print("Result mean:", C.mean().item())
```

**运行**：

```bash
export LD_PRELOAD=/path/to/libvgpu_preload_init.so:/path/to/libcuda.so:/path/to/libcudart.so
python my_script.py
```

### 示例 2：CNN 卷积操作

```python
import torch
import torch.nn as nn

# 创建卷积层
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1).cuda()

# 输入数据
x = torch.randn(8, 3, 32, 32).cuda()

# 卷积操作（被拦截）
y = conv(x)

print("Output shape:", y.shape)
```

### 示例 3：Transformer 自注意力

```python
import torch

batch_size, seq_len, d = 32, 100, 512

Q = torch.randn(batch_size, seq_len, d).cuda()
K = torch.randn(batch_size, seq_len, d).cuda()
V = torch.randn(batch_size, seq_len, d).cuda()

# 注意力计算：Q @ K^T（大矩阵乘）
scores = torch.matmul(Q, K.transpose(-2, -1))
print("Attention scores shape:", scores.shape)

# Softmax + V（继续用伪 CUDA）
attn = torch.softmax(scores, dim=-1)
output = torch.matmul(attn, V)
print("Output shape:", output.shape)
```

## 五、调试与故障排查

### 问题 1：dlopen 拦截不工作

**症状**：
```
[pytorch] CUDA available: False
```

**检查**：
```bash
# 确认 LD_PRELOAD 顺序正确
echo $LD_PRELOAD

# 应该看到 libvgpu_preload_init.so 在最前面
# /path/to/libvgpu_preload_init.so:/path/to/libcuda.so:...
```

**解决**：
```bash
export LD_PRELOAD=/path/to/libvgpu_preload_init.so:/path/to/libcuda.so:/path/to/libcudart.so
```

### 问题 2：Symbol not found

**症状**：
```
undefined symbol: cuLaunchKernel
```

**原因**：libcuda.so 没有正确加载

**解决**：
```bash
# 验证库文件存在
ls -la /path/to/libcuda.so
ldd /path/to/libcuda.so

# 验证符号导出
nm -D /path/to/libcuda.so | grep cuLaunchKernel
```

### 问题 3：RPC 连接失败

**症状**：
```
Failed to connect to VGPU server
```

**检查**：
```bash
# 确认服务器运行
ps aux | grep vgpu_server

# 检查 socket 文件
ls -la /tmp/vgpu_server.sock

# 检查服务器日志
tail -f /tmp/vgpu_server.log
```

**解决**：
```bash
# 启动服务
VGPU_SERVER_SOCK=/tmp/vgpu_server.sock /path/to/vgpu_server &
```

### 问题 4：结果不正确

**症状**：计算结果与 GPU 版本不符

**检查清单**：
- 服务器上是否有真实 GPU？
- CUDA 驱动是否安装正确？
- 是否有足够的显存？

**调试**：
```bash
# 启用详细日志
export VGPU_VERBOSE=1
python my_script.py
```

## 六、性能优化

### 1. 减少 RPC 往返

避免频繁的小操作，合并成大操作：

```python
# 不好：每次操作都是 RPC 调用
for i in range(1000):
    x = torch.randn(10, 10).cuda()
    y = torch.randn(10, 10).cuda()
    z = torch.matmul(x, y)

# 更好：批量处理
x = torch.randn(1000, 10, 10).cuda()
y = torch.randn(1000, 10, 10).cuda()
z = torch.matmul(x, y)  # 一次 RPC 调用
```

### 2. 使用适当的精度

```python
# float16 参数更少，RPC 负载更小
x = torch.randn(1024, 1024, dtype=torch.float16).cuda()
y = torch.randn(1024, 1024, dtype=torch.float16).cuda()
z = torch.matmul(x, y)
```

### 3. 本地预热

```python
# 第一个操作可能较慢（初始化、连接等）
# 预热以确保缓存就绪
_ = torch.matmul(torch.randn(10, 10).cuda(), torch.randn(10, 10).cuda())

# 后续操作更快
for _ in range(100):
    result = torch.matmul(x, y)
```

## 七、环境变量控制

### 调试相关

```bash
# 启用 dlopen 拦截日志
export VGPU_DEBUG=1

# 启用详细服务器日志
export VGPU_VERBOSE=1
```

### 服务通信

```bash
# 指定 socket 路径
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock

# 指定超时时间（毫秒）
export VGPU_RPC_TIMEOUT_MS=5000
```

## 八、完整的部署流程

### 在远程服务器上（带有真实 GPU）

```bash
# 编译 Virtual-GPU
cd /path/to/Virtual-GPU
cmake -S . -B build && cmake --build build -j

# 启动服务（使用高可用性方案）
nohup env VGPU_SERVER_SOCK=/tmp/vgpu_server.sock \
  /path/to/Virtual-GPU/build/vgpu_server > /tmp/vgpu_server.log 2>&1 &

# 或使用 systemd
sudo systemctl start vgpu-server
```

### 在客户端计算机上（无 GPU）

```bash
# 设置环境
cat > ~/.bashrc_pytorch << 'EOF'
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
export LD_PRELOAD=/path/to/Virtual-GPU/build/libvgpu_preload_init.so:\
/path/to/Virtual-GPU/build/libcuda.so:\
/path/to/Virtual-GPU/build/libcudart.so
export VGPU_DEBUG=1
EOF

source ~/.bashrc_pytorch

# 运行 PyTorch
python my_pytorch_training_script.py
```

## 九、常见问题 (FAQ)

**Q: PyTorch 会检测到 GPU 吗？**  
A: 是的，torch.cuda.is_available() 会返回 True，因为我们的伪 cuGetDeviceCount 总是返回成功。

**Q: 性能会下降多少？**  
A: 取决于 RPC 延迟。本地 Unix Socket 通常 < 1ms，网络可能 10-50ms。对于大操作（秒级计算），开销可以忽略不计。

**Q: 可以在多个客户端上同时运行吗？**  
A: 是的，Virtual-GPU 服务器支持多个客户端并发连接。

**Q: 如果服务器在不同机器上呢？**  
A: 需要使用 TCP Socket 而非 Unix Socket。当前实现仅支持 Unix Socket，需要 SSH 端口转发或修改源码。

**Q: 支持 GPU 之间的通信吗？**  
A: 目前不支持 NCCL。需要额外实现 NCCL 的伪实现。

