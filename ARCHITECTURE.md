# Virtual-GPU 系统架构设计文档

## 1. 系统概览

### 1.1 目标与定位

Virtual-GPU 是一个**分布式 CUDA 拦截系统**，通过动态链接库拦截和转发 CUDA API 调用，使没有本地 GPU 的客户端也能运行 CUDA 程序。

**核心目标**：
- 无需修改应用代码
- 透明转发 CUDA 操作到远程/后端执行
- 支持现代 AI 框架（PyTorch、TensorFlow）
- 支持 cuBLAS、cuDNN 等高性能库

### 1.2 应用场景

```
典型场景 1：CPU 集群 + 共享 GPU
  ┌──────────┐  ┌──────────┐
  │ CPU 节点1 │  │ CPU 节点2 │
  │ (no GPU) │  │ (no GPU) │
  └────┬─────┘  └────┬─────┘
       │ RPC         │ RPC
       └──────┬──────┘
              │ Unix Socket / TCP
         ┌────▼────┐
         │  GPU    │
         │ Server  │
         └─────────┘
         
典型场景 2：PyTorch 分布式训练跨机房
  ┌────────────────┐
  │ PyTorch Master │ (no local GPU)
  │ + dlopen hooks │
  └────────┬───────┘
           │ RPC (all CUDA ops intercepted)
      ┌────▼─────┐
      │ GPU Farm │
      │ (backend)│
      └──────────┘
```

---

## 2. 高层架构

### 2.1 三层架构

```
┌──────────────────────────────────────────────────────┐
│          应用层 (Applications)                        │
│   PyTorch、TensorFlow、Custom CUDA Programs          │
└────────────────────┬─────────────────────────────────┘
                     │ CUDA API calls (cudaMalloc, cuLaunchKernel, etc.)
┌────────────────────▼──────────────────────────────────┐
│     前端层 (Frontend - Client Side)                   │
│  ┌─────────────────────────────────────────────────┐  │
│  │ dlopen 钩子层                                   │  │
│  │ ├─ dlopen_hook.cpp: 拦截动态库加载             │  │
│  │ └─ cuda_preload_init.cpp: 高优先级初始化       │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────┐  │
│  │ API 拦截层                                      │  │
│  │ ├─ interceptor.cpp: Runtime API (cudaMalloc等) │  │
│  │ └─ driver_interceptor.cpp: Driver API (cu*)   │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────┐  │
│  │ 共享基础层                                      │  │
│  │ ├─ rpc_client.cpp: RPC 通信                    │  │
│  │ ├─ kernel_registry.cpp: 内核信息映射          │  │
│  │ ├─ fatbin_parser.cpp: 二进制解析              │  │
│  │ └─ context_registry.cpp: 上下文管理           │  │
│  └─────────────────────────────────────────────────┘  │
└────────────────────┬──────────────────────────────────┘
                     │ RPC Protocol (Unix Socket / TCP)
┌────────────────────▼──────────────────────────────────┐
│     后端层 (Backend - Server Side)                    │
│  ┌─────────────────────────────────────────────────┐  │
│  │ server_main.cpp: RPC 分发器                    │  │
│  │ ├─ 解析 RPC 请求                             │  │
│  │ ├─ 路由到 Runtime/Driver 处理器              │  │
│  │ └─ 构造响应                                 │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────┐  │
│  │ CUDA 加载器                                    │  │
│  │ ├─ cuda_runtime_loader.cpp: Runtime 动态加载   │  │
│  │ └─ cuda_driver_loader.cpp: Driver 动态加载    │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────┐  │
│  │ 真实 CUDA Runtime/Driver                       │  │
│  │ ├─ libcudart.so                              │  │
│  │ └─ libcuda.so                                │  │
│  └─────────────────────────────────────────────────┘  │
└────────────────────┬──────────────────────────────────┘
                     │ NVIDIA GPU
┌────────────────────▼──────────────────────────────────┐
│          硬件层 (GPU Hardware)                        │
│   NVIDIA GPU with CUDA Compute Capability             │
└───────────────────────────────────────────────────────┘
```

### 2.2 模块职责

| 模块 | 位置 | 职责 |
|------|------|------|
| **dlopen Hook** | `src/common/dlopen_hook.cpp` | 拦截 PyTorch 的动态 CUDA 库加载 |
| **Runtime API 拦截** | `src/frontend/interceptor.cpp` | 导出 `cuda*` 符号（内存、流等） |
| **Driver API 拦截** | `src/frontend/driver_interceptor.cpp` | 导出 `cu*` 符号（内核启动等） |
| **RPC 客户端** | `src/common/rpc_client.cpp` | Unix Socket 通信 |
| **内核映射表** | `src/common/kernel_registry.cpp` | 维护假句柄与真实 ID 的映射 |
| **Fatbin 解析** | `src/common/fatbin_parser.cpp` | 从二进制中提取内核参数信息 |
| **上下文管理** | `src/common/context_registry.cpp` | 应用 ID、上下文 ID 映射 |
| **RPC 分发器** | `src/backend/server_main.cpp` | 解析请求、路由、分发 |
| **Runtime 加载器** | `src/backend/cuda_runtime_loader.cpp` | 动态加载 libcudart.so |
| **Driver 加载器** | `src/backend/cuda_driver_loader.cpp` | 动态加载 libcuda.so |

---

## 3. dlopen 拦截机制

### 3.1 为什么需要 dlopen 拦截？

PyTorch、TensorFlow 等框架在运行时通过 `dlopen()` 动态加载 CUDA 库，而非直接链接。

```cpp
// PyTorch 内部代码
void* libcuda_handle = dlopen("/usr/local/cuda/lib64/libcuda.so", RTLD_LAZY);
auto cuLaunchKernel = (cuLaunchKernel_t)dlsym(libcuda_handle, "cuLaunchKernel");
```

**问题**：LD_PRELOAD 无法拦截 dlopen 调用。

**解决**：通过钩子拦截 dlopen，强制加载我们的伪库。

### 3.2 dlopen 钩子实现

```cpp
// src/common/dlopen_hook.cpp
extern "C" void* dlopen(const char* filename, int flags) {
    initOriginalFunctions();
    
    if (g_in_dlopen_hook.exchange(true)) {
        return g_original_dlopen(filename, flags);
    }
    
    // 检测 CUDA 库
    if (isCudaLibrary(filename)) {
        const char* shim_name = getShimLibraryPath(filename);
        fprintf(stderr, "[vGPU] dlopen interception: %s -> %s\n",
                filename, shim_name);
        
        // 加载虚拟库，而非真实库
        void* result = g_original_dlopen(shim_name, flags | RTLD_GLOBAL);
        g_in_dlopen_hook = false;
        return result;
    }
    
    g_in_dlopen_hook = false;
    return g_original_dlopen(filename, flags);
}
```

### 3.3 加载顺序

```
程序启动
  ↓
LD_PRELOAD 加载 libvgpu_preload_init.so (第一个)
  ↓
cuda_preload_init.cpp 的 constructor (优先级 101) 运行
  ↓
初始化 dlopen/dlsym 钩子
  ↓
应用代码开始执行
  ↓
PyTorch 调用 dlopen("/usr/local/cuda/lib64/libcuda.so")
  ↓
dlopen 钩子拦截 → 加载 libcuda.so (来自 LD_PRELOAD)
  ↓
PyTorch 获得 Virtual-GPU 的伪库 ✓
```

---

## 4. API 拦截与转发

### 4.1 Runtime API 拦截流程

```cpp
// interceptor.cpp 导出的 API
extern "C" cudaError_t cudaMalloc(void** devPtr, size_t size) {
    // 1. 构建元数据
    Meta m = buildMeta();  // app_id = getpid(), device, context_id
    
    // 2. 构造请求
    RpcMemAllocReq req{};
    req.size = size;
    
    // 3. 调用 RPC
    RpcResult r = callServer(RpcOp::kCudaMalloc, &req, sizeof(req), nullptr, 0);
    
    // 4. 返回结果
    if (r.status == cudaSuccess && r.payload.size() >= sizeof(uint64_t)) {
        uint64_t device_ptr = *reinterpret_cast<uint64_t*>(r.payload.data());
        *devPtr = reinterpret_cast<void*>(device_ptr);
    }
    return r.status;
}

// 关键特性：
// - 所有内存指针是虚拟的 (客户端 uint64_t，不指向实际内存)
// - 后端维护真实指针 ↔ 虚拟指针映射
// - 数据转移通过额外的 payload 进行
```

### 4.2 Driver API - 内核启动

```cpp
// driver_interceptor.cpp
extern "C" CUresult cuLaunchKernel(
    CUfunction f,                    // 虚拟函数句柄
    unsigned gridDimX, unsigned gridDimY, unsigned gridDimZ,
    unsigned blockDimX, unsigned blockDimY, unsigned blockDimZ,
    unsigned sharedMemBytes,
    CUstream hStream,
    void** kernelParams,             // 参数指针数组
    void** extra) {                  // 可选：预打包的参数缓冲
    
    // 1. 从注册表查找函数信息
    const KernelEntry* ke = kernel_registry.findDriverFunc(f);
    if (!ke) return CUDA_ERROR_INVALID_HANDLE;
    
    // 2. 打包参数
    std::vector<uint8_t> packed;
    if (extra && *extra == CU_LAUNCH_PARAM_BUFFER_POINTER) {
        // 模式 A：参数已预打包 (来自 libcudart.so)
        packed.assign(...);
    } else if (kernelParams) {
        // 模式 B：需要自己打包
        packed = packArgs(ke->params, kernelParams);
    }
    
    // 3. 构造 RPC 请求
    RpcLaunchKernelReq req{};
    req.func_id = ke->func_id;
    req.gridX = gridDimX;
    ...
    
    // 4. 发送到服务器
    RpcResult r = callServer(
        RpcOp::kCuLaunchKernel,
        &req, sizeof(req),
        packed.data(), packed.size()  // 参数作为 extra payload
    );
    
    return r.status;
}
```

### 4.3 参数打包

参数打包遵循 NVIDIA 的规则：

```cpp
// 从 PTX 中提取参数信息 (fatbin_parser.cpp)
struct ParamInfo {
    uint32_t offset;      // 在打包缓冲中的偏移
    uint32_t size;        // 参数大小 (4, 8, 16 等)
    bool is_pointer;      // 是否为指针
};

// 打包示例：
// 内核：__global__ void kernel(int a, float* ptr, double b)
// 打包结果：[4 bytes: a][8 bytes: ptr][8 bytes: b] = 20 bytes

std::vector<uint8_t> packArgs(
    const std::vector<ParamInfo>& params,
    void** kernelParams) {
    
    std::vector<uint8_t> result(total_size);
    for (const auto& p : params) {
        std::memcpy(result.data() + p.offset,
                    kernelParams[p.index],
                    p.size);
    }
    return result;
}
```

---

## 5. RPC 协议设计

### 5.1 消息格式

```
┌─────────────────────────────────────────────────────────┐
│ RPC 请求                                                │
├─────────────────────────────────────────────────────────┤
│ 字段              │ 类型      │ 说明                    │
├─────────────────────────────────────────────────────────┤
│ op                │ uint16    │ 操作码 (如 kCudaMalloc) │
│ app_id            │ uint64    │ 应用 PID (getpid())    │
│ context_id        │ uint64    │ 上下文 ID              │
│ device            │ int32     │ 设备编号               │
│ payload_size      │ uint32    │ 主负载大小             │
│ extra_payload_size│ uint32    │ 额外负载大小           │
│ [payload]         │ bytes     │ 请求数据               │
│ [extra_payload]   │ bytes     │ 额外数据 (fatbin/参数) │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ RPC 响应                                                │
├─────────────────────────────────────────────────────────┤
│ status            │ uint32    │ 错误码                 │
│ payload_size      │ uint32    │ 响应数据大小           │
│ [payload]         │ bytes     │ 响应数据               │
└─────────────────────────────────────────────────────────┘
```

### 5.2 操作码

```cpp
enum class RpcOp : uint16 {
    // Memory operations
    kCudaMalloc,
    kCudaFree,
    kCudaMemcpy,
    kCudaMemcpyAsync,
    kCudaMemset,
    
    // Kernel operations
    kCuModuleLoadData,
    kCuModuleGetFunction,
    kCuLaunchKernel,
    
    // Stream operations
    kCudaStreamCreate,
    kCudaStreamDestroy,
    kCudaStreamSynchronize,
    
    // Device operations
    kCudaGetDeviceCount,
    kCudaGetDevice,
    kCudaSetDevice,
    
    // ... more operations
};
```

### 5.3 通信实现

```cpp
// src/common/rpc_client.cpp
class RpcClient {
private:
    int socket_fd;
    std::mutex socket_lock;
    
public:
    RpcResult call(RpcOp op, uint64_t app_id, uint64_t context_id,
                   int device, const void* payload, size_t payload_size,
                   const void* extra_payload, size_t extra_payload_size) {
        std::lock_guard<std::mutex> lock(socket_lock);
        
        // 1. 序列化请求
        RpcHeader header{op, app_id, context_id, device,
                        (uint32_t)payload_size, (uint32_t)extra_payload_size};
        
        // 2. 发送请求（三部分）
        write(socket_fd, &header, sizeof(header));
        write(socket_fd, payload, payload_size);
        write(socket_fd, extra_payload, extra_payload_size);
        
        // 3. 接收响应
        RpcHeader response_header;
        read(socket_fd, &response_header, sizeof(response_header));
        
        // 4. 构造结果
        RpcResult result;
        result.status = response_header.status;
        result.payload.resize(response_header.payload_size);
        read(socket_fd, result.payload.data(), response_header.payload_size);
        
        return result;
    }
};
```

---

## 6. 后端执行引擎

### 6.1 服务器主循环

```cpp
// src/backend/server_main.cpp
void server_main() {
    int server_socket = socket(AF_UNIX, SOCK_STREAM, 0);
    bind(server_socket, &addr, sizeof(addr));
    listen(server_socket, 16);  // 支持多个客户端
    
    while (true) {
        int client_socket = accept(server_socket, nullptr, nullptr);
        
        // 为每个客户端创建线程
        std::thread([client_socket]() {
            handle_client(client_socket);
        }).detach();
    }
}

void handle_client(int socket) {
    while (true) {
        // 1. 读取请求头
        RpcHeader header;
        if (read(socket, &header, sizeof(header)) <= 0) break;
        
        // 2. 读取负载
        std::vector<uint8_t> payload(header.payload_size);
        read(socket, payload.data(), payload_size);
        
        // 3. 路由到处理器
        RpcResult result = dispatch(header.op, header, payload);
        
        // 4. 返回响应
        write(socket, &result.header, sizeof(result.header));
        write(socket, result.payload.data(), result.payload.size());
    }
}
```

### 6.2 请求分发

```cpp
RpcResult dispatch(RpcOp op, const RpcHeader& header,
                   const std::vector<uint8_t>& payload) {
    switch (op) {
        case RpcOp::kCudaMalloc: {
            RpcMemAllocReq* req = (RpcMemAllocReq*)payload.data();
            void* device_ptr;
            cudaError_t st = cudaMalloc(&device_ptr, req->size);
            
            // 存储映射：虚拟指针 -> 真实指针
            store_mapping(header.app_id, req->virt_addr, device_ptr);
            
            // 构造响应
            return make_result(st, &device_ptr, sizeof(device_ptr));
        }
        
        case RpcOp::kCuLaunchKernel: {
            RpcLaunchKernelReq* req = (RpcLaunchKernelReq*)payload.data();
            CUmodule module = get_module(header.app_id, req->module_id);
            CUfunction func = get_function(module, req->func_id);
            
            // extra_payload 包含打包的内核参数
            void* args[] = { (void*)extra_payload.data() };
            
            return make_result(
                cuLaunchKernel(func,
                    req->gridX, req->gridY, req->gridZ,
                    req->blockX, req->blockY, req->blockZ,
                    0, nullptr,
                    args, nullptr)
            );
        }
        
        // ... more cases
    }
}
```

### 6.3 模块与函数缓存

```cpp
// 服务器端的缓存管理
struct AppContext {
    uint64_t app_id;
    std::map<uint64_t, CUmodule> modules;      // 模块 ID -> 真实模块
    std::map<uint64_t, CUfunction> functions;  // 函数 ID -> 真实函数
    std::map<void*, uint64_t> alloc_map;       // 真实指针 -> 分配大小
};

std::map<uint64_t, AppContext> g_contexts;
std::mutex g_contexts_lock;
```

---

## 7. 内核信息管理

### 7.1 Fatbin 解析

NVIDIA fatbin 格式包含多个二进制镜像 (PTX、cubin 等)。

```cpp
// src/common/fatbin_parser.cpp
struct FatbinHeader {
    uint32_t magic;        // 0xBA55ED50 (fatbin) 或 0x466243B1 (wrapper)
    uint32_t version;
    uint32_t size;
    // ... more fields
    // 后跟多个 binary sections
};

void parse_fatbin(const void* fatbin_data, size_t size) {
    // 1. 检查魔数
    if (magic == 0xBA55ED50) {
        // 直接 fatbin 格式
        parse_fatbin_sections(fatbin_data);
    } else if (magic == 0x466243B1) {
        // Wrapper 格式，先解析包装器
        ...
    }
    
    // 2. 遍历每个 section，提取 PTX
    for (auto section : sections) {
        if (is_ptx(section)) {
            std::string ptx_source(section.data, section.size);
            parse_ptx_kernels(ptx_source);
        }
    }
}

void parse_ptx_kernels(const std::string& ptx_source) {
    // 正则表达式提取内核信息
    // .entry kernel_name(
    //     .param .u64 arg0,
    //     .param .f32 arg1,
    //     ...
    // )
    
    for (auto match : regex_match(ptx_source, kernel_pattern)) {
        std::string kernel_name = match.group(1);
        auto params = parse_params(match.group(2));
        
        kernel_registry.add(kernel_name, params);
    }
}
```

### 7.2 内核注册表

客户端维护的映射：

```cpp
// src/common/kernel_registry.cpp
class KernelRegistry {
private:
    std::map<void*, uint64_t> module_map;      // CUmodule 假句柄 -> 服务器 ID
    std::map<void*, KernelEntry> func_map;     // CUfunction 假句柄 -> 信息
    std::atomic<uint64_t> next_fake_handle{0x10000};
    
public:
    void add_driver_module(void* cu_module, uint64_t module_id) {
        module_map[cu_module] = module_id;
    }
    
    void add_driver_func(void* cu_func, const KernelEntry& entry) {
        func_map[cu_func] = entry;
    }
    
    // KernelEntry 包含：
    // - func_id: 服务器分配的 ID
    // - params: 参数信息列表
    // - total_param_bytes: 打包后的参数缓冲大小
};
```

---

## 8. 上下文管理

### 8.1 多进程隔离

Virtual-GPU 支持多个独立应用并发运行：

```cpp
// src/common/context_registry.cpp
class ContextRegistry {
private:
    std::map<std::pair<uint64_t, int>, uint64_t> app_device_to_context;
    std::atomic<uint64_t> next_context_id{1};
    
public:
    uint64_t acquire_context_id(uint64_t app_id, int device) {
        auto key = std::make_pair(app_id, device);
        
        // 为每个 (app_id, device) 对创建唯一的上下文 ID
        if (app_device_to_context.find(key) == app_device_to_context.end()) {
            app_device_to_context[key] = next_context_id++;
        }
        
        return app_device_to_context[key];
    }
};

// 调用流程中的使用：
Meta m = buildMeta();  // app_id = getpid()
m.context_id = contexts().acquire_context_id(m.app_id, m.device);

// 后端服务器：
// 如果收到来自 app_id=12345, device=0 的请求，
// 自动使用相同的 context_id 来访问该应用的资源
```

---

## 9. 数据流示例

### 9.1 矩阵乘的完整流程

```
┌─ 客户端 (PyTorch) ──────────────────────┐
│                                         │
│ A = torch.randn(1024, 1024).cuda()    │
│     ↓                                   │
│ 调用 cudaMalloc(&d_A, ...)            │
│     ↓                                   │
│ interceptor.cpp: cudaMalloc()          │
│     ↓                                   │
│ 构造 RpcMemAllocReq + RPC 调用        │
│     ↓                                   │
└─────────────────────────────────────────┘
     │ RPC over Unix Socket
┌────▼─ 服务器 (Backend) ─────────────────┐
│                                         │
│ server_main.cpp: 收到请求              │
│     ↓                                   │
│ dispatch → kCudaMalloc 处理器         │
│     ↓                                   │
│ 真实 cudaMalloc(&d_A_real, 1024*1024*4) │
│     ↓                                   │
│ 存储映射：d_A (虚拟) → d_A_real (真实) │
│     ↓                                   │
│ 响应包含 d_A (虚拟指针)                 │
│     ↓                                   │
└─────────────────────────────────────────┘
     │ RPC 响应
┌────▼─ 客户端 (PyTorch) ──────────────────┐
│                                         │
│ *devPtr = d_A (虚拟指针)               │
│     ↓                                   │
│ C = torch.matmul(A, B) → cuBLAS      │
│     ↓                                   │
│ cuLaunchKernel(gemm_kernel, ...)     │
│     ↓                                   │
│ driver_interceptor.cpp: 打包参数      │
│     ↓                                   │
│ RpcLaunchKernelReq + 参数缓冲         │
│     ↓                                   │
└─────────────────────────────────────────┘
     │ RPC over Unix Socket
┌────▼─ 服务器 (Backend) ─────────────────┐
│                                         │
│ 查找映射：d_A_virt → d_A_real        │
│          d_B_virt → d_B_real         │
│          d_C_virt → d_C_real         │
│     ↓                                   │
│ 执行真实内核：                          │
│ cuLaunchKernel(gemm_kernel,           │
│     [d_A_real, d_B_real, d_C_real],  │
│     ...)                              │
│     ↓                                   │
│ GPU 执行计算                            │
│     ↓                                   │
│ 响应成功                                │
│     ↓                                   │
└─────────────────────────────────────────┘
     │ RPC 响应
┌────▼─ 客户端 (PyTorch) ──────────────────┐
│                                         │
│ 矩阵乘完成                              │
│ (结果在虚拟指针 d_C 指向的"远程位置")  │
│                                         │
└─────────────────────────────────────────┘
```

### 9.2 内存复制流程

```
客户端：
  cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost)
      ↓
  构造 RpcMemcpyReq + RPC 调用
      ↓
  等待响应

服务器：
  查找映射：d_src_virt → d_src_real
      ↓
  真实 cudaMemcpy(buf, d_src_real, size, ...)
      ↓
  buf 内容作为 RPC 响应的 payload 返回
      ↓

客户端：
  接收响应，从 payload 复制数据到 h_dst
      ↓
  完成
```

---

## 10. 并发与线程安全

### 10.1 客户端线程安全

```cpp
// RPC 客户端：所有操作通过单一全局连接
thread_local RpcClient g_rpc_client;

// 每个线程有独立的 RPC 客户端以避免线程争用
RpcClient& client() {
    static thread_local RpcClient c;
    return c;
}
```

### 10.2 服务器并发处理

```cpp
// 每个客户端连接分配一个线程
for (;;) {
    int client_socket = accept(server_socket, ...);
    std::thread([client_socket]() {
        handle_client(client_socket);
    }).detach();
}

// AppContext 通过 app_id 隔离
std::map<uint64_t, AppContext> g_contexts;  // 每个 app 独立

// 对全局状态的访问用 mutex 保护
std::mutex g_contexts_lock;
```

---

## 11. 性能考虑

### 11.1 延迟来源

| 操作 | 延迟 | 优化空间 |
|------|------|--------|
| RPC 往返 (Unix Socket) | 0.1-1ms | 消息批处理 |
| 参数打包 | <0.1ms | SIMD 优化 |
| 上下文查询 | <0.01ms | 缓存热数据 |
| GPU 执行 | 毫秒级 | 内核融合 |

### 11.2 优化策略

1. **消息融合**：多个小操作合并为一个 RPC
2. **参数缓存**：缓存频繁使用的内核参数信息
3. **预热**：第一次调用初始化缓存
4. **异步执行**：支持 CUDA Stream 异步操作

---

## 12. 错误处理

### 12.1 错误传播

```
客户端异常 ← RPC 调用失败 ← 后端错误
                    ↓
            返回 CUresult / cudaError_t
                    ↓
            应用层错误处理
```

### 12.2 常见错误

| 错误 | 原因 | 处理 |
|------|------|------|
| CUDA_ERROR_INVALID_HANDLE | 虚拟指针无效 | 验证映射表 |
| cudaErrorMemoryAllocation | 后端显存不足 | 增加 GPU 显存或释放 |
| Socket 连接失败 | 服务器未启动 | 启动服务器 |

---

## 13. 限制与已知问题

1. **本地无 GPU 要求**：后端服务必须能访问真实 GPU
2. **不支持 NCCL**：多 GPU 通信需要额外实现
3. **不支持 Graph API**：CUDA Graph 功能暂未支持
4. **不支持 Unified Memory**：虚拟统一内存需要特殊处理

---

## 总结

Virtual-GPU 通过以下关键创新实现了对 PyTorch 等 AI 框架的透明 CUDA 拦截：

1. **dlopen 钩子**：强制所有动态 CUDA 库加载走 Virtual-GPU
2. **参数打包**：根据 PTX 信息动态打包内核参数
3. **RPC 转发**：统一的 Unix Socket 通信协议
4. **多应用隔离**：基于 app_id + context_id 的资源隔离

这个架构的核心优势是**透明性**和**通用性** —— 无需修改应用代码，支持任何使用 CUDA 的框架。

