# Virtual-GPU

Virtual-GPU is a CUDA API interception and remote execution prototype for single-machine GPU backends.

Client-side shim libraries (`libcudart.so.12`, `libcuda.so`, `libvgpu_preload_init.so`) intercept CUDA Runtime and Driver API calls, then forward them over a Unix Domain Socket to `vgpu_server`, which executes them against the real CUDA libraries on a physical GPU.

## What Works

**Core memory management (Runtime & Driver)**
- `cudaMalloc` / `cudaFree`, `cuMemAlloc_v2` / `cuMemFree_v2`
- `cudaMemcpy` / `cudaMemcpyAsync` (H2D, D2H, D2D)
- `cudaMemset` / `cudaMemsetAsync`
- `cuMemcpyHtoD_v2` / `cuMemcpyDtoH_v2` / `cuMemcpyDtoD_v2`

**Kernel launch (Runtime & Driver)**
- NVCC fatbin registration: `__cudaRegisterFatBinary`, `__cudaRegisterFunction`
- Fatbin / PTX parameter layout parsing
- `cudaLaunchKernel`, `cuLaunchKernel`
- `cuModuleLoadData` / `cuModuleLoadDataEx` / `cuModuleLoad`
- `cuModuleGetFunction`

**Device, context, stream, event management**
- `cudaGetDeviceCount` / `cudaGetDevice` / `cudaSetDevice`
- `cudaStreamCreate` / `Destroy` / `Synchronize` / `Query` / `WaitEvent`
- `cudaEventCreate` / `Destroy` / `Record` / `Synchronize` / `Query`
- Common Driver API device / context / stream / event query stubs

**Dynamic library interception**
- `dlopen` / `dlsym` hooks redirect PyTorch and other frameworks to shim libraries
- Controlled via `LD_PRELOAD` of `libvgpu_preload_init.so`

**Real driver passthrough for non-intercepted symbols**
- `cuGetProcAddress` resolves non-intercepted symbols from the real `libcuda.so.1`
- `cuGetExportTable` forwards to the real driver when available
- This lets cuBLAS, cuDNN, and other CUDA libraries function on machines with a real GPU

**PyTorch support**
- `torch.cuda.is_available()` returns `True`
- `torch.matmul()` works (via cuBLAS using real driver passthrough)
- Memory operations (`.cuda()`, `.cpu()`, `torch.zeros` on GPU) work via RPC

## What Does Not Work

**Not a complete CUDA compatibility layer.** Many APIs return `cudaErrorNotSupported` / `CUDA_ERROR_NOT_SUPPORTED` by design:

- CUDA Graph capture and execution
- IPC (inter-process communication) memory handles
- Unified Virtual Memory (virtual memory management)
- Peer access between devices
- Memory pools (`cuMemPool*`)
- Memory advise / prefetch (`cuMemAdvise`, `cuMemPrefetchAsync`)
- External semaphores
- Occupancy API
- Profiler control (`cudaProfilerStart` / `cudaProfilerStop`)
- Stream memory operations (`cuStreamWaitValue*`, `cuStreamWriteValue*`, `cuStreamBatchMemOp`)
- Host memory registration (`cudaHostRegister`, `cuMemHostRegister`)

**cuBLAS / cuDNN use real driver passthrough, not RPC proxy.** cuBLAS and cuDNN call CUDA Driver API functions (export tables, context management) that are resolved from the real `libcuda.so.1` rather than forwarded through the server. This means:

- cuBLAS/cuDNN operate against the local GPU directly for internal operations
- Memory allocation and kernel launch still go through the vGPU server via RPC
- This is a pragmatic hybrid approach, not a full proxy

**No network transport.** Only Unix Domain Socket on the same machine.

**No multi-GPU scheduling or isolation.**

## Known Boundaries

- Ordinary CUDA programs using core Runtime / Driver paths work correctly.
- Framework initialization that probes optional APIs will see reasonable defaults or explicit `not_supported`, rather than silent misbehavior.
- cuBLAS / cuDNN work via real driver passthrough — they need a real GPU on the same machine.
- Optional symbols (Graph, capture, occupancy) are hidden by default to prevent frameworks from taking unsupported code paths. Set `VGPU_EXPOSE_OPTIONAL_CUDA_SYMBOLS=1` to expose them.

## Components

| Component | Source | Description |
| --- | --- | --- |
| `libvgpu_preload_init.so` | `src/common/dlopen_hook.cpp`, `src/common/cuda_preload_init.cpp` | Preload library; hooks `dlopen` / `dlsym` |
| `libcudart.so.12` | `src/frontend/interceptor.cpp` | Runtime API shim |
| `libcuda.so` | `src/frontend/driver_interceptor.cpp` | Driver API shim |
| `vgpu_server` | `src/backend/server_main.cpp` | Server process; loads real CUDA libs |

## Quick Start

### Build

```bash
cd /path/to/vGPU
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Start the server

```bash
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
./build/vgpu_server
```

### Run a CUDA program

```bash
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
export LD_LIBRARY_PATH=/path/to/vGPU/build:$LD_LIBRARY_PATH
./your_cuda_app
```

### Run a PyTorch program

```bash
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
export LD_PRELOAD=/path/to/vGPU/build/libvgpu_preload_init.so:/path/to/vGPU/build/libcuda.so:/path/to/vGPU/build/libcudart.so
export LD_LIBRARY_PATH=/path/to/vGPU/build:$LD_LIBRARY_PATH
python your_pytorch_script.py
```

## Tests

### C++ tests

```bash
# Smoke test: basic memory operations (requires running vgpu_server)
./build/vgpu_runtime_shim_smoke_test

# Boundary tests: verify not_supported semantics (no server needed)
./build/vgpu_runtime_not_supported_semantics_test
./build/vgpu_runtime_capability_boundary_test
./build/vgpu_driver_capability_boundary_test
```

### Python tests

```bash
# cuBLAS boundary test (requires running vgpu_server)
python -m pytest tests/python/test_capability_boundaries.py -v

# torch.matmul end-to-end test (requires running vgpu_server + PyTorch)
python -m pytest tests/python/test_torch_matmul.py -v
```

### Capability boundary test runner

```bash
bash tests/run_capability_boundaries.sh
```

## Environment Variables

| Variable | Description |
| --- | --- |
| `VGPU_SERVER_SOCK` | Server Unix Domain Socket path |
| `VGPU_RPC_TIMEOUT_MS` | RPC timeout in milliseconds |
| `VGPU_DEBUG` | Enable verbose client-side logging |
| `VGPU_VERBOSE` | Enable verbose server-side request logging |
| `VGPU_FORCE_DEVICE` | Force server to bind to a specific device |
| `VGPU_DELAY_US` | Inject artificial delay on server (microseconds) |
| `VGPU_MEMCPY_CLAMP_BYTES` | Clamp single memcpy size on server |
| `VGPU_LOG_CU_GETPROC` | Log `cuGetProcAddress` / `cuGetExportTable` calls |
| `VGPU_EXPOSE_OPTIONAL_CUDA_SYMBOLS` | Expose optional CUDA symbols via `cuGetProcAddress` |
| `VGPU_OPTIONAL_SYMBOLS_RETURN_NOT_SUPPORTED` | Optional symbols return `CUDA_ERROR_NOT_SUPPORTED` |
| `VGPU_USE_FAKE_CU_EXPORT_TABLES` | Force fake `cuGetExportTable` tables (override real driver) |
| `VGPU_CU_EXPORT_TABLE_SUCCESS_NULL` | Force `cuGetExportTable` to return success with null table |

## Directory Layout

```
src/
  frontend/          Client shim implementations (interceptor.cpp, driver_interceptor.cpp)
  backend/           Server implementation and real CUDA loaders
  common/            Fatbin parser, RPC client, kernel registry, dlopen hook
include/vgpu/
  common/            Protocol definitions, CUDA ABI types
  frontend/          Shim utility helpers
  backend/           Server-side headers
tests/
  cpp/               C++ boundary and smoke tests
  python/            Python capability boundary and matmul tests
```

## Design

See [DESIGN.md](DESIGN.md) for architecture details.
