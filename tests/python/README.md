# Virtual-GPU Python Tests

Python tests for Virtual-GPU capability boundaries.

## Tests

### test_capability_boundaries.py

Tests the known boundary between what Virtual-GPU supports and what it does not:

- Starts `vgpu_server` in a clean environment
- Runs `cublas_probe.py` to verify that cuBLAS initialization fails as expected
- Confirms that `torch.cuda.is_available()` can return `True` while cuBLAS `cublasCreate_v2` still fails
- This is the documented boundary: CUDA Runtime/Driver core paths work, but cuBLAS/cuDNN proxying is not implemented

### cublas_probe.py

Standalone probe script that:
- Imports PyTorch and checks `torch.cuda.is_available()`
- Attempts to initialize cuBLAS via `ctypes.CDLL("libcublas.so.12")`
- Exits with code 1 when `cublasCreate_v2` fails (expected behavior)

## Usage

### Run capability boundary tests

```bash
# Via pytest
python -m pytest tests/python/test_capability_boundaries.py -v

# Via the test runner (runs both C++ and Python boundary tests)
bash tests/run_capability_boundaries.sh
```

### Run cublas probe manually

```bash
export LD_PRELOAD=build/libvgpu_preload_init.so:build/libcuda.so:build/libcudart.so
export LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
# Start vgpu_server first, then:
python tests/python/cublas_probe.py
```

## Why no PyTorch compute tests?

Virtual-GPU does not proxy cuBLAS or cuDNN. This means:

- `torch.cuda.is_available()` may return `True` (CUDA Runtime API works)
- `torch.matmul()` on CUDA tensors will fail (requires cuBLAS)
- `torch.nn.Conv2d` forward on CUDA will fail (requires cuDNN)

Tests for these operations were removed because they test functionality that is intentionally not implemented, not bugs to be fixed.
