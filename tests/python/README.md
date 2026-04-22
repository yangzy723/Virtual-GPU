# Virtual-GPU Python Test Suite

Comprehensive test suite for Virtual-GPU CUDA interception with PyTorch and CUDA Runtime API.

## Test Coverage

### test_pytorch_matmul.py
- Basic 2D matrix multiplication
- Batched operations
- Different data types (float32, float64)
- Large matrix operations
- Correctness verification
- Vector-matrix operations
- Linear layer forward/backward
- Convolution operations
- Memory management
- Stream and synchronization

### test_pytorch_conv.py
- Conv2d with various kernel sizes
- Strided and dilated convolutions
- Depthwise and grouped convolutions
- Conv1d and Conv3d
- Complete CNN architectures
- Functional convolution API
- Gradient computation
- Performance characteristics

### test_cuda_runtime.py
- Memory allocation/deallocation
- Host-to-Device, Device-to-Host, Device-to-Device transfers
- Stream creation and synchronization
- Event recording and timing
- Device properties and management
- Kernel launch operations
- Error handling

### pytorch_matmul_demo.py
- Minimal end-to-end demo script
- Uses the same LD_PRELOAD interception path as tests
- Suitable for quick manual validation before running full pytest

## Quick Start

### Prerequisites

1. Virtual-GPU compiled:
```bash
cd /path/to/Virtual-GPU
cmake -S . -B build && cmake --build build -j
```

2. Backend server running:
```bash
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
./build/vgpu_server &
```

3. PyTorch installed:
```bash
pip install torch pytest
```

### Run All Tests

```bash
cd /path/to/Virtual-GPU

# Set up environment
export LD_PRELOAD=/path/to/build/libvgpu_preload_init.so:\
/path/to/build/libcuda.so:/path/to/build/libcudart.so
export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock

# Run all tests
python -m pytest tests/python/ -v

# With output
python -m pytest tests/python/ -v -s
```

### Run Specific Test File

```bash
# Matrix multiplication tests only
python -m pytest tests/python/test_pytorch_matmul.py -v

# Convolution tests only
python -m pytest tests/python/test_pytorch_conv.py -v

# CUDA Runtime tests only
python -m pytest tests/python/test_cuda_runtime.py -v

# Run the demo script (not a pytest test)
python tests/python/pytorch_matmul_demo.py
```

### Run Specific Test

```bash
# Single test
python -m pytest tests/python/test_pytorch_matmul.py::TestPyTorchMatmul::test_basic_matmul_2d -v

# Test class
python -m pytest tests/python/test_pytorch_matmul.py::TestPyTorchMatmul -v

# With pattern
python -m pytest tests/python/ -k "matmul and not batch" -v
```

### Debug Mode

```bash
# Enable Virtual-GPU debug output
export VGPU_DEBUG=1

# Run with pytest output capture disabled
python -m pytest tests/python/ -v -s

# Run with Python debugger
python -m pdb -m pytest tests/python/test_pytorch_matmul.py::TestPyTorchMatmul::test_basic_matmul_2d

# Verbose logging
export VGPU_VERBOSE=1
python -m pytest tests/python/ -v -s
```

## Test Organization

### Fixtures (in conftest.py)

- `torch_available`: Check PyTorch is installed
- `check_cuda`: Verify CUDA available via Virtual-GPU (skips if not)
- `print_environment`: Print test environment info

### Running Fixtures

Tests that use `check_cuda` fixture will:
- Skip if `torch.cuda.is_available()` returns False
- This happens when Virtual-GPU server isn't running

Example skip message:
```
SKIPPED [50%] tests/python/test_pytorch_matmul.py::TestPyTorchMatmul::test_basic_matmul_2d 
  CUDA not available (check if vgpu_server is running)
```

## Expected Output

### Successful Run
```
=================== test session starts ====================
platform linux -- Python 3.10.0, pytest-7.x.x, ...

======= Virtual-GPU Python Test Environment ========
PyTorch version: 2.0.0
CUDA available: True
CUDA device: NVIDIA A100-SXM4-40GB
CUDA device count: 1
LD_PRELOAD: /path/to/build/libvgpu_preload_init.so:...
================================================================

tests/python/test_pytorch_matmul.py::TestPyTorchMatmul::test_basic_matmul_2d PASSED [ 0%]
tests/python/test_pytorch_matmul.py::TestPyTorchMatmul::test_matmul_batch PASSED [ 1%]
...

==================== 45 passed in 2.34s ====================
```

### Troubleshooting Failed Tests

1. **Import errors**
   ```
   ModuleNotFoundError: No module named 'torch'
   ```
   → Run: `pip install torch`

2. **CUDA not available**
   ```
   SKIPPED - CUDA not available (check if vgpu_server is running)
   ```
   → Start server: `./build/vgpu_server &`
   → Check LD_PRELOAD is set correctly

3. **Socket connection failed**
   ```
   Failed to connect to VGPU server at /tmp/vgpu_server.sock
   ```
   → Verify: `ls -la /tmp/vgpu_server.sock`
   → Restart server if missing

4. **Symbol not found**
   ```
   undefined symbol: cuLaunchKernel
   ```
   → Check LD_PRELOAD order (libvgpu_preload_init.so MUST be first)

## Performance Testing

### Run with Timing

```bash
python -m pytest tests/python/ -v --durations=10
```

This shows the 10 slowest tests, useful for identifying performance issues.

### Custom Performance Test

Add to any test:
```python
import time

def test_performance(check_cuda):
    start = time.time()
    
    # Your test
    A = torch.randn(1024, 1024, device='cuda')
    B = torch.randn(1024, 1024, device='cuda')
    C = torch.matmul(A, B)
    
    elapsed = time.time() - start
    print(f"Time: {elapsed*1000:.2f} ms")
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Virtual-GPU Python Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: self-hosted  # Requires GPU
    steps:
      - uses: actions/checkout@v2
      - name: Build
        run: |
          cmake -S . -B build
          cmake --build build -j
      - name: Start Server
        run: |
          export VGPU_SERVER_SOCK=/tmp/vgpu_server.sock
          ./build/vgpu_server &
          sleep 1
      - name: Run Tests
        run: |
          export LD_PRELOAD=...
          python -m pytest tests/python/ -v
```

## Adding New Tests

### Test Template

```python
import torch
import pytest

class TestNewFeature:
    """Test description."""
    
    def test_something(self, check_cuda):
        """Test specific behavior."""
        # Arrange
        x = torch.randn(256, 256, device='cuda')
        
        # Act
        y = torch.matmul(x, x)
        
        # Assert
        assert y.shape == (256, 256)
        assert y.device.type == 'cuda'
```

### Guidelines

1. Use descriptive test names: `test_<operation>_<scenario>`
2. Use `check_cuda` fixture to verify CUDA availability
3. Include docstrings explaining what is tested
4. Group related tests in classes
5. Use assertions with clear messages

## Debugging pytest

### Show print statements
```bash
python -m pytest tests/python/ -s
```

### Stop on first failure
```bash
python -m pytest tests/python/ -x
```

### Show local variables on failure
```bash
python -m pytest tests/python/ -l
```

### Verbose traceback
```bash
python -m pytest tests/python/ --tb=long
```

## FAQ

**Q: Tests skip with "CUDA not available"**
A: Server not running. Start with `./build/vgpu_server &`

**Q: Tests timeout**
A: Server may be overloaded. Increase timeout or run fewer tests in parallel.

**Q: Memory error**
A: Backend GPU out of memory. Reduce test size or free GPU memory.

**Q: LD_PRELOAD errors**
A: Verify order: init hook first, then libcuda.so, then libcudart.so

## Performance Expectations

On local Unix Socket with GPU:
- Simple operations: 1-10 ms
- Matrix multiply (256x256): 0.5-2 ms
- Matrix multiply (1024x1024): 5-15 ms
- Convolution (64x64): 2-10 ms

RPC overhead typically < 1 ms on local communication.

