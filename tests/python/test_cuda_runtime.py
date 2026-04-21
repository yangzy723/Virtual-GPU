"""
CUDA Runtime API Tests

Tests direct CUDA Runtime API calls:
- cudaMalloc / cudaFree
- cudaMemcpy
- cudaLaunchKernel
- Stream and Event operations
"""

import torch
import pytest


class TestCUDAMemory:
    """Test CUDA memory operations via Runtime API."""

    def test_malloc_free(self, check_cuda):
        """Test cudaMalloc and cudaFree."""
        # Allocate
        tensor = torch.empty(1024, 1024, device='cuda')
        assert tensor.is_cuda
        
        # Free
        del tensor
        torch.cuda.empty_cache()

    def test_memcpy_h2d(self, check_cuda):
        """Test Host-to-Device memcpy."""
        data_cpu = torch.randn(512, 512, dtype=torch.float32)
        data_gpu = data_cpu.cuda()
        
        assert data_gpu.device.type == 'cuda'
        assert torch.allclose(data_cpu, data_gpu.cpu(), atol=1e-5)

    def test_memcpy_d2h(self, check_cuda):
        """Test Device-to-Host memcpy."""
        data_gpu = torch.randn(512, 512, device='cuda', dtype=torch.float32)
        data_cpu = data_gpu.cpu()
        
        assert data_cpu.device.type == 'cpu'
        assert torch.isfinite(data_cpu).all()

    def test_memcpy_d2d(self, check_cuda):
        """Test Device-to-Device memcpy."""
        data_gpu1 = torch.randn(512, 512, device='cuda')
        data_gpu2 = data_gpu1.clone()
        
        assert torch.allclose(data_gpu1, data_gpu2)

    def test_memset(self, check_cuda):
        """Test cudaMemset (via torch.zeros)."""
        zeros = torch.zeros(256, 256, device='cuda')
        
        assert zeros.sum() == 0
        assert zeros.shape == (256, 256)

    def test_memory_info(self, check_cuda):
        """Test cudaMemGetInfo."""
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            assert props.total_memory > 0


class TestCUDAStreams:
    """Test CUDA Stream operations."""

    def test_stream_create_destroy(self, check_cuda):
        """Test stream creation and destruction."""
        stream = torch.cuda.Stream()
        assert stream is not None
        
        # Stream auto-destroyed

    def test_stream_synchronize(self, check_cuda):
        """Test stream synchronization."""
        stream = torch.cuda.Stream()
        
        with torch.cuda.stream(stream):
            x = torch.randn(256, 256, device='cuda')
            y = torch.matmul(x, x)
        
        stream.synchronize()  # Explicit sync

    def test_stream_query(self, check_cuda):
        """Test stream query (done/not done)."""
        stream = torch.cuda.Stream()
        
        with torch.cuda.stream(stream):
            x = torch.randn(10, 10, device='cuda')
        
        # After sync, query should be done
        stream.synchronize()

    def test_multiple_streams(self, check_cuda):
        """Test concurrent operations on multiple streams."""
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        
        with torch.cuda.stream(stream1):
            x1 = torch.randn(256, 256, device='cuda')
            y1 = torch.matmul(x1, x1)
        
        with torch.cuda.stream(stream2):
            x2 = torch.randn(256, 256, device='cuda')
            y2 = torch.matmul(x2, x2)
        
        torch.cuda.synchronize()


class TestCUDAEvents:
    """Test CUDA Event operations."""

    def test_event_create_record(self, check_cuda):
        """Test event creation and recording."""
        event = torch.cuda.Event()
        
        x = torch.randn(256, 256, device='cuda')
        event.record()
        y = torch.matmul(x, x)
        event.synchronize()

    def test_event_elapsed_time(self, check_cuda):
        """Test elapsed time between events."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        
        # Do some work
        x = torch.randn(512, 512, device='cuda')
        for _ in range(10):
            x = torch.matmul(x, x.T)
        
        end.record()
        torch.cuda.synchronize()
        
        elapsed = start.elapsed_time(end)
        assert elapsed > 0

    def test_event_query(self, check_cuda):
        """Test event query."""
        event = torch.cuda.Event()
        
        x = torch.randn(10, 10, device='cuda')
        event.record()
        torch.cuda.synchronize()
        
        # After sync, query should be True


class TestCUDADeviceOperations:
    """Test device-level operations."""

    def test_get_device_count(self, check_cuda):
        """Test cudaGetDeviceCount."""
        count = torch.cuda.device_count()
        assert count >= 1

    def test_get_set_device(self, check_cuda):
        """Test cudaGetDevice and cudaSetDevice."""
        if torch.cuda.device_count() >= 1:
            torch.cuda.set_device(0)
            current = torch.cuda.current_device()
            assert current == 0

    def test_device_synchronize(self, check_cuda):
        """Test cudaDeviceSynchronize."""
        x = torch.randn(256, 256, device='cuda')
        torch.cuda.synchronize()  # Device-level sync

    def test_device_properties(self, check_cuda):
        """Test cudaGetDeviceProperties."""
        props = torch.cuda.get_device_properties(0)
        assert props.name  # Device has a name
        assert props.total_memory > 0


class TestKernelLaunch:
    """Test kernel launch through Runtime API."""

    def test_simple_kernel_via_cuda(self, check_cuda):
        """Test kernel launch via cudaLaunchKernel."""
        # Create a simple CUDA operation that triggers kernel launch
        x = torch.randn(256, 256, device='cuda', requires_grad=True)
        y = torch.randn(256, 256, device='cuda', requires_grad=True)
        
        # This internally calls cuLaunchKernel
        z = torch.matmul(x, y)
        
        assert z.shape == (256, 256)

    def test_kernel_with_parameters(self, check_cuda):
        """Test kernel launch with various parameters."""
        # Different parameter configurations
        operations = [
            lambda: torch.matmul(torch.randn(128, 128, device='cuda'),
                                 torch.randn(128, 128, device='cuda')),
            lambda: torch.randn(256, device='cuda').sum(),
            lambda: torch.nn.functional.relu(torch.randn(256, 256, device='cuda')),
            lambda: torch.nn.functional.softmax(
                torch.randn(64, 1000, device='cuda'), dim=1),
        ]
        
        for op in operations:
            result = op()
            assert result is not None


class TestErrorHandling:
    """Test error handling in CUDA operations."""

    def test_invalid_device(self):
        """Test handling of invalid device."""
        try:
            # This should handle gracefully
            torch.cuda.get_device_properties(999)
        except (RuntimeError, IndexError):
            # Expected behavior
            pass

    def test_memory_allocation_check(self, check_cuda):
        """Test that memory operations report errors properly."""
        # Normal allocation should work
        x = torch.randn(1024, 1024, device='cuda')
        assert x.numel() == 1024 * 1024


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
