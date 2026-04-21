"""
PyTorch Matrix Multiplication Tests

Tests torch.matmul() which internally uses cuBLAS and cuLaunchKernel.
This tests the full chain: PyTorch -> dlopen hook -> Virtual-GPU -> cuBLAS -> GPU
"""

import torch
import pytest
import math


class TestPyTorchMatmul:
    """Basic PyTorch matrix multiplication tests."""

    def test_basic_matmul_2d(self, check_cuda):
        """Test basic 2D matrix multiplication."""
        A = torch.randn(256, 256, device='cuda', dtype=torch.float32)
        B = torch.randn(256, 256, device='cuda', dtype=torch.float32)
        
        C = torch.matmul(A, B)
        
        assert C.shape == (256, 256)
        assert C.device.type == 'cuda'
        assert C.dtype == torch.float32
        
        # Verify result is finite
        assert torch.isfinite(C).all(), "Result contains NaN or Inf"

    def test_matmul_batch(self, check_cuda):
        """Test batched matrix multiplication."""
        batch_size = 32
        m, k, n = 64, 128, 64
        
        A = torch.randn(batch_size, m, k, device='cuda')
        B = torch.randn(batch_size, k, n, device='cuda')
        
        C = torch.matmul(A, B)
        
        assert C.shape == (batch_size, m, n)

    def test_matmul_different_dtypes(self, check_cuda):
        """Test matrix multiplication with different data types."""
        sizes = [128, 256]
        dtypes = [torch.float32, torch.float64]
        
        for size in sizes:
            for dtype in dtypes:
                A = torch.randn(size, size, device='cuda', dtype=dtype)
                B = torch.randn(size, size, device='cuda', dtype=dtype)
                
                C = torch.matmul(A, B)
                
                assert C.shape == (size, size)
                assert C.dtype == dtype

    def test_matmul_large(self, check_cuda):
        """Test larger matrix multiplication."""
        size = 1024
        
        A = torch.randn(size, size, device='cuda')
        B = torch.randn(size, size, device='cuda')
        
        C = torch.matmul(A, B)
        
        assert C.shape == (size, size)
        # Check statistics are reasonable
        assert abs(C.mean().item()) < 100  # Should be around 0

    def test_matmul_result_correctness_small(self, check_cuda):
        """Verify correctness on small matrices (CPU reference)."""
        A_gpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
        B_gpu = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
        
        C_gpu = torch.matmul(A_gpu, B_gpu)
        
        # Expected result (computed by hand):
        # [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
        # [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
        expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], device='cuda')
        
        assert torch.allclose(C_gpu, expected, atol=1e-5)

    def test_vector_matrix_product(self, check_cuda):
        """Test vector-matrix multiplication."""
        v = torch.randn(256, device='cuda')
        M = torch.randn(256, 512, device='cuda')
        
        result = torch.matmul(v, M)
        
        assert result.shape == (512,)

    def test_batched_vector_matrix(self, check_cuda):
        """Test batched vector-matrix multiplication."""
        batch_size = 16
        v = torch.randn(batch_size, 256, device='cuda')
        M = torch.randn(batch_size, 256, 512, device='cuda')
        
        result = torch.matmul(v.unsqueeze(1), M).squeeze(1)
        
        assert result.shape == (batch_size, 512)


class TestLinearLayer:
    """Test PyTorch Linear layer (uses GEMM internally)."""

    def test_linear_forward(self, check_cuda):
        """Test linear layer forward pass."""
        in_features = 512
        out_features = 256
        batch_size = 32
        
        linear = torch.nn.Linear(in_features, out_features, device='cuda')
        x = torch.randn(batch_size, in_features, device='cuda')
        
        y = linear(x)
        
        assert y.shape == (batch_size, out_features)

    def test_linear_no_bias(self, check_cuda):
        """Test linear layer without bias."""
        linear = torch.nn.Linear(128, 64, bias=False, device='cuda')
        x = torch.randn(16, 128, device='cuda')
        
        y = linear(x)
        
        assert y.shape == (16, 64)

    def test_linear_backward(self, check_cuda):
        """Test linear layer gradient computation."""
        linear = torch.nn.Linear(128, 64, device='cuda')
        x = torch.randn(16, 128, device='cuda', requires_grad=True)
        
        y = linear(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert linear.weight.grad is not None


class TestConvolution:
    """Test PyTorch convolution (uses cuDNN internally)."""

    def test_conv2d_basic(self, check_cuda):
        """Test basic 2D convolution."""
        conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1, device='cuda')
        x = torch.randn(8, 3, 32, 32, device='cuda')
        
        y = conv(x)
        
        assert y.shape == (8, 16, 32, 32)

    def test_conv2d_different_configs(self, check_cuda):
        """Test convolution with different configurations."""
        configs = [
            (1, 1, 1),    # in_channels, out_channels, kernel_size
            (3, 16, 3),
            (16, 32, 5),
            (32, 64, 3),
        ]
        
        for in_c, out_c, k_size in configs:
            conv = torch.nn.Conv2d(in_c, out_c, kernel_size=k_size, 
                                   padding=k_size//2, device='cuda')
            x = torch.randn(4, in_c, 64, 64, device='cuda')
            
            y = conv(x)
            
            assert y.shape == (4, out_c, 64, 64)


class TestMemoryOperations:
    """Test CUDA memory operations."""

    def test_cuda_malloc_free(self, check_cuda):
        """Test GPU memory allocation and freeing."""
        # Allocate
        x = torch.randn(1024, 1024, device='cuda')
        assert x.is_cuda
        
        # Free (implicit with del)
        del x
        torch.cuda.synchronize()

    def test_host_device_transfer(self, check_cuda):
        """Test data transfer between host and device."""
        # Host to Device
        x_cpu = torch.randn(256, 256)
        x_gpu = x_cpu.cuda()
        assert x_gpu.device.type == 'cuda'
        
        # Device to Host
        x_cpu2 = x_gpu.cpu()
        assert x_cpu2.device.type == 'cpu'
        
        # Verify values are preserved
        assert torch.allclose(x_cpu, x_cpu2, atol=1e-5)

    def test_memcpy_large(self, check_cuda):
        """Test large data transfer."""
        size = 10 * 1024 * 1024  # 10 MB
        x_cpu = torch.randn(size, dtype=torch.float32)
        
        # CPU -> GPU
        x_gpu = x_cpu.cuda()
        
        # GPU -> CPU
        x_cpu2 = x_gpu.cpu()
        
        # Verify
        assert torch.allclose(x_cpu, x_cpu2, atol=1e-5)


class TestStreamAndSync:
    """Test CUDA stream and synchronization."""

    def test_stream_create(self, check_cuda):
        """Test stream creation and usage."""
        stream = torch.cuda.Stream()
        
        with torch.cuda.stream(stream):
            x = torch.randn(256, 256, device='cuda')
            y = torch.randn(256, 256, device='cuda')
            z = torch.matmul(x, y)
        
        torch.cuda.synchronize()

    def test_device_synchronize(self, check_cuda):
        """Test device synchronization."""
        x = torch.randn(512, 512, device='cuda')
        y = torch.randn(512, 512, device='cuda')
        
        z = torch.matmul(x, y)
        torch.cuda.synchronize()  # Explicit sync
        
        assert z.shape == (512, 512)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
