"""
PyTorch Convolution Tests

Tests torch.nn.Conv2d and other convolution operations.
These typically use cuDNN on the backend, which calls cuLaunchKernel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest


class TestConv2dBasic:
    """Basic Conv2d tests."""

    def test_conv2d_simple(self, check_cuda):
        """Test simple Conv2d layer."""
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, device='cuda')
        x = torch.randn(8, 3, 32, 32, device='cuda')
        
        y = conv(x)
        
        assert y.shape == (8, 16, 32, 32)
        assert y.device.type == 'cuda'

    def test_conv2d_different_kernel_sizes(self, check_cuda):
        """Test Conv2d with different kernel sizes."""
        kernel_sizes = [1, 3, 5, 7]
        
        for k_size in kernel_sizes:
            conv = nn.Conv2d(3, 16, kernel_size=k_size, padding=k_size//2, 
                            device='cuda')
            x = torch.randn(4, 3, 64, 64, device='cuda')
            
            y = conv(x)
            
            assert y.shape == (4, 16, 64, 64)

    def test_conv2d_stride(self, check_cuda):
        """Test Conv2d with stride."""
        conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, device='cuda')
        x = torch.randn(4, 3, 64, 64, device='cuda')
        
        y = conv(x)
        
        assert y.shape == (4, 16, 32, 32)

    def test_conv2d_dilation(self, check_cuda):
        """Test Conv2d with dilation."""
        conv = nn.Conv2d(3, 16, kernel_size=3, dilation=2, padding=2, device='cuda')
        x = torch.randn(4, 3, 64, 64, device='cuda')
        
        y = conv(x)
        
        # With dilation=2 and padding=2, output size is same as input
        assert y.shape == (4, 16, 64, 64)

    def test_conv2d_no_bias(self, check_cuda):
        """Test Conv2d without bias."""
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False, 
                        device='cuda')
        x = torch.randn(4, 3, 32, 32, device='cuda')
        
        y = conv(x)
        
        assert y.shape == (4, 16, 32, 32)


class TestConv2dBackwardPass:
    """Test gradient computation for Conv2d."""

    def test_conv2d_backward(self, check_cuda):
        """Test backward pass through Conv2d."""
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, device='cuda')
        x = torch.randn(4, 3, 32, 32, device='cuda', requires_grad=True)
        
        y = conv(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert conv.weight.grad is not None
        assert conv.bias.grad is not None

    def test_conv2d_gradient_values(self, check_cuda):
        """Test that gradients have reasonable values."""
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, device='cuda')
        x = torch.randn(4, 3, 32, 32, device='cuda', requires_grad=True)
        
        y = conv(x)
        loss = y.sum()
        loss.backward()
        
        # Gradients should be non-zero
        assert (x.grad != 0).any()
        assert (conv.weight.grad != 0).any()


class TestDepthwiseConvolution:
    """Test depthwise convolutions."""

    def test_depthwise_conv(self, check_cuda):
        """Test depthwise Conv2d (groups = in_channels)."""
        in_channels = 16
        conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                        groups=in_channels, padding=1, device='cuda')
        x = torch.randn(4, in_channels, 32, 32, device='cuda')
        
        y = conv(x)
        
        assert y.shape == (4, in_channels, 32, 32)

    def test_grouped_conv(self, check_cuda):
        """Test grouped convolution."""
        conv = nn.Conv2d(32, 64, kernel_size=3, groups=4, padding=1, device='cuda')
        x = torch.randn(4, 32, 32, 32, device='cuda')
        
        y = conv(x)
        
        assert y.shape == (4, 64, 32, 32)


class TestConv1dConv3d:
    """Test 1D and 3D convolutions."""

    def test_conv1d(self, check_cuda):
        """Test Conv1d."""
        conv = nn.Conv1d(3, 16, kernel_size=3, padding=1, device='cuda')
        x = torch.randn(4, 3, 128, device='cuda')
        
        y = conv(x)
        
        assert y.shape == (4, 16, 128)

    def test_conv3d(self, check_cuda):
        """Test Conv3d."""
        conv = nn.Conv3d(3, 16, kernel_size=3, padding=1, device='cuda')
        x = torch.randn(2, 3, 16, 16, 16, device='cuda')
        
        y = conv(x)
        
        assert y.shape == (2, 16, 16, 16, 16)


class TestConvolutionalNetworks:
    """Test complete convolutional networks."""

    def test_simple_cnn(self, check_cuda):
        """Test simple CNN."""
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
            
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                return x
        
        model = SimpleCNN().to('cuda')
        x = torch.randn(4, 3, 64, 64, device='cuda')
        
        y = model(x)
        
        assert y.shape == (4, 32, 16, 16)

    def test_resnet_block(self, check_cuda):
        """Test ResNet-style residual block."""
        class ResBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, 
                                      padding=1)
                self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, 
                                      padding=1)
            
            def forward(self, x):
                return x + F.relu(self.conv2(F.relu(self.conv1(x))))
        
        model = ResBlock(32).to('cuda')
        x = torch.randn(4, 32, 64, 64, device='cuda')
        
        y = model(x)
        
        assert y.shape == (4, 32, 64, 64)


class TestFunctionalConvolution:
    """Test functional convolution operations."""

    def test_functional_conv2d(self, check_cuda):
        """Test F.conv2d."""
        weight = torch.randn(16, 3, 3, 3, device='cuda')
        bias = torch.randn(16, device='cuda')
        x = torch.randn(4, 3, 32, 32, device='cuda')
        
        y = F.conv2d(x, weight, bias, padding=1)
        
        assert y.shape == (4, 16, 32, 32)

    def test_functional_conv2d_no_bias(self, check_cuda):
        """Test F.conv2d without bias."""
        weight = torch.randn(16, 3, 3, 3, device='cuda')
        x = torch.randn(4, 3, 32, 32, device='cuda')
        
        y = F.conv2d(x, weight, padding=1)
        
        assert y.shape == (4, 16, 32, 32)


class TestConvolutionPerformance:
    """Test convolution performance characteristics."""

    def test_conv_output_size_calculation(self, check_cuda):
        """Verify output size calculation for various conv configs."""
        configs = [
            # (in_size, out_channels, kernel, stride, padding, expected_out_size)
            (32, 16, 3, 1, 1, 32),
            (32, 16, 3, 1, 0, 30),
            (64, 32, 5, 1, 2, 64),
            (64, 32, 5, 2, 2, 32),
            (128, 64, 3, 1, 1, 128),
        ]
        
        for in_size, out_ch, kernel, stride, padding, exp_size in configs:
            conv = nn.Conv2d(3, out_ch, kernel_size=kernel, stride=stride, 
                            padding=padding, device='cuda')
            x = torch.randn(4, 3, in_size, in_size, device='cuda')
            
            y = conv(x)
            
            assert y.shape == (4, out_ch, exp_size, exp_size), \
                f"Failed for config: in={in_size}, k={kernel}, s={stride}, p={padding}"

    def test_large_conv(self, check_cuda):
        """Test larger convolution."""
        conv = nn.Conv2d(64, 128, kernel_size=3, padding=1, device='cuda')
        x = torch.randn(8, 64, 224, 224, device='cuda')
        
        y = conv(x)
        
        assert y.shape == (8, 128, 224, 224)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
