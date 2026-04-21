"""
Virtual-GPU Python Test Suite

Tests for PyTorch and CUDA Runtime interception.

Usage:
    # Run all tests
    python -m pytest tests/python/ -v
    
    # Run specific test
    python -m pytest tests/python/test_pytorch_matmul.py::test_basic_matmul -v
    
    # With debug output
    export VGPU_DEBUG=1
    python -m pytest tests/python/ -v -s
"""

import os
import sys
import torch
import pytest


# Fixture to check PyTorch is available
@pytest.fixture(scope="session")
def torch_available():
    """Check if PyTorch and CUDA are available through Virtual-GPU."""
    assert torch is not None, "PyTorch not installed"
    return True


@pytest.fixture
def check_cuda():
    """Verify CUDA is available via Virtual-GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available (check if vgpu_server is running)")
    return True


@pytest.fixture(scope="session", autouse=True)
def print_environment():
    """Print environment info for debugging."""
    print("\n" + "="*70)
    print("Virtual-GPU Python Test Environment")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"LD_PRELOAD: {os.environ.get('LD_PRELOAD', 'not set')[:80]}...")
    print("="*70 + "\n")
    yield
