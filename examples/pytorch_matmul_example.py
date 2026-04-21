#!/usr/bin/env python3
"""
PyTorch CUDA 矩阵乘示例 - 演示 Virtual-GPU 拦截

运行方式：
  # 启动后端服务
  VGPU_SERVER_SOCK=/tmp/vgpu_server.sock /path/to/vgpu_server &
  
  # 使用 dlopen 钩子运行该脚本
  export LD_PRELOAD=/path/to/libvgpu_preload_init.so:/path/to/libcuda.so:/path/to/libcudart.so
  export VGPU_DEBUG=1
  python3 pytorch_matmul_example.py
"""

import sys
import os

# 检查 PyTorch 是否可用
try:
    import torch
except ImportError:
    print("Error: PyTorch not installed")
    print("Install with: pip install torch")
    sys.exit(1)

def main():
    print("[pytorch] PyTorch version:", torch.__version__)
    print("[pytorch] CUDA available:", torch.cuda.is_available())
    
    if not torch.cuda.is_available():
        print("[pytorch] Warning: CUDA not detected")
        print("[pytorch] Make sure Virtual-GPU server is running and LD_PRELOAD is set correctly")
        print("[pytorch] Continuing anyway...")
    
    # 设置矩阵维度
    size = 256
    print(f"[pytorch] Creating {size}x{size} matrices on CUDA...")
    
    try:
        # 在 CUDA 上分配矩阵
        A = torch.randn(size, size, device='cuda', dtype=torch.float32)
        B = torch.randn(size, size, device='cuda', dtype=torch.float32)
        
        print(f"[pytorch] Matrix A shape: {A.shape}, device: {A.device}")
        print(f"[pytorch] Matrix B shape: {B.shape}, device: {B.device}")
        
        # 执行矩阵乘法（这会触发 cuBLAS -> cuLaunchKernel）
        print("[pytorch] Starting matrix multiplication (torch.matmul)...")
        C = torch.matmul(A, B)
        
        print(f"[pytorch] Matrix multiplication completed")
        print(f"[pytorch] Result shape: {C.shape}, device: {C.device}")
        
        # 计算一些统计信息
        result_sum = C.sum().item()
        result_mean = C.mean().item()
        result_max = C.max().item()
        result_min = C.min().item()
        
        print(f"[pytorch] Result statistics:")
        print(f"  - Sum:  {result_sum:.2f}")
        print(f"  - Mean: {result_mean:.6f}")
        print(f"  - Max:  {result_max:.6f}")
        print(f"  - Min:  {result_min:.6f}")
        
        # 验证结果合理性
        expected_mean = 0.0  # randn 应该平均为 0
        if abs(result_mean) < 10:  # 合理范围
            print("[pytorch] ✓ Result appears valid")
        else:
            print("[pytorch] ✗ Result seems suspicious")
        
        # 额外测试：在线性层中使用矩阵乘
        print("\n[pytorch] Testing linear layer (uses GEMM internally)...")
        linear = torch.nn.Linear(size, size, bias=False, device='cuda')
        x = torch.randn(16, size, device='cuda')  # 批处理
        
        y = linear(x)
        print(f"[pytorch] Linear layer output shape: {y.shape}")
        print(f"[pytorch] Linear layer output mean: {y.mean().item():.6f}")
        
        print("\n[pytorch] ✓ All tests passed!")
        return 0
        
    except Exception as e:
        print(f"[pytorch] ✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
