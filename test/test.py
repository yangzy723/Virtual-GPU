import torch

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建两个随机矩阵，并放到 GPU
A = torch.randn(3, 4, device=device)  # 3x4 矩阵
B = torch.randn(4, 5, device=device)  # 4x5 矩阵

# 进行矩阵乘法
C = torch.matmul(A, B)  # 或者 A @ B

print(f"当前计算设备: {device}")
print("\n矩阵 A:")
print(A)
print("\n矩阵 B:")
print(B)
print("\n矩阵乘法结果 C:")
print(C)
