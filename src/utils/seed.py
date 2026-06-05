"""全局随机种子设置模块。

本模块确保深度学习实验的可复现性（reproducibility）。
通过统一设置 Python random、NumPy 和 PyTorch 的随机种子，
保证在相同硬件、相同代码和相同种子的条件下，多次运行产生完全一致的结果。

随机性来源及控制：
- Python random: 影响数据划分等 Python 层面的随机操作
- NumPy: 影响 NumPy 生成的随机数组（如噪声、采样）
- PyTorch CPU: 影响权重初始化、dropout 等 CPU 端随机操作
- PyTorch CUDA: 影响 GPU 端随机操作
- cuDNN: 通过 deterministic 模式消除卷积等 cuDNN 操作的算法非确定性

注意事项：
  - cuDNN deterministic 模式可能降低训练速度（约 10-20%），
    原因是禁用了 cuDNN 的自动算法择优（benchmark 模式）
  - 完全可复现需要相同的 PyTorch/CUDA/cuDNN 版本

在整个项目中的角色：
  被 scripts/train.py 在训练开始前调用，确保实验可复现。
  评估脚本可选择性地也调用此函数，但不强制。
"""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """设置所有随机数生成器的种子，确保实验可复现。

    按顺序设置 Python random、NumPy、PyTorch CPU 和 PyTorch CUDA 的种子，
    并启用 cuDNN 确定性模式以消除 GPU 算法选择导致的不确定性。

    参数:
        seed: 整数随机种子。建议使用固定值（如 42）或从配置中读取。

    副作用:
        - 修改全局随机状态（Python、NumPy、PyTorch）
        - 如果 CUDA 可用，修改 cuDNN 全局设置（deterministic 和 benchmark）
        - 这些修改是全局的、不可逆的（在当前进程中）

    使用建议:
        - 在训练脚本的最开始调用，在任何随机操作之前
        - 评估脚本中如果涉及 dropout 等随机操作也应调用
    """
    # Python 内置 random 模块
    random.seed(seed)
    # NumPy 随机数生成器
    np.random.seed(seed)
    # PyTorch CPU 随机数生成器
    torch.manual_seed(seed)
    # CUDA 相关设置（仅在 GPU 可用时有意义）
    if torch.cuda.is_available():
        # 为所有 GPU 设置相同的种子
        torch.cuda.manual_seed_all(seed)
        # 启用确定性模式：cuDNN 使用确定性算法，牺牲少量性能换取可复现性
        torch.backends.cudnn.deterministic = True
        # 禁用 benchmark 模式：不自动搜索最快算法，因为算法选择可能非确定性
        torch.backends.cudnn.benchmark = False
