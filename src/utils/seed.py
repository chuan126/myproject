"""全局随机种子设置。

确保 Python、NumPy、PyTorch 的随机性可复现。
"""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """设置所有随机数生成器的种子。

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
