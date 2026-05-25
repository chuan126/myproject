"""注意力池化层。

通过学习每个时间步的标量分数，加权求和得到定长表示。
"""

import torch
from torch import nn


class AttentionPooling(nn.Module):
    """软注意力池化：score = Linear(encoded)，weights = softmax(score)，output = sum(encoded * weights)。

    Args:
        feature_dim: 每个时间步的特征维度
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.score = nn.Linear(feature_dim, 1)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            encoded: (batch, seq_len, feature_dim)

        Returns:
            (batch, feature_dim) 加权求和后的特征
        """
        weights = torch.softmax(self.score(encoded), dim=1)
        return (encoded * weights).sum(dim=1)
