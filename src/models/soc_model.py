"""SOC 估计主模型。

组装编码器、池化层和回归头，形成完整的 Encoder → Pooling → Head 流水线。
"""

from torch import nn


class SOCModel(nn.Module):
    """SOC 估计模型：Encoder → Pooling → Head。

    Args:
        encoder: 时序编码器，将 (batch, seq_len, input_dim) 编码为 (batch, seq_len, feature_dim)
        pooling: 池化层，将 (batch, seq_len, feature_dim) 聚合为 (batch, feature_dim)
        head: 回归头，将 (batch, feature_dim) 映射为 (batch, 1)
    """

    def __init__(self, encoder: nn.Module, pooling: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.head = head

    def forward(self, inputs):
        """前向传播。"""
        encoded = self.encoder(inputs)
        pooled = self.pooling(encoded)
        return self.head(pooled)
