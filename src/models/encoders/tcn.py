"""时序卷积网络编码器 (TCN Encoder)。

使用膨胀因果卷积逐层扩大感受野，每层 dilation = 2^layer_index。
保持时序长度不变（通过适当的 padding）。
"""

import torch
from torch import nn


class TCNEncoder(nn.Module):
    """时序卷积网络编码器。

    参数字典:
        input_dim: 输入特征维度
        hidden_size: 每层卷积的输出通道数
        num_layers: 卷积层数
        kernel_size: 卷积核大小，必须为奇数
        dropout: 每层后的 Dropout 概率
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("TCN kernel_size must be odd so the temporal length is preserved.")
        self.output_dim = hidden_size
        layers: list[nn.Module] = []
        in_channels = input_dim
        for layer_index in range(num_layers):
            dilation = 2**layer_index  # 膨胀因子指数增长
            padding = dilation * (kernel_size - 1) // 2  # 保持长度不变
            layers.extend(
                [
                    nn.Conv1d(in_channels, hidden_size, kernel_size, padding=padding, dilation=dilation),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = hidden_size
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            inputs: (batch, seq_len, input_dim)

        Returns:
            (batch, seq_len, hidden_size)
        """
        # Conv1d 期望 (batch, channels, seq_len)
        encoded = self.network(inputs.transpose(1, 2))
        return encoded.transpose(1, 2)
