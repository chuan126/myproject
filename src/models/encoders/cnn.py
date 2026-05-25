"""一维卷积神经网络编码器 (CNN1D Encoder)。

渐进通道减半结构：hidden_channels → hidden_channels/2 → hidden_channels/4 → ...
每层包含 Conv1d → BatchNorm1d → ReLU → Dropout。
"""

import torch
from torch import nn


class CNNEncoder(nn.Module):
    """一维卷积编码器，沿时间轴提取局部模式。

    参数字典:
        input_dim: 输入特征维度（作为卷积的输入通道数）
        hidden_channels: 第一层卷积输出通道数，后续层逐层减半
        num_layers: 卷积层数量
        kernel_size: 卷积核大小，必须为奇数以保证时序长度不变
        dropout: Dropout 概率
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("CNNEncoder num_layers must be at least 1.")
        if kernel_size % 2 == 0:
            raise ValueError("CNNEncoder kernel_size must be odd for same-length output.")

        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()

        # 第一层: input_dim → hidden_channels
        padding = kernel_size // 2
        self.conv_layers.append(nn.Conv1d(input_dim, hidden_channels, kernel_size, padding=padding))
        self.batch_norm_layers.append(nn.BatchNorm1d(hidden_channels))

        # 渐进通道减半
        for i in range(1, num_layers):
            in_channels = hidden_channels // (2 ** (i - 1))
            out_channels = hidden_channels // (2 ** i)
            if out_channels < 1:
                raise ValueError(
                    f"CNNEncoder hidden_channels={hidden_channels} too small for num_layers={num_layers}. "
                    f"Layer {i} would have {out_channels} channels."
                )
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding))
            self.batch_norm_layers.append(nn.BatchNorm1d(out_channels))

        # 最终输出通道数
        self.output_dim = hidden_channels // (2 ** (num_layers - 1))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            inputs: (batch, seq_len, input_dim) 形状的张量

        Returns:
            (batch, seq_len, output_dim) 形状的张量
        """
        # Conv1d 期望 (batch, channels, seq_len)，需要转置
        x = inputs.transpose(1, 2)
        for conv_layer, batch_norm_layer in zip(self.conv_layers, self.batch_norm_layers):
            x = self.relu(batch_norm_layer(conv_layer(x)))
            x = self.dropout(x)
        # 转回 (batch, seq_len, channels)
        return x.transpose(1, 2)
