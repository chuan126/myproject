"""全连接网络编码器 (FCN Encoder)。

渐进减半结构：hidden_size → hidden_size/2 → hidden_size/4 → ...
每层包含 Linear → BatchNorm1d → ReLU → Dropout。
"""

import torch
from torch import nn


class FCNEncoder(nn.Module):
    """全连接网络编码器，逐时间步独立处理序列特征。

    参数字典:
        input_dim: 输入特征维度
        hidden_size: 第一层隐层大小，后续层逐层减半
        num_layers: 隐层数量（不含输出层）
        dropout: Dropout 概率
    """

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError("FCNEncoder num_layers must be at least 1.")

        self.hidden_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()

        # 第一层: input_dim → hidden_size
        self.hidden_layers.append(nn.Linear(input_dim, hidden_size))
        self.batch_norm_layers.append(nn.BatchNorm1d(hidden_size))

        # 渐进减半: hidden_size → hidden_size/2 → hidden_size/4 → ...
        for i in range(1, num_layers):
            in_features = hidden_size // (2 ** (i - 1))
            out_features = hidden_size // (2 ** i)
            if out_features < 1:
                raise ValueError(
                    f"FCNEncoder hidden_size={hidden_size} too small for num_layers={num_layers}. "
                    f"Layer {i} would have {out_features} features."
                )
            self.hidden_layers.append(nn.Linear(in_features, out_features))
            self.batch_norm_layers.append(nn.BatchNorm1d(out_features))

        # 最终输出维度 = 最后一层隐层大小
        self.output_dim = hidden_size // (2 ** (num_layers - 1))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            inputs: (batch, seq_len, input_dim) 形状的张量

        Returns:
            (batch, seq_len, output_dim) 形状的张量
        """
        batch, seq_len, _ = inputs.shape
        # 将序列维并入批次维，逐时间步独立处理
        x = inputs.reshape(batch * seq_len, -1)
        for hidden_layer, batch_norm_layer in zip(self.hidden_layers, self.batch_norm_layers):
            x = self.relu(batch_norm_layer(hidden_layer(x)))
            x = self.dropout(x)
        # 恢复序列维
        return x.reshape(batch, seq_len, self.output_dim)
