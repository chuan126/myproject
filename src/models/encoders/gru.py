"""GRU 编码器。"""

from torch import nn


class GRUEncoder(nn.Module):
    """多层 GRU 编码器，batch_first 模式。

    参数字典:
        input_dim: 输入特征维度
        hidden_size: 隐层大小
        num_layers: GRU 层数
        dropout: 层间 Dropout 概率（仅 num_layers > 1 时生效）
    """

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.output_dim = hidden_size
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, inputs):
        """前向传播。

        Args:
            inputs: (batch, seq_len, input_dim)

        Returns:
            (batch, seq_len, hidden_size)
        """
        outputs, _ = self.gru(inputs)
        return outputs
