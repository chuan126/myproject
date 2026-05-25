"""回归头模块。

将编码器输出的特征映射为单个 SOC 预测值。
支持可选的一层隐藏层 + Dropout。
"""

from torch import nn


class RegressionHead(nn.Module):
    """回归头：特征 → SOC 预测值。

    参数字典:
        input_dim: 输入特征维度
        hidden_size: 中间隐层大小，为 None 或 0 时退化为单层 Linear
        dropout: Dropout 概率
    """

    def __init__(self, input_dim: int, hidden_size: int | None = None, dropout: float = 0.0):
        super().__init__()
        if hidden_size:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.network = nn.Linear(input_dim, 1)

    def forward(self, features):
        """前向传播。

        Args:
            features: (batch, feature_dim)

        Returns:
            (batch, 1) SOC 预测值
        """
        return self.network(features)
