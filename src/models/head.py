"""回归头模块。

本模块定义 RegressionHead 类，负责将编码器和池化层提取的特征向量映射为
单个 SOC（State of Charge）预测值。这是模型的最终输出层。

在整个项目中的角色：
- 作为模型的预测输出端，接收池化层或融合层的特征向量
- 支持可选的一层隐藏层 + ReLU + Dropout，提供非线性变换能力
- hidden_size 为 None 或 0 时退化为纯线性层，节省参数和计算
"""

from torch import nn


class RegressionHead(nn.Module):
    """回归头：将特征向量映射为 SOC 预测值。

    支持两种模式：
    - 有隐藏层模式：Linear → ReLU → Dropout → Linear(→ 1)，提供非线性能力
    - 无隐藏层模式（退化）：单层 Linear → 1，简洁高效

    使用场景：
    - 作为单流模型的最后的预测头
    - 作为双流模型中融合层之后的预测头
    - 可通过配置控制是否使用隐藏层及 Dropout 率

    Args:
        input_dim: 输入特征维度，来自上游编码器或融合层的 output_dim
        hidden_size: 中间隐藏层大小。None 或 0 时退化为单层 Linear。
            默认 None。
        dropout: Dropout 概率，仅在启用隐藏层时生效。默认 0.0。
    """

    def __init__(self, input_dim: int, hidden_size: int | None = None, dropout: float = 0.0):
        super().__init__()
        # 当 hidden_size 有效（非 None 且 > 0）时，构建两层网络
        if hidden_size is not None and hidden_size > 0:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
        else:
            # 退化为单层线性层，直接将特征映射到标量预测值
            self.network = nn.Linear(input_dim, 1)

    def forward(self, features):
        """前向传播：特征向量 → SOC 预测值。

        Args:
            features: 输入特征，形状 (batch, feature_dim)

        Returns:
            SOC 预测值，形状 (batch, 1)
        """
        return self.network(features)
