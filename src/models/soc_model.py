"""SOC 估计主模型。

本模块定义 SOCModel 类，它本身不做计算，而是作为容器组装三个子模块
形成完整的 Encoder → Pooling → Head 流水线。这种设计遵循"组合优于继承"
的原则，允许编码器、池化层和预测头独立开发和测试。

在整个项目中的角色：
- 是单流架构的模型载体，被 registry.py 中的 _build_encoder_pooling_head 组装
- forward 方法串联编码→池化→预测的数据流，对外表现为一个完整的端到端模型
"""

from torch import nn


class SOCModel(nn.Module):
    """SOC 估计模型：Encoder → Pooling → Head。

    这是一个组合式模型容器，将时序编码器、序列池化层和回归头串联起来。
    数据流为：输入(batch, seq_len, input_dim) → encoder → (batch, seq_len, feature_dim)
    → pooling → (batch, feature_dim) → head → (batch, 1)。

    设计意图：
    - 解耦编码、池化、预测三个环节，每个环节可由不同实现替换
    - 模型本身不包含任何可学习参数，所有参数由子模块提供
    - 通过构造函数注入子模块，支持依赖注入风格的对象组装

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
        """前向传播：依次执行编码、池化、预测。

        Args:
            inputs: 输入张量，形状 (batch, seq_len, input_dim)

        Returns:
            SOC 预测值，形状 (batch, 1)
        """
        encoded = self.encoder(inputs)
        pooled = self.pooling(encoded)
        return self.head(pooled)
