"""池化层模块对外公开接口。

本模块是池化层子包的入口，统一导出所有可用的序列池化实现：
- LastPooling: 取最后时间步作为序列表示
- MeanPooling: 对所有时间步取均值
- MaxPooling: 对所有时间步取最大值
- AttentionPooling: 通过学习注意力权重对时间步加权平均

所有池化层遵循统一的接口约定：
- __init__(feature_dim) 初始化（其中 AttentionPooling 需要 feature_dim，
  Last/Mean/Max 无需参数）
- forward(x) 接受 (batch, seq_len, feature_dim)，返回 (batch, feature_dim)

池化层的作用是将编码器输出的时序特征压缩为固定长度的向量表示，
消除序列长度差异，供后续的融合层或预测头使用。
"""

from .attention import AttentionPooling
from .last import LastPooling
from .max import MaxPooling
from .mean import MeanPooling

__all__ = ["AttentionPooling", "LastPooling", "MaxPooling", "MeanPooling"]
