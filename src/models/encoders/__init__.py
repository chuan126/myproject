"""编码器模块对外公开接口。

本模块是编码器子包的入口，统一导出所有可用的时序编码器实现：
- CNNEncoder: 一维卷积编码器
- FCNEncoder: 全连接网络编码器
- GRUEncoder: 门控循环单元编码器
- InformerEncoder: Informer Transformer 编码器（长序列专用）
- LSTMEncoder: 长短期记忆网络编码器
- TCNEncoder: 时序卷积网络编码器

所有编码器遵循统一的接口约定：
- __init__(input_dim, hidden_size, num_layers, ...) 初始化
- output_dim 属性：编码后的特征维度
- forward(x) 接受 (batch, seq_len, input_dim)，返回 (batch, seq_len, output_dim)

这些编码器被 registry.py 中的构建器工厂引用，通过注册表模式实现可插拔切换。
"""

from .cnn import CNNEncoder
from .fcn import FCNEncoder
from .gru import GRUEncoder
from .informer import InformerEncoder
from .lstm import LSTMEncoder
from .tcn import TCNEncoder

__all__ = ["CNNEncoder", "FCNEncoder", "GRUEncoder", "InformerEncoder", "LSTMEncoder", "TCNEncoder"]
