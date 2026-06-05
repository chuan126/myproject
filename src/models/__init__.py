"""模型层（models）对外公开接口。

本模块是模型包的入口，集中导出项目中所有可直接使用的公共接口：
- 模型类：SOCModel（单流）、DualStreamSOCModel（双流）
- 构建函数：build_model（顶层入口，根据配置构建完整模型）
- 注册函数：register_encoder / register_head / register_model / register_pooling
           （供外部扩展组件库使用）

使用方式：
    from src.models import build_model
    model = build_model(config, input_dim)

组件注册表、编码器实现、池化层实现等内部细节通过本模块的选择性导出
对外隐藏，外部代码只需导入本模块即可使用所有模型相关功能。
"""

from .dual_stream import DualStreamSOCModel
from .registry import build_model, register_encoder, register_head, register_model, register_pooling
from .soc_model import SOCModel

__all__ = [
    "DualStreamSOCModel",
    "SOCModel",
    "build_model",
    "register_encoder",
    "register_head",
    "register_model",
    "register_pooling",
]
