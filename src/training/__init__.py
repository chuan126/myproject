"""训练子包。

本包汇聚了模型训练流程的所有核心组件：

- trainer: 训练循环编排（Trainer 类、TrainingResult 数据类、predict 推理函数）
- losses: 损失函数注册与构建（MSE、MAE、Smooth L1 等）
- optimizers: 优化器注册与构建（Adam、AdamW、SGD 等）
- checkpoint: 模型检查点的保存与加载

通过 __init__.py 统一导出公共 API，外部模块只需
`from src.training import ...` 即可访问所有训练相关功能。
"""

from .checkpoint import load_checkpoint, save_checkpoint
from .losses import build_loss, register_loss
from .optimizers import build_optimizer, register_optimizer
from .trainer import Trainer, TrainingResult, predict

# 显式定义公共 API，控制导出范围，避免内部实现细节被意外暴露
__all__ = [
    "Trainer",
    "TrainingResult",
    "build_loss",
    "build_optimizer",
    "load_checkpoint",
    "predict",
    "register_loss",
    "register_optimizer",
    "save_checkpoint",
]
