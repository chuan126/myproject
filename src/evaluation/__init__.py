"""评估子包。

本子包汇集了模型评估所需的所有功能，包括：
- 回归指标计算（metrics.py）
- 评估结果可视化（plots.py）

对外暴露的公共 API 通过 __all__ 明确声明，便于外部模块直接通过
`from src.evaluation import ...` 的方式导入，屏蔽内部模块结构变化。

在整个项目中的角色：
  作为 src 包的子包，为 experiment.py 的训练后评估流程提供指标计算
  和图表生成功能。
"""

from .metrics import regression_metrics
from .plots import (
    save_error_curve,
    save_gate_weights,
    save_prediction_curve,
    save_prediction_scatter,
    save_soc_by_sequence,
    save_soc_curves_by_sequence,
    save_training_curve,
)

# 显式声明本模块的公共接口，便于 IDE 自动补全和静态分析
__all__ = [
    "regression_metrics",
    "save_error_curve",
    "save_gate_weights",
    "save_prediction_curve",
    "save_prediction_scatter",
    "save_soc_by_sequence",
    "save_soc_curves_by_sequence",
    "save_training_curve",
]
