"""损失函数注册与构建模块。

本模块提供可插拔的回归损失函数注册表，在整个项目中的角色是：
- 定义统一的损失函数注册和构建接口
- 内置常用回归损失（MSE、MAE/L1、Smooth L1）
- 支持通过配置文件名称或结构化字典动态选择损失函数

设计模式采用注册表模式（Registry Pattern），将损失函数的构建逻辑
与使用方解耦。使用者通过字符串名称引用损失函数，无需硬编码类名。
新增自定义损失只需调用 register_loss 即可接入整个训练管线。
"""

from collections.abc import Callable
from inspect import signature
from typing import Any

from torch import nn

# 损失函数构建器的类型别名：接受配置字典，返回 nn.Module 实例
LossBuilder = Callable[..., nn.Module]

# 全局注册表：小写名称 → 构建器函数的映射
# 键统一为小写，使得名称匹配不区分大小写，提升配置文件的容错性
LOSS_BUILDERS: dict[str, LossBuilder] = {}


def register_loss(name: str, builder: LossBuilder, replace: bool = False) -> None:
    """注册一个损失函数构建器到全局注册表。

    自动检测构建器的参数签名，兼容两种形式：
    - 无参数的构建器（如 lambda config: nn.MSELoss()）：包装为忽略 config 参数的 lambda
    - 带参数的构建器：直接使用原始 builder

    名称统一转为小写存储，实现大小写不敏感的注册与查找。

    Args:
        name: 损失函数的名称标识符，不区分大小写
        builder: 构建器函数，签名为 (config: dict) -> nn.Module
        replace: 是否允许覆盖已注册的同名损失函数。默认为 False，
            重复注册同名函数会抛出 ValueError，防止意外覆盖

    Raises:
        ValueError: 当 name 已存在且 replace=False 时抛出
    """
    normalized_name = name.lower()
    if normalized_name in LOSS_BUILDERS and not replace:
        raise ValueError(f"Loss already registered: {normalized_name}")
    # 检查 builder 是否需要 config 参数：参数个数为 0 则包装一层忽略 config 的 lambda
    if len(signature(builder).parameters) == 0:
        LOSS_BUILDERS[normalized_name] = lambda config: builder()
    else:
        LOSS_BUILDERS[normalized_name] = builder


# ---- 内置损失函数注册 ----

# 均方误差损失：回归任务最常用的损失函数，对大误差惩罚更重（平方关系）
register_loss("mse", lambda config: nn.MSELoss())

# 平均绝对误差损失：对异常值更鲁棒，因为误差惩罚是线性的
register_loss("mae", lambda config: nn.L1Loss())

# L1 损失：与 MAE 相同，提供别名便于配置
register_loss("l1", lambda config: nn.L1Loss())

# Smooth L1 损失：结合 L1 和 L2 的优点，在 beta 阈值内使用 L2（平滑），
# 阈值外使用 L1（鲁棒），常用于对异常值敏感度适中的场景
# beta 参数控制平滑区域的大小，默认 1.0
register_loss("smooth_l1", lambda config: nn.SmoothL1Loss(beta=float(config.get("beta", 1.0))))


def build_loss(config_value: str | dict[str, Any]) -> nn.Module:
    """根据配置构建损失函数模块。

    支持两种配置格式：
    - 简单字符串：如 "mse"，自动转换为 {"name": "mse"}
    - 结构化字典：如 {"name": "smooth_l1", "beta": 0.5}

    从注册表中查找对应名称的构建器并调用，未找到时抛出明确的错误信息。

    Args:
        config_value: 损失函数配置，可以是字符串名称或包含 name 键的字典

    Returns:
        构建好的 nn.Module 损失函数实例

    Raises:
        ValueError: 当配置中的名称未在注册表中找到时抛出
    """
    # 统一处理字符串和字典两种输入格式
    config = {"name": config_value} if isinstance(config_value, str) else dict(config_value)
    name = str(config.get("name", "mse"))
    normalized_name = name.lower()
    try:
        builder = LOSS_BUILDERS[normalized_name]
    except KeyError as error:
        raise ValueError(f"Unknown train.loss: {name}") from error
    return builder(config)
