"""优化器注册与构建模块。

本模块提供可插拔的优化器注册表，在整个项目中的角色是：
- 定义统一的优化器注册和构建接口
- 内置常用优化器（Adam、AdamW、SGD）
- 支持通过配置文件名称或结构化字典动态选择优化器及超参数

与 losses.py 采用相同的注册表模式（Registry Pattern），
将优化器的构建逻辑与训练循环解耦。构建器函数签名为
(parameters, config) -> optimizer，其中 config 合并了
训练配置和优化器特定参数。
"""

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn

# 优化器构建器的类型别名：接受模型参数和配置字典，返回优化器实例
OptimizerBuilder = Callable[[Iterable[nn.Parameter], dict[str, Any]], torch.optim.Optimizer]

# 全局注册表：小写名称 → 构建器函数的映射
OPTIMIZER_BUILDERS: dict[str, OptimizerBuilder] = {}


def register_optimizer(name: str, builder: OptimizerBuilder, replace: bool = False) -> None:
    """注册一个优化器构建器到全局注册表。

    与 register_loss 类似，名称统一转为小写存储，支持大小写不敏感的查找。

    Args:
        name: 优化器的名称标识符，不区分大小写
        builder: 构建器函数，签名为 (parameters, config) -> optimizer
        replace: 是否允许覆盖已注册的同名优化器。默认为 False，
            重复注册会抛出 ValueError

    Raises:
        ValueError: 当 name 已存在且 replace=False 时抛出
    """
    normalized_name = name.lower()
    if normalized_name in OPTIMIZER_BUILDERS and not replace:
        raise ValueError(f"Optimizer already registered: {normalized_name}")
    OPTIMIZER_BUILDERS[normalized_name] = builder


# ---- 内置优化器构建函数 ----

# 这些私有函数封装了 PyTorch 原生优化器的构造逻辑，
# 将配置字典中的键映射到优化器构造参数，提供默认值兜底


def _build_adam(parameters: Iterable[nn.Parameter], config: dict[str, Any]) -> torch.optim.Optimizer:
    """构建 Adam 优化器。

    Adam 结合了动量（Momentum）和 RMSProp 的优点，是大多数深度学习
    任务的默认选择。学习率是唯一必需参数，权重衰减默认为 0。

    Args:
        parameters: 模型的可训练参数迭代器
        config: 配置字典，必须包含 'learning_rate'，可选 'weight_decay'

    Returns:
        配置好的 torch.optim.Adam 实例
    """
    return torch.optim.Adam(
        parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


def _build_adamw(parameters: Iterable[nn.Parameter], config: dict[str, Any]) -> torch.optim.Optimizer:
    """构建 AdamW 优化器。

    AdamW 将权重衰减与自适应学习率解耦，相比 Adam 在正则化效果上
    更优，是现代 Transformer 模型训练的首选优化器。

    Args:
        parameters: 模型的可训练参数迭代器
        config: 配置字典，必须包含 'learning_rate'，可选 'weight_decay'

    Returns:
        配置好的 torch.optim.AdamW 实例
    """
    return torch.optim.AdamW(
        parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


def _build_sgd(parameters: Iterable[nn.Parameter], config: dict[str, Any]) -> torch.optim.Optimizer:
    """构建 SGD（随机梯度下降）优化器。

    SGD 是最基础的优化算法，配合动量（momentum）在某些任务上仍有
    竞争力。学习率和动量均可配置，权重衰减默认为 0。

    Args:
        parameters: 模型的可训练参数迭代器
        config: 配置字典，必须包含 'learning_rate'，可选 'momentum' 和 'weight_decay'

    Returns:
        配置好的 torch.optim.SGD 实例
    """
    return torch.optim.SGD(
        parameters,
        lr=float(config["learning_rate"]),
        momentum=float(config.get("momentum", 0.0)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


# ---- 注册内置优化器到全局表 ----

register_optimizer("adam", _build_adam)
register_optimizer("adamw", _build_adamw)
register_optimizer("sgd", _build_sgd)


def build_optimizer(
    config_value: str | dict[str, Any],
    parameters: Iterable[nn.Parameter],
    train_config: dict[str, Any],
) -> torch.optim.Optimizer:
    """根据配置构建优化器。

    支持两种配置格式：
    - 简单字符串：如 "adam"，自动转换为 {"name": "adam"}
    - 结构化字典：如 {"name": "adamw", "weight_decay": 1e-4}

    配置合并逻辑：先将 train_config（训练级别的通用配置，如 learning_rate）
    作为基础，再用 config_value 中的值覆盖同名键。这样设计的原因是
    learning_rate 通常定义在训练配置顶层，而非优化器子配置中，
    合并后保证构建函数能正确读取到所需参数。

    Args:
        config_value: 优化器配置，可以是字符串名称或包含 name 键的字典
        parameters: 模型的可训练参数迭代器，通常通过 model.parameters() 获取
        train_config: 训练配置字典，包含 learning_rate 等通用超参数

    Returns:
        配置好的 PyTorch 优化器实例

    Raises:
        ValueError: 当配置中的名称未在注册表中找到时抛出
    """
    # 统一处理字符串和字典两种输入格式
    values = {"name": config_value} if isinstance(config_value, str) else dict(config_value)
    # 弹出 name 键，避免在合并后重复传递
    name = str(values.pop("name", "adam"))
    # 合并训练配置和优化器特定配置：训练配置为基础，优化器配置覆盖同名键
    config = {**train_config, **values}
    normalized_name = name.lower()
    try:
        builder = OPTIMIZER_BUILDERS[normalized_name]
    except KeyError as error:
        raise ValueError(f"Unknown train.optimizer: {name}") from error
    return builder(parameters, config)
