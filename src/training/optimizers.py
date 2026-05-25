"""优化器注册表。

提供可插拔的优化器构建器，支持按名称和配置构建。
"""

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn

OptimizerBuilder = Callable[[Iterable[nn.Parameter], dict[str, Any]], torch.optim.Optimizer]
OPTIMIZER_BUILDERS: dict[str, OptimizerBuilder] = {}


def register_optimizer(name: str, builder: OptimizerBuilder, replace: bool = False) -> None:
    """注册优化器构建器，签名为 ``(parameters, config) -> optimizer``。"""
    normalized_name = name.lower()
    if normalized_name in OPTIMIZER_BUILDERS and not replace:
        raise ValueError(f"Optimizer already registered: {normalized_name}")
    OPTIMIZER_BUILDERS[normalized_name] = builder


def _build_adam(parameters: Iterable[nn.Parameter], config: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


def _build_adamw(parameters: Iterable[nn.Parameter], config: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


def _build_sgd(parameters: Iterable[nn.Parameter], config: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        parameters,
        lr=float(config["learning_rate"]),
        momentum=float(config.get("momentum", 0.0)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


register_optimizer("adam", _build_adam)
register_optimizer("adamw", _build_adamw)
register_optimizer("sgd", _build_sgd)


def build_optimizer(
    config_value: str | dict[str, Any],
    parameters: Iterable[nn.Parameter],
    train_config: dict[str, Any],
) -> torch.optim.Optimizer:
    """按字符串名称或结构化配置构建已注册的优化器。"""
    values = {"name": config_value} if isinstance(config_value, str) else dict(config_value)
    name = str(values.pop("name", "adam"))
    config = {**train_config, **values}
    normalized_name = name.lower()
    try:
        builder = OPTIMIZER_BUILDERS[normalized_name]
    except KeyError as error:
        raise ValueError(f"Unknown train.optimizer: {name}") from error
    return builder(parameters, config)
