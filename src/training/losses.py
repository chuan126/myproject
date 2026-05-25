"""损失函数注册表。

提供可插拔的回归损失函数，支持按名称构建。
"""

from collections.abc import Callable
from inspect import signature
from typing import Any

from torch import nn

LossBuilder = Callable[..., nn.Module]
LOSS_BUILDERS: dict[str, LossBuilder] = {}


def register_loss(name: str, builder: LossBuilder, replace: bool = False) -> None:
    """注册损失函数构建器，兼容 ``()`` 与 ``(config)`` 两种签名。"""
    normalized_name = name.lower()
    if normalized_name in LOSS_BUILDERS and not replace:
        raise ValueError(f"Loss already registered: {normalized_name}")
    if len(signature(builder).parameters) == 0:
        LOSS_BUILDERS[normalized_name] = lambda config: builder()
    else:
        LOSS_BUILDERS[normalized_name] = builder


register_loss("mse", lambda config: nn.MSELoss())
register_loss("mae", lambda config: nn.L1Loss())
register_loss("l1", lambda config: nn.L1Loss())
register_loss("smooth_l1", lambda config: nn.SmoothL1Loss(beta=float(config.get("beta", 1.0))))


def build_loss(config_value: str | dict[str, Any]) -> nn.Module:
    """按字符串名称或结构化配置构建已注册的损失函数。"""
    config = {"name": config_value} if isinstance(config_value, str) else dict(config_value)
    name = str(config.get("name", "mse"))
    normalized_name = name.lower()
    try:
        builder = LOSS_BUILDERS[normalized_name]
    except KeyError as error:
        raise ValueError(f"Unknown train.loss: {name}") from error
    return builder(config)
