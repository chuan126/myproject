"""特征融合模块。

本模块定义双流架构中用于合并主分支和力学分支特征的融合层。
提供两种融合策略：简单拼接（ConcatFusion）和门控融合（GatedFusion）。
同时提供注册表机制，支持按名称查找并实例化融合层。

在整个项目中的角色：
- 被 registry.py 中的 _build_dual_stream 调用，作为双流模型的融合组件
- 提供可插拔的融合策略，通过配置即可切换不同融合方式
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

# 融合层构建器的类型别名：签名为 (config, main_dim, mech_dim) -> nn.Module
FusionBuilder = Callable[[dict[str, Any], int, int], nn.Module]

# 全局融合层注册表：名称（小写）→ 构建器函数
FUSION_BUILDERS: dict[str, FusionBuilder] = {}


class ConcatFusion(nn.Module):
    """拼接融合层：直接将两个分支的特征向量沿最后一维拼接。

    这是最简单的融合方式，不引入任何可学习参数。
    输出维度为两个输入维度之和。

    使用场景：
    - 当不确定两个分支特征的关系时，保留所有信息供后续层学习
    - 计算开销最小的融合方式

    Args:
        main_dim: 主分支特征维度
        mech_dim: 力学分支特征维度
    """

    def __init__(self, main_dim: int, mech_dim: int):
        super().__init__()
        # 融合后的输出维度为两个分支维度之和
        self.output_dim = main_dim + mech_dim

    def forward(self, main_features: torch.Tensor, mech_features: torch.Tensor) -> torch.Tensor:
        """前向传播：沿最后一维拼接两个输入张量。

        Args:
            main_features: 主分支特征，形状 (batch, main_dim)
            mech_features: 力学分支特征，形状 (batch, mech_dim)

        Returns:
            拼接后的特征，形状 (batch, main_dim + mech_dim)
        """
        return torch.cat([main_features, mech_features], dim=-1)


class GatedFusion(nn.Module):
    """门控融合层：通过可学习的门控权重动态混合两个分支的特征。

    门控机制根据两个分支的拼接特征计算一个逐元素的权重向量 gate，
    然后以 gate * main + (1 - gate) * mech 的方式加权混合。
    这是一种软选择机制，允许模型在不同样本上自适应地调整两个分支的贡献。

    注意事项：
    - 要求两个分支的特征维度必须相等（因为要逐元素加权求和）
    - 每次 forward 后将门控权重保存到 self.last_gate，供外部可视化或分析使用
    - last_gate 只保留最后一次 forward 的结果，不适合并发推理场景

    Args:
        main_dim: 主分支特征维度
        mech_dim: 力学分支特征维度

    Raises:
        ValueError: 当 main_dim != mech_dim 时，门控融合无法直接使用
    """

    def __init__(self, main_dim: int, mech_dim: int):
        super().__init__()
        if main_dim != mech_dim:
            raise ValueError(
                "Gated fusion requires matching branch dimensions: "
                f"main_dim={main_dim}, mech_dim={mech_dim}"
            )
        self.output_dim = main_dim
        # 门控网络：将两个分支特征拼接后，经 Linear + Sigmoid 生成归一化的门控权重
        self.gate = nn.Linear(main_dim + mech_dim, main_dim)
        # 保存最近一次 forward 的 gate 值，供外部访问（如可视化分析）
        self.last_gate: torch.Tensor | None = None

    def forward(self, main_features: torch.Tensor, mech_features: torch.Tensor) -> torch.Tensor:
        """前向传播：计算门控权重并加权混合两个分支的特征。

        Args:
            main_features: 主分支特征，形状 (batch, main_dim)
            mech_features: 力学分支特征，形状 (batch, mech_dim)，必须与 main_dim 等长

        Returns:
            融合后的特征，形状 (batch, output_dim)，output_dim == main_dim

        Side Effect:
            将本次的 gate 张量写入 self.last_gate，覆盖之前的值
        """
        # 将两个分支特征拼接后送入门控网络，经 sigmoid 得到 (0,1) 范围的权重
        gate = torch.sigmoid(self.gate(torch.cat([main_features, mech_features], dim=-1)))
        self.last_gate = gate
        # 加权求和：gate 控制主分支贡献，(1-gate) 控制力学分支贡献
        return gate * main_features + (1.0 - gate) * mech_features


def register_fusion(name: str, builder: FusionBuilder, replace: bool = False) -> None:
    """注册融合层构建器到全局注册表。

    对名称做小写归一化，默认不允许覆盖已注册的构建器。

    Args:
        name: 融合层名称（如 "concat"、"gated"），大小写不敏感
        builder: 融合层构建器函数
        replace: 是否允许覆盖已存在的同名注册，默认 False

    Raises:
        ValueError: 当同名融合层已注册且 replace=False 时
    """
    normalized_name = name.lower()
    if normalized_name in FUSION_BUILDERS and not replace:
        raise ValueError(f"Fusion already registered: {normalized_name}")
    FUSION_BUILDERS[normalized_name] = builder


def build_fusion(config: dict[str, Any], main_dim: int, mech_dim: int) -> nn.Module:
    """根据配置构建融合层。

    从 config 中获取融合层名称（默认为 "concat"），在 FUSION_BUILDERS
    注册表中查找对应的构建器并调用。

    Args:
        config: 融合层配置字典，可包含 "name" 键指定融合方式
        main_dim: 主分支编码器输出维度
        mech_dim: 力学分支编码器输出维度

    Returns:
        实例化后的融合层模块

    Raises:
        ValueError: 当指定的融合层名称未注册时
    """
    name = config.get("name", "concat").lower()
    try:
        builder = FUSION_BUILDERS[name]
    except KeyError as error:
        raise ValueError(f"Unknown model.fusion.name: {name}") from error
    return builder(config, main_dim, mech_dim)


# ── 注册内置融合层 ───────────────────────────────────────────────────────────
# 在模块导入时自动完成注册

# 拼接融合：简单高效，直接将两个分支特征拼接
register_fusion("concat", lambda config, main_dim, mech_dim: ConcatFusion(main_dim, mech_dim))
# 门控融合：可学习权重，动态平衡两个分支的贡献
register_fusion("gated", lambda config, main_dim, mech_dim: GatedFusion(main_dim, mech_dim))
