"""模型组件注册表与构建器。

通过注册表模式实现编码器、池化、头部和完整模型的可插拔组合。
"""

from collections.abc import Callable
from typing import Any

from torch import nn

from .encoders import CNNEncoder, FCNEncoder, GRUEncoder, LSTMEncoder, TCNEncoder
from .head import RegressionHead
from .pooling import AttentionPooling, LastPooling, MaxPooling, MeanPooling
from .soc_model import SOCModel

EncoderBuilder = Callable[[dict[str, Any], int], nn.Module]
PoolingBuilder = Callable[[int], nn.Module]
HeadBuilder = Callable[[dict[str, Any], int], nn.Module]
ModelBuilder = Callable[[dict[str, Any], int], nn.Module]

ENCODER_BUILDERS: dict[str, EncoderBuilder] = {}
POOLING_BUILDERS: dict[str, PoolingBuilder] = {}
HEAD_BUILDERS: dict[str, HeadBuilder] = {}
MODEL_BUILDERS: dict[str, ModelBuilder] = {}


def _register(registry: dict[str, Any], name: str, builder: Any, replace: bool) -> None:
    normalized_name = name.lower()
    if normalized_name in registry and not replace:
        raise ValueError(f"Component already registered: {normalized_name}")
    registry[normalized_name] = builder


def register_encoder(name: str, builder: EncoderBuilder, replace: bool = False) -> None:
    """注册编码器构建器，签名为 ``(config, input_dim) -> module``。"""
    _register(ENCODER_BUILDERS, name, builder, replace)


def register_pooling(name: str, builder: PoolingBuilder, replace: bool = False) -> None:
    """注册池化层构建器，签名为 ``(feature_dim) -> module``。"""
    _register(POOLING_BUILDERS, name, builder, replace)


def register_head(name: str, builder: HeadBuilder, replace: bool = False) -> None:
    """注册回归头构建器，签名为 ``(config, feature_dim) -> module``。"""
    _register(HEAD_BUILDERS, name, builder, replace)


def register_model(name: str, builder: ModelBuilder, replace: bool = False) -> None:
    """注册完整模型构建器，签名为 ``(config, input_dim) -> module``。"""
    _register(MODEL_BUILDERS, name, builder, replace)


def _recurrent_builder(encoder_type: type[nn.Module]) -> EncoderBuilder:
    def build(config: dict[str, Any], input_dim: int) -> nn.Module:
        return encoder_type(
            input_dim,
            int(config.get("hidden_size", 64)),
            int(config.get("num_layers", 1)),
            float(config.get("dropout", 0.0)),
        )

    return build


def _convolutional_builder(encoder_type: type[nn.Module]) -> EncoderBuilder:
    def build(config: dict[str, Any], input_dim: int) -> nn.Module:
        return encoder_type(
            input_dim,
            int(config.get("hidden_size", 64)),
            int(config.get("num_layers", 1)),
            int(config.get("kernel_size", 3)),
            float(config.get("dropout", 0.0)),
        )

    return build


register_encoder("lstm", _recurrent_builder(LSTMEncoder))
register_encoder("gru", _recurrent_builder(GRUEncoder))
register_encoder("fcn", _recurrent_builder(FCNEncoder))
register_encoder("tcn", _convolutional_builder(TCNEncoder))
register_encoder("cnn", _convolutional_builder(CNNEncoder))

register_pooling("last", lambda feature_dim: LastPooling())
register_pooling("mean", lambda feature_dim: MeanPooling())
register_pooling("max", lambda feature_dim: MaxPooling())
register_pooling("attention", lambda feature_dim: AttentionPooling(feature_dim))


def _regression_head_builder(config: dict[str, Any], feature_dim: int) -> nn.Module:
    return RegressionHead(
        feature_dim,
        hidden_size=config.get("hidden_size"),
        dropout=float(config.get("dropout", 0.0)),
    )


register_head("regression", _regression_head_builder)


def build_encoder(config: dict[str, Any], input_dim: int) -> nn.Module:
    """根据配置条目构建编码器。"""
    name = config["name"].lower()
    try:
        builder = ENCODER_BUILDERS[name]
    except KeyError as error:
        raise ValueError(f"Unknown model.name: {name}") from error
    return builder(config, input_dim)


def build_pooling(name: str, feature_dim: int) -> nn.Module:
    """根据配置条目构建池化层。"""
    normalized_name = name.lower()
    try:
        builder = POOLING_BUILDERS[normalized_name]
    except KeyError as error:
        raise ValueError(f"Unknown model.pooling.name: {normalized_name}") from error
    return builder(feature_dim)


def build_head(config: dict[str, Any], feature_dim: int) -> nn.Module:
    """根据配置条目构建预测头。"""
    name = config.get("name", "regression").lower()
    try:
        builder = HEAD_BUILDERS[name]
    except KeyError as error:
        raise ValueError(f"Unknown model.head.name: {name}") from error
    return builder(config, feature_dim)


def _build_encoder_pooling_head(config: dict[str, Any], input_dim: int) -> SOCModel:
    """构建 Encoder → Pooling → Head 完整 SOC 模型。"""
    encoder = build_encoder(config, input_dim)
    pooling = build_pooling(config.get("pooling", {}).get("name", "last"), encoder.output_dim)
    head = build_head(config.get("head", {}), encoder.output_dim)
    return SOCModel(encoder, pooling, head)


register_model("encoder_pooling_head", _build_encoder_pooling_head)


def build_model(config: dict[str, Any], input_dim: int) -> nn.Module:
    """根据配置构建完整的 SOC 模型架构。"""
    name = config.get("architecture", {}).get("name", "encoder_pooling_head").lower()
    try:
        builder = MODEL_BUILDERS[name]
    except KeyError as error:
        raise ValueError(f"Unknown model.architecture.name: {name}") from error
    return builder(config, input_dim)
