"""模型组件注册表与构建器。

本模块是模型层的核心组装模块，采用注册表模式（Registry Pattern）实现编码器（Encoder）、
池化层（Pooling）、预测头（Head）以及完整模型的可插拔组合。每种组件类型维护一个独立的
构建器字典，通过名称字符串查找对应的工厂函数来实例化组件。

在整个项目中的角色：
- 作为模型构建的入口，提供 `build_model()` 供 `src/experiment.py` 调用
- 集中管理所有可用的编码器/池化/头部组件的注册和查询
- 通过 `_build_encoder_pooling_head` 和 `_build_dual_stream` 实现两种预定义的
  模型架构（单流和双流）的组装逻辑
- 支持外部通过 `register_encoder` 等函数动态扩展组件库而不修改核心代码
"""

from collections.abc import Callable
from typing import Any

from torch import nn

from .dual_stream import DualStreamSOCModel
from .encoders import CNNEncoder, FCNEncoder, GRUEncoder, InformerEncoder, LSTMEncoder, TCNEncoder
from .fusion import build_fusion
from .head import RegressionHead
from .pooling import AttentionPooling, LastPooling, MaxPooling, MeanPooling
from .soc_model import SOCModel

# ── 类型别名：各组件构建器的函数签名 ─────────────────────────────────────────
# 编码器构建器：接收配置字典和输入维度，返回 nn.Module
EncoderBuilder = Callable[[dict[str, Any], int], nn.Module]
# 池化层构建器：只需特征维度，返回 nn.Module（池化层通常无额外配置）
PoolingBuilder = Callable[[int], nn.Module]
# 预测头构建器：接收配置字典和特征维度，返回 nn.Module
HeadBuilder = Callable[[dict[str, Any], int], nn.Module]
# 完整模型构建器：接收配置字典和输入维度，返回 nn.Module
ModelBuilder = Callable[[dict[str, Any], int], nn.Module]

# ── 组件注册表：名称（小写）→ 构建器函数 ─────────────────────────────────────
# 所有组件名在注册时被归一化为小写，确保查找时不区分大小写
ENCODER_BUILDERS: dict[str, EncoderBuilder] = {}
POOLING_BUILDERS: dict[str, PoolingBuilder] = {}
HEAD_BUILDERS: dict[str, HeadBuilder] = {}
MODEL_BUILDERS: dict[str, ModelBuilder] = {}


def _register(registry: dict[str, Any], name: str, builder: Any, replace: bool) -> None:
    """通用注册辅助函数，将构建器注册到指定的注册表中。

    对名称做小写归一化处理，确保"LSTM"和"lstm"指向同一个构建器。
    默认不允许覆盖已注册的组件，除非显式设置 replace=True。

    Args:
        registry: 目标注册表字典
        name: 组件名称（大小写不敏感）
        builder: 构建器函数或可调用对象
        replace: 是否允许覆盖已存在的同名注册

    Raises:
        ValueError: 当同名组件已注册且 replace=False 时抛出
    """
    normalized_name = name.lower()
    if normalized_name in registry and not replace:
        raise ValueError(f"Component already registered: {normalized_name}")
    registry[normalized_name] = builder


def register_encoder(name: str, builder: EncoderBuilder, replace: bool = False) -> None:
    """注册编码器构建器。

    编码器构建器的签名为 (config, input_dim) -> module，其中 config 是包含
    超参数（如 hidden_size、num_layers、dropout 等）的字典。

    Args:
        name: 编码器名称（如 "lstm"、"informer"）
        builder: 编码器构建器函数
        replace: 是否允许覆盖已存在的同名注册，默认 False
    """
    _register(ENCODER_BUILDERS, name, builder, replace)


def register_pooling(name: str, builder: PoolingBuilder, replace: bool = False) -> None:
    """注册池化层构建器。

    池化层构建器的签名为 (feature_dim) -> module，因为池化层通常不需要
    额外的超参数配置，只需知道特征维度即可实例化。

    Args:
        name: 池化层名称（如 "mean"、"attention"）
        builder: 池化层构建器函数
        replace: 是否允许覆盖已存在的同名注册，默认 False
    """
    _register(POOLING_BUILDERS, name, builder, replace)


def register_head(name: str, builder: HeadBuilder, replace: bool = False) -> None:
    """注册回归头构建器。

    预测头构建器的签名为 (config, feature_dim) -> module，其中 config 可以是
    包含 hidden_size、dropout 等超参数的字典。

    Args:
        name: 头部名称（如 "regression"）
        builder: 头部构建器函数
        replace: 是否允许覆盖已存在的同名注册，默认 False
    """
    _register(HEAD_BUILDERS, name, builder, replace)


def register_model(name: str, builder: ModelBuilder, replace: bool = False) -> None:
    """注册完整模型构建器。

    模型构建器的签名为 (config, input_dim) -> module，它是最高层级的组装器，
    内部会调用编码器、池化、头部等子组件的构建器来组装完整模型。

    Args:
        name: 模型架构名称（如 "encoder_pooling_head"、"dual_stream"）
        builder: 模型构建器函数
        replace: 是否允许覆盖已存在的同名注册，默认 False
    """
    _register(MODEL_BUILDERS, name, builder, replace)


# ── 编码器构建器工厂 ─────────────────────────────────────────────────────────
# 以下工厂函数为不同编码器类型提供参数提取和默认值填充逻辑

def _recurrent_builder(encoder_type: type[nn.Module]) -> EncoderBuilder:
    """循环神经网络（RNN 类）编码器的通用构建器工厂。

    适用于 LSTM、GRU、FCN（全连接网络也复用同一参数集）等编码器类型。
    从 config 中提取 hidden_size、num_layers、dropout，并应用默认值。

    Args:
        encoder_type: 具体的编码器类（如 LSTMEncoder、GRUEncoder）

    Returns:
        一个闭包函数，签名为 (config, input_dim) -> nn.Module
    """
    def build(config: dict[str, Any], input_dim: int) -> nn.Module:
        return encoder_type(
            input_dim,
            int(config.get("hidden_size", 64)),
            int(config.get("num_layers", 1)),
            float(config.get("dropout", 0.0)),
        )

    return build


def _convolutional_builder(encoder_type: type[nn.Module]) -> EncoderBuilder:
    """卷积神经网络编码器的通用构建器工厂。

    适用于 TCN、CNN 等卷积编码器类型。与循环网络构建器相比，额外提取
    kernel_size 参数，控制卷积核大小。

    Args:
        encoder_type: 具体的编码器类（如 TCNEncoder、CNNEncoder）

    Returns:
        一个闭包函数，签名为 (config, input_dim) -> nn.Module
    """
    def build(config: dict[str, Any], input_dim: int) -> nn.Module:
        return encoder_type(
            input_dim,
            int(config.get("hidden_size", 64)),
            int(config.get("num_layers", 1)),
            int(config.get("kernel_size", 3)),
            float(config.get("dropout", 0.0)),
        )

    return build


def _informer_builder(config: dict[str, Any], input_dim: int) -> nn.Module:
    """Informer 编码器的独立构建器。

    Informer 是专为长序列时间序列预测设计的 Transformer 变体，其参数集
    与循环/卷积网络差异较大，因此使用独立的构建器而非工厂模式。
    需要提取 attention 类型、n_heads、d_ff、distil 等特有参数。

    Args:
        config: 包含 Informer 超参数的配置字典
        input_dim: 输入特征维度

    Returns:
        实例化后的 InformerEncoder
    """
    return InformerEncoder(
        input_dim,
        int(config.get("hidden_size", 64)),
        int(config.get("num_layers", 1)),
        int(config.get("n_heads", 4)),
        # d_ff 默认为 hidden_size 的 4 倍，这是 Transformer 的常见设置
        int(config.get("d_ff", int(config.get("hidden_size", 64)) * 4)),
        float(config.get("dropout", 0.0)),
        str(config.get("attention", "prob")).lower(),
        int(config.get("factor", 5)),
        bool(config.get("distil", False)),
        str(config.get("activation", "gelu")).lower(),
    )


# ── 注册所有内置编码器 ───────────────────────────────────────────────────────
# 在模块导入时自动完成注册，外部代码导入本模块即可直接使用这些编码器

register_encoder("lstm", _recurrent_builder(LSTMEncoder))
register_encoder("gru", _recurrent_builder(GRUEncoder))
register_encoder("fcn", _recurrent_builder(FCNEncoder))
register_encoder("tcn", _convolutional_builder(TCNEncoder))
register_encoder("cnn", _convolutional_builder(CNNEncoder))
register_encoder("informer", _informer_builder)

# ── 注册所有内置池化层 ───────────────────────────────────────────────────────
# 池化层不需要额外配置参数，使用 lambda 直接实例化
# 注意：AttentionPooling 需要 feature_dim 来初始化内部的可学习参数
register_pooling("last", lambda feature_dim: LastPooling())
register_pooling("mean", lambda feature_dim: MeanPooling())
register_pooling("max", lambda feature_dim: MaxPooling())
register_pooling("attention", lambda feature_dim: AttentionPooling(feature_dim))


def _regression_head_builder(config: dict[str, Any], feature_dim: int) -> nn.Module:
    """回归头构建器。

    从 config 中提取 hidden_size 和 dropout 参数。
    hidden_size 为 None 或不存在时，RegressionHead 内部会退化为单层线性层。
    dropout 默认为 0.0（不使用 Dropout）。

    Args:
        config: 包含 head 配置的字典
        feature_dim: 编码器输出的特征维度

    Returns:
        RegressionHead 实例
    """
    return RegressionHead(
        feature_dim,
        hidden_size=config.get("hidden_size"),
        dropout=float(config.get("dropout", 0.0)),
    )


register_head("regression", _regression_head_builder)


# ── 组件构建入口函数 ─────────────────────────────────────────────────────────

def build_encoder(config: dict[str, Any], input_dim: int) -> nn.Module:
    """根据配置字典构建编码器。

    从 config["name"] 获取编码器名称，在 ENCODER_BUILDERS 注册表中查找
    对应的构建器并调用。名称不区分大小写。

    Args:
        config: 包含 "name" 键的编码器配置字典
        input_dim: 输入特征维度，传递给构建器

    Returns:
        实例化后的编码器模块

    Raises:
        ValueError: 当 config["name"] 指定的编码器未注册时
    """
    name = config["name"].lower()
    try:
        builder = ENCODER_BUILDERS[name]
    except KeyError as error:
        raise ValueError(f"Unknown model.name: {name}") from error
    return builder(config, input_dim)


def build_pooling(name: str, feature_dim: int) -> nn.Module:
    """根据名称构建池化层。

    在 POOLING_BUILDERS 注册表中查找对应的构建器并调用。
    名称不区分大小写。

    Args:
        name: 池化层名称（如 "mean"、"attention"）
        feature_dim: 编码器输出的特征维度

    Returns:
        实例化后的池化层模块

    Raises:
        ValueError: 当指定的池化层名称未注册时
    """
    normalized_name = name.lower()
    try:
        builder = POOLING_BUILDERS[normalized_name]
    except KeyError as error:
        raise ValueError(f"Unknown model.pooling.name: {normalized_name}") from error
    return builder(feature_dim)


def build_head(config: dict[str, Any], feature_dim: int) -> nn.Module:
    """根据配置构建预测头。

    从 config 中获取头部名称（默认为 "regression"），在 HEAD_BUILDERS 注册表中
    查找并调用对应的构建器。

    Args:
        config: 包含 "name" 键的头部配置字典，name 缺失时默认用 "regression"
        feature_dim: 编码器/融合层输出的特征维度

    Returns:
        实例化后的预测头模块

    Raises:
        ValueError: 当指定的头部名称未注册时
    """
    name = config.get("name", "regression").lower()
    try:
        builder = HEAD_BUILDERS[name]
    except KeyError as error:
        raise ValueError(f"Unknown model.head.name: {name}") from error
    return builder(config, feature_dim)


# ── 预定义的模型架构组装器 ────────────────────────────────────────────────────

def _build_encoder_pooling_head(config: dict[str, Any], input_dim: int) -> SOCModel:
    """构建标准的 Encoder → Pooling → Head 单流 SOC 模型。

    这是最基础也是默认的模型架构：输入数据经过编码器提取时序特征，
    再通过池化层压缩时间维度，最后由回归头输出 SOC 预测值。
    所有配置（编码器类型、池化方式、头部超参数）均从 config 字典中提取。

    Args:
        config: 完整的模型配置字典，应包含 encoder、pooling、head 子配置
        input_dim: 原始输入数据的特征维度

    Returns:
        已组装完整的 SOCModel 实例
    """
    encoder = build_encoder(config, input_dim)
    # pooling.name 缺省时使用 "last"，即取最后时间步作为序列表示
    pooling = build_pooling(config.get("pooling", {}).get("name", "last"), encoder.output_dim)
    # head.name 缺省时使用 "regression"（在 build_head 内部处理）
    head = build_head(config.get("head", {}), encoder.output_dim)
    return SOCModel(encoder, pooling, head)


register_model("encoder_pooling_head", _build_encoder_pooling_head)


def _feature_indices(all_features: list[str], branch_features: list[str], branch_name: str) -> list[int]:
    """将特征列名映射为特征索引。

    在双流架构中，需要从完整特征列表中找出属于某个分支（主分支或力学分支）
    的特征列的索引，以便在前向传播时用切片选取对应数据。

    Args:
        all_features: 全部特征的列名列表（按输入数据的列顺序排列）
        branch_features: 当前分支要使用的特征列名列表
        branch_name: 分支名称（用于错误消息），如 "main" 或 "mech"

    Returns:
        整数索引列表，指示 branch_features 在 all_features 中的位置

    Raises:
        ValueError: 当 branch_features 为空或包含未在 all_features 中出现的列名时
    """
    if not branch_features:
        raise ValueError(f"{branch_name} branch must define at least one feature column.")

    # 构造特征名到索引的映射
    index_by_feature = {feature: index for index, feature in enumerate(all_features)}
    indices = []
    for feature in branch_features:
        try:
            indices.append(index_by_feature[feature])
        except KeyError as error:
            raise ValueError(f"Unknown {branch_name} branch feature column: {feature}") from error
    return indices


def _require_feature_columns(config: dict[str, Any], input_dim: int) -> list[str]:
    """验证并返回完整特征列名列表。

    双流架构强制要求配置中提供 feature_columns，用于在 forward 时按列索引切片。
    此函数执行三项校验：feature_columns 存在性、长度与 input_dim 一致性、
    无重复列名。

    Args:
        config: 包含 "feature_columns" 键的配置字典
        input_dim: 输入数据的特征维度

    Returns:
        特征列名列表（转成 list 以确保可重复索引）

    Raises:
        ValueError: 校验不通过时抛出，包含具体的错误描述
    """
    all_features = config.get("feature_columns")
    if not all_features:
        raise ValueError("model.feature_columns must be provided for dual_stream architecture.")
    if len(all_features) != input_dim:
        raise ValueError(
            "model.feature_columns length must match input_dim: "
            f"len(feature_columns)={len(all_features)}, input_dim={input_dim}"
        )
    if len(set(all_features)) != len(all_features):
        raise ValueError("model.feature_columns must not contain duplicates.")
    return list(all_features)


def _build_branch(branch_config: dict[str, Any], input_dim: int, branch_name: str) -> tuple[nn.Module, nn.Module]:
    """构建双流架构中单个分支的编码器和池化层。

    每个分支独立处理一组特征列，有自己的编码器和池化层配置。
    该函数负责从分支配置中提取 encoder 和 pooling 子配置并实例化。

    Args:
        branch_config: 分支的配置字典，应包含 "encoder" 键和可选的 "pooling" 键
        input_dim: 该分支的输入特征维度（已按分支特征列数量确定）
        branch_name: 分支名称，用于错误消息

    Returns:
        (encoder, pooling) 元组

    Raises:
        ValueError: 当分支配置缺少 "encoder" 键时
    """
    if "encoder" not in branch_config:
        raise ValueError(f"model.{branch_name}_branch.encoder must be provided for dual_stream architecture.")
    encoder = build_encoder(branch_config["encoder"], input_dim)
    pooling = build_pooling(branch_config.get("pooling", {}).get("name", "last"), encoder.output_dim)
    return encoder, pooling


def _build_dual_stream(config: dict[str, Any], input_dim: int) -> DualStreamSOCModel:
    """构建双流（Dual Stream）SOC 模型。

    双流架构将输入特征分为两组——主分支（main_branch）和力学分支（mech_branch）——
    分别通过各自的编码器和池化层独立处理，然后通过融合层（fusion）组合两路特征，
    最后由预测头输出 SOC 估计值。

    此架构适用于需要显式分离不同性质特征（如电学特征 vs 力学特征）的场景，
    允许每个分支使用不同的编码器类型和超参数。

    Args:
        config: 完整的模型配置字典，必须包含 feature_columns、main_branch、mech_branch
        input_dim: 原始输入数据的特征维度

    Returns:
        已组装完整的 DualStreamSOCModel 实例

    Raises:
        ValueError: 当缺少必要配置项时
    """
    all_features = _require_feature_columns(config, input_dim)

    try:
        main_branch = config["main_branch"]
        mech_branch = config["mech_branch"]
    except KeyError as error:
        raise ValueError(f"model.{error.args[0]} must be provided for dual_stream architecture.") from error

    # 计算各分支特征列在完整特征列表中的索引
    main_indices = _feature_indices(all_features, main_branch.get("feature_columns", []), "main")
    mech_indices = _feature_indices(all_features, mech_branch.get("feature_columns", []), "mech")

    # 构建各分支的编码器和池化层
    main_encoder, main_pooling = _build_branch(main_branch, len(main_indices), "main")
    mech_encoder, mech_pooling = _build_branch(mech_branch, len(mech_indices), "mech")

    # 融合层根据两个分支的输出维度构建
    fusion = build_fusion(config.get("fusion", {}), main_encoder.output_dim, mech_encoder.output_dim)
    # 预测头接收融合后的特征
    head = build_head(config.get("head", {}), fusion.output_dim)

    return DualStreamSOCModel(
        main_indices,
        mech_indices,
        main_encoder,
        main_pooling,
        mech_encoder,
        mech_pooling,
        fusion,
        head,
    )


register_model("dual_stream", _build_dual_stream)


def build_model(config: dict[str, Any], input_dim: int) -> nn.Module:
    """模型构建的顶层入口函数。

    从 config 中获取模型架构名称（默认为 "encoder_pooling_head"），
    在 MODEL_BUILDERS 注册表中查找对应的构建器并调用。

    这是整个模型层对外暴露的主要接口，被 src/experiment.py 调用
    以根据 YAML 配置文件实例化完整的模型。

    Args:
        config: 完整的模型配置字典，其中 model.architecture.name 决定使用哪种架构
        input_dim: 输入数据的特征维度

    Returns:
        已组装完成的 nn.Module 模型实例

    Raises:
        ValueError: 当 model.architecture.name 指定的架构未注册时
    """
    name = config.get("architecture", {}).get("name", "encoder_pooling_head").lower()
    try:
        builder = MODEL_BUILDERS[name]
    except KeyError as error:
        raise ValueError(f"Unknown model.architecture.name: {name}") from error
    return builder(config, input_dim)
