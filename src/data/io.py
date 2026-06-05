"""数据集 IO 模块。

提供统一的数据文件加载接口，支持通过 glob 模式匹配多个 CSV 文件
并合并为一个 DataFrame，同时支持加载可选的 YAML manifest 元数据文件。
加载后通过 schema.validate_canonical_frame 对数据做基本校验。

在整个项目中的角色：
- 位于 data 层，是所有数据进入系统的入口
- 被 dataset.py 中的 _load_frames 调用
- 支持相对路径解析和 glob 通配符，方便组织多文件数据集
"""

from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .schema import validate_canonical_frame


def resolve_pattern(root: Path, pattern: str) -> str:
    """将相对路径模式解析为绝对路径字符串。

    如果 pattern 已经是绝对路径则直接返回，否则拼接 root 作为前缀。
    此函数用于将配置文件中的相对路径转换为绝对路径，避免工作目录依赖。

    Args:
        root: 项目根目录，用作相对路径的基准
        pattern: 文件路径或 glob 模式，可以是相对或绝对路径

    Returns:
        解析后的绝对路径字符串
    """
    path = Path(pattern)
    return str(path if path.is_absolute() else root / path)


def load_canonical_csv(root: Path, data_config: dict[str, Any]) -> pd.DataFrame:
    """加载规范化 CSV 文件并合并为单个 DataFrame。

    支持：
    - 单个文件路径或文件路径列表
    - glob 通配符模式（如 "data/**/*.csv"）
    - 通过 set 去重（同一文件不会被重复加载）
    - 自动按路径排序以保证加载顺序一致

    加载后调用 validate_canonical_frame 进行基本校验，
    保留数据中的所有额外列（超出核心列和特征列的部分不被丢弃）。

    Args:
        root: 项目根目录，用于解析相对路径
        data_config: 数据配置字典，需包含 "path" 键（字符串或字符串列表）

    Returns:
        合并后的 DataFrame，包含所有匹配文件的数据

    Raises:
        FileNotFoundError: 没有匹配到任何 CSV 文件时抛出
        ValueError: 数据校验失败时抛出（由 validate_canonical_frame 触发）
    """
    patterns = data_config["path"]
    values = [patterns] if isinstance(patterns, str) else patterns
    paths = sorted(
        {
            path
            for pattern in values
            for path in glob(resolve_pattern(root, pattern), recursive=True)
        }
    )
    if not paths:
        raise FileNotFoundError(f"No canonical CSV files matched: {data_config['path']}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    validate_canonical_frame(frame, data_config)
    return frame


def load_manifest(root: Path, data_config: dict[str, Any]) -> dict[str, Any] | None:
    """读取配置中指定的数据集 manifest（元数据文件）。

    Manifest 是一个 YAML 文件，包含数据集的元信息（来源、采样率、
    SOC 计算方法等），由数据准备阶段生成。如果配置中未指定 manifest
    路径，则静默返回 None。

    Args:
        root: 项目根目录，用于解析相对路径
        data_config: 数据配置字典，可选包含 "manifest" 键

    Returns:
        解析后的 manifest 字典，如果未配置则返回 None；如果 YAML 文件
        为空则返回空字典 {}

    Raises:
        FileNotFoundError: 配置了 manifest 路径但文件不存在时抛出
    """
    manifest_path = data_config.get("manifest")
    if not manifest_path:
        return None
    path = Path(resolve_pattern(root, manifest_path))
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
