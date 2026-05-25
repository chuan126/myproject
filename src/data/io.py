"""数据集 IO 模块。

提供统一的数据文件加载接口，支持 CSV 数据集和可选的 manifest 元数据。
"""

from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .schema import validate_canonical_frame


def resolve_pattern(root: Path, pattern: str) -> str:
    """将相对路径模式解析为绝对路径。"""
    path = Path(pattern)
    return str(path if path.is_absolute() else root / path)


def load_canonical_csv(root: Path, data_config: dict[str, Any]) -> pd.DataFrame:
    """加载规范化 CSV 文件。

    保留数据中的任意额外列，仅对核心列做校验。
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

    如果未配置则返回 None。
    """
    manifest_path = data_config.get("manifest")
    if not manifest_path:
        return None
    path = Path(resolve_pattern(root, manifest_path))
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
