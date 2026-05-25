"""配置加载与合并模块。

支持 YAML 配置文件读取、深度合并和写入。
"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """深度合并两个字典。

    嵌套字典递归合并，其余键直接覆盖。

    Args:
        base: 基础配置字典
        override: 覆盖配置字典

    Returns:
        合并后的新字典
    """
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def read_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 文件为字典。

    Args:
        path: YAML 文件路径

    Returns:
        解析后的字典（空文件返回空字典）
    """
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _read_with_extends(path: Path) -> dict[str, Any]:
    """解析 ``extends`` 继承链后再应用本地覆盖。"""
    values = read_yaml(path)
    extends = values.pop("extends", [])
    if isinstance(extends, str):
        extends = [extends]
    if not isinstance(extends, list) or not all(isinstance(item, str) for item in extends):
        raise ValueError(f"extends must be a path or list of paths in: {path}")
    resolved: dict[str, Any] = {}
    for item in extends:
        resolved = deep_merge(resolved, _read_with_extends((path.parent / item).resolve()))
    return deep_merge(resolved, values)


def load_config(
    base_path: Path,
    experiment_path: Path | None = None,
) -> dict[str, Any]:
    """加载并合并配置。

    先加载基础配置，再用实验配置深度覆盖。

    Args:
        base_path: 基础 YAML 配置文件路径
        experiment_path: 实验特定 YAML 配置文件路径（可选）
    Returns:
        合并后的完整配置字典
    """
    config = _read_with_extends(base_path)
    if experiment_path is not None:
        config = deep_merge(config, _read_with_extends(experiment_path))
    return config


def write_yaml(data: dict[str, Any], path: Path) -> None:
    """将字典写入 YAML 文件。

    Args:
        data: 配置字典
        path: 输出文件路径
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)
