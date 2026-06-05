"""配置加载与合并模块。

本模块提供 YAML 配置文件的读取、深度合并和写入功能。
核心特性包括：
- 配置文件继承链（extends 机制）：一个配置可以声明继承另一个配置文件，
  子配置的值覆盖父配置
- 深度合并：嵌套字典递归合并而非简单替换
- 基础配置 + 实验配置的两层覆盖模式：先加载全局基础配置，
  再用实验特定配置覆盖差异项

在整个项目中的角色：
  被 scripts/train.py 和 scripts/eval.py 在启动时调用，是配置管理的基础设施。
  所有超参数、数据路径、模型配置等均通过本模块加载，确保配置的一致性和可复现性。
"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """深度合并两个字典。

    对于嵌套字典，递归合并子字典（保留 base 中存在但 override 中不存在的键）；
    对于非字典类型的值，直接用 override 覆盖 base 中的对应值。
    所有值均通过 deepcopy 复制，确保不会意外修改原始字典。

    参数:
        base: 基础配置字典，其内容不会被修改。
        override: 覆盖配置字典，其内容也不会被修改。

    返回:
        合并后的新字典，base 和 override 均不受影响。

    使用示例:
        >>> base = {"a": 1, "b": {"x": 1, "y": 2}}
        >>> override = {"b": {"x": 10}, "c": 3}
        >>> deep_merge(base, override)
        {"a": 1, "b": {"x": 10, "y": 2}, "c": 3}
    """
    merged = deepcopy(base)
    for key, value in override.items():
        # 如果两个字典在同一键上都是字典，则递归合并
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            # 否则直接覆盖（包含 override 中新增的键）
            merged[key] = deepcopy(value)
    return merged


def read_yaml(path: Path) -> dict[str, Any]:
    """安全读取 YAML 文件为 Python 字典。

    使用 yaml.safe_load 以避免任意代码执行风险。
    空文件（或仅含注释的文件）返回空字典。

    参数:
        path: YAML 文件的路径。

    返回:
        解析后的字典；如果文件为空，返回空字典 {}。

    异常:
        FileNotFoundError: 文件不存在时由 Python 抛出。
        yaml.YAMLError: YAML 格式错误时由 PyYAML 抛出。
    """
    with path.open("r", encoding="utf-8") as file:
        # safe_load 返回 None 时（空文件），用 or {} 转为空字典
        return yaml.safe_load(file) or {}


def _read_with_extends(path: Path) -> dict[str, Any]:
    """递归解析带 extends 继承链的配置文件。

    配置文件可以通过 extends 字段声明一个或多个父配置文件（相对于当前文件的路径）。
    本函数先递归解析所有父配置，再用当前文件的配置覆盖。

    处理流程：
    1. 读取当前文件的原始配置
    2. 提取 extends 字段（支持单个字符串或字符串列表）
    3. 递归解析每个父配置文件
    4. 将解析结果按顺序深度合并（先声明者优先级更低）
    5. 应用当前文件自身的配置覆盖

    参数:
        path: 当前配置文件的路径。

    返回:
        解析继承链后的完整配置字典。

    异常:
        ValueError: extends 字段格式不正确时抛出（既非字符串也非字符串列表）。
    """
    values = read_yaml(path)
    # 弹出 extends 字段，避免其成为最终配置的一部分
    extends = values.pop("extends", [])
    # 单个路径字符串转为列表，统一后续处理逻辑
    if isinstance(extends, str):
        extends = [extends]
    # 类型校验：extends 必须是字符串列表
    if not isinstance(extends, list) or not all(isinstance(item, str) for item in extends):
        raise ValueError(f"extends must be a path or list of paths in: {path}")
    resolved: dict[str, Any] = {}
    # 按声明顺序逐个解析父配置并合并
    for item in extends:
        # 父配置文件路径相对于当前文件所在目录解析
        resolved = deep_merge(resolved, _read_with_extends((path.parent / item).resolve()))
    # 最后合并当前文件自身的值（优先级最高）
    return deep_merge(resolved, values)


def load_config(
    base_path: Path,
    experiment_path: Path | None = None,
) -> dict[str, Any]:
    """加载并合并配置（基础配置 + 可选实验配置）。

    这是配置加载的主入口函数。先加载基础配置（含其继承链），
    如果提供了实验配置路径，再用实验配置深度覆盖基础配置。
    这种两层机制使得实验配置只需声明与基础配置不同的字段。

    参数:
        base_path: 基础 YAML 配置文件路径（支持 extends 继承）。
        experiment_path: 实验特定的 YAML 配置文件路径（可选，同样支持 extends）。

    返回:
        合并后的完整配置字典，可直接用于实验的各个组件。

    使用示例:
        >>> config = load_config(
        ...     Path("configs/base.yaml"),
        ...     Path("configs/experiments/my_exp.yaml"),
        ... )
    """
    config = _read_with_extends(base_path)
    if experiment_path is not None:
        config = deep_merge(config, _read_with_extends(experiment_path))
    return config


def write_yaml(data: dict[str, Any], path: Path) -> None:
    """将 Python 字典写入 YAML 文件。

    自动创建父目录，键按插入顺序排列（sort_keys=False），
    使用 safe_dump 避免将 Python 对象序列化为危险的 YAML 标签。

    参数:
        data: 需要写入的配置字典。
        path: 输出文件的完整路径。

    副作用:
        - 如果路径的父目录不存在，会递归创建
        - 覆盖已存在的文件
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)
