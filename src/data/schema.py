"""规范化数据契约与校验模块。

定义 SOC 估计任务所需的数据列常量、数据帧校验规则和元数据清单格式。
运行时的强制要求仅针对序列标识、时间顺序和 SOC 目标列；
特征列和元数据列保持开放，便于在不改动数据流水线的前提下
引入新的传感器信号或派生特征。

在整个项目中的角色：
- 位于 data 层，为所有数据加载和转换提供契约基准
- 导出 CORE_COLUMNS、TIME_COLUMN、SOC_COLUMN、SEQUENCE_COLUMN 供其他模块引用
- validate_canonical_frame 确保数据在进入窗口化之前满足基本约束
- canonical_manifest 为每个数据集生成标准化的元数据清单
"""

from typing import Any

import numpy as np
import pandas as pd

# 核心列名常量：时间列、SOC 目标列、序列标识列
TIME_COLUMN = "time"
SOC_COLUMN = "soc"
SEQUENCE_COLUMN = "sequence_id"
# 核心列元组，用于批量引用
CORE_COLUMNS = (TIME_COLUMN, SOC_COLUMN, SEQUENCE_COLUMN)


def validate_canonical_frame(frame: pd.DataFrame, data_config: dict[str, Any]) -> None:
    """校验规范化数据集是否满足运行时的基本约束。

    校验步骤（按顺序）：
    1. 检查 data_config 中是否定义了至少一个特征列
    2. 检查 DataFrame 中是否包含所有必需的列（核心列 + 特征列 + 可选的划分列）
    3. 检查核心数值列是否全部为有效的有限数值（排除 NaN、inf）
    4. 检查 SOC 值是否在 [0, 1] 合法范围内
    5. 检查每个序列内部时间是否单调递增

    Args:
        frame: 待校验的规范化 DataFrame
        data_config: 数据配置字典，至少包含 "feature_columns" 键

    Raises:
        ValueError: 缺少必需列时抛出
        ValueError: 数值列包含缺失值、非数值或非法无穷值时抛出
        ValueError: SOC 值超出 [0, 1] 范围时抛出
        ValueError: 某序列内时间不单调递增时抛出
    """
    features = list(data_config.get("feature_columns", []))
    if not features:
        raise ValueError("data.feature_columns must define at least one model input column.")
    required = [*CORE_COLUMNS, *features]
    split_column = data_config.get("split_column")
    if split_column:
        required.append(split_column)
    # 使用 dict.fromkeys 去重但保持顺序
    missing = [column for column in dict.fromkeys(required) if column not in frame.columns]
    if missing:
        raise ValueError(f"Canonical data is missing required columns: {missing}")

    # 仅对核心列和特征列做数值校验
    numeric_columns = [TIME_COLUMN, SOC_COLUMN, *features]
    values = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    invalid = [column for column in numeric_columns if values[column].isna().any()]
    if invalid:
        raise ValueError(f"Canonical numeric columns contain missing or non-numeric values: {invalid}")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Canonical numeric columns must contain only finite values.")
    if not values[SOC_COLUMN].between(0.0, 1.0).all():
        raise ValueError("Canonical soc values must be within [0, 1].")

    # 逐序列检查时间单调性
    for sequence_id, group in frame.assign(**{TIME_COLUMN: values[TIME_COLUMN]}).groupby(
        SEQUENCE_COLUMN, sort=False
    ):
        if not group[TIME_COLUMN].is_monotonic_increasing:
            raise ValueError(f"Canonical time must be monotonic within sequence_id={sequence_id}.")


def canonical_manifest(
    *,
    dataset_name: str,
    source_type: str,
    frame: pd.DataFrame,
    sampling_period_s: float | None = None,
    soc_method: str | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """创建规范化数据集的元数据清单。

    不限制额外特征列，仅记录核心统计信息和来源信息。
    生成的 manifest 可供下游模块追溯数据集的来源、采样率和 SOC 计算方法。

    Args:
        dataset_name: 数据集名称（必传，通过关键字强制指定）
        source_type: 数据来源类型标识（如 "canonical_csv"、"cycler_workbook" 等）
        frame: 待描述的数据帧，用于统计序列数和行数
        sampling_period_s: 采样周期（秒），可选
        soc_method: SOC 计算方法说明（如 "cycle_charge_discharge_coulomb_counting"），可选
        **metadata: 任意额外的元数据键值对，将被合并到返回字典中

    Returns:
        标准化的数据集元数据字典，包含 schema_version、列信息、统计量等
    """
    values: dict[str, Any] = {
        "schema_version": 1,
        "dataset_name": dataset_name,
        "source_type": source_type,
        "columns": frame.columns.tolist(),
        "sequence_count": int(frame[SEQUENCE_COLUMN].nunique()),
        "row_count": int(len(frame)),
    }
    if sampling_period_s is not None:
        values["sampling_period_s"] = float(sampling_period_s)
    if soc_method is not None:
        values["soc_method"] = soc_method
    # 合并调用方传入的额外元数据
    values.update(metadata)
    return values
