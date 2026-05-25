"""SOC 实验的规范化表格数据契约。

运行时仅对序列标识、时间顺序和 SOC 目标列有固定要求。
特征列和元数据列保持开放，便于在不改动数据流水线的前提下
引入新的传感器或派生信号。
"""

from typing import Any

import numpy as np
import pandas as pd

TIME_COLUMN = "time"
SOC_COLUMN = "soc"
SEQUENCE_COLUMN = "sequence_id"
CORE_COLUMNS = (TIME_COLUMN, SOC_COLUMN, SEQUENCE_COLUMN)


def validate_canonical_frame(frame: pd.DataFrame, data_config: dict[str, Any]) -> None:
    """校验规范化数据集。

    检查必需列是否存在、数值列是否合法、SOC 范围是否在 [0,1]、
    每个序列内时间是否单调递增。
    """
    features = list(data_config.get("feature_columns", []))
    if not features:
        raise ValueError("data.feature_columns must define at least one model input column.")
    required = [*CORE_COLUMNS, *features]
    split_column = data_config.get("split_column")
    if split_column:
        required.append(split_column)
    missing = [column for column in dict.fromkeys(required) if column not in frame.columns]
    if missing:
        raise ValueError(f"Canonical data is missing required columns: {missing}")

    numeric_columns = [TIME_COLUMN, SOC_COLUMN, *features]
    values = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    invalid = [column for column in numeric_columns if values[column].isna().any()]
    if invalid:
        raise ValueError(f"Canonical numeric columns contain missing or non-numeric values: {invalid}")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Canonical numeric columns must contain only finite values.")
    if not values[SOC_COLUMN].between(0.0, 1.0).all():
        raise ValueError("Canonical soc values must be within [0, 1].")

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
    """创建数据集元数据。

    不限制额外特征列，仅记录核心统计信息。
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
    values.update(metadata)
    return values
