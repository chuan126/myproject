"""规范化 SOC 数据集降采样模块。

将已处理的规范化 CSV 数据集按指定时间间隔进行降采样，
重新计算力学派生特征，并生成新的 manifest 元数据文件。
支持覆盖写入模式，可清理目标目录中不属于当前降采样的旧文件。

在整个项目中的角色：
- 位于 data 层，作为数据准备工具链的一部分
- 被 scripts/downsample_data.py 脚本调用
- 输入为经过 prepare_cycler_workbook 处理后的规范化数据集
- 输出为降采样后的规范化数据集，可直接用于训练流水线
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .converters.features import derive_mechanical_features
from .converters.cycler_workbook import raw_file_signature
from .schema import SEQUENCE_COLUMN, SOC_COLUMN, TIME_COLUMN, canonical_manifest

# 力学派生特征列名元组，用于在重新计算前清除旧值
MECHANICAL_FEATURE_COLUMNS = ("delta_f", "delta_q", "df_dt", "df_dq", "force_slope")


def _source_sequence_paths(input_dir: Path) -> list[Path]:
    """扫描输入目录下的所有序列 CSV 文件。

    期望结构为 input_dir/sequences/*.csv。

    Args:
        input_dir: 包含 sequences/ 子目录的数据集目录

    Returns:
        按路径排序的 CSV 文件路径列表

    Raises:
        FileNotFoundError: 没有找到任何序列 CSV 文件时抛出
    """
    sequence_dir = input_dir / "sequences"
    paths = sorted(sequence_dir.rglob("*.csv")) if sequence_dir.exists() else []
    if not paths:
        raise FileNotFoundError(f"No canonical CSV files found under: {sequence_dir}")
    return paths


def _is_on_grid(values: pd.Series, origin: float, interval_s: float) -> np.ndarray:
    """判断时间序列中的每个值是否落在以 origin 为起点、
    以 interval_s 为间隔的等距网格上。

    使用 np.isclose 处理浮点舍入误差，容差为 1e-9。

    Args:
        values: 时间值序列
        origin: 网格起点（第一个时间值）
        interval_s: 网格间隔（秒）

    Returns:
        布尔数组，True 表示该时间点落在网格上
    """
    offsets = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float) - origin
    remainders = np.mod(offsets, interval_s)
    return np.isclose(remainders, 0.0, atol=1e-9) | np.isclose(remainders, interval_s, atol=1e-9)


def downsample_sequence_frame(frame: pd.DataFrame, interval_s: float) -> pd.DataFrame:
    """对单个序列的数据帧进行降采样。

    降采样流程：
    1. 按时间列排序
    2. 以第一个时间点为原点，按 interval_s 等间隔选取行
    3. 如果没有行正好落在网格上，则保留第一行（避免丢失整个序列）
    4. 重新编号 id 列
    5. 如果存在 voltage 和 current 列，重新计算 power
    6. 清除旧的力学派生特征，在降采样后的数据上重新计算
    7. 保持输出列顺序与输入一致

    Args:
        frame: 单个序列的 DataFrame（需包含 time 列）
        interval_s: 目标采样间隔（秒），必须为正数

    Returns:
        降采样后的 DataFrame，列顺序与输入相同

    Raises:
        ValueError: interval_s 非正、frame 为空或缺少 time 列时抛出
    """
    if interval_s <= 0:
        raise ValueError("interval_s must be positive.")
    if frame.empty:
        raise ValueError("Cannot downsample an empty sequence frame.")
    if TIME_COLUMN not in frame.columns:
        raise ValueError(f"Canonical CSV is missing required column: {TIME_COLUMN}")

    values = frame.copy()
    values[TIME_COLUMN] = pd.to_numeric(values[TIME_COLUMN], errors="raise")
    values = values.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)
    origin = float(values[TIME_COLUMN].iloc[0])
    # 选取落在等距网格上的行
    sampled = values.loc[_is_on_grid(values[TIME_COLUMN], origin, float(interval_s))].copy()
    if sampled.empty:
        # 如果没有行恰好落在网格上，至少保留第一行
        sampled = values.iloc[[0]].copy()

    sampled = sampled.reset_index(drop=True)
    # 重新编号 id，从 1 开始
    sampled["id"] = np.arange(1, len(sampled) + 1)
    # 重新计算 power（电压 × 电流）
    if {"voltage", "current"}.issubset(sampled.columns):
        sampled["power"] = sampled["voltage"].astype(float) * sampled["current"].astype(float)

    # 清除旧的力学派生特征，在降采样后的时间网格上重新计算
    if {"force", "cc_capacity"}.issubset(sampled.columns):
        rows = sampled.drop(columns=[column for column in MECHANICAL_FEATURE_COLUMNS if column in sampled.columns])
        records: list[dict[str, Any]] = rows.to_dict(orient="records")
        derive_mechanical_features(records)
        sampled = pd.DataFrame(records)

    return sampled[frame.columns.tolist()]


def downsample_canonical_dataset(
    input_dir: Path,
    output_dir: Path,
    interval_s: float,
    overwrite: bool = False,
) -> list[Path]:
    """对整个规范化数据集进行降采样，生成新的数据集目录。

    流程：
    1. 扫描输入目录下所有序列 CSV 文件
    2. 对每个序列执行降采样
    3. 写入输出目录（如果 overwrite=True 则清理旧文件）
    4. 合并所有降采样后的数据，生成新的 manifest.yaml

    Args:
        input_dir: 输入数据集目录（需包含 sequences/ 子目录和 manifest.yaml）
        output_dir: 输出数据集目录
        interval_s: 目标采样间隔（秒）
        overwrite: 是否覆盖输出目录中已有的序列文件，
                   同时会清理不属于当前降采样的旧 CSV 文件

    Returns:
        生成的 CSV 文件路径列表

    Raises:
        FileExistsError: 输出目录已有序列文件且 overwrite=False 时抛出
    """
    source_paths = _source_sequence_paths(input_dir)
    output_paths = [output_dir / "sequences" / path.name for path in source_paths]
    if any(path.exists() for path in output_paths) and not overwrite:
        raise FileExistsError(f"Output dataset already contains sequence CSV files: {output_dir}")

    outputs: list[tuple[Path, pd.DataFrame]] = []
    frames: list[pd.DataFrame] = []
    for source_path, output_path in zip(source_paths, output_paths):
        frame = pd.read_csv(source_path)
        downsampled = downsample_sequence_frame(frame, interval_s)
        outputs.append((output_path, downsampled))
        frames.append(downsampled)

    # 覆盖模式：清理目标目录中不属于当前降采样的旧文件
    if overwrite:
        sequence_dir = output_dir / "sequences"
        for existing_path in sequence_dir.rglob("*.csv") if sequence_dir.exists() else ():
            if existing_path not in output_paths:
                existing_path.unlink()

    generated: list[Path] = []
    for output_path, frame in outputs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        generated.append(output_path)

    # 读取源 manifest，为输出数据集生成新的 manifest
    source_manifest_path = input_dir / "manifest.yaml"
    source_manifest: dict[str, Any] = {}
    if source_manifest_path.exists():
        with source_manifest_path.open("r", encoding="utf-8") as file:
            source_manifest = yaml.safe_load(file) or {}

    combined = pd.concat(frames, ignore_index=True)
    manifest = canonical_manifest(
        dataset_name=output_dir.name,
        source_type="canonical_csv_downsample",
        frame=combined,
        sampling_period_s=float(interval_s),
        soc_method=source_manifest.get("soc_method"),
        source_dataset_name=source_manifest.get("dataset_name", input_dir.name),
        source_sampling_period_s=source_manifest.get("sampling_period_s"),
        source_manifest=str(source_manifest_path) if source_manifest_path.exists() else None,
        source_files=[str(path) for path in source_paths],
        source_file_signatures=[raw_file_signature(path) for path in source_paths],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False, allow_unicode=True)
    return generated
