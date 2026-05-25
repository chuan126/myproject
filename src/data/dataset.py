"""数据集、数据加载与窗口化模块。

提供 SOCDataset、滑动窗口构建、数据加载器构建和 DataBundle。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .io import load_canonical_csv, load_manifest
from .preprocess import Standardizer
from .schema import SEQUENCE_COLUMN, SOC_COLUMN, TIME_COLUMN
from .window import WindowedData, build_windows


class SOCDataset(Dataset):
    """SOC 估计任务的 PyTorch Dataset。

    内部持有预计算好的窗口化特征和目标值张量。
    """

    def __init__(self, windowed: WindowedData):
        self.features = torch.as_tensor(windowed.features, dtype=torch.float32)
        self.targets = torch.as_tensor(windowed.targets, dtype=torch.float32).unsqueeze(-1)
        self.sequence_ids = windowed.sequence_ids
        self.times = windowed.times

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.features[index], self.targets[index], index


@dataclass
class DataBundle:
    """封装训练/验证/测试的数据加载器、数据集和预处理产物。"""

    loaders: dict[str, DataLoader]
    datasets: dict[str, SOCDataset]
    artifacts: dict[str, Any]
    input_dim: int


def _load_frames(root: Path, data_config: dict[str, Any]) -> pd.DataFrame:
    """加载规范化输入数据，供训练流水线使用。"""
    format_name = data_config.get("format", "canonical_csv")
    if format_name != "canonical_csv":
        raise ValueError(f"Runtime data.format must be canonical_csv, got: {format_name}")
    return load_canonical_csv(root, data_config)


def _create_split_map(sequence_ids: list[str], split: dict[str, Any], seed: int) -> dict[str, str]:
    """按序列维度随机划分训练/验证/测试集。"""
    ids = np.asarray(sorted(set(sequence_ids)), dtype=object)
    if len(ids) < 3:
        raise ValueError("At least three sequence_id values are required for train/val/test splitting.")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    train_ratio = float(split["train"])
    val_ratio = float(split["val"])
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("data.split ratios must provide non-empty train, val and test portions.")
    train_end = min(max(1, int(len(ids) * train_ratio)), len(ids) - 2)
    val_end = max(train_end + 1, int(len(ids) * (train_ratio + val_ratio)))
    val_end = min(val_end, len(ids) - 1)
    assignment: dict[str, str] = {}
    for sequence_id in ids[:train_end]:
        assignment[str(sequence_id)] = "train"
    for sequence_id in ids[train_end:val_end]:
        assignment[str(sequence_id)] = "val"
    for sequence_id in ids[val_end:]:
        assignment[str(sequence_id)] = "test"
    return assignment


def _assign_splits(
    frame: pd.DataFrame,
    data_config: dict[str, Any],
    seed: int,
    saved_assignment: dict[str, str] | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """将每一行分配到 train/val/test 划分。

    优先使用已保存的划分方案，其次使用数据中的 split 列，最后按序列随机划分。
    """
    sequence_column = SEQUENCE_COLUMN
    split_column = data_config.get("split_column")
    if saved_assignment:
        assignment = saved_assignment
    elif split_column and split_column in frame:
        values = frame[[sequence_column, split_column]].drop_duplicates()
        if values[sequence_column].duplicated().any():
            raise ValueError("Each sequence_id must belong to exactly one split.")
        assignment = dict(zip(values[sequence_column].astype(str), values[split_column]))
    else:
        assignment = _create_split_map(
            frame[sequence_column].astype(str).tolist(),
            data_config["split"],
            seed,
        )
    assignment = {str(key): str(value) for key, value in assignment.items()}
    required = {"train", "val", "test"}
    invalid_splits = sorted(set(assignment.values()) - required)
    if invalid_splits:
        raise ValueError(f"Split assignment contains unsupported values: {invalid_splits}")

    frame = frame.copy()
    frame[sequence_column] = frame[sequence_column].astype(str)
    frame["_split"] = frame[sequence_column].map(assignment)
    if frame["_split"].isna().any():
        raise ValueError("Saved split assignment does not cover all sequences in the processed data.")
    if not required.issubset(set(frame["_split"])):
        raise ValueError("The processed data must contain train, val and test splits.")
    return frame, assignment


def build_dataloaders(
    config: dict[str, Any],
    root: Path,
    saved_artifacts: dict[str, Any] | None = None,
) -> DataBundle:
    """从配置构建数据加载器、数据集和预处理产物。

    Args:
        config: 完整配置字典
        root: 项目根目录
        saved_artifacts: 之前保存的标准化器和划分方案（用于评估时复现）

    Returns:
        DataBundle 包含 train/val/test 的 DataLoader 和 SOCDataset
    """
    data_config = config["data"]
    frame = _load_frames(root, data_config)
    saved_assignment = saved_artifacts.get("split_assignment") if saved_artifacts else None
    split_seed = int(data_config.get("split_seed", config["seed"]))
    frame, assignment = _assign_splits(frame, data_config, split_seed, saved_assignment)

    features = list(data_config["feature_columns"])
    # 仅在训练集上拟合标准化器，避免数据泄露
    train_values = frame.loc[frame["_split"] == "train", features].to_numpy(dtype=np.float32)
    scaler = (
        Standardizer.from_dict(saved_artifacts["scaler"])
        if saved_artifacts
        else Standardizer.fit(train_values)
    )
    frame.loc[:, features] = scaler.transform(frame[features].to_numpy(dtype=np.float32))

    datasets: dict[str, SOCDataset] = {}
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        subset = frame[frame["_split"] == split]
        windowed = build_windows(
            subset,
            feature_columns=features,
            target_column=SOC_COLUMN,
            sequence_column=SEQUENCE_COLUMN,
            time_column=TIME_COLUMN,
            window_size=int(data_config["window_size"]),
            stride=int(data_config.get("stride", 1)),
        )
        dataset = SOCDataset(windowed)
        datasets[split] = dataset
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(config["train"]["batch_size"]),
            shuffle=split == "train",
            num_workers=int(data_config.get("num_workers", 0)),
        )

    artifacts = {
        "scaler": scaler.to_dict(),
        "split_assignment": assignment,
        "feature_columns": features,
    }
    manifest = load_manifest(root, data_config)
    if manifest is not None:
        artifacts["manifest"] = manifest
    return DataBundle(loaders=loaders, datasets=datasets, artifacts=artifacts, input_dim=len(features))


def build_evaluation_dataloader(
    config: dict[str, Any],
    root: Path,
    saved_artifacts: dict[str, Any],
) -> DataBundle:
    """将外部数据集整体构建为 test loader，并复用训练时保存的特征标准化器。"""
    if "scaler" not in saved_artifacts:
        raise ValueError("Checkpoint data artifacts must contain a fitted scaler for external evaluation.")

    data_config = config["data"]
    frame = _load_frames(root, data_config)
    features = list(saved_artifacts.get("feature_columns", data_config["feature_columns"]))
    scaler = Standardizer.from_dict(saved_artifacts["scaler"])
    frame.loc[:, features] = scaler.transform(frame[features].to_numpy(dtype=np.float32))
    windowed = build_windows(
        frame,
        feature_columns=features,
        target_column=SOC_COLUMN,
        sequence_column=SEQUENCE_COLUMN,
        time_column=TIME_COLUMN,
        window_size=int(data_config["window_size"]),
        stride=int(data_config.get("stride", 1)),
    )
    dataset = SOCDataset(windowed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 0)),
    )
    artifacts: dict[str, Any] = {
        "scaler": scaler.to_dict(),
        "feature_columns": features,
    }
    manifest = load_manifest(root, data_config)
    if manifest is not None:
        artifacts["manifest"] = manifest
    return DataBundle(
        loaders={"test": loader},
        datasets={"test": dataset},
        artifacts=artifacts,
        input_dim=len(features),
    )
