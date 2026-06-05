"""数据集构建、数据加载与窗口化模块。

本模块是 data 层的核心编排组件，负责将原始 CSV 数据转换为 PyTorch 可用的
DataLoader。主要流程为：
1. 加载规范化 CSV 文件（通过 io 模块）
2. 按序列维度划分训练/验证/测试集（支持随机划分和基于规则的划分）
3. 在训练集上拟合 Standardizer，并变换全部数据
4. 对每个划分构建滑动窗口，生成 SOCDataset 和 DataLoader

在整个项目中的角色：
- 位于 data 层，是连接原始数据和模型训练的桥梁
- 被 scripts/train.py 和 scripts/eval.py 调用
- 导出 build_dataloaders（训练用）和 build_evaluation_dataloader（评估用）
- 输出 DataBundle，封装了 DataLoader、Dataset 和预处理产物
"""

from dataclasses import dataclass
from fnmatch import fnmatchcase
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
    与普通 Dataset 不同，它不进行即时窗口化，
    而是在构造前由 build_windows 一次性完成所有窗口的预计算，
    使得训练过程中的数据加载非常高效。

    设计意图：
    - 将 WindowedData 中的 numpy 数组转换为 PyTorch 张量
    - 目标值增加最后一维（从 (n_windows,) 变为 (n_windows, 1)），匹配模型输出形状
    - __getitem__ 返回 (features, target, index) 三元组，index 用于 debug 和跟踪
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
    """封装训练/验证/测试的数据加载器、数据集和预处理产物的容器。

    设计意图：
    - 将构建数据流水线的所有输出统一封装，方便训练和评估流程使用
    - loaders 和 datasets 通过 "train"/"val"/"test" 键访问
    - artifacts 包含标准化器参数和划分方案，用于 checkpoint 保存和评估复现

    Attributes:
        loaders: 键为划分名（"train"/"val"/"test"），值为对应的 DataLoader
        datasets: 键为划分名，值为对应的 SOCDataset（可用于手动迭代和检查）
        artifacts: 预处理产物字典，包含 scaler（标准化器参数）、
                   split_assignment（序列到划分的映射）、feature_columns 等
        input_dim: 特征维度（特征列数量），用于模型初始化
    """

    loaders: dict[str, DataLoader]
    datasets: dict[str, SOCDataset]
    artifacts: dict[str, Any]
    input_dim: int


def _load_frames(root: Path, data_config: dict[str, Any]) -> pd.DataFrame:
    """加载规范化输入数据，供训练流水线使用。

    当前仅支持 canonical_csv 格式，未来可扩展支持其他格式。
    会通过 schema.validate_canonical_frame 对数据进行校验。

    Args:
        root: 项目根目录，用于解析相对路径
        data_config: 数据配置字典，需包含 "format" 和 "path" 键

    Returns:
        合并后的规范化 DataFrame

    Raises:
        ValueError: 当 data_config["format"] 不是 "canonical_csv" 时抛出
    """
    format_name = data_config.get("format", "canonical_csv")
    if format_name != "canonical_csv":
        raise ValueError(f"Runtime data.format must be canonical_csv, got: {format_name}")
    return load_canonical_csv(root, data_config)


def _create_split_map(sequence_ids: list[str], split: dict[str, Any], seed: int) -> dict[str, str]:
    """按序列维度随机划分训练/验证/测试集。

    划分在序列级别进行（而非行级别），确保同一序列的所有窗口
    属于同一个划分，避免数据泄露。

    算法：
    1. 对序列 ID 列表排序并随机打乱
    2. 按 train/val 比例切分，余下的归为 test
    3. 保证每个划分至少有一个序列

    Args:
        sequence_ids: 所有序列 ID 的列表
        split: 包含 "train" 和 "val" 键的字典，值为比例（如 0.7, 0.15），
               余下的自动归入 test
        seed: 随机种子，用于保证划分的可复现性

    Returns:
        序列 ID 到划分名（"train"/"val"/"test"）的映射字典
    """
    ids = np.asarray(sorted(set(sequence_ids)), dtype=object)
    if len(ids) < 3:
        raise ValueError("At least three sequence_id values are required for train/val/test splitting.")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    train_ratio = float(split["train"])
    val_ratio = float(split["val"])
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("data.split ratios must provide non-empty train, val and test portions.")
    # 确保 train 和 val 至少各有一个序列，且 test 也至少有一个
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


def _matches_pattern(value: str, pattern: str) -> bool:
    # 统一转小写后使用 fnmatch 比较，消除平台大小写差异带来的匹配不一致
    return fnmatchcase(value.lower(), pattern.lower())


def _split_unmatched_sequences(
    sequence_ids: list[str],
    split_ratios: dict[str, Any],
    seed: int,
) -> dict[str, str]:
    """将未匹配到规则的序列按比例随机分配到 train/val/test。

    用于基于规则的划分方案中，处理那些未被显式规则覆盖的序列。
    保证每个划分至少分到一个序列。

    Args:
        sequence_ids: 未被规则匹配到的序列 ID 列表
        split_ratios: 划分比例字典，键为 "train"/"val"/"test"，值为浮点数比例
        seed: 随机种子

    Returns:
        序列 ID 到划分名的映射字典
    """
    required = {"train", "val", "test"}
    invalid_splits = sorted(set(split_ratios) - required)
    if invalid_splits:
        raise ValueError(f"Unsupported data.split_rules.remaining key(s): {invalid_splits}")
    ratios = {str(key): float(value) for key, value in split_ratios.items()}
    if not ratios or any(value <= 0.0 for value in ratios.values()):
        raise ValueError("data.split_rules.remaining must contain positive split ratios.")

    ids = np.asarray(sorted(set(sequence_ids)), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    split_names = list(ratios)
    total = sum(ratios.values())
    assignment: dict[str, str] = {}
    start = 0
    for index, split_name in enumerate(split_names):
        if index == len(split_names) - 1:
            # 最后一个划分取所有剩余序列
            end = len(ids)
        else:
            end = start + int(len(ids) * ratios[split_name] / total)
            remaining_splits = len(split_names) - index - 1
            # 确保剩余划分至少各有一个序列
            end = min(max(end, start + 1), len(ids) - remaining_splits)
        for sequence_id in ids[start:end]:
            assignment[str(sequence_id)] = split_name
        start = end
    return assignment


def _create_rule_based_split_map(
    sequence_ids: list[str],
    split_rules: dict[str, Any],
    seed: int,
) -> dict[str, str]:
    """根据序列 ID 的模式匹配分配训练/验证/测试集。

    支持灵活的基于命名规则的划分方案，适用于需要按实验条件
    （如温度、电流倍率）划分数据集的场景。

    配置示例：
        split_rules:
          test:
            - "*2000N*"       # 将 ID 包含 "2000N" 的序列划入 test
          remaining:
            train: 0.67       # 剩余序列按 67%/33% 随机分配
            val: 0.33

    还支持 default 键，用于将未匹配序列统一分配到某个划分。

    Args:
        sequence_ids: 所有序列 ID 的列表
        split_rules: 划分规则字典，键为 "train"/"val"/"test"/"default"/"remaining"
        seed: 随机种子（用于 "remaining" 的随机分配）

    Returns:
        序列 ID 到划分名的映射字典

    Raises:
        ValueError: 某个序列匹配到多个规则时抛出
        ValueError: 存在未匹配序列且未配置 remaining 或 default 时抛出
    """
    required = {"train", "val", "test"}
    default_split = str(split_rules.get("default", "")).strip()
    if default_split and default_split not in required:
        raise ValueError(f"data.split_rules.default must be one of {sorted(required)}.")
    remaining_rules = split_rules.get("remaining")

    # 收集各划分的 glob 模式列表
    rules: dict[str, list[str]] = {}
    for split_name, patterns in split_rules.items():
        if split_name in {"default", "remaining"}:
            continue
        if split_name not in required:
            raise ValueError(f"Unsupported data.split_rules key: {split_name}")
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list) or not all(isinstance(pattern, str) for pattern in patterns):
            raise ValueError(f"data.split_rules.{split_name} must be a string or list of strings.")
        rules[split_name] = patterns

    # 逐序列匹配规则
    assignment: dict[str, str] = {}
    unmatched: list[str] = []
    for sequence_id in sorted(set(map(str, sequence_ids))):
        matches = [
            split_name
            for split_name, patterns in rules.items()
            if any(_matches_pattern(sequence_id, pattern) for pattern in patterns)
        ]
        if len(matches) > 1:
            raise ValueError(f"sequence_id matches multiple split rules: {sequence_id} -> {matches}")
        if matches:
            assignment[sequence_id] = matches[0]
        else:
            unmatched.append(sequence_id)
    # 处理未匹配序列：优先使用 remaining 规则，其次 default，否则报错
    if unmatched and remaining_rules:
        if not isinstance(remaining_rules, dict):
            raise ValueError("data.split_rules.remaining must be a mapping of split ratios.")
        assignment.update(_split_unmatched_sequences(unmatched, remaining_rules, seed))
    elif unmatched and default_split:
        assignment.update({sequence_id: default_split for sequence_id in unmatched})
    elif unmatched:
        raise ValueError(f"sequence_id does not match any split rule: {unmatched[0]}")
    return assignment


def _assign_splits(
    frame: pd.DataFrame,
    data_config: dict[str, Any],
    seed: int,
    saved_assignment: dict[str, str] | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """将 DataFrame 中的每一行分配到 train/val/test 划分。

    划分优先级（从高到低）：
    1. 使用已保存的划分方案（saved_assignment），用于评估时复现训练时的划分
    2. 使用数据中已有的 split 列，支持外部预划分的数据集
    3. 使用基于规则的划分（split_rules），支持按序列 ID 模式分配
    4. 使用随机按比例划分（split），作为默认方案

    分配结果以 "_split" 列的形式附加到 DataFrame 副本上。

    Args:
        frame: 待划分的 DataFrame
        data_config: 数据配置字典
        seed: 随机种子
        saved_assignment: 之前保存的序列到划分的映射（评估时使用）

    Returns:
        (附加了 "_split" 列的 DataFrame 副本, 序列到划分的映射字典)
    """
    sequence_column = SEQUENCE_COLUMN
    split_column = data_config.get("split_column")
    if saved_assignment:
        # 评估模式：使用训练时保存的划分方案
        assignment = saved_assignment
    elif split_column and split_column in frame:
        # 数据中已有划分列，提取唯一映射
        values = frame[[sequence_column, split_column]].drop_duplicates()
        if values[sequence_column].duplicated().any():
            raise ValueError("Each sequence_id must belong to exactly one split.")
        assignment = dict(zip(values[sequence_column].astype(str), values[split_column]))
    elif data_config.get("split_rules"):
        # 基于序列 ID 模式匹配的规则划分
        assignment = _create_rule_based_split_map(
            frame[sequence_column].astype(str).tolist(),
            data_config["split_rules"],
            seed,
        )
    else:
        # 默认：随机按比例划分
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
    """从配置构建数据加载器、数据集和预处理产物（完整流水线）。

    这是训练脚本调用的主入口。完整流程：
    1. 加载 CSV 数据
    2. 分配 train/val/test 划分
    3. 在训练集上拟合 Standardizer，并变换全部数据（避免数据泄露）
    4. 对每个划分构建滑动窗口 → SOCDataset → DataLoader
    5. 收集预处理产物（标准化器参数、划分方案等）用于 checkpoint 保存

    Args:
        config: 完整配置字典，需包含 "data" 和 "train" 子配置
        root: 项目根目录，用于解析数据路径
        saved_artifacts: 之前保存的标准化器和划分方案（可选），
                         传入时跳过重新拟合，直接使用已有参数

    Returns:
        DataBundle 包含 train/val/test 的 DataLoader、SOCDataset 和预处理产物

    注意事项：
        - 标准化器仅在训练集上拟合，然后变换全部划分，这是关键的反泄露措施
        - 训练集的 DataLoader 设置了 shuffle=True，验证/测试集为 False
    """
    data_config = config["data"]
    frame = _load_frames(root, data_config)
    saved_assignment = saved_artifacts.get("split_assignment") if saved_artifacts else None
    split_seed = int(data_config.get("split_seed", config["seed"]))
    frame, assignment = _assign_splits(frame, data_config, split_seed, saved_assignment)

    features = list(data_config["feature_columns"])
    # 仅在训练集上拟合标准化器，避免数据泄露到验证/测试集
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
    """为外部评估数据集构建 test loader，复用训练时保存的特征标准化器。

    与 build_dataloaders 的区别：
    - 不做 train/val/test 划分，整个数据集作为 test 集
    - 不拟合新的标准化器，而是从 saved_artifacts 中恢复训练时的参数
    - 不 shuffle 数据，保证评估结果可复现

    使用场景：对新的电池循环数据或不同的实验条件数据进行评估，
    使用与训练时完全相同的预处理管道。

    Args:
        config: 完整配置字典
        root: 项目根目录
        saved_artifacts: 必须包含 "scaler" 键的训练产物，
                         可选包含 "feature_columns" 键

    Returns:
        DataBundle，其中仅包含 test 的 DataLoader 和 SOCDataset

    Raises:
        ValueError: 当 saved_artifacts 中缺少 "scaler" 时抛出
    """
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
