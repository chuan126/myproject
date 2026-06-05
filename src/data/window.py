"""滑动窗口构建模块。

本模块负责将时序数据按序列分组后构建固定长度的滑动窗口，
为后续模型训练准备 (window_size, n_features) 形状的输入数组。
每个窗口的标签取窗口内最后一个时间步的目标值，
窗口不会跨越不同序列的边界，保证训练/验证/测试集之间的独立性。

在整个项目中的角色：
- 位于 data 层，是数据预处理流水线的核心组件之一
- 被 dataset.py 中的 build_dataloaders 和 build_evaluation_dataloader 调用
- 输出 WindowedData 作为 SOCDataset 的构造参数
- 支持通过 stride 参数控制窗口密度，平衡样本量与计算开销
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WindowedData:
    """窗口化后的数据容器。

    将滑动窗口构建的结果组织为结构化的数据对象，
    包含特征数组、目标值数组以及元信息（序列标识、时间戳）。

    设计意图：
    - 使用 dataclass 保持数据结构简洁、透明、可类型检查
    - features 形状为 (n_windows, window_size, n_features)，可直接传入时序模型
    - 保留 sequence_ids 和 times 元信息，用于后续的序列级分析和可视化

    Attributes:
        features: 形状为 (n_windows, window_size, n_features) 的特征数组，
                  每个窗口是一个连续时间步的特征子序列
        targets: 形状为 (n_windows,) 的目标值数组，
                 每个值为对应窗口最后一个时间步的目标值
        sequence_ids: 每个窗口所属的原始序列标识列表，长度等于 n_windows
        times: 每个窗口对应的时间戳列表（取自窗口最后一个时间步的时间值）
    """

    features: np.ndarray
    targets: np.ndarray
    sequence_ids: list[str]
    times: list[object]


def build_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    sequence_column: str,
    time_column: str | None,
    window_size: int,
    stride: int = 1,
) -> WindowedData:
    """对 DataFrame 按序列分组构建滑动窗口。

    核心算法：
    1. 按 sequence_column 分组，每组独立构建窗口，窗口不会跨序列边界
    2. 如果提供了 time_column，组内按时间排序以确保时序一致性
    3. 以 stride 为步长在序列上滑动，每个窗口取 window_size 个连续时间步
    4. 窗口标签取窗口内最后一个时间步的目标值（即 "用历史窗口预测当前时刻"）
    5. 长度不足以构成一个完整窗口的序列会被自动跳过

    Args:
        frame: 包含特征列、目标列和序列标识列的 DataFrame
        feature_columns: 用作模型输入的特征列名列表
        target_column: 预测目标列名（如 SOC 列）
        sequence_column: 区分不同独立序列的标识列名（如电池循环编号）
        time_column: 用于组内排序的时间列名，可以为 None（此时保持原始行序）
        window_size: 每个窗口包含的时间步数量，必须 >= 1
        stride: 窗口滑动的步长，默认为 1（生成密集窗口），增大可减少样本量

    Returns:
        WindowedData 实例，包含堆叠好的窗口特征数组、目标数组和元信息

    Raises:
        ValueError: 当 window_size 或 stride 不是正整数时抛出
        ValueError: 当所有序列长度都不足以构建至少一个窗口时抛出

    注意事项：
        - 特征值统一转换为 float32，以节省内存并匹配模型默认精度
        - 目标值取窗口最后一个时间步，对应 "用过去 window_size 步预测当前步" 的任务设定
        - 如果传入 time_column=None，组内行序按 DataFrame 当前顺序，不做额外排序
    """
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be positive integers")

    windows: list[np.ndarray] = []
    targets: list[float] = []
    sequence_ids: list[str] = []
    times: list[object] = []

    for sequence_id, group in frame.groupby(sequence_column, sort=False):
        # 如果配置了时间列，先按时间排序确保组内时序一致
        if time_column and time_column in group.columns:
            group = group.sort_values(time_column)
        feature_values = group[feature_columns].to_numpy(dtype=np.float32)
        target_values = group[target_column].to_numpy(dtype=np.float32)
        time_values = group[time_column].tolist() if time_column and time_column in group else group.index.tolist()

        # 以 stride 为步长在序列上滑动，提取 (window_size) 长度的连续子段
        for start in range(0, len(group) - window_size + 1, stride):
            end = start + window_size
            windows.append(feature_values[start:end])
            # 取窗口最后一个时间步的目标值作为该窗口的标签
            targets.append(float(target_values[end - 1]))
            sequence_ids.append(str(sequence_id))
            times.append(time_values[end - 1])

    if not windows:
        raise ValueError("No windows were created. Check sequence lengths and data.window_size.")

    return WindowedData(
        features=np.stack(windows),
        targets=np.asarray(targets, dtype=np.float32),
        sequence_ids=sequence_ids,
        times=times,
    )
