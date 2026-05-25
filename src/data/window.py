"""滑动窗口构建模块。

将时序数据按序列分组，构建固定长度的滑动窗口，窗口标签为窗口最后一个时间步的目标值。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WindowedData:
    """窗口化后的数据容器。

    Attributes:
        features: (n_windows, window_size, n_features) 形状的特征数组
        targets: (n_windows,) 形状的目标值数组，每个窗口取最后一个时间步
        sequence_ids: 每个窗口所属的序列标识
        times: 每个窗口对应的时间戳
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

    窗口不会跨序列边界。如果某序列长度不足一个窗口则跳过。

    Args:
        frame: 包含特征、目标、序列列的数据帧
        feature_columns: 特征列名列表
        target_column: 目标列名
        sequence_column: 序列标识列名
        time_column: 时间列名（可选），用于排序
        window_size: 窗口大小（时间步数）
        stride: 滑动步长

    Returns:
        WindowedData 包含堆叠好的窗口数组和元信息
    """
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be positive integers")

    windows: list[np.ndarray] = []
    targets: list[float] = []
    sequence_ids: list[str] = []
    times: list[object] = []

    for sequence_id, group in frame.groupby(sequence_column, sort=False):
        if time_column and time_column in group.columns:
            group = group.sort_values(time_column)
        feature_values = group[feature_columns].to_numpy(dtype=np.float32)
        target_values = group[target_column].to_numpy(dtype=np.float32)
        time_values = group[time_column].tolist() if time_column and time_column in group else group.index.tolist()

        for start in range(0, len(group) - window_size + 1, stride):
            end = start + window_size
            windows.append(feature_values[start:end])
            targets.append(float(target_values[end - 1]))  # 取窗口最后一个时间步的目标
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
