"""评估可视化模块。

本模块负责生成模型评估阶段所需的各种可视化图表，包括：
- 训练过程中的损失曲线
- 测试集上的 SOC 预测对比曲线
- 预测误差随时间变化图
- 预测值与真实值散点图
- 按序列分组的 SOC 对比图
- 门控融合权重可视化

所有图表均通过 matplotlib 生成并保存为 PNG 文件，使用 "Agg" 非交互后端以支持
无图形界面的服务器环境。

在整个项目中的角色：
  被 experiment.py 中的 evaluate_and_save 函数调用，将评估结果可视化为
  图表文件，辅助研究人员直观判断模型性能。
"""

from pathlib import Path
import re

import matplotlib

# 必须在导入 pyplot 之前设置非交互后端，否则在无 GUI 环境中会抛出 TclError
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 文件名中不允许出现的字符（Windows/Linux 共同限制）
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
# Windows 保留设备名（不区分大小写），用作文件名 stem 时可能导致写入失败
_WINDOWS_RESERVED_STEMS = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def _save_current_figure(path: Path) -> None:
    """保存当前 matplotlib 图形到文件并释放资源。

    这是一个内部辅助函数，统一处理 tight_layout 调整、目录创建、
    保存和关闭操作，减少重复代码。

    参数:
        path: 输出 PNG 文件的完整路径。父目录不存在时会自动创建。

    副作用:
        - 创建 path 的父目录（如不存在的話）
        - 关闭当前 matplotlib 图形，释放内存
    """
    fig = plt.gcf()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _safe_filename_stem(value: object) -> str:
    """将任意值转换为安全的文件名字干（不含扩展名）。

    处理三类问题：
    1. 替换文件名中不允许的字符为双下划线
    2. 去除首尾空格和点号（Windows 不允许以点结尾的文件名）
    3. 若与 Windows 保留设备名冲突，添加下划线前缀以避免冲突

    参数:
        value: 需要转换为文件名的任意对象，会通过 str() 转为字符串。

    返回:
        安全的文件名字干字符串。
    """
    stem = _INVALID_FILENAME_CHARS.sub("__", str(value)).rstrip(" .")
    stem = stem or "sequence"
    return f"_{stem}" if stem.lower() in _WINDOWS_RESERVED_STEMS else stem


def save_training_curve(history: dict[str, list[float]], path: Path) -> None:
    """保存训练/验证损失随 epoch 变化的曲线图。

    横轴为 epoch 数，纵轴为损失值。同时绘制训练损失和验证损失两条曲线，
    便于判断模型是否过拟合或欠拟合。

    参数:
        history: 训练历史字典，必须包含键 "train_loss" 和 "val_loss"，
                 每个键对应一个按 epoch 顺序排列的浮点数列表。
        path: 输出 PNG 文件的路径。

    副作用:
        - 创建并关闭一个 matplotlib 图形
        - 将图片写入 path 指定的位置
    """
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    _save_current_figure(path)


def save_prediction_curve(actual: np.ndarray, predicted: np.ndarray, path: Path) -> None:
    """保存真实 SOC 与预测 SOC 的逐样本对比曲线。

    将两个数组按样本索引并排绘制，直观展示预测值与真实值的吻合程度。

    参数:
        actual: 真实 SOC 值的一维数组。
        predicted: 模型预测 SOC 值的一维数组，长度与 actual 相同。
        path: 输出 PNG 文件的路径。

    副作用:
        - 创建并关闭一个 matplotlib 图形
        - 将图片写入 path 指定的位置
    """
    plt.figure(figsize=(10, 4))
    plt.plot(actual, label="actual_soc", linewidth=1)
    plt.plot(predicted, label="predicted_soc", linewidth=1)
    plt.xlabel("Sample")
    plt.ylabel("SOC")
    plt.legend()
    _save_current_figure(path)


def save_error_curve(predictions: pd.DataFrame, path: Path) -> None:
    """保存预测误差随时间变化的曲线图。

    如果数据中包含多个序列，每条序列单独绘制一条误差曲线；
    如果序列数不超过 6 个，则显示图例以区分不同序列。
    图中绘制一条 y=0 的水平虚线作为零误差参考线。

    参数:
        predictions: 包含预测结果的 DataFrame，必须包含以下列：
                     - sequence_id: 序列标识符
                     - time: 时间戳（秒）
                     - error: 预测误差（predicted - actual）
        path: 输出 PNG 文件的路径。

    副作用:
        - 创建并关闭一个 matplotlib 图形
        - 将图片写入 path 指定的位置
    """
    # 按序列和时间排序，确保曲线按时间顺序绘制
    frame = predictions.sort_values(["sequence_id", "time"])
    plt.figure(figsize=(10, 4))
    if "sequence_id" in frame:
        for sequence_id, group in frame.groupby("sequence_id", sort=False):
            plt.plot(group["time"], group["error"], label=str(sequence_id), linewidth=1)
        # 序列过多时图例过于拥挤，仅当序列数 <= 6 时显示图例
        if frame["sequence_id"].nunique() <= 6:
            plt.legend()
    else:
        plt.plot(frame["time"], frame["error"], linewidth=1)
    # 零误差参考线，便于判断误差的偏置方向
    plt.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Prediction Error [SOC]")
    _save_current_figure(path)


def save_prediction_scatter(actual: np.ndarray, predicted: np.ndarray, path: Path) -> None:
    """保存预测 SOC 与真实 SOC 的散点图。

    散点图用于评估预测值在整体分布上与真实值的一致性。
    图中包含一条 y=x 的红色对角线作为完美预测参考线。
    坐标轴范围固定为 [0, 1]，因为 SOC 的取值范围为 0~1。

    参数:
        actual: 真实 SOC 值的一维数组。
        predicted: 模型预测 SOC 值的一维数组。
        path: 输出 PNG 文件的路径。

    副作用:
        - 创建并关闭一个 matplotlib 图形
        - 将图片写入 path 指定的位置
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.5)
    plt.xlabel("True Values [SOC]")
    plt.ylabel("Predictions [SOC]")
    # 确保 x 轴和 y 轴比例一致，使 y=x 线呈 45 度角
    plt.axis("equal")
    plt.axis("square")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # 完美预测参考线：散点越靠近此线，预测越准确
    plt.plot([0, 1], [0, 1], color="red")
    plt.title("Predicted SOC vs True SOC")
    _save_current_figure(path)


def save_soc_by_sequence(predictions: pd.DataFrame, path: Path) -> None:
    """保存按序列分组的 SOC 对比图（紧凑布局）。

    将所有测试序列的 SOC 预测结果绘制在同一个图的多个子图中，
    每个子图对应一个序列，便于快速浏览所有序列的预测效果。

    参数:
        predictions: 包含预测结果的 DataFrame，必须包含以下列：
                     - sequence_id: 序列标识符
                     - time: 时间戳
                     - actual_soc: 真实 SOC 值
                     - predicted_soc: 预测 SOC 值
        path: 输出 PNG 文件的路径。

    副作用:
        - 创建并关闭一个 matplotlib 图形
        - 将图片写入 path 指定的位置
    """
    sequence_count = int(predictions["sequence_id"].nunique())
    # 每个序列占一行子图，确保单个序列也能正常绘制
    rows = max(1, sequence_count)
    fig, axes = plt.subplots(rows, 1, figsize=(12, max(3, 2.8 * rows)), sharex=False)
    # np.atleast_1d 确保单序列时 axes 仍为可迭代数组
    axes_array = np.atleast_1d(axes)
    for axis, (sequence_id, frame) in zip(axes_array, predictions.groupby("sequence_id", sort=False)):
        frame = frame.sort_values("time")
        axis.plot(frame["time"], frame["actual_soc"], label="True SOC", color="blue", linewidth=1)
        axis.plot(frame["time"], frame["predicted_soc"], label="Predicted SOC", color="red", linewidth=1)
        axis.set_title(str(sequence_id))
        axis.set_xlabel("Time [s]")
        axis.set_ylabel("SOC")
        axis.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_soc_curves_by_sequence(predictions: pd.DataFrame, output_dir: Path) -> None:
    """为每个测试序列单独保存一张 SOC 随时间变化图。

    与 save_soc_by_sequence 不同，本函数为每个序列生成独立的 PNG 文件，
    适合序列数量较多、需要逐一详细查看的场景。
    文件名基于序列标识符生成，自动处理特殊字符和 Windows 保留名。

    参数:
        predictions: 包含预测结果的 DataFrame，列要求同 save_soc_by_sequence。
        output_dir: 输出目录，会在其中为每个序列生成一张 PNG 图片。

    副作用:
        - 创建 output_dir 目录（如不存在）
        - 为每个序列创建并关闭一个 matplotlib 图形
        - 将多张图片写入 output_dir
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for sequence_id, frame in predictions.groupby("sequence_id", sort=False):
        frame = frame.sort_values("time")
        safe_id = _safe_filename_stem(sequence_id)
        plt.figure(figsize=(12, 6))
        plt.plot(frame["time"], frame["actual_soc"], label="True SOC", color="blue")
        plt.plot(frame["time"], frame["predicted_soc"], label="Predicted SOC", color="red")
        plt.title(f"Sequence: {sequence_id}")
        plt.xlabel("Time [s]")
        plt.ylabel("SOC")
        plt.legend()
        _save_current_figure(output_dir / f"{safe_id}_soc_over_time.png")


def save_gate_weights(gate_weights: pd.DataFrame, path: Path) -> None:
    """保存门控融合权重的均值随时间变化图。

    门控融合机制用于动态融合多模态特征（如电压、电流、温度等不同来源的信息），
    权重值在 0 到 1 之间。本函数绘制每个序列的门控权重均值随时间的变化，
    帮助分析模型在不同时间点对不同模态的依赖程度。

    参数:
        gate_weights: 包含门控权重的 DataFrame，必须包含以下列：
                      - sequence_id: 序列标识符
                      - time: 时间戳
                      - gate_mean: 门控权重的逐样本均值（标量）
        path: 输出 PNG 文件的路径。

    副作用:
        - 创建并关闭一个 matplotlib 图形
        - 将图片写入 path 指定的位置
    """
    frame = gate_weights.sort_values(["sequence_id", "time"])
    plt.figure(figsize=(10, 4))
    for sequence_id, group in frame.groupby("sequence_id", sort=False):
        plt.plot(group["time"], group["gate_mean"], label=str(sequence_id), linewidth=1)
    # 门控权重值域为 [0, 1]，固定 y 轴范围以统一所有图片的刻度
    plt.ylim(0.0, 1.0)
    plt.xlabel("Time [s]")
    plt.ylabel("Mean Gate Weight")
    # 序列过多时图例过于拥挤，仅当序列数 <= 6 时显示图例
    if frame["sequence_id"].nunique() <= 6:
        plt.legend()
    _save_current_figure(path)
