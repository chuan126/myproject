"""评估可视化模块。

生成训练曲线图和 SOC 预测曲线图。
"""

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")  # 非交互后端，支持无 GUI 环境
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_WINDOWS_RESERVED_STEMS = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def _save_current_figure(path: Path) -> None:
    """保存并关闭当前 matplotlib 图形。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _safe_filename_stem(value: object) -> str:
    """将序列标识符转换为安全的文件名。"""
    stem = _INVALID_FILENAME_CHARS.sub("__", str(value)).rstrip(" .")
    stem = stem or "sequence"
    return f"_{stem}" if stem.lower() in _WINDOWS_RESERVED_STEMS else stem


def save_training_curve(history: dict[str, list[float]], path: Path) -> None:
    """保存训练/验证损失曲线。

    Args:
        history: 包含 train_loss 和 val_loss 列表的字典
        path: 输出图片路径
    """
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    _save_current_figure(path)


def save_prediction_curve(actual: np.ndarray, predicted: np.ndarray, path: Path) -> None:
    """保存真实 vs 预测 SOC 对比曲线。

    Args:
        actual: 真实 SOC 数组
        predicted: 预测 SOC 数组
        path: 输出图片路径
    """
    plt.figure(figsize=(10, 4))
    plt.plot(actual, label="actual_soc", linewidth=1)
    plt.plot(predicted, label="predicted_soc", linewidth=1)
    plt.xlabel("Sample")
    plt.ylabel("SOC")
    plt.legend()
    _save_current_figure(path)


def save_prediction_scatter(actual: np.ndarray, predicted: np.ndarray, path: Path) -> None:
    """保存真实 SOC vs 预测 SOC 散点图。"""
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.5)
    plt.xlabel("True Values [SOC]")
    plt.ylabel("Predictions [SOC]")
    plt.axis("equal")
    plt.axis("square")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.plot([0, 1], [0, 1], color="red")
    plt.title("Predicted SOC vs True SOC")
    _save_current_figure(path)


def save_soc_curves_by_sequence(predictions: pd.DataFrame, output_dir: Path) -> None:
    """保存每个测试序列的 SOC 随时间变化图。"""
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
