"""实验管理模块。

设备选择、JSON 写入、测试集评估与结果保存。
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data import DataBundle
from src.evaluation import (
    regression_metrics,
    save_prediction_curve,
    save_prediction_scatter,
    save_soc_curves_by_sequence,
)
from src.training import predict


def select_device(value: str) -> torch.device:
    """解析设备字符串。

    Args:
        value: "auto" 自动选择 CUDA/CPU，或具体设备名如 "cuda:0"

    Returns:
        torch.device 实例
    """
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def write_json(values: dict[str, Any], path: Path) -> None:
    """将字典写入 JSON 文件。

    Args:
        values: 可序列化的字典
        path: 输出文件路径
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(values, file, ensure_ascii=False, indent=2)


def evaluate_and_save(
    model: torch.nn.Module,
    bundle: DataBundle,
    device: torch.device,
    output_dir: Path,
) -> dict[str, float]:
    """在测试集上评估模型并保存结果。

    生成 predictions.csv、metrics.json 和预测曲线图。

    Args:
        model: 训练好的模型
        bundle: DataBundle 包含测试集数据
        device: 计算设备
        output_dir: 输出目录

    Returns:
        评估指标字典
    """
    actual, predicted, indices = predict(model, bundle.loaders["test"], device)
    dataset = bundle.datasets["test"]
    rows = []
    for array_index, dataset_index in enumerate(indices):
        rows.append(
            {
                "sequence_id": dataset.sequence_ids[dataset_index],
                "time": dataset.times[dataset_index],
                "actual_soc": float(actual[array_index]),
                "predicted_soc": float(predicted[array_index]),
                "error": float(predicted[array_index] - actual[array_index]),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics = regression_metrics(actual, predicted)
    write_json(metrics, output_dir / "metrics.json")
    plot_dir = output_dir / "plots"
    save_prediction_curve(np.asarray(actual), np.asarray(predicted), plot_dir / "soc_prediction_curve.png")
    save_prediction_scatter(np.asarray(actual), np.asarray(predicted), plot_dir / "pred_vs_true.png")
    save_soc_curves_by_sequence(predictions, plot_dir / "soc_over_time_by_sequence")
    return metrics
