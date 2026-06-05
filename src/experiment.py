"""实验管理模块。

本模块是训练与评估流程的编排层，负责：
- 计算设备的选择与解析
- 配置序列化为 JSON
- 测试集评估与结果持久化（预测值 CSV、评估指标 JSON、可视化图表）

在整个项目中的角色：
  被 scripts/eval.py 和 scripts/train.py 调用，是脚本层与模型/数据层之间的
  桥梁，封装了评估阶段的通用逻辑，避免在多个脚本中重复代码。
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
    save_error_curve,
    save_gate_weights,
    save_prediction_curve,
    save_prediction_scatter,
    save_soc_by_sequence,
)


def _last_gate(model: torch.nn.Module) -> torch.Tensor | None:
    """从模型中提取最后一次前向传播的门控权重。

    这是一个内部辅助函数，通过访问模型的 fusion 子模块获取门控权重张量。
    如果模型不使用门控融合机制，则返回 None。

    参数:
        model: PyTorch 模型实例，可能包含 fusion 子模块。

    返回:
        已 detach 并移至 CPU 的门控权重张量；如果模型没有门控机制则返回 None。
    """
    fusion = getattr(model, "fusion", None)
    gate = getattr(fusion, "last_gate", None)
    # last_gate 是前向传播缓存的张量，需要 detach 断开计算图并移至 CPU
    return gate.detach().cpu() if isinstance(gate, torch.Tensor) else None


def select_device(value: str) -> torch.device:
    """解析设备字符串为 PyTorch device 对象。

    支持 "auto" 自动选择（CUDA 可用时优先 GPU，否则回退 CPU），
    以及具体设备名如 "cuda:0"、"cpu"、"mps" 等。

    参数:
        value: 设备描述字符串。"auto" 或任何 torch.device 接受的字符串。

    返回:
        对应的 torch.device 实例。

    使用示例:
        >>> device = select_device("auto")  # GPU 可用时返回 cuda:0，否则 cpu
        >>> device = select_device("cuda:1")  # 指定第 2 块 GPU
    """
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def write_json(values: dict[str, Any], path: Path) -> None:
    """将 Python 字典写入 JSON 文件。

    自动创建父目录，使用 UTF-8 编码，确保中文等非 ASCII 字符可读（ensure_ascii=False），
    并使用 2 空格缩进提高可读性。

    参数:
        values: 需要序列化的字典，所有值必须可 JSON 序列化。
        path: 输出文件的完整路径。

    副作用:
        - 如果路径的父目录不存在，会递归创建
        - 覆盖已存在的文件
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
    """在测试集上评估模型并保存所有结果。

    此函数是评估流程的核心入口，执行以下步骤：
    1. 遍历测试集 DataLoader，逐批次前向传播收集预测值
    2. 将每个样本的预测值、真实值、元信息（序列ID、时间戳）整理为 DataFrame
    3. 计算 MSE、MAE、RMSE、Max Error、R² 五个回归指标
    4. 保存 predictions.csv（每个样本的详细预测结果）
    5. 生成并保存多张可视化图表（SOC 对比图、误差图、散点图、按序列对比图）
    6. 如果模型使用门控融合，额外保存门控权重图和 CSV

    参数:
        model: 已完成训练的 PyTorch 模型，将被设为 eval 模式。
        bundle: DataBundle 实例，必须包含 "test" 键的数据集和数据加载器。
        device: 计算设备，模型和数据将被移至该设备。
        output_dir: 输出根目录，会在其中创建 plots/ 子目录存放图表。

    返回:
        包含五个评估指标（mse, mae, rmse, max_error, r2）的字典。

    副作用:
        - 将模型设为 eval 模式（影响 dropout、batch norm 等层的行为）
        - 在 output_dir 下创建 predictions.csv
        - 在 output_dir/plots/ 下创建多张 PNG 图表
        - 如果有门控权重，还会创建 gate_weights.csv
    """
    dataset = bundle.datasets["test"]
    rows = []
    gate_rows = []
    actual: list[float] = []
    predicted: list[float] = []
    model.eval()
    # 使用 torch.no_grad() 禁用梯度计算，节省显存并加速推理
    with torch.no_grad():
        for features, targets, batch_indices in bundle.loaders["test"]:
            # 数据移至 GPU（如果可用），前向传播后移回 CPU 转为 NumPy
            output = model(features.to(device)).cpu().numpy().reshape(-1)
            target_values = targets.numpy().reshape(-1)
            # 尝试提取门控权重（仅对使用门控融合的模型有效）
            gate = _last_gate(model)
            for batch_offset, dataset_index in enumerate(batch_indices.tolist()):
                target_value = float(target_values[batch_offset])
                prediction = float(output[batch_offset])
                actual.append(target_value)
                predicted.append(prediction)
                # 每个样本记录一行，包含序列 ID、时间戳、真值、预测值和误差
                rows.append(
                    {
                        "sequence_id": dataset.sequence_ids[dataset_index],
                        "time": dataset.times[dataset_index],
                        "actual_soc": target_value,
                        "predicted_soc": prediction,
                        "error": prediction - target_value,
                    }
                )
                # 如果提取到了门控权重，逐样本记录各维度的权重值
                if gate is not None:
                    gate_values = gate[batch_offset].numpy()
                    gate_row = {
                        "sequence_id": dataset.sequence_ids[dataset_index],
                        "time": dataset.times[dataset_index],
                        "gate_mean": float(gate_values.mean()),
                    }
                    # 逐维度记录门控权重，便于分析各模态的贡献度
                    gate_row.update({f"gate_dim_{index}": float(value) for index, value in enumerate(gate_values)})
                    gate_rows.append(gate_row)
    # 输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    # 保存每个样本的详细预测结果
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    # 计算全局回归指标
    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)
    metrics = regression_metrics(actual_array, predicted_array)
    # 生成评估图表
    plot_dir = output_dir / "plots"
    save_prediction_curve(actual_array, predicted_array, plot_dir / "soc_prediction.png")
    save_error_curve(predictions, plot_dir / "soc_error.png")
    save_prediction_scatter(actual_array, predicted_array, plot_dir / "pred_vs_true.png")
    save_soc_by_sequence(predictions, plot_dir / "soc_by_sequence.png")
    # 仅当模型使用门控融合时才生成门控权重图
    if gate_rows:
        save_gate_weights(pd.DataFrame(gate_rows), plot_dir / "gate_weights.png")
    return metrics
