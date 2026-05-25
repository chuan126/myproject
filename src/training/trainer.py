"""训练器模块。

提供 Trainer 类（早停训练循环）和 predict 函数（批量推理）。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint


@dataclass
class TrainingResult:
    """训练结果容器。

    Attributes:
        history: 每个 epoch 的 train_loss 和 val_loss 列表
        best_epoch: 验证损失最小的 epoch
        best_val_loss: 最佳验证损失值
        checkpoint_path: 最佳模型检查点路径
    """

    history: dict[str, list[float]]
    best_epoch: int
    best_val_loss: float
    checkpoint_path: Path


class Trainer:
    """带早停机制的模型训练器。

    Args:
        model: PyTorch 模型
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        patience: 早停耐心值（连续无改善 epoch 数）
        min_delta: 判定改善的最小损失下降量
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        patience: int,
        min_delta: float = 0.0,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.min_delta = min_delta

    def _epoch(self, loader: DataLoader, training: bool) -> float:
        """执行一个 epoch 的训练或验证。

        Args:
            loader: 数据加载器
            training: True 为训练模式，False 为评估模式

        Returns:
            该 epoch 的平均损失
        """
        self.model.train(training)
        total_loss = 0.0
        sample_count = 0
        for features, targets, _ in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            if training:
                self.optimizer.zero_grad()
            with torch.set_grad_enabled(training):
                predictions = self.model(features)
                loss = self.criterion(predictions, targets)
                if training:
                    loss.backward()
                    self.optimizer.step()
            total_loss += float(loss.item()) * len(features)
            sample_count += len(features)
        return total_loss / sample_count

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_path: Path,
        checkpoint_context: dict[str, Any],
    ) -> TrainingResult:
        """执行完整训练流程。

        每个 epoch 后在验证集上评估，保存最佳模型并在早停触发时停止。

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 最大训练轮数
            checkpoint_path: 最佳模型保存路径
            checkpoint_context: 随检查点保存的额外上下文（配置、产物等）

        Returns:
            TrainingResult 包含训练历史和最佳 epoch 信息
        """
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0

        for epoch in range(1, epochs + 1):
            train_loss = self._epoch(train_loader, training=True)
            val_loss = self._epoch(val_loader, training=False)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                stale_epochs = 0
                save_checkpoint(
                    checkpoint_path,
                    {
                        **checkpoint_context,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                    },
                )
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        return TrainingResult(history, best_epoch, best_val_loss, checkpoint_path)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """对 DataLoader 中所有样本进行推理。

    Args:
        model: 已训练的模型
        loader: 数据加载器
        device: 计算设备

    Returns:
        (实际值数组, 预测值数组, 数据集索引列表)
    """
    model.eval()
    predicted: list[np.ndarray] = []
    actual: list[np.ndarray] = []
    indices: list[int] = []
    for features, targets, batch_indices in loader:
        output = model(features.to(device)).cpu().numpy().reshape(-1)
        predicted.append(output)
        actual.append(targets.numpy().reshape(-1))
        indices.extend(batch_indices.tolist())
    return np.concatenate(actual), np.concatenate(predicted), indices
