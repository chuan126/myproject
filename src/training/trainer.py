"""训练器模块。

本模块是训练流程的核心编排层，负责：
- 定义训练结果的数据结构（TrainingResult）
- 实现带早停机制的训练循环（Trainer 类）
- 提供批量推理接口（predict 函数）

在整个项目中的角色：连接数据加载、模型、损失函数、优化器和检查点保存，
将各个独立组件组装成完整的训练与评估流程。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoint import save_checkpoint


@dataclass
class TrainingResult:
    """训练结果容器，用于封装一次完整训练流程的产出。

    设计为不可变数据类，调用方通过属性读取训练历史与最佳模型信息，
    便于下游评估、绘图和日志记录。

    Attributes:
        history: 字典，包含 'train_loss' 和 'val_loss' 两个列表，
            每个元素对应一个 epoch 的平均损失值
        best_epoch: 验证损失最小的 epoch 编号（从 1 开始计数）
        best_val_loss: 最佳验证损失值
        checkpoint_path: 最佳模型检查点的保存路径
    """

    history: dict[str, list[float]]
    best_epoch: int
    best_val_loss: float
    checkpoint_path: Path


class Trainer:
    """带早停机制的模型训练器。

    设计意图：封装标准训练循环，支持训练/验证交替执行、最佳模型自动保存、
    以及基于验证损失无改善的早停策略。使用者只需提供模型、损失函数、优化器
    和数据加载器，调用 fit() 即可完成训练。

    使用场景：适用于回归问题（损失下降方向为越小越好），分类问题需调整
    损失比较方向。早停机制通过 patience 和 min_delta 两个参数控制灵敏度。

    Args:
        model: PyTorch 模型实例，需实现 forward 方法
        criterion: 损失函数模块，接受 (predictions, targets) 并返回标量损失
        optimizer: PyTorch 优化器实例
        device: 计算设备（cpu 或 cuda）
        patience: 早停耐心值，即连续多少个 epoch 验证损失无改善后停止训练
        min_delta: 判定"改善"的最小损失下降量，仅当 val_loss < best_val_loss - min_delta
            时才视为有效改善，用于过滤微小波动导致的伪改善
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

    def _epoch(self, loader: DataLoader, training: bool, epoch: int, epochs: int) -> float:
        """执行一个 epoch 的训练或验证。

        基于 `training` 标志自动切换模型模式和梯度计算上下文，
        复用同一段代码处理训练和验证两种阶段，减少重复逻辑。

        训练模式下：启用梯度计算，执行前向传播、损失计算、反向传播和参数更新。
        验证模式下：禁用梯度计算（torch.no_grad），仅执行前向传播和损失计算，
        大幅减少显存占用和计算开销。

        损失统计采用按样本数加权平均的方式，确保不同 batch 大小下的结果可比。

        Args:
            loader: 数据加载器，每个 batch 返回 (features, targets, indices) 三元组
            training: True 表示训练模式，False 表示评估模式
            epoch: 当前 epoch 编号（从 1 开始），仅用于进度条显示
            epochs: 总 epoch 数，仅用于进度条显示

        Returns:
            该 epoch 所有样本的平均损失（标量浮点数）
        """
        self.model.train(training)
        total_loss = 0.0
        sample_count = 0
        phase = "train" if training else "val"
        progress = tqdm(
            loader,
            desc=f"Epoch {epoch}/{epochs} {phase}",
            unit="batch",
            leave=False,
        )
        for features, targets, _ in progress:
            # 将数据移动到指定设备（GPU 或 CPU）
            features = features.to(self.device)
            targets = targets.to(self.device)

            if training:
                # 每个 batch 前清零梯度，防止梯度累加
                self.optimizer.zero_grad()

            # 使用上下文管理器控制梯度开关，比手动 torch.no_grad() 更简洁
            with torch.set_grad_enabled(training):
                predictions = self.model(features)
                loss = self.criterion(predictions, targets)
                if training:
                    loss.backward()
                    self.optimizer.step()

            # 累加总损失（乘以样本数以支持不同大小的最后一个 batch）
            total_loss += float(loss.item()) * len(features)
            sample_count += len(features)
            # 实时更新进度条上的当前平均损失
            progress.set_postfix(loss=f"{total_loss / sample_count:.6f}")

        # 返回按总样本数归一化的平均损失
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

        每个 epoch 依次进行以下步骤：
        1. 在训练集上执行一个训练 epoch
        2. 在验证集上执行一个评估 epoch
        3. 比较当前验证损失与历史最佳值，若改善则保存检查点
        4. 检查早停条件，若连续 patience 个 epoch 无改善则停止

        早停策略：通过 stale_epochs 计数器跟踪连续无改善的 epoch 数。
        最佳模型仅在验证损失严格优于历史最佳（考虑 min_delta 容差）时更新。

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 最大训练轮数（早停触发时可提前结束）
            checkpoint_path: 最佳模型检查点的保存路径，父目录不存在时会自动创建
            checkpoint_context: 随检查点一起保存的额外上下文信息，通常包括
                配置参数、数据产物路径等，用于后续恢复训练或复现实验

        Returns:
            TrainingResult 包含完整的训练历史和最佳 epoch 信息
        """
        history = {"train_loss": [], "val_loss": []}
        # 初始化为无穷大，确保第一个 epoch 总会触发保存
        best_val_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0

        epoch_progress = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
        for epoch in epoch_progress:
            # 执行训练和验证各一个 epoch
            train_loss = self._epoch(train_loader, training=True, epoch=epoch, epochs=epochs)
            val_loss = self._epoch(val_loader, training=False, epoch=epoch, epochs=epochs)

            # 记录本轮损失
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # 判断是否有改善：损失下降幅度需超过 min_delta
            improved = val_loss < best_val_loss - self.min_delta

            postfix = {
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "best_epoch": best_epoch,
            }

            if improved:
                # 更新最佳记录
                best_val_loss = val_loss
                best_epoch = epoch
                stale_epochs = 0
                postfix["best_epoch"] = best_epoch

                # 保存包含完整状态的检查点，便于后续恢复训练或直接用于推理
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
                    # 触发早停，更新进度条后退出循环
                    epoch_progress.set_postfix(postfix)
                    break

            epoch_progress.set_postfix(postfix)

        return TrainingResult(history, best_epoch, best_val_loss, checkpoint_path)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """对 DataLoader 中所有样本进行批量推理。

    该函数是独立于 Trainer 的推理入口，适用于训练完成后的评估阶段。
    使用装饰器 @torch.no_grad() 全局禁用梯度计算，确保推理时零显存
    用于存储中间激活和梯度。

    处理流程：遍历所有 batch，将特征送入模型得到预测值，收集实际值、
    预测值和样本索引，最后拼接为连续数组返回。每个样本的预测值被
    reshape 为一维，适用于单输出回归任务。

    Args:
        model: 已训练的 PyTorch 模型，调用前应确保处于 eval 模式
        loader: 数据加载器，每个 batch 返回 (features, targets, batch_indices) 三元组
        device: 计算设备，特征会被移到该设备上

    Returns:
        三元组 (actual, predictions, indices)：
        - actual: shape (N,) 的 numpy 数组，所有样本的真实值
        - predictions: shape (N,) 的 numpy 数组，所有样本的预测值
        - indices: 长度为 N 的列表，每个元素是样本在原始数据集中的索引号
    """
    model.eval()
    predicted: list[np.ndarray] = []
    actual: list[np.ndarray] = []
    indices: list[int] = []
    for features, targets, batch_indices in loader:
        # 将特征移到设备上进行前向计算，结果移回 CPU 转为 numpy
        output = model(features.to(device)).cpu().numpy().reshape(-1)
        predicted.append(output)
        actual.append(targets.numpy().reshape(-1))
        indices.extend(batch_indices.tolist())
    # 将所有 batch 的结果沿第一维拼接，得到完整的数据集输出
    return np.concatenate(actual), np.concatenate(predicted), indices
