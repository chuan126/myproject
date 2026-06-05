"""模型检查点保存与加载模块。

本模块是训练流程中持久化层的薄封装，在整个项目中的角色是：
- 提供统一的检查点保存接口（save_checkpoint）
- 提供统一的检查点加载接口（load_checkpoint）
- 封装 PyTorch 的序列化机制，对外隐藏实现细节

检查点中保存的内容包括模型权重、优化器状态、当前 epoch、
验证损失以及调用方提供的上下文信息（配置、数据产物路径等），
这样可以支持完整的训练恢复和实验复现。
"""

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """将检查点保存到磁盘。

    保存前自动创建目标路径的父目录（如果不存在），确保不会因
    目录缺失而中断训练流程。使用 PyTorch 原生的 torch.save 进行
    序列化，支持所有 PyTorch 内置类型和自定义对象。

    Args:
        path: 目标文件路径，通常以 .pt 或 .pth 为扩展名
        payload: 要保存的内容字典，典型键包括：
            - model_state: 模型 state_dict
            - optimizer_state: 优化器 state_dict
            - epoch: 当前 epoch 编号
            - val_loss: 验证损失值
            - 其他调用方自定义的上下文信息

    Side effects:
        在磁盘上创建或覆盖目标文件，其父目录会被创建（如果不存在）
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """从磁盘加载检查点。

    使用 weights_only=False 加载是因为检查点中不仅包含张量数据
    （模型权重、优化器状态），还包含 Python 字典、配置对象等
    非张量数据。PyTorch 的 safe 模式（weights_only=True）仅允许
    加载张量，无法处理这些混合数据。

    重要安全提示：如果你从不可信来源加载检查点，请务必先校验文件
    的完整性和来源，因为 weights_only=False 允许任意 Python 对象
    的反序列化，存在潜在的安全风险（如 pickle 注入攻击）。

    Args:
        path: 检查点文件路径
        device: 目标计算设备，模型张量会被映射到该设备上

    Returns:
        检查点内容字典，结构与保存时一致

    Side effects:
        读取磁盘文件，将张量加载到指定设备的显存/内存中

    Note:
        weights_only=False 是因为 checkpoint 中保存了 config 字典、
        data_artifacts 等非张量数据，无法用 safe 模式加载。
    """
    return torch.load(path, map_location=device, weights_only=False)
