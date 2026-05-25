"""模型检查点保存与加载。"""

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """保存检查点到磁盘。

    Args:
        path: 目标文件路径
        payload: 包含 model_state、optimizer_state 等内容的字典
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """从磁盘加载检查点。

    Args:
        path: 检查点文件路径
        device: 加载到的设备

    Returns:
        检查点内容字典
    """
    return torch.load(path, map_location=device, weights_only=False)
