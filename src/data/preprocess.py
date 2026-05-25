"""数据预处理模块。

提供 Standardizer（Z-score 标准化），仅在训练集上拟合以避免数据泄露。
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    """Z-score 标准化器： (x - mean) / scale。

    零方差特征的 scale 设为 1.0，避免除零错误。
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "Standardizer":
        """在特征数组上拟合标准化器。

        Args:
            features: (n_samples, n_features) 形状的 numpy 数组

        Returns:
            拟合好的 Standardizer 实例
        """
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale == 0.0] = 1.0  # 避免除零
        return cls(mean=mean, scale=scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """对特征数组应用标准化变换。"""
        return (features - self.mean) / self.scale

    def to_dict(self) -> dict[str, list[float]]:
        """序列化为字典，用于 checkpoint 保存。"""
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, values: dict[str, list[float]]) -> "Standardizer":
        """从字典反序列化，用于评估时复现。"""
        return cls(
            mean=np.asarray(values["mean"], dtype=np.float32),
            scale=np.asarray(values["scale"], dtype=np.float32),
        )
