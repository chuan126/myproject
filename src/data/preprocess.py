"""数据预处理模块。

提供 Standardizer（Z-score 标准化器），用于将特征缩放到零均值单位方差。
核心原则：仅在训练集上拟合标准化器参数（均值和标准差），
然后在训练/验证/测试集上统一应用相同的变换，避免数据泄露。

在整个项目中的角色：
- 位于 data 层，是数据预处理流水线的一部分
- 被 dataset.py 中的 build_dataloaders 调用，拟合后应用于全部划分
- 标准化器参数会被序列化保存到 checkpoint，评估时反序列化复现
- 零方差特征通过将 scale 设为 1.0 来安全处理，避免除零错误
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    """Z-score 标准化器：对特征执行 (x - mean) / scale 变换。

    设计意图：
    - 使用 dataclass 保持参数透明、可序列化
    - 提供 fit（拟合）/ transform（变换）经典接口
    - 支持 to_dict / from_dict 用于 checkpoint 的保存和恢复

    使用场景：
    - 训练时：在训练集上调用 fit()，然后用 transform() 应用到全部划分
    - 评估时：从 checkpoint 中通过 from_dict() 恢复参数，直接 transform()

    Attributes:
        mean: 形状为 (n_features,) 的各特征均值数组
        scale: 形状为 (n_features,) 的各特征标准差数组，
               零方差特征对应的 scale 被设为 1.0
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "Standardizer":
        """在给定特征数组上拟合标准化器参数。

        计算每个特征的均值和标准差。对于标准差为零的特征（常量特征），
        将其 scale 设为 1.0 以避免 transform 时的除零错误，
        这意味着该特征在变换后保持不变。

        Args:
            features: 形状为 (n_samples, n_features) 的 numpy 数组，
                      通常应仅包含训练集数据

        Returns:
            拟合好的 Standardizer 实例，其 mean 和 scale 属性已设置
        """
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        # 将零方差特征的 scale 设为 1.0，避免后续除零
        scale[scale == 0.0] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """对特征数组应用标准化变换。

        执行 (x - mean) / scale，返回变换后的数组。
        注意：变换不会修改输入数组，返回新数组。

        Args:
            features: 形状为 (n_samples, n_features) 的特征数组

        Returns:
            标准化后的特征数组，形状与输入相同
        """
        return (features - self.mean) / self.scale

    def to_dict(self) -> dict[str, list[float]]:
        """将标准化器参数序列化为可 JSON 兼容的字典。

        用于在 checkpoint 中保存标准化器状态，便于评估时复现相同的变换。

        Returns:
            包含 "mean" 和 "scale" 两个键的字典，值均为 Python 列表
        """
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, values: dict[str, list[float]]) -> "Standardizer":
        """从字典反序列化恢复 Standardizer 实例。

        用于从 checkpoint 中恢复训练时拟合的标准化器参数，
        确保评估阶段使用与训练阶段完全相同的变换。

        Args:
            values: 包含 "mean" 和 "scale" 键的字典，值均为数值列表

        Returns:
            恢复的 Standardizer 实例
        """
        return cls(
            mean=np.asarray(values["mean"], dtype=np.float32),
            scale=np.asarray(values["scale"], dtype=np.float32),
        )
