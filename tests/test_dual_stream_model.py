"""测试双流（Dual Stream）SOC 估计模型的功能。

双流模型由两个独立的编码器分支组成：
- main_branch：处理主特征（电压、电流、温度、力）
- mech_branch：处理力学派生特征（delta_f、df_dt、delta_q、df_dq、force_slope）
两个分支的输出通过融合模块（concat 或 gated）合并后送入回归头。

测试验证：
1. 前向传播输出形状正确
2. concat 融合允许两个分支输出不同维度
3. gated 融合要求两个分支输出维度相同
4. 未知特征列名被正确拒绝
5. 模型注册表能正确构建 DualStreamSOCModel
6. GatedFusion 的 gate 值在 [0,1] 范围内
"""

import unittest

import torch

from src.models import DualStreamSOCModel, build_model
from src.models.fusion import GatedFusion


def _dual_stream_config(fusion_name: str = "gated") -> dict:
    """构造双流模型的配置字典。

    定义 9 个特征列，其中 4 个分配给主分支（电压、电流、温度、力），
    5 个分配给力学分支（delta_f、df_dt、delta_q、df_dq、force_slope）。
    每个分支使用 LSTM 编码器 + last pooling。

    Args:
        fusion_name: 融合方式，可选 "gated" 或 "concat"。

    Returns:
        双流模型配置字典。
    """
    return {
        "architecture": {"name": "dual_stream"},
        "feature_columns": [
            "voltage",
            "current",
            "temperature",
            "force",
            "delta_f",
            "df_dt",
            "delta_q",
            "df_dq",
            "force_slope",
        ],
        "main_branch": {
            "feature_columns": ["voltage", "current", "temperature", "force"],
            "encoder": {
                "name": "lstm",
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "pooling": {"name": "last"},
        },
        "mech_branch": {
            "feature_columns": ["delta_f", "df_dt", "delta_q", "df_dq", "force_slope"],
            "encoder": {
                "name": "lstm",
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "pooling": {"name": "last"},
        },
        "fusion": {"name": fusion_name},
        "head": {"name": "regression", "hidden_size": None, "dropout": 0.0},
    }


class DualStreamModelTests(unittest.TestCase):
    """测试双流模型的构建、前向传播和特征索引机制。"""

    def test_dual_stream_forward_shape(self) -> None:
        """验证双流模型前向传播的输出形状。

        输入 (batch=4, window=20, features=9)，输出应为 (4, 1)。
        """
        model = build_model(_dual_stream_config(), input_dim=9)
        inputs = torch.randn(4, 20, 9)

        predictions = model(inputs)

        self.assertEqual(predictions.shape, (4, 1))

    def test_dual_stream_concat_allows_different_dims(self) -> None:
        """验证 concat 融合方式允许主分支和力学分支输出不同维度。

        主分支 hidden_size=8，力学分支 hidden_size=5，
        concat 后 fusion.output_dim 应为 8+5=13。
        """
        config = _dual_stream_config("concat")
        config["main_branch"]["encoder"]["hidden_size"] = 8
        config["mech_branch"]["encoder"]["hidden_size"] = 5

        model = build_model(config, input_dim=9)
        predictions = model(torch.randn(4, 20, 9))

        self.assertEqual(model.fusion.output_dim, 13)
        self.assertEqual(predictions.shape, (4, 1))

    def test_dual_stream_gated_requires_same_dims(self) -> None:
        """验证 gated 融合方式要求两个分支的隐藏维度相同。

        当 main_branch hidden_size=8、mech_branch hidden_size=5 时，
        构建模型应抛出 ValueError，因为 gated 需要两个输入维度一致。
        """
        config = _dual_stream_config("gated")
        config["main_branch"]["encoder"]["hidden_size"] = 8
        config["mech_branch"]["encoder"]["hidden_size"] = 5

        with self.assertRaisesRegex(ValueError, "Gated fusion requires matching branch dimensions"):
            build_model(config, input_dim=9)

    def test_dual_stream_rejects_unknown_feature_column(self) -> None:
        """验证配置了不在全局 feature_columns 中的分支特征列时抛出 ValueError。

        当 main_branch.feature_columns 包含 "missing_feature" 而该列不在
        顶层的 feature_columns 列表中时，应拒绝构建。
        """
        config = _dual_stream_config()
        config["main_branch"]["feature_columns"] = ["voltage", "missing_feature"]

        with self.assertRaisesRegex(ValueError, "Unknown main branch feature column: missing_feature"):
            build_model(config, input_dim=9)

    def test_registry_builds_dual_stream_model(self) -> None:
        """验证通过模型注册表 build_model 正确构建 DualStreamSOCModel 实例。

        同时验证特征索引正确：
        - main_indices 对应电压、电流、温度、力 → [0, 1, 2, 3]
        - mech_indices 对应力学特征 → [4, 5, 6, 7, 8]
        """
        model = build_model(_dual_stream_config(), input_dim=9)

        self.assertIsInstance(model, DualStreamSOCModel)
        self.assertEqual(model.main_indices, [0, 1, 2, 3])
        self.assertEqual(model.mech_indices, [4, 5, 6, 7, 8])

    def test_gated_fusion_exposes_last_gate(self) -> None:
        """验证 GatedFusion 模块的 gate（门控）值在合理的 [0, 1] 范围内。

        GatedFusion 使用 sigmoid 激活的 gate 来加权融合两个分支的输出，
        gate 值应始终在 [0, 1] 之间，且 last_gate 属性在前向传播后被赋值。
        """
        fusion = GatedFusion(main_dim=4, mech_dim=4)
        output = fusion(torch.randn(3, 4), torch.randn(3, 4))

        self.assertEqual(output.shape, (3, 4))
        self.assertIsNotNone(fusion.last_gate)
        assert fusion.last_gate is not None
        self.assertTrue(torch.all(fusion.last_gate >= 0.0))
        self.assertTrue(torch.all(fusion.last_gate <= 1.0))


if __name__ == "__main__":
    unittest.main()
