"""测试 Informer 编码器的构建和前向传播。

Informer 是一种基于 ProbSparse 自注意力的长序列时间序列编码器，
支持可选的蒸馏（distilling）机制来逐步缩减序列长度。

测试验证：
1. 不启用蒸馏时输出序列长度与输入一致
2. 通过模型注册表构建完整的 Informer SOC 模型
3. 多头注意力的头数必须能整除隐藏维度
"""

import unittest

import torch

from src.models import build_model
from src.models.encoders import InformerEncoder


class InformerEncoderTests(unittest.TestCase):
    """测试 InformerEncoder 的构建和前向传播行为。"""

    def test_informer_encoder_preserves_window_shape_without_distilling(self) -> None:
        """验证不启用蒸馏（distil=False）时，输出序列长度与输入保持一致。

        输入 (batch=3, seq_len=20, features=6)，
        不蒸馏时输出形状应为 (3, 20, hidden_size=16)。
        编码器的 output_dim 应为 hidden_size。
        """
        encoder = InformerEncoder(
            input_dim=6,
            hidden_size=16,
            num_layers=2,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            distil=False,
        )

        outputs = encoder(torch.randn(3, 20, 6))

        self.assertEqual(outputs.shape, (3, 20, 16))
        self.assertEqual(encoder.output_dim, 16)

    def test_registry_builds_informer_model(self) -> None:
        """验证通过 build_model 注册表能正确构建完整的 Informer SOC 模型。

        配置使用 last pooling + regression head，
        输入 (3, 20, 6)，输出应为 (3, 1)。
        """
        model = build_model(
            {
                "name": "informer",
                "hidden_size": 16,
                "num_layers": 1,
                "n_heads": 4,
                "d_ff": 32,
                "dropout": 0.0,
                "pooling": {"name": "last"},
                "head": {"name": "regression", "hidden_size": None, "dropout": 0.0},
            },
            input_dim=6,
        )

        predictions = model(torch.randn(3, 20, 6))

        self.assertEqual(predictions.shape, (3, 1))

    def test_informer_requires_heads_to_divide_hidden_size(self) -> None:
        """验证多头注意力的头数 (n_heads) 必须能整除隐藏维度 (hidden_size)。

        当 hidden_size=10、n_heads=4 时，10 % 4 != 0，
        应抛出 ValueError 提示隐藏维度必须能被头数整除。
        """
        with self.assertRaisesRegex(ValueError, "divisible by n_heads"):
            InformerEncoder(
                input_dim=6,
                hidden_size=10,
                num_layers=1,
                n_heads=4,
                d_ff=32,
                dropout=0.0,
            )


if __name__ == "__main__":
    unittest.main()
