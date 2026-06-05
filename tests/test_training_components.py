"""测试训练组件（损失函数、优化器）的结构化配置构建功能。

验证 build_loss 和 build_optimizer 能正确处理两种配置格式：
1. 结构化字典格式：{"name": "smooth_l1", "beta": 0.25}，支持额外参数
2. 扁平字符串格式："mse"、"adam"，向后兼容

同时验证：
1. 自定义注册的损失函数也能通过结构化配置构建
2. 优化器参数（lr、weight_decay、momentum）被正确传递
"""

import unittest

import torch

from src.training import build_loss, build_optimizer, register_loss


class StructuredTrainingComponentConfigTests(unittest.TestCase):
    """测试结构化训练组件配置的构建兼容性。"""

    def test_builds_structured_smooth_l1_loss(self) -> None:
        """验证结构化字典格式能正确构建 SmoothL1Loss 并设置 beta 参数。

        使用 {"name": "smooth_l1", "beta": 0.25} 构建的损失函数
        应为 torch.nn.SmoothL1Loss 实例，且 beta 属性为 0.25。
        """
        loss = build_loss({"name": "smooth_l1", "beta": 0.25})

        self.assertIsInstance(loss, torch.nn.SmoothL1Loss)
        self.assertEqual(loss.beta, 0.25)

    def test_builds_structured_sgd_optimizer(self) -> None:
        """验证结构化字典格式能正确构建 SGD 优化器并传递所有参数。

        使用 {"name": "sgd", "weight_decay": 0.01, "momentum": 0.9}
        配合学习率 0.1，验证参数组的 lr、weight_decay、momentum 属性。
        """
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = build_optimizer(
            {"name": "sgd", "weight_decay": 0.01, "momentum": 0.9},
            [parameter],
            {"learning_rate": 0.1},
        )

        group = optimizer.param_groups[0]
        self.assertEqual(group["lr"], 0.1)
        self.assertEqual(group["weight_decay"], 0.01)
        self.assertEqual(group["momentum"], 0.9)

    def test_keeps_flat_string_config_compatible(self) -> None:
        """验证扁平字符串格式（旧版 API）仍然向后兼容。

        build_loss("mse") 应返回 torch.nn.MSELoss 实例；
        build_optimizer("adam", params, train_config) 应返回 torch.optim.Adam 实例。
        """
        parameter = torch.nn.Parameter(torch.tensor([1.0]))

        self.assertIsInstance(build_loss("mse"), torch.nn.MSELoss)
        self.assertIsInstance(
            build_optimizer("adam", [parameter], {"learning_rate": 0.001}),
            torch.optim.Adam,
        )

    def test_keeps_zero_argument_registered_loss_compatible(self) -> None:
        """验证自定义注册的零参数损失函数也能通过结构化配置等方式构建。

        注册一个名为 "legacy_test_loss" 的无参损失（L1Loss），
        通过 {"name": "legacy_test_loss"} 构建应返回 L1Loss 实例。
        """
        register_loss("legacy_test_loss", lambda: torch.nn.L1Loss(), replace=True)

        self.assertIsInstance(build_loss({"name": "legacy_test_loss"}), torch.nn.L1Loss)


if __name__ == "__main__":
    unittest.main()
