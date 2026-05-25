import unittest

import torch

from src.training import build_loss, build_optimizer, register_loss


class StructuredTrainingComponentConfigTests(unittest.TestCase):
    def test_builds_structured_smooth_l1_loss(self) -> None:
        loss = build_loss({"name": "smooth_l1", "beta": 0.25})

        self.assertIsInstance(loss, torch.nn.SmoothL1Loss)
        self.assertEqual(loss.beta, 0.25)

    def test_builds_structured_sgd_optimizer(self) -> None:
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
        parameter = torch.nn.Parameter(torch.tensor([1.0]))

        self.assertIsInstance(build_loss("mse"), torch.nn.MSELoss)
        self.assertIsInstance(
            build_optimizer("adam", [parameter], {"learning_rate": 0.001}),
            torch.optim.Adam,
        )

    def test_keeps_zero_argument_registered_loss_compatible(self) -> None:
        register_loss("legacy_test_loss", lambda: torch.nn.L1Loss(), replace=True)

        self.assertIsInstance(build_loss({"name": "legacy_test_loss"}), torch.nn.L1Loss)


if __name__ == "__main__":
    unittest.main()
