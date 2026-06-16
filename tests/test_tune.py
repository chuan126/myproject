"""测试 Optuna 调参脚本的配置写回逻辑。"""

import importlib.util
import unittest

HAS_OPTUNA = importlib.util.find_spec("optuna") is not None

if HAS_OPTUNA:
    import optuna

    from scripts.tune import apply_best_params, suggest_params


@unittest.skipUnless(HAS_OPTUNA, "optuna is not installed")
class TuneConfigTests(unittest.TestCase):
    """验证 best params 能按项目配置结构写回。"""

    def test_apply_best_params_updates_single_stream_config(self) -> None:
        config = {
            "train": {"learning_rate": 0.001, "optimizer": {"name": "adam", "weight_decay": 0.0}},
            "model": {
                "name": "lstm",
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.0,
                "head": {"dropout": 0.0},
            },
        }
        search_space = {
            "learning_rate": {"targets": ["train.learning_rate"]},
            "weight_decay": {"targets": ["train.optimizer.weight_decay"]},
            "hidden_size": {"targets": ["model.hidden_size"]},
            "num_layers": {"targets": ["model.num_layers"]},
            "dropout": {"targets": ["model.dropout"]},
            "head_dropout": {"targets": ["model.head.dropout"]},
        }
        params = {
            "learning_rate": 0.0001,
            "weight_decay": 0.001,
            "hidden_size": 96,
            "num_layers": 3,
            "dropout": 0.2,
            "head_dropout": 0.3,
        }

        tuned = apply_best_params(config, params, search_space)

        self.assertEqual(tuned["train"]["learning_rate"], 0.0001)
        self.assertEqual(tuned["train"]["optimizer"]["weight_decay"], 0.001)
        self.assertEqual(tuned["model"]["hidden_size"], 96)
        self.assertEqual(tuned["model"]["num_layers"], 3)
        self.assertEqual(tuned["model"]["dropout"], 0.2)
        self.assertEqual(tuned["model"]["head"]["dropout"], 0.3)
        self.assertEqual(config["model"]["hidden_size"], 64)

    def test_apply_best_params_updates_dual_stream_encoders(self) -> None:
        config = {
            "train": {"learning_rate": 0.001, "optimizer": {"name": "adam", "weight_decay": 0.0}},
            "model": {
                "architecture": {"name": "dual_stream"},
                "main_branch": {"encoder": {"hidden_size": 64, "num_layers": 2, "dropout": 0.0}},
                "mech_branch": {"encoder": {"hidden_size": 64, "num_layers": 2, "dropout": 0.0}},
                "head": {"dropout": 0.0},
            },
        }
        search_space = {
            "learning_rate": {"targets": ["train.learning_rate"]},
            "weight_decay": {"targets": ["train.optimizer.weight_decay"]},
            "hidden_size": {
                "targets": [
                    "model.main_branch.encoder.hidden_size",
                    "model.mech_branch.encoder.hidden_size",
                ]
            },
            "num_layers": {
                "targets": [
                    "model.main_branch.encoder.num_layers",
                    "model.mech_branch.encoder.num_layers",
                ]
            },
            "dropout": {
                "targets": [
                    "model.main_branch.encoder.dropout",
                    "model.mech_branch.encoder.dropout",
                ]
            },
            "head_dropout": {"targets": ["model.head.dropout"]},
        }
        params = {
            "learning_rate": 0.0002,
            "weight_decay": 0.0003,
            "hidden_size": 128,
            "num_layers": 1,
            "dropout": 0.1,
            "head_dropout": 0.2,
        }

        tuned = apply_best_params(config, params, search_space)

        self.assertEqual(tuned["model"]["main_branch"]["encoder"]["hidden_size"], 128)
        self.assertEqual(tuned["model"]["mech_branch"]["encoder"]["hidden_size"], 128)
        self.assertEqual(tuned["model"]["main_branch"]["encoder"]["num_layers"], 1)
        self.assertEqual(tuned["model"]["mech_branch"]["encoder"]["dropout"], 0.1)
        self.assertEqual(tuned["model"]["head"]["dropout"], 0.2)

    def test_suggest_params_uses_configured_search_space(self) -> None:
        search_space = {
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
            "hidden_size": {"type": "categorical", "choices": [32, 64, 128]},
            "num_layers": {"type": "int", "low": 1, "high": 3},
        }
        trial = optuna.trial.FixedTrial(
            {"learning_rate": 0.0001, "hidden_size": 64, "num_layers": 2}
        )

        params = suggest_params(trial, search_space)

        self.assertEqual(
            params,
            {"learning_rate": 0.0001, "hidden_size": 64, "num_layers": 2},
        )


if __name__ == "__main__":
    unittest.main()
