"""端到端烟雾测试。

使用 synthetic（合成）数据验证从数据构建到模型训练再到推理的完整流程不会崩溃。
测试覆盖：
1. LSTM 模型的完整流程（数据→DataLoader→模型→训练 1 epoch→推理）
2. GRU + Attention 池化组合的构建和训练

这些测试不验证数值精度，只确保管道各组件能协同工作。
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data import DataBundle, SOCDataset, build_dataloaders
from src.data.window import build_windows
from src.models import build_model
from src.training import Trainer, build_loss, build_optimizer


def _make_synthetic_data(
    output_dir: Path,
    n_sequences: int = 4,
    seq_length: int = 50,
    n_features: int = 3,
    window_size: int = 10,
) -> dict:
    """生成 synthetic（合成）规范化 CSV 数据集和对应的实验配置。

    使用正弦波 + 噪声生成模拟的 SOC 曲线和特征列，
    以固定种子保证可重复性。

    Args:
        output_dir: 输出目录（临时目录）。
        n_sequences: 合成序列数量。
        seq_length: 每个序列的长度（时间步数）。
        n_features: 特征列数量（模拟电压、电流、噪声等）。
        window_size: 滑动窗口大小。

    Returns:
        完整的实验配置字典，包含 data、model、train 三个部分。
    """
    rng = np.random.default_rng(42)
    frames: list[pd.DataFrame] = []
    for seq_idx in range(n_sequences):
        t = np.arange(seq_length, dtype=np.float32)
        # 使用正弦函数 + 噪声生成模拟 SOC 曲线
        soc = 0.5 + 0.3 * np.sin(2 * np.pi * t / seq_length) + rng.normal(0, 0.02, seq_length)
        soc = np.clip(soc, 0.0, 1.0)
        # 从 SOC 派生模拟特征：电压型、电流型、纯噪声型
        features = np.column_stack([
            soc + rng.normal(0, 0.01, seq_length),
            0.5 * soc + rng.normal(0, 0.01, seq_length),
            rng.normal(0, 1, seq_length),
        ]).astype(np.float32)
        frame = pd.DataFrame(features, columns=[f"f{i}" for i in range(n_features)])
        frame["time"] = t
        frame["soc"] = soc.astype(np.float32)
        frame["sequence_id"] = f"synth_seq_{seq_idx}"
        frames.append(frame[["time", "soc", "sequence_id", *[f"f{i}" for i in range(n_features)]]])
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / "synth.csv", index=False)
    # 生成 manifest 元信息文件
    manifest = {
        "schema_version": 1,
        "dataset_name": "synth_smoke",
        "source_type": "synthetic",
        "columns": combined.columns.tolist(),
        "sequence_count": n_sequences,
        "row_count": len(combined),
        "sampling_period_s": 1.0,
    }
    with (output_dir / "manifest.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False, allow_unicode=True)

    return {
        "data": {
            "format": "canonical_csv",
            "path": str(output_dir / "synth.csv"),
            "manifest": str(output_dir / "manifest.yaml"),
            "feature_columns": [f"f{i}" for i in range(n_features)],
            "window_size": window_size,
            "stride": 1,
            "split_seed": 12,
            "split": {"train": 0.5, "val": 0.25, "test": 0.25},
            "num_workers": 0,
        },
        "model": {
            "name": "lstm",
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "pooling": {"name": "last"},
        },
        "train": {
            "batch_size": 8,
            "epochs": 1,
            "learning_rate": 0.001,
            "optimizer": {"name": "adam", "weight_decay": 0.0},
            "loss": {"name": "mse"},
            "patience": 5,
            "min_delta": 0.0,
            "device": "cpu",
        },
        "seed": 42,
    }


class SmokeTest(unittest.TestCase):
    """端到端集成测试：验证完整管道各个阶段能正常执行。"""

    def test_full_pipeline_one_epoch(self) -> None:
        """验证完整流程：构建数据→模型→训练 1 epoch→推理无崩溃。

        执行步骤：
        1. 从合成配置构建 DataLoader（train/val/test 三个数据集）
        2. 构建 LSTM 模型并确认有可训练参数
        3. 使用 MSE 损失和 Adam 优化器训练 1 个 epoch
        4. 载入检查点后执行一次推理，验证输出形状正确
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _make_synthetic_data(root)

            # 步骤 1：构建 DataLoader
            bundle = build_dataloaders(config, root)

            self.assertIsInstance(bundle, DataBundle)
            for split in ("train", "val", "test"):
                self.assertIn(split, bundle.loaders)
                self.assertIsInstance(bundle.datasets[split], SOCDataset)
                self.assertGreater(len(bundle.datasets[split]), 0)
            self.assertEqual(bundle.input_dim, 3)

            # 步骤 2：构建模型
            device = torch.device("cpu")
            model = build_model(config["model"], bundle.input_dim).to(device)
            self.assertGreater(sum(p.numel() for p in model.parameters()), 0)

            # 步骤 3：训练 1 个 epoch
            criterion = build_loss(config["train"]["loss"])
            optimizer = build_optimizer(
                config["train"]["optimizer"],
                model.parameters(),
                config["train"],
            )
            checkpoint = root / "checkpoint.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            trainer = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                patience=config["train"]["patience"],
            )
            result = trainer.fit(
                bundle.loaders["train"],
                bundle.loaders["val"],
                epochs=1,
                checkpoint_path=checkpoint,
                checkpoint_context={"config": config, "input_dim": bundle.input_dim, "data_artifacts": bundle.artifacts},
            )

            # 验证训练历史中有损失记录
            self.assertGreater(len(result.history["train_loss"]), 0)
            self.assertGreater(len(result.history["val_loss"]), 0)
            self.assertTrue(checkpoint.exists())

            # 步骤 4：推理验证
            model.eval()
            with torch.no_grad():
                for features, targets, _ in bundle.loaders["test"]:
                    output = model(features)
                    self.assertEqual(output.shape, (len(features), 1))
                    break

    def test_smoke_with_gru_and_attention(self) -> None:
        """验证 GRU 编码器 + Attention 池化的组合也能正常构建和训练。

        使用不同的模型架构（GRU 替代 LSTM，attention 替代 last pooling），
        确保模型注册和构建系统对不同架构组合具有鲁棒性。
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _make_synthetic_data(root)
            config["model"] = {
                "name": "gru",
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "pooling": {"name": "attention"},
            }

            bundle = build_dataloaders(config, root)
            device = torch.device("cpu")
            model = build_model(config["model"], bundle.input_dim).to(device)
            criterion = build_loss(config["train"]["loss"])
            optimizer = build_optimizer(
                config["train"]["optimizer"],
                model.parameters(),
                config["train"],
            )
            checkpoint = root / "checkpoint.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            trainer = Trainer(model, criterion, optimizer, device, patience=5)
            result = trainer.fit(
                bundle.loaders["train"],
                bundle.loaders["val"],
                epochs=1,
                checkpoint_path=checkpoint,
                checkpoint_context={"config": config, "input_dim": bundle.input_dim, "data_artifacts": bundle.artifacts},
            )

            self.assertGreater(len(result.history["train_loss"]), 0)


if __name__ == "__main__":
    unittest.main()
