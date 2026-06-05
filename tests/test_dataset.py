"""测试数据集拆分逻辑和外部评估数据加载功能。

验证以下行为：
1. 基于 sequence_id 规则的 train/val/test 拆分（split_rules）
2. 基于 CSV 中 split_column 的拆分
3. 拆分分配的确定性（给定种子后结果可重复）
4. 外部评估数据加载：所有序列作为 test 集，复用已保存的 scaler
"""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import _assign_splits, build_evaluation_dataloader


class AssignSplitsTests(unittest.TestCase):
    """测试 _assign_splits 函数的拆分逻辑。"""

    def test_assigns_splits_from_sequence_id_rules(self) -> None:
        """验证基于 sequence_id 通配符规则的拆分。

        规则：包含 "2000N" 的序列分配到 test 集；
        剩余的按 train=0.67、val=0.33 随机分配。
        给定种子 seed=1，验证分配结果具有确定性和正确的比例。
        """
        frame = pd.DataFrame(
            {
                "sequence_id": [
                    "25C_0.5C_0N",
                    "15C_0.5C_0N",
                    "35C_0.5C_2000N",
                    "45C_0.5C_3000N",
                    "15C_0.5C_3000N",
                    "35C_0.5C_0N",
                ]
            }
        )

        assigned, assignment = _assign_splits(
            frame,
            {
                "split_rules": {
                    "test": ["*2000N*"],
                    "remaining": {"train": 0.67, "val": 0.33},
                }
            },
            seed=1,
            saved_assignment=None,
        )

        # "35C_0.5C_2000N" 匹配通配符 "*2000N*"，应分配到 test
        self.assertEqual(assignment["35C_0.5C_2000N"], "test")
        # 其余 5 个序列按 67:33 随机分配
        remaining_assignment = {
            key: value for key, value in assignment.items() if key != "35C_0.5C_2000N"
        }
        self.assertEqual(list(remaining_assignment.values()).count("train"), 3)
        self.assertEqual(list(remaining_assignment.values()).count("val"), 2)
        # 验证 DataFrame 中 _split 列已正确写入
        self.assertEqual(assigned.loc[2, "_split"], "test")

    def test_rejects_unsupported_split_column_value(self) -> None:
        """验证对不支持的分割值（如 "holdout"）抛出 ValueError。

        split_column 中仅允许 train、val、test 三种值。
        """
        frame = pd.DataFrame(
            {
                "sequence_id": ["a", "b", "c", "d"],
                "split": ["train", "val", "test", "holdout"],
            }
        )

        with self.assertRaisesRegex(ValueError, "unsupported values.*holdout"):
            _assign_splits(frame, {"split_column": "split"}, seed=1, saved_assignment=None)

    def test_rejects_unsupported_saved_split_value(self) -> None:
        """验证已保存的分配中包含不支持的分割值时也抛出 ValueError。

        当使用 saved_assignment 参数恢复之前的拆分时，
        其中的值也需要是 train/val/test 之一。
        """
        frame = pd.DataFrame({"sequence_id": ["a", "b", "c", "d"]})
        assignment = {"a": "train", "b": "val", "c": "test", "d": "holdout"}

        with self.assertRaisesRegex(ValueError, "unsupported values.*holdout"):
            _assign_splits(frame, {}, seed=1, saved_assignment=assignment)


class ExternalEvaluationDataTests(unittest.TestCase):
    """测试外部评估数据的 DataLoader 构建。"""

    def test_builds_all_external_sequences_as_test_with_saved_scaler(self) -> None:
        """验证外部数据全部作为测试集，并使用已保存的 scaler 进行标准化。

        构造两个新序列（new_a, new_b），验证：
        - bundle.datasets 仅包含 "test"
        - 序列 ID 列表与输入一致
        - 特征值经过了与训练时相同的标准化处理
          (voltage=10 -> (10-10)/10=0, voltage=20 -> (20-10)/10=1)
        """
        frame = pd.DataFrame(
            {
                "time": [0.0, 1.0, 0.0, 1.0],
                "soc": [0.1, 0.2, 0.3, 0.4],
                "sequence_id": ["new_a", "new_a", "new_b", "new_b"],
                "voltage": [10.0, 20.0, 30.0, 40.0],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "external.csv"
            frame.to_csv(data_path, index=False)
            config = {
                "data": {
                    "format": "canonical_csv",
                    "path": [str(data_path)],
                    "feature_columns": ["voltage"],
                    "window_size": 2,
                    "stride": 1,
                    "num_workers": 0,
                },
                "train": {"batch_size": 8},
            }
            # 使用预训练的 scaler 参数（mean=10, scale=10）
            artifacts = {
                "scaler": {"mean": [10.0], "scale": [10.0]},
                "feature_columns": ["voltage"],
            }
            bundle = build_evaluation_dataloader(config, Path(temp_dir), artifacts)

        self.assertEqual(set(bundle.datasets), {"test"})
        self.assertEqual(bundle.datasets["test"].sequence_ids, ["new_a", "new_b"])
        # 标准化后的特征值：(10-10)/10=0, (20-10)/10=1
        np.testing.assert_allclose(
            bundle.datasets["test"].features[0].numpy().reshape(-1),
            np.asarray([0.0, 1.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
