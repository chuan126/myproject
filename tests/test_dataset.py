import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import _assign_splits, build_evaluation_dataloader


class AssignSplitsTests(unittest.TestCase):
    def test_rejects_unsupported_split_column_value(self) -> None:
        frame = pd.DataFrame(
            {
                "sequence_id": ["a", "b", "c", "d"],
                "split": ["train", "val", "test", "holdout"],
            }
        )

        with self.assertRaisesRegex(ValueError, "unsupported values.*holdout"):
            _assign_splits(frame, {"split_column": "split"}, seed=1, saved_assignment=None)

    def test_rejects_unsupported_saved_split_value(self) -> None:
        frame = pd.DataFrame({"sequence_id": ["a", "b", "c", "d"]})
        assignment = {"a": "train", "b": "val", "c": "test", "d": "holdout"}

        with self.assertRaisesRegex(ValueError, "unsupported values.*holdout"):
            _assign_splits(frame, {}, seed=1, saved_assignment=assignment)


class ExternalEvaluationDataTests(unittest.TestCase):
    def test_builds_all_external_sequences_as_test_with_saved_scaler(self) -> None:
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
            artifacts = {
                "scaler": {"mean": [10.0], "scale": [10.0]},
                "feature_columns": ["voltage"],
            }
            bundle = build_evaluation_dataloader(config, Path(temp_dir), artifacts)

        self.assertEqual(set(bundle.datasets), {"test"})
        self.assertEqual(bundle.datasets["test"].sequence_ids, ["new_a", "new_b"])
        np.testing.assert_allclose(
            bundle.datasets["test"].features[0].numpy().reshape(-1),
            np.asarray([0.0, 1.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
