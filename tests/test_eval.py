import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.eval import prepare_external_evaluation_data


class PrepareExternalEvaluationDataTests(unittest.TestCase):
    @patch("scripts.eval.resolve_raw_paths")
    @patch("scripts.eval.prepare_cycler_workbooks")
    def test_uses_only_generated_files_for_external_evaluation(self, prepare_mock, resolve_mock) -> None:
        resolve_mock.return_value = [Path("new.xlsx")]
        config = {
            "data": {
                "raw_path": "data/raw/training.xlsx",
                "split_column": "split",
                "extra_record_columns": ["resistance"],
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir, patch("scripts.eval.ROOT", Path(temp_dir)):
            generated = Path(temp_dir) / "new_cycle_1.csv"
            prepare_mock.return_value = [generated]
            external = prepare_external_evaluation_data(config, ["data/raw/new.xlsx"], "new_set")
            expected_output = Path(temp_dir) / "data" / "processed" / "evaluation" / "new_set"

        prepare_mock.assert_called_once_with(
            [Path("new.xlsx")],
            expected_output,
            overwrite=True,
            extra_record_columns=("resistance",),
        )
        self.assertEqual(external["data"]["path"], [str(generated)])
        self.assertEqual(external["data"]["manifest"], str(expected_output / "manifest.yaml"))
        self.assertIsNone(external["data"]["split_column"])
        self.assertNotIn("raw_path", external["data"])


if __name__ == "__main__":
    unittest.main()
