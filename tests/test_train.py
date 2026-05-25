import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.train import has_processed_training_data, prepare_training_data, resolve_processed_data_paths
from src.data import raw_file_signature


class PrepareTrainingDataTests(unittest.TestCase):
    def write_processed_data(
        self,
        root: Path,
        rows_by_file: dict[str, list[tuple[str, str]]],
        raw_files: list[str] | None = None,
        raw_file_signatures: list[dict] | None = None,
    ) -> None:
        output_dir = root / "data" / "processed" / "own_cell"
        sequence_dir = output_dir / "sequences"
        sequence_dir.mkdir(parents=True)
        total_rows = sum(len(rows) for rows in rows_by_file.values())
        sequence_ids = {sequence_id for rows in rows_by_file.values() for _, sequence_id in rows}
        metadata = {
            "columns": ["time", "soc", "sequence_id"],
            "sequence_count": len(sequence_ids),
            "row_count": total_rows,
        }
        if raw_files is not None:
            metadata["raw_files"] = raw_files
        if raw_file_signatures is not None:
            metadata["raw_file_signatures"] = raw_file_signatures
        (output_dir / "manifest.yaml").write_text(
            yaml.safe_dump(metadata, sort_keys=False),
            encoding="utf-8",
        )
        for name, rows in rows_by_file.items():
            content = "time,soc,sequence_id\n" + "".join(
                f"{time},0.5,{sequence_id}\n" for time, sequence_id in rows
            )
            (sequence_dir / name).write_text(content, encoding="utf-8")

    def test_detects_existing_processed_dataset(self) -> None:
        data_config = {
            "path": "data/processed/own_cell/sequences/**/*.csv",
            "manifest": "data/processed/own_cell/manifest.yaml",
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch("scripts.train.ROOT", Path(temp_dir)):
            self.write_processed_data(Path(temp_dir), {"cycle_1.csv": [("0", "cycle_1")]})

            self.assertTrue(has_processed_training_data(data_config))

    def test_rejects_processed_dataset_with_missing_sequence_csv(self) -> None:
        data_config = {
            "path": "data/processed/own_cell/sequences/**/*.csv",
            "manifest": "data/processed/own_cell/manifest.yaml",
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch("scripts.train.ROOT", Path(temp_dir)):
            self.write_processed_data(
                Path(temp_dir),
                {
                    "cycle_1.csv": [("0", "cycle_1")],
                    "cycle_2.csv": [("0", "cycle_2")],
                },
            )
            (Path(temp_dir) / "data" / "processed" / "own_cell" / "sequences" / "cycle_2.csv").unlink()

            self.assertFalse(has_processed_training_data(data_config))

    def test_rejects_processed_dataset_with_missing_rows(self) -> None:
        data_config = {
            "path": "data/processed/own_cell/sequences/**/*.csv",
            "manifest": "data/processed/own_cell/manifest.yaml",
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch("scripts.train.ROOT", Path(temp_dir)):
            self.write_processed_data(
                Path(temp_dir),
                {"cycle_1.csv": [("0", "cycle_1"), ("1", "cycle_1")]},
            )
            sequence = Path(temp_dir) / "data" / "processed" / "own_cell" / "sequences" / "cycle_1.csv"
            sequence.write_text("time,soc,sequence_id\n0,0.5,cycle_1\n", encoding="utf-8")

            self.assertFalse(has_processed_training_data(data_config))

    def test_rejects_processed_dataset_with_invalid_manifest(self) -> None:
        data_config = {
            "path": "data/processed/own_cell/sequences/**/*.csv",
            "manifest": "data/processed/own_cell/manifest.yaml",
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch("scripts.train.ROOT", Path(temp_dir)):
            self.write_processed_data(Path(temp_dir), {"cycle_1.csv": [("0", "cycle_1")]})
            manifest = Path(temp_dir) / "data" / "processed" / "own_cell" / "manifest.yaml"
            manifest.write_text("columns: [time\n", encoding="utf-8")

            self.assertFalse(has_processed_training_data(data_config))

    def test_rejects_processed_dataset_when_raw_inputs_changed(self) -> None:
        data_config = {
            "path": "data/processed/own_cell/sequences/**/*.csv",
            "manifest": "data/processed/own_cell/manifest.yaml",
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch("scripts.train.ROOT", Path(temp_dir)):
            self.write_processed_data(
                Path(temp_dir),
                {"cycle_1.csv": [("0", "cycle_1")]},
                raw_files=[str(Path("a.xlsx").resolve())],
            )

            self.assertFalse(has_processed_training_data(data_config, [Path("b.xlsx")]))

    @patch("scripts.train.resolve_raw_paths")
    @patch("scripts.train.prepare_cycler_workbooks")
    def test_reuses_existing_processed_dataset_by_default(self, prepare_mock, resolve_mock) -> None:
        config = {
            "data": {
                "raw_path": "data/raw/*.xlsx",
                "path": "data/processed/own_cell/sequences/**/*.csv",
                "manifest": "data/processed/own_cell/manifest.yaml",
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch("scripts.train.ROOT", Path(temp_dir)):
            raw_path = Path(temp_dir) / "data" / "raw" / "a.xlsx"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_bytes(b"source-data")
            resolve_mock.return_value = [raw_path]
            self.write_processed_data(
                Path(temp_dir),
                {"cycle_1.csv": [("0", "cycle_1")]},
                raw_files=[str(raw_path)],
                raw_file_signatures=[raw_file_signature(raw_path)],
            )

            prepared = prepare_training_data(config)

        self.assertFalse(prepared)
        resolve_mock.assert_called_once_with("data/raw/*.xlsx")
        prepare_mock.assert_not_called()

    @patch("scripts.train.resolve_raw_paths")
    @patch("scripts.train.prepare_cycler_workbooks")
    def test_upgrades_legacy_manifest_signature_without_reprocessing(self, prepare_mock, resolve_mock) -> None:
        config = {
            "data": {
                "raw_path": "data/raw/*.xlsx",
                "path": "data/processed/own_cell/sequences/**/*.csv",
                "manifest": "data/processed/own_cell/manifest.yaml",
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch("scripts.train.ROOT", Path(temp_dir)):
            raw_path = Path(temp_dir) / "data" / "raw" / "a.xlsx"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_bytes(b"source-data")
            resolve_mock.return_value = [raw_path]
            self.write_processed_data(
                Path(temp_dir),
                {"cycle_1.csv": [("0", "cycle_1")]},
                raw_files=[str(raw_path)],
            )
            prepared = prepare_training_data(config)
            expected_signatures = [raw_file_signature(raw_path)]
            manifest = Path(temp_dir) / "data" / "processed" / "own_cell" / "manifest.yaml"
            metadata = yaml.safe_load(manifest.read_text(encoding="utf-8"))

        self.assertFalse(prepared)
        prepare_mock.assert_not_called()
        self.assertEqual(metadata["raw_file_signatures"], expected_signatures)

    @patch("scripts.train.resolve_raw_paths")
    @patch("scripts.train.prepare_cycler_workbooks")
    def test_rebuilds_incomplete_processed_dataset(self, prepare_mock, resolve_mock) -> None:
        resolve_mock.return_value = [Path("a.xlsx")]
        config = {
            "data": {
                "raw_path": "data/raw/*.xlsx",
                "path": "data/processed/own_cell/sequences/**/*.csv",
                "manifest": "data/processed/own_cell/manifest.yaml",
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch("scripts.train.ROOT", Path(temp_dir)):
            self.write_processed_data(
                Path(temp_dir),
                {
                    "cycle_1.csv": [("0", "cycle_1")],
                    "cycle_2.csv": [("0", "cycle_2")],
                },
            )
            (Path(temp_dir) / "data" / "processed" / "own_cell" / "sequences" / "cycle_2.csv").unlink()
            prepared = prepare_training_data(config)
            expected_output = (Path(temp_dir) / "data" / "processed" / "own_cell").resolve()

        self.assertTrue(prepared)
        prepare_mock.assert_called_once_with(
            [Path("a.xlsx")],
            expected_output,
            overwrite=True,
            extra_record_columns=(),
        )

    @patch("scripts.train.resolve_raw_paths")
    @patch("scripts.train.prepare_cycler_workbooks")
    def test_prepares_the_same_dataset_only_once_per_train_command(self, prepare_mock, resolve_mock) -> None:
        resolve_mock.return_value = [Path("a.xlsx")]
        config = {
            "data": {
                "raw_path": "data/raw/0.?C.xlsx",
                "manifest": "data/processed/own_cell/manifest.yaml",
            }
        }
        prepared_datasets: set[tuple] = set()

        with tempfile.TemporaryDirectory() as temp_dir, patch("scripts.train.ROOT", Path(temp_dir)):
            prepare_training_data(config, prepared_datasets)
            prepare_training_data(config, prepared_datasets)
            expected_output = (Path(temp_dir) / "data" / "processed" / "own_cell").resolve()

        prepare_mock.assert_called_once_with(
            [Path("a.xlsx")],
            expected_output,
            overwrite=True,
            extra_record_columns=(),
        )

    @patch("scripts.train.resolve_raw_paths")
    @patch("scripts.train.prepare_cycler_workbooks")
    def test_rebuilds_dataset_when_raw_content_changes(self, prepare_mock, resolve_mock) -> None:
        config = {
            "data": {
                "raw_path": "data/raw/*.xlsx",
                "path": "data/processed/own_cell/sequences/**/*.csv",
                "manifest": "data/processed/own_cell/manifest.yaml",
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch("scripts.train.ROOT", Path(temp_dir)):
            raw_path = Path(temp_dir) / "data" / "raw" / "a.xlsx"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_bytes(b"before")
            resolve_mock.return_value = [raw_path]
            self.write_processed_data(
                Path(temp_dir),
                {"cycle_1.csv": [("0", "cycle_1")]},
                raw_files=[str(raw_path)],
                raw_file_signatures=[raw_file_signature(raw_path)],
            )
            raw_path.write_bytes(b"after")
            prepared = prepare_training_data(config)
            expected_output = (Path(temp_dir) / "data" / "processed" / "own_cell").resolve()

        self.assertTrue(prepared)
        prepare_mock.assert_called_once_with(
            [raw_path],
            expected_output,
            overwrite=True,
            extra_record_columns=(),
        )

    def test_requires_manifest_for_automatic_preparation(self) -> None:
        with self.assertRaisesRegex(ValueError, "data.manifest is required"):
            prepare_training_data({"data": {"raw_path": "data/raw/*.xlsx"}})

    def test_dataset_name_infers_processed_paths(self) -> None:
        data_config = {"dataset_name": "new_dataset"}

        resolve_processed_data_paths(data_config)

        self.assertEqual(data_config["path"], "data\\processed\\new_dataset\\sequences\\**\\*.csv")
        self.assertEqual(data_config["manifest"], "data\\processed\\new_dataset\\manifest.yaml")

    def test_dataset_name_rejects_parent_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "single directory name"):
            resolve_processed_data_paths({"dataset_name": "../outside"})

    @patch("scripts.train.resolve_raw_paths")
    @patch("scripts.train.prepare_cycler_workbooks")
    def test_dataset_name_allows_preparation_without_manual_paths(self, prepare_mock, resolve_mock) -> None:
        resolve_mock.return_value = [Path("a.xlsx")]
        config = {"data": {"dataset_name": "new_dataset", "raw_path": "data/raw/*.xlsx"}}

        with tempfile.TemporaryDirectory() as temp_dir, patch("scripts.train.ROOT", Path(temp_dir)):
            prepare_training_data(config)
            expected_output = (Path(temp_dir) / "data" / "processed" / "new_dataset").resolve()

        prepare_mock.assert_called_once_with(
            [Path("a.xlsx")],
            expected_output,
            overwrite=True,
            extra_record_columns=(),
        )


if __name__ == "__main__":
    unittest.main()
