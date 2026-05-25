import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import yaml

from src.data.converters.cycler_workbook import prepare_cycler_workbook, prepare_cycler_workbooks


def _sequence(sequence_id: str) -> dict[str, pd.DataFrame]:
    return {
        "1": pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "soc": [0.0, 1.0],
                "sequence_id": [sequence_id, sequence_id],
                "voltage": [3.0, 4.2],
            }
        )
    }


class PrepareCyclerWorkbooksTests(unittest.TestCase):
    @patch("src.data.converters.cycler_workbook.tqdm")
    @patch("src.data.converters.cycler_workbook.convert_cycler_workbook")
    def test_shows_tqdm_progress_by_default(self, convert_mock, tqdm_mock) -> None:
        convert_mock.return_value = _sequence("a_cycle_1")
        progress = MagicMock()
        tqdm_mock.return_value = progress

        with tempfile.TemporaryDirectory() as temp_dir:
            workbook_path = Path(temp_dir) / "a.xlsx"
            workbook_path.write_bytes(b"a")
            progress.__iter__.return_value = iter([workbook_path])
            prepare_cycler_workbooks([workbook_path], Path(temp_dir), overwrite=True)

        tqdm_mock.assert_called_once_with(
            [workbook_path],
            desc="Converting Excel workbooks",
            unit="file",
            disable=False,
        )
        progress.set_postfix_str.assert_called_once_with("a.xlsx")

    @patch("src.data.converters.cycler_workbook.convert_cycler_workbook")
    def test_writes_one_manifest_for_multiple_workbooks(self, convert_mock) -> None:
        convert_mock.side_effect = [_sequence("a_cycle_1"), _sequence("b_cycle_1")]

        with tempfile.TemporaryDirectory() as temp_dir:
            a_path = Path(temp_dir) / "a.xlsx"
            b_path = Path(temp_dir) / "b.xlsx"
            a_path.write_bytes(b"a")
            b_path.write_bytes(b"b")
            output_dir = Path(temp_dir) / "own_cell"
            generated = prepare_cycler_workbooks(
                [a_path, b_path],
                output_dir,
                overwrite=True,
                show_progress=False,
            )
            with (output_dir / "manifest.yaml").open(encoding="utf-8") as file:
                manifest = yaml.safe_load(file)

        self.assertEqual(len(generated), 2)
        self.assertEqual(manifest["dataset_name"], "own_cell")
        self.assertEqual(manifest["sequence_count"], 2)
        self.assertEqual(manifest["row_count"], 4)
        self.assertEqual(manifest["raw_files"], [str(a_path), str(b_path)])
        self.assertEqual(len(manifest["raw_file_signatures"]), 2)
        self.assertEqual(manifest["raw_file_signatures"][0]["path"], str(a_path.resolve()))

    @patch("src.data.converters.cycler_workbook.convert_cycler_workbook")
    def test_overwrite_removes_stale_sequence_csv(self, convert_mock) -> None:
        convert_mock.return_value = _sequence("a_cycle_1")

        with tempfile.TemporaryDirectory() as temp_dir:
            workbook_path = Path(temp_dir) / "a.xlsx"
            workbook_path.write_bytes(b"a")
            output_dir = Path(temp_dir) / "own_cell"
            stale_path = output_dir / "sequences" / "removed_cycle_1.csv"
            stale_path.parent.mkdir(parents=True)
            stale_path.write_text("time,soc,sequence_id\n0,0.5,removed_cycle_1\n", encoding="utf-8")

            prepare_cycler_workbooks([workbook_path], output_dir, overwrite=True, show_progress=False)

            self.assertFalse(stale_path.exists())
            self.assertTrue((output_dir / "sequences" / "a_cycle_1.csv").exists())

    @patch("src.data.converters.cycler_workbook.convert_cycler_workbook")
    def test_single_workbook_api_keeps_raw_file_metadata(self, convert_mock) -> None:
        convert_mock.return_value = _sequence("a_cycle_1")

        with tempfile.TemporaryDirectory() as temp_dir:
            workbook_path = Path(temp_dir) / "a.xlsx"
            workbook_path.write_bytes(b"a")
            output_dir = Path(temp_dir) / "output"
            prepare_cycler_workbook(workbook_path, output_dir, overwrite=True, show_progress=False)
            with (output_dir / "manifest.yaml").open(encoding="utf-8") as file:
                manifest = yaml.safe_load(file)

        self.assertEqual(manifest["dataset_name"], "a")
        self.assertEqual(manifest["raw_file"], str(workbook_path))
        self.assertNotIn("raw_files", manifest)
        self.assertEqual(manifest["raw_file_signatures"][0]["path"], str(workbook_path.resolve()))


if __name__ == "__main__":
    unittest.main()
