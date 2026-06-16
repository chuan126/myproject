"""Tests for paper result table generation."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_paper_results import PAPER_TABLES, collect_results, write_paper_tables


class SummarizePaperResultsTests(unittest.TestCase):
    """Validate paper_tables.md generation from summary.json files."""

    def write_complete_run(self, run_dir: Path) -> None:
        """Create all expected summary.json files in a temporary run directory."""
        value = 0.1
        for table in PAPER_TABLES:
            for experiment in table.experiments:
                summary_dir = run_dir / table.table_dir / experiment
                summary_dir.mkdir(parents=True, exist_ok=True)
                (summary_dir / "summary.json").write_text(
                    json.dumps({"mae": value, "mse": value * value}, indent=2),
                    encoding="utf-8",
                )
                value += 0.01

    def test_writes_only_paper_tables_markdown(self) -> None:
        """Generate one compact Markdown artifact in the run directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "20260616_223804"
            self.write_complete_run(run_dir)

            output_path = write_paper_tables(run_dir)

            self.assertEqual(output_path, run_dir / "paper_tables.md")
            self.assertTrue(output_path.exists())
            self.assertFalse((run_dir / "paper_tables.xlsx").exists())
            content = output_path.read_text(encoding="utf-8")
            self.assertIn("# Paper Tables", content)
            self.assertIn("## 表 1 基线模型对比结果", content)
            self.assertIn("## 表 4 采样间隔适用性结果", content)
            self.assertIn("**0.100000**", content)

    def test_collect_results_requires_all_summaries(self) -> None:
        """Fail fast when one expected experiment summary is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "20260616_223804"
            self.write_complete_run(run_dir)
            missing = run_dir / "table4_downsampling" / "c4_30s_dsmi_li" / "summary.json"
            missing.unlink()

            with self.assertRaisesRegex(FileNotFoundError, "Missing summary.json"):
                collect_results(run_dir)

    def test_collect_results_requires_mae_and_mse(self) -> None:
        """Fail fast when summary.json does not contain the paper metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "20260616_223804"
            self.write_complete_run(run_dir)
            broken = run_dir / "table1_baselines" / "m1_lstm_uitf" / "summary.json"
            broken.write_text(json.dumps({"mae": 0.1}), encoding="utf-8")

            with self.assertRaisesRegex(KeyError, "mse"):
                collect_results(run_dir)


if __name__ == "__main__":
    unittest.main()
