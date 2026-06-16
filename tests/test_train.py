"""测试训练脚本中的数据处理预备和校验逻辑。

验证以下功能：
1. has_processed_training_data：校验规范化数据完整性
   - 正常检测完整数据集
   - 检测缺失的 CSV 文件
   - 检测行数不匹配
   - 检测损坏的 manifest
   - 检测原始文件变更（路径不同、内容签名不同）
2. prepare_training_data：按需触发数据准备
   - 复用已有完整数据（不重复转换）
   - 为旧版 manifest 补写签名
   - 重建不完整数据集
   - 同一训练命令中相同数据集仅处理一次
   - 原始文件内容变更时重新处理
3. resolve_processed_data_paths：根据 dataset_name 推演路径
   - 正确生成规范的 data/processed/<name> 路径
   - 拒绝包含父目录引用的 dataset_name
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.train import (
    has_processed_training_data,
    prepare_output_dir,
    prepare_training_data,
    resolve_experiment_output_dir,
    resolve_processed_data_paths,
)
from src.data import raw_file_signature


class PrepareTrainingDataTests(unittest.TestCase):
    """测试训练数据预备流程的完整校验和触发逻辑。"""

    def write_processed_data(
        self,
        root: Path,
        rows_by_file: dict[str, list[tuple[str, str]]],
        raw_files: list[str] | None = None,
        raw_file_signatures: list[dict] | None = None,
    ) -> None:
        """辅助方法：在临时目录中构造一个完整的规范化数据集目录结构。

        创建 data/processed/own_cell/ 目录，包含：
        - sequences/ 子目录中的 CSV 文件
        - manifest.yaml 元信息文件

        Args:
            root: 临时根目录。
            rows_by_file: 文件名到 (time, sequence_id) 行列表的映射。
            raw_files: 可选的原始文件路径列表，写入 manifest。
            raw_file_signatures: 可选的原始文件签名列表，写入 manifest。
        """
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
        """验证 has_processed_training_data 能正确识别完整的数据集。

        构造一个与 manifest 完全一致的规范化数据集，应返回 True。
        """
        data_config = {
            "path": "data/processed/own_cell/sequences/**/*.csv",
            "manifest": "data/processed/own_cell/manifest.yaml",
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch("scripts.train.ROOT", Path(temp_dir)):
            self.write_processed_data(Path(temp_dir), {"cycle_1.csv": [("0", "cycle_1")]})

            self.assertTrue(has_processed_training_data(data_config))

    def test_rejects_processed_dataset_with_missing_sequence_csv(self) -> None:
        """验证 manifest 记录的 CSV 文件缺失时返回 False。

        manifest 声明了 2 个序列，但删除了其中一个 CSV 文件，
        应检测到文件数不匹配。
        """
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
        """验证 CSV 文件行数少于 manifest 声明时返回 False。

        manifest 声明 2 行，但修改 CSV 只剩 1 行，应检测到不匹配。
        """
        data_config = {
            "path": "data/processed/own_cell/sequences/**/*.csv",
            "manifest": "data/processed/own_cell/manifest.yaml",
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch("scripts.train.ROOT", Path(temp_dir)):
            self.write_processed_data(
                Path(temp_dir),
                {"cycle_1.csv": [("0", "cycle_1"), ("1", "cycle_1")]},
            )
            # 覆盖写入，减少行数
            sequence = Path(temp_dir) / "data" / "processed" / "own_cell" / "sequences" / "cycle_1.csv"
            sequence.write_text("time,soc,sequence_id\n0,0.5,cycle_1\n", encoding="utf-8")

            self.assertFalse(has_processed_training_data(data_config))

    def test_rejects_processed_dataset_with_invalid_manifest(self) -> None:
        """验证 manifest 文件格式损坏时返回 False。

        将 manifest 内容写入为无效 YAML 格式，应捕获解析异常。
        """
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
        """验证原始输入文件路径与 manifest 记录不一致时返回 False。

        manifest 记录的原始文件是 a.xlsx，但当前传入的原始文件是 b.xlsx，
        应检测到路径不匹配。
        """
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
        """验证已有完整数据时不会重复调用数据转换。

        当 manifest 签名与实际原始文件一致时：
        - prepare_training_data 返回 False（未新建数据）
        - prepare_cycler_workbooks 未被调用
        """
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
        """验证对旧版 manifest（无 raw_file_signatures）自动升级签名而不重新处理。

        旧版 manifest 可能只有 raw_files 列表没有签名字段。
        此时应自动计算并补写 raw_file_signatures，不触发数据重新转换。
        """
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
            # 构造无签名的旧版数据集
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
        # 验证 manifest 已被升级，包含了签名
        self.assertEqual(metadata["raw_file_signatures"], expected_signatures)

    @patch("scripts.train.resolve_raw_paths")
    @patch("scripts.train.prepare_cycler_workbooks")
    def test_rebuilds_incomplete_processed_dataset(self, prepare_mock, resolve_mock) -> None:
        """验证数据集不完整时（缺失 CSV 文件）自动触发重建。

        manifest 声明了 2 个序列但只有 1 个 CSV 文件存在，
        prepare_training_data 应触发 prepare_cycler_workbooks 重新生成。
        """
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
        """验证在同一训练命令中，相同的数据集只处理一次（去重机制）。

        使用 prepared_datasets 集合跟踪已处理的数据集键值，
        第二次调用 prepare_training_data 时应跳过转换。
        """
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

        # 验证只调用了一次 prepare_cycler_workbooks
        prepare_mock.assert_called_once_with(
            [Path("a.xlsx")],
            expected_output,
            overwrite=True,
            extra_record_columns=(),
        )

    @patch("scripts.train.resolve_raw_paths")
    @patch("scripts.train.prepare_cycler_workbooks")
    def test_rebuilds_dataset_when_raw_content_changes(self, prepare_mock, resolve_mock) -> None:
        """验证原始文件内容变更后自动触发数据重建。

        先创建与 manifest 签名一致的数据集，然后修改原始文件内容，
        此时文件签名会变化，应检测到不一致并触发重新转换。
        """
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
            # 修改文件内容，使签名变化
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
        """验证配置了 raw_path 但未配置 manifest 时抛出 ValueError。

        自动数据准备需要 manifest 路径来定位输出位置和校验完整性。
        """
        with self.assertRaisesRegex(ValueError, "data.manifest is required"):
            prepare_training_data({"data": {"raw_path": "data/raw/*.xlsx"}})

    def test_dataset_name_infers_processed_paths(self) -> None:
        """验证 resolve_processed_data_paths 根据 dataset_name 正确生成默认路径。

        给定 dataset_name="new_dataset"，应生成：
        - path: data/processed/new_dataset/sequences/**/*.csv
        - manifest: data/processed/new_dataset/manifest.yaml
        """
        data_config = {"dataset_name": "new_dataset"}

        resolve_processed_data_paths(data_config)

        self.assertEqual(
            Path(data_config["path"]),
            Path("data") / "processed" / "new_dataset" / "sequences" / "**" / "*.csv",
        )
        self.assertEqual(
            Path(data_config["manifest"]),
            Path("data") / "processed" / "new_dataset" / "manifest.yaml",
        )

    def test_dataset_name_rejects_parent_paths(self) -> None:
        """验证 dataset_name 包含父目录引用（如 "../outside"）时抛出 ValueError。

        dataset_name 必须是单一目录名，不允许路径穿越以防止安全问题。
        """
        with self.assertRaisesRegex(ValueError, "single directory name"):
            resolve_processed_data_paths({"dataset_name": "../outside"})

    @patch("scripts.train.resolve_raw_paths")
    @patch("scripts.train.prepare_cycler_workbooks")
    def test_dataset_name_allows_preparation_without_manual_paths(self, prepare_mock, resolve_mock) -> None:
        """验证只配置 dataset_name（不配置 path/manifest）也能自动完成数据准备。

        当配置中只有 dataset_name 和 raw_path 时，
        系统应自动推断 path 和 manifest 路径，并正确执行数据转换。
        """
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

    def test_resolves_legacy_experiment_output_dir_without_run_name(self) -> None:
        """验证未配置 run_name 时保持旧版输出目录兼容。"""
        config = {
            "experiment": {"name": "paper/table1/m1_lstm"},
            "output": {"dir": "outputs/experiments"},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = resolve_experiment_output_dir(config, Path(temp_dir))

        self.assertEqual(
            output_dir,
            Path(temp_dir) / "outputs" / "experiments" / "paper" / "table1" / "m1_lstm",
        )

    @patch("scripts.train.datetime")
    def test_auto_run_name_adds_timestamp_subdirectory(self, datetime_mock) -> None:
        """验证 run_name: auto 会在实验名首段后插入时间戳目录。"""
        datetime_mock.now.return_value.strftime.return_value = "20260616_153000"
        config = {
            "experiment": {"name": "paper/table1/m1_lstm", "run_name": "auto"},
            "output": {"dir": "outputs/experiments"},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = resolve_experiment_output_dir(config, Path(temp_dir))

        self.assertEqual(config["experiment"]["run_name"], "20260616_153000")
        self.assertEqual(
            output_dir,
            Path(temp_dir)
            / "outputs"
            / "experiments"
            / "paper"
            / "20260616_153000"
            / "table1"
            / "m1_lstm"
        )

    def test_auto_run_name_can_be_shared_across_batch(self) -> None:
        """验证批量训练可为多个配置复用同一个自动运行名。"""
        config = {
            "experiment": {"name": "paper/table2/a1_uit", "run_name": "auto"},
            "output": {"dir": "outputs/experiments"},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = resolve_experiment_output_dir(config, Path(temp_dir), auto_run_name="20260616_210000")

        self.assertEqual(config["experiment"]["run_name"], "20260616_210000")
        self.assertEqual(
            output_dir,
            Path(temp_dir)
            / "outputs"
            / "experiments"
            / "paper"
            / "20260616_210000"
            / "table2"
            / "a1_uit",
        )

    def test_run_name_rejects_nested_paths(self) -> None:
        """验证 run_name 只能是单级目录名，避免把运行目录打散。"""
        config = {
            "experiment": {"name": "m1_lstm", "run_name": "../bad"},
            "output": {"dir": "outputs/experiments"},
        }

        with self.assertRaisesRegex(ValueError, "single directory name"):
            resolve_experiment_output_dir(config, Path("root"))

    def test_prepare_output_dir_blocks_existing_files_by_default(self) -> None:
        """验证默认不覆盖已有训练输出产物。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "experiment"
            output_dir.mkdir()
            (output_dir / "summary.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already contains files"):
                prepare_output_dir(output_dir, overwrite=False)

    def test_prepare_output_dir_allows_existing_files_when_overwrite_enabled(self) -> None:
        """验证显式开启 output.overwrite 后允许复用已有目录。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "experiment"
            output_dir.mkdir()
            (output_dir / "summary.json").write_text("{}", encoding="utf-8")

            prepare_output_dir(output_dir, overwrite=True)

            self.assertTrue(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
