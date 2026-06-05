"""测试评估脚本 (scripts/eval.py) 的外部数据准备逻辑。

验证 prepare_external_evaluation_data 函数能：
1. 将外部 Excel 工作簿转换为规范化格式
2. 修改配置使其仅指向新生成的文件
3. 移除 split_column 和 raw_path（外部数据全部作为测试集）
4. 保留额外记录列（extra_record_columns）
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.eval import prepare_external_evaluation_data


class PrepareExternalEvaluationDataTests(unittest.TestCase):
    """测试外部评估数据的准备和配置修改逻辑。"""

    @patch("scripts.eval.resolve_raw_paths")
    @patch("scripts.eval.prepare_cycler_workbooks")
    def test_uses_only_generated_files_for_external_evaluation(self, prepare_mock, resolve_mock) -> None:
        """验证外部评估时配置正确指向新生成的文件，并移除训练相关配置。

        原始配置中包含 raw_path 和 split_column，
        外部评估时应：
        - 调用 prepare_cycler_workbooks 转换新的 Excel 文件
        - data.path 指向生成的新 CSV 文件
        - data.manifest 指向新生成的 manifest.yaml
        - data.split_column 被设为 None（不区分 train/val/test）
        - data.raw_path 被移除（不再需要原始路径）
        - extra_record_columns 被正确传递（转为 tuple 以用于缓存键）
        """
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
