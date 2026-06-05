"""测试评估结果可视化图表生成功能。

验证以下行为：
1. 序列标识符中的特殊字符（如 :, ?, /）被安全替换为合法的文件名
2. 每序列 SOC 曲线图正确保存为 PNG 文件
3. 紧凑型实验图表（误差曲线、序列对比图、门控权重图）能正确生成
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.plots import (
    save_error_curve,
    save_gate_weights,
    save_soc_by_sequence,
    save_soc_curves_by_sequence,
)


class SocCurvePlotTests(unittest.TestCase):
    """测试 SOC 曲线和评估图表的生成功能。"""

    def _predictions(self) -> pd.DataFrame:
        """构造标准预测结果 DataFrame（用于紧凑图表测试）。

        包含 2 个序列 (seq_a, seq_b)，每个序列 2 个时间步，
        包含实际 SOC、预测 SOC 和误差列。

        Returns:
            预测结果 DataFrame。
        """
        return pd.DataFrame(
            {
                "sequence_id": ["seq_a", "seq_a", "seq_b", "seq_b"],
                "time": [0.0, 1.0, 0.0, 1.0],
                "actual_soc": [0.4, 0.5, 0.6, 0.7],
                "predicted_soc": [0.45, 0.48, 0.59, 0.72],
                "error": [0.05, -0.02, -0.01, 0.02],
            }
        )

    def test_sequence_identifiers_create_portable_output_names(self) -> None:
        """验证序列标识符中的特殊字符被安全地转换为合法的文件名。

        序列 ID "folder/a:b?" 包含路径分隔符和 Windows 非法字符，
        应被替换为安全字符；"CON" 是 Windows 保留名，应加下划线前缀。
        所有输出文件名不应包含 '<>:"/\\|?*' 中的任何字符。
        """
        predictions = pd.DataFrame(
            {
                "sequence_id": ["folder/a:b?", "CON"],
                "time": [0.0, 0.0],
                "actual_soc": [0.4, 0.5],
                "predicted_soc": [0.45, 0.55],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            save_soc_curves_by_sequence(predictions, output_dir)
            names = sorted(path.name for path in output_dir.glob("*.png"))

        self.assertEqual(len(names), 2)
        self.assertIn("_CON_soc_over_time.png", names)
        # 验证所有文件名不包含 Windows 非法字符
        self.assertTrue(all(not set('<>:"/\\|?*').intersection(name) for name in names))

    def test_compact_experiment_plots_are_saved(self) -> None:
        """验证紧凑型实验图表（误差曲线、序列对比图、门控权重图）能正确保存。

        包括三种图表：
        - soc_error.png：预测误差分布图
        - soc_by_sequence.png：各序列 SOC 预测对比散点图
        - gate_weights.png：双流模型门控权重变化图
        """
        predictions = self._predictions()
        # 构造门控权重数据（模拟双流模型中的 gate 值）
        gates = pd.DataFrame(
            {
                "sequence_id": ["seq_a", "seq_a", "seq_b", "seq_b"],
                "time": [0.0, 1.0, 0.0, 1.0],
                "gate_mean": [0.4, 0.5, 0.55, 0.6],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            save_error_curve(predictions, output_dir / "soc_error.png")
            save_soc_by_sequence(predictions, output_dir / "soc_by_sequence.png")
            save_gate_weights(gates, output_dir / "gate_weights.png")

            self.assertTrue((output_dir / "soc_error.png").exists())
            self.assertTrue((output_dir / "soc_by_sequence.png").exists())
            self.assertTrue((output_dir / "gate_weights.png").exists())


if __name__ == "__main__":
    unittest.main()
