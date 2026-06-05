"""测试规范化 SOC 数据集的降采样功能。

验证以下行为：
1. 按照指定时间间隔对序列进行降采样
2. 降采样后自动重新计算 id 列、力学派生特征
3. 降采样数据集正确输出 CSV 文件和 manifest.yaml
4. manifest 中正确记录降采样元信息（source_type、sampling_period_s、source_dataset_name 等）
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from src.data.downsample import downsample_canonical_dataset, downsample_sequence_frame


def _frame(sequence_id: str = "seq_a") -> pd.DataFrame:
    """构造一个包含完整列的标准测试 DataFrame。

    包含 6 行数据（时间 0-5 秒，间隔 1 秒），涵盖所有规范列：
    id, time, voltage, current, power, cc_capacity, force, temperature,
    delta_f, delta_q, df_dt, df_dq, force_slope, soc, sequence_id。

    Args:
        sequence_id: 序列标识符。

    Returns:
        含有 6 行数据的测试 DataFrame。
    """
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "voltage": [3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
            "current": [1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
            "power": [3.0, 3.1, 3.2, -3.3, -3.4, -3.5],
            "cc_capacity": [0.0, 0.1, 0.2, 0.1, 0.0, -0.1],
            "force": [100.0, 101.0, 104.0, 109.0, 116.0, 125.0],
            "temperature": [25.0, 25.1, 25.2, 25.3, 25.4, 25.5],
            "delta_f": [0.0, 1.0, 4.0, 9.0, 16.0, 25.0],
            "delta_q": [0.0, 0.1, 0.2, 0.1, 0.0, -0.1],
            "df_dt": [0.0, 1.0, 3.0, 5.0, 7.0, 9.0],
            "df_dq": [0.0, 10.0, 30.0, -50.0, -70.0, -90.0],
            "force_slope": [0.0, 10.0, 20.0, 90.0, 0.0, -250.0],
            "soc": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "sequence_id": [sequence_id] * 6,
        }
    )


class DownsampleTests(unittest.TestCase):
    """测试降采样功能的单序列和完整数据集处理。"""

    def test_downsamples_on_time_grid_and_recomputes_ids_and_mechanical_features(self) -> None:
        """验证按时间网格降采样并重新计算 id 和力学特征。

        原始数据时间 [0,1,2,3,4,5]（1 秒间隔），降采样到 2 秒间隔：
        - 保留时间点 [0, 2, 4]，共 3 行
        - id 重新编号为 [1, 2, 3]
        - soc 取对应时间点的值 [0.0, 0.4, 0.8]
        - 力学特征（delta_f 等）相对于新采样点重新计算
        """
        sampled = downsample_sequence_frame(_frame(), interval_s=2)

        self.assertEqual(sampled["id"].tolist(), [1, 2, 3])
        self.assertEqual(sampled["time"].tolist(), [0.0, 2.0, 4.0])
        self.assertEqual(sampled["soc"].tolist(), [0.0, 0.4, 0.8])
        self.assertEqual(sampled["delta_f"].tolist(), [0.0, 4.0, 16.0])
        self.assertEqual(sampled["delta_q"].tolist(), [0.0, 0.2, 0.0])
        self.assertEqual(sampled["df_dt"].tolist(), [0.0, 2.0, 6.0])
        self.assertEqual(sampled["df_dq"].tolist(), [0.0, 20.0, -60.0])
        self.assertEqual(sampled["force_slope"].tolist(), [0.0, 20.0, 0.0])

    def test_writes_downsampled_dataset_and_manifest(self) -> None:
        """验证降采样数据集正确写入 CSV 文件和 manifest.yaml。

        构造含 2 个序列的原始数据集，降采样到 2 秒间隔后，验证：
        - 生成了 2 个 CSV 文件
        - manifest 中 dataset_name 包含降采样间隔标识
        - source_type 为 "canonical_csv_downsample"
        - sampling_period_s 更新为新的间隔
        - source_dataset_name 保留原始数据集名称
        - sequence_count 和 row_count 正确反映降采样后的数据
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "base"
            output_dir = root / "base_2s"
            sequence_dir = input_dir / "sequences"
            sequence_dir.mkdir(parents=True)
            _frame("seq_a").to_csv(sequence_dir / "seq_a.csv", index=False)
            _frame("seq_b").to_csv(sequence_dir / "seq_b.csv", index=False)
            (input_dir / "manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "dataset_name": "base",
                        "sampling_period_s": 1.0,
                        "soc_method": "test_soc",
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            generated = downsample_canonical_dataset(input_dir, output_dir, interval_s=2, overwrite=True)
            manifest = yaml.safe_load((output_dir / "manifest.yaml").read_text(encoding="utf-8"))

        self.assertEqual(len(generated), 2)
        self.assertEqual(manifest["dataset_name"], "base_2s")
        self.assertEqual(manifest["source_type"], "canonical_csv_downsample")
        self.assertEqual(manifest["sampling_period_s"], 2.0)
        self.assertEqual(manifest["source_dataset_name"], "base")
        # 验证原始 manifest 中的额外字段被保留
        self.assertEqual(manifest["soc_method"], "test_soc")
        self.assertEqual(manifest["sequence_count"], 2)
        self.assertEqual(manifest["row_count"], 6)


if __name__ == "__main__":
    unittest.main()
