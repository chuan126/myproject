"""测试滑动窗口构建器 (window.py) 和标准化器 (preprocess.py) 的功能。

验证以下行为：
Standardizer（Z-score 标准化器）：
1. fit 正确计算均值和标准差
2. 零方差特征处理（scale 设为 1.0 避免除零）
3. transform 正确执行 Z-score 标准化
4. to_dict/from_dict 序列化往返一致

WindowedData 构建器：
1. 从时间序列 DataFrame 正确构建滑动窗口
2. 窗口目标值取最后一个时间步的标签
3. 窗口不跨越序列边界（sequence_id 变化时重置）
4. 长度小于窗口的序列被正确跳过
5. 无可生成窗口时抛出异常
6. 无效参数（window_size <= 0, stride <= 0）被拒绝
7. 无时间列时按原始行序构建窗口
8. 特征列顺序严格遵循 feature_columns 参数
"""

import unittest

import numpy as np
import pandas as pd

from src.data.preprocess import Standardizer
from src.data.window import WindowedData, build_windows


class StandardizerTests(unittest.TestCase):
    """测试 Standardizer（Z-score 标准化器）的拟合、变换和序列化。"""

    def test_fit_computes_mean_and_std(self) -> None:
        """验证 fit 方法正确计算每列的均值和标准差。

        使用三行两列的简单数据：
        - 均值应为 [2.0, 20.0]
        - scale 应为每列的标准差（ddof=0，即总体标准差）
        """
        features = np.asarray([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]], dtype=np.float32)
        scaler = Standardizer.fit(features)

        np.testing.assert_allclose(scaler.mean, [2.0, 20.0])
        np.testing.assert_allclose(scaler.scale, features.std(axis=0))

    def test_fit_handles_zero_variance_feature(self) -> None:
        """验证零方差特征的 scale 被设为 1.0 以避免除零错误。

        第一列所有值为 5.0（方差为 0），scale 应为 1.0；
        第二列方差非零，scale 应 > 0。
        """
        features = np.asarray([[5.0, 0.0], [5.0, 2.0]], dtype=np.float32)
        scaler = Standardizer.fit(features)

        self.assertEqual(float(scaler.scale[0]), 1.0)
        self.assertGreater(float(scaler.scale[1]), 0.0)

    def test_transform_applies_zscore(self) -> None:
        """验证 transform 正确执行 (x - mean) / scale 的 Z-score 标准化。

        对 fit 所用的同一数据做变换，应与手动计算一致。
        """
        features = np.asarray([[0.0], [2.0], [4.0]], dtype=np.float32)
        scaler = Standardizer.fit(features)
        transformed = scaler.transform(features)

        expected = (features - scaler.mean) / scaler.scale
        np.testing.assert_allclose(transformed, expected)

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """验证 to_dict 和 from_dict 的序列化往返一致。

        fit 一个 scaler，序列化为字典再从字典恢复，
        恢复后的 scaler 对相同数据的 transform 结果应与原始一致。
        """
        features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        original = Standardizer.fit(features)
        restored = Standardizer.from_dict(original.to_dict())

        np.testing.assert_allclose(restored.mean, original.mean)
        np.testing.assert_allclose(restored.scale, original.scale)
        np.testing.assert_allclose(
            restored.transform(features),
            original.transform(features),
        )


class BuildWindowsTests(unittest.TestCase):
    """测试滑动窗口构建函数 build_windows 的各种场景。"""

    def _make_frame(self, rows: list[dict]) -> pd.DataFrame:
        """辅助方法：从字典列表构造 DataFrame。"""
        return pd.DataFrame(rows)

    def test_builds_simple_window(self) -> None:
        """验证从简单序列正确构建滑动窗口。

        3 行数据，window_size=2, stride=1：
        - 应生成 2 个窗口：窗口 0 覆盖索引 [0,1]，窗口 1 覆盖索引 [1,2]
        - 标签取窗口最后一个时间步的 soc 值
        - features 形状为 (2, 2, 1)：2 个窗口，窗口长度 2，1 个特征
        """
        frame = self._make_frame([
            {"time": 0, "soc": 0.1, "seq": "a", "v": 3.0},
            {"time": 1, "soc": 0.2, "seq": "a", "v": 3.1},
            {"time": 2, "soc": 0.3, "seq": "a", "v": 3.2},
        ])
        result = build_windows(
            frame,
            feature_columns=["v"],
            target_column="soc",
            sequence_column="seq",
            time_column="time",
            window_size=2,
            stride=1,
        )

        self.assertIsInstance(result, WindowedData)
        self.assertEqual(result.features.shape, (2, 2, 1))
        self.assertEqual(result.targets.shape, (2,))
        np.testing.assert_allclose(result.features[0], [[3.0], [3.1]])
        np.testing.assert_allclose(result.targets, [0.2, 0.3])
        self.assertEqual(result.sequence_ids, ["a", "a"])
        self.assertEqual(result.times, [1, 2])

    def test_window_target_takes_last_timestep(self) -> None:
        """验证窗口目标值取自窗口最后一个时间步。

        4 行数据，window_size=3, stride=2：
        - 只生成 1 个窗口（因为 4-3+1=2，stride=2 只能取 1 个）
        - 目标值为索引 2 的 soc=0.7
        """
        frame = self._make_frame([
            {"time": 0, "soc": 0.5, "seq": "x", "v": 1.0},
            {"time": 1, "soc": 0.6, "seq": "x", "v": 2.0},
            {"time": 2, "soc": 0.7, "seq": "x", "v": 3.0},
            {"time": 3, "soc": 0.8, "seq": "x", "v": 4.0},
        ])
        result = build_windows(
            frame,
            feature_columns=["v"],
            target_column="soc",
            sequence_column="seq",
            time_column="time",
            window_size=3,
            stride=2,
        )

        self.assertEqual(result.features.shape, (1, 3, 1))
        self.assertAlmostEqual(float(result.targets[0]), 0.7)
        self.assertEqual(result.times, [2])

    def test_respects_sequence_boundaries(self) -> None:
        """验证窗口不跨越不同 sequence_id 的边界。

        两个序列各 2 行，window_size=2：
        序列 "a" 索引 [0,1] 生成 1 个窗口，序列 "b" 索引 [2,3] 生成 1 个窗口。
        不应出现跨序列的窗口（如索引 [1,2]）。
        """
        frame = self._make_frame([
            {"time": 0, "soc": 0.1, "seq": "a", "v": 1.0},
            {"time": 1, "soc": 0.2, "seq": "a", "v": 1.1},
            {"time": 0, "soc": 0.3, "seq": "b", "v": 2.0},
            {"time": 1, "soc": 0.4, "seq": "b", "v": 2.1},
        ])
        result = build_windows(
            frame,
            feature_columns=["v"],
            target_column="soc",
            sequence_column="seq",
            time_column="time",
            window_size=2,
            stride=1,
        )

        self.assertEqual(result.features.shape, (2, 2, 1))
        self.assertEqual(result.sequence_ids, ["a", "b"])

    def test_skips_short_sequences(self) -> None:
        """验证长度小于窗口大小的序列被跳过。

        序列 "short" 只有 1 行，window_size=2 无法构造窗口，应被跳过；
        序列 "ok" 有 2 行，可以正常生成 1 个窗口。
        """
        frame = self._make_frame([
            {"time": 0, "soc": 0.1, "seq": "short", "v": 1.0},
            {"time": 0, "soc": 0.2, "seq": "ok", "v": 2.0},
            {"time": 1, "soc": 0.3, "seq": "ok", "v": 2.1},
        ])
        result = build_windows(
            frame,
            feature_columns=["v"],
            target_column="soc",
            sequence_column="seq",
            time_column="time",
            window_size=2,
            stride=1,
        )

        self.assertEqual(result.features.shape, (1, 2, 1))
        self.assertEqual(result.sequence_ids, ["ok"])

    def test_raises_when_no_windows_created(self) -> None:
        """验证所有序列都无法构造窗口时抛出 ValueError。

        只有 1 行数据，window_size=10，无法生成任何窗口。
        """
        frame = self._make_frame([
            {"time": 0, "soc": 0.1, "seq": "a", "v": 1.0},
        ])
        with self.assertRaisesRegex(ValueError, "No windows were created"):
            build_windows(
                frame,
                feature_columns=["v"],
                target_column="soc",
                sequence_column="seq",
                time_column="time",
                window_size=10,
                stride=1,
            )

    def test_rejects_invalid_window_size(self) -> None:
        """验证 window_size <= 0 时抛出 ValueError。

        window_size 必须为正整数（> 0）。
        """
        frame = self._make_frame([{"time": 0, "soc": 0.1, "seq": "a", "v": 1.0}])
        with self.assertRaisesRegex(ValueError, "positive"):
            build_windows(
                frame,
                feature_columns=["v"],
                target_column="soc",
                sequence_column="seq",
                time_column="time",
                window_size=0,
                stride=1,
            )

    def test_rejects_invalid_stride(self) -> None:
        """验证 stride <= 0 时抛出 ValueError。

        stride 必须为正整数（> 0）。
        """
        frame = self._make_frame([{"time": 0, "soc": 0.1, "seq": "a", "v": 1.0}])
        with self.assertRaisesRegex(ValueError, "positive"):
            build_windows(
                frame,
                feature_columns=["v"],
                target_column="soc",
                sequence_column="seq",
                time_column="time",
                window_size=1,
                stride=0,
            )

    def test_handles_missing_time_column(self) -> None:
        """验证无时间列时按 DataFrame 原始行序构建窗口。

        time_column=None 时不依赖时间信息，
        仅按行索引位置顺序构建窗口。
        """
        frame = pd.DataFrame({
            "soc": [0.1, 0.2, 0.3],
            "seq": ["a", "a", "a"],
            "v": [1.0, 2.0, 3.0],
        })
        result = build_windows(
            frame,
            feature_columns=["v"],
            target_column="soc",
            sequence_column="seq",
            time_column=None,
            window_size=2,
            stride=1,
        )

        self.assertEqual(result.features.shape, (2, 2, 1))
        np.testing.assert_allclose(result.targets, [0.2, 0.3])

    def test_explicit_column_order(self) -> None:
        """验证特征列顺序严格遵循 feature_columns 参数而非 DataFrame 列顺序。

        DataFrame 列顺序为 [v, i]，但 feature_columns=["i", "v"]，
        因此每个窗口的特征应为 [[i, v], [i, v]] 而非 [[v, i], [v, i]]。
        """
        frame = self._make_frame([
            {"time": 0, "soc": 0.1, "seq": "a", "v": 4.0, "i": -1.0},
            {"time": 1, "soc": 0.2, "seq": "a", "v": 5.0, "i": -2.0},
        ])
        result = build_windows(
            frame,
            feature_columns=["i", "v"],
            target_column="soc",
            sequence_column="seq",
            time_column="time",
            window_size=2,
        )

        np.testing.assert_allclose(result.features[0], [[-1.0, 4.0], [-2.0, 5.0]])


if __name__ == "__main__":
    unittest.main()
