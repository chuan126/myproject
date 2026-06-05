"""数据降采样脚本入口。

从已处理的规范化 SOC 数据集创建降采样版本。
按照指定的时间间隔（interval_s）对时间序列进行降采样，
并重新计算衍生特征（如力学特征 delta_f、df_dt 等）。

使用方式：
    python scripts/downsample_data.py --input data/processed/my_dataset --output data/processed/my_dataset_10s --interval-s 10.0
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import downsample_canonical_dataset


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    必需参数：
        --input: 已处理的规范化数据集目录（包含 manifest.yaml）。
        --output: 降采样后的输出目录。
        --interval-s: 目标采样间隔（秒），必须大于原始数据的采样周期。

    可选参数：
        --overwrite: 是否覆盖已有输出。

    Returns:
        argparse.Namespace: 解析后的命令行参数。
    """
    parser = argparse.ArgumentParser(description="Downsample a processed canonical SOC dataset.")
    parser.add_argument("--input", type=Path, required=True, help="Source processed dataset directory.")
    parser.add_argument("--output", type=Path, required=True, help="Downsampled processed dataset directory.")
    parser.add_argument("--interval-s", type=float, required=True, help="Sampling interval in seconds.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """主入口：解析参数并执行降采样。

    调用 downsample_canonical_dataset 完成降采样，
    输出新的 CSV 文件和更新的 manifest.yaml。
    """
    args = parse_args()
    generated = downsample_canonical_dataset(
        args.input,
        args.output,
        interval_s=args.interval_s,
        overwrite=args.overwrite,
    )
    print(f"Generated {len(generated)} downsampled CSV file(s) in {args.output}.")


if __name__ == "__main__":
    main()
