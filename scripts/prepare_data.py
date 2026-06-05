"""数据准备脚本入口。

将循环测试设备（cycler）导出的 Excel 工作簿转换为规范化的 SOC 数据集格式。
输出为标准的 CSV 文件（含 time、soc、sequence_id 等列），
同时生成 manifest.yaml 记录数据集元信息（序列数、行数、列名、原始文件签名等）。

使用方式：
    python scripts/prepare_data.py --input "data/raw/*.xlsx" --output data/processed/my_dataset
"""

import argparse
from glob import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import prepare_cycler_workbooks


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    必需参数：
        --input: 一个或多个 Excel 工作簿路径或 glob 模式。
        --output: 规范化数据集输出目录。

    可选参数：
        --overwrite: 是否覆盖已有输出。
        --extra-record-column: 额外保留的 record 工作表列名（可多次指定）。

    Returns:
        argparse.Namespace: 解析后的命令行参数。
    """
    parser = argparse.ArgumentParser(description="Prepare canonical SOC data from cycler Excel workbooks.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Workbook path(s) or glob pattern(s).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Canonical dataset output directory.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--extra-record-column",
        action="append",
        default=[],
        help="Retain an additional record worksheet column.",
    )
    return parser.parse_args()


def resolve_inputs(values: list[str]) -> list[Path]:
    """展开命令行输入路径或 glob 模式，保持确定性顺序。

    对每个输入值，先尝试 glob 展开；若无匹配则检查是否为直接路径。
    结果按字母序排序并去重，确保输出可重复。

    Args:
        values: 输入路径或 glob 模式列表。

    Returns:
        去重后的 Path 对象列表。

    Raises:
        FileNotFoundError: 某个模式未匹配到任何文件且路径不存在。
    """
    paths: list[Path] = []
    for value in values:
        matches = sorted(Path(path) for path in glob(value, recursive=True))
        if not matches:
            path = Path(value)
            if path.exists():
                matches = [path]
            else:
                raise FileNotFoundError(f"No input paths matched: {value}")
        for path in matches:
            if path not in paths:
                paths.append(path)
    return paths


def main() -> None:
    """主入口：解析参数并执行数据转换。

    调用 prepare_cycler_workbooks 完成 Excel -> CSV 转换，
    并输出生成的 CSV 文件数量。
    """
    args = parse_args()
    inputs = resolve_inputs(args.input)
    generated = prepare_cycler_workbooks(
        inputs,
        args.output,
        overwrite=args.overwrite,
        extra_record_columns=args.extra_record_column,
    )
    print(f"Generated {len(generated)} canonical CSV file(s) in {args.output}.")


if __name__ == "__main__":
    main()
