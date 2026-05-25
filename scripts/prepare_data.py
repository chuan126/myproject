"""将自有设备导出的 Excel 工作簿转换为规范化 SOC 数据集格式。"""

import argparse
from glob import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import prepare_cycler_workbooks


def parse_args() -> argparse.Namespace:
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
    """展开命令行输入路径或 glob，并保持确定性顺序。"""
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
