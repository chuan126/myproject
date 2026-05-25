"""评估脚本入口。

加载已训练检查点，在测试集上评估并输出指标。
"""

import argparse
import copy
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_dataloaders, build_evaluation_dataloader, prepare_cycler_workbooks
from src.experiment import evaluate_and_save, select_device
from src.models import build_model
from src.training import load_checkpoint
from src.utils import get_logger, load_plugins


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Evaluate an SOC estimation checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--raw-input",
        nargs="+",
        default=None,
        help="External cycler Excel workbook paths or glob patterns to evaluate.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Name used for external processed data and evaluation outputs.",
    )
    return parser.parse_args()


def resolve_raw_paths(patterns: list[str]) -> list[Path]:
    """展开相对于项目目录的外部 Excel 文件路径或 glob 模式。"""
    resolved: list[Path] = []
    for value in patterns:
        pattern = str(Path(value) if Path(value).is_absolute() else ROOT / value)
        matches = sorted(Path(path).resolve() for path in glob.glob(pattern, recursive=True))
        if not matches:
            raise FileNotFoundError(f"No raw workbook files matched: {value}")
        for path in matches:
            if path not in resolved:
                resolved.append(path)
    return resolved


def prepare_external_evaluation_data(config: dict, raw_inputs: list[str], dataset_name: str) -> dict:
    """将外部 Excel 转换为 canonical CSV，并返回仅指向本次生成文件的评估配置。"""
    external_config = copy.deepcopy(config)
    output_dir = ROOT / "data" / "processed" / "evaluation" / dataset_name
    generated = prepare_cycler_workbooks(
        resolve_raw_paths(raw_inputs),
        output_dir,
        overwrite=True,
        extra_record_columns=tuple(config["data"].get("extra_record_columns", [])),
    )
    external_config["data"]["path"] = [str(path) for path in generated]
    external_config["data"]["manifest"] = str(output_dir / "manifest.yaml")
    external_config["data"].pop("raw_path", None)
    external_config["data"]["split_column"] = None
    return external_config


def main() -> None:
    """评估主流程。"""
    args = parse_args()
    logger = get_logger()
    device = select_device("auto")
    if bool(args.raw_input) != bool(args.dataset_name):
        raise ValueError("--raw-input and --dataset-name must be provided together for external evaluation.")

    # 加载检查点，重建数据和模型
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    load_plugins(config)
    if args.raw_input:
        logger.info("Preparing external evaluation dataset '%s'.", args.dataset_name)
        config = prepare_external_evaluation_data(config, args.raw_input, args.dataset_name)
        bundle = build_evaluation_dataloader(config, ROOT, checkpoint["data_artifacts"])
    else:
        bundle = build_dataloaders(config, ROOT, saved_artifacts=checkpoint["data_artifacts"])
    model = build_model(config["model"], int(checkpoint["input_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state"])

    # 评估并保存结果
    default_output_dir = args.checkpoint.parent.parent / "evaluation"
    if args.dataset_name:
        default_output_dir = default_output_dir / args.dataset_name
    output_dir = args.output_dir or default_output_dir
    metrics = evaluate_and_save(model, bundle, device, output_dir)
    logger.info("Evaluated checkpoint from epoch %d on %s.", checkpoint["epoch"], device)
    logger.info(
        "Test MSE: %.6f | Test MAE: %.6f | Test RMSE: %.6f",
        metrics["mse"],
        metrics["mae"],
        metrics["rmse"],
    )


if __name__ == "__main__":
    main()
