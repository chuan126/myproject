"""评估脚本入口。

从已保存的训练检查点加载模型，在测试集（或外部数据）上评估模型性能。
支持两种评估模式：
1. 内部评估：使用训练时保存的数据划分，在测试集上评估
2. 外部评估：通过 --raw-input 和 --dataset-name 指定外部 Excel 文件，
   先转换为规范化格式再评估

评估结果包括：MSE、MAE、RMSE 等指标，以及误差曲线和 SOC 对比图。
"""

import argparse
import copy
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_dataloaders, build_evaluation_dataloader, prepare_cycler_workbooks
from src.experiment import evaluate_and_save, select_device, write_json
from src.models import build_model
from src.training import load_checkpoint
from src.utils import get_logger, load_plugins


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    必需参数：
        --checkpoint: 训练保存的检查点文件路径。

    可选参数（用于外部数据评估，需同时提供）：
        --raw-input: 外部 Excel 工作簿文件路径或 glob 模式。
        --dataset-name: 外部数据集名称，用于输出目录命名。
        --output-dir: 自定义输出目录，不指定时自动推断。

    Returns:
        argparse.Namespace: 解析后的命令行参数。
    """
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
    """展开外部 Excel 文件路径或 glob 模式（相对于项目根目录）。

    按字母序排序以保证确定性，并去重。

    Args:
        patterns: 文件路径或 glob 模式列表。

    Returns:
        去重后的文件绝对路径列表。

    Raises:
        FileNotFoundError: 某个模式未匹配到任何文件。
    """
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
    """将外部 Excel 工作簿转换为规范化 CSV，并构建仅指向新生成数据的评估配置。

    流程：
    1. 深拷贝原有配置以避免副作用
    2. 确定输出目录（data/processed/evaluation/<dataset_name>）
    3. 调用 prepare_cycler_workbooks 转换 Excel 到 CSV
    4. 修改配置指向新生成的 CSV 文件
    5. 移除 raw_path 和 split_column，因为外部数据全部作为测试集

    Args:
        config: 原始实验配置（来自检查点）。
        raw_inputs: 外部 Excel 文件路径或模式列表。
        dataset_name: 数据集名称，用于输出目录命名。

    Returns:
        修改后的评估配置字典，data 部分指向新生成的规范化数据。
    """
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
    # 外部评估不区分 train/val/test，全部作为测试数据
    external_config["data"]["split_column"] = None
    return external_config


def main() -> None:
    """评估主流程。

    执行步骤：
    1. 解析命令行参数
    2. 从检查点恢复模型、配置和数据产物
    3. 如果指定了外部数据，进行转换并构建评估 DataLoader
    4. 否则使用训练时的测试集 DataLoader
    5. 运行评估，计算 MSE/MAE/RMSE 等指标
    6. 保存评估结果（图表和 summary.json）
    """
    args = parse_args()
    logger = get_logger()
    device = select_device("auto")
    # 外部评估要求 --raw-input 和 --dataset-name 同时提供
    if bool(args.raw_input) != bool(args.dataset_name):
        raise ValueError("--raw-input and --dataset-name must be provided together for external evaluation.")

    # 加载检查点，从中恢复配置、数据产物和模型状态
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

    # 推断输出目录：默认在检查点同级目录下创建 evaluation 子目录
    checkpoint_root = (
        args.checkpoint.parent.parent
        if args.checkpoint.parent.name == "checkpoints"
        else args.checkpoint.parent
    )
    default_output_dir = checkpoint_root / "evaluation"
    if args.dataset_name:
        default_output_dir = default_output_dir / args.dataset_name
    output_dir = args.output_dir or default_output_dir
    # 执行评估并保存结果
    metrics = evaluate_and_save(model, bundle, device, output_dir)
    write_json(metrics, output_dir / "summary.json")
    # 输出关键指标
    logger.info("Evaluated checkpoint from epoch %d on %s.", checkpoint["epoch"], device)
    logger.info(
        "Test MSE: %.6f | Test MAE: %.6f | Test RMSE: %.6f",
        metrics["mse"],
        metrics["mae"],
        metrics["rmse"],
    )


if __name__ == "__main__":
    main()
