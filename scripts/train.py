"""训练脚本入口。

从 YAML 实验配置文件读取参数，完成以下流程：
1. 解析配置文件路径（支持 glob 模式），展开为确定性的实验列表
2. 检查或生成规范化的 CSV 训练数据（从 Excel 原始文件转换）
3. 构建 DataLoader、模型、损失函数和优化器
4. 运行训练循环，支持早停（early stopping）和检查点保存
5. 在测试集上评估最佳模型，保存指标和摘要 JSON

支持在一次启动中依次训练多个实验配置，避免重复处理相同的数据集。
"""

import argparse
import csv
import glob
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_dataloaders, prepare_cycler_workbooks, raw_file_signature
from src.experiment import evaluate_and_save, select_device, write_json
from src.models import build_model
from src.training import Trainer, build_loss, build_optimizer, load_checkpoint
from src.utils import get_logger, load_config, load_plugins, seed_everything

# 基础配置文件，所有实验都会继承其中的默认值
BASE_CONFIG = ROOT / "configs" / "base" / "default.yaml"
# 默认的实验配置 glob 模式，当命令行未指定时可扫描所有实验
DEFAULT_CONFIG_PATTERN = "configs/experiments/*.yaml"
AUTO_RUN_NAME = "auto"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    支持传入一个或多个 YAML 配置文件路径或 glob 模式，
    用于批量执行多个实验。

    Returns:
        argparse.Namespace: 包含 --configs 参数列表的命名空间。
    """
    parser = argparse.ArgumentParser(description="Train SOC estimation experiment configurations.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[DEFAULT_CONFIG_PATTERN],
        help="YAML paths or glob patterns, for example configs/experiments/*.yaml.",
    )
    return parser.parse_args()


def resolve_config_paths(patterns: list[str]) -> list[Path]:
    """展开 glob 模式并按确定性顺序返回实验配置路径列表。

    对每个模式值，解析为相对于项目根目录的路径并展开 glob，
    结果按字母序排序以保证可重复性。同时验证文件必须为 YAML 格式。

    Args:
        patterns: 用户传入的路径模式列表（可包含 glob 通配符）。

    Returns:
        去重后的实验配置文件绝对路径列表。

    Raises:
        FileNotFoundError: 某个模式未匹配到任何文件。
        ValueError: 匹配到的文件不是 .yaml 或 .yml 后缀。
    """
    resolved: list[Path] = []
    for value in patterns:
        pattern = str(Path(value) if Path(value).is_absolute() else ROOT / value)
        matches = sorted(Path(path).resolve() for path in glob.glob(pattern, recursive=True))
        if not matches:
            raise FileNotFoundError(f"No experiment config files matched: {value}")
        for path in matches:
            if path.suffix.lower() not in {".yaml", ".yml"}:
                raise ValueError(f"Experiment config must be a YAML file: {path}")
            if path not in resolved:
                resolved.append(path)
    return resolved


def resolve_experiment_output_dir(
    config: dict,
    root: Path = ROOT,
    auto_run_name: str | None = None,
) -> Path:
    """根据实验配置解析训练输出目录。

    `experiment.name` 表示稳定的实验身份；可选的 `experiment.run_name`
    表示一次具体运行。当 run_name 为 "auto" 时自动生成时间戳目录名。

    Args:
        config: 完整实验配置字典。
        root: 项目根目录，测试中可注入临时目录。
        auto_run_name: 本次训练命令共享的自动运行名。

    Returns:
        训练输出目录的绝对路径。

    Raises:
        ValueError: run_name 不是单级目录名。
    """
    output_dir = Path(config["output"]["dir"])
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    experiment_path = Path(config["experiment"]["name"])

    run_name = config["experiment"].get("run_name")
    if run_name in (None, ""):
        return output_dir / experiment_path
    run_name = str(run_name)
    if run_name.lower() == AUTO_RUN_NAME:
        run_name = auto_run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        config["experiment"]["run_name"] = run_name
    run_path = Path(run_name)
    if run_path.name != run_name or run_name in {".", ".."}:
        raise ValueError("experiment.run_name must be a single directory name or 'auto'.")
    parts = experiment_path.parts
    if not parts:
        raise ValueError("experiment.name must not be empty.")
    return output_dir / parts[0] / run_name / Path(*parts[1:])


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """创建训练输出目录，并在未允许覆盖时阻止覆盖已有产物。"""
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already contains files: {output_dir}. "
            "Set experiment.run_name: auto for a new run, or set output.overwrite: true to reuse it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def resolve_raw_paths(patterns: str | list[str]) -> list[Path]:
    """展开原始 Excel 文件路径或 glob 模式（相对于项目根目录）。

    与 resolve_config_paths 类似，但用于原始工作簿文件而非配置文件。

    Args:
        patterns: 单个字符串或字符串列表，表示原始工作簿路径/模式。

    Returns:
        去重后的原始工作簿文件绝对路径列表。

    Raises:
        FileNotFoundError: 某个模式未匹配到任何文件。
    """
    values = [patterns] if isinstance(patterns, str) else patterns
    resolved: list[Path] = []
    for value in values:
        pattern = str(Path(value) if Path(value).is_absolute() else ROOT / value)
        matches = sorted(Path(path).resolve() for path in glob.glob(pattern, recursive=True))
        if not matches:
            raise FileNotFoundError(f"No raw workbook files matched: {value}")
        for path in matches:
            if path not in resolved:
                resolved.append(path)
    return resolved


def resolve_processed_data_paths(data_config: dict) -> None:
    """根据 dataset_name 自动补全规范化 CSV 与 manifest 的默认路径。

    当配置中指定了 dataset_name 但未显式给出 path 和 manifest 时，
    根据约定（data/processed/<dataset_name>/sequences/**/*.csv）
    自动填充这两个字段。

    Args:
        data_config: 数据配置字典（会被原地修改）。

    Raises:
        ValueError: dataset_name 不是有效的单一目录名。
    """
    dataset_name = data_config.get("dataset_name")
    if not dataset_name:
        return
    name = str(dataset_name)
    # 确保 dataset_name 是简单目录名，不包含路径分隔符
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("data.dataset_name must be a single directory name.")
    processed_dir = Path("data") / "processed" / name
    data_config.setdefault("path", str(processed_dir / "sequences" / "**" / "*.csv"))
    data_config.setdefault("manifest", str(processed_dir / "manifest.yaml"))


def has_processed_training_data(data_config: dict, raw_inputs: list[Path] | None = None) -> bool:
    """校验已有规范化 CSV 产物是否完整且与原始数据一致。

    通过对比 manifest 中记录的元信息（序列数、行数、列名、原始文件签名）
    与实际 CSV 文件内容，判断已处理数据是否可用。

    校验项目包括：
    1. manifest 文件是否存在且可解析
    2. CSV 文件数量是否与 manifest 中 sequence_count 一致
    3. 总行数是否与 manifest 中 row_count 一致
    4. CSV 列集合是否包含 manifest 中声明的所有列
    5. 如果传入了 raw_inputs，校验原始文件路径和签名是否匹配
    6. 如果 manifest 中有原始文件签名，逐项比较一致性
    7. 如果只有 raw_files 没有签名，则通过文件修改时间粗略判断

    Args:
        data_config: 数据配置字典，必须包含 path 和 manifest 键。
        raw_inputs: 可选的原始工作簿路径列表，用于校验数据是否过期。

    Returns:
        bool: 如果规范化数据完整且未过期返回 True，否则返回 False。
    """
    manifest_path = data_config.get("manifest")
    patterns = data_config.get("path")
    if not manifest_path or not patterns:
        return False
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    if not manifest.exists():
        return False
    values = [patterns] if isinstance(patterns, str) else patterns
    csv_paths = sorted(
        {
            Path(path)
            for value in values
            for path in glob.glob(
                str(Path(value) if Path(value).is_absolute() else ROOT / value),
                recursive=True,
            )
        }
    )
    if not csv_paths:
        return False
    # 解析 manifest YAML 文件
    try:
        with manifest.open("r", encoding="utf-8") as file:
            metadata = yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(metadata, dict):
        return False
    expected_sequences = metadata.get("sequence_count")
    expected_rows = metadata.get("row_count")
    expected_columns = set(metadata.get("columns", []))
    if not isinstance(expected_sequences, int) or not isinstance(expected_rows, int):
        return False
    if len(csv_paths) != expected_sequences:
        return False
    # 逐文件统计实际行数和序列 ID
    actual_rows = 0
    actual_sequences: set[str] = set()
    for csv_path in csv_paths:
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                columns = set(reader.fieldnames or [])
                if "sequence_id" not in columns or not expected_columns.issubset(columns):
                    return False
                for row in reader:
                    actual_rows += 1
                    actual_sequences.add(str(row["sequence_id"]))
        except (OSError, csv.Error, UnicodeDecodeError):
            return False
    if actual_rows != expected_rows or len(actual_sequences) != expected_sequences:
        return False
    # 校验原始文件的一致性
    stored_raw_files = metadata.get("raw_files")
    if stored_raw_files is None and metadata.get("raw_file"):
        stored_raw_files = [metadata["raw_file"]]
    if raw_inputs is not None and stored_raw_files:
        recorded = {str(Path(path).resolve()).lower() for path in stored_raw_files}
        current = {str(Path(path).resolve()).lower() for path in raw_inputs}
        if recorded != current:
            return False
        stored_signatures = metadata.get("raw_file_signatures")
        if stored_signatures:
            # 有签名时精确比较每个文件的签名
            current_signatures = {item["path"].lower(): item for item in map(raw_file_signature, raw_inputs)}
            recorded_signatures = {
                str(item.get("path", "")).lower(): item
                for item in stored_signatures
                if isinstance(item, dict)
            }
            if current_signatures != recorded_signatures:
                return False
        elif any(path.stat().st_mtime_ns > manifest.stat().st_mtime_ns for path in raw_inputs):
            # 无签名时回退到修改时间比较
            return False
    return True


def upgrade_manifest_signatures(data_config: dict, raw_inputs: list[Path] | None) -> None:
    """为旧版完整产物补写原始文件指纹到 manifest 中。

    当数据已存在但 manifest 中没有 raw_file_signatures 字段时，
    追加该字段以避免后续每次都要重新计算签名。此操作不触发数据重新转换。

    Args:
        data_config: 数据配置字典，必须包含 manifest 键。
        raw_inputs: 原始工作簿文件路径列表。
    """
    if not raw_inputs:
        return
    manifest = Path(data_config["manifest"])
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    with manifest.open("r", encoding="utf-8") as file:
        metadata = yaml.safe_load(file) or {}
    if metadata.get("raw_file_signatures"):
        return
    metadata["raw_file_signatures"] = [raw_file_signature(path) for path in raw_inputs]
    with manifest.open("w", encoding="utf-8") as file:
        yaml.safe_dump(metadata, file, sort_keys=False, allow_unicode=True)


def prepare_training_data(config: dict, prepared_datasets: set[tuple] | None = None) -> bool:
    """按需生成规范化 CSV 数据，避免重复处理相同数据集。

    检查配置中是否配置了 raw_path（原始 Excel 数据路径）：
    - 如果已有完整的规范化数据，直接复用
    - 如果没有或不完整，则调用 prepare_cycler_workbooks 进行转换
    - 使用 prepared_datasets 集合跟踪已处理的数据集，防止同一命令中重复处理

    Args:
        config: 完整实验配置字典。
        prepared_datasets: 可选集合，记录本次运行中已处理的数据集键值，
                           避免对同一数据集重复调用转换。

    Returns:
        bool: 本次是否实际执行了离线数据处理（True 表示新建了数据）。

    Raises:
        ValueError: 配置了 raw_path 但未配置 manifest。
    """
    data_config = config["data"]
    resolve_processed_data_paths(data_config)
    raw_path = data_config.get("raw_path")
    if not raw_path:
        return False
    manifest_path = data_config.get("manifest")
    if not manifest_path:
        raise ValueError("data.manifest is required when data.raw_path is configured.")
    inputs: list[Path] | None = None
    try:
        inputs = resolve_raw_paths(raw_path)
    except FileNotFoundError:
        inputs = None
    # 检查是否已有完整的已处理数据
    if has_processed_training_data(data_config, inputs):
        upgrade_manifest_signatures(data_config, inputs)
        return False
    output_dir = Path(manifest_path)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.parent.resolve()
    if inputs is None:
        inputs = resolve_raw_paths(raw_path)
    # 用 (输入文件, 输出目录, 额外列) 元组作为去重键
    extra_columns = tuple(data_config.get("extra_record_columns", []))
    key = (tuple(inputs), output_dir, extra_columns)
    if prepared_datasets is not None and key in prepared_datasets:
        return False
    prepare_cycler_workbooks(
        inputs,
        output_dir,
        overwrite=True,
        extra_record_columns=extra_columns,
    )
    if prepared_datasets is not None:
        prepared_datasets.add(key)
    return True


def run_experiment(
    config_path: Path,
    prepared_datasets: set[tuple] | None = None,
    auto_run_name: str | None = None,
) -> None:
    """训练并评估单个实验配置。

    完整流程：
    1. 加载配置（合并 base 和实验配置），加载插件
    2. 设置随机种子，创建输出目录
    3. 按需准备训练数据
    4. 构建 DataLoader、模型、损失函数、优化器
    5. 运行训练循环（含早停和检查点保存）
    6. 加载最佳模型，在测试集上评估并保存结果

    Args:
        config_path: 实验 YAML 配置文件的绝对路径。
        prepared_datasets: 可选集合，用于跨实验去重数据准备。
        auto_run_name: 本次训练命令共享的自动运行名。
    """
    config = load_config(BASE_CONFIG, config_path)
    load_plugins(config)
    seed_everything(int(config["seed"]))
    logger = get_logger()
    output_dir = resolve_experiment_output_dir(config, auto_run_name=auto_run_name)
    device = select_device(config["train"].get("device", "auto"))

    # 按需准备规范化数据
    prepared = prepare_training_data(config, prepared_datasets)
    if config["data"].get("raw_path"):
        message = "Prepared canonical data" if prepared else "Using existing canonical data"
        logger.info("%s for experiment '%s'.", message, config["experiment"]["name"])
    # 构建数据加载器
    bundle = build_dataloaders(config, ROOT)
    # 构建模型、损失函数和优化器
    model = build_model(config["model"], bundle.input_dim).to(device)
    criterion = build_loss(config["train"]["loss"])
    optimizer = build_optimizer(config["train"].get("optimizer", "adam"), model.parameters(), config["train"])

    overwrite_output = config["output"].get("overwrite", False)
    if not isinstance(overwrite_output, bool):
        raise ValueError("output.overwrite must be a boolean.")
    prepare_output_dir(output_dir, overwrite_output)
    logger.info("Writing experiment outputs to %s.", output_dir)
    checkpoint_path = output_dir / "best.pt"

    # 创建训练器并开始训练
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        patience=int(config["train"]["patience"]),
        min_delta=float(config["train"].get("min_delta", 0.0)),
    )
    result = trainer.fit(
        bundle.loaders["train"],
        bundle.loaders["val"],
        epochs=int(config["train"]["epochs"]),
        checkpoint_path=checkpoint_path,
        checkpoint_context={
            "config": config,
            "input_dim": bundle.input_dim,
            "data_artifacts": bundle.artifacts,
        },
    )

    # 加载最佳检查点并在测试集上评估
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate_and_save(model, bundle, device, output_dir)
    summary = {"best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss, **metrics}
    write_json(summary, output_dir / "summary.json")

    # 输出训练结果摘要
    logger.info("Experiment '%s' completed on %s.", config["experiment"]["name"], device)
    logger.info(
        "Best epoch: %d | Test MSE: %.6f | Test MAE: %.6f | Test RMSE: %.6f",
        result.best_epoch,
        metrics["mse"],
        metrics["mae"],
        metrics["rmse"],
    )


def main() -> None:
    """主入口：解析命令行参数并按顺序执行所有实验配置。

    使用 prepared_datasets 集合跟踪已处理的数据集，
    避免多个实验使用相同数据源时重复进行数据转换。
    """
    args = parse_args()
    prepared_datasets: set[tuple] = set()
    auto_run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    for config_path in resolve_config_paths(args.configs):
        run_experiment(config_path, prepared_datasets, auto_run_name=auto_run_name)


if __name__ == "__main__":
    main()
