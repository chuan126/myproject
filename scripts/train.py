"""从 YAML 配置文件训练 SOC 估计实验。"""

import argparse
import csv
import glob
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_dataloaders, prepare_cycler_workbooks, raw_file_signature
from src.evaluation import save_training_curve
from src.experiment import evaluate_and_save, select_device, write_json
from src.models import build_model
from src.training import Trainer, build_loss, build_optimizer, load_checkpoint
from src.utils import get_logger, load_config, load_plugins, seed_everything, write_yaml

BASE_CONFIG = ROOT / "configs" / "base" / "default.yaml"
DEFAULT_CONFIG_PATTERN = "configs/experiments/*.yaml"


def parse_args() -> argparse.Namespace:
    """解析一个或多个实验配置文件路径或 glob 模式。"""
    parser = argparse.ArgumentParser(description="Train SOC estimation experiment configurations.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[DEFAULT_CONFIG_PATTERN],
        help="YAML paths or glob patterns, for example configs/experiments/*.yaml.",
    )
    return parser.parse_args()


def resolve_config_paths(patterns: list[str]) -> list[Path]:
    """展开 glob 模式并按确定性顺序返回实验配置路径列表。"""
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


def resolve_raw_paths(patterns: str | list[str]) -> list[Path]:
    """展开相对于项目目录的原始 Excel 路径或 glob 模式。"""
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
    """根据 dataset_name 补全 canonical CSV 与 manifest 的默认路径。"""
    dataset_name = data_config.get("dataset_name")
    if not dataset_name:
        return
    name = str(dataset_name)
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("data.dataset_name must be a single directory name.")
    processed_dir = Path("data") / "processed" / name
    data_config.setdefault("path", str(processed_dir / "sequences" / "**" / "*.csv"))
    data_config.setdefault("manifest", str(processed_dir / "manifest.yaml"))


def has_processed_training_data(data_config: dict, raw_inputs: list[Path] | None = None) -> bool:
    """根据 manifest 校验已有 canonical CSV 产物是否完整。"""
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
            current_signatures = {item["path"].lower(): item for item in map(raw_file_signature, raw_inputs)}
            recorded_signatures = {
                str(item.get("path", "")).lower(): item
                for item in stored_signatures
                if isinstance(item, dict)
            }
            if current_signatures != recorded_signatures:
                return False
        elif any(path.stat().st_mtime_ns > manifest.stat().st_mtime_ns for path in raw_inputs):
            return False
    return True


def upgrade_manifest_signatures(data_config: dict, raw_inputs: list[Path] | None) -> None:
    """为旧版完整产物补写 raw 文件指纹，不重新转换 canonical CSV。"""
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
    """按需生成 canonical CSV；返回本次是否实际执行了离线处理。"""
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
    if has_processed_training_data(data_config, inputs):
        upgrade_manifest_signatures(data_config, inputs)
        return False
    output_dir = Path(manifest_path)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.parent.resolve()
    if inputs is None:
        inputs = resolve_raw_paths(raw_path)
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


def run_experiment(config_path: Path, prepared_datasets: set[tuple] | None = None) -> None:
    """训练并评估一个实验配置。"""
    config = load_config(BASE_CONFIG, config_path)
    load_plugins(config)
    seed_everything(int(config["seed"]))
    logger = get_logger()
    output_dir = ROOT / config["output"]["dir"] / config["experiment"]["name"]
    device = select_device(config["train"].get("device", "auto"))

    prepared = prepare_training_data(config, prepared_datasets)
    if config["data"].get("raw_path"):
        message = "Prepared canonical data" if prepared else "Using existing canonical data"
        logger.info("%s for experiment '%s'.", message, config["experiment"]["name"])
    bundle = build_dataloaders(config, ROOT)
    model = build_model(config["model"], bundle.input_dim).to(device)
    criterion = build_loss(config["train"]["loss"])
    optimizer = build_optimizer(config["train"].get("optimizer", "adam"), model.parameters(), config["train"])

    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(config, output_dir / "resolved_config.yaml")
    if "manifest" in bundle.artifacts:
        write_yaml(bundle.artifacts["manifest"], output_dir / "data_manifest.yaml")
    checkpoint_path = output_dir / "checkpoints" / "best.pt"

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

    write_json(result.history, output_dir / "history.json")
    save_training_curve(result.history, output_dir / "plots" / "training_curve.png")

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate_and_save(model, bundle, device, output_dir)
    summary = {"best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss, **metrics}
    write_json(summary, output_dir / "summary.json")

    logger.info("Experiment '%s' completed on %s.", config["experiment"]["name"], device)
    logger.info(
        "Best epoch: %d | Test MSE: %.6f | Test MAE: %.6f | Test RMSE: %.6f",
        result.best_epoch,
        metrics["mse"],
        metrics["mae"],
        metrics["rmse"],
    )


def main() -> None:
    """依次训练命令行选中的所有实验配置。"""
    args = parse_args()
    prepared_datasets: set[tuple] = set()
    for config_path in resolve_config_paths(args.configs):
        run_experiment(config_path, prepared_datasets)


if __name__ == "__main__":
    main()
