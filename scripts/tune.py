"""使用 Optuna 搜索单个实验配置的超参数。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import BASE_CONFIG, prepare_training_data
from src.data import build_dataloaders
from src.experiment import select_device
from src.models import build_model
from src.training import Trainer, build_loss, build_optimizer
from src.utils import load_config, load_plugins, seed_everything, write_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune one SOC experiment with Optuna.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tuning"))
    return parser.parse_args()


def _set_by_path(values: dict, path: str, value: object) -> None:
    """按点分路径写入配置值。"""
    current = values
    keys = path.split(".")
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            raise KeyError(f"Cannot set tuning target '{path}': missing '{key}'")
        current = current[key]
    current[keys[-1]] = value


def suggest_params(trial: optuna.Trial, search_space: dict) -> dict:
    """根据配置中的搜索空间生成 Optuna trial 参数。"""
    params = {}
    for name, spec in search_space.items():
        param_type = spec["type"]
        if param_type == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif param_type == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        elif param_type == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported tuning parameter type '{param_type}' for '{name}'")
    return params


def apply_best_params(config: dict, params: dict, search_space: dict | None = None) -> dict:
    """将 Optuna 参数按 search_space.targets 写回配置。"""
    tuned = deepcopy(config)
    if search_space is None:
        search_space = tuned.get("tuning", {}).get("search_space", {})
    if not search_space:
        raise ValueError("Missing tuning.search_space in experiment config.")

    for name, value in params.items():
        if name not in search_space:
            raise KeyError(f"Parameter '{name}' is not defined in tuning.search_space.")
        targets = search_space[name].get("targets")
        if not targets:
            raise ValueError(f"Missing targets for tuning parameter '{name}'.")
        for target in targets:
            _set_by_path(tuned, target, value)
    return tuned


def train_trial(config: dict, output_dir: Path) -> float:
    """训练一次 trial，返回最佳验证损失。"""
    seed_everything(int(config["seed"]))
    load_plugins(config)
    device = select_device(config["train"].get("device", "auto"))
    bundle = build_dataloaders(config, ROOT)
    model = build_model(config["model"], bundle.input_dim).to(device)
    criterion = build_loss(config["train"]["loss"])
    optimizer = build_optimizer(config["train"].get("optimizer", "adam"), model.parameters(), config["train"])

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
        checkpoint_path=output_dir / "best.pt",
        checkpoint_context={"config": config, "input_dim": bundle.input_dim, "data_artifacts": bundle.artifacts},
    )
    return result.best_val_loss


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    study_dir = output_dir / args.study_name
    trial_root = study_dir / "trials"
    trial_root.mkdir(parents=True, exist_ok=True)

    base_config = load_config(BASE_CONFIG, config_path)
    tuning_config = base_config.get("tuning", {})
    search_space = tuning_config.get("search_space")
    if not search_space:
        raise ValueError("Missing tuning.search_space in experiment config.")
    prepare_training_data(base_config, prepared_datasets=set())

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, search_space)
        trial_config = apply_best_params(base_config, params, search_space)
        trial_config["experiment"]["name"] = f"tuning/{args.study_name}/trial_{trial.number:03d}"
        trial_dir = trial_root / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        write_yaml(trial_config, trial_dir / "config.yaml")
        val_loss = train_trial(trial_config, trial_dir)
        with (trial_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump({"best_val_loss": val_loss, "params": params}, file, ensure_ascii=False, indent=2)
        return val_loss

    study = optuna.create_study(study_name=args.study_name, direction=tuning_config.get("direction", "minimize"))
    study.optimize(objective, n_trials=args.trials)

    best_hyperparams = study.best_trial.params
    print(f"Best trial: {study.best_trial}")
    print("Best hyperparameters:", best_hyperparams)

    best_config = apply_best_params(base_config, best_hyperparams, search_space)
    write_yaml(best_config, study_dir / "best_config.yaml")
    with (study_dir / "best_params.json").open("w", encoding="utf-8") as file:
        json.dump(best_hyperparams, file, ensure_ascii=False, indent=2)

    with (study_dir / "trials.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["number", "value", *best_hyperparams.keys()])
        writer.writeheader()
        for trial in study.trials:
            writer.writerow({"number": trial.number, "value": trial.value, **trial.params})


if __name__ == "__main__":
    main()
