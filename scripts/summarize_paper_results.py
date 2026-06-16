"""Generate manuscript-ready paper result tables from one paper run directory."""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class PaperTable:
    """Static table definition plus experiment mapping."""

    title: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    table_dir: str
    experiments: tuple[str, ...]
    note: str


PAPER_TABLES: tuple[PaperTable, ...] = (
    PaperTable(
        title="表 1 基线模型对比结果",
        headers=("模型", "输入", "编码与融合", "MAE", "MSE"),
        rows=(
            ("LSTM", "`U,I,T,F`", "LSTM"),
            ("Informer", "`U,I,T,F`", "Informer"),
            ("DSMI-LI", "`X_main + X_mech`", "LSTM + Informer + gated"),
        ),
        table_dir="table1_baselines",
        experiments=("m1_lstm_uitf", "m2_informer_uitf", "m3_dsmi_li"),
        note="表注建议：测试集为独立 `2000N` 初始预紧力数据；训练/验证集包含其它完整充放电序列，最优结果加粗。",
    ),
    PaperTable(
        title="表 2 输入特征消融结果",
        headers=("编号", "输入", "特征组", "MAE", "MSE"),
        rows=(
            ("A1", "`U,I,T`", "电-热基线"),
            ("A2", "`U,I,T,F`", "加入原始力信号"),
            ("A3", "`U,I,T,df_dq`", "加入关键力学导数"),
            ("A4", "`U,I,T,F,df_dq`", "原始力 + 关键导数"),
            ("A5", "`U,I,T,F,X_mech`", "完整特征"),
        ),
        table_dir="table2_feature_ablation",
        experiments=("a1_uit", "a2_uitf", "a3_uit_dfdq", "a4_uitf_dfdq", "a5_uitf_xmech"),
        note="表注建议：所有输入组合使用相同数据划分、训练策略和评价指标。",
    ),
    PaperTable(
        title="表 3 结构消融结果",
        headers=("编号", "结构", "主分支", "力学分支", "融合", "MAE", "MSE"),
        rows=(
            ("B1", "`LSTM_main`", "LSTM", "-", "-"),
            ("B2", "`LSTM_all`", "LSTM", "-", "-"),
            ("B3", "`Informer_all`", "Informer", "-", "-"),
            ("B4", "`Dual-stream concat`", "LSTM", "Informer", "concat"),
            ("B5", "`DSMI-LI`", "LSTM", "Informer", "gated"),
        ),
        table_dir="table3_structure_ablation",
        experiments=("b1_lstm_main", "b2_lstm_all", "b3_informer_all", "b4_dual_stream_concat", "b5_dsmi_li_gated"),
        note="表注建议：gated 融合中两个分支输出维度保持一致。",
    ),
    PaperTable(
        title="表 4 采样间隔适用性结果",
        headers=("采样间隔", "数据生成方式", "窗口长度", "MAE", "MSE"),
        rows=(
            ("`1s`", "原始 canonical CSV", "`20`"),
            ("`5s`", "降采样 canonical CSV", "`20`"),
            ("`10s`", "降采样 canonical CSV", "`20`"),
            ("`30s`", "降采样 canonical CSV", "`20`"),
        ),
        table_dir="table4_downsampling",
        experiments=("c1_1s_dsmi_li", "c2_5s_dsmi_li", "c3_10s_dsmi_li", "c4_30s_dsmi_li"),
        note="表注建议：窗口长度单位为时间步；降采样后需要重新计算力学导数特征；训练、验证和测试序列划分规则保持一致。",
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate paper_tables.md from paper experiment summaries.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Paper run directory, for example outputs/experiments/paper/20260616_223804.",
    )
    return parser.parse_args()


def _resolve_run_dir(run_dir: Path) -> Path:
    """Resolve a run directory relative to the project root."""
    return run_dir if run_dir.is_absolute() else ROOT / run_dir


def _load_metrics(summary_path: Path) -> dict[str, float]:
    """Load MAE and MSE metrics from one summary.json."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as file:
        summary: dict[str, Any] = json.load(file)

    metrics: dict[str, float] = {}
    for name in ("mae", "mse"):
        if name not in summary:
            raise KeyError(f"Metric '{name}' is missing in {summary_path}")
        value = float(summary[name])
        if not math.isfinite(value):
            raise ValueError(f"Metric '{name}' is not finite in {summary_path}: {summary[name]}")
        metrics[name] = value
    return metrics


def collect_results(run_dir: Path) -> dict[str, dict[str, float]]:
    """Collect metrics for all expected paper experiments."""
    resolved = _resolve_run_dir(run_dir)
    if not resolved.exists():
        raise FileNotFoundError(f"Run directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Run path is not a directory: {resolved}")

    results: dict[str, dict[str, float]] = {}
    for table in PAPER_TABLES:
        for experiment in table.experiments:
            summary_path = resolved / table.table_dir / experiment / "summary.json"
            results[experiment] = _load_metrics(summary_path)
    return results


def _format_metric(value: float, best_value: float) -> str:
    """Format one metric, bolding the best value in the table."""
    formatted = f"{value:.6f}"
    if math.isclose(value, best_value, rel_tol=0.0, abs_tol=1e-12):
        return f"**{formatted}**"
    return formatted


def render_table(table: PaperTable, results: dict[str, dict[str, float]]) -> str:
    """Render one Markdown table."""
    mae_values = [results[experiment]["mae"] for experiment in table.experiments]
    mse_values = [results[experiment]["mse"] for experiment in table.experiments]
    best_mae = min(mae_values)
    best_mse = min(mse_values)

    lines = [
        f"## {table.title}",
        "",
        "| " + " | ".join(table.headers) + " |",
        "|" + "|".join("---:" if header in {"MAE", "MSE"} else "---" for header in table.headers) + "|",
    ]
    for static_cells, experiment in zip(table.rows, table.experiments, strict=True):
        metrics = results[experiment]
        row = (
            *static_cells,
            _format_metric(metrics["mae"], best_mae),
            _format_metric(metrics["mse"], best_mse),
        )
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(["", table.note])
    return "\n".join(lines)


def render_paper_tables(run_dir: Path, results: dict[str, dict[str, float]]) -> str:
    """Render all paper tables into one Markdown document."""
    resolved = _resolve_run_dir(run_dir)
    sections = [
        "# Paper Tables",
        "",
        f"Source run: `{resolved}`",
        "",
    ]
    for table in PAPER_TABLES:
        sections.append(render_table(table, results))
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def write_paper_tables(run_dir: Path) -> Path:
    """Write paper_tables.md into the paper run directory."""
    resolved = _resolve_run_dir(run_dir)
    results = collect_results(resolved)
    content = render_paper_tables(resolved, results)
    output_path = resolved / "paper_tables.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    output_path = write_paper_tables(args.run_dir)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
