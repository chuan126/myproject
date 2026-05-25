"""将电池循环仪导出的 Excel 工作簿转换为规范化 SOC CSV 文件。"""

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd
import yaml
from tqdm.auto import tqdm

from ..schema import SEQUENCE_COLUMN, SOC_COLUMN, TIME_COLUMN, canonical_manifest


def raw_file_signature(path: Path) -> dict[str, Any]:
    """返回可用于识别原始文件内容变化的稳定指纹。"""
    resolved = Path(path).resolve()
    digest = sha256()
    with resolved.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _find_header(sheet: Any, required_tokens: Iterable[str]) -> tuple[int, list[str]]:
    for row_number, row in enumerate(sheet.iter_rows(values_only=True), start=1):
        headers = [str(value).strip() if value is not None else "" for value in row]
        if all(token in headers for token in required_tokens):
            return row_number, headers
    raise ValueError(f"Worksheet '{sheet.title}' has no expected header row: {list(required_tokens)}")


def _column(headers: list[str], candidates: Iterable[str]) -> int:
    for candidate in candidates:
        if candidate in headers:
            return headers.index(candidate)
    raise ValueError(f"Missing expected workbook column; accepted names: {list(candidates)}")


def _as_float(value: Any, column_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Workbook value in '{column_name}' is not numeric: {value!r}") from error


def _normalized_current(current: float, step_type: str) -> float:
    if "充电" in step_type or "charge" in step_type.lower():
        return abs(current)
    if "放电" in step_type or "discharge" in step_type.lower():
        return -abs(current)
    return current


def _read_temperature_map(workbook: Any, requested_ids: set[Any]) -> dict[Any, float]:
    if "auxTemp" not in workbook.sheetnames:
        raise ValueError("Cycler workbook must contain an auxTemp worksheet for temperature data.")
    sheet = workbook["auxTemp"]
    header_row, headers = _find_header(sheet, ["数据序号"])
    serial_index = _column(headers, ["数据序号", "record_id"])
    temperature_indices = [
        index
        for index, header in enumerate(headers)
        if ("温度" in header and "温差" not in header)
        or header.lower().startswith("temperature")
        or re.fullmatch(r"T\d+", header, re.IGNORECASE)
    ]
    if not temperature_indices:
        raise ValueError("auxTemp worksheet contains no temperature measurement columns.")
    temperatures: dict[Any, float] = {}
    for row in sheet.iter_rows(min_row=header_row + 1, values_only=True):
        serial = row[serial_index]
        if serial not in requested_ids:
            continue
        values = [float(row[index]) for index in temperature_indices if row[index] is not None]
        if values:
            temperatures[serial] = sum(values) / len(values)
    missing = requested_ids.difference(temperatures)
    if missing:
        raise ValueError(f"auxTemp worksheet is missing temperature values for {len(missing)} sampled records.")
    return temperatures


def convert_cycler_workbook(
    workbook_path: Path,
    extra_record_columns: Iterable[str] = (),
) -> dict[str, pd.DataFrame]:
    """从导出的工作簿中提取每个循环的规范化数据帧。

    执行 1 Hz 去重采样、安时积分、温度映射和基于充放电容量的 SOC 计算。
    """
    try:
        from openpyxl import load_workbook
    except ImportError as error:
        raise ImportError("Converting cycler workbooks requires the optional dependency openpyxl.") from error

    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    if "record" not in workbook.sheetnames:
        raise ValueError("Cycler workbook must contain a record worksheet.")
    record = workbook["record"]
    header_row, headers = _find_header(record, ["数据序号", "工步类型"])
    serial_index = _column(headers, ["数据序号", "record_id"])
    cycle_index = _column(headers, ["循环号", "cycle_id"])
    step_type_index = _column(headers, ["工步类型", "step_type"])
    time_index = _column(headers, ["总时间(s)", "time", "时间(s)"])
    current_index = _column(headers, ["电流(A)", "current"])
    voltage_index = _column(headers, ["电压(V)", "voltage"])
    extra_indices = {column: _column(headers, [column]) for column in extra_record_columns}

    sampled: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sampled_seconds: dict[str, set[int]] = defaultdict(set)
    totals: dict[str, dict[str, float]] = defaultdict(lambda: {"charge": 0.0, "discharge": 0.0})
    last_time: dict[str, float] = {}
    origin_time: dict[str, float] = {}
    selected_ids: set[Any] = set()

    for row in record.iter_rows(min_row=header_row + 1, values_only=True):
        if row[serial_index] is None:
            continue
        cycle = str(row[cycle_index])
        time_s = _as_float(row[time_index], headers[time_index])
        origin_time.setdefault(cycle, time_s)
        sequence_time = time_s - origin_time[cycle]
        step_type = str(row[step_type_index] or "")
        current = _normalized_current(_as_float(row[current_index], headers[current_index]), step_type)
        voltage = _as_float(row[voltage_index], headers[voltage_index])
        previous = last_time.get(cycle, time_s)
        delta_hours = max(0.0, time_s - previous) / 3600.0
        last_time[cycle] = time_s
        totals[cycle]["charge"] += max(current, 0.0) * delta_hours
        totals[cycle]["discharge"] += max(-current, 0.0) * delta_hours
        rounded_second = round(sequence_time)
        if rounded_second in sampled_seconds[cycle]:
            continue
        sampled_seconds[cycle].add(rounded_second)
        selected_ids.add(row[serial_index])
        values: dict[str, Any] = {
            "record_id": row[serial_index],
            TIME_COLUMN: float(rounded_second),
            "voltage": voltage,
            "current": current,
            "power": voltage * current,
            "cc_capacity": totals[cycle]["charge"] - totals[cycle]["discharge"],
            "step_type": step_type,
            "cycle_id": cycle,
            "_charge_ah": totals[cycle]["charge"],
            "_discharge_ah": totals[cycle]["discharge"],
        }
        for column, index in extra_indices.items():
            values[column] = row[index]
        sampled[cycle].append(values)

    temperature_map = _read_temperature_map(workbook, selected_ids)
    sequences: dict[str, pd.DataFrame] = {}
    for cycle, rows in sampled.items():
        charge_total = totals[cycle]["charge"]
        discharge_total = totals[cycle]["discharge"]
        if charge_total <= 0.0 or discharge_total <= 0.0:
            raise ValueError(f"Cycle {cycle} must contain both charging and discharging records to derive SOC.")
        for values in rows:
            values["temperature"] = temperature_map[values["record_id"]]
            step_type = str(values["step_type"])
            if "充电" in step_type or "charge" in step_type.lower():
                soc = values["_charge_ah"] / charge_total
            elif "放电" in step_type or "discharge" in step_type.lower():
                soc = 1.0 - values["_discharge_ah"] / discharge_total
            elif values["_charge_ah"] <= 0.0:
                soc = 0.0
            elif values["_discharge_ah"] <= 0.0:
                soc = 1.0
            else:
                soc = 1.0 - values["_discharge_ah"] / discharge_total
            values[SOC_COLUMN] = min(1.0, max(0.0, float(soc)))
            values[SEQUENCE_COLUMN] = f"{workbook_path.stem}_cycle_{cycle}"
            values["source"] = "cycler_workbook"
            del values["_charge_ah"]
            del values["_discharge_ah"]
        sequences[cycle] = pd.DataFrame(rows)
    workbook.close()
    if not sequences:
        raise ValueError(f"Cycler workbook contains no usable record rows: {workbook_path}")
    return sequences


def prepare_cycler_workbook(
    workbook_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    extra_record_columns: Iterable[str] = (),
    show_progress: bool = True,
) -> list[Path]:
    """将工作簿各循环的规范化序列写入 CSV，并生成 manifest。"""
    return prepare_cycler_workbooks(
        [workbook_path],
        output_dir,
        overwrite=overwrite,
        extra_record_columns=extra_record_columns,
        show_progress=show_progress,
    )


def prepare_cycler_workbooks(
    workbook_paths: Iterable[Path],
    output_dir: Path,
    overwrite: bool = False,
    extra_record_columns: Iterable[str] = (),
    show_progress: bool = True,
) -> list[Path]:
    """将多个工作簿合并转换为一个规范化数据集，并生成统一 manifest。"""
    paths = list(dict.fromkeys(Path(path) for path in workbook_paths))
    if not paths:
        raise ValueError("At least one cycler workbook path is required.")

    frames: list[pd.DataFrame] = []
    outputs: list[tuple[Path, pd.DataFrame]] = []
    output_paths: set[Path] = set()
    generated: list[Path] = []
    progress = tqdm(paths, desc="Converting Excel workbooks", unit="file", disable=not show_progress)
    for workbook_path in progress:
        progress.set_postfix_str(workbook_path.name)
        sequences = convert_cycler_workbook(workbook_path, extra_record_columns=extra_record_columns)
        for cycle, frame in sequences.items():
            output_path = output_dir / "sequences" / f"{workbook_path.stem}_cycle_{cycle}.csv"
            if output_path in output_paths:
                raise ValueError(f"Multiple workbooks resolve to the same sequence output file: {output_path}")
            output_paths.add(output_path)
            outputs.append((output_path, frame))
            frames.append(frame)

    if overwrite:
        sequence_dir = output_dir / "sequences"
        for existing_path in sequence_dir.rglob("*.csv") if sequence_dir.exists() else ():
            if existing_path not in output_paths:
                existing_path.unlink()

    for output_path, frame in outputs:
        if overwrite or not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output_path, index=False)
            generated.append(output_path)

    combined = pd.concat(frames, ignore_index=True)
    metadata: dict[str, Any]
    if len(paths) == 1:
        metadata = {"raw_file": str(paths[0])}
    else:
        metadata = {"raw_files": [str(path) for path in paths]}
    metadata["raw_file_signatures"] = [raw_file_signature(path) for path in paths]
    manifest = canonical_manifest(
        dataset_name=paths[0].stem if len(paths) == 1 else output_dir.name,
        source_type="cycler_workbook",
        frame=combined,
        sampling_period_s=1.0,
        soc_method="cycle_charge_discharge_coulomb_counting",
        current_sign="charge_positive",
        **metadata,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False, allow_unicode=True)
    return generated
