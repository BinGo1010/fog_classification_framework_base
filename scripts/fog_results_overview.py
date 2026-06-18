from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

PREFERRED_COLUMNS = [
    "updated_at",
    "sweep",
    "experiment",
    "feature_set",
    "model_name",
    "status",
    "fold_count",
    "test_f1_macro_mean",
    "test_f1_macro_std",
    "test_recall_macro_mean",
    "test_recall_macro_std",
    "test_pr_auc_macro_mean",
    "test_pr_auc_macro_std",
    "pre_fog_recall_mean",
    "pre_fog_recall_std",
    "pre_fog_f1_mean",
    "pre_fog_f1_std",
    "pre_fog_support_sum",
    "fog_recall_mean",
    "fog_f1_mean",
    "fog_support_sum",
    "normal_recall_mean",
    "normal_f1_mean",
    "normal_support_sum",
    "best_val_f1_macro_mean",
    "best_val_f1_macro_std",
    "test_balanced_accuracy_mean",
    "test_balanced_accuracy_std",
    "test_accuracy_mean",
    "test_accuracy_std",
    "confusion_matrix_test_sum",
    "cm_true_normal_pred_pre_fog",
    "cm_true_fog_pred_pre_fog",
    "window_seconds",
    "long_window_seconds",
    "stride_seconds",
    "pre_fog_seconds",
    "multi_window_mode",
    "trend_features",
    "small_kernel_size",
    "large_kernel_size",
    "short_kernel_size",
    "long_kernel_size",
    "input_channels",
    "raw_in_channels",
    "window_size",
    "long_window_size",
    "stride",
    "epochs",
    "batch_size",
    "returncode",
    "elapsed_sec",
    "config",
    "output_dir",
    "summary_path",
]

CLASS_NAMES_BY_COUNT = {
    2: ["normal", "fog"],
    3: ["normal", "pre_fog", "fog"],
}


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _stringify(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return None if number is None else int(number)


def _mean(values: list[float]) -> float | None:
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    values = [float(value) for value in values if value is not None]
    if len(values) < 2:
        return 0.0 if values else None
    avg = sum(values) / len(values)
    return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _class_key(class_index: int, class_count: int) -> str:
    names = CLASS_NAMES_BY_COUNT.get(int(class_count), [])
    if 0 <= int(class_index) < len(names):
        return names[int(class_index)]
    return f"class_{int(class_index)}"


def _metric_mean(aggregate: dict[str, Any], key: str) -> Any:
    value = aggregate.get(key)
    if isinstance(value, dict):
        return value.get("mean")
    return None


def _metric_std(aggregate: dict[str, Any], key: str) -> Any:
    value = aggregate.get(key)
    if isinstance(value, dict):
        return value.get("std")
    return None


def _resolve_existing_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = resolve_repo_path(value)
    return path if path.exists() else None


def _summary_path_for_row(row: dict[str, Any]) -> Path | None:
    summary_path = _resolve_existing_path(row.get("summary_path"))
    if summary_path is not None:
        return summary_path
    output_dir = _resolve_existing_path(row.get("output_dir"))
    if output_dir is not None:
        candidate = output_dir / "loso_summary.json"
        if candidate.exists():
            return candidate
    return None


def _output_dir_for_row(row: dict[str, Any], summary_path: Path | None) -> Path | None:
    output_dir = _resolve_existing_path(row.get("output_dir"))
    if output_dir is not None:
        return output_dir
    if summary_path is not None and summary_path.name == "loso_summary.json":
        return summary_path.parent
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _enrich_from_loso_summary(row: dict[str, Any], summary_path: Path | None) -> None:
    if summary_path is None:
        return
    summary = _read_json(summary_path)
    if not summary:
        return
    aggregate = summary.get("aggregate") or {}
    if isinstance(aggregate, dict):
        metric_map = {
            "test_f1_macro": "test_f1_macro_mean",
            "test_recall_macro": "test_recall_macro_mean",
            "test_pr_auc_macro": "test_pr_auc_macro_mean",
            "test_balanced_accuracy": "test_balanced_accuracy_mean",
            "test_accuracy": "test_accuracy_mean",
            "best_val_f1_macro": "best_val_f1_macro_mean",
            "test_pre_fog_recall": "pre_fog_recall_mean",
            "test_pre_fog_f1": "pre_fog_f1_mean",
            "test_pre_fog_support": "pre_fog_support_mean",
            "test_fog_recall": "fog_recall_mean",
            "test_fog_f1": "fog_f1_mean",
            "test_normal_recall": "normal_recall_mean",
            "test_normal_f1": "normal_f1_mean",
        }
        for source_key, target_key in metric_map.items():
            value = _metric_mean(aggregate, source_key)
            if value is not None:
                row[target_key] = value
            std_value = _metric_std(aggregate, source_key)
            if std_value is not None and target_key.endswith("_mean"):
                row[f"{target_key[:-5]}_std"] = std_value
    if summary.get("num_folds") is not None:
        row["fold_count"] = summary.get("num_folds")
    matrix = summary.get("confusion_matrix_test_sum")
    if matrix:
        _add_confusion_matrix(row, matrix)


def _read_per_class_file(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _per_class_files(output_dir: Path) -> list[Path]:
    files = sorted(output_dir.glob("loso_subject_*/per_class_metrics_test.csv"))
    direct = output_dir / "per_class_metrics_test.csv"
    if direct.exists():
        files.append(direct)
    return files


def _confusion_matrix_files(output_dir: Path) -> list[Path]:
    files = sorted(output_dir.glob("loso_subject_*/confusion_matrix_test.csv"))
    direct = output_dir / "confusion_matrix_test.csv"
    if direct.exists():
        files.append(direct)
    return files


def _read_confusion_matrix(path: Path) -> list[list[int]] | None:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            matrix = []
            for csv_row in reader:
                values = csv_row[1:] if len(csv_row) > 1 else csv_row
                matrix.append([int(float(value)) for value in values if value != ""])
    except OSError:
        return None
    return matrix or None


def _sum_matrices(matrices: list[list[list[int]]]) -> list[list[int]] | None:
    if not matrices:
        return None
    rows = len(matrices[0])
    cols = len(matrices[0][0]) if rows else 0
    total = [[0 for _ in range(cols)] for _ in range(rows)]
    for matrix in matrices:
        if len(matrix) != rows or any(len(row) != cols for row in matrix):
            continue
        for i in range(rows):
            for j in range(cols):
                total[i][j] += int(matrix[i][j])
    return total


def _add_confusion_matrix(row: dict[str, Any], matrix: list[list[int]]) -> None:
    row["confusion_matrix_test_sum"] = matrix
    class_count = len(matrix)
    for true_index, matrix_row in enumerate(matrix):
        true_key = _class_key(true_index, class_count)
        for pred_index, value in enumerate(matrix_row):
            pred_key = _class_key(pred_index, class_count)
            row[f"cm_true_{true_key}_pred_{pred_key}"] = int(value)


def _enrich_from_per_class_artifacts(row: dict[str, Any], output_dir: Path | None) -> None:
    if output_dir is None:
        return
    per_class_files = _per_class_files(output_dir)
    by_class: dict[int, list[dict[str, Any]]] = {}
    max_class_index = -1
    for path in per_class_files:
        for record in _read_per_class_file(path):
            class_index = int(float(record["class"]))
            max_class_index = max(max_class_index, class_index)
            by_class.setdefault(class_index, []).append(record)
    class_count = max_class_index + 1
    for class_index, records in by_class.items():
        class_key = _class_key(class_index, class_count)
        precisions = [_to_float(record.get("precision")) for record in records]
        recalls = [_to_float(record.get("recall_sensitivity") or record.get("recall")) for record in records]
        f1s = [_to_float(record.get("f1")) for record in records]
        supports = [_to_int(record.get("support")) or 0 for record in records]
        row[f"{class_key}_precision_mean"] = _mean([value for value in precisions if value is not None])
        row[f"{class_key}_recall_mean"] = _mean([value for value in recalls if value is not None])
        row[f"{class_key}_recall_std"] = _std([value for value in recalls if value is not None])
        row[f"{class_key}_f1_mean"] = _mean([value for value in f1s if value is not None])
        row[f"{class_key}_f1_std"] = _std([value for value in f1s if value is not None])
        row[f"{class_key}_support_sum"] = sum(supports)
        row[f"{class_key}_fold_count"] = len(records)

    matrices = [
        matrix
        for matrix in (_read_confusion_matrix(path) for path in _confusion_matrix_files(output_dir))
        if matrix is not None
    ]
    matrix_sum = _sum_matrices(matrices)
    if matrix_sum is not None:
        _add_confusion_matrix(row, matrix_sum)
        row["confusion_matrix_fold_count"] = len(matrices)


def enrich_overview_row(row: dict[str, Any]) -> dict[str, Any]:
    """Add acceptance-focused metrics from LOSO artifacts when they exist."""

    next_row = dict(row)
    summary_path = _summary_path_for_row(next_row)
    output_dir = _output_dir_for_row(next_row, summary_path)
    _enrich_from_loso_summary(next_row, summary_path)
    _enrich_from_per_class_artifacts(next_row, output_dir)
    return next_row


def _row_key(row: dict[str, Any]) -> str:
    for key in ("output_dir", "experiment", "config", "summary_path"):
        value = row.get(key)
        if value not in (None, ""):
            return f"{key}:{value}"
    raise ValueError("Overview row needs at least one of output_dir, experiment, config, or summary_path.")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    columns = [column for column in PREFERRED_COLUMNS if any(column in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = _columns(rows)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def update_overview(
    overview_csv: str | Path,
    row: dict[str, Any],
    *,
    sweep: str,
) -> Path:
    """Upsert one completed experiment row into the shared overview CSV."""

    path = resolve_repo_path(overview_csv)
    rows = _read_csv(path)
    next_row = enrich_overview_row(row)
    next_row = {key: _stringify(value) for key, value in next_row.items()}
    next_row["sweep"] = sweep
    next_row["updated_at"] = datetime.now().isoformat(timespec="seconds")
    next_key = _row_key(next_row)

    replaced = False
    for idx, existing in enumerate(rows):
        if _row_key(existing) == next_key:
            rows[idx] = {**existing, **next_row}
            replaced = True
            break
    if not replaced:
        rows.append(next_row)

    _write_csv_atomic(path, rows)
    return path


def update_overview_many(
    overview_csv: str | Path,
    rows: list[dict[str, Any]],
    *,
    sweep: str,
) -> Path:
    path = resolve_repo_path(overview_csv)
    for row in rows:
        path = update_overview(path, row, sweep=sweep)
    return path
