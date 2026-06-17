from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

PREFERRED_COLUMNS = [
    "updated_at",
    "sweep",
    "experiment",
    "model_name",
    "status",
    "fold_count",
    "test_f1_macro_mean",
    "test_balanced_accuracy_mean",
    "test_accuracy_mean",
    "best_val_f1_macro_mean",
    "window_seconds",
    "long_window_seconds",
    "stride_seconds",
    "pre_fog_seconds",
    "multi_window_mode",
    "trend_features",
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


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _stringify(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    return value


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
    next_row = {key: _stringify(value) for key, value in row.items()}
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
