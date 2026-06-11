from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and value is not None
    }


def summarize(rows: Iterable[dict[str, Any]], metric_keys: Iterable[str]) -> dict[str, dict[str, float | None]]:
    rows = list(rows)
    summary = {}
    for key in metric_keys:
        values = [row.get(key) for row in rows]
        values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
        if not values:
            summary[key] = {"mean": None, "std": None}
            continue
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
