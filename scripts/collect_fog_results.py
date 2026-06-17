#!/usr/bin/env python
"""Collect FOG experiment aggregate metrics into one CSV/JSON table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect aggregate metrics from FOG experiment output directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "output_dirs",
        nargs="+",
        type=Path,
        help="Experiment output directories, or parent directories to scan recursively.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/multimodal_results_summary.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/multimodal_results_summary.json"),
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Find aggregate.json files recursively under each output directory.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_read_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def find_aggregate_paths(output_dirs: list[Path], recursive: bool) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    names = {"aggregate.json", "loso_summary.json"}
    for root in output_dirs:
        root = root.resolve()
        candidates = root.rglob("*.json") if recursive else root.glob("*.json")
        for path in candidates:
            if path.name not in names:
                continue
            path = path.resolve()
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return sorted(paths)


def parse_aggregate_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], int | None, float | None]:
    if "aggregate" in payload and isinstance(payload["aggregate"], dict):
        aggregate = payload["aggregate"]
        fold_count = len(payload.get("folds", [])) if isinstance(payload.get("folds"), list) else None
        elapsed_sec = payload.get("elapsed_sec")
        return aggregate, fold_count, elapsed_sec
    return payload, None, None


def infer_experiment_dirs(aggregate_path: Path) -> tuple[Path, str, Path | None]:
    parent = aggregate_path.parent
    if (parent / "summary.csv").exists() and (parent / "config.json").exists():
        return parent, parent.name, parent / "summary.csv"
    if (parent / "summary.csv").exists() and (parent.parent / "config.json").exists():
        return parent.parent, parent.name, parent / "summary.csv"
    return parent, parent.name, parent / "summary.csv" if (parent / "summary.csv").exists() else None


def metric_columns(aggregate: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for metric, stats in sorted(aggregate.items()):
        if isinstance(stats, dict):
            for stat in ("mean", "std", "min", "max"):
                if stat in stats:
                    row[f"{metric}_{stat}"] = stats[stat]
        elif isinstance(stats, (int, float)) or stats is None:
            row[metric] = stats
    return row


def summary_fold_count(summary_path: Path | None) -> int | None:
    if summary_path is None or not summary_path.exists():
        return None
    try:
        return int(len(pd.read_csv(summary_path)))
    except Exception:
        return None


def data_config_from_training_config(config: dict[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    data_dir_value = config.get("data_dir")
    if data_dir_value is None:
        return None, {}
    data_dir = Path(str(data_dir_value))
    window_config = maybe_read_json(data_dir / "config.json") if data_dir.exists() else {}
    return data_dir, window_config


def class_names_from_config(training_config: dict[str, Any], window_config: dict[str, Any]) -> list[str]:
    classes = training_config.get("class_names")
    if classes is None:
        classes = window_config.get("class_names")
    return [str(item) for item in classes] if isinstance(classes, list) else []


def input_channels_from_config(training_config: dict[str, Any], window_config: dict[str, Any]) -> int | None:
    value = training_config.get("input_channels")
    if value is not None:
        return int(value)
    feature_names = window_config.get("feature_names")
    if isinstance(feature_names, list):
        return int(len(feature_names))
    return None


def infer_trainer(experiment: str, variant: str, training_config: dict[str, Any]) -> str:
    model_name = str(training_config.get("model_name", ""))
    if model_name:
        return model_name
    text = f"{experiment} {variant}".lower()
    if "sleepyco" in text or variant in {"seq2one_gru", "seq2seq_gru", "seq2seq_tcn"}:
        return "sleepyco"
    if "tcn" in text or {"levels", "kernel_size"} & set(training_config):
        return "tcn"
    return "unknown"


def maybe_read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def yaml_loso_config(experiment_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    configs = sorted(experiment_dir.glob("loso_subject_*/config_resolved.yaml"))
    if not configs:
        return {}, {}
    cfg = maybe_read_yaml(configs[0])
    model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    data = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    windowing = data.get("windowing") if isinstance(data.get("windowing"), dict) else {}
    train = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    return {
        "model_name": model.get("name"),
        "class_names": None,
        "input_channels": model.get("in_channels"),
        "epochs": train.get("epochs"),
        "batch_size": data.get("batch_size"),
        "lr": train.get("lr"),
        "loss": train.get("loss"),
    }, {
        "window_seconds": (
            float(windowing["window_size"]) / float(windowing.get("sampling_rate_hz", 1))
            if windowing.get("window_size") is not None
            else None
        ),
        "stride_seconds": (
            float(windowing["stride"]) / float(windowing.get("sampling_rate_hz", 1))
            if windowing.get("stride") is not None
            else None
        ),
        "target_hz": windowing.get("sampling_rate_hz"),
        "target_len": windowing.get("window_size"),
        "label_mode": windowing.get("label_mode"),
        "pre_fog_seconds": windowing.get("pre_fog_seconds"),
    }


def collect_loso_summary(summary_path: Path) -> dict[str, Any]:
    payload = read_json(summary_path)
    aggregate, fold_count, elapsed_sec = parse_aggregate_payload(payload)
    experiment_dir = summary_path.parent
    training_config, window_config = yaml_loso_config(experiment_dir)
    row: dict[str, Any] = {
        "experiment": experiment_dir.name,
        "variant": "loso",
        "trainer": infer_trainer(experiment_dir.name, "loso", training_config),
        "output_dir": str(experiment_dir),
        "aggregate_path": str(summary_path),
        "summary_path": str(experiment_dir / "loso_summary.csv")
        if (experiment_dir / "loso_summary.csv").exists()
        else "",
        "fold_count": fold_count,
        "elapsed_sec": elapsed_sec,
        "data_dir": "",
        "class_names": "",
        "num_classes": None,
        "input_channels": input_channels_from_config(training_config, window_config),
        "window_seconds": window_config.get("window_seconds"),
        "stride_seconds": window_config.get("stride_seconds"),
        "target_hz": window_config.get("target_hz"),
        "target_len": window_config.get("target_len"),
        "label_mode": window_config.get("label_mode"),
        "pre_fog_seconds": window_config.get("pre_fog_seconds"),
        "epochs": training_config.get("epochs"),
        "batch_size": training_config.get("batch_size"),
        "lr": training_config.get("lr"),
        "loss": training_config.get("loss"),
    }
    row.update(metric_columns(aggregate))
    return row


def collect_one(aggregate_path: Path) -> dict[str, Any]:
    if aggregate_path.name == "loso_summary.json":
        return collect_loso_summary(aggregate_path)

    payload = read_json(aggregate_path)
    aggregate, fold_count_from_payload, elapsed_sec = parse_aggregate_payload(payload)
    experiment_dir, variant, summary_path = infer_experiment_dirs(aggregate_path)
    training_config = maybe_read_json(experiment_dir / "config.json")
    data_dir, window_config = data_config_from_training_config(training_config)
    class_names = class_names_from_config(training_config, window_config)
    trainer = infer_trainer(experiment_dir.name, variant, training_config)

    row: dict[str, Any] = {
        "experiment": experiment_dir.name,
        "variant": variant,
        "trainer": trainer,
        "output_dir": str(experiment_dir),
        "aggregate_path": str(aggregate_path),
        "summary_path": str(summary_path) if summary_path is not None and summary_path.exists() else "",
        "fold_count": fold_count_from_payload
        if fold_count_from_payload is not None
        else summary_fold_count(summary_path),
        "elapsed_sec": elapsed_sec,
        "data_dir": str(data_dir) if data_dir is not None else "",
        "class_names": "|".join(class_names),
        "num_classes": len(class_names) if class_names else None,
        "input_channels": input_channels_from_config(training_config, window_config),
        "window_seconds": window_config.get("window_seconds"),
        "stride_seconds": window_config.get("stride_seconds"),
        "target_hz": window_config.get("target_hz"),
        "target_len": window_config.get("target_len"),
        "label_mode": window_config.get("label_mode"),
        "pre_fog_seconds": window_config.get("pre_fog_seconds"),
    }
    row.update(metric_columns(aggregate))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    preferred = [
        "experiment",
        "variant",
        "trainer",
        "fold_count",
        "test_f1_macro_mean",
        "test_balanced_accuracy_mean",
        "test_accuracy_mean",
        "best_val_f1_macro_mean",
        "window_seconds",
        "pre_fog_seconds",
        "epochs",
        "batch_size",
    ]
    columns = [column for column in preferred if any(column in row for row in rows)]
    widths = {
        column: max(len(column), *(len(format_cell(row.get(column))) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(format_cell(row.get(column)).ljust(widths[column]) for column in columns))


def main() -> None:
    args = parse_args()
    aggregate_paths = find_aggregate_paths(args.output_dirs, args.recursive)
    if not aggregate_paths:
        raise FileNotFoundError("No aggregate.json files found.")

    rows = [collect_one(path) for path in aggregate_paths]
    write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print_table(rows)
    print(
        json.dumps(
            {
                "experiments": len(rows),
                "output_csv": str(args.output_csv.resolve()),
                "output_json": str(args.output_json.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
