#!/usr/bin/env python
"""Audit whether a FOG experiment suite produced complete aggregate results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.collect_fog_results import collect_one
except ImportError:  # pragma: no cover - used when executed from scripts/
    from collect_fog_results import collect_one


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check aggregate outputs after a JSON-configured FOG suite finishes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def experiment_config_path(entry: str | dict[str, Any]) -> Path:
    if isinstance(entry, str):
        return resolve_path(entry)
    if isinstance(entry, dict) and "config" in entry:
        return resolve_path(entry["config"])
    raise ValueError(f"Invalid experiment entry: {entry!r}")


def add_issue(report: dict[str, Any], level: str, message: str, **context: Any) -> None:
    report[level].append({"message": message, **context})


def arg_get(args: dict[str, Any] | list[Any], key: str, default: Any = None) -> Any:
    if isinstance(args, dict):
        return args.get(key, default)
    option = f"--{key.replace('_', '-')}"
    for index, item in enumerate(args[:-1]):
        if item == option:
            return args[index + 1]
    return default


def split_csv_arg(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_requested_fold_count(value: Any, full_fold_count: int | None) -> int | None:
    if value is None or str(value).strip().lower() == "all":
        return full_fold_count
    if isinstance(value, int):
        return 1
    if isinstance(value, (list, tuple)):
        return len(value)
    parts = split_csv_arg(value)
    return len(parts) if parts else full_fold_count


def inspect_window_output(config: dict[str, Any], report: dict[str, Any], experiment: str) -> dict[str, Any]:
    windowing = config.get("windowing") or {}
    output_dir_value = windowing.get("output_dir")
    if not output_dir_value:
        add_issue(report, "errors", "Experiment has no windowing.output_dir", experiment=experiment)
        return {}

    output_dir = resolve_path(output_dir_value)
    windows_path = output_dir / "windows.npz"
    folds_path = output_dir / "loso_folds.npz"
    info: dict[str, Any] = {"data_dir": str(output_dir)}

    if not windows_path.exists():
        add_issue(report, "errors", "Missing windows.npz", experiment=experiment, path=str(windows_path))
        return info
    if not folds_path.exists():
        add_issue(report, "errors", "Missing loso_folds.npz", experiment=experiment, path=str(folds_path))
        return info

    with np.load(windows_path, allow_pickle=True) as windows:
        x_shape = tuple(int(value) for value in windows["X"].shape)
        class_names = windows["class_names"].astype(str).tolist() if "class_names" in windows.files else []
    with np.load(folds_path, allow_pickle=True) as folds:
        fold_count = int(len(folds["fold_test_subjects"]))

    info.update(
        {
            "x_shape": x_shape,
            "input_channels": int(x_shape[2]),
            "class_names": class_names,
            "num_classes": len(class_names),
            "fold_count": fold_count,
        }
    )
    return info


def infer_output_dir(config: dict[str, Any]) -> Path | None:
    training = config.get("training") or {}
    args = training.get("args") or {}
    value = arg_get(args, "output_dir")
    return resolve_path(value) if value else None


def expected_result_files(config: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    training = config.get("training") or {}
    args = training.get("args") or {}
    script = str(training.get("script", "")).lower()

    if "sleepyco" in script:
        baselines = split_csv_arg(arg_get(args, "baselines", "seq2one_gru"))
        return [
            {
                "trainer": "sleepyco",
                "variant": baseline,
                "aggregate_path": output_dir / baseline / "aggregate.json",
                "summary_path": output_dir / baseline / "summary.csv",
            }
            for baseline in baselines
        ]

    if "run_tcn_loso_npz" in script or "tcn" in output_dir.name.lower():
        return [
            {
                "trainer": "tcn",
                "variant": output_dir.name,
                "aggregate_path": output_dir / "aggregate.json",
                "summary_path": output_dir / "summary.csv",
            }
        ]

    return [
        {
            "trainer": "unknown",
            "variant": output_dir.name,
            "aggregate_path": output_dir / "aggregate.json",
            "summary_path": output_dir / "summary.csv",
        }
    ]


def count_summary_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def audit_expected_file(
    expected: dict[str, Any],
    expected_folds: int | None,
    window_info: dict[str, Any],
    report: dict[str, Any],
    experiment: str,
) -> dict[str, Any]:
    aggregate_path = expected["aggregate_path"]
    summary_path = expected["summary_path"]
    item = {
        "trainer": expected["trainer"],
        "variant": expected["variant"],
        "aggregate_path": str(aggregate_path),
        "summary_path": str(summary_path),
        "exists": aggregate_path.exists(),
    }

    if not aggregate_path.exists():
        add_issue(
            report,
            "errors",
            "Missing aggregate.json",
            experiment=experiment,
            variant=expected["variant"],
            path=str(aggregate_path),
        )
        return item

    row = collect_one(aggregate_path)
    observed_folds = row.get("fold_count")
    summary_rows = count_summary_rows(summary_path)
    item.update(
        {
            "fold_count": observed_folds,
            "summary_rows": summary_rows,
            "class_names": row.get("class_names", ""),
            "input_channels": row.get("input_channels"),
            "f1_macro_mean": row.get("f1_macro_mean"),
        }
    )

    if not summary_path.exists():
        add_issue(
            report,
            "errors",
            "Missing summary.csv",
            experiment=experiment,
            variant=expected["variant"],
            path=str(summary_path),
        )
    if expected_folds is not None and observed_folds is not None and int(observed_folds) != expected_folds:
        add_issue(
            report,
            "errors",
            "Fold count mismatch",
            experiment=experiment,
            variant=expected["variant"],
            expected=expected_folds,
            actual=observed_folds,
        )
    if expected_folds is not None and summary_rows is not None and summary_rows != expected_folds:
        add_issue(
            report,
            "errors",
            "Summary row count mismatch",
            experiment=experiment,
            variant=expected["variant"],
            expected=expected_folds,
            actual=summary_rows,
        )

    expected_class_names = window_info.get("class_names") or []
    if expected_class_names:
        observed_class_names = str(row.get("class_names", "")).split("|") if row.get("class_names") else []
        if observed_class_names != expected_class_names:
            add_issue(
                report,
                "errors",
                "Class names mismatch",
                experiment=experiment,
                variant=expected["variant"],
                expected=expected_class_names,
                actual=observed_class_names,
            )
    if window_info.get("input_channels") is not None and row.get("input_channels") is not None:
        if int(row["input_channels"]) != int(window_info["input_channels"]):
            add_issue(
                report,
                "errors",
                "Input channel count mismatch",
                experiment=experiment,
                variant=expected["variant"],
                expected=window_info["input_channels"],
                actual=row["input_channels"],
            )
    return item


def experiment_complete(exp_report: dict[str, Any]) -> bool:
    results = exp_report.get("results") or []
    if not results:
        return False
    expected_folds = exp_report.get("expected_folds")
    window = exp_report.get("window") or {}
    if "fold_count" not in window:
        return False
    expected_class_names = window.get("class_names") or []
    expected_channels = window.get("input_channels")

    for result in results:
        if not result.get("exists"):
            return False
        if expected_folds is not None:
            if result.get("fold_count") is None or int(result["fold_count"]) != int(expected_folds):
                return False
            if result.get("summary_rows") is None or int(result["summary_rows"]) != int(expected_folds):
                return False
        if expected_class_names:
            class_names = str(result.get("class_names", "")).split("|") if result.get("class_names") else []
            if class_names != expected_class_names:
                return False
        if expected_channels is not None and result.get("input_channels") is not None:
            if int(result["input_channels"]) != int(expected_channels):
                return False
    return True


def audit_suite_results(config_path: Path) -> dict[str, Any]:
    suite_path = config_path.resolve()
    suite = load_json(suite_path)
    report: dict[str, Any] = {
        "suite": suite.get("name", suite_path.stem),
        "suite_config": str(suite_path),
        "experiments": [],
        "expected_aggregates": 0,
        "found_aggregates": 0,
        "warnings": [],
        "errors": [],
    }

    for entry in suite.get("experiments") or []:
        exp_config_path = experiment_config_path(entry)
        config = load_json(exp_config_path)
        name = config.get("name", exp_config_path.stem)
        training = config.get("training") or {}
        training_args = training.get("args") or {}
        output_dir = infer_output_dir(config)
        window_info = inspect_window_output(config, report, name)
        expected_folds = parse_requested_fold_count(arg_get(training_args, "folds", "all"), window_info.get("fold_count"))
        exp_report = {
            "name": name,
            "config": str(exp_config_path),
            "output_dir": str(output_dir) if output_dir is not None else "",
            "expected_folds": expected_folds,
            "window": window_info,
            "results": [],
        }
        if output_dir is None:
            add_issue(report, "errors", "Training output_dir could not be inferred", experiment=name)
            exp_report["expected_aggregates"] = 0
            exp_report["found_aggregates"] = 0
            exp_report["ok"] = False
            report["experiments"].append(exp_report)
            continue

        expected_files = expected_result_files(config, output_dir)
        exp_report["expected_aggregates"] = len(expected_files)
        report["expected_aggregates"] += len(expected_files)
        for expected in expected_files:
            item = audit_expected_file(expected, expected_folds, window_info, report, name)
            exp_report["results"].append(item)
            report["found_aggregates"] += int(bool(item.get("exists")))
        exp_report["found_aggregates"] = int(sum(1 for item in exp_report["results"] if item.get("exists")))
        exp_report["ok"] = experiment_complete(exp_report)
        report["experiments"].append(exp_report)

    report["ok"] = not report["errors"]
    return report


def main() -> None:
    args = parse_args()
    report = audit_suite_results(args.config)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
