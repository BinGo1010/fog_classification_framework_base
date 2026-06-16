#!/usr/bin/env python
"""Preflight checks for FOG experiment suite configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT_OVERRIDE: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a FOG suite config before running local/server experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Override relative dataset/... paths. Useful for synthetic preflight datasets.",
    )
    parser.add_argument(
        "--require-windows",
        action="store_true",
        help="Require every unique window output directory to already contain valid windows.",
    )
    parser.add_argument(
        "--allow-missing-processed",
        action="store_true",
        help="Warn instead of failing when processed_dir is absent. Useful before preprocessing.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the preflight report.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if DATASET_ROOT_OVERRIDE is not None and path.parts and path.parts[0] == "dataset":
        return (DATASET_ROOT_OVERRIDE / Path(*path.parts[1:])).resolve()
    return (REPO_ROOT / path).resolve()


def normalized_json(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): normalize(item[key]) for key in sorted(item)}
        if isinstance(item, list):
            return [normalize(part) for part in item]
        return item

    return json.dumps(normalize(value), sort_keys=True, ensure_ascii=False)


def experiment_config_path(entry: str | dict[str, Any]) -> Path:
    if isinstance(entry, str):
        return resolve_path(entry)
    if isinstance(entry, dict) and "config" in entry:
        return resolve_path(entry["config"])
    raise ValueError(f"Invalid experiment entry: {entry!r}")


def add_issue(report: dict[str, Any], level: str, message: str, **context: Any) -> None:
    report[level].append({"message": message, **context})


def required_keys(config: dict[str, Any], keys: list[str], label: str, report: dict[str, Any]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        add_issue(report, "errors", f"{label} is missing required keys", missing=missing)


def critical_window_config(windowing: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "processed_dir",
        "output_dir",
        "window_seconds",
        "stride_seconds",
        "overlap",
        "label_mode",
        "pre_fog_seconds",
        "label_rule",
        "target_hz",
        "require_success",
        "num_folds",
        "fold_seed",
        "max_records",
    ]
    out = {key: windowing.get(key) for key in keys if key in windowing}
    if "processed_dir" in out:
        out["processed_dir"] = str(resolve_path(out["processed_dir"]))
    if "output_dir" in out:
        out["output_dir"] = str(resolve_path(out["output_dir"]))
    return out


def window_key(windowing: dict[str, Any]) -> tuple[str, str]:
    output_dir = windowing.get("output_dir")
    if not output_dir:
        raise ValueError("windowing config is missing output_dir")
    critical = critical_window_config(windowing)
    return str(resolve_path(output_dir)), normalized_json(critical)


def compare_existing_window_config(
    output_dir: Path,
    requested: dict[str, Any],
    report: dict[str, Any],
) -> None:
    if not output_dir.exists():
        return
    config_path = output_dir / "config.json"
    if not config_path.exists():
        add_issue(report, "warnings", "Window output exists but has no config.json", output_dir=str(output_dir))
        return
    existing = load_json(config_path)
    compare_keys = [
        "window_seconds",
        "stride_seconds",
        "overlap",
        "label_mode",
        "pre_fog_seconds",
        "label_rule",
        "target_hz",
        "require_success",
    ]
    mismatches = {}
    for key in compare_keys:
        requested_value = requested.get(key)
        existing_value = existing.get(key)
        if requested_value is not None and existing_value != requested_value:
            mismatches[key] = {"requested": requested_value, "existing": existing_value}
    if mismatches:
        add_issue(
            report,
            "errors",
            "Existing window config does not match requested config",
            output_dir=str(output_dir),
            mismatches=mismatches,
        )


def inspect_window_output(
    output_dir: Path,
    expected_classes: int | None,
    expected_channels: int | None,
    report: dict[str, Any],
) -> dict[str, Any] | None:
    if not output_dir.exists():
        return None
    windows_path = output_dir / "windows.npz"
    folds_path = output_dir / "loso_folds.npz"
    if not windows_path.exists() or not folds_path.exists():
        add_issue(
            report,
            "warnings",
            "Window output is incomplete",
            output_dir=str(output_dir),
            missing=[
                name
                for name, path in (("windows.npz", windows_path), ("loso_folds.npz", folds_path))
                if not path.exists()
            ],
        )
        return None

    with np.load(windows_path, allow_pickle=True) as data:
        x_shape = tuple(int(value) for value in data["X"].shape)
        class_names = data["class_names"].astype(str).tolist() if "class_names" in data.files else []
    with np.load(folds_path, allow_pickle=True) as folds:
        fold_count = int(len(folds["fold_test_subjects"]))
    if expected_channels is not None and x_shape[2] != expected_channels:
        add_issue(
            report,
            "errors",
            "Window channel count mismatch",
            output_dir=str(output_dir),
            expected=expected_channels,
            actual=x_shape[2],
        )
    if expected_classes is not None and len(class_names) != expected_classes:
        add_issue(
            report,
            "errors",
            "Window class count mismatch",
            output_dir=str(output_dir),
            expected=expected_classes,
            actual=len(class_names),
            class_names=class_names,
        )
    return {
        "output_dir": str(output_dir),
        "x_shape": x_shape,
        "class_names": class_names,
        "fold_count": fold_count,
    }


def check_processed_success_marker(processed_dir: Path, experiment: str, report: dict[str, Any]) -> None:
    success_path = processed_dir / "_SUCCESS.json"
    if not success_path.exists():
        add_issue(
            report,
            "errors",
            "Processed directory is missing required success marker",
            experiment=experiment,
            path=str(success_path),
        )
        return
    try:
        marker = load_json(success_path)
    except Exception as exc:
        add_issue(
            report,
            "errors",
            f"Cannot read processed success marker: {exc}",
            experiment=experiment,
            path=str(success_path),
        )
        return
    if marker.get("status") != "complete":
        add_issue(
            report,
            "errors",
            "Processed success marker is not complete",
            experiment=experiment,
            path=str(success_path),
            status=marker.get("status"),
        )


def infer_training_output_dir(config: dict[str, Any]) -> Path | None:
    training = config.get("training") or {}
    args = training.get("args") or {}
    if isinstance(args, dict) and args.get("output_dir"):
        return resolve_path(args["output_dir"])
    if isinstance(args, list):
        for idx, item in enumerate(args[:-1]):
            if item == "--output-dir":
                return resolve_path(args[idx + 1])
    return None


def check_suite(args: argparse.Namespace) -> dict[str, Any]:
    suite_path = args.config.resolve()
    suite = load_json(suite_path)
    experiments = suite.get("experiments") or []
    report: dict[str, Any] = {
        "suite": suite.get("name", suite_path.stem),
        "suite_config": str(suite_path),
        "experiments": [],
        "unique_windows": [],
        "training_outputs": [],
        "collection": {},
        "warnings": [],
        "errors": [],
    }
    if not experiments:
        add_issue(report, "errors", "Suite has no experiments", suite_config=str(suite_path))
        return report

    window_seen: dict[str, tuple[str, Path]] = {}
    unique_window_entries: dict[str, dict[str, Any]] = {}
    for entry in experiments:
        try:
            config_path = experiment_config_path(entry)
            config = load_json(config_path)
        except Exception as exc:
            add_issue(report, "errors", f"Cannot read experiment config: {exc}", entry=entry)
            continue

        name = config.get("name", config_path.stem)
        windowing = config.get("windowing") or {}
        validation = config.get("validation") or {}
        training = config.get("training") or {}
        training_args = training.get("args") or {}
        report["experiments"].append({"name": name, "config": str(config_path)})

        if windowing.get("enabled", True):
            required_keys(
                windowing,
                ["processed_dir", "output_dir", "window_seconds"],
                f"{name}.windowing",
                report,
            )
            processed_dir = resolve_path(windowing.get("processed_dir", ""))
            allow_missing_processed = bool(getattr(args, "allow_missing_processed", False))
            if not processed_dir.exists():
                issue_level = "warnings" if allow_missing_processed else "errors"
                add_issue(
                    report,
                    issue_level,
                    "Processed directory does not exist",
                    experiment=name,
                    path=str(processed_dir),
                )
            else:
                for required in ("records", "manifest.csv"):
                    if not (processed_dir / required).exists():
                        add_issue(
                            report,
                            "errors",
                            "Processed directory is missing required artifact",
                            experiment=name,
                            path=str(processed_dir / required),
                        )
                if not (processed_dir / "config.json").exists() and not (processed_dir / "schema.json").exists():
                    add_issue(
                        report,
                        "errors",
                        "Processed directory is missing required artifact",
                        experiment=name,
                        path=f"{processed_dir / 'config.json'} or {processed_dir / 'schema.json'}",
                    )
                if bool(windowing.get("require_success", False)):
                    check_processed_success_marker(processed_dir, name, report)
            try:
                key, fingerprint = window_key(windowing)
                if key in window_seen and window_seen[key][0] != fingerprint:
                    add_issue(
                        report,
                        "errors",
                        "Conflicting windowing configs share an output directory",
                        output_dir=key,
                        first_config=str(window_seen[key][1]),
                        second_config=str(config_path),
                    )
                else:
                    window_seen.setdefault(key, (fingerprint, config_path))
                    unique_window_entries.setdefault(
                        key,
                        {
                            "config_path": config_path,
                            "windowing": windowing,
                            "validation": validation,
                        },
                    )
            except Exception as exc:
                add_issue(report, "errors", f"Invalid windowing config: {exc}", experiment=name)

        if training.get("enabled", True):
            script = resolve_path(training.get("script", "scripts/run_sleepyco_fog_two_stage.py"))
            if not script.exists():
                add_issue(report, "errors", "Training script does not exist", experiment=name, script=str(script))
            if isinstance(training_args, dict) and "data_dir" not in training_args and "output_dir" not in windowing:
                add_issue(report, "errors", "Training has no data_dir and no windowing.output_dir", experiment=name)
            training_output = infer_training_output_dir(config)
            if training_output is None:
                add_issue(report, "warnings", "Training output_dir could not be inferred", experiment=name)
            else:
                report["training_outputs"].append({"experiment": name, "output_dir": str(training_output)})

    for output_dir_text, entry in sorted(unique_window_entries.items()):
        output_dir = resolve_path(output_dir_text)
        validation = entry["validation"]
        expected_classes = validation.get("expected_classes")
        expected_channels = validation.get("expected_channels")
        compare_existing_window_config(output_dir, entry["windowing"], report)
        info = inspect_window_output(output_dir, expected_classes, expected_channels, report)
        if info is None and args.require_windows:
            add_issue(report, "errors", "Required window output is missing or incomplete", output_dir=str(output_dir))
            info = {"output_dir": str(output_dir), "exists": False}
        elif info is None:
            info = {"output_dir": str(output_dir), "exists": False}
        else:
            info["exists"] = True
        info["source_config"] = str(entry["config_path"])
        report["unique_windows"].append(info)

    collection = suite.get("collection") or {}
    if collection.get("enabled", True):
        report["collection"] = {
            "output_csv": str(resolve_path(collection.get("output_csv", "outputs/fog_suite_summary.csv"))),
            "output_json": str(resolve_path(collection.get("output_json", "outputs/fog_suite_summary.json"))),
            "recursive": bool(collection.get("recursive", True)),
        }
    return report


def main() -> None:
    global DATASET_ROOT_OVERRIDE
    args = parse_args()
    DATASET_ROOT_OVERRIDE = args.dataset_root.resolve() if args.dataset_root else None
    report = check_suite(args)
    ok = not report["errors"]
    report["ok"] = ok
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
