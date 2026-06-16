#!/usr/bin/env python
"""Run a configurable FOG windowing/validation/training pipeline.

The runner is intentionally thin. It delegates dataset windowing, validation,
and training to separate scripts so each part remains replaceable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a JSON-configured FOG experiment pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for subprocesses.",
    )
    parser.add_argument(
        "--only",
        choices=("all", "windowing", "validation", "training"),
        default="all",
        help="Run the whole pipeline or only one stage.",
    )
    parser.add_argument("--skip-windowing", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands but do not execute them.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def cli_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def append_arg(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        command.append(cli_name(key) if value else "--no-" + key.replace("_", "-"))
        return
    if isinstance(value, (list, tuple)):
        command.append(cli_name(key))
        command.extend(str(item) for item in value)
        return
    command.extend([cli_name(key), str(value)])


def format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run_command(command: list[str], dry_run: bool) -> None:
    print(f"[CMD] {format_command(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def should_run(stage: str, args: argparse.Namespace, enabled: bool) -> bool:
    if not enabled:
        return False
    if args.only != "all" and args.only != stage:
        return False
    if stage == "windowing" and args.skip_windowing:
        return False
    if stage == "validation" and args.skip_validation:
        return False
    if stage == "training" and args.skip_training:
        return False
    return True


def build_windowing_command(python_exe: str, config: dict[str, Any]) -> list[str]:
    script = resolve_path(config.get("script", "scripts/prepare_processed_record_windows.py"))
    command = [python_exe, str(script)]
    required = ("processed_dir", "output_dir", "window_seconds")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"windowing config is missing required keys: {missing}")

    path_keys = {"processed_dir", "output_dir"}
    ordered_keys = [
        "processed_dir",
        "output_dir",
        "window_seconds",
        "stride_seconds",
        "overlap",
        "label_mode",
        "pre_fog_seconds",
        "label_rule",
        "target_hz",
        "nan_policy",
        "require_success",
        "num_folds",
        "fold_seed",
        "max_records",
        "dry_run",
        "compress",
        "overwrite",
    ]
    for key in ordered_keys:
        if key not in config:
            continue
        value = resolve_path(config[key]) if key in path_keys else config[key]
        append_arg(command, key, value)
    return command


def build_validation_command(
    python_exe: str,
    validation: dict[str, Any],
    windowing: dict[str, Any] | None,
) -> list[str]:
    script = resolve_path(validation.get("script", "scripts/validate_window_dataset.py"))
    data_dir = validation.get("data_dir")
    if data_dir is None and windowing is not None:
        data_dir = windowing.get("output_dir")
    if data_dir is None:
        raise ValueError("validation needs data_dir or windowing.output_dir")

    command = [python_exe, str(script), str(resolve_path(data_dir))]
    for key in ("expected_channels", "expected_classes", "allow_empty_train"):
        if key in validation:
            append_arg(command, key, validation[key])
    return command


def build_training_command(
    python_exe: str,
    training: dict[str, Any],
    windowing: dict[str, Any] | None,
) -> list[str]:
    script = resolve_path(training.get("script", "scripts/run_sleepyco_fog_two_stage.py"))
    command = [python_exe, str(script)]

    args = training.get("args")
    if args is None:
        args = {}
    if isinstance(args, list):
        command.extend(str(resolve_path(item)) if idx and args[idx - 1] in {"--data-dir", "--output-dir"} else str(item)
                       for idx, item in enumerate(args))
    elif isinstance(args, dict):
        merged = dict(args)
        if "data_dir" not in merged and windowing is not None and "output_dir" in windowing:
            merged["data_dir"] = windowing["output_dir"]
        for key, value in merged.items():
            if key in {"data_dir", "output_dir"}:
                value = resolve_path(value)
            append_arg(command, key, value)
    else:
        raise TypeError("training.args must be a list or dict")
    return command


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    print(f"[INFO] experiment={config.get('name', config_path.stem)}", flush=True)
    print(f"[INFO] config={config_path}", flush=True)
    print(f"[INFO] repo_root={REPO_ROOT}", flush=True)

    windowing = config.get("windowing") or {}
    validation = config.get("validation") or {}
    training = config.get("training") or {}

    if should_run("windowing", args, bool(windowing.get("enabled", True))):
        run_command(build_windowing_command(args.python, windowing), args.dry_run)
    if should_run("validation", args, bool(validation.get("enabled", True))):
        run_command(build_validation_command(args.python, validation, windowing), args.dry_run)
    if should_run("training", args, bool(training.get("enabled", True))):
        run_command(build_training_command(args.python, training, windowing), args.dry_run)


if __name__ == "__main__":
    main()
