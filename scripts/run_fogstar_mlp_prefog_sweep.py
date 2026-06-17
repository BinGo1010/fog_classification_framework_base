#!/usr/bin/env python
"""Run the FoG-STAR MLP Pre-FOG duration sweep sequentially."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    REPO_ROOT / "configs" / "fogstar_mlp_loso_win2s_prefog0p5.yaml",
    REPO_ROOT / "configs" / "fogstar_mlp_loso_win2s_prefog1.yaml",
    REPO_ROOT / "configs" / "fogstar_mlp_loso_win2s_prefog1p5.yaml",
    REPO_ROOT / "configs" / "fogstar_mlp_loso_win2s_prefog2.yaml",
    REPO_ROOT / "configs" / "fogstar_mlp_loso_win2s_prefog3.yaml",
    REPO_ROOT / "configs" / "fogstar_mlp_loso_win2s_prefog4.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run six FoG-STAR MLP LOSO configs: Pre-FOG 0.5, 1, 1.5, 2, 3, 4 seconds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra run.py override applied to every config, for example train.epochs=1.",
    )
    parser.add_argument("--no-collect", action="store_true")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/fogstar_mlp_prefog_sweep_summary.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/fogstar_mlp_prefog_sweep_summary.json"),
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return loaded


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def build_command(args: argparse.Namespace, config_path: Path) -> list[str]:
    command = [
        args.python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        str(REPO_ROOT / "run.py"),
        "--config",
        str(config_path),
    ]
    for override in args.override:
        command.extend(["--override", override])
    return command


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def run_command(command: list[str], dry_run: bool) -> int:
    print(f"[CMD] {command_text(command)}", flush=True)
    if dry_run:
        return 0
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
    return int(completed.returncode)


def metric_mean(aggregate: dict[str, Any], key: str) -> Any:
    value = aggregate.get(key)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def collect_result(config_path: Path, returncode: int, elapsed_sec: float) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    project = cfg.get("project", {})
    data = cfg.get("data", {})
    windowing = data.get("windowing", {})
    output_dir = resolve_repo_path(project["output_dir"])
    summary_path = output_dir / "loso_summary.json"

    row: dict[str, Any] = {
        "config": str(config_path.relative_to(REPO_ROOT)),
        "experiment": project.get("name"),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "pre_fog_seconds": windowing.get("pre_fog_seconds"),
        "window_size": windowing.get("window_size"),
        "stride": windowing.get("stride"),
        "epochs": (cfg.get("train") or {}).get("epochs"),
        "batch_size": data.get("batch_size"),
        "returncode": returncode,
        "elapsed_sec": round(elapsed_sec, 3),
        "summary_path": str(summary_path.relative_to(REPO_ROOT)),
    }
    if not summary_path.exists():
        row["status"] = "missing_summary"
        return row

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    aggregate = summary.get("aggregate", {})
    row.update(
        {
            "status": "ok" if returncode == 0 else "failed",
            "fold_count": summary.get("num_folds"),
            "test_f1_macro_mean": metric_mean(aggregate, "test_f1_macro"),
            "test_balanced_accuracy_mean": metric_mean(aggregate, "test_balanced_accuracy"),
            "test_accuracy_mean": metric_mean(aggregate, "test_accuracy"),
            "best_val_f1_macro_mean": metric_mean(aggregate, "best_val_f1_macro"),
        }
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = resolve_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path = resolve_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(rows: list[dict[str, Any]]) -> None:
    columns = [
        "pre_fog_seconds",
        "status",
        "fold_count",
        "test_f1_macro_mean",
        "test_balanced_accuracy_mean",
        "test_accuracy_mean",
        "best_val_f1_macro_mean",
        "elapsed_sec",
    ]
    widths = {
        column: max(len(column), *(len("" if row.get(column) is None else str(row.get(column))) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(("" if row.get(column) is None else str(row.get(column))).ljust(widths[column]) for column in columns))


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for config_path in CONFIGS:
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        cfg = load_yaml(config_path)
        print(f"\n===== {cfg.get('project', {}).get('name', config_path.stem)} =====", flush=True)
        command = build_command(args, config_path)
        start = time.perf_counter()
        returncode = run_command(command, args.dry_run)
        elapsed = time.perf_counter() - start
        row = collect_result(config_path, returncode, elapsed)
        rows.append(row)
        write_csv(args.summary_csv, rows)
        write_json(args.summary_json, rows)
        if returncode != 0 and not args.continue_on_error:
            print_summary(rows)
            raise SystemExit(returncode)

    if not args.no_collect:
        print("\n===== Sweep summary =====")
        print_summary(rows)
        print(f"\n[SUMMARY] csv={resolve_repo_path(args.summary_csv)}")
        print(f"[SUMMARY] json={resolve_repo_path(args.summary_json)}")


if __name__ == "__main__":
    main()

