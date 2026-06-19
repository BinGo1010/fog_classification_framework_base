#!/usr/bin/env python
"""Run one FoG-STAR raw long-short MLP experiment.

The input uses short_plus_long_raw, so MLP sees raw short/long windows instead
of handcrafted trend or frequency features.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from fog_results_overview import enrich_overview_row, update_overview
from run_fogstar_mlp_long_short_sweep import (
    REPO_ROOT,
    collect_result,
    seconds_slug,
    samples,
    write_csv,
    write_json,
)


BASE_CONFIG = REPO_ROOT / "configs" / "fogstar_mlp_raw_loso_win1s_long6s_stride0p1s_prefog0p5.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FoG-STAR MLP with raw short/long window input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--short-seconds", type=float, default=1.0)
    parser.add_argument("--long-seconds", type=float, default=6.0)
    parser.add_argument("--stride-seconds", type=float, default=0.1)
    parser.add_argument("--pre-fog-seconds", type=float, default=0.5)
    parser.add_argument(
        "--generated-config-dir",
        type=Path,
        default=Path("outputs/generated_configs/fogstar_mlp_raw_long_short"),
    )
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra run.py override, for example train.epochs=1.",
    )
    parser.add_argument(
        "--overview-csv",
        type=Path,
        default=Path("outputs/fog_results_overview.csv"),
    )
    parser.add_argument("--no-overview", action="store_true")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/fogstar_mlp_raw_long_short_summary.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/fogstar_mlp_raw_long_short_summary.json"),
    )
    return parser.parse_args()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return loaded


def validate_seconds(short_seconds: float, long_seconds: float, stride_seconds: float) -> None:
    if short_seconds <= 0 or long_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("short-seconds, long-seconds, and stride-seconds must be positive.")
    if long_seconds < short_seconds:
        raise ValueError("long-seconds must be >= short-seconds.")


def materialize_config(args: argparse.Namespace) -> Path:
    validate_seconds(args.short_seconds, args.long_seconds, args.stride_seconds)
    cfg = copy.deepcopy(load_yaml(resolve_repo_path(args.base_config)))
    wcfg = cfg.setdefault("data", {}).setdefault("windowing", {})
    multi_window = wcfg.setdefault("multi_window", {})
    model = cfg.setdefault("model", {})
    sampling_rate_hz = float(wcfg.get("sampling_rate_hz", 60))
    short_samples = samples(args.short_seconds, sampling_rate_hz)
    long_samples = samples(args.long_seconds, sampling_rate_hz)
    stride_samples = samples(args.stride_seconds, sampling_rate_hz)

    short_slug = seconds_slug(args.short_seconds)
    long_slug = seconds_slug(args.long_seconds)
    stride_slug = seconds_slug(args.stride_seconds)
    prefog_slug = seconds_slug(args.pre_fog_seconds)
    run_name = (
        f"fogstar_mlp_raw_loso_win{short_slug}s_long{long_slug}s_"
        f"stride{stride_slug}s_prefog{prefog_slug}"
    )
    data_dir = f"data/{run_name}"

    cfg.setdefault("project", {})["model_id"] = run_name
    cfg["project"]["name"] = run_name
    cfg["project"]["output_dir"] = f"outputs/{run_name}"
    cfg.setdefault("experiment", {})["loso_root"] = data_dir
    cfg["data"]["root"] = data_dir
    wcfg["out_dir"] = data_dir
    wcfg["window_size"] = short_samples
    wcfg["stride"] = stride_samples
    wcfg["pre_fog_seconds"] = float(args.pre_fog_seconds)
    multi_window["enabled"] = True
    multi_window["mode"] = "short_plus_long_raw"
    multi_window["long_window_size"] = long_samples
    multi_window.pop("trend_features", None)
    multi_window.setdefault("pad", "edge")

    raw_channels = int(model.get("raw_in_channels") or 24)
    model["name"] = "MLPClassifier"
    model["raw_in_channels"] = raw_channels
    model["in_channels"] = raw_channels * 2
    model["dec_in"] = raw_channels * 2
    model["seq_len"] = long_samples
    model["short_seq_len"] = short_samples
    model["long_seq_len"] = long_samples

    output_dir = resolve_repo_path(args.generated_config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"{run_name}.yaml"
    tmp_path = config_path.with_name(f".{config_path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    tmp_path.replace(config_path)
    return config_path


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


def run_command(command: list[str], dry_run: bool) -> int:
    print(f"[CMD] {shlex.join(command)}", flush=True)
    if dry_run:
        return 0
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
    return int(completed.returncode)


def summary_path_for(config_path: Path) -> Path:
    cfg = load_yaml(config_path)
    return resolve_repo_path(cfg["project"]["output_dir"]) / "loso_summary.json"


def main() -> None:
    args = parse_args()
    config_path = materialize_config(args)
    print(f"[CONFIG] {config_path}", flush=True)
    existing_summary = summary_path_for(config_path)
    if args.skip_existing and existing_summary.exists() and not args.dry_run:
        print(f"[SKIP] existing summary: {existing_summary}", flush=True)
        row = collect_result(config_path, 0, 0.0)
    else:
        command = build_command(args, config_path)
        start = time.perf_counter()
        returncode = run_command(command, args.dry_run)
        elapsed = time.perf_counter() - start
        row = collect_result(config_path, returncode, elapsed)

    row["feature_set"] = "raw_short_long"
    row = enrich_overview_row(row)
    write_csv(args.summary_csv, [row])
    write_json(args.summary_json, [row])
    if not args.dry_run and not args.no_overview:
        overview_path = update_overview(args.overview_csv, row, sweep="mlp_raw_long_short")
        print(f"[OVERVIEW] updated {overview_path}", flush=True)
    print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
