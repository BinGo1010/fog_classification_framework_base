#!/usr/bin/env python
"""Run FoG-STAR raw long-short DualWindowCNN configs sequentially."""

from __future__ import annotations

import argparse
import copy
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

from fog_results_overview import update_overview


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "fogstar_dualcnn_raw_loso_win1p5s_long4s_prefog0p5.yaml"

FOCUSED_COMBOS = [
    (0.5, 3.0),
    (0.5, 4.0),
    (1.0, 2.0),
    (1.0, 3.0),
    (1.0, 4.0),
    (1.0, 5.0),
    (1.5, 2.0),
    (1.5, 3.0),
    (1.5, 4.0),
    (1.5, 5.0),
    (1.5, 6.0),
    (2.0, 3.0),
    (2.0, 4.0),
    (2.0, 5.0),
]

FULL_COMBOS = FOCUSED_COMBOS + [
    (0.5, 2.0),
    (0.5, 5.0),
    (1.0, 6.0),
    (2.0, 6.0),
    (2.5, 4.0),
    (2.5, 5.0),
    (2.5, 6.0),
]

PRESETS = {
    "focused": FOCUSED_COMBOS,
    "full": FULL_COMBOS,
    "custom": [],
}

MODEL_SPECS = {
    "cnn": {
        "name": "DualWindowCNN",
        "slug": "dualcnn",
        "sweep": "dualcnn_raw_long_short",
    },
    "cnn_gru": {
        "name": "DualWindowCNNGRU",
        "slug": "dualcnn_gru",
        "sweep": "dualcnn_gru_raw_long_short",
    },
    "cnn_transformer": {
        "name": "DualWindowCNNTransformer",
        "slug": "dualcnn_transformer",
        "sweep": "dualcnn_transformer_raw_long_short",
    },
}

MODEL_SUITES = {
    "all": ["cnn", "cnn_gru", "cnn_transformer"],
    "cnn": ["cnn"],
    "cnn_gru": ["cnn_gru"],
    "cnn_transformer": ["cnn_transformer"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FoG-STAR DualWindowCNN raw long-short LOSO sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    parser.add_argument(
        "--model-suite",
        choices=sorted(MODEL_SUITES),
        default="all",
        help="Model family to run for every window combination.",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(MODEL_SPECS),
        default=[],
        help="Add one model family explicitly. Overrides --model-suite when present.",
    )
    parser.add_argument(
        "--combo",
        action="append",
        default=[],
        help="Add a short,long pair in seconds, for example --combo 1.5,4.0.",
    )
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument(
        "--generated-config-dir",
        type=Path,
        default=Path("outputs/generated_configs/fogstar_dualcnn_raw_long_short_sweep"),
    )
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra run.py override applied to every config, for example train.epochs=1.",
    )
    parser.add_argument("--no-collect", action="store_true")
    parser.add_argument(
        "--overview-csv",
        type=Path,
        default=Path("outputs/fog_results_overview.csv"),
        help="Shared CSV updated after each completed experiment.",
    )
    parser.add_argument("--no-overview", action="store_true")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/fogstar_dualwindow_raw_long_short_sweep_summary.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/fogstar_dualwindow_raw_long_short_sweep_summary.json"),
    )
    parser.add_argument("--rank-by", default="test_f1_macro_mean")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return loaded


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_combo(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Combination must be short,long seconds, got: {value}")
    short_seconds, long_seconds = float(parts[0]), float(parts[1])
    if short_seconds <= 0 or long_seconds <= 0:
        raise ValueError(f"Window seconds must be positive, got: {value}")
    if long_seconds <= short_seconds:
        raise ValueError(f"Long window must be longer than short window, got: {value}")
    return short_seconds, long_seconds


def unique_combos(combos: list[tuple[float, float]]) -> list[tuple[float, float]]:
    seen = set()
    out = []
    for short_seconds, long_seconds in combos:
        key = (float(short_seconds), float(long_seconds))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def selected_combos(args: argparse.Namespace) -> list[tuple[float, float]]:
    combos = list(PRESETS[args.preset])
    combos.extend(parse_combo(value) for value in args.combo)
    combos = unique_combos(combos)
    if not combos:
        raise ValueError("No combinations selected. Use --preset focused/full or add --combo short,long.")
    return combos


def selected_models(args: argparse.Namespace) -> list[dict[str, str]]:
    keys = args.model if args.model else MODEL_SUITES[args.model_suite]
    return [MODEL_SPECS[key] for key in keys]


def seconds_slug(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return ("%g" % float(value)).replace(".", "p")


def samples(seconds: float, sampling_rate_hz: float) -> int:
    return max(1, int(round(float(seconds) * float(sampling_rate_hz))))


def materialize_config(
    base_cfg: dict[str, Any],
    short_seconds: float,
    long_seconds: float,
    model_spec: dict[str, str],
    generated_config_dir: Path,
) -> Path:
    cfg = copy.deepcopy(base_cfg)
    wcfg = cfg.setdefault("data", {}).setdefault("windowing", {})
    multi_window = wcfg.setdefault("multi_window", {})
    model = cfg.setdefault("model", {})
    sampling_rate_hz = float(wcfg.get("sampling_rate_hz", 60))
    short_samples = samples(short_seconds, sampling_rate_hz)
    long_samples = samples(long_seconds, sampling_rate_hz)
    stride_samples = max(1, int(round(short_samples * 0.5)))
    short_slug = seconds_slug(short_seconds)
    long_slug = seconds_slug(long_seconds)
    run_name = f"fogstar_{model_spec['slug']}_raw_loso_win{short_slug}s_long{long_slug}s_prefog0p5"
    data_dir = f"data/{run_name}"

    cfg.setdefault("project", {})["model_id"] = run_name
    cfg["project"]["name"] = run_name
    cfg["project"]["output_dir"] = f"outputs/{run_name}"
    cfg.setdefault("experiment", {})["loso_root"] = data_dir
    cfg["data"]["root"] = data_dir
    wcfg["out_dir"] = data_dir
    wcfg["window_size"] = short_samples
    wcfg["stride"] = stride_samples
    multi_window["enabled"] = True
    multi_window["mode"] = "short_plus_long_raw"
    multi_window["long_window_size"] = long_samples
    multi_window.pop("trend_features", None)
    multi_window.setdefault("pad", "edge")

    raw_channels = int(model.get("raw_in_channels") or int(model.get("in_channels", 48)) // 2)
    model["name"] = model_spec["name"]
    model["raw_in_channels"] = raw_channels
    model["in_channels"] = raw_channels * 2
    model["dec_in"] = raw_channels * 2
    model["seq_len"] = long_samples
    model["short_seq_len"] = short_samples
    model["long_seq_len"] = long_samples

    generated_config_dir = resolve_repo_path(generated_config_dir)
    generated_config_dir.mkdir(parents=True, exist_ok=True)
    config_path = generated_config_dir / f"{run_name}.yaml"
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
    model = cfg.get("model", {})
    windowing = data.get("windowing", {})
    multi_window = windowing.get("multi_window") or {}
    output_dir = resolve_repo_path(project["output_dir"])
    summary_path = output_dir / "loso_summary.json"

    sampling_rate_hz = float(windowing.get("sampling_rate_hz", 1))
    window_size = windowing.get("window_size")
    stride = windowing.get("stride")
    long_window_size = multi_window.get("long_window_size")
    row: dict[str, Any] = {
        "config": str(config_path.relative_to(REPO_ROOT)),
        "experiment": project.get("name"),
        "model_name": model.get("name"),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "window_seconds": float(window_size) / sampling_rate_hz if window_size is not None else None,
        "long_window_seconds": float(long_window_size) / sampling_rate_hz if long_window_size is not None else None,
        "stride_seconds": float(stride) / sampling_rate_hz if stride is not None else None,
        "pre_fog_seconds": windowing.get("pre_fog_seconds"),
        "multi_window_mode": multi_window.get("mode"),
        "short_kernel_size": model.get("short_kernel_size"),
        "long_kernel_size": model.get("long_kernel_size"),
        "input_channels": model.get("in_channels"),
        "raw_in_channels": model.get("raw_in_channels"),
        "window_size": window_size,
        "long_window_size": long_window_size,
        "stride": stride,
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
        "model_name",
        "window_seconds",
        "long_window_seconds",
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


def print_ranked(rows: list[dict[str, Any]], rank_by: str, top_k: int) -> None:
    ranked = [
        row for row in rows
        if row.get("status") == "ok" and isinstance(row.get(rank_by), (int, float))
    ]
    if not ranked:
        return
    ranked.sort(key=lambda row: row[rank_by], reverse=True)
    ranked = ranked[:max(1, int(top_k))]
    columns = ["model_name", "window_seconds", "long_window_seconds", rank_by, "test_balanced_accuracy_mean", "test_accuracy_mean"]
    widths = {
        column: max(len(column), *(len("" if row.get(column) is None else str(row.get(column))) for row in ranked))
        for column in columns
    }
    print(f"\n===== Top {len(ranked)} by {rank_by} =====")
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in ranked:
        print("  ".join(("" if row.get(column) is None else str(row.get(column))).ljust(widths[column]) for column in columns))


def summary_path_for(config_path: Path) -> Path:
    cfg = load_yaml(config_path)
    return resolve_repo_path(cfg["project"]["output_dir"]) / "loso_summary.json"


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(resolve_repo_path(args.base_config))
    combos = selected_combos(args)
    models = selected_models(args)
    print(
        f"[INFO] preset={args.preset} combinations={len(combos)} "
        f"models={','.join(model['name'] for model in models)} total_runs={len(combos) * len(models)}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for model_spec in models:
        for short_seconds, long_seconds in combos:
            config_path = materialize_config(
                base_cfg,
                short_seconds,
                long_seconds,
                model_spec,
                args.generated_config_dir,
            )
            cfg = load_yaml(config_path)
            print(f"\n===== {cfg.get('project', {}).get('name', config_path.stem)} =====", flush=True)
            existing_summary = summary_path_for(config_path)
            if args.skip_existing and existing_summary.exists() and not args.dry_run:
                print(f"[SKIP] existing summary: {existing_summary}", flush=True)
                row = collect_result(config_path, 0, 0.0)
                rows.append(row)
                write_csv(args.summary_csv, rows)
                write_json(args.summary_json, rows)
                if not args.no_overview:
                    overview_path = update_overview(args.overview_csv, row, sweep=model_spec["sweep"])
                    print(f"[OVERVIEW] updated {overview_path}", flush=True)
                continue

            command = build_command(args, config_path)
            start = time.perf_counter()
            returncode = run_command(command, args.dry_run)
            elapsed = time.perf_counter() - start
            row = collect_result(config_path, returncode, elapsed)
            rows.append(row)
            write_csv(args.summary_csv, rows)
            write_json(args.summary_json, rows)
            if not args.dry_run and not args.no_overview:
                overview_path = update_overview(args.overview_csv, row, sweep=model_spec["sweep"])
                print(f"[OVERVIEW] updated {overview_path}", flush=True)
            if returncode != 0 and not args.continue_on_error:
                print_summary(rows)
                raise SystemExit(returncode)

    if not args.no_collect:
        print("\n===== Sweep summary =====")
        print_summary(rows)
        print_ranked(rows, args.rank_by, args.top_k)
        print(f"\n[SUMMARY] csv={resolve_repo_path(args.summary_csv)}")
        print(f"[SUMMARY] json={resolve_repo_path(args.summary_json)}")


if __name__ == "__main__":
    main()
