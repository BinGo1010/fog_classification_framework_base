#!/usr/bin/env python
"""Run FoG-STAR single-window MultiKernelCNN kernel-size sweep."""

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
BASE_CONFIG = REPO_ROOT / "configs" / "fogstar_multikernel_cnn_loso_win1s_prefog0p5.yaml"

FOCUSED_SMALL_KERNELS = [3, 5, 7]
FOCUSED_LARGE_KERNELS = [15, 31, 51]
FULL_SMALL_KERNELS = [3, 5, 7]
FULL_LARGE_KERNELS = [15, 21, 31, 41, 51]

PRESETS = {
    "focused": (FOCUSED_SMALL_KERNELS, FOCUSED_LARGE_KERNELS),
    "full": (FULL_SMALL_KERNELS, FULL_LARGE_KERNELS),
    "custom": ([], []),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FoG-STAR MultiKernelCNN LOSO sweep over small/large kernel pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    parser.add_argument(
        "--small-kernel",
        action="append",
        type=int,
        default=[],
        help="Small kernel size. Repeat to form a Cartesian product with --large-kernel.",
    )
    parser.add_argument(
        "--large-kernel",
        action="append",
        type=int,
        default=[],
        help="Large kernel size. Repeat to form a Cartesian product with --small-kernel.",
    )
    parser.add_argument(
        "--kernel-pair",
        action="append",
        default=[],
        help="Add one explicit small,large pair, for example --kernel-pair 5,31.",
    )
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument(
        "--generated-config-dir",
        type=Path,
        default=Path("outputs/generated_configs/fogstar_multikernel_cnn_win1s_sweep"),
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
        default=Path("outputs/fogstar_multikernel_cnn_win1s_sweep_summary.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/fogstar_multikernel_cnn_win1s_sweep_summary.json"),
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


def parse_kernel_pair(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Kernel pair must be small,large, got: {value}")
    return validate_pair(int(parts[0]), int(parts[1]))


def validate_kernel(value: int, name: str) -> int:
    value = int(value)
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} kernel must be a positive odd integer, got {value}.")
    return value


def validate_pair(small_kernel: int, large_kernel: int) -> tuple[int, int]:
    small_kernel = validate_kernel(small_kernel, "Small")
    large_kernel = validate_kernel(large_kernel, "Large")
    if large_kernel <= small_kernel:
        raise ValueError(f"Large kernel must be greater than small kernel, got {small_kernel},{large_kernel}.")
    return small_kernel, large_kernel


def unique_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen = set()
    out = []
    for small_kernel, large_kernel in pairs:
        key = (int(small_kernel), int(large_kernel))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def selected_pairs(args: argparse.Namespace) -> list[tuple[int, int]]:
    preset_small, preset_large = PRESETS[args.preset]
    small_kernels = args.small_kernel or preset_small
    large_kernels = args.large_kernel or preset_large
    pairs = [validate_pair(small, large) for small in small_kernels for large in large_kernels]
    pairs.extend(parse_kernel_pair(value) for value in args.kernel_pair)
    pairs = unique_pairs(pairs)
    if not pairs:
        raise ValueError(
            "No kernel pairs selected. Use --preset focused/full, "
            "--small-kernel with --large-kernel, or --kernel-pair small,large."
        )
    return pairs


def samples(seconds: float, sampling_rate_hz: float) -> int:
    return max(1, int(round(float(seconds) * float(sampling_rate_hz))))


def seconds_slug(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return ("%g" % float(value)).replace(".", "p")


def materialize_config(
    base_cfg: dict[str, Any],
    small_kernel: int,
    large_kernel: int,
    generated_config_dir: Path,
) -> Path:
    cfg = copy.deepcopy(base_cfg)
    project = cfg.setdefault("project", {})
    data = cfg.setdefault("data", {})
    wcfg = data.setdefault("windowing", {})
    model = cfg.setdefault("model", {})
    sampling_rate_hz = float(wcfg.get("sampling_rate_hz", 60))
    window_size = int(wcfg.get("window_size") or samples(1.0, sampling_rate_hz))
    stride = int(wcfg.get("stride") or max(1, window_size // 2))
    if large_kernel > window_size:
        raise ValueError(
            f"large_kernel_size={large_kernel} is longer than the window_size={window_size}. "
            "Use a smaller large kernel or a longer base window."
        )
    window_seconds = window_size / sampling_rate_hz
    prefog = float(wcfg.get("pre_fog_seconds", 0.5))
    win_slug = seconds_slug(window_seconds)
    prefog_slug = seconds_slug(prefog)
    run_name = (
        "fogstar_multikernel_cnn_loso_"
        f"win{win_slug}s_k{small_kernel}_{large_kernel}_prefog{prefog_slug}"
    )
    data_dir = f"data/fogstar_multikernel_cnn_loso_win{win_slug}s_prefog{prefog_slug}"

    project["model_id"] = run_name
    project["name"] = run_name
    project["output_dir"] = f"outputs/{run_name}"
    cfg.setdefault("experiment", {})["loso_root"] = data_dir
    data["root"] = data_dir
    wcfg["out_dir"] = data_dir
    wcfg["window_size"] = window_size
    wcfg["stride"] = stride
    wcfg.pop("multi_window", None)

    model["name"] = "MultiKernelCNN"
    model["in_channels"] = int(model.get("in_channels", 24))
    model["dec_in"] = int(model.get("dec_in", model["in_channels"]))
    model["seq_len"] = window_size
    model["small_kernel_size"] = int(small_kernel)
    model["large_kernel_size"] = int(large_kernel)
    model.pop("raw_in_channels", None)
    model.pop("short_seq_len", None)
    model.pop("long_seq_len", None)
    model.pop("short_kernel_size", None)
    model.pop("long_kernel_size", None)

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
    output_dir = resolve_repo_path(project["output_dir"])
    summary_path = output_dir / "loso_summary.json"

    sampling_rate_hz = float(windowing.get("sampling_rate_hz", 1))
    window_size = windowing.get("window_size")
    stride = windowing.get("stride")
    row: dict[str, Any] = {
        "config": str(config_path.relative_to(REPO_ROOT)),
        "experiment": project.get("name"),
        "model_name": model.get("name"),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "window_seconds": float(window_size) / sampling_rate_hz if window_size is not None else None,
        "long_window_seconds": None,
        "stride_seconds": float(stride) / sampling_rate_hz if stride is not None else None,
        "pre_fog_seconds": windowing.get("pre_fog_seconds"),
        "multi_window_mode": "",
        "small_kernel_size": model.get("small_kernel_size"),
        "large_kernel_size": model.get("large_kernel_size"),
        "input_channels": model.get("in_channels"),
        "raw_in_channels": "",
        "window_size": window_size,
        "long_window_size": "",
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
            "test_recall_macro_mean": metric_mean(aggregate, "test_recall_macro"),
            "test_pr_auc_macro_mean": metric_mean(aggregate, "test_pr_auc_macro"),
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
        "small_kernel_size",
        "large_kernel_size",
        "status",
        "fold_count",
        "test_f1_macro_mean",
        "test_recall_macro_mean",
        "test_pr_auc_macro_mean",
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
    columns = ["small_kernel_size", "large_kernel_size", rank_by, "test_recall_macro_mean", "test_pr_auc_macro_mean", "test_balanced_accuracy_mean", "test_accuracy_mean"]
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
    pairs = selected_pairs(args)
    print(
        f"[INFO] preset={args.preset} kernel_pairs={len(pairs)} "
        f"total_runs={len(pairs)}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for small_kernel, large_kernel in pairs:
        config_path = materialize_config(
            base_cfg,
            small_kernel,
            large_kernel,
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
                overview_path = update_overview(args.overview_csv, row, sweep="multikernel_cnn")
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
            overview_path = update_overview(args.overview_csv, row, sweep="multikernel_cnn")
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
