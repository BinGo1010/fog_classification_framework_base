#!/usr/bin/env python
"""Run the current long-short FFT MLP baseline on processed FOG datasets.

The script stays inside the existing framework:

1. dataset-specific preprocessing creates standardized sample records;
2. prepare_processed_record_windows.py creates LOSO window data;
3. export_loso_fold_npz.py creates run.py-compatible fold directories;
4. run.py trains the MLP LOSO experiment via torchrun.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fog_results_overview import enrich_overview_row, update_overview


REPO_ROOT = Path(__file__).resolve().parents[1]
TREND_FEATURES = [
    "mean",
    "std",
    "delta",
    "slope",
    "fft_energy",
    "fft_entropy",
    "fft_centroid",
    "fft_peak_freq",
]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    processed_dir: Path
    preprocess_command: list[str]


def rel(path: Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def seconds_slug(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return ("%g" % value).replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run current best long-short FFT MLP baseline on Daphnet, Multimodal, and Stanford subsets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=("daphnet", "multimodal", "stanford_imus6", "stanford_imus11"),
        default=[],
        help="Dataset key to run. Repeatable. Defaults to all four.",
    )
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--long-window-seconds", type=float, default=6.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--pre-fog-seconds", type=float, default=0.5)
    parser.add_argument(
        "--target-hz",
        type=float,
        default=60.0,
        help="Resample short windows to this Hz before training; 0 keeps native Hz.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nan-policy", choices=("error", "zero"), default="error")
    parser.add_argument("--num-folds", type=int, default=0)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--work-root", type=Path, default=Path("data/cross_dataset_sota_mlp"))
    parser.add_argument(
        "--generated-config-dir",
        type=Path,
        default=Path("outputs/generated_configs/cross_dataset_sota_mlp"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/cross_dataset_sota_mlp_summary.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/cross_dataset_sota_mlp_summary.json"),
    )
    parser.add_argument(
        "--overview-csv",
        type=Path,
        default=Path("outputs/fog_results_overview.csv"),
    )
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-windowing", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite-processed", action="store_true")
    parser.add_argument("--overwrite-windows", action="store_true")
    parser.add_argument("--overwrite-folds", action="store_true")
    parser.add_argument("--no-overview", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def dataset_specs(args: argparse.Namespace) -> dict[str, DatasetSpec]:
    root = args.data_root
    py = args.python
    daphnet_processed = root / "1.Daphnet Freezing of Gait Dataset" / "processed"
    multimodal_processed = root / "4.Multimodal Dataset" / "processed"
    stanford_processed_root = root / "5.Stanford imu-fog-detection" / "processed"
    return {
        "daphnet": DatasetSpec(
            key="daphnet",
            display_name="1.Daphnet Freezing of Gait Dataset",
            processed_dir=daphnet_processed,
            preprocess_command=[
                py,
                str(REPO_ROOT / "scripts" / "preprocess_daphnet_binary.py"),
                "--data-dir",
                str(root / "1.Daphnet Freezing of Gait Dataset" / "dataset"),
                "--output-dir",
                str(daphnet_processed),
            ],
        ),
        "multimodal": DatasetSpec(
            key="multimodal",
            display_name="4.Multimodal Dataset",
            processed_dir=multimodal_processed,
            preprocess_command=[
                py,
                str(REPO_ROOT / "scripts" / "preprocess_multimodal_binary.py"),
                "--data-dir",
                str(root / "4.Multimodal Dataset" / "Filtered Data"),
                "--output-dir",
                str(multimodal_processed),
            ],
        ),
        "stanford_imus6": DatasetSpec(
            key="stanford_imus6",
            display_name="5.Stanford imu-fog-detection / imus6_subjects7",
            processed_dir=stanford_processed_root / "imus6_subjects7",
            preprocess_command=[
                py,
                str(REPO_ROOT / "scripts" / "preprocess_stanford_binary.py"),
                "--data-root",
                str(root / "5.Stanford imu-fog-detection" / "data"),
                "--output-dir",
                str(stanford_processed_root),
                "--subsets",
                "imus6_subjects7",
            ],
        ),
        "stanford_imus11": DatasetSpec(
            key="stanford_imus11",
            display_name="5.Stanford imu-fog-detection / imus11_subjects4",
            processed_dir=stanford_processed_root / "imus11_subjects4",
            preprocess_command=[
                py,
                str(REPO_ROOT / "scripts" / "preprocess_stanford_binary.py"),
                "--data-root",
                str(root / "5.Stanford imu-fog-detection" / "data"),
                "--output-dir",
                str(stanford_processed_root),
                "--subsets",
                "imus11_subjects4",
            ],
        ),
    }


def command_text(command: list[str]) -> str:
    return shlex.join(str(part) for part in command)


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


def processed_ready(path: Path) -> bool:
    return (path / "manifest.csv").exists() and (path / "records").exists()


def windows_ready(path: Path) -> bool:
    return (path / "windows.npz").exists() and (path / "loso_folds.npz").exists() and (path / "config.json").exists()


def folds_ready(path: Path) -> bool:
    return any(path.glob("loso_subject_*/train.npz"))


def experiment_name(dataset_key: str, args: argparse.Namespace) -> str:
    return (
        f"{dataset_key}_mlp_loso_win{seconds_slug(args.window_seconds)}s"
        f"_long{seconds_slug(args.long_window_seconds)}s"
        f"_stride{seconds_slug(args.stride_seconds)}s"
        f"_prefog{seconds_slug(args.pre_fog_seconds)}"
        "_fftglobal"
    )


def dataset_work_dirs(dataset_key: str, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    name = experiment_name(dataset_key, args)
    work_root = args.work_root / name
    return work_root / "windows", work_root / "folds", Path("outputs") / name


def window_command(spec: DatasetSpec, window_dir: Path, args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        str(REPO_ROOT / "scripts" / "prepare_processed_record_windows.py"),
        "--processed-dir",
        str(spec.processed_dir),
        "--output-dir",
        str(window_dir),
        "--window-seconds",
        str(args.window_seconds),
        "--stride-seconds",
        str(args.stride_seconds),
        "--label-mode",
        "three-class",
        "--pre-fog-seconds",
        str(args.pre_fog_seconds),
        "--label-rule",
        "priority",
        "--target-hz",
        str(args.target_hz),
        "--nan-policy",
        str(args.nan_policy),
        "--multi-window-mode",
        "short_plus_long_trend",
        "--long-window-seconds",
        str(args.long_window_seconds),
        "--trend-features",
        ",".join(TREND_FEATURES),
        "--multi-window-pad",
        "edge",
        "--num-folds",
        str(args.num_folds),
        "--fold-seed",
        str(args.fold_seed),
    ]
    if args.overwrite_windows:
        command.append("--overwrite")
    return command


def export_command(window_dir: Path, fold_dir: Path, args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        str(REPO_ROOT / "scripts" / "export_loso_fold_npz.py"),
        "--data-dir",
        str(window_dir),
        "--output-dir",
        str(fold_dir),
    ]
    if args.overwrite_folds:
        command.append("--overwrite")
    return command


def load_window_metadata(window_dir: Path) -> dict[str, Any]:
    path = window_dir / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing window metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def materialize_training_config(
    spec: DatasetSpec,
    window_dir: Path,
    fold_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    meta = load_window_metadata(window_dir)
    feature_names = meta.get("output_feature_names") or meta.get("feature_names") or []
    if not feature_names:
        raise ValueError(f"No feature names found in {window_dir / 'config.json'}")
    seq_len = int(meta["target_len"])
    num_classes = 3
    name = experiment_name(spec.key, args)
    cfg = {
        "project": {
            "task_name": "fog_classification",
            "is_training": 1,
            "model_id": name,
            "name": name,
            "output_dir": rel(output_dir),
            "seed": args.seed,
            "device": "auto",
        },
        "experiment": {
            "mode": "loso",
            "loso_root": rel(fold_dir),
        },
        "data": {
            "name": "NPZTimeSeriesDataset",
            "data": "FOG",
            "root": rel(fold_dir),
            "data_path": "train.npz",
            "train_file": "train.npz",
            "val_file": "val.npz",
            "test_file": "test.npz",
            "x_key": "X",
            "y_key": "y",
            "input_format": "NTC",
            "normalize": "zscore",
            "features": "M",
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": True,
        },
        "model": {
            "name": "MLPClassifier",
            "in_channels": int(len(feature_names)),
            "dec_in": int(len(feature_names)),
            "c_out": num_classes,
            "num_classes": num_classes,
            "seq_len": seq_len,
            "pred_len": 0,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        },
        "train": {
            "epochs": args.epochs,
            "optimizer": "AdamW",
            "optimizer_foreach": False,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": "cosine",
            "class_weight": "auto",
            "loss": "weighted_ce",
            "grad_clip": 1.0,
            "amp": True,
            "distributed": "auto",
            "distributed_backend": None,
            "ddp_find_unused_parameters": "auto",
            "early_stopping_patience": 5,
            "monitor": "val_f1_macro",
            "monitor_mode": "max",
            "save_best_checkpoint": True,
            "save_last_checkpoint": False,
            "checkpoint_include_optimizer": False,
            "checkpoint_include_scheduler": False,
            "show_progress": True,
            "print_epoch_metrics": False,
        },
        "metrics": {
            "top_k": [1, 2],
            "save_predictions": False,
            "save_confusion_matrix": True,
            "save_per_class": True,
        },
        "run_metadata": {
            "dataset_key": spec.key,
            "dataset_name": spec.display_name,
            "window_dir": rel(window_dir),
            "fold_dir": rel(fold_dir),
            "window_seconds": args.window_seconds,
            "long_window_seconds": args.long_window_seconds,
            "stride_seconds": args.stride_seconds,
            "pre_fog_seconds": args.pre_fog_seconds,
            "target_hz": meta.get("target_hz"),
            "multi_window_mode": "short_plus_long_trend",
            "trend_features": TREND_FEATURES,
        },
    }
    config_dir = (REPO_ROOT / args.generated_config_dir).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{name}.yaml"
    tmp_path = config_path.with_name(f".{config_path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    tmp_path.replace(config_path)
    return config_path


def training_command(config_path: Path, args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        str(REPO_ROOT / "run.py"),
        "--config",
        str(config_path),
    ]


def metric_mean(aggregate: dict[str, Any], key: str) -> Any:
    value = aggregate.get(key)
    return value.get("mean") if isinstance(value, dict) else None


def metric_std(aggregate: dict[str, Any], key: str) -> Any:
    value = aggregate.get(key)
    return value.get("std") if isinstance(value, dict) else None


def collect_result(
    spec: DatasetSpec,
    config_path: Path,
    output_dir: Path,
    window_dir: Path,
    fold_dir: Path,
    returncode: int,
    elapsed_sec: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    meta = load_window_metadata(window_dir)
    feature_names = meta.get("output_feature_names") or meta.get("feature_names") or []
    summary_path = output_dir / "loso_summary.json"
    row: dict[str, Any] = {
        "dataset_key": spec.key,
        "dataset_name": spec.display_name,
        "feature_set": "fft_global",
        "model_name": "MLPClassifier",
        "config": rel(config_path),
        "experiment": experiment_name(spec.key, args),
        "output_dir": rel(output_dir),
        "window_dir": rel(window_dir),
        "fold_dir": rel(fold_dir),
        "window_seconds": args.window_seconds,
        "long_window_seconds": args.long_window_seconds,
        "stride_seconds": args.stride_seconds,
        "pre_fog_seconds": args.pre_fog_seconds,
        "target_hz": meta.get("target_hz"),
        "multi_window_mode": "short_plus_long_trend",
        "trend_features": ",".join(TREND_FEATURES),
        "input_channels": len(feature_names),
        "raw_in_channels": len(meta.get("feature_names") or []),
        "window_size": int(meta.get("target_len", 0)),
        "long_window_size": int(round(float(args.long_window_seconds) * float(meta.get("target_hz", 0) or 0))),
        "stride": int(round(float(args.stride_seconds) * float(meta.get("target_hz", 0) or 0))),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "returncode": returncode,
        "elapsed_sec": round(elapsed_sec, 3),
        "summary_path": rel(summary_path),
    }
    if not summary_path.exists():
        row["status"] = "missing_summary" if returncode == 0 else "failed"
        return enrich_overview_row(row)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    aggregate = summary.get("aggregate", {})
    row.update(
        {
            "status": "ok" if returncode == 0 else "failed",
            "fold_count": summary.get("num_folds"),
            "test_f1_macro_mean": metric_mean(aggregate, "test_f1_macro"),
            "test_f1_macro_std": metric_std(aggregate, "test_f1_macro"),
            "test_recall_macro_mean": metric_mean(aggregate, "test_recall_macro"),
            "test_recall_macro_std": metric_std(aggregate, "test_recall_macro"),
            "test_pr_auc_macro_mean": metric_mean(aggregate, "test_pr_auc_macro"),
            "test_pr_auc_macro_std": metric_std(aggregate, "test_pr_auc_macro"),
            "pre_fog_recall_mean": metric_mean(aggregate, "test_pre_fog_recall"),
            "pre_fog_recall_std": metric_std(aggregate, "test_pre_fog_recall"),
            "pre_fog_f1_mean": metric_mean(aggregate, "test_pre_fog_f1"),
            "pre_fog_f1_std": metric_std(aggregate, "test_pre_fog_f1"),
            "test_balanced_accuracy_mean": metric_mean(aggregate, "test_balanced_accuracy"),
            "test_balanced_accuracy_std": metric_std(aggregate, "test_balanced_accuracy"),
            "test_accuracy_mean": metric_mean(aggregate, "test_accuracy"),
            "test_accuracy_std": metric_std(aggregate, "test_accuracy"),
            "best_val_f1_macro_mean": metric_mean(aggregate, "best_val_f1_macro"),
            "best_val_f1_macro_std": metric_std(aggregate, "best_val_f1_macro"),
        }
    )
    return enrich_overview_row(row)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = REPO_ROOT / path if not path.is_absolute() else path
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
    path = REPO_ROOT / path if not path.is_absolute() else path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = [
        "dataset_key",
        "status",
        "fold_count",
        "test_f1_macro_mean",
        "test_recall_macro_mean",
        "test_pr_auc_macro_mean",
        "pre_fog_recall_mean",
        "pre_fog_f1_mean",
        "test_balanced_accuracy_mean",
        "elapsed_sec",
    ]
    widths = {
        column: max(len(column), *(len("" if row.get(column) is None else str(row.get(column))) for row in rows))
        for column in columns
    }
    print("\n===== Cross-dataset SOTA MLP summary =====")
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(("" if row.get(column) is None else str(row.get(column))).ljust(widths[column]) for column in columns))


def main() -> None:
    args = parse_args()
    specs = dataset_specs(args)
    selected = args.dataset or ["daphnet", "multimodal", "stanford_imus6", "stanford_imus11"]
    rows: list[dict[str, Any]] = []
    print(
        "[INFO] current baseline: MLP + short_plus_long_trend + fft_global "
        f"win={args.window_seconds}s long={args.long_window_seconds}s "
        f"stride={args.stride_seconds}s prefog={args.pre_fog_seconds}s",
        flush=True,
    )

    for dataset_key in selected:
        spec = specs[dataset_key]
        window_dir, fold_dir, output_dir = dataset_work_dirs(dataset_key, args)
        print(f"\n===== {dataset_key}: {spec.display_name} =====", flush=True)

        if not args.skip_preprocess:
            if processed_ready(spec.processed_dir) and not args.overwrite_processed:
                print(f"[SKIP] processed records exist: {spec.processed_dir}", flush=True)
            else:
                command = list(spec.preprocess_command)
                if args.overwrite_processed:
                    command.append("--overwrite")
                code = run_command(command, args.dry_run)
                if code != 0 and not args.continue_on_error:
                    raise SystemExit(code)

        if not args.skip_windowing:
            if windows_ready(window_dir) and not args.overwrite_windows:
                print(f"[SKIP] window data exists: {window_dir}", flush=True)
            else:
                command = window_command(spec, window_dir, args)
                code = run_command(command, args.dry_run)
                if code != 0 and not args.continue_on_error:
                    raise SystemExit(code)

        if not args.skip_export:
            if folds_ready(fold_dir) and not args.overwrite_folds:
                print(f"[SKIP] fold exports exist: {fold_dir}", flush=True)
            else:
                command = export_command(window_dir, fold_dir, args)
                code = run_command(command, args.dry_run)
                if code != 0 and not args.continue_on_error:
                    raise SystemExit(code)

        config_path = materialize_training_config(spec, window_dir, fold_dir, output_dir, args)
        summary_path = output_dir / "loso_summary.json"
        if args.skip_training:
            returncode = 0 if summary_path.exists() else 1
            elapsed = 0.0
        elif args.skip_existing and summary_path.exists():
            print(f"[SKIP] existing training summary: {summary_path}", flush=True)
            returncode = 0
            elapsed = 0.0
        else:
            command = training_command(config_path, args)
            start = time.perf_counter()
            returncode = run_command(command, args.dry_run)
            elapsed = time.perf_counter() - start

        row = collect_result(spec, config_path, output_dir, window_dir, fold_dir, returncode, elapsed, args)
        rows.append(row)
        write_csv(args.summary_csv, rows)
        write_json(args.summary_json, rows)
        if not args.no_overview and not args.dry_run:
            overview_path = update_overview(args.overview_csv, row, sweep="cross_dataset_sota_mlp")
            print(f"[OVERVIEW] updated {overview_path}", flush=True)
        if returncode != 0 and not args.continue_on_error:
            print_summary(rows)
            raise SystemExit(returncode)

    print_summary(rows)
    print(f"\n[SUMMARY] csv={rel(args.summary_csv)}")
    print(f"[SUMMARY] json={rel(args.summary_json)}")


if __name__ == "__main__":
    main()
