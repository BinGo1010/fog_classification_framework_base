from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_daphnet_full_subject_nbm_residual_binary as exp
import run_daphnet_full_subject_nbm_residual_inceptiontime as full
from cnbr_fog.data import DaphnetDataset


EXPERIMENT = "daphnet_full_subject_raw_inceptiontime_k359_v1"
METHOD = "B0"
METHOD_NAME = "Raw-InceptionTime-K3-5-9"
METHOD_DIR = "B0_raw_inceptiontime_k359"
KERNEL_SIZES = (3, 5, 9)
TOTAL_RUNS = 30 * len(exp.SEEDS)


class SmallKernelInceptionTimeClassifier(nn.Module):
    """Six-module Raw InceptionTime using temporal kernels 3/5/9."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        modules: list[nn.Module] = []
        channels = in_channels
        for _ in range(full.MODULE_COUNT):
            module = full.InceptionModule(channels, kernel_sizes=KERNEL_SIZES)
            modules.append(module)
            channels = module.out_channels
        self.modules_inception = nn.ModuleList(modules)
        self.residual_1 = full.ResidualProjection(in_channels, channels)
        self.residual_2 = full.ResidualProjection(channels, channels)
        self.residual_activation = nn.ReLU()
        self.classifier = nn.Linear(channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        block_input = inputs
        features = inputs
        for module in self.modules_inception[:3]:
            features = module(features)
        features = self.residual_activation(features + self.residual_1(block_input))
        block_input = features
        for module in self.modules_inception[3:]:
            features = module(features)
        features = self.residual_activation(features + self.residual_2(block_input))
        return self.classifier(features.mean(dim=-1)).squeeze(1)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": "inceptiontime_6module_k359",
            "in_channels": self.in_channels,
            "module_count": full.MODULE_COUNT,
            "residual_after_modules": [3, 6],
            "bottleneck_channels": full.BOTTLENECK_CHANNELS,
            "branch_channels": full.BRANCH_CHANNELS,
            "kernel_sizes": list(KERNEL_SIZES),
            "pool_branch": "MaxPool1d(3, stride=1)+Conv1d(1)",
            "normalization": "BatchNorm1d",
            "activation": "ReLU",
            "pooling": "global_average",
            "output": "one logit; sigmoid applied for probability",
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }


def configure_experiment(resume: bool = True) -> None:
    """Patch the audited base protocol to train Raw + small-kernel InceptionTime only."""
    full.RESUME_ALLOWED = bool(resume)
    full.InceptionTimeClassifier = SmallKernelInceptionTimeClassifier
    exp.EXPERIMENT = EXPERIMENT
    exp.METHODS = (METHOD,)
    exp.METHOD_NAMES = {METHOD: METHOD_NAME}
    exp.METHOD_CHANNELS = {METHOD: 9}
    exp.METHOD_DIRS = {METHOD: METHOD_DIR}
    exp.train_classifier = full.train_classifier


def write_protocol(root: Path, source_root: Path, config: Path) -> None:
    exp.write_json(
        root / "splits" / "frozen_protocol_raw_inceptiontime_k359.json",
        {
            "experiment": EXPERIMENT,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": str(config.resolve()),
            "source_representation_root": str(source_root),
            "representation_reused_without_retraining": True,
            "subjects": list(exp.SUBJECTS),
            "methods": [METHOD],
            "outer": "leave-one-complete-valid-record-out",
            "inner": "3-fold record-first purged OOF",
            "classifier": "Raw-InceptionTime-6module-K3-5-9",
            "kernel_sizes": list(KERNEL_SIZES),
            "classifier_seeds": list(exp.SEEDS),
            "test_used_for_selection": False,
            "resume_granularity": "classifier epoch and completed method-seed run",
            "invalid_record_exclusion": {"S03_seg003": "valid_fraction=0"},
        },
    )


def create_balanced_fold_plan(root: Path, devices: list[str]) -> Path:
    summary_path = root / "splits" / "outer_folds" / "outer_fold_summary.csv"
    rows = exp.read_csv(summary_path)
    if len(rows) != 30:
        raise ValueError(f"expected 30 outer folds in {summary_path}, found {len(rows)}")
    workers = [
        {"device": device, "folds": [], "estimated_train_windows": 0}
        for device in devices
    ]
    for row in sorted(rows, key=lambda item: int(item["train_windows"]), reverse=True):
        worker = min(
            workers,
            key=lambda item: (item["estimated_train_windows"], len(item["folds"])),
        )
        worker["folds"].append(f"{row['subject_id']}/{row['fold_id']}")
        worker["estimated_train_windows"] += int(row["train_windows"])
    path = root / "splits" / "raw_inceptiontime_k359_parallel_plan.json"
    exp.write_json(
        path,
        {
            "strategy": "greedy_lpt_by_outer_train_windows",
            "worker_count": len(devices),
            "workers": {str(index): worker for index, worker in enumerate(workers)},
        },
    )
    return path


def launch_parallel_workers(args: argparse.Namespace, root: Path, source_root: Path) -> None:
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one CUDA device")
    fold_plan = create_balanced_fold_plan(root, devices)
    log_dir = root / "logs" / "parallel_workers"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = set(range(len(devices)))
    attempts = {index: 0 for index in pending}
    statuses: dict[int, dict[str, Any]] = {}
    running: dict[int, tuple[subprocess.Popen[Any], Any, Any]] = {}
    try:
        while pending:
            running = {}
            for index in sorted(pending):
                attempts[index] += 1
                stdout_path = log_dir / f"worker{index}_attempt{attempts[index]}.out.log"
                stderr_path = log_dir / f"worker{index}_attempt{attempts[index]}.err.log"
                stdout_handle = stdout_path.open("a", encoding="utf-8")
                stderr_handle = stderr_path.open("a", encoding="utf-8")
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--source-root",
                    str(source_root),
                    "--output-root",
                    str(root),
                    "--data-dir",
                    str(args.data_dir.resolve()),
                    "--config",
                    str(args.config.resolve()),
                    "--device",
                    devices[index],
                    "--threads",
                    str(args.threads),
                    "--classifier-max-epochs",
                    str(args.classifier_max_epochs),
                    "--classifier-patience",
                    str(args.classifier_patience),
                    "--bootstrap-samples",
                    str(args.bootstrap_samples),
                    "--shard-index",
                    str(index),
                    "--shard-count",
                    str(len(devices)),
                    "--fold-plan",
                    str(fold_plan),
                ]
                if args.no_resume:
                    command.append("--no-resume")
                popen_options: dict[str, Any] = {
                    "cwd": str(ROOT),
                    "stdout": stdout_handle,
                    "stderr": stderr_handle,
                }
                if os.name == "nt":
                    popen_options["creationflags"] = subprocess.CREATE_NO_WINDOW
                else:
                    popen_options["start_new_session"] = True
                process = subprocess.Popen(command, **popen_options)
                running[index] = (process, stdout_handle, stderr_handle)
                statuses[index] = {
                    "device": devices[index],
                    "pid": process.pid,
                    "attempt": attempts[index],
                    "state": "running",
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                }
                print(f"LAUNCH worker={index} device={devices[index]} pid={process.pid}", flush=True)
            exp.write_json(log_dir / "parallel_status.json", statuses)
            failed: set[int] = set()
            for index, (process, stdout_handle, stderr_handle) in running.items():
                return_code = process.wait()
                stdout_handle.close()
                stderr_handle.close()
                statuses[index]["return_code"] = return_code
                statuses[index]["state"] = "complete" if return_code == 0 else "failed"
                if return_code != 0:
                    failed.add(index)
                print(f"EXIT worker={index} device={devices[index]} code={return_code}", flush=True)
            exp.write_json(log_dir / "parallel_status.json", statuses)
            exhausted = [index for index in failed if attempts[index] > args.max_retries]
            if exhausted:
                raise RuntimeError(f"workers exhausted retries: {exhausted}; inspect {log_dir}")
            pending = failed
    except KeyboardInterrupt:
        for process, stdout_handle, stderr_handle in running.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.time() + 15
        for process, stdout_handle, stderr_handle in running.values():
            try:
                process.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                process.kill()
            stdout_handle.close()
            stderr_handle.close()
        exp.write_json(
            log_dir / "parallel_status.json",
            {
                **statuses,
                "interrupted": True,
                "resume_command": "rerun the identical --launch-parallel command",
            },
        )
        print("INTERRUPTED: epoch checkpoints retained; rerun the same command to resume", flush=True)
        raise


def progress_summary(root: Path) -> dict[str, Any]:
    completed = sum(
        1
        for path in (root / METHOD_DIR).glob("*/*/seed*/run_metrics.json")
        if (path.parent / "test_predictions.csv").exists()
    )
    resume_files = list((root / METHOD_DIR).glob("*/*/seed*/inceptiontime_resume.pt"))
    return {
        "output_root": str(root),
        "method": METHOD_NAME,
        "kernel_sizes": list(KERNEL_SIZES),
        "completed_runs": completed,
        "total_runs": TOTAL_RUNS,
        "completion_percent": 100.0 * completed / TOTAL_RUNS,
        "incomplete_epoch_checkpoints": len(resume_files),
        "ready_to_finalize": completed == TOTAL_RUNS,
    }


def aggregate_raw_results(root: Path, bootstrap_samples: int) -> dict[str, Any]:
    (root / "tables").mkdir(parents=True, exist_ok=True)
    subject_seed_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for subject in exp.SUBJECTS:
        for seed in exp.SEEDS:
            paths = sorted((root / METHOD_DIR / subject).glob(f"*/seed{seed}/test_predictions.csv"))
            if not paths:
                raise FileNotFoundError(f"missing predictions {subject} seed={seed}")
            rows = [row for path in paths for row in exp.read_csv(path)]
            keys = [(row["record_id"], int(row["window_start"])) for row in rows]
            if len(keys) != len(set(keys)):
                raise AssertionError(f"duplicate outer predictions {subject} seed={seed}")
            y = np.asarray([int(row["y_true"]) for row in rows])
            probability = np.asarray([float(row["y_prob"]) for row in rows])
            prediction = np.asarray([int(row["y_pred"]) for row in rows])
            metrics = exp.binary_metrics(y, probability, prediction)
            events = exp.event_metrics(rows)
            subject_seed_rows.append(
                {
                    "subject_id": subject,
                    "method": METHOD,
                    "method_name": METHOD_NAME,
                    "seed": seed,
                    **metrics,
                    **events,
                }
            )
            for row in rows:
                prediction_rows.append(
                    {
                        **row,
                        "subject_id": subject,
                        "method": METHOD,
                        "seed": seed,
                        "window_start": int(row["window_start"]),
                        "y_true": int(row["y_true"]),
                        "y_prob": float(row["y_prob"]),
                        "y_pred": int(row["y_pred"]),
                    }
                )
            exp.write_csv(root / "predictions" / subject / f"{METHOD}_seed{seed}.csv", rows)

    exp.write_csv(root / "metrics" / "subject_seed_metrics.csv", subject_seed_rows)
    frame = pd.DataFrame(subject_seed_rows)
    numeric = list(exp.CLASSIFICATION_METRICS) + [
        "tn",
        "fp",
        "fn",
        "tp",
        "prevalence",
        "windows",
        "event_sensitivity",
        "detected_events",
        "total_events",
        "false_alarm_episodes",
        "false_alarms_per_minute",
        "median_detection_latency_seconds",
        "average_alarm_duration_seconds",
    ]
    subject_median = frame.groupby(
        ["subject_id", "method", "method_name"], as_index=False
    )[numeric].median(numeric_only=True)
    subject_median.to_csv(
        root / "tables" / "subject_level_main_results.csv", index=False, encoding="utf-8-sig"
    )

    macro: dict[str, Any] = {
        "method": METHOD,
        "method_name": METHOD_NAME,
        "subjects": len(subject_median),
    }
    for metric in exp.CLASSIFICATION_METRICS:
        values = subject_median[metric].to_numpy(float)
        ci_low, ci_high = exp.bootstrap_ci(
            values, "mean", bootstrap_samples, 20260804 + exp.stable_int(metric) % 100000
        )
        macro[f"macro_{metric}"] = float(np.nanmean(values))
        macro[f"median_{metric}"] = float(np.nanmedian(values))
        macro[f"iqr_{metric}"] = float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))
        macro[f"macro_{metric}_ci_low"] = ci_low
        macro[f"macro_{metric}_ci_high"] = ci_high
    exp.write_csv(root / "tables" / "all_subject_summary.csv", [macro])

    predictions = pd.DataFrame(prediction_rows)
    keys = ["subject_id", "method", "record_id", "window_start"]
    pooled = predictions.groupby(keys, as_index=False).agg(
        y_true=("y_true", "first"),
        y_prob=("y_prob", "median"),
        y_pred=("y_pred", lambda values: int(np.median(values) >= 0.5)),
    )
    pooled.to_csv(
        root / "predictions" / "seed_median_pooled_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pooled_metrics = exp.binary_metrics(
        pooled["y_true"].to_numpy(), pooled["y_prob"].to_numpy(), pooled["y_pred"].to_numpy()
    )
    exp.write_csv(
        root / "tables" / "pooled_window_metrics.csv",
        [{"method": METHOD, "method_name": METHOD_NAME, **pooled_metrics}],
    )
    result = {
        "experiment": EXPERIMENT,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "subjects": list(exp.SUBJECTS),
        "method": METHOD,
        "method_name": METHOD_NAME,
        "kernel_sizes": list(KERNEL_SIZES),
        "classifier_seeds": list(exp.SEEDS),
        "macro_results": [macro],
        "pooled_window_metrics": pooled_metrics,
        "test_data_used_for_selection": False,
    }
    exp.write_json(root / "FINAL_RESULTS.json", result)
    write_report(root, macro, pooled_metrics)
    return result


def write_report(root: Path, macro: dict[str, Any], pooled: dict[str, Any]) -> None:
    lines = [
        "# Daphnet Raw + InceptionTime小卷积核复核",
        "",
        f"- 卷积核：{list(KERNEL_SIZES)}",
        "- 结构：6个Inception Module，在第3和第6模块后残差连接。",
        "- 外层：被试内留一完整记录测试；测试数据不参与训练、early stopping或阈值选择。",
        f"- 分类器种子：{list(exp.SEEDS)}",
        "",
        "## 8被试宏平均",
        "",
        "| PR-AUC | ROC-AUC | FoG F1 | Recall | Specificity | BAcc | MCC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {macro['macro_pr_auc']:.4f} | {macro['macro_roc_auc']:.4f} | "
            f"{macro['macro_fog_f1']:.4f} | {macro['macro_recall']:.4f} | "
            f"{macro['macro_specificity']:.4f} | {macro['macro_balanced_accuracy']:.4f} | "
            f"{macro['macro_mcc']:.4f} |"
        ),
        "",
        "## 种子中位池化窗口结果",
        "",
        "| PR-AUC | ROC-AUC | FoG F1 | Recall | Specificity | BAcc | MCC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {pooled['pr_auc']:.4f} | {pooled['roc_auc']:.4f} | {pooled['fog_f1']:.4f} | "
            f"{pooled['recall']:.4f} | {pooled['specificity']:.4f} | "
            f"{pooled['balanced_accuracy']:.4f} | {pooled['mcc']:.4f} |"
        ),
    ]
    path = root / "reports" / "raw_inceptiontime_k359_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "daphnet_full_subject_tcndae_inceptiontime_server_v1"
            / "full_subject_binary_experiment"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "daphnet_full_subject_raw_inceptiontime_k359_server_v1"
            / "full_subject_binary_experiment"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "daphnet_full_subject_raw_inceptiontime_k359.yaml",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--classifier-max-epochs", type=int, default=50)
    parser.add_argument("--classifier-patience", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--only-fold", default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--fold-plan", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6")
    parser.add_argument("--launch-parallel", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_experiment(resume=not args.no_resume)
    source_root = args.source_root.resolve()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.status_only:
        print(json.dumps(progress_summary(root), ensure_ascii=False, indent=2), flush=True)
        return
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: {device}")
        torch.cuda.set_device(device)
    architecture = SmallKernelInceptionTimeClassifier(9).architecture_config()
    print(
        f"CONFIG method={METHOD} kernels={architecture['kernel_sizes']} "
        f"parameters={architecture['parameter_count']} total_runs={TOTAL_RUNS}",
        flush=True,
    )

    if not args.finalize_only:
        if not args.worker:
            full.hardlink_tree(source_root / "splits", root / "splits")
            write_protocol(root, source_root, args.config)
        if args.prepare_only:
            print(f"PREPARED {root}", flush=True)
            return
        if args.launch_parallel and not args.worker:
            launch_parallel_workers(args, root, source_root)
            aggregate_raw_results(root, args.bootstrap_samples)
            print(f"COMPLETE {root} method={METHOD} kernels={KERNEL_SIZES}", flush=True)
            return

        dataset = DaphnetDataset.load(args.data_dir.resolve())
        items = {subject: exp.build_subject_windows(dataset, subject) for subject in exp.SUBJECTS}
        all_folds: list[tuple[str, dict[str, Any]]] = []
        for subject, item in items.items():
            all_folds.extend((subject, fold) for fold in exp.outer_folds(item))
        selected = all_folds
        if args.only_fold:
            wanted_subject, wanted_fold = args.only_fold.split("/", 1)
            selected = [
                (subject, fold)
                for subject, fold in selected
                if subject == wanted_subject and str(fold["fold_id"]) == wanted_fold
            ]
            if len(selected) != 1:
                raise ValueError(f"unknown --only-fold {args.only_fold}")
        elif args.fold_plan is not None:
            plan = json.loads(args.fold_plan.resolve().read_text(encoding="utf-8"))
            wanted = set(plan["workers"][str(args.shard_index)]["folds"])
            selected = [
                (subject, fold)
                for subject, fold in selected
                if f"{subject}/{fold['fold_id']}" in wanted
            ]
            if len(selected) != len(wanted):
                raise ValueError(f"fold plan mismatch for worker {args.shard_index}")
        elif args.shard_count > 1:
            if not 0 <= args.shard_index < args.shard_count:
                raise ValueError("shard-index must be in [0, shard-count)")
            selected = [
                entry for index, entry in enumerate(selected) if index % args.shard_count == args.shard_index
            ]
        if args.smoke:
            selected = selected[:1]
        for position, (subject, fold) in enumerate(selected, 1):
            print(
                f"OUTER {position}/{len(selected)} {subject}/{fold['fold_id']} device={device}",
                flush=True,
            )
            exp.run_outer_fold(
                items[subject],
                fold,
                root,
                device,
                1,
                1,
                min(args.classifier_max_epochs, 2) if args.smoke else args.classifier_max_epochs,
                min(args.classifier_patience, 1) if args.smoke else args.classifier_patience,
            )
    if args.smoke or args.only_fold or args.shard_count > 1:
        print(f"PARTIAL COMPLETE {root}", flush=True)
        return
    aggregate_raw_results(root, args.bootstrap_samples)
    print(f"COMPLETE {root} method={METHOD} kernels={KERNEL_SIZES}", flush=True)


if __name__ == "__main__":
    main()
