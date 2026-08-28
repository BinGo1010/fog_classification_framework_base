#!/usr/bin/env python3
"""Evaluate all frozen Private robustness pipelines on 8 GPUs and aggregate CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import t as student_t


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.resume import atomic_json_dump, canonical_fingerprint, sha256_file
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts import evaluate_private_gru_ngm_robustness as worker
from scripts import train_private_gru_ngm_robustness_tcn as training
from scripts.launch_daphnet_residual_calibration_abcd_7gpu import run_pool


WORKER = REPO_ROOT / "scripts" / "evaluate_private_gru_ngm_robustness.py"
DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "0.Private" / "processed_NBM_Exp"
DEFAULT_TRAINED_ROOT = (
    REPO_ROOT / "outputs" / "private_gru_ngm_robustness_matched_tcn"
)
PER_FOLD_CSV = "ROBUSTNESS_PER_FOLD.csv"
SUBJECT_SEED_CSV = "ROBUSTNESS_SUBJECT_SEED.csv"
SUBJECT_SUMMARY_CSV = "ROBUSTNESS_SUBJECT_SUMMARY.csv"
OVERALL_SEED_CSV = "ROBUSTNESS_OVERALL_SEED.csv"
CURVE_LONG_CSV = "ROBUSTNESS_CURVE_SUMMARY_LONG.csv"
FIG1_CSV = "FIG1_GAUSSIAN_NOISE_AP.csv"
FIG2_CSV = "FIG2_TEMPORAL_MASK_AP.csv"


def parse_csv_values(text: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in text.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid unique comma-separated values: {text}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--trained-root", type=Path, default=DEFAULT_TRAINED_ROOT)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--batch-size", type=int, default=worker.EVALUATION_BATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_global_context(
    data_dir: Path,
    trained_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = trained_root.resolve()
    plan = training.load_plan(root)
    barrier_path = root / "TCN_TRAINING_BARRIER.json"
    done_path = root / "DONE_TCN_TRAINING.json"
    if not barrier_path.is_file() or not done_path.is_file():
        raise FileNotFoundError(
            "matched TCN training is incomplete; expected "
            "TCN_TRAINING_BARRIER.json and DONE_TCN_TRAINING.json"
        )
    barrier = json.loads(barrier_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if barrier.get("status") != (
        "all_matched_ngm_tcn_pipelines_frozen_before_robustness_test"
    ):
        raise AssertionError("invalid matched-TCN training barrier status")
    if barrier.get("plan_id") != plan.get("plan_id"):
        raise AssertionError("training barrier/plan mismatch")
    if done.get("training_barrier_sha256") != sha256_file(barrier_path):
        raise AssertionError("DONE_TCN_TRAINING barrier hash mismatch")
    if barrier.get("job_count") != plan.get("job_count"):
        raise AssertionError("training barrier job count mismatch")
    if len(barrier.get("jobs", {})) != plan.get("job_count"):
        raise AssertionError("training barrier job inventory mismatch")
    scientific = processed_nbm_scientific_manifest(data_dir.resolve())
    if scientific["sha256"] != plan.get("data_scientific_sha256"):
        raise AssertionError("Private scientific dataset differs from training plan")
    if str(data_dir.resolve()) != plan.get("data_dir"):
        raise AssertionError("evaluation data path differs from frozen training plan")
    return plan, barrier


def common_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--data-dir",
        str(args.data_dir.resolve()),
        "--trained-root",
        str(args.trained_root.resolve()),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.overwrite:
        values.append("--overwrite")
    return values


def jobs(args: argparse.Namespace, plan: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for subject in plan["subjects"]:
        for fold in plan["folds"]:
            for seed in plan["seeds"]:
                for arm in training.ARMS:
                    output.append(
                        {
                            "id": f"{subject}_fold{fold}_seed{seed}_{arm}_robustness",
                            "command": [
                                args.python,
                                "-u",
                                str(WORKER),
                                "--arm",
                                arm,
                                "--subject",
                                subject,
                                "--fold",
                                str(fold),
                                "--seed",
                                str(seed),
                                "--device",
                                "cuda:0",
                                *common_args(args),
                            ],
                        }
                    )
    if len(output) != plan["job_count"]:
        raise AssertionError(
            f"evaluation job count mismatch: {len(output)} != {plan['job_count']}"
        )
    return output


def read_metric_rows(path: Path) -> list[dict[str, Any]]:
    integer_columns = {
        "fold",
        "seed",
        "mask_samples",
        "condition_seed",
        "n_windows",
        "n_nonfog",
        "n_fog",
    }
    float_columns = {
        "x_value",
        "x_percent",
        "realized_mask_fraction",
        "ap",
        "clipped_fraction",
        "maximum_absolute_feature",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    converted: list[dict[str, Any]] = []
    for row in rows:
        output: dict[str, Any] = dict(row)
        for name in integer_columns:
            output[name] = int(row[name])
        for name in float_columns:
            output[name] = float(row[name]) if row[name] != "" else ""
        converted.append(output)
    return converted


def collect_per_fold_rows(
    root: Path,
    plan: dict[str, Any],
    barrier: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_levels = {
        "gaussian": set(worker.GAUSSIAN_SIGMAS),
        "temporal_mask": set(worker.MASK_RHOS),
    }
    for subject in plan["subjects"]:
        for fold in plan["folds"]:
            for seed in plan["seeds"]:
                for arm in training.ARMS:
                    destination = training.run_dir(root, arm, subject, fold, seed)
                    if not worker.completed_evaluation_is_valid(
                        destination, barrier
                    ):
                        raise FileNotFoundError(
                            f"robustness evaluation incomplete: {destination}"
                        )
                    metrics_path = (
                        destination / "robustness_test" / worker.METRICS_NAME
                    )
                    selected = read_metric_rows(metrics_path)
                    if len(selected) != 10:
                        raise AssertionError(
                            f"expected 10 robustness rows: {metrics_path}"
                        )
                    for row in selected:
                        identity = (
                            row["arm"],
                            row["subject"],
                            row["fold"],
                            row["seed"],
                        )
                        if identity != (arm, subject, fold, seed):
                            raise AssertionError(
                                f"robustness row identity mismatch: {metrics_path}"
                            )
                    for corruption_type, levels in expected_levels.items():
                        actual = {
                            float(row["x_value"])
                            for row in selected
                            if row["corruption_type"] == corruption_type
                        }
                        if actual != levels:
                            raise AssertionError(
                                f"condition grid mismatch: {metrics_path}"
                            )
                    rows.extend(selected)
    expected_count = int(plan["job_count"]) * 10
    if len(rows) != expected_count:
        raise AssertionError(
            f"per-fold robustness row count mismatch: {len(rows)} != {expected_count}"
        )
    clean: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if float(row["x_value"]) == 0.0:
            key = (row["arm"], row["subject"], row["fold"], row["seed"])
            clean[key][row["corruption_type"]] = float(row["ap"])
    for key, values in clean.items():
        if set(values) != {"gaussian", "temporal_mask"}:
            raise AssertionError(f"missing clean baseline: {key}")
        if values["gaussian"] != values["temporal_mask"]:
            raise AssertionError(f"clean AP mismatch between figures: {key}")
    return sorted(
        rows,
        key=lambda row: (
            training.ARMS.index(row["arm"]),
            plan["subjects"].index(row["subject"]),
            int(row["seed"]),
            int(row["fold"]),
            0 if row["corruption_type"] == "gaussian" else 1,
            float(row["x_value"]),
        ),
    )


def population_summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary requires non-empty finite values")
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=0))
    if len(array) > 1:
        sample_std = float(np.std(array, ddof=1))
        sem = float(sample_std / math.sqrt(len(array)))
        half_width = float(student_t.ppf(0.975, len(array) - 1) * sem)
    else:
        sem = 0.0
        half_width = 0.0
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "n": int(len(array)),
    }


def grouped(
    rows: Sequence[dict[str, Any]],
    keys: Sequence[str],
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    output: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[tuple(row[key] for key in keys)].append(row)
    return dict(output)


def condition_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "arm_display_name": row["arm_display_name"],
        "x_name": row["x_name"],
        "x_percent": row["x_percent"],
        "mask_samples": row["mask_samples"],
        "realized_mask_fraction": row["realized_mask_fraction"],
    }


def build_subject_seed_rows(
    rows: Sequence[dict[str, Any]],
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    keys = ("arm", "subject", "seed", "corruption_type", "x_value")
    output: list[dict[str, Any]] = []
    for key, selected in grouped(rows, keys).items():
        if len(selected) != len(plan["folds"]):
            raise AssertionError(f"subject/seed fold count mismatch: {key}")
        summary = population_summary(float(row["ap"]) for row in selected)
        first = selected[0]
        output.append(
            {
                **dict(zip(keys, key)),
                **condition_fields(first),
                "ap": summary["mean"],
                "ap_fold_std": summary["std"],
                "n_folds": summary["n"],
            }
        )
    return sorted(
        output,
        key=lambda row: (
            training.ARMS.index(row["arm"]),
            plan["subjects"].index(row["subject"]),
            plan["seeds"].index(row["seed"]),
            0 if row["corruption_type"] == "gaussian" else 1,
            float(row["x_value"]),
        ),
    )


def build_subject_summary_rows(
    subject_seed_rows: Sequence[dict[str, Any]],
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    keys = ("arm", "subject", "corruption_type", "x_value")
    output: list[dict[str, Any]] = []
    for key, selected in grouped(subject_seed_rows, keys).items():
        if len(selected) != len(plan["seeds"]):
            raise AssertionError(f"subject seed count mismatch: {key}")
        summary = population_summary(float(row["ap"]) for row in selected)
        output.append(
            {
                **dict(zip(keys, key)),
                **condition_fields(selected[0]),
                "ap_mean": summary["mean"],
                "ap_std": summary["std"],
                "ap_sem": summary["sem"],
                "ap_ci95_low": summary["ci95_low"],
                "ap_ci95_high": summary["ci95_high"],
                "n_seeds": summary["n"],
            }
        )
    return sorted(
        output,
        key=lambda row: (
            training.ARMS.index(row["arm"]),
            plan["subjects"].index(row["subject"]),
            0 if row["corruption_type"] == "gaussian" else 1,
            float(row["x_value"]),
        ),
    )


def build_overall_seed_rows(
    subject_seed_rows: Sequence[dict[str, Any]],
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    keys = ("arm", "seed", "corruption_type", "x_value")
    output: list[dict[str, Any]] = []
    for key, selected in grouped(subject_seed_rows, keys).items():
        if len(selected) != len(plan["subjects"]):
            raise AssertionError(f"overall subject count mismatch: {key}")
        summary = population_summary(float(row["ap"]) for row in selected)
        output.append(
            {
                **dict(zip(keys, key)),
                **condition_fields(selected[0]),
                "ap": summary["mean"],
                "ap_subject_std": summary["std"],
                "n_subjects": summary["n"],
            }
        )
    baseline = {
        (row["arm"], row["seed"], row["corruption_type"]): float(row["ap"])
        for row in output
        if float(row["x_value"]) == 0.0
    }
    for row in output:
        clean_ap = baseline[(row["arm"], row["seed"], row["corruption_type"])]
        row["clean_ap"] = clean_ap
        row["ap_drop_from_clean"] = clean_ap - float(row["ap"])
        row["ap_retention"] = float(row["ap"]) / clean_ap if clean_ap > 0 else ""
    return sorted(
        output,
        key=lambda row: (
            training.ARMS.index(row["arm"]),
            plan["seeds"].index(row["seed"]),
            0 if row["corruption_type"] == "gaussian" else 1,
            float(row["x_value"]),
        ),
    )


def build_curve_rows(
    overall_seed_rows: Sequence[dict[str, Any]],
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    keys = ("arm", "corruption_type", "x_value")
    output: list[dict[str, Any]] = []
    for key, selected in grouped(overall_seed_rows, keys).items():
        if len(selected) != len(plan["seeds"]):
            raise AssertionError(f"curve seed count mismatch: {key}")
        ap = population_summary(float(row["ap"]) for row in selected)
        drop = population_summary(
            float(row["ap_drop_from_clean"]) for row in selected
        )
        retention = population_summary(
            float(row["ap_retention"]) for row in selected
        )
        first = selected[0]
        output.append(
            {
                **dict(zip(keys, key)),
                "figure": (
                    "Fig1 Gaussian-noise robustness"
                    if key[1] == "gaussian"
                    else "Fig2 Temporal-masking robustness"
                ),
                **condition_fields(first),
                "ap_mean": ap["mean"],
                "ap_std": ap["std"],
                "ap_sem": ap["sem"],
                "ap_ci95_low": ap["ci95_low"],
                "ap_ci95_high": ap["ci95_high"],
                "ap_drop_from_clean_mean": drop["mean"],
                "ap_drop_from_clean_std": drop["std"],
                "ap_retention_mean": retention["mean"],
                "ap_retention_std": retention["std"],
                "n_training_seeds": ap["n"],
                "n_subjects": len(plan["subjects"]),
                "n_folds": len(plan["folds"]),
                "aggregation": (
                    "3-fold macro within subject/seed; subject macro within "
                    "seed; mean and population SD over 5 training seeds"
                ),
                "ci95_method": (
                    "two-sided Student-t interval over training seeds using "
                    "sample-SD standard error"
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            0 if row["corruption_type"] == "gaussian" else 1,
            float(row["x_value"]),
            training.ARMS.index(row["arm"]),
        ),
    )


def paired_arm_difference(
    overall_seed_rows: Sequence[dict[str, Any]],
    corruption_type: str,
    x_value: float,
    seeds: Sequence[int],
) -> dict[str, float | int]:
    lookup = {
        (row["arm"], row["seed"]): float(row["ap"])
        for row in overall_seed_rows
        if row["corruption_type"] == corruption_type
        and float(row["x_value"]) == float(x_value)
    }
    differences = [
        lookup[("gaussian_mask", seed)] - lookup[("none", seed)]
        for seed in seeds
    ]
    return population_summary(differences)


def build_figure_wide_rows(
    curve_rows: Sequence[dict[str, Any]],
    overall_seed_rows: Sequence[dict[str, Any]],
    corruption_type: str,
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    selected = [row for row in curve_rows if row["corruption_type"] == corruption_type]
    levels = sorted({float(row["x_value"]) for row in selected})
    output: list[dict[str, Any]] = []
    for level in levels:
        by_arm = {
            row["arm"]: row
            for row in selected
            if float(row["x_value"]) == level
        }
        if set(by_arm) != set(training.ARMS):
            raise AssertionError(f"missing figure arm at {corruption_type}={level}")
        difference = paired_arm_difference(
            overall_seed_rows, corruption_type, level, plan["seeds"]
        )
        row: dict[str, Any] = {
            "x_value": level,
            "gaussian_mask_minus_none_ap_mean": difference["mean"],
            "gaussian_mask_minus_none_ap_std": difference["std"],
            "gaussian_mask_minus_none_ap_ci95_low": difference["ci95_low"],
            "gaussian_mask_minus_none_ap_ci95_high": difference["ci95_high"],
        }
        if corruption_type == "gaussian":
            row["sigma_test"] = level
        else:
            row["rho_mask"] = level
            row["rho_mask_percent"] = level * 100.0
            row["mask_samples"] = by_arm["none"]["mask_samples"]
            row["realized_mask_fraction"] = by_arm["none"][
                "realized_mask_fraction"
            ]
        for arm, prefix in (
            ("none", "no_perturbation"),
            ("gaussian_mask", "gaussian_mask"),
        ):
            curve = by_arm[arm]
            for name in (
                "ap_mean",
                "ap_std",
                "ap_sem",
                "ap_ci95_low",
                "ap_ci95_high",
                "ap_drop_from_clean_mean",
                "ap_retention_mean",
            ):
                row[f"{prefix}_{name}"] = curve[name]
        output.append(row)
    return output


def aggregate_results(
    root: Path,
    plan: dict[str, Any],
    barrier: dict[str, Any],
) -> dict[str, Path]:
    per_fold_rows = collect_per_fold_rows(root, plan, barrier)
    subject_seed_rows = build_subject_seed_rows(per_fold_rows, plan)
    subject_summary_rows = build_subject_summary_rows(subject_seed_rows, plan)
    overall_seed_rows = build_overall_seed_rows(subject_seed_rows, plan)
    curve_rows = build_curve_rows(overall_seed_rows, plan)
    fig1_rows = build_figure_wide_rows(
        curve_rows, overall_seed_rows, "gaussian", plan
    )
    fig2_rows = build_figure_wide_rows(
        curve_rows, overall_seed_rows, "temporal_mask", plan
    )
    paths = {
        "per_fold": root / PER_FOLD_CSV,
        "subject_seed": root / SUBJECT_SEED_CSV,
        "subject_summary": root / SUBJECT_SUMMARY_CSV,
        "overall_seed": root / OVERALL_SEED_CSV,
        "curve_long": root / CURVE_LONG_CSV,
        "fig1": root / FIG1_CSV,
        "fig2": root / FIG2_CSV,
    }
    for path, rows in (
        (paths["per_fold"], per_fold_rows),
        (paths["subject_seed"], subject_seed_rows),
        (paths["subject_summary"], subject_summary_rows),
        (paths["overall_seed"], overall_seed_rows),
        (paths["curve_long"], curve_rows),
        (paths["fig1"], fig1_rows),
        (paths["fig2"], fig2_rows),
    ):
        worker.write_csv(path, rows)

    summary = {
        "schema": worker.EXPERIMENT_SCHEMA,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": plan["plan_id"],
        "barrier_id": barrier["barrier_id"],
        "evaluation_contract_id": worker.evaluation_contract_id(),
        "subjects": plan["subjects"],
        "folds": plan["folds"],
        "seeds": plan["seeds"],
        "arms": list(training.ARMS),
        "per_fold_row_count": len(per_fold_rows),
        "aggregation": (
            "3-fold macro within subject/seed; subject macro within seed; "
            "mean and population SD over 5 training seeds"
        ),
        "csv_sha256": {name: sha256_file(path) for name, path in paths.items()},
    }
    summary["summary_id"] = canonical_fingerprint(
        {key: value for key, value in summary.items() if key != "created_utc"}
    )
    summary_path = root / "ROBUSTNESS_EVALUATION_SUMMARY.json"
    atomic_json_dump(summary, summary_path)
    atomic_json_dump(
        {
            "schema": worker.EXPERIMENT_SCHEMA,
            "status": "complete",
            "summary_id": summary["summary_id"],
            "summary_sha256": sha256_file(summary_path),
            "fig1_sha256": sha256_file(paths["fig1"]),
            "fig2_sha256": sha256_file(paths["fig2"]),
        },
        root / "DONE_ROBUSTNESS_EVALUATION.json",
    )
    return paths


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.trained_root = args.trained_root.resolve()
    if args.batch_size != worker.EVALUATION_BATCH_SIZE:
        raise ValueError(
            f"--batch-size is frozen to {worker.EVALUATION_BATCH_SIZE}"
        )
    gpu_ids = list(parse_csv_values(args.gpu_ids))
    if len(gpu_ids) != 8 or any(not value.isdigit() for value in gpu_ids):
        raise ValueError(
            "--gpu-ids must contain exactly eight unique non-negative integers"
        )
    plan, barrier = load_global_context(args.data_dir, args.trained_root)
    evaluation_jobs = jobs(args, plan)
    print(
        f"EVALUATION PLAN jobs={len(evaluation_jobs)} "
        f"conditions_per_job=10 gpus={','.join(gpu_ids)} "
        f"contract={worker.evaluation_contract_id()}",
        flush=True,
    )
    if args.dry_run:
        print(
            "DRY RUN: training barrier and dataset preflight passed; no test roles read",
            flush=True,
        )
        print(
            "FIRST JOB:",
            subprocess.list2cmdline(evaluation_jobs[0]["command"]),
            flush=True,
        )
        print(
            "LAST JOB:",
            subprocess.list2cmdline(evaluation_jobs[-1]["command"]),
            flush=True,
        )
        return

    run_pool("robustness_eval", evaluation_jobs, gpu_ids, args.trained_root)
    paths = aggregate_results(args.trained_root, plan, barrier)
    print(
        f"ROBUSTNESS EVALUATION COMPLETE jobs={len(evaluation_jobs)}",
        flush=True,
    )
    print(f"FIG1 CSV: {paths['fig1']}", flush=True)
    print(f"FIG2 CSV: {paths['fig2']}", flush=True)


if __name__ == "__main__":
    main()
