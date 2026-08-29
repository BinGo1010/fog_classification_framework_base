#!/usr/bin/env python3
"""Evaluate a frozen 64-Hz Daphnet RAW+TCN under additive Gaussian noise.

This program is evaluation-only.  It never trains a network and never changes
the validation-selected checkpoint or threshold.  Noise is injected into the
raw 2-second IMU window before the frozen role-4 RobustScaler and the existing
per-window/per-axis centering step.

For every window and every sensor axis, signal power is measured after removing
that axis' temporal mean.  Independent Gaussian noise is then scaled to the
requested SNR:

    P_signal = mean_t((x - mean_t(x)) ** 2)
    P_noise  = P_signal / 10 ** (SNR_dB / 10)
    x_noisy  = x + Normal(0, P_noise)

The corruption seed depends on (fold, SNR) only.  Consequently all five frozen
TCN seeds in one fold see bit-identical noisy test windows, enabling a paired
robustness comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.pop("MKL_SERVICE_FORCE_INTEL", None)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.evaluation import binary_metrics
from scripts.run_daphnet_nbm300_c_vs_raw_ablation import (
    FOLDS,
    SUBJECTS,
    build_test_data_manifest,
    job_dir as source_job_dir,
    load_and_validate_barrier,
    load_records_rows,
    load_scaler_only,
    raw_features,
    raw_windows_dynamic,
    stable_json_hash,
)
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    RepresentationTCNM,
)
from scripts.run_daphnet_residual_calibration_abcd import sha256_file
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    classifier_predict,
    write_csv,
    write_json,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import save_npz

REQUIRED_SEEDS = (0, 52, 161, 5216, 52161)
SNR_LEVELS = (30, 20, 10, 0)
PRIMARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "auprc",
    "auroc",
)
SOURCE_BATCH_SIZE = 128
SOURCE_TCN_MAX_EPOCHS = 5
SOURCE_TCN_PATIENCE = 2
SOURCE_TCN_LEARNING_RATE = 3e-3
SOURCE_TCN_WEIGHT_DECAY = 1e-3
WINDOW_SAMPLES = 128


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("evaluate", "aggregate"), required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "dataset"
            / "1.Daphnet Freezing of Gait Dataset"
            / "processed_NBM"
        ),
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--scaler-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--snr-db", type=int)
    parser.add_argument("--seeds", default="0,52,161,5216,52161")
    parser.add_argument("--snr-levels", default="30,20,10,0")
    parser.add_argument("--batch-size", type=int, default=SOURCE_BATCH_SIZE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def noise_seed(fold: int, snr_db: int) -> int:
    """Intentionally independent of the frozen model seed."""
    if fold not in FOLDS or snr_db not in SNR_LEVELS:
        raise ValueError(f"unsupported fold/SNR: {fold}/{snr_db}")
    return 7_310_000 + fold * 100 + snr_db


def add_gaussian_noise_at_snr(
    raw: np.ndarray,
    snr_db: int,
    fold: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add per-window/per-axis AC-power-referenced Gaussian noise."""
    if raw.ndim != 3 or raw.shape[1:] != (WINDOW_SAMPLES, 9):
        raise AssertionError(f"expected raw [N,128,9], got {raw.shape}")
    x = np.asarray(raw, dtype=np.float64)
    ac = x - np.mean(x, axis=1, keepdims=True)
    signal_power = np.mean(np.square(ac), axis=1, keepdims=True)
    target_noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    rng = np.random.default_rng(noise_seed(fold, snr_db))
    standard_normal = rng.standard_normal(x.shape)
    noise = standard_normal * np.sqrt(target_noise_power)
    noisy = np.ascontiguousarray(x + noise, dtype=np.float32)
    noise32 = np.ascontiguousarray(noise, dtype=np.float32)

    valid = signal_power[:, 0, :] > 0.0
    actual_noise_power = np.mean(np.square(noise), axis=1)
    signal_power_2d = signal_power[:, 0, :]
    valid_signal = float(np.sum(signal_power_2d[valid]))
    valid_noise = float(np.sum(actual_noise_power[valid]))
    realized = (
        float(10.0 * np.log10(valid_signal / valid_noise))
        if valid_signal > 0.0 and valid_noise > 0.0
        else float("nan")
    )
    contract = {
        "schema": "raw_test_gaussian_snr.v1",
        "snr_db": int(snr_db),
        "noise_seed": noise_seed(fold, snr_db),
        "seed_scope": "fold_and_snr_only; shared by all five TCN model seeds",
        "injection_point": "raw window before frozen role4 RobustScaler",
        "power_reference": "per-window/per-axis temporal-mean-removed AC power",
        "formula": (
            "Ps=mean_t((x-mean_t(x))^2); Pn=Ps/10^(SNR_dB/10); "
            "x_noisy=x+N(0,Pn)"
        ),
        "zero_ac_power_axis_windows": int(valid.size - np.sum(valid)),
        "realized_pooled_snr_db": realized,
        "noise_float32_sha256": hashlib.sha256(noise32.tobytes()).hexdigest(),
    }
    contract["contract_sha256"] = stable_json_hash(contract)
    return noisy, contract


def robustness_job_dir(root: Path, snr_db: int, fold: int, seed: int) -> Path:
    return root / "runs" / f"snr_{snr_db}" / f"fold_{fold}" / f"seed_{seed}"


def load_source_job(
    source_root: Path,
    fold: int,
    seed: int,
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    barrier_path = source_root / "TRAINING_BARRIER.json"
    if not barrier_path.is_file():
        raise FileNotFoundError(f"source TRAINING_BARRIER missing: {barrier_path}")
    barrier = load_and_validate_barrier(barrier_path)
    if "RAW" not in tuple(barrier.get("methods", ())):
        raise AssertionError("source barrier does not contain RAW")
    expected_job_id = f"fold{fold}_methodRAW_seed{seed}"
    jobs = {str(item["job_id"]): item for item in barrier["jobs"]}
    if expected_job_id not in jobs:
        raise AssertionError(f"source job absent from barrier: {expected_job_id}")
    sealed = jobs[expected_job_id]
    directory = source_job_dir(source_root, fold, "RAW", seed)
    frozen_path = directory / "frozen_validation.json"
    checkpoint = directory / "checkpoints" / "tcn.pt"
    if not frozen_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"source frozen/checkpoint missing: {directory}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("job_id") != expected_job_id or frozen.get("method") != "RAW":
        raise AssertionError("source frozen job identity mismatch")
    if int(frozen["fold"]) != fold or int(frozen["tcn_seed"]) != seed:
        raise AssertionError("source frozen fold/seed mismatch")
    if list(frozen.get("input_shape", ())) != ["B", 9, 128]:
        raise AssertionError("source is not a 9-channel 64-Hz RAW model")
    training = frozen["training"]
    expected_training = {
        "maximum_epochs": SOURCE_TCN_MAX_EPOCHS,
        "patience": SOURCE_TCN_PATIENCE,
        "batch_size": SOURCE_BATCH_SIZE,
        "learning_rate": SOURCE_TCN_LEARNING_RATE,
        "weight_decay": SOURCE_TCN_WEIGHT_DECAY,
    }
    missing = []
    for key, expected in expected_training.items():
        if key not in training or not np.isclose(float(training[key]), float(expected)):
            missing.append(f"{key}={training.get(key)!r}, expected {expected!r}")
    if missing:
        raise AssertionError("wrong source training configuration: " + "; ".join(missing))
    checkpoint_sha = sha256_file(checkpoint)
    if frozen.get("checkpoint_sha256") != checkpoint_sha:
        raise AssertionError("source checkpoint hash differs from frozen validation")
    if sealed.get("checkpoint_sha256") != checkpoint_sha:
        raise AssertionError("source checkpoint hash differs from barrier")
    if not np.isclose(float(sealed["threshold"]), float(frozen["threshold"])):
        raise AssertionError("source threshold differs from barrier")
    return barrier, checkpoint, frozen, sealed


def validate_current_test_manifest(
    data_dir: Path,
    barrier: dict[str, Any],
) -> str:
    rows_by_fold = {fold: load_records_rows(data_dir, fold)[1] for fold in FOLDS}
    current = build_test_data_manifest(data_dir, rows_by_fold)
    sealed_sha = barrier.get("test_data_manifest", {}).get("sha256")
    if sealed_sha is not None and current["sha256"] != sealed_sha:
        raise AssertionError("current permanent-test data differs from source seal")
    return str(current["sha256"])


def subject_metrics(
    subject_ids: np.ndarray,
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for subject in SUBJECTS:
        mask = subject_ids == subject
        if np.any(mask):
            result[subject] = binary_metrics(y_true[mask], probability[mask], threshold)
    return result


def run_evaluate(args: argparse.Namespace) -> None:
    if args.fold not in FOLDS or args.seed not in REQUIRED_SEEDS:
        raise ValueError("evaluate requires a valid --fold and one of the five seeds")
    if args.snr_db not in SNR_LEVELS:
        raise ValueError(f"evaluate requires --snr-db in {SNR_LEVELS}")
    if args.batch_size != SOURCE_BATCH_SIZE:
        raise ValueError(f"this evaluation requires batch_size={SOURCE_BATCH_SIZE}")

    data_dir = args.data_dir.resolve()
    source_root = args.source_root.resolve()
    output = robustness_job_dir(
        args.output_root.resolve(), args.snr_db, args.fold, args.seed
    )
    done_path = output / "DONE_TEST.json"
    if done_path.exists() and not args.overwrite:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        expected_files = {
            "metrics_sha256": output / "metrics.json",
            "predictions_sha256": output / "test_predictions.csv",
            "probabilities_sha256": output / "test_probabilities.npz",
        }
        if done.get("status") != "complete":
            raise AssertionError(f"invalid completed marker: {done_path}")
        for key, path in expected_files.items():
            if not path.is_file() or done.get(key) != sha256_file(path):
                raise AssertionError(f"completed output failed resume audit: {path}")
        print(f"SKIP complete {output}", flush=True)
        return

    barrier, checkpoint, frozen, sealed = load_source_job(
        source_root, args.fold, args.seed
    )
    test_manifest_sha = validate_current_test_manifest(data_dir, barrier)
    records, rows = load_records_rows(data_dir, args.fold)
    test_rows = rows.take_role(0, 1)
    raw = raw_windows_dynamic(records, test_rows, WINDOW_SAMPLES)
    noisy_raw, noise_contract = add_gaussian_noise_at_snr(
        raw, args.snr_db, args.fold
    )

    scaler_root = args.scaler_source_root.resolve() / f"seed_{args.seed}"
    scaler, scaler_artifact, _ = load_scaler_only(scaler_root, args.fold, "gru")
    sealed_scaler_sha = sealed.get("scaler_sha256")
    if sealed_scaler_sha and scaler_artifact["scaler_sha256"] != sealed_scaler_sha:
        raise AssertionError("role4 scaler differs from source barrier")
    features = raw_features(scaler, noisy_raw, WINDOW_SAMPLES)

    device = resolve_device(args.device)
    model = RepresentationTCNM(9).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if int(payload.get("input_channels", -1)) != 9:
        raise AssertionError("source checkpoint is not a 9-channel TCN")
    model.load_state_dict(payload["model_state"], strict=True)
    y_true, probability = classifier_predict(
        model,
        features,
        np.asarray(test_rows.label, dtype=np.int8),
        device,
        batch_size=args.batch_size,
    )
    threshold = float(frozen["threshold"])
    metrics = binary_metrics(y_true, probability, threshold)
    by_subject = subject_metrics(
        np.asarray(test_rows.subject_id), y_true, probability, threshold
    )
    prediction = (probability >= threshold).astype(np.int8)

    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "test_predictions.csv"
    probabilities_path = output / "test_probabilities.npz"
    metrics_path = output / "metrics.json"
    prediction_rows = []
    for index in range(len(test_rows)):
        prediction_rows.append(
            {
                "subject_id": str(test_rows.subject_id[index]),
                "record_id": str(test_rows.record_id[index]),
                "window_id": str(test_rows.window_id[index]),
                "start_index": int(test_rows.start[index]),
                "end_index_exclusive": int(test_rows.end[index]),
                "y_true": int(y_true[index]),
                "probability": float(probability[index]),
                "threshold": threshold,
                "y_pred": int(prediction[index]),
                "snr_db": int(args.snr_db),
                "noise_seed": int(noise_contract["noise_seed"]),
            }
        )
    write_csv(predictions_path, prediction_rows)
    save_npz(
        probabilities_path,
        y_true=y_true,
        probability=probability.astype(np.float32),
        y_pred=prediction,
        subject_id=np.asarray(test_rows.subject_id).astype(str),
    )
    result = {
        "schema": "daphnet_raw_tcn_test_noise_result.v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "fold": int(args.fold),
        "model_seed": int(args.seed),
        "condition": f"SNR{args.snr_db}",
        "snr_db": int(args.snr_db),
        "threshold": threshold,
        "threshold_policy": "frozen from clean roles2/3; never retuned on noisy test",
        "test": metrics,
        "test_by_subject": by_subject,
        "noise": noise_contract,
        "source": {
            "source_root": str(source_root),
            "source_barrier_id": barrier.get("barrier_id"),
            "source_job_id": frozen["job_id"],
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": sha256_file(checkpoint),
            "scaler_sha256": scaler_artifact["scaler_sha256"],
            "test_data_manifest_sha256": test_manifest_sha,
        },
    }
    write_json(metrics_path, result)
    write_json(
        done_path,
        {
            "status": "complete",
            "completed_utc": result["completed_utc"],
            "fold": int(args.fold),
            "model_seed": int(args.seed),
            "snr_db": int(args.snr_db),
            "noise_contract_sha256": noise_contract["contract_sha256"],
            "source_checkpoint_sha256": result["source"]["source_checkpoint_sha256"],
            "metrics_sha256": sha256_file(metrics_path),
            "predictions_sha256": sha256_file(predictions_path),
            "probabilities_sha256": sha256_file(probabilities_path),
        },
    )
    print(
        f"SNR{args.snr_db} fold={args.fold} seed={args.seed} "
        f"sens={metrics['sensitivity']:.6f} spec={metrics['specificity']:.6f} "
        f"pr_auc={metrics['auprc']:.6f}",
        flush=True,
    )


def mean_std(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "n": int(len(array)),
    }


def clean_result(source_root: Path, fold: int, seed: int) -> dict[str, Any]:
    path = source_job_dir(source_root, fold, "RAW", seed) / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"clean source test metrics missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_aggregate(args: argparse.Namespace) -> None:
    seeds = parse_csv_ints(args.seeds)
    snr_levels = parse_csv_ints(args.snr_levels)
    if seeds != REQUIRED_SEEDS or snr_levels != SNR_LEVELS:
        raise ValueError("aggregate requires exact five seeds and SNR 30,20,10,0")
    root = args.output_root.resolve()
    source_root = args.source_root.resolve()
    run_rows: list[dict[str, Any]] = []
    run_data: dict[tuple[str, int, int], dict[str, Any]] = {}

    for fold in FOLDS:
        for seed in seeds:
            clean = clean_result(source_root, fold, seed)
            run_data[("CLEAN", fold, seed)] = clean
            clean_test = clean["test"]
            run_rows.append(
                {"condition": "CLEAN", "snr_db": "", "fold": fold, "seed": seed,
                 **{metric: clean_test[metric] for metric in PRIMARY_METRICS}}
            )
            for snr_db in snr_levels:
                directory = robustness_job_dir(root, snr_db, fold, seed)
                done = directory / "DONE_TEST.json"
                metrics_path = directory / "metrics.json"
                if not done.is_file() or not metrics_path.is_file():
                    raise FileNotFoundError(
                        f"noise evaluation incomplete: SNR{snr_db}/fold{fold}/seed{seed}"
                    )
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                if sha256_file(metrics_path) != json.loads(
                    done.read_text(encoding="utf-8")
                )["metrics_sha256"]:
                    raise AssertionError(f"metrics hash mismatch: {metrics_path}")
                condition = f"SNR{snr_db}"
                run_data[(condition, fold, seed)] = payload
                test = payload["test"]
                run_rows.append(
                    {"condition": condition, "snr_db": snr_db, "fold": fold, "seed": seed,
                     **{metric: test[metric] for metric in PRIMARY_METRICS}}
                )

    conditions = ("CLEAN",) + tuple(f"SNR{value}" for value in snr_levels)
    seed_rows: list[dict[str, Any]] = []
    for condition in conditions:
        for seed in seeds:
            selected = [
                row for row in run_rows
                if row["condition"] == condition and int(row["seed"]) == seed
            ]
            seed_rows.append(
                {"condition": condition, "seed": seed,
                 **{metric: float(np.mean([row[metric] for row in selected]))
                    for metric in PRIMARY_METRICS}}
            )

    summary: dict[str, Any] = {}
    for condition in conditions:
        selected = [row for row in seed_rows if row["condition"] == condition]
        summary[condition] = {
            metric: mean_std(row[metric] for row in selected)
            for metric in PRIMARY_METRICS
        }

    paired_delta: dict[str, Any] = {}
    clean_by_seed = {
        int(row["seed"]): row for row in seed_rows if row["condition"] == "CLEAN"
    }
    for condition in conditions[1:]:
        condition_by_seed = {
            int(row["seed"]): row
            for row in seed_rows if row["condition"] == condition
        }
        paired_delta[condition] = {
            metric: mean_std(
                condition_by_seed[seed][metric] - clean_by_seed[seed][metric]
                for seed in seeds
            )
            for metric in PRIMARY_METRICS
        }

    subject_rows: list[dict[str, Any]] = []
    for condition in conditions:
        for subject in SUBJECTS:
            subject_seed_metrics: dict[int, dict[str, float]] = {}
            for seed in seeds:
                folds = []
                for fold in FOLDS:
                    payload = run_data[(condition, fold, seed)]
                    if subject not in payload["test_by_subject"]:
                        raise AssertionError(f"subject {subject} missing from {condition}")
                    folds.append(payload["test_by_subject"][subject])
                subject_seed_metrics[seed] = {
                    metric: float(np.mean([item[metric] for item in folds]))
                    for metric in PRIMARY_METRICS
                }
            row: dict[str, Any] = {"condition": condition, "subject_id": subject}
            for metric in PRIMARY_METRICS:
                value = mean_std(subject_seed_metrics[seed][metric] for seed in seeds)
                row[f"{metric}_mean"] = value["mean"]
                row[f"{metric}_std"] = value["std"]
            subject_rows.append(row)

    write_csv(root / "run_metrics_75.csv", run_rows)
    write_csv(root / "seed_macro_over_3folds.csv", seed_rows)
    write_csv(root / "subject_summary.csv", subject_rows)
    payload = {
        "schema": "daphnet_raw_tcn_test_noise_summary.v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "aggregation": (
            "within each model seed macro-average the three folds; then mean and "
            "population SD over the five model seeds"
        ),
        "source_root": str(source_root),
        "conditions": list(conditions),
        "summary": summary,
        "paired_delta_vs_clean": paired_delta,
    }
    write_json(root / "summary.json", payload)
    write_json(
        root / "DONE.json",
        {
            "status": "complete",
            "completed_utc": payload["completed_utc"],
            "noise_run_count": 60,
            "clean_reference_run_count": 15,
            "summary_sha256": sha256_file(root / "summary.json"),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.stage == "evaluate":
        run_evaluate(args)
    else:
        run_aggregate(args)


if __name__ == "__main__":
    main()
