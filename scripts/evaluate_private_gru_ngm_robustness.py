#!/usr/bin/env python3
"""Evaluate one frozen Private GRU-NGM/TCN under paired test corruptions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.resume import atomic_json_dump, canonical_fingerprint, sha256_file
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts import train_private_gru_ngm_robustness_tcn as training


base = training.base
EXPERIMENT_SCHEMA = "private_gru_ngm_robustness_evaluation.v1"
GAUSSIAN_SIGMAS = (0.0, 0.02, 0.04, 0.08, 0.12)
MASK_RHOS = (0.0, 0.025, 0.05, 0.10, 0.15)
CORRUPTION_SEED = 20260828
EVALUATION_BATCH_SIZE = 128
METRICS_NAME = "robustness_metrics.csv"
METADATA_NAME = "ROBUSTNESS_EVALUATION.json"
DONE_NAME = "DONE_ROBUSTNESS_EVALUATION.json"


def evaluation_contract() -> dict[str, Any]:
    return {
        "test_roles": [0, 1],
        "metric": "average_precision_score over test windows",
        "gaussian_sigma_test": list(GAUSSIAN_SIGMAS),
        "gaussian_definition": (
            "iid N(0,sigma_test^2) added after the frozen role-4 RobustScaler "
            "and per-window/per-axis centering"
        ),
        "temporal_mask_rho": list(MASK_RHOS),
        "temporal_mask_definition": (
            "one uniformly located contiguous interval per window; all 30 "
            "channels replaced by zero in centered/scaled space"
        ),
        "temporal_mask_samples": {
            str(rho): mask_sample_count(rho) for rho in MASK_RHOS
        },
        "mask_length_rule": "floor(rho*128+0.5)",
        "perturbation_location": "observed input before frozen GRU-NGM",
        "residual_definition": "perturbed observed X minus NGM reconstruction",
        "scheme_c": (
            "q=clip((X_observed-Xhat)/(sigma_role5+1e-6),-12,12); "
            "r=q-mean_t(q); [r,abs(r),delta_t(r)]"
        ),
        "paired_randomization": (
            "corruption seed depends on subject/fold/type/level only, never on "
            "training seed or perturbation arm"
        ),
        "corruption_seed": CORRUPTION_SEED,
        "test_time_training": False,
    }


def evaluation_contract_id() -> str:
    return canonical_fingerprint(evaluation_contract())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--trained-root", type=Path, required=True)
    parser.add_argument("--arm", choices=training.ARMS, required=True)
    parser.add_argument("--subject", choices=training.SUBJECTS, required=True)
    parser.add_argument("--fold", type=int, choices=training.FOLDS, required=True)
    parser.add_argument("--seed", type=int, choices=training.SEEDS, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=EVALUATION_BATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def mask_sample_count(rho: float, samples: int = base.WINDOW_SAMPLES) -> int:
    if not 0.0 <= float(rho) <= 1.0:
        raise ValueError(f"mask ratio must be in [0,1], got {rho}")
    return int(np.floor(float(rho) * int(samples) + 0.5))


def paired_condition_seed(
    subject: str,
    fold: int,
    corruption_type: str,
    level: float,
) -> int:
    text = (
        f"{CORRUPTION_SEED}|{subject}|fold_{fold}|{corruption_type}|"
        f"{float(level):.8f}"
    )
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def apply_gaussian_noise(
    clean_centered_scaled: np.ndarray,
    sigma_test: float,
    condition_seed: int,
) -> np.ndarray:
    clean = np.asarray(clean_centered_scaled, dtype=np.float32)
    if sigma_test < 0:
        raise ValueError(f"negative Gaussian sigma: {sigma_test}")
    if sigma_test == 0:
        return np.ascontiguousarray(clean.copy())
    rng = np.random.default_rng(condition_seed)
    noise = rng.normal(0.0, float(sigma_test), size=clean.shape).astype(np.float32)
    return np.ascontiguousarray(clean + noise, dtype=np.float32)


def apply_contiguous_time_mask(
    clean_centered_scaled: np.ndarray,
    rho_mask: float,
    condition_seed: int,
) -> tuple[np.ndarray, int]:
    clean = np.asarray(clean_centered_scaled, dtype=np.float32)
    if clean.ndim != 3 or clean.shape[1:] != (
        base.WINDOW_SAMPLES,
        base.RAW_CHANNELS,
    ):
        raise ValueError(f"expected [B,128,30], got {clean.shape}")
    length = mask_sample_count(rho_mask)
    output = np.ascontiguousarray(clean.copy())
    if length == 0:
        return output, length
    rng = np.random.default_rng(condition_seed)
    starts = rng.integers(
        0,
        base.WINDOW_SAMPLES - length + 1,
        size=len(output),
        endpoint=False,
    )
    for index, start in enumerate(starts.tolist()):
        output[index, start : start + length, :] = 0.0
    return output, length


def scheme_c_from_observed(
    model: torch.nn.Module,
    sigma_role5: np.ndarray,
    observed: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.ascontiguousarray(observed, dtype=np.float32)
    x_hat = base.reconstruct(model, x, device, batch_size)
    unbounded = (x - x_hat) / (sigma_role5[None, None, :] + 1e-6)
    q = np.clip(unbounded, -12.0, 12.0)
    r = q - q.mean(axis=1, keepdims=True)
    features = np.concatenate(
        (r, np.abs(r), np.diff(r, axis=1, prepend=r[:, :1, :])),
        axis=2,
    ).astype(np.float32, copy=False)
    if features.shape[1:] != (
        base.WINDOW_SAMPLES,
        base.TCN_INPUT_CHANNELS,
    ):
        raise AssertionError(f"unexpected Scheme-C shape: {features.shape}")
    features_bct = np.ascontiguousarray(features.transpose(0, 2, 1))
    return features_bct, {
        "clipped_fraction": float(np.mean(np.abs(unbounded) > 12.0)),
        "maximum_absolute_feature": float(np.max(np.abs(features_bct))),
    }


def load_training_context(
    trained_root: Path,
    arm: str,
    subject: str,
    fold: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    root = trained_root.resolve()
    plan = training.load_plan(root)
    barrier_path = root / "TCN_TRAINING_BARRIER.json"
    done_training_path = root / "DONE_TCN_TRAINING.json"
    if not barrier_path.is_file() or not done_training_path.is_file():
        raise FileNotFoundError(
            "robustness test is locked until matched TCN training and its global "
            "barrier are complete"
        )
    barrier = json.loads(barrier_path.read_text(encoding="utf-8"))
    done_training = json.loads(done_training_path.read_text(encoding="utf-8"))
    if barrier.get("status") != (
        "all_matched_ngm_tcn_pipelines_frozen_before_robustness_test"
    ):
        raise AssertionError("invalid or unsealed matched-TCN training barrier")
    if barrier.get("plan_id") != plan.get("plan_id"):
        raise AssertionError("training barrier/plan mismatch")
    if done_training.get("training_barrier_sha256") != sha256_file(barrier_path):
        raise AssertionError("DONE_TCN_TRAINING barrier hash mismatch")
    if barrier.get("job_count") != plan.get("job_count"):
        raise AssertionError("training barrier job count mismatch")

    key = training.job_key(arm, subject, fold, seed)
    sealed = barrier.get("jobs", {}).get(key)
    source = plan.get("source_jobs", {}).get(key)
    if sealed is None or source is None:
        raise KeyError(f"job absent from frozen training plan/barrier: {key}")
    destination = training.run_dir(root, arm, subject, fold, seed)
    if not training.completed_training_is_valid(destination, plan, source):
        raise FileNotFoundError(f"matched TCN training incomplete: {destination}")
    frozen = json.loads(
        (destination / "FROZEN_TCN.json").read_text(encoding="utf-8")
    )
    expected = {
        "source_ngm_checkpoint_sha256": frozen[
            "source_ngm_checkpoint_sha256"
        ],
        "tcn_checkpoint_sha256": frozen["tcn_checkpoint_sha256"],
        "calibration_sha256": frozen["calibration_sha256"],
        "frozen_id": frozen["frozen_id"],
    }
    for name, value in expected.items():
        if sealed.get(name) != value:
            raise AssertionError(f"training barrier {name} mismatch for {key}")
    return plan, barrier, frozen, destination


def load_tcn(
    destination: Path,
    frozen: dict[str, Any],
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = destination / "checkpoints" / "tcn.pt"
    if sha256_file(checkpoint) != frozen["tcn_checkpoint_sha256"]:
        raise AssertionError(f"frozen TCN checkpoint changed: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("seed") != seed:
        raise AssertionError(f"TCN checkpoint seed mismatch: {checkpoint}")
    if payload.get("input_channels") != base.TCN_INPUT_CHANNELS:
        raise AssertionError(f"TCN input-channel mismatch: {checkpoint}")
    model = base.RepresentationTCNM(base.TCN_INPUT_CHANNELS).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != (
        base.TCN_PARAMETER_COUNT
    ):
        raise AssertionError("90-channel TCN parameter contract changed")
    model.eval()
    return model


def completed_evaluation_is_valid(
    destination: Path,
    barrier: dict[str, Any],
) -> bool:
    result_dir = destination / "robustness_test"
    done_path = result_dir / DONE_NAME
    if not done_path.is_file():
        return False
    metrics_path = result_dir / METRICS_NAME
    metadata_path = result_dir / METADATA_NAME
    if not metrics_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"incomplete robustness evaluation: {result_dir}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    valid = (
        done.get("barrier_id") == barrier.get("barrier_id")
        and metadata.get("barrier_id") == barrier.get("barrier_id")
        and done.get("evaluation_contract_id") == evaluation_contract_id()
        and metadata.get("evaluation_contract_id") == evaluation_contract_id()
        and done.get("metrics_sha256") == sha256_file(metrics_path)
        and done.get("metadata_sha256") == sha256_file(metadata_path)
    )
    if not valid:
        raise AssertionError(f"completed robustness artifacts invalid: {result_dir}")
    return True


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    base.write_csv(path, rows)


def metric_row(
    *,
    arm: str,
    subject: str,
    fold: int,
    seed: int,
    corruption_type: str,
    x_name: str,
    x_value: float,
    mask_samples: int,
    condition_seed: int,
    labels: np.ndarray,
    probability: np.ndarray,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    ap = float(average_precision_score(labels, probability))
    if not np.isfinite(ap):
        raise FloatingPointError("non-finite test AP")
    return {
        "arm": arm,
        "arm_display_name": training.ARM_DISPLAY_NAMES[arm],
        "subject": subject,
        "fold": fold,
        "seed": seed,
        "corruption_type": corruption_type,
        "x_name": x_name,
        "x_value": float(x_value),
        "x_percent": (
            float(x_value) * 100.0 if corruption_type == "temporal_mask" else ""
        ),
        "mask_samples": mask_samples if corruption_type == "temporal_mask" else 0,
        "realized_mask_fraction": (
            mask_samples / base.WINDOW_SAMPLES
            if corruption_type == "temporal_mask"
            else 0.0
        ),
        "condition_seed": condition_seed,
        "n_windows": int(len(labels)),
        "n_nonfog": int(np.sum(labels == 0)),
        "n_fog": int(np.sum(labels == 1)),
        "ap": ap,
        **diagnostics,
    }


def evaluate_observed(
    *,
    ngm: torch.nn.Module,
    tcn: torch.nn.Module,
    sigma_role5: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    features, diagnostics = scheme_c_from_observed(
        ngm, sigma_role5, observed, device, batch_size
    )
    predicted_labels, probability = base.predict(
        tcn, features, labels, device, batch_size
    )
    if not np.array_equal(predicted_labels, labels):
        raise AssertionError("test label order changed during inference")
    return probability, diagnostics


def run_evaluate(args: argparse.Namespace) -> None:
    args.data_dir = args.data_dir.resolve()
    args.trained_root = args.trained_root.resolve()
    if args.batch_size != EVALUATION_BATCH_SIZE:
        raise ValueError(
            f"evaluation batch size is frozen to {EVALUATION_BATCH_SIZE}"
        )
    plan, barrier, frozen, destination = load_training_context(
        args.trained_root,
        args.arm,
        args.subject,
        args.fold,
        args.seed,
    )
    if not args.overwrite and completed_evaluation_is_valid(destination, barrier):
        print(f"SKIP validated robustness evaluation: {destination}", flush=True)
        return
    current_scientific = processed_nbm_scientific_manifest(args.data_dir)["sha256"]
    if current_scientific != plan["data_scientific_sha256"]:
        raise AssertionError("Private scientific dataset changed after training")
    if str(args.data_dir) != plan["data_dir"]:
        raise AssertionError(
            f"evaluation data path differs from frozen plan: {args.data_dir}"
        )

    dataset = DaphnetDataset.load(args.data_dir)
    if (
        dataset.sampling_rate_hz != base.SAMPLING_RATE_HZ
        or dataset.n_channels != base.RAW_CHANNELS
    ):
        raise AssertionError("expected Private 64-Hz/30-channel dataset")
    rows = base.raw_base.load_subject_rows(
        args.data_dir, dataset, args.subject, args.fold
    )
    test_rows = rows.take_role(0, 1)
    if len(test_rows) == 0:
        raise ValueError("test roles 0/1 are empty")
    expected_labels = np.isin(test_rows.role, [1]).astype(np.int8)
    if not np.array_equal(test_rows.label, expected_labels):
        raise AssertionError("test labels do not match roles 0/1")
    labels = test_rows.label.astype(np.int8, copy=False)
    if len(np.unique(labels)) != 2:
        raise ValueError("test roles 0/1 must contain both classes for AP")

    source = frozen["source_ngm"]
    scaler = training.scaler_from_dict(source["scaler"])
    clean = base.centered_scaled_ntc(
        scaler, base.raw_base.raw_windows(dataset, test_rows)
    )
    calibration_path = destination / "calibration_role5.json"
    if sha256_file(calibration_path) != frozen["calibration_sha256"]:
        raise AssertionError(f"role-5 calibration changed: {calibration_path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    sigma_role5 = np.asarray(calibration["sigma"], dtype=np.float32)
    if sigma_role5.shape != (base.RAW_CHANNELS,) or np.any(sigma_role5 < 0.05):
        raise AssertionError("invalid frozen role-5 calibration sigma")

    device = base.resolve_device(args.device)
    ngm = training.load_source_model(source, device)
    tcn = load_tcn(destination, frozen, args.seed, device)
    result_rows: list[dict[str, Any]] = []

    clean_seed = paired_condition_seed(args.subject, args.fold, "clean", 0.0)
    clean_probability, clean_diagnostics = evaluate_observed(
        ngm=ngm,
        tcn=tcn,
        sigma_role5=sigma_role5,
        observed=clean,
        labels=labels,
        device=device,
        batch_size=args.batch_size,
    )
    result_rows.append(
        metric_row(
            arm=args.arm,
            subject=args.subject,
            fold=args.fold,
            seed=args.seed,
            corruption_type="gaussian",
            x_name="sigma_test",
            x_value=0.0,
            mask_samples=0,
            condition_seed=clean_seed,
            labels=labels,
            probability=clean_probability,
            diagnostics=clean_diagnostics,
        )
    )
    result_rows.append(
        metric_row(
            arm=args.arm,
            subject=args.subject,
            fold=args.fold,
            seed=args.seed,
            corruption_type="temporal_mask",
            x_name="rho_mask",
            x_value=0.0,
            mask_samples=0,
            condition_seed=clean_seed,
            labels=labels,
            probability=clean_probability,
            diagnostics=clean_diagnostics,
        )
    )

    for sigma_test in GAUSSIAN_SIGMAS[1:]:
        condition_seed = paired_condition_seed(
            args.subject, args.fold, "gaussian", sigma_test
        )
        observed = apply_gaussian_noise(clean, sigma_test, condition_seed)
        probability, diagnostics = evaluate_observed(
            ngm=ngm,
            tcn=tcn,
            sigma_role5=sigma_role5,
            observed=observed,
            labels=labels,
            device=device,
            batch_size=args.batch_size,
        )
        result_rows.append(
            metric_row(
                arm=args.arm,
                subject=args.subject,
                fold=args.fold,
                seed=args.seed,
                corruption_type="gaussian",
                x_name="sigma_test",
                x_value=sigma_test,
                mask_samples=0,
                condition_seed=condition_seed,
                labels=labels,
                probability=probability,
                diagnostics=diagnostics,
            )
        )

    for rho_mask in MASK_RHOS[1:]:
        condition_seed = paired_condition_seed(
            args.subject, args.fold, "temporal_mask", rho_mask
        )
        observed, length = apply_contiguous_time_mask(
            clean, rho_mask, condition_seed
        )
        probability, diagnostics = evaluate_observed(
            ngm=ngm,
            tcn=tcn,
            sigma_role5=sigma_role5,
            observed=observed,
            labels=labels,
            device=device,
            batch_size=args.batch_size,
        )
        result_rows.append(
            metric_row(
                arm=args.arm,
                subject=args.subject,
                fold=args.fold,
                seed=args.seed,
                corruption_type="temporal_mask",
                x_name="rho_mask",
                x_value=rho_mask,
                mask_samples=length,
                condition_seed=condition_seed,
                labels=labels,
                probability=probability,
                diagnostics=diagnostics,
            )
        )

    result_rows.sort(
        key=lambda row: (
            0 if row["corruption_type"] == "gaussian" else 1,
            float(row["x_value"]),
        )
    )
    if len(result_rows) != len(GAUSSIAN_SIGMAS) + len(MASK_RHOS):
        raise AssertionError("robustness condition count mismatch")
    clean_ap = [
        row["ap"] for row in result_rows if float(row["x_value"]) == 0.0
    ]
    if len(clean_ap) != 2 or clean_ap[0] != clean_ap[1]:
        raise AssertionError("clean AP differs between the two figure baselines")

    result_dir = destination / "robustness_test"
    metrics_path = result_dir / METRICS_NAME
    metadata_path = result_dir / METADATA_NAME
    write_csv(metrics_path, result_rows)
    metadata = {
        "schema": EXPERIMENT_SCHEMA,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": plan["plan_id"],
        "barrier_id": barrier["barrier_id"],
        "evaluation_contract_id": evaluation_contract_id(),
        "evaluation_contract": evaluation_contract(),
        "arm": args.arm,
        "subject": args.subject,
        "fold": args.fold,
        "seed": args.seed,
        "source_ngm_checkpoint_sha256": frozen[
            "source_ngm_checkpoint_sha256"
        ],
        "tcn_checkpoint_sha256": frozen["tcn_checkpoint_sha256"],
        "calibration_sha256": frozen["calibration_sha256"],
        "test_window_count": int(len(labels)),
        "test_nonfog_count": int(np.sum(labels == 0)),
        "test_fog_count": int(np.sum(labels == 1)),
        "condition_count": len(result_rows),
        "test_time_training": False,
    }
    atomic_json_dump(metadata, metadata_path)
    atomic_json_dump(
        {
            "schema": EXPERIMENT_SCHEMA,
            "status": "complete",
            "barrier_id": barrier["barrier_id"],
            "evaluation_contract_id": evaluation_contract_id(),
            "metrics_sha256": sha256_file(metrics_path),
            "metadata_sha256": sha256_file(metadata_path),
        },
        result_dir / DONE_NAME,
    )
    print(
        f"ROBUSTNESS COMPLETE arm={args.arm} subject={args.subject} "
        f"fold={args.fold} seed={args.seed} clean_ap={clean_ap[0]:.7f}",
        flush=True,
    )


def main() -> None:
    run_evaluate(parse_args())


if __name__ == "__main__":
    main()
