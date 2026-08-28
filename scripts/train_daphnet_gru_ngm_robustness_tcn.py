#!/usr/bin/env python3
"""Train one clean Scheme-C TCN matched to one frozen Daphnet GRU-NGM.

This training-only stage reads roles 4/5/6/7/2/3.  It recomputes role-5
calibration from the frozen NGM, trains the TCN on clean roles 6/7, selects the
TCN checkpoint on clean roles 2/3 AP, and never materializes roles 0/1.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.resume import atomic_json_dump, canonical_fingerprint, sha256_file
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts import run_daphnet_nbm300_c_vs_raw_ablation as ablation
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    RobustScaler,
    choose_document_threshold,
    classifier_predict,
    residual_diagnostics,
    write_csv,
)
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    train_representation_tcn,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    GRUReconstructionNBM,
    calibrate,
    prepare_nbm_windows,
    reconstruct,
)


EXPERIMENT_SCHEMA = "daphnet_gru_ngm_robustness_matched_tcn.v1"
PLAN_SCHEMA = "daphnet_gru_ngm_robustness_matched_tcn_plan.v1"
ARMS = ("none", "gaussian_mask")
ARM_DISPLAY_NAMES = {
    "none": "No perturbation",
    "gaussian_mask": "Gaussian + Mask",
}
FOLDS = (0, 1, 2)
SEEDS = (0, 52, 161, 5216, 52161)
SOURCE_CHECKPOINT_NAMES = ("gru_ngm_best.pt", "gru_nbm_best.pt")
TCN_MAX_EPOCHS = 5
TCN_PATIENCE = 2
TCN_BATCH_SIZE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--fold", type=int, choices=FOLDS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tcn-max-epochs", type=int, default=TCN_MAX_EPOCHS)
    parser.add_argument("--tcn-patience", type=int, default=TCN_PATIENCE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    return device


def job_key(arm: str, fold: int, seed: int) -> str:
    return f"{arm}/fold_{fold}/seed_{seed}"


def run_dir(root: Path, arm: str, fold: int, seed: int) -> Path:
    return root / "runs" / arm / f"fold_{fold}" / f"seed_{seed}"


def _checkpoint_in(directory: Path) -> Path | None:
    for name in SOURCE_CHECKPOINT_NAMES:
        for path in (directory / "checkpoints" / name, directory / name):
            if path.is_file():
                return path
    return None


def source_fold_candidates(root: Path, fold: int, seed: int) -> tuple[Path, ...]:
    """Supported layouts for one arm's 3-fold x 5-seed NGM artifacts."""

    return (
        root / f"seed_{seed}" / f"fold_{fold}",
        root / f"seed{seed}" / f"fold_{fold}",
        root / f"fold_{fold}" / f"seed_{seed}",
        root / "runs" / f"fold_{fold}" / f"seed_{seed}",
        root / f"fold_{fold}",
    )


def resolve_source_fold_dir(root: Path, fold: int, seed: int) -> Path:
    matches = [
        candidate.resolve()
        for candidate in source_fold_candidates(root.resolve(), fold, seed)
        if _checkpoint_in(candidate) is not None
    ]
    unique = tuple(dict.fromkeys(matches))
    if not unique:
        expected = "\n".join(str(path) for path in source_fold_candidates(root, fold, seed))
        raise FileNotFoundError(
            f"no GRU-NGM checkpoint found for fold={fold}, seed={seed}; checked:\n{expected}"
        )
    if len(unique) != 1:
        raise RuntimeError(
            f"ambiguous GRU-NGM source for fold={fold}, seed={seed}: {unique}"
        )
    return unique[0]


def _scaler_source(directory: Path) -> Path:
    for name in ("scaler_role4.json", "nbm_frozen.json"):
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"role-4 scaler missing under {directory}; expected scaler_role4.json "
        "or nbm_frozen.json"
    )


def scaler_dict_from_path(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scaler = payload.get("scaler", payload)
    required = ("median", "iqr", "epsilon")
    if not all(key in scaler for key in required):
        raise KeyError(f"invalid scaler artifact: {path}")
    median = np.asarray(scaler["median"], dtype=np.float32)
    iqr = np.asarray(scaler["iqr"], dtype=np.float32)
    if median.shape != (9,) or iqr.shape != (9,):
        raise ValueError(f"Daphnet scaler must contain nine channels: {path}")
    if not np.all(np.isfinite(median)) or not np.all(np.isfinite(iqr)):
        raise FloatingPointError(f"non-finite scaler values: {path}")
    return {
        "median": median.astype(float).tolist(),
        "iqr": iqr.astype(float).tolist(),
        "epsilon": float(scaler["epsilon"]),
    }


def scaler_from_dict(payload: dict[str, Any]) -> RobustScaler:
    return RobustScaler(
        median=np.asarray(payload["median"], dtype=np.float32),
        iqr=np.asarray(payload["iqr"], dtype=np.float32),
        epsilon=float(payload["epsilon"]),
    )


def _optional_json_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def inspect_source_artifacts(
    root: Path,
    fold: int,
    seed: int,
    scientific_data_sha256: str | None = None,
) -> dict[str, Any]:
    directory = resolve_source_fold_dir(root, fold, seed)
    checkpoint = _checkpoint_in(directory)
    if checkpoint is None:
        raise FileNotFoundError(directory / "checkpoints" / SOURCE_CHECKPOINT_NAMES[0])
    scaler_source = _scaler_source(directory)
    scaler = scaler_dict_from_path(scaler_source)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model_state" not in payload:
        raise KeyError(f"source checkpoint has no model_state: {checkpoint}")
    if payload.get("seed") is not None and int(payload["seed"]) != seed:
        raise AssertionError(
            f"source checkpoint seed mismatch: {payload.get('seed')} != {seed}"
        )
    probe = GRUReconstructionNBM(channels=9, hidden=64, bottleneck=16)
    probe.load_state_dict(payload["model_state"], strict=True)
    parameter_count = sum(parameter.numel() for parameter in probe.parameters())
    del probe

    frozen_path = directory / "nbm_frozen.json"
    done_path = directory / "DONE_NBM.json"
    for metadata_path in (frozen_path, done_path, scaler_source):
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_data_hash = metadata.get("scientific_data_sha256")
        if (
            scientific_data_sha256 is not None
            and source_data_hash is not None
            and source_data_hash != scientific_data_sha256
        ):
            raise AssertionError(
                f"source NGM dataset mismatch in {metadata_path}: "
                f"{source_data_hash} != {scientific_data_sha256}"
            )
        if metadata_path == done_path and metadata.get("checkpoint_sha256") not in (
            None,
            sha256_file(checkpoint),
        ):
            raise AssertionError(f"source DONE_NBM checkpoint hash mismatch: {done_path}")

    return {
        "source_root": str(root.resolve()),
        "source_fold_dir": str(directory),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_name": checkpoint.name,
        "checkpoint_seed": int(payload["seed"]) if payload.get("seed") is not None else None,
        "checkpoint_step": payload.get("step"),
        "checkpoint_epoch": payload.get("epoch"),
        "parameter_count": parameter_count,
        "scaler_source": str(scaler_source.resolve()),
        "scaler_source_sha256": sha256_file(scaler_source),
        "scaler": scaler,
        "scaler_sha256": canonical_fingerprint(scaler),
        "nbm_frozen_sha256": _optional_json_sha256(frozen_path),
        "done_nbm_sha256": _optional_json_sha256(done_path),
    }


def load_plan(plan_root: Path) -> dict[str, Any]:
    path = plan_root.resolve() / "EXPERIMENT_PLAN.json"
    if not path.is_file():
        raise FileNotFoundError(f"TCN experiment plan missing: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise AssertionError(f"unexpected TCN plan schema: {plan.get('schema')}")
    return plan


def validate_args_against_plan(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    if args.tcn_max_epochs != TCN_MAX_EPOCHS or args.tcn_patience != TCN_PATIENCE:
        raise ValueError(
            f"matched TCN is frozen to max_epochs={TCN_MAX_EPOCHS}, "
            f"patience={TCN_PATIENCE}"
        )
    expected = {
        "data_dir": str(args.data_dir.resolve()),
        "output_root": str(args.output_root.resolve()),
        "tcn_max_epochs": args.tcn_max_epochs,
        "tcn_patience": args.tcn_patience,
        "tcn_batch_size": TCN_BATCH_SIZE,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise AssertionError(
                f"plan/worker mismatch for {key}: {plan.get(key)!r} != {value!r}"
            )
    if job_key(args.arm, args.fold, args.seed) not in plan.get("source_jobs", {}):
        raise KeyError(f"job absent from frozen plan: {job_key(args.arm, args.fold, args.seed)}")


def load_source_model(
    source: dict[str, Any], device: torch.device
) -> GRUReconstructionNBM:
    checkpoint = Path(source["checkpoint"])
    if sha256_file(checkpoint) != source["checkpoint_sha256"]:
        raise AssertionError(f"source GRU-NGM checkpoint changed: {checkpoint}")
    scaler_source = Path(source["scaler_source"])
    if sha256_file(scaler_source) != source["scaler_source_sha256"]:
        raise AssertionError(f"source role-4 scaler artifact changed: {scaler_source}")
    if canonical_fingerprint(scaler_dict_from_path(scaler_source)) != source["scaler_sha256"]:
        raise AssertionError(f"source role-4 scaler values changed: {scaler_source}")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = GRUReconstructionNBM(channels=9, hidden=64, bottleneck=16).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


def scheme_c_features(
    model: GRUReconstructionNBM,
    scaler: RobustScaler,
    sigma: np.ndarray,
    raw: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaled = prepare_nbm_windows(scaler, raw, center=True)
    reconstruction = reconstruct(model, scaled, device)
    error_bct = np.ascontiguousarray(
        (scaled - reconstruction).transpose(0, 2, 1), dtype=np.float32
    )
    return ablation.build_scheme_c_features(
        error_bct,
        labels,
        sigma,
        window_samples=128,
        expand=True,
    )


def completed_training_is_valid(
    destination: Path,
    plan: dict[str, Any],
    source: dict[str, Any],
) -> bool:
    done_path = destination / "DONE_TCN.json"
    if not done_path.is_file():
        return False
    frozen_path = destination / "FROZEN_TCN.json"
    checkpoint = destination / "checkpoints" / "tcn.pt"
    history = destination / "logs" / "tcn_history.csv"
    calibration = destination / "calibration_role5.json"
    required = (frozen_path, checkpoint, history, calibration)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"incomplete completed TCN job: {destination}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    valid = (
        done.get("frozen_sha256") == sha256_file(frozen_path)
        and done.get("frozen_id") == frozen.get("frozen_id")
        and frozen.get("plan_id") == plan.get("plan_id")
        and frozen.get("source_ngm_checkpoint_sha256")
        == source.get("checkpoint_sha256")
        and frozen.get("tcn_checkpoint_sha256") == sha256_file(checkpoint)
        and frozen.get("tcn_history_sha256") == sha256_file(history)
        and frozen.get("calibration_sha256") == sha256_file(calibration)
        and frozen.get("test_roles_accessed") is False
    )
    if not valid:
        raise AssertionError(f"completed TCN artifacts failed validation: {destination}")
    return True


def run_train(args: argparse.Namespace) -> None:
    args.data_dir = args.data_dir.resolve()
    args.plan_root = args.plan_root.resolve()
    args.output_root = args.output_root.resolve()
    plan = load_plan(args.plan_root)
    validate_args_against_plan(args, plan)
    source = plan["source_jobs"][job_key(args.arm, args.fold, args.seed)]
    destination = run_dir(args.output_root, args.arm, args.fold, args.seed)
    if not args.overwrite and completed_training_is_valid(destination, plan, source):
        print(f"SKIP validated completed TCN job: {destination}", flush=True)
        return

    current_scientific = processed_nbm_scientific_manifest(args.data_dir)["sha256"]
    if current_scientific != plan["data_scientific_sha256"]:
        raise AssertionError("Daphnet scientific dataset changed after plan freeze")
    records, rows = ablation.load_records_rows(args.data_dir, args.fold)
    role4 = rows.take_role(4)
    role5 = rows.take_role(5)
    role67 = rows.take_role(6, 7)
    role23 = rows.take_role(2, 3)
    if min(len(role4), len(role5), len(role67), len(role23)) <= 0:
        raise ValueError("roles 4/5/6/7/2/3 must all be non-empty")
    if not np.array_equal(role67.label, np.isin(role67.role, [7]).astype(np.int8)):
        raise AssertionError("classifier training labels do not match roles 6/7")
    if not np.array_equal(role23.label, np.isin(role23.role, [3]).astype(np.int8)):
        raise AssertionError("classifier validation labels do not match roles 2/3")

    scaler = scaler_from_dict(source["scaler"])
    device = resolve_device(args.device)
    model = load_source_model(source, device)
    role5_x = prepare_nbm_windows(
        scaler,
        ablation.raw_windows_dynamic(records, role5, 128),
        center=True,
    )
    bias, sigma, calibration = calibrate(model, role5_x, device)
    if bias.shape != (9,) or sigma.shape != (9,) or np.any(sigma < 0.05):
        raise AssertionError("invalid clean role-5 GRU-NGM calibration")
    train_x, train_clip = scheme_c_features(
        model,
        scaler,
        sigma,
        ablation.raw_windows_dynamic(records, role67, 128),
        role67.label,
        device,
    )
    validation_x, validation_clip = scheme_c_features(
        model,
        scaler,
        sigma,
        ablation.raw_windows_dynamic(records, role23, 128),
        role23.label,
        device,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    initial_state, initialization = ablation.paired_initialization(
        args.seed, "FULL_C"
    )
    tcn, training = train_representation_tcn(
        "r_abs_delta",
        train_x,
        role67.label,
        validation_x,
        role23.label,
        destination,
        device,
        args.seed,
        args.num_workers,
        args.tcn_max_epochs,
        args.tcn_patience,
        initial_state,
        reset_seed_after_loading=True,
    )
    val_true, val_probability = classifier_predict(
        tcn, validation_x, role23.label, device
    )
    threshold, validation_metrics = choose_document_threshold(
        val_true, val_probability
    )
    del tcn

    destination.mkdir(parents=True, exist_ok=True)
    calibration_path = destination / "calibration_role5.json"
    atomic_json_dump(
        {
            "schema": EXPERIMENT_SCHEMA,
            "arm": args.arm,
            "fold": args.fold,
            "seed": args.seed,
            "source_role": 5,
            "source_ngm_checkpoint_sha256": source["checkpoint_sha256"],
            "bias_used_to_estimate_sigma_only": True,
            "scheme_c_subtracts_bias": False,
            **calibration,
        },
        calibration_path,
    )
    checkpoint = destination / "checkpoints" / "tcn.pt"
    history_path = destination / "logs" / "tcn_history.csv"
    frozen = {
        "schema": EXPERIMENT_SCHEMA,
        "status": "frozen_before_robustness_test",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": plan["plan_id"],
        "arm": args.arm,
        "arm_display_name": ARM_DISPLAY_NAMES[args.arm],
        "fold": args.fold,
        "seed": args.seed,
        "source_ngm": source,
        "source_ngm_checkpoint_sha256": source["checkpoint_sha256"],
        "role4_scaler_sha256": source["scaler_sha256"],
        "calibration_sha256": sha256_file(calibration_path),
        "tcn_checkpoint": str(checkpoint.resolve()),
        "tcn_checkpoint_sha256": sha256_file(checkpoint),
        "tcn_history_sha256": sha256_file(history_path),
        "initialization": initialization,
        "representation": {
            "name": "Scheme C",
            "formula": (
                "e=X-Xhat; q=clip(e/(sigma+1e-6),-12,12); "
                "r=q-mean_t(q); F=[r,abs(r),delta_t(r)]"
            ),
            "input_shape": ["B", 128, 27],
            "train_clip_statistics": train_clip,
            "validation_clip_statistics": validation_clip,
        },
        "roles": {
            "ngm_calibration": [5],
            "tcn_train": [6, 7],
            "tcn_validation": [2, 3],
            "test_not_accessed": [0, 1],
        },
        "training": {key: value for key, value in training.items() if key != "history"},
        "threshold": float(threshold),
        "threshold_source_roles": [2, 3],
        "threshold_not_used_for_ap": True,
        "validation": validation_metrics,
        "feature_diagnostics": {
            "roles_6_7_train": residual_diagnostics(train_x, role67.label),
            "roles_2_3_validation": residual_diagnostics(
                validation_x, role23.label
            ),
        },
        "test_roles_accessed": False,
    }
    frozen["frozen_id"] = canonical_fingerprint(frozen)
    frozen_path = destination / "FROZEN_TCN.json"
    atomic_json_dump(frozen, frozen_path)
    atomic_json_dump(
        {
            "schema": EXPERIMENT_SCHEMA,
            "status": "train_complete",
            "arm": args.arm,
            "fold": args.fold,
            "seed": args.seed,
            "frozen_id": frozen["frozen_id"],
            "frozen_sha256": sha256_file(frozen_path),
            "test_roles_accessed": False,
        },
        destination / "DONE_TCN.json",
    )
    print(
        f"TCN TRAIN COMPLETE arm={args.arm} fold={args.fold} seed={args.seed} "
        f"best_epoch={training['best_epoch']} "
        f"validation_ap={training['best_validation_pr_auc']:.7f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    run_train(args)


if __name__ == "__main__":
    main()
