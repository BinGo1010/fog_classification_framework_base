#!/usr/bin/env python3
"""Strict paired FULL-C versus RAW TCN ablation on processed_NBM.

FULL_C
    role-4 RobustScaler -> per-window/per-axis centering -> selected frozen NBM
    -> e=X-Xhat -> scheme-C r -> [r,abs(r),delta(r)] -> TCN [B,27,128].

RAW
    the identical role-4 RobustScaler -> per-window/per-axis centering
    -> TCN [B,9,128].  The NBM, role-5 b/sigma, and residual path are absent.

Stages enforce a global test barrier: all 18 classifiers and validation-only
thresholds must be frozen before roles 0/1 can be materialized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.evaluation import binary_metrics
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    ROLES,
    SUBJECTS,
    RobustScaler,
    audit_protocol,
    choose_document_threshold,
    classifier_predict,
    load_fold_rows,
    raw_windows,
    residual_diagnostics,
    write_csv,
    write_json,
)
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    RepresentationTCNM,
    centered_scaled_bct,
    paired_tcn_initial_states,
    plot_classifier_training,
    train_representation_tcn,
)
from scripts.run_daphnet_residual_calibration_abcd import (
    build_abcd_features,
    load_frozen_nbm,
    reconstruction_error,
    sha256_file,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    GRUReconstructionNBM,
    prepare_nbm_windows,
    reconstruct as reconstruct_gru,
)

FOLDS = (0, 1, 2)
METHODS = ("FULL_C", "RAW")
REQUIRED_SEEDS = (0, 52, 161)
METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "auprc",
    "auroc",
)


def parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid unique integer list: {value}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("train", "seal", "evaluate", "aggregate"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed_NBM",
    )
    parser.add_argument(
        "--nbm-source-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_conv_tcn_nbm300_C_vs_raw_tcn_ep10pat2_seedset_0_52_161"
        / "nbm_source",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_conv_tcn_nbm300_C_vs_raw_tcn_ep10pat2_seedset_0_52_161",
    )
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--nbm-kind", choices=("conv_tcn", "gru"), default="conv_tcn")
    parser.add_argument("--nbm-seed", type=int)
    parser.add_argument("--tcn-seed", type=int)
    parser.add_argument("--nbm-seeds", default="0,52,161")
    parser.add_argument("--tcn-seeds", default="0,52,161")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tcn-max-epochs", type=int, default=10)
    parser.add_argument("--tcn-patience", type=int, default=2)
    parser.add_argument("--required-nbm-max-epochs", type=int, default=300)
    parser.add_argument("--required-nbm-patience", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    return device


def require_job_args(args: argparse.Namespace) -> None:
    if (
        args.fold is None
        or args.method is None
        or args.nbm_seed is None
        or args.tcn_seed is None
    ):
        raise ValueError(
            f"--fold, --method, --nbm-seed and --tcn-seed are required for {args.stage}"
        )
    if args.nbm_seed != args.tcn_seed:
        raise ValueError("strict paired repeats require NBM seed == TCN seed")
    if args.nbm_seed not in REQUIRED_SEEDS:
        raise ValueError(f"seed must be one of {REQUIRED_SEEDS}")


def job_id(fold: int, method: str, seed: int) -> str:
    return f"fold{fold}_method{method}_seed{seed}"


def job_dir(root: Path, fold: int, method: str, seed: int) -> Path:
    return root / "runs" / f"fold_{fold}" / f"method_{method}" / f"seed_{seed}"


def expected_jobs(seeds: tuple[int, ...]) -> list[tuple[int, str, int]]:
    return [(fold, method, seed) for fold in FOLDS for method in METHODS for seed in seeds]


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_records_rows(data_dir: Path, fold: int) -> tuple[dict[str, Any], Any]:
    dataset = DaphnetDataset.load(data_dir.resolve())
    records = {record.record_id: record for record in dataset.records}
    rows = load_fold_rows(data_dir.resolve(), fold)
    expected = np.isin(rows.role, [1, 3, 7]).astype(np.int8)
    if not np.array_equal(rows.label, expected):
        raise AssertionError(f"role/label mismatch in fold {fold}")
    return records, rows


def load_scaler_only(
    source_root: Path,
    fold: int,
    nbm_kind: str,
) -> tuple[RobustScaler, dict[str, Any], dict[str, Any]]:
    """Load only the role-4 scaler; do not instantiate NBM or read b/sigma."""
    fold_dir = source_root.resolve() / f"fold_{fold}"
    frozen_path = fold_dir / "nbm_frozen.json"
    checkpoint_name = "conv_tcn_nbm_best.pt" if nbm_kind == "conv_tcn" else "gru_nbm_best.pt"
    checkpoint = fold_dir / "checkpoints" / checkpoint_name
    if not frozen_path.exists() or not checkpoint.exists():
        raise FileNotFoundError(f"frozen NBM/scaler artifacts missing: {fold_dir}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    payload = frozen["scaler"]
    scaler = RobustScaler(
        median=np.asarray(payload["median"], dtype=np.float32),
        iqr=np.asarray(payload["iqr"], dtype=np.float32),
        epsilon=float(payload.get("epsilon", 1e-6)),
    )
    manifest = {
        "fold": fold,
        "nbm_kind": nbm_kind,
        "scaler_fit_role": int(frozen["scaler_fit_role"]),
        "scaler_unique_raw_points": int(frozen["scaler_unique_raw_points"]),
        "scaler_sha256": stable_json_hash(payload),
        "frozen_json": str(frozen_path.resolve()),
        "frozen_json_sha256": sha256_file(frozen_path),
        "nbm_checkpoint": str(checkpoint.resolve()),
        "nbm_checkpoint_sha256": sha256_file(checkpoint),
    }
    return scaler, manifest, frozen


def validate_nbm_contract(frozen: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    training = frozen["training"]
    augmentation = training["augmentation"]
    expected_aug = {
        "clean_probability": 0.40,
        "gaussian_probability": 0.40,
        "mask_probability": 0.20,
        "gaussian_std": 0.04,
        "mask_minimum_samples": 4,
        "mask_maximum_samples": 8,
        "mask_all_channels": True,
    }
    checks = {
        "maximum_epochs": int(training["maximum_epochs"]) == args.required_nbm_max_epochs,
        "patience": int(training["patience"]) == args.required_nbm_patience,
        "loss": str(training["loss"]) == "SmoothL1(beta=1.0)",
        "optimizer_lr": "lr=0.001" in str(training["optimizer"]),
        "augmentation": all(augmentation.get(key) == value for key, value in expected_aug.items()),
        "best_checkpoint_restored_before_calibration": bool(
            frozen.get("best_checkpoint_restored_before_calibration", False)
        ),
        "validation_unaugmented": frozen.get("validation_mask_or_noise") is False,
        "exact_seed": (
            args.nbm_seed is not None
            and int(training["seed"]) == args.nbm_seed
        ),
        "architecture": (
            str(training["architecture"]["name"]).startswith("conv_tcn")
            if args.nbm_kind == "conv_tcn"
            else str(training["architecture"]["name"]).startswith("gru_reconstruction")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"frozen NBM does not match requested best configuration: {failed}")
    return {
        "max_epochs": args.required_nbm_max_epochs,
        "patience": args.required_nbm_patience,
        "loss": "SmoothL1(beta=1.0)",
        "optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)",
        "augmentation": expected_aug,
        "checkpoint_rule": "lowest unaugmented role-5 validation SmoothL1",
        "all_checks_passed": True,
        "seed": args.nbm_seed,
        "nbm_kind": args.nbm_kind,
    }


def load_frozen_gru_nbm(
    source_root: Path,
    fold: int,
    device: torch.device,
) -> tuple[GRUReconstructionNBM, RobustScaler, np.ndarray, np.ndarray, dict[str, Any]]:
    scaler, artifact, frozen = load_scaler_only(source_root, fold, "gru")
    training = frozen["training"]
    if training["architecture"]["name"] != "gru_reconstruction_nbm_v1":
        raise AssertionError("unexpected frozen GRU-NBM architecture")
    model = GRUReconstructionNBM(channels=9, hidden=64, bottleneck=16).to(device)
    checkpoint = Path(artifact["nbm_checkpoint"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()
    calibration = frozen["calibration"]
    bias = np.asarray(calibration["bias"], dtype=np.float32)
    sigma = np.asarray(calibration["sigma"], dtype=np.float32)
    if bias.shape != (9,) or sigma.shape != (9,) or np.any(sigma < 0.05):
        raise AssertionError("invalid frozen GRU-NBM role-5 calibration")
    manifest = {
        **artifact,
        "best_epoch": int(training["best_epoch"]),
        "best_validation_loss": float(training["best_validation_huber"]),
        "best_validation_metric": "role5_validation_SmoothL1",
        "calibration_role": 5,
        "scaler_role": 4,
    }
    return model, scaler, bias, sigma, manifest


def raw_features(scaler: RobustScaler, raw: np.ndarray) -> np.ndarray:
    """RobustScaler then per-window/per-axis centering; return [N,128,9]."""
    bct = centered_scaled_bct(scaler, raw)
    if bct.shape[1:] != (9, 128):
        raise AssertionError(f"unexpected RAW tensor shape: {bct.shape}")
    if not np.all(np.isfinite(bct)):
        raise FloatingPointError("RAW tensor contains NaN or infinity after scaling/centering")
    # ``bct`` is intentionally float32 because it is the tensor sent to the
    # classifier.  Summing 128 potentially large scaled values in float32 can
    # leave a small apparent mean (observed around 1.6e-5) even though the
    # window mean was subtracted correctly.  Recheck with float64 accumulation
    # and use a scale-aware float32 tolerance so this audit catches genuine
    # centering failures without rejecting ordinary round-off.
    axis_means = np.mean(bct, axis=2, dtype=np.float64)
    maximum_axis_mean = float(np.max(np.abs(axis_means)))
    maximum_signal = float(np.max(np.abs(bct)))
    centering_tolerance = max(
        1e-5,
        64.0 * float(np.finfo(np.float32).eps) * max(1.0, maximum_signal),
    )
    if maximum_axis_mean > centering_tolerance:
        raise AssertionError(
            "RAW per-window/per-axis centering failed: "
            f"max_mean={maximum_axis_mean}, tolerance={centering_tolerance}, "
            f"max_signal={maximum_signal}"
        )
    return np.ascontiguousarray(bct.transpose(0, 2, 1), dtype=np.float32)


def make_features(
    method: str,
    scaler: RobustScaler,
    raw: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    nbm_source_root: Path,
    fold: int,
    nbm_kind: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if method == "RAW":
        values = raw_features(scaler, raw)
        return values, {
            "formula": "RobustScaler(role4); Xc=X-mean_t(X), input Xc",
            "shape": ["B", 9, 128],
            "uses_nbm": False,
            "uses_role5_b_sigma": False,
            "uses_residual": False,
            "removed_nbm_kind": nbm_kind,
        }
    if nbm_kind == "conv_tcn":
        nbm, full_scaler, bias, sigma, nbm_manifest = load_frozen_nbm(
            nbm_source_root, fold, device
        )
        error = reconstruction_error(nbm, full_scaler, raw, device)
    else:
        nbm, full_scaler, bias, sigma, nbm_manifest = load_frozen_gru_nbm(
            nbm_source_root, fold, device
        )
        scaled = prepare_nbm_windows(full_scaler, raw, center=True)
        reconstruction = reconstruct_gru(nbm, scaled, device)
        error_ntc = (scaled - reconstruction).astype(np.float32, copy=False)
        error = np.ascontiguousarray(error_ntc.transpose(0, 2, 1))
    if stable_json_hash(full_scaler.as_dict()) != stable_json_hash(scaler.as_dict()):
        raise AssertionError("FULL_C and RAW scalers differ")
    values, clip_stats = build_abcd_features(error, labels, "C", bias, sigma)
    del nbm, error
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return values, {
        "formula": "e=X-Xhat; q=clip(e/(sigma+1e-6),-12,12); r=q-mean_t(q); F=[r,abs(r),delta_t(r)]",
        "shape": ["B", 27, 128],
        "uses_nbm": True,
        "uses_role5_b_sigma": True,
        "uses_bias_b": False,
        "uses_sigma": True,
        "nbm_kind": nbm_kind,
        "nbm": nbm_manifest,
        "clip_statistics": clip_stats,
    }


def paired_initialization(seed: int, method: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    raw_state, full_state, hashes = paired_tcn_initial_states(seed)
    pair_id = hashlib.sha256(
        f"{seed}:{hashes['r']}:{hashes['r_abs_delta']}".encode("utf-8")
    ).hexdigest()
    return (full_state if method == "FULL_C" else raw_state), {
        "seed": seed,
        "pair_id": pair_id,
        "raw_9ch_state_sha256": hashes["r"],
        "full_c_27ch_state_sha256": hashes["r_abs_delta"],
        "pairing_rule": (
            "all shape-compatible weights are identical; the 27-channel first layer "
            "copies the first 9 channels and initializes the extra 18 channels to zero"
        ),
    }


def run_train(args: argparse.Namespace, device: torch.device) -> None:
    require_job_args(args)
    directory = job_dir(args.output_root.resolve(), args.fold, args.method, args.tcn_seed)
    done_path = directory / "DONE_TRAIN.json"
    if done_path.exists() and not args.overwrite:
        print(f"SKIP frozen training job: {done_path}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    records, rows = load_records_rows(args.data_dir, args.fold)
    role67 = rows.take_role(6, 7)
    role23 = rows.take_role(2, 3)
    scaler, artifact_manifest, frozen_nbm = load_scaler_only(
        args.nbm_source_root, args.fold, args.nbm_kind
    )
    nbm_contract = validate_nbm_contract(frozen_nbm, args)
    train_x, train_feature = make_features(
        args.method, scaler, raw_windows(records, role67), role67.label,
        device, args.nbm_source_root, args.fold, args.nbm_kind,
    )
    validation_x, validation_feature = make_features(
        args.method, scaler, raw_windows(records, role23), role23.label,
        device, args.nbm_source_root, args.fold, args.nbm_kind,
    )
    initial_state, initialization = paired_initialization(args.tcn_seed, args.method)
    representation = "r_abs_delta" if args.method == "FULL_C" else "r"
    model, training = train_representation_tcn(
        representation,
        train_x,
        role67.label,
        validation_x,
        role23.label,
        directory,
        device,
        args.tcn_seed,
        args.num_workers,
        args.tcn_max_epochs,
        args.tcn_patience,
        initial_state,
        reset_seed_after_loading=True,
    )
    val_true, val_prob = classifier_predict(model, validation_x, role23.label, device)
    threshold, validation_metrics = choose_document_threshold(val_true, val_prob)
    checkpoint = directory / "checkpoints" / "tcn.pt"
    frozen = {
        "job_id": job_id(args.fold, args.method, args.tcn_seed),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "method": args.method,
        "nbm_kind": args.nbm_kind,
        "nbm_seed": args.nbm_seed,
        "tcn_seed": args.tcn_seed,
        "input_shape": train_feature["shape"],
        "feature": train_feature,
        "validation_feature": validation_feature,
        "role4_scaler_artifact": artifact_manifest,
        "nbm_contract": nbm_contract,
        "raw_ablation_role5_policy": (
            "RAW reads only the role-4 scaler fields; it does not load NBM weights, b, sigma, "
            "or use role-5 windows"
        ),
        "roles": {"classifier_train": [6, 7], "classifier_validation": [2, 3], "test_not_accessed": [0, 1]},
        "test_roles_accessed": False,
        "initialization": initialization,
        "training": {key: value for key, value in training.items() if key != "history"},
        "threshold": float(threshold),
        "threshold_source_roles": [2, 3],
        "threshold_rule": "max balanced accuracy; ties FoG F1 then higher threshold; 0.05..0.95 step 0.01",
        "validation": validation_metrics,
        "feature_diagnostics": {
            "roles_6_7_train": residual_diagnostics(train_x, role67.label),
            "roles_2_3_validation": residual_diagnostics(validation_x, role23.label),
        },
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    write_json(directory / "frozen_validation.json", frozen)
    write_json(done_path, {
        "status": "frozen",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": frozen["job_id"],
        "checkpoint_sha256": frozen["checkpoint_sha256"],
        "threshold": threshold,
        "test_roles_accessed": False,
    })
    print(
        f"TRAIN FROZEN {frozen['job_id']} best_epoch={training['best_epoch']} threshold={threshold:.2f}",
        flush=True,
    )


def run_seal(args: argparse.Namespace) -> None:
    nbm_seeds = parse_csv_ints(args.nbm_seeds)
    seeds = parse_csv_ints(args.tcn_seeds)
    if nbm_seeds != seeds:
        raise ValueError("strict paired repeats require identical NBM and TCN seed lists")
    if seeds != REQUIRED_SEEDS:
        raise ValueError(f"this experiment requires seeds {REQUIRED_SEEDS}")
    root = args.output_root.resolve()
    entries = []
    for fold, method, seed in expected_jobs(seeds):
        directory = job_dir(root, fold, method, seed)
        frozen_path = directory / "frozen_validation.json"
        checkpoint = directory / "checkpoints" / "tcn.pt"
        if not (directory / "DONE_TRAIN.json").exists() or not frozen_path.exists() or not checkpoint.exists():
            raise FileNotFoundError(f"training job incomplete: {job_id(fold, method, seed)}")
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen["test_roles_accessed"] is not False:
            raise AssertionError(f"premature test access: {frozen['job_id']}")
        if sha256_file(checkpoint) != frozen["checkpoint_sha256"]:
            raise AssertionError(f"classifier checkpoint changed: {frozen['job_id']}")
        entries.append({
            "job_id": frozen["job_id"], "fold": fold, "method": method, "tcn_seed": seed,
            "nbm_seed": frozen["nbm_seed"],
            "nbm_kind": frozen["nbm_kind"],
            "threshold": frozen["threshold"], "checkpoint_sha256": frozen["checkpoint_sha256"],
            "pair_id": frozen["initialization"]["pair_id"],
            "scaler_sha256": frozen["role4_scaler_artifact"]["scaler_sha256"],
            "nbm_checkpoint_sha256": frozen["role4_scaler_artifact"]["nbm_checkpoint_sha256"],
            "pos_weight": frozen["training"]["pos_weight"],
            "tcn_max_epochs": frozen["training"]["maximum_epochs"],
            "tcn_patience": frozen["training"]["patience"],
        })
    for fold in FOLDS:
        same_fold = [item for item in entries if item["fold"] == fold]
        for key in ("scaler_sha256", "pos_weight"):
            if len({item[key] for item in same_fold}) != 1:
                raise AssertionError(f"fold {fold} mismatch: {key}")
        for seed in seeds:
            paired = [item for item in same_fold if item["tcn_seed"] == seed]
            if len(paired) != 2 or len({item["pair_id"] for item in paired}) != 1:
                raise AssertionError(f"fold {fold}, seed {seed} is not a valid paired initialization")
            if len({item["nbm_seed"] for item in paired}) != 1 or paired[0]["nbm_seed"] != seed:
                raise AssertionError(f"fold {fold}, seed {seed} NBM/TCN seeds are not paired")
            if len({item["nbm_checkpoint_sha256"] for item in paired}) != 1:
                raise AssertionError(f"fold {fold}, seed {seed} methods do not share one NBM")
            if len({item["nbm_kind"] for item in paired}) != 1:
                raise AssertionError(f"fold {fold}, seed {seed} NBM backbone mismatch")
    if len({item["nbm_kind"] for item in entries}) != 1:
        raise AssertionError("one experiment cannot mix NBM backbone kinds")
    if any(item["tcn_max_epochs"] != args.tcn_max_epochs for item in entries):
        raise AssertionError("TCN maximum epoch mismatch")
    if any(item["tcn_patience"] != args.tcn_patience for item in entries):
        raise AssertionError("TCN patience mismatch")
    dataset = DaphnetDataset.load(args.data_dir.resolve())
    records = {record.record_id: record for record in dataset.records}
    rows_by_fold = {fold: load_fold_rows(args.data_dir.resolve(), fold) for fold in FOLDS}
    source_audit = audit_protocol(args.data_dir.resolve(), rows_by_fold, records)
    barrier = {
        "status": "all_FULL_C_and_RAW_classifiers_and_thresholds_frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "folds": list(FOLDS), "methods": list(METHODS),
        "nbm_seeds": list(nbm_seeds), "tcn_seeds": list(seeds),
        "job_count": len(entries),
        "strict_test_gate": "roles 0/1 may be accessed only after this global barrier",
        "source_audit": source_audit,
        "jobs": entries,
    }
    write_json(root / "TRAINING_BARRIER.json", barrier)
    write_json(root / "experiment_config.json", {
        "experiment": f"strict_paired_{entries[0]['nbm_kind']}_NBM_schemeC_vs_centered_scaled_RAW_TCN",
        "full": "role4 scaler + window-axis centering + NBM + scheme C [r,abs(r),delta] [B,27,128]",
        "ablation": "role4 scaler + window-axis centering + RAW [B,9,128]",
        "nbm": "max_epoch=300, patience=20, SmoothL1, lr=1e-3, augmentation=40% clean/40% Gaussian(std=.04)/20% mask",
        "paired_seeds": list(seeds),
        "nbm_kind": entries[0]["nbm_kind"],
        "seed_policy": "exact seeds; no hidden fold offset",
        "tcn": f"max_epoch={args.tcn_max_epochs}, patience={args.tcn_patience}, paired seed/loader order",
        "roles": {str(key): value for key, value in ROLES.items()},
        "threshold": "roles 2/3 balanced accuracy; ties FoG F1 then higher threshold",
        "global_test_barrier_jobs": len(entries),
    })
    print(f"GLOBAL TRAINING BARRIER SEALED jobs={len(entries)}", flush=True)


def sealed_job(args: argparse.Namespace) -> dict[str, Any]:
    barrier_path = args.output_root.resolve() / "TRAINING_BARRIER.json"
    if not barrier_path.exists():
        raise FileNotFoundError("TRAINING_BARRIER.json missing; roles 0/1 access forbidden")
    barrier = json.loads(barrier_path.read_text(encoding="utf-8"))
    target = job_id(args.fold, args.method, args.tcn_seed)
    matches = [item for item in barrier["jobs"] if item["job_id"] == target]
    if len(matches) != 1:
        raise AssertionError(f"job not sealed: {target}")
    if matches[0]["nbm_kind"] != args.nbm_kind:
        raise AssertionError("requested NBM backbone differs from the sealed experiment")
    return matches[0]


def load_history(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["epoch"] = int(row["epoch"])
        for key in ("train_weighted_bce", "validation_weighted_bce", "validation_pr_auc"):
            row[key] = float(row[key])
    return rows


def run_evaluate(args: argparse.Namespace, device: torch.device) -> None:
    require_job_args(args)
    sealed = sealed_job(args)
    directory = job_dir(args.output_root.resolve(), args.fold, args.method, args.tcn_seed)
    done_path = directory / "DONE_TEST.json"
    if done_path.exists() and not args.overwrite:
        print(f"SKIP completed test job: {done_path}", flush=True)
        return
    checkpoint = directory / "checkpoints" / "tcn.pt"
    if sha256_file(checkpoint) != sealed["checkpoint_sha256"]:
        raise AssertionError("sealed TCN checkpoint changed")
    frozen = json.loads((directory / "frozen_validation.json").read_text(encoding="utf-8"))
    if float(frozen["threshold"]) != float(sealed["threshold"]):
        raise AssertionError("sealed threshold changed")

    # The permanent test roles are first requested only after the global barrier.
    records, rows = load_records_rows(args.data_dir, args.fold)
    test_rows = rows.take_role(0, 1)
    scaler, artifact_manifest, _ = load_scaler_only(
        args.nbm_source_root, args.fold, args.nbm_kind
    )
    if artifact_manifest["scaler_sha256"] != sealed["scaler_sha256"]:
        raise AssertionError("sealed role-4 scaler changed")
    test_x, test_feature = make_features(
        args.method, scaler, raw_windows(records, test_rows), test_rows.label,
        device, args.nbm_source_root, args.fold, args.nbm_kind,
    )
    input_channels = 27 if args.method == "FULL_C" else 9
    model = RepresentationTCNM(input_channels).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    test_true, test_prob = classifier_predict(model, test_x, test_rows.label, device)
    threshold = float(sealed["threshold"])
    metrics = binary_metrics(test_true, test_prob, threshold)
    test_pred = (test_prob >= threshold).astype(np.int8)
    by_subject = {}
    for subject in SUBJECTS:
        mask = test_rows.subject_id == subject
        by_subject[subject] = binary_metrics(test_true[mask], test_prob[mask], threshold)
    result = {
        "job_id": sealed["job_id"], "completed_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold, "method": args.method, "tcn_seed": args.tcn_seed,
        "nbm_seed": args.nbm_seed,
        "nbm_kind": args.nbm_kind,
        "threshold": threshold, "threshold_source_roles": [2, 3],
        "strict_global_test_barrier_verified": True, "test_roles": [0, 1],
        "test": metrics, "test_by_subject": by_subject,
        "test_feature": test_feature,
        "test_feature_diagnostics": residual_diagnostics(test_x, test_true),
        "tcn_checkpoint_sha256": sealed["checkpoint_sha256"],
    }
    write_json(directory / "metrics.json", result)
    write_csv(directory / "test_predictions.csv", [{
        "fold": args.fold, "method": args.method, "tcn_seed": args.tcn_seed,
        "subject_id": str(test_rows.subject_id[i]), "record_id": str(test_rows.record_id[i]),
        "window_id": str(test_rows.window_id[i]), "start_index": int(test_rows.start[i]),
        "end_index_exclusive": int(test_rows.end[i]), "role_code": int(test_rows.role[i]),
        "y_true": int(test_true[i]), "fog_probability": float(test_prob[i]),
        "threshold": threshold, "y_pred": int(test_pred[i]),
    } for i in range(len(test_rows))])
    np.savez_compressed(
        directory / "test_probabilities.npz", y_true=test_true, y_prob=test_prob,
        y_pred=test_pred, subject_id=test_rows.subject_id, window_id=test_rows.window_id,
        threshold=np.asarray(threshold),
    )
    history = load_history(directory / "logs" / "tcn_history.csv")
    plot_classifier_training(
        directory, args.method,
        {**frozen["training"], "history": history}, metrics["confusion_matrix"],
    )
    write_json(done_path, {
        "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": sealed["job_id"], "test": metrics,
    })
    print(
        f"TEST COMPLETE {sealed['job_id']} acc={metrics['accuracy']:.6f} "
        f"recall={metrics['sensitivity']:.6f} spec={metrics['specificity']:.6f} "
        f"pr_auc={metrics['auprc']:.6f}", flush=True,
    )


def mean_std(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0)), "n": int(len(array))}


def run_aggregate(args: argparse.Namespace) -> None:
    nbm_seeds = parse_csv_ints(args.nbm_seeds)
    seeds = parse_csv_ints(args.tcn_seeds)
    if nbm_seeds != seeds:
        raise ValueError("strict paired repeats require identical NBM and TCN seed lists")
    if seeds != REQUIRED_SEEDS:
        raise ValueError(f"this experiment requires seeds {REQUIRED_SEEDS}")
    root = args.output_root.resolve()
    if not (root / "TRAINING_BARRIER.json").exists():
        raise FileNotFoundError("cannot aggregate without global barrier")
    results = []
    for fold, method, seed in expected_jobs(seeds):
        directory = job_dir(root, fold, method, seed)
        if not (directory / "DONE_TEST.json").exists():
            raise FileNotFoundError(f"test incomplete: {job_id(fold, method, seed)}")
        results.append(json.loads((directory / "metrics.json").read_text(encoding="utf-8")))
    run_rows = [{
        "fold": r["fold"], "method": r["method"], "nbm_kind": r["nbm_kind"],
        "nbm_seed": r["nbm_seed"], "tcn_seed": r["tcn_seed"],
        "threshold": r["threshold"],
        **{key: r["test"][key] for key in METRIC_KEYS},
        **{key: r["test"][key] for key in ("tn", "fp", "fn", "tp")},
    } for r in results]
    write_csv(root / "run_metrics_18.csv", run_rows)
    seed_rows = []
    for method in METHODS:
        for seed in seeds:
            subset = [r for r in results if r["method"] == method and r["tcn_seed"] == seed]
            seed_rows.append({
                "method": method, "tcn_seed": seed,
                **{key: float(np.mean([r["test"][key] for r in subset])) for key in METRIC_KEYS},
            })
    write_csv(root / "seed_macro_over_3folds.csv", seed_rows)
    summary: dict[str, Any] = {}
    summary_rows = []
    for method in METHODS:
        method_rows = [row for row in seed_rows if row["method"] == method]
        summary[method] = {key: mean_std(row[key] for row in method_rows) for key in METRIC_KEYS}
        for key in METRIC_KEYS:
            summary_rows.append({"method": method, "metric": key, **summary[method][key]})
    write_csv(root / "method_summary_3seed_mean_std.csv", summary_rows)
    deltas = []
    for seed in seeds:
        full = next(row for row in seed_rows if row["method"] == "FULL_C" and row["tcn_seed"] == seed)
        raw = next(row for row in seed_rows if row["method"] == "RAW" and row["tcn_seed"] == seed)
        deltas.append({"tcn_seed": seed, **{key: full[key] - raw[key] for key in METRIC_KEYS}})
    delta_summary = {key: mean_std(row[key] for row in deltas) for key in METRIC_KEYS}
    write_csv(root / "paired_delta_FULL_C_minus_RAW_by_seed.csv", deltas)
    write_csv(root / "paired_delta_FULL_C_minus_RAW_summary.csv", [
        {"metric": key, **value} for key, value in delta_summary.items()
    ])
    subject_rows = []
    subject_json: dict[str, Any] = {method: {} for method in METHODS}
    for method in METHODS:
        for subject in SUBJECTS:
            per_seed = []
            for seed in seeds:
                subset = [r for r in results if r["method"] == method and r["tcn_seed"] == seed]
                per_seed.append({
                    key: float(np.mean([r["test_by_subject"][subject][key] for r in subset]))
                    for key in METRIC_KEYS
                })
            stats = {key: mean_std(item[key] for item in per_seed) for key in METRIC_KEYS}
            subject_json[method][subject] = stats
            subject_rows.append({
                "method": method, "subject_id": subject,
                **{f"{key}_mean": stats[key]["mean"] for key in METRIC_KEYS},
                **{f"{key}_std": stats[key]["std"] for key in METRIC_KEYS},
            })
    write_csv(root / "subject_metrics_3seed_mean_std.csv", subject_rows)
    final = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metrics": summary,
        "paired_delta_FULL_C_minus_RAW": delta_summary,
        "subject_metrics": subject_json,
        "definition": "within each seed macro-average 3 folds; mean±population SD across 3 seeds",
        "strict_global_test_barrier": True,
        "nbm_kind": results[0]["nbm_kind"],
        "run_count": len(results),
    }
    write_json(root / "summary.json", final)
    write_json(root / "DONE.json", {
        "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(results), "methods": list(METHODS),
        "nbm_seeds": list(seeds), "tcn_seeds": list(seeds),
    })
    print(json.dumps(final["primary_metrics"], ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    if args.stage == "seal":
        run_seal(args)
        return
    if args.stage == "aggregate":
        run_aggregate(args)
        return
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    if args.stage == "train":
        run_train(args, device)
    else:
        run_evaluate(args, device)


if __name__ == "__main__":
    main()
