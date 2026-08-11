#!/usr/bin/env python3
"""Strict RAW / GRU-v1-C / GRU-v1.5-C three-arm Daphnet experiment.

The two reconstruction models differ in exactly one architectural variable:
GRU-v1.5 keeps the GRU-v1 encoder and global 16-dimensional bottleneck, while
increasing only the all-zero-input decoder hidden size from 64 to 96.

Roles 4/5 are reserved for Scaler/NBM fit and NBM early stopping/calibration;
roles 6/7 train the classifier; roles 2/3 select its checkpoint and threshold.
Permanent-test roles 0/1 are materialized only after all 45 classifier jobs are
sealed by one strict_test_barrier.v2 manifest.
"""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.evaluation import binary_metrics
from cnbr_fog.resume import atomic_json_dump
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts.run_daphnet_gru_v15_nbm300_fold import (
    ARCHITECTURE_NAME as GRU_V15_ARCHITECTURE_NAME,
    GRUV15Decoder96NBM,
    reconstruct_gru_v15,
)
from scripts.run_daphnet_nbm300_c_vs_raw_ablation import (
    audit_protocol_dynamic,
    barrier_identity_payload,
    build_scheme_c_features,
    build_test_data_manifest,
    load_and_validate_barrier,
    load_history,
    load_records_rows,
    parse_csv_ints,
    raw_features,
    raw_windows_dynamic,
    resolve_device,
    stable_json_hash,
    validate_completed_test_artifacts,
)
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    ROLES,
    SUBJECTS,
    RobustScaler,
    choose_document_threshold,
    classifier_predict,
    load_fold_rows,
    residual_diagnostics,
    write_csv,
)
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    RepresentationTCNM,
    paired_tcn_initial_states,
    plot_classifier_training,
    train_representation_tcn,
)
from scripts.run_daphnet_residual_calibration_abcd import sha256_file
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    GRUReconstructionNBM,
    prepare_nbm_windows,
    reconstruct as reconstruct_gru_v1,
)

FOLDS = (0, 1, 2)
METHODS = ("RAW", "GRU_V1_C", "GRU_V15_C")
REQUIRED_SEEDS = (0, 52, 161, 5216, 52161)
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
SOURCE_GRU_V1 = "gru_v1"
SOURCE_GRU_V15 = "gru_v15_decoder96"
GRU_V1_ARCHITECTURE_NAME = "gru_reconstruction_nbm_v1"
GRU_V1_PARAMETER_COUNT = 31_513
GRU_V15_PARAMETER_COUNT = 48_761
CRITICAL_CODE_PATHS = (
    Path(__file__).resolve(),
    REPO_ROOT / "scripts" / "run_daphnet_gru_nbm300_fold.py",
    REPO_ROOT / "scripts" / "run_daphnet_gru_v15_nbm300_fold.py",
    REPO_ROOT / "scripts" / "run_daphnet_nbm300_c_vs_raw_ablation.py",
    REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py",
    REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_centered_residual_tcn.py",
    REPO_ROOT / "scripts" / "run_daphnet_s01_nonfog_gru_reconstruction_tcnm.py",
    REPO_ROOT / "cnbr_fog" / "evaluation.py",
    REPO_ROOT / "cnbr_fog" / "resume.py",
    REPO_ROOT / "cnbr_fog" / "scientific_fingerprint.py",
)

FEATURE_CONTRACTS = {
    "RAW": {
        "name": "centered_scaled_raw.v1",
        "formula": "RobustScaler(role4); Xc=X-mean_t(X); F=Xc",
        "shape": ["B", 9, 128],
        "uses_nbm": False,
        "subtracts_role5_bias": False,
        "sigma_estimation_centered_by_role5_bias": False,
        "uses_role5_sigma": False,
    },
    "GRU_V1_C": {
        "name": "scheme_c_r_abs_delta.v1",
        "formula": (
            "e=X-Xhat; q=clip(e/(sigma+1e-6),-12,12); "
            "r=q-mean_t(q); F=[r,abs(r),delta_t(r)]"
        ),
        "shape": ["B", 27, 128],
        "uses_nbm": True,
        "subtracts_role5_bias": False,
        "sigma_estimation_centered_by_role5_bias": True,
        "uses_role5_sigma": True,
    },
    "GRU_V15_C": {
        "name": "scheme_c_r_abs_delta.v1",
        "formula": (
            "e=X-Xhat; q=clip(e/(sigma+1e-6),-12,12); "
            "r=q-mean_t(q); F=[r,abs(r),delta_t(r)]"
        ),
        "shape": ["B", 27, 128],
        "uses_nbm": True,
        "subtracts_role5_bias": False,
        "sigma_estimation_centered_by_role5_bias": True,
        "uses_role5_sigma": True,
    },
}


def critical_code_sha256() -> dict[str, str]:
    return {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in CRITICAL_CODE_PATHS
    }


def feature_contract(method: str) -> dict[str, Any]:
    if method not in FEATURE_CONTRACTS:
        raise ValueError(f"unsupported method: {method}")
    contract = dict(FEATURE_CONTRACTS[method])
    return {**contract, "sha256": stable_json_hash(contract)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=("train", "seal", "evaluate", "aggregate")
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT
        / "dataset"
        / "1.Daphnet Freezing of Gait Dataset"
        / "processed_NBM",
    )
    parser.add_argument("--gru-v1-source-root", type=Path, required=True)
    parser.add_argument("--gru-v15-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--nbm-seed", type=int)
    parser.add_argument("--tcn-seed", type=int)
    parser.add_argument("--nbm-seeds", default="0,52,161,5216,52161")
    parser.add_argument("--tcn-seeds", default="0,52,161,5216,52161")
    parser.add_argument("--required-seeds", default="0,52,161,5216,52161")
    parser.add_argument("--sampling-rate-hz", type=int, default=64)
    parser.add_argument("--window-samples", type=int, default=128)
    parser.add_argument("--stride-samples", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tcn-max-epochs", type=int, default=5)
    parser.add_argument("--tcn-patience", type=int, default=2)
    parser.add_argument("--required-nbm-max-epochs", type=int, default=300)
    parser.add_argument("--required-nbm-patience", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_global_contract(args: argparse.Namespace) -> tuple[int, ...]:
    nbm_seeds = parse_csv_ints(args.nbm_seeds)
    tcn_seeds = parse_csv_ints(args.tcn_seeds)
    required = parse_csv_ints(args.required_seeds)
    if nbm_seeds != REQUIRED_SEEDS or tcn_seeds != REQUIRED_SEEDS:
        raise ValueError(f"this experiment requires paired seeds {REQUIRED_SEEDS}")
    if required != REQUIRED_SEEDS:
        raise ValueError(f"--required-seeds must equal {REQUIRED_SEEDS}")
    if args.sampling_rate_hz != 64 or args.window_samples != 128 or args.stride_samples != 64:
        raise ValueError("this experiment is frozen to 64 Hz, window=128, stride=64")
    if args.tcn_max_epochs != 5 or args.tcn_patience != 2:
        raise ValueError("the paired TCN experiment requires max_epoch=5, patience=2")
    if args.required_nbm_max_epochs != 300 or args.required_nbm_patience != 20:
        raise ValueError("both GRU NBMs require max_epoch=300, patience=20")
    return required


def require_job_args(args: argparse.Namespace) -> None:
    if any(
        value is None
        for value in (args.fold, args.method, args.nbm_seed, args.tcn_seed)
    ):
        raise ValueError(
            f"--fold, --method, --nbm-seed and --tcn-seed are required for {args.stage}"
        )
    if args.nbm_seed != args.tcn_seed:
        raise ValueError("strict paired repeats require NBM seed == TCN seed")
    if args.nbm_seed not in REQUIRED_SEEDS:
        raise ValueError(f"seed must be one of {REQUIRED_SEEDS}")


def expected_jobs() -> list[tuple[int, str, int]]:
    return [
        (fold, method, seed)
        for fold in FOLDS
        for method in METHODS
        for seed in REQUIRED_SEEDS
    ]


def job_id(fold: int, method: str, seed: int) -> str:
    return f"fold{fold}_method{method}_seed{seed}"


def job_dir(root: Path, fold: int, method: str, seed: int) -> Path:
    return root / "runs" / f"fold_{fold}" / f"method_{method}" / f"seed_{seed}"


def source_for_method(args: argparse.Namespace, method: str) -> tuple[str, Path, bool]:
    if method == "GRU_V15_C":
        return SOURCE_GRU_V15, args.gru_v15_source_root.resolve(), True
    if method == "GRU_V1_C":
        return SOURCE_GRU_V1, args.gru_v1_source_root.resolve(), True
    if method == "RAW":
        # RAW reads only the deterministic role-4 scaler fields from the v1
        # source.  It never instantiates the NBM or accesses b/sigma.
        return SOURCE_GRU_V1, args.gru_v1_source_root.resolve(), False
    raise ValueError(f"unsupported method: {method}")


def checkpoint_name(source_kind: str) -> str:
    return {
        SOURCE_GRU_V1: "gru_nbm_best.pt",
        SOURCE_GRU_V15: "gru_v15_nbm_best.pt",
    }[source_kind]


def load_source_metadata(
    source_root: Path,
    fold: int,
    source_kind: str,
    seed: int,
    scientific_data_sha256: str,
) -> tuple[RobustScaler, dict[str, Any], dict[str, Any]]:
    fold_dir = source_root / f"fold_{fold}"
    frozen_path = fold_dir / "nbm_frozen.json"
    done_path = fold_dir / "DONE_NBM.json"
    scaler_path = fold_dir / "scaler_role4.json"
    checkpoint = fold_dir / "checkpoints" / checkpoint_name(source_kind)
    for path in (frozen_path, done_path, scaler_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"frozen {source_kind} source missing: {path}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    role4_scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    if done.get("status") != "frozen":
        raise AssertionError(f"{source_kind} DONE_NBM is not frozen")
    if int(done.get("fold", -1)) != fold or int(done.get("seed", -1)) != seed:
        raise AssertionError(f"{source_kind} DONE_NBM fold/seed mismatch")
    if done.get("scientific_data_sha256") != scientific_data_sha256:
        raise AssertionError(f"{source_kind} DONE_NBM scientific dataset changed")
    if frozen.get("scientific_data_sha256") != scientific_data_sha256:
        raise AssertionError(f"{source_kind} frozen scientific dataset changed")
    if (
        int(role4_scaler.get("fold", -1)) != fold
        or int(role4_scaler.get("seed", -1)) != seed
        or int(role4_scaler.get("scaler_fit_role", -1)) != 4
    ):
        raise AssertionError(f"{source_kind} role-4 scaler identity mismatch")
    if role4_scaler.get("scientific_data_sha256") != scientific_data_sha256:
        raise AssertionError(f"{source_kind} role-4 scaler dataset changed")
    scaler_payload = frozen["scaler"]
    if role4_scaler.get("scaler") != scaler_payload:
        raise AssertionError(f"{source_kind} role-4/frozen scaler mismatch")
    checkpoint_sha256 = sha256_file(checkpoint)
    if done.get("checkpoint_sha256") != checkpoint_sha256:
        raise AssertionError(f"{source_kind} DONE_NBM checkpoint hash mismatch")
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    training = frozen["training"]
    if int(checkpoint_payload.get("seed", -1)) != seed:
        raise AssertionError(f"{source_kind} checkpoint seed mismatch")
    if int(checkpoint_payload.get("epoch", -1)) != int(training["best_epoch"]):
        raise AssertionError(f"{source_kind} checkpoint best epoch mismatch")
    if not np.isclose(
        float(checkpoint_payload.get("validation_huber", np.nan)),
        float(training["best_validation_huber"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(f"{source_kind} checkpoint validation loss mismatch")
    if source_kind == SOURCE_GRU_V15 and (
        checkpoint_payload.get("architecture") != training["architecture"]
    ):
        raise AssertionError("GRU-v1.5 checkpoint architecture mismatch")
    scaler = RobustScaler(
        median=np.asarray(scaler_payload["median"], dtype=np.float32),
        iqr=np.asarray(scaler_payload["iqr"], dtype=np.float32),
        epsilon=float(scaler_payload.get("epsilon", 1e-6)),
    )
    artifact = {
        "fold": fold,
        "source_kind": source_kind,
        "scaler_fit_role": int(frozen["scaler_fit_role"]),
        "scaler_unique_raw_points": int(frozen["scaler_unique_raw_points"]),
        "scaler_sha256": stable_json_hash(scaler_payload),
        "scaler_json": str(scaler_path.resolve()),
        "scaler_json_sha256": sha256_file(scaler_path),
        "frozen_json": str(frozen_path.resolve()),
        "frozen_json_sha256": sha256_file(frozen_path),
        "done_nbm_json": str(done_path.resolve()),
        "done_nbm_json_sha256": sha256_file(done_path),
        "nbm_checkpoint": str(checkpoint.resolve()),
        "nbm_checkpoint_sha256": checkpoint_sha256,
        "scientific_data_sha256": scientific_data_sha256,
    }
    return scaler, artifact, frozen


def load_role4_scaler_metadata(
    source_root: Path,
    fold: int,
    seed: int,
    scientific_data_sha256: str,
) -> tuple[RobustScaler, dict[str, Any], dict[str, Any]]:
    """Load RAW scaling without parsing any role-5/NBM calibration artifact."""
    scaler_path = source_root / f"fold_{fold}" / "scaler_role4.json"
    if not scaler_path.is_file():
        raise FileNotFoundError(
            f"role-4-only scaler missing: {scaler_path}; rerun the GRU-v1 NBM job "
            "with the current worker before training RAW"
        )
    payload = json.loads(scaler_path.read_text(encoding="utf-8"))
    if int(payload.get("fold", -1)) != fold:
        raise AssertionError("role-4 scaler fold mismatch")
    if int(payload.get("seed", -1)) != seed:
        raise AssertionError("role-4 scaler seed mismatch")
    if int(payload.get("scaler_fit_role", -1)) != 4:
        raise AssertionError("RAW scaler was not fitted exclusively on role 4")
    if payload.get("scientific_data_sha256") != scientific_data_sha256:
        raise AssertionError("RAW role-4 scaler scientific dataset changed")
    forbidden = {"calibration", "bias", "b", "sigma"}.intersection(payload)
    if forbidden:
        raise AssertionError(
            f"role-4-only scaler unexpectedly contains role-5 fields: {forbidden}"
        )
    scaler_payload = payload["scaler"]
    scaler = RobustScaler(
        median=np.asarray(scaler_payload["median"], dtype=np.float32),
        iqr=np.asarray(scaler_payload["iqr"], dtype=np.float32),
        epsilon=float(scaler_payload.get("epsilon", 1e-6)),
    )
    artifact = {
        "fold": fold,
        "source_kind": SOURCE_GRU_V1,
        "scaler_fit_role": 4,
        "scaler_unique_raw_points": int(payload["scaler_unique_raw_points"]),
        "scaler_sha256": stable_json_hash(scaler_payload),
        "scaler_json": str(scaler_path.resolve()),
        "scaler_json_sha256": sha256_file(scaler_path),
        "frozen_json": None,
        "frozen_json_sha256": None,
        "done_nbm_json": None,
        "done_nbm_json_sha256": None,
        "nbm_checkpoint": None,
        "nbm_checkpoint_sha256": None,
        "scientific_data_sha256": scientific_data_sha256,
    }
    raw_contract = {
        "source_kind": "role4_scaler_only",
        "all_checks_passed": True,
        "architecture": {"name": "none"},
        "seed": seed,
        "fit_role": 4,
        "uses_role5_calibration": False,
        "uses_nbm": False,
    }
    return scaler, artifact, raw_contract


def validate_source_contract(
    frozen: dict[str, Any],
    source_kind: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    training = frozen["training"]
    architecture = training["architecture"]
    augmentation = training["augmentation"]
    expected_augmentation = {
        "clean_probability": 0.40,
        "gaussian_probability": 0.40,
        "mask_probability": 0.20,
        "gaussian_std": 0.04,
        "mask_minimum_samples": 4,
        "mask_maximum_samples": 8,
        "mask_all_channels": True,
    }
    common = {
        "fit_role": int(frozen["nbm_train_role"]) == 4,
        "validation_and_calibration_role": int(
            frozen["nbm_earlystop_and_calibration_role"]
        )
        == 5,
        "restored_best": bool(
            frozen.get("best_checkpoint_restored_before_calibration", False)
        ),
        "validation_clean": frozen.get("validation_mask_or_noise") is False,
        "maximum_epochs": int(training["maximum_epochs"])
        == args.required_nbm_max_epochs,
        "patience": int(training["patience"]) == args.required_nbm_patience,
        "seed": int(training["seed"]) == seed,
        "loss": str(training["loss"]) == "SmoothL1(beta=1.0)",
        "optimizer": "lr=0.001" in str(training["optimizer"]),
        "augmentation": all(
            augmentation.get(key) == value
            for key, value in expected_augmentation.items()
        ),
    }
    if source_kind == SOURCE_GRU_V1:
        architecture_checks = {
            "name": architecture.get("name") == GRU_V1_ARCHITECTURE_NAME,
            "parameter_count": int(architecture.get("parameter_count", -1))
            == GRU_V1_PARAMETER_COUNT,
            "encoder_hidden": architecture.get("encoder_gru", {}).get("hidden_size")
            == 64,
            "bottleneck": architecture.get("latent_shape") == ["B", 16],
            "decoder_hidden": architecture.get("decoder_gru", {}).get("hidden_size")
            == 64,
            "zero_input": architecture.get("decoder_gru", {}).get("input")
            == "all-zero sequence",
            "skip_free": architecture.get("skip_connections") is False,
        }
    elif source_kind == SOURCE_GRU_V15:
        architecture_checks = {
            "name": architecture.get("name") == GRU_V15_ARCHITECTURE_NAME,
            "parameter_count": int(architecture.get("parameter_count", -1))
            == GRU_V15_PARAMETER_COUNT,
            "encoder_hidden": architecture.get("encoder_gru", {}).get("hidden_size")
            == 64,
            "bottleneck": architecture.get("latent_shape") == ["B", 16],
            "decoder_hidden": architecture.get("decoder_gru", {}).get("hidden_size")
            == 96,
            "zero_input": architecture.get("decoder_gru", {}).get("input")
            == "all-zero sequence",
            "skip_free": architecture.get("skip_connections") is False,
            "teacher_forcing": architecture.get("teacher_forcing") is False,
            "time_code": architecture.get("time_code") is False,
        }
    else:
        raise ValueError(f"unsupported source kind: {source_kind}")
    checks = {**common, **architecture_checks}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(
            f"frozen {source_kind} source violates the experiment contract: {failed}"
        )
    return {
        "source_kind": source_kind,
        "all_checks_passed": True,
        "architecture": architecture,
        "seed": seed,
        "maximum_epochs": args.required_nbm_max_epochs,
        "patience": args.required_nbm_patience,
        "checkpoint_rule": "lowest clean role-5 validation SmoothL1",
    }


def load_reconstruction_model(
    source_kind: str,
    artifact: dict[str, Any],
    frozen: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, np.ndarray]:
    checkpoint = Path(artifact["nbm_checkpoint"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if source_kind == SOURCE_GRU_V1:
        model: torch.nn.Module = GRUReconstructionNBM(
            channels=9, hidden=64, bottleneck=16
        ).to(device)
    elif source_kind == SOURCE_GRU_V15:
        model = GRUV15Decoder96NBM().to(device)
        checkpoint_architecture = payload.get("architecture")
        if checkpoint_architecture != frozen["training"]["architecture"]:
            raise AssertionError("GRU-v1.5 checkpoint/frozen architecture mismatch")
    else:
        raise ValueError(f"unsupported source kind: {source_kind}")
    model.load_state_dict(payload["model_state"])
    model.eval()
    calibration = frozen["calibration"]
    sigma = np.asarray(calibration["sigma"], dtype=np.float32)
    bias = np.asarray(calibration["bias"], dtype=np.float32)
    if sigma.shape != (9,) or bias.shape != (9,) or np.any(sigma < 0.05):
        raise AssertionError(f"invalid role-5 calibration for {source_kind}")
    return model, sigma


def make_method_features(
    method: str,
    scaler: RobustScaler,
    artifact: dict[str, Any],
    frozen: dict[str, Any] | None,
    raw: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    window_samples: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if method == "RAW":
        contract = feature_contract(method)
        values = raw_features(scaler, raw, window_samples)
        return values, {
            "formula": contract["formula"],
            "shape": contract["shape"],
            "feature_contract_sha256": contract["sha256"],
            "uses_nbm": False,
            "subtracts_role5_bias": False,
            "sigma_estimation_centered_by_role5_bias": False,
            "uses_role5_sigma": False,
            "uses_residual": False,
            "scaler_source_kind": artifact["source_kind"],
        }
    if frozen is None:
        raise AssertionError("a residual arm requires frozen NBM metadata")
    source_kind = artifact["source_kind"]
    model, sigma = load_reconstruction_model(source_kind, artifact, frozen, device)
    scaled = prepare_nbm_windows(scaler, raw, center=True)
    if source_kind == SOURCE_GRU_V1:
        reconstruction = reconstruct_gru_v1(model, scaled, device)
    elif source_kind == SOURCE_GRU_V15:
        reconstruction = reconstruct_gru_v15(model, scaled, device)
    else:
        raise ValueError(f"unsupported source kind: {source_kind}")
    error_ntc = (scaled - reconstruction).astype(np.float32, copy=False)
    error_bct = np.ascontiguousarray(error_ntc.transpose(0, 2, 1))
    values, clip_stats = build_scheme_c_features(
        error_bct, labels, sigma, window_samples
    )
    del model, reconstruction, error_ntc, error_bct
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return values, {
        "formula": feature_contract(method)["formula"],
        "shape": feature_contract(method)["shape"],
        "feature_contract_sha256": feature_contract(method)["sha256"],
        "uses_nbm": True,
        "subtracts_role5_bias": False,
        "sigma_estimation_centered_by_role5_bias": True,
        "uses_role5_sigma": True,
        "source_kind": source_kind,
        "clip_statistics": clip_stats,
    }


def paired_initialization(
    seed: int, method: str
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    raw_state, residual_state, hashes = paired_tcn_initial_states(seed)
    pair_id = hashlib.sha256(
        f"{seed}:{hashes['r']}:{hashes['r_abs_delta']}".encode("utf-8")
    ).hexdigest()
    selected = raw_state if method == "RAW" else residual_state
    return selected, {
        "seed": seed,
        "pair_id": pair_id,
        "raw_9ch_state_sha256": hashes["r"],
        "residual_27ch_state_sha256": hashes["r_abs_delta"],
        "selected_state_sha256": (
            hashes["r"] if method == "RAW" else hashes["r_abs_delta"]
        ),
        "pairing_rule": (
            "all shape-compatible weights are identical; each 27-channel arm "
            "copies the RAW first 9 channels and zero-initializes channels 10..27"
        ),
    }


def validate_frozen_training_job(
    args: argparse.Namespace,
    directory: Path,
    fold: int,
    method: str,
    seed: int,
    scientific_data_sha256: str,
) -> dict[str, Any]:
    done_path = directory / "DONE_TRAIN.json"
    frozen_path = directory / "frozen_validation.json"
    checkpoint = directory / "checkpoints" / "tcn.pt"
    for path in (done_path, frozen_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete frozen classifier job: {path}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    expected_job_id = job_id(fold, method, seed)
    expected_source = SOURCE_GRU_V15 if method == "GRU_V15_C" else SOURCE_GRU_V1
    expected_uses_nbm = method != "RAW"
    expected_contract = feature_contract(method)
    expected_frozen = {
        "job_id": expected_job_id,
        "fold": fold,
        "method": method,
        "source_kind": expected_source,
        "uses_nbm": expected_uses_nbm,
        "nbm_seed": seed,
        "tcn_seed": seed,
        "test_roles_accessed": False,
        "scientific_data_sha256": scientific_data_sha256,
        "experiment_code_sha256": critical_code_sha256(),
    }
    for key, value in expected_frozen.items():
        if frozen.get(key) != value:
            raise AssertionError(
                f"frozen classifier identity mismatch {expected_job_id}: {key}"
            )
    if frozen.get("feature_contract") != expected_contract:
        raise AssertionError(f"feature contract changed: {expected_job_id}")
    checkpoint_sha256 = sha256_file(checkpoint)
    if frozen.get("checkpoint_sha256") != checkpoint_sha256:
        raise AssertionError(f"TCN checkpoint changed: {expected_job_id}")
    expected_done = {
        "status": "frozen",
        "job_id": expected_job_id,
        "fold": fold,
        "method": method,
        "source_kind": expected_source,
        "uses_nbm": expected_uses_nbm,
        "nbm_seed": seed,
        "tcn_seed": seed,
        "checkpoint_sha256": checkpoint_sha256,
        "frozen_validation_sha256": sha256_file(frozen_path),
        "scientific_data_sha256": scientific_data_sha256,
        "feature_contract_sha256": expected_contract["sha256"],
        "test_roles_accessed": False,
    }
    for key, value in expected_done.items():
        if done.get(key) != value:
            raise AssertionError(
                f"DONE_TRAIN identity mismatch {expected_job_id}: {key}"
            )
    artifact = frozen["role4_scaler_artifact"]
    if method == "RAW":
        source_root = Path(artifact["scaler_json"]).resolve().parent.parent
        _, current_artifact, _ = load_role4_scaler_metadata(
            source_root, fold, seed, scientific_data_sha256
        )
    else:
        source_root = Path(artifact["frozen_json"]).resolve().parent.parent
        _, current_artifact, current_frozen = load_source_metadata(
            source_root,
            fold,
            expected_source,
            seed,
            scientific_data_sha256,
        )
        validate_source_contract(current_frozen, expected_source, seed, args)
    for key in (
        "scaler_sha256",
        "scaler_json_sha256",
        "frozen_json_sha256",
        "done_nbm_json_sha256",
        "nbm_checkpoint_sha256",
        "scientific_data_sha256",
    ):
        if artifact.get(key) != current_artifact.get(key):
            raise AssertionError(
                f"upstream source changed {expected_job_id}: {key}"
            )
    return frozen


def run_train(args: argparse.Namespace, device: torch.device) -> None:
    require_job_args(args)
    directory = job_dir(
        args.output_root.resolve(), args.fold, args.method, args.tcn_seed
    )
    done_path = directory / "DONE_TRAIN.json"
    scientific_data = processed_nbm_scientific_manifest(args.data_dir.resolve())
    if done_path.exists() and not args.overwrite:
        validate_frozen_training_job(
            args,
            directory,
            args.fold,
            args.method,
            args.tcn_seed,
            scientific_data["sha256"],
        )
        print(f"SKIP frozen training job: {done_path}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    records, rows = load_records_rows(args.data_dir, args.fold)
    role67 = rows.take_role(6, 7)
    role23 = rows.take_role(2, 3)
    source_kind, source_root, uses_nbm = source_for_method(args, args.method)
    if args.method == "RAW":
        scaler, artifact, nbm_contract = load_role4_scaler_metadata(
            source_root,
            args.fold,
            args.nbm_seed,
            scientific_data["sha256"],
        )
        frozen_nbm = None
    else:
        scaler, artifact, frozen_nbm = load_source_metadata(
            source_root,
            args.fold,
            source_kind,
            args.nbm_seed,
            scientific_data["sha256"],
        )
        nbm_contract = validate_source_contract(
            frozen_nbm, source_kind, args.nbm_seed, args
        )
    train_x, train_feature = make_method_features(
        args.method,
        scaler,
        artifact,
        frozen_nbm,
        raw_windows_dynamic(records, role67, args.window_samples),
        role67.label,
        device,
        args.window_samples,
    )
    validation_x, validation_feature = make_method_features(
        args.method,
        scaler,
        artifact,
        frozen_nbm,
        raw_windows_dynamic(records, role23, args.window_samples),
        role23.label,
        device,
        args.window_samples,
    )
    initial_state, initialization = paired_initialization(args.tcn_seed, args.method)
    representation = "r" if args.method == "RAW" else "r_abs_delta"
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
    val_true, val_prob = classifier_predict(
        model, validation_x, role23.label, device
    )
    threshold, validation_metrics = choose_document_threshold(val_true, val_prob)
    checkpoint = directory / "checkpoints" / "tcn.pt"
    frozen = {
        "job_id": job_id(args.fold, args.method, args.tcn_seed),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "method": args.method,
        "source_kind": source_kind,
        "uses_nbm": uses_nbm,
        "nbm_seed": args.nbm_seed,
        "tcn_seed": args.tcn_seed,
        "input_shape": train_feature["shape"],
        "feature": train_feature,
        "validation_feature": validation_feature,
        "role4_scaler_artifact": artifact,
        "nbm_contract": nbm_contract,
        "raw_ablation_role5_policy": (
            "RAW uses only role-4 scaler values; it does not instantiate an NBM "
            "or access role-5 b/sigma"
        ),
        "roles": {
            "classifier_train": [6, 7],
            "classifier_validation": [2, 3],
            "test_not_accessed": [0, 1],
        },
        "test_roles_accessed": False,
        "initialization": initialization,
        "training": {key: value for key, value in training.items() if key != "history"},
        "threshold": float(threshold),
        "threshold_source_roles": [2, 3],
        "threshold_rule": (
            "max balanced accuracy; ties FoG F1 then higher threshold; "
            "0.05..0.95 step 0.01"
        ),
        "validation": validation_metrics,
        "feature_diagnostics": {
            "roles_6_7_train": residual_diagnostics(train_x, role67.label),
            "roles_2_3_validation": residual_diagnostics(
                validation_x, role23.label
            ),
        },
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "experiment_code_sha256": critical_code_sha256(),
        "scientific_data_sha256": scientific_data["sha256"],
        "feature_contract": feature_contract(args.method),
    }
    frozen_path = directory / "frozen_validation.json"
    atomic_json_dump(frozen, frozen_path)
    atomic_json_dump(
        {
            "status": "frozen",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": frozen["job_id"],
            "fold": args.fold,
            "method": args.method,
            "source_kind": source_kind,
            "uses_nbm": uses_nbm,
            "nbm_seed": args.nbm_seed,
            "tcn_seed": args.tcn_seed,
            "checkpoint_sha256": frozen["checkpoint_sha256"],
            "frozen_validation_sha256": sha256_file(frozen_path),
            "scientific_data_sha256": scientific_data["sha256"],
            "feature_contract_sha256": frozen["feature_contract"]["sha256"],
            "threshold": threshold,
            "test_roles_accessed": False,
        },
        done_path,
    )
    print(
        f"TRAIN FROZEN {frozen['job_id']} "
        f"best_epoch={training['best_epoch']} threshold={threshold:.2f}",
        flush=True,
    )


def run_seal(args: argparse.Namespace) -> None:
    validate_global_contract(args)
    root = args.output_root.resolve()
    current_code_sha256 = critical_code_sha256()
    scientific_data = processed_nbm_scientific_manifest(args.data_dir.resolve())
    entries: list[dict[str, Any]] = []
    for fold, method, seed in expected_jobs():
        directory = job_dir(root, fold, method, seed)
        frozen_path = directory / "frozen_validation.json"
        frozen = validate_frozen_training_job(
            args,
            directory,
            fold,
            method,
            seed,
            scientific_data["sha256"],
        )
        artifact = frozen["role4_scaler_artifact"]
        entries.append(
            {
                "job_id": frozen["job_id"],
                "fold": fold,
                "method": method,
                "tcn_seed": seed,
                "nbm_seed": frozen["nbm_seed"],
                "source_kind": frozen["source_kind"],
                "uses_nbm": frozen["uses_nbm"],
                "threshold": frozen["threshold"],
                "checkpoint_sha256": frozen["checkpoint_sha256"],
                "frozen_validation_sha256": sha256_file(frozen_path),
                "pair_id": frozen["initialization"]["pair_id"],
                "selected_state_sha256": frozen["initialization"][
                    "selected_state_sha256"
                ],
                "scaler_sha256": artifact["scaler_sha256"],
                "scaler_json_sha256": artifact["scaler_json_sha256"],
                "nbm_frozen_sha256": artifact["frozen_json_sha256"],
                "done_nbm_sha256": artifact["done_nbm_json_sha256"],
                "nbm_checkpoint_sha256": artifact["nbm_checkpoint_sha256"],
                "nbm_architecture_name": frozen["nbm_contract"]["architecture"][
                    "name"
                ],
                "pos_weight": frozen["training"]["pos_weight"],
                "tcn_max_epochs": frozen["training"]["maximum_epochs"],
                "tcn_patience": frozen["training"]["patience"],
                "experiment_code_sha256": frozen["experiment_code_sha256"],
                "scientific_data_sha256": frozen["scientific_data_sha256"],
                "feature_contract_sha256": frozen["feature_contract"]["sha256"],
            }
        )
    for fold in FOLDS:
        fold_entries = [entry for entry in entries if entry["fold"] == fold]
        if len({entry["scaler_sha256"] for entry in fold_entries}) != 1:
            raise AssertionError(f"fold {fold} role-4 scalers differ across arms")
        if len({entry["pos_weight"] for entry in fold_entries}) != 1:
            raise AssertionError(f"fold {fold} pos_weight differs across arms")
        if any(
            entry["scientific_data_sha256"] != scientific_data["sha256"]
            for entry in fold_entries
        ):
            raise AssertionError(f"fold {fold} scientific dataset mismatch")
        for seed in REQUIRED_SEEDS:
            paired = [
                entry for entry in fold_entries if entry["tcn_seed"] == seed
            ]
            if {entry["method"] for entry in paired} != set(METHODS):
                raise AssertionError(f"fold {fold}, seed {seed} lacks three arms")
            if len({entry["pair_id"] for entry in paired}) != 1:
                raise AssertionError(f"fold {fold}, seed {seed} initialization mismatch")
            if any(entry["nbm_seed"] != seed for entry in paired):
                raise AssertionError(f"fold {fold}, seed {seed} NBM seed mismatch")
            by_method = {entry["method"]: entry for entry in paired}
            if by_method["RAW"]["uses_nbm"] is not False:
                raise AssertionError("RAW unexpectedly uses an NBM")
            if by_method["GRU_V1_C"]["uses_nbm"] is not True:
                raise AssertionError("GRU-v1 arm does not use its NBM")
            if by_method["GRU_V15_C"]["uses_nbm"] is not True:
                raise AssertionError("GRU-v1.5 arm does not use its NBM")
            if by_method["RAW"]["nbm_checkpoint_sha256"] is not None:
                raise AssertionError("RAW unexpectedly binds an NBM checkpoint")
            if by_method["RAW"]["nbm_frozen_sha256"] is not None:
                raise AssertionError("RAW unexpectedly binds role-5 calibration")
            if by_method["GRU_V1_C"]["nbm_checkpoint_sha256"] is None:
                raise AssertionError("GRU-v1 arm lacks its NBM checkpoint")
            if by_method["GRU_V15_C"]["nbm_checkpoint_sha256"] is None:
                raise AssertionError("GRU-v1.5 arm lacks its NBM checkpoint")
            if by_method["GRU_V1_C"]["nbm_frozen_sha256"] is None:
                raise AssertionError("GRU-v1 arm lacks frozen role-5 calibration")
            if by_method["GRU_V15_C"]["nbm_frozen_sha256"] is None:
                raise AssertionError("GRU-v1.5 arm lacks frozen role-5 calibration")
            if (
                by_method["GRU_V1_C"]["selected_state_sha256"]
                != by_method["GRU_V15_C"]["selected_state_sha256"]
            ):
                raise AssertionError("the two residual arms do not share TCN initialization")
            if (
                by_method["RAW"]["selected_state_sha256"]
                == by_method["GRU_V1_C"]["selected_state_sha256"]
            ):
                raise AssertionError("RAW and 27-channel state hashes unexpectedly match")
    if any(entry["tcn_max_epochs"] != 5 for entry in entries):
        raise AssertionError("TCN maximum epoch mismatch")
    if any(entry["tcn_patience"] != 2 for entry in entries):
        raise AssertionError("TCN patience mismatch")
    rows_by_fold = {
        fold: load_fold_rows(args.data_dir.resolve(), fold) for fold in FOLDS
    }
    source_audit = audit_protocol_dynamic(
        args.data_dir.resolve(), rows_by_fold, 64, 128, 64
    )
    source_audit["experiment_code_sha256"] = current_code_sha256
    source_audit["scientific_data_manifest"] = scientific_data
    test_manifest = build_test_data_manifest(args.data_dir.resolve(), rows_by_fold)
    barrier = {
        "barrier_schema": "strict_test_barrier.v2",
        "status": "all_RAW_GRU_V1_C_GRU_V15_C_classifiers_and_thresholds_frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "folds": list(FOLDS),
        "methods": list(METHODS),
        "nbm_seeds": list(REQUIRED_SEEDS),
        "tcn_seeds": list(REQUIRED_SEEDS),
        "job_count": len(entries),
        "strict_test_gate": "roles 0/1 may be accessed only after this global barrier",
        "source_audit": source_audit,
        "test_data_manifest": test_manifest,
        "jobs": entries,
    }
    barrier["barrier_id"] = stable_json_hash(barrier_identity_payload(barrier))
    atomic_json_dump(barrier, root / "TRAINING_BARRIER.json")
    atomic_json_dump(
        {
            "experiment": "strict_three_arm_GRU_v1_vs_GRU_v15_decoder96_vs_RAW",
            "methods": list(METHODS),
            "raw": "role4 RobustScaler + window-axis centering [B,9,128]",
            "gru_v1": "GRU-v1 + scheme C [r,abs(r),delta] [B,27,128]",
            "gru_v15": (
                "GRU-v1 encoder H64/global z16; zero-input decoder H96 + "
                "scheme C [B,27,128]"
            ),
            "single_architecture_variable": "decoder hidden 64 -> 96",
            "nbm_training": (
                "max300/pat20, SmoothL1, AdamW lr1e-3, "
                "40% clean/40% Gaussian/20% mask"
            ),
            "tcn_training": "max5/pat2, weighted BCE, AdamW lr1e-3",
            "paired_seeds": list(REQUIRED_SEEDS),
            "sampling_rate_hz": 64,
            "window_samples": 128,
            "stride_samples": 64,
            "roles": {str(key): value for key, value in ROLES.items()},
            "threshold": (
                "roles 2/3 max balanced accuracy; ties FoG F1 then higher threshold"
            ),
            "barrier_schema": barrier["barrier_schema"],
            "barrier_id": barrier["barrier_id"],
            "test_data_manifest_sha256": test_manifest["sha256"],
            "adaptive_benchmark_notice": (
                "roles 0/1 have been inspected in prior architecture studies; "
                "use an external holdout for confirmatory claims"
            ),
            "pre_registered_success": {
                "comparison": "GRU_V15_C minus GRU_V1_C",
                "sensitivity_mean_delta_min": 0.010,
                "auprc_mean_delta_min": -0.005,
                "specificity_mean_delta_min": -0.010,
                "precision_mean_delta_min": -0.010,
                "sensitivity_positive_seed_count_min": 4,
            },
        },
        root / "experiment_config.json",
    )
    print(f"GLOBAL TRAINING BARRIER SEALED jobs={len(entries)}", flush=True)


def sealed_job(args: argparse.Namespace) -> dict[str, Any]:
    barrier_path = args.output_root.resolve() / "TRAINING_BARRIER.json"
    if not barrier_path.is_file():
        raise FileNotFoundError("TRAINING_BARRIER.json missing; roles 0/1 forbidden")
    barrier = load_and_validate_barrier(barrier_path)
    if barrier.get("barrier_schema") != "strict_test_barrier.v2":
        raise RuntimeError("the three-arm experiment requires strict_test_barrier.v2")
    target = job_id(args.fold, args.method, args.tcn_seed)
    matches = [entry for entry in barrier["jobs"] if entry["job_id"] == target]
    if len(matches) != 1:
        raise AssertionError(f"job is not uniquely sealed: {target}")
    sealed = dict(matches[0])
    expected_source, _, expected_uses_nbm = source_for_method(args, args.method)
    if sealed["source_kind"] != expected_source:
        raise AssertionError("requested feature source differs from sealed source")
    if sealed["uses_nbm"] is not expected_uses_nbm:
        raise AssertionError("requested NBM usage differs from sealed method")
    sealed["barrier_schema"] = barrier["barrier_schema"]
    sealed["barrier_id"] = barrier["barrier_id"]
    sealed["test_data_manifest_sha256"] = barrier["test_data_manifest"]["sha256"]
    return sealed


def run_evaluate(args: argparse.Namespace, device: torch.device) -> None:
    require_job_args(args)
    sealed = sealed_job(args)
    if sealed.get("experiment_code_sha256") != critical_code_sha256():
        raise AssertionError("experiment code changed after the global seal")
    scientific_data = processed_nbm_scientific_manifest(args.data_dir.resolve())
    if sealed.get("scientific_data_sha256") != scientific_data["sha256"]:
        raise AssertionError("scientific training/test dataset changed after seal")
    if sealed.get("feature_contract_sha256") != feature_contract(args.method)["sha256"]:
        raise AssertionError("feature representation contract changed after seal")
    directory = job_dir(
        args.output_root.resolve(), args.fold, args.method, args.tcn_seed
    )
    done_path = directory / "DONE_TEST.json"
    rows_by_fold = {
        fold: load_fold_rows(args.data_dir.resolve(), fold) for fold in FOLDS
    }
    current_manifest = build_test_data_manifest(args.data_dir.resolve(), rows_by_fold)
    if current_manifest["sha256"] != sealed["test_data_manifest_sha256"]:
        raise AssertionError("permanent-test data changed after the global seal")
    validate_frozen_training_job(
        args,
        directory,
        args.fold,
        args.method,
        args.tcn_seed,
        scientific_data["sha256"],
    )
    if done_path.exists() and not args.overwrite:
        validate_completed_test_artifacts(directory, sealed)
        print(f"SKIP completed test job: {done_path}", flush=True)
        return
    checkpoint = directory / "checkpoints" / "tcn.pt"
    if sha256_file(checkpoint) != sealed["checkpoint_sha256"]:
        raise AssertionError("sealed TCN checkpoint changed")
    frozen_path = directory / "frozen_validation.json"
    if sha256_file(frozen_path) != sealed["frozen_validation_sha256"]:
        raise AssertionError("sealed classifier validation artifact changed")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if float(frozen["threshold"]) != float(sealed["threshold"]):
        raise AssertionError("sealed threshold changed")
    records, rows = load_records_rows(args.data_dir, args.fold)
    test_rows = rows.take_role(0, 1)
    source_kind, source_root, _ = source_for_method(args, args.method)
    if args.method == "RAW":
        scaler, artifact, _ = load_role4_scaler_metadata(
            source_root,
            args.fold,
            args.nbm_seed,
            scientific_data["sha256"],
        )
        frozen_nbm = None
    else:
        scaler, artifact, frozen_nbm = load_source_metadata(
            source_root,
            args.fold,
            source_kind,
            args.nbm_seed,
            scientific_data["sha256"],
        )
    if artifact["scaler_sha256"] != sealed["scaler_sha256"]:
        raise AssertionError("sealed role-4 scaler changed")
    if artifact["nbm_checkpoint_sha256"] != sealed["nbm_checkpoint_sha256"]:
        raise AssertionError("sealed NBM checkpoint changed")
    if artifact["frozen_json_sha256"] != sealed["nbm_frozen_sha256"]:
        raise AssertionError("sealed role-5 calibration changed")
    if artifact["done_nbm_json_sha256"] != sealed["done_nbm_sha256"]:
        raise AssertionError("sealed NBM completion identity changed")
    test_x, test_feature = make_method_features(
        args.method,
        scaler,
        artifact,
        frozen_nbm,
        raw_windows_dynamic(records, test_rows, args.window_samples),
        test_rows.label,
        device,
        args.window_samples,
    )
    input_channels = 9 if args.method == "RAW" else 27
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
        by_subject[subject] = binary_metrics(
            test_true[mask], test_prob[mask], threshold
        )
    result = {
        "job_id": sealed["job_id"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "method": args.method,
        "source_kind": source_kind,
        "uses_nbm": sealed["uses_nbm"],
        "tcn_seed": args.tcn_seed,
        "nbm_seed": args.nbm_seed,
        "threshold": threshold,
        "threshold_source_roles": [2, 3],
        "strict_global_test_barrier_verified": True,
        "test_roles": [0, 1],
        "barrier_schema": sealed["barrier_schema"],
        "barrier_id": sealed["barrier_id"],
        "test_data_manifest_sha256": sealed["test_data_manifest_sha256"],
        "test": metrics,
        "test_by_subject": by_subject,
        "test_feature": test_feature,
        "test_feature_diagnostics": residual_diagnostics(test_x, test_true),
        "tcn_checkpoint_sha256": sealed["checkpoint_sha256"],
        "nbm_checkpoint_sha256": sealed["nbm_checkpoint_sha256"],
        "nbm_frozen_sha256": sealed["nbm_frozen_sha256"],
        "done_nbm_sha256": sealed["done_nbm_sha256"],
        "scaler_sha256": sealed["scaler_sha256"],
        "frozen_validation_sha256": sealed["frozen_validation_sha256"],
        "scientific_data_sha256": sealed["scientific_data_sha256"],
        "feature_contract_sha256": sealed["feature_contract_sha256"],
    }
    atomic_json_dump(result, directory / "metrics.json")
    write_csv(
        directory / "test_predictions.csv",
        [
            {
                "fold": args.fold,
                "method": args.method,
                "tcn_seed": args.tcn_seed,
                "subject_id": str(test_rows.subject_id[index]),
                "record_id": str(test_rows.record_id[index]),
                "window_id": str(test_rows.window_id[index]),
                "start_index": int(test_rows.start[index]),
                "end_index_exclusive": int(test_rows.end[index]),
                "role_code": int(test_rows.role[index]),
                "y_true": int(test_true[index]),
                "fog_probability": float(test_prob[index]),
                "threshold": threshold,
                "y_pred": int(test_pred[index]),
            }
            for index in range(len(test_rows))
        ],
    )
    np.savez_compressed(
        directory / "test_probabilities.npz",
        y_true=test_true,
        y_prob=test_prob,
        y_pred=test_pred,
        subject_id=test_rows.subject_id,
        window_id=test_rows.window_id,
        threshold=np.asarray(threshold),
    )
    history = load_history(directory / "logs" / "tcn_history.csv")
    plot_classifier_training(
        directory,
        args.method,
        {**frozen["training"], "history": history},
        metrics["confusion_matrix"],
    )
    metrics_path = directory / "metrics.json"
    predictions_path = directory / "test_predictions.csv"
    probabilities_path = directory / "test_probabilities.npz"
    atomic_json_dump(
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": sealed["job_id"],
            "test": metrics,
            "barrier_id": sealed["barrier_id"],
            "test_data_manifest_sha256": sealed["test_data_manifest_sha256"],
            "tcn_checkpoint_sha256": sealed["checkpoint_sha256"],
            "nbm_checkpoint_sha256": sealed["nbm_checkpoint_sha256"],
            "nbm_frozen_sha256": sealed["nbm_frozen_sha256"],
            "done_nbm_sha256": sealed["done_nbm_sha256"],
            "scaler_sha256": sealed["scaler_sha256"],
            "frozen_validation_sha256": sealed["frozen_validation_sha256"],
            "scientific_data_sha256": sealed["scientific_data_sha256"],
            "feature_contract_sha256": sealed["feature_contract_sha256"],
            "metrics_sha256": sha256_file(metrics_path),
            "predictions_sha256": sha256_file(predictions_path),
            "probabilities_sha256": sha256_file(probabilities_path),
        },
        done_path,
    )
    print(
        f"TEST COMPLETE {sealed['job_id']} "
        f"sens={metrics['sensitivity']:.6f} precision={metrics['precision']:.6f} "
        f"spec={metrics['specificity']:.6f} pr_auc={metrics['auprc']:.6f}",
        flush=True,
    )


def mean_std(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "n": int(len(array)),
    }


def paired_deltas(
    seed_rows: list[dict[str, Any]],
    left: str,
    right: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for seed in REQUIRED_SEEDS:
        left_row = next(
            row
            for row in seed_rows
            if row["method"] == left and row["tcn_seed"] == seed
        )
        right_row = next(
            row
            for row in seed_rows
            if row["method"] == right and row["tcn_seed"] == seed
        )
        rows.append(
            {
                "tcn_seed": seed,
                **{key: left_row[key] - right_row[key] for key in METRIC_KEYS},
            }
        )
    summary = {key: mean_std(row[key] for row in rows) for key in METRIC_KEYS}
    return rows, summary


def run_aggregate(args: argparse.Namespace) -> None:
    validate_global_contract(args)
    root = args.output_root.resolve()
    barrier_path = root / "TRAINING_BARRIER.json"
    if not barrier_path.is_file():
        raise FileNotFoundError("cannot aggregate without TRAINING_BARRIER.json")
    barrier = load_and_validate_barrier(barrier_path)
    if barrier.get("barrier_schema") != "strict_test_barrier.v2":
        raise RuntimeError("the three-arm experiment requires strict_test_barrier.v2")
    if (
        barrier.get("source_audit", {}).get("experiment_code_sha256")
        != critical_code_sha256()
    ):
        raise AssertionError("experiment code changed after the global seal")
    scientific_data = processed_nbm_scientific_manifest(args.data_dir.resolve())
    sealed_scientific = barrier.get("source_audit", {}).get(
        "scientific_data_manifest", {}
    )
    if sealed_scientific.get("sha256") != scientific_data["sha256"]:
        raise AssertionError("scientific dataset changed after the global seal")
    current_manifest = build_test_data_manifest(
        args.data_dir.resolve(),
        {fold: load_fold_rows(args.data_dir.resolve(), fold) for fold in FOLDS},
    )
    if current_manifest["sha256"] != barrier["test_data_manifest"]["sha256"]:
        raise AssertionError("permanent-test data changed after the global seal")
    barrier_jobs = {entry["job_id"]: entry for entry in barrier["jobs"]}
    results = []
    for fold, method, seed in expected_jobs():
        target = job_id(fold, method, seed)
        if target not in barrier_jobs:
            raise AssertionError(f"job absent from barrier: {target}")
        directory = job_dir(root, fold, method, seed)
        validate_frozen_training_job(
            args,
            directory,
            fold,
            method,
            seed,
            scientific_data["sha256"],
        )
        sealed = {
            **barrier_jobs[target],
            "barrier_schema": barrier["barrier_schema"],
            "barrier_id": barrier["barrier_id"],
            "test_data_manifest_sha256": barrier["test_data_manifest"]["sha256"],
        }
        result = validate_completed_test_artifacts(directory, sealed)
        if (
            int(result["fold"]) != fold
            or result["method"] != method
            or int(result["tcn_seed"]) != seed
            or result["source_kind"] != sealed["source_kind"]
        ):
            raise AssertionError(f"test identity mismatch: {target}")
        results.append(result)
    run_rows = [
        {
            "fold": result["fold"],
            "method": result["method"],
            "source_kind": result["source_kind"],
            "nbm_seed": result["nbm_seed"],
            "tcn_seed": result["tcn_seed"],
            "threshold": result["threshold"],
            **{key: result["test"][key] for key in METRIC_KEYS},
            **{key: result["test"][key] for key in ("tn", "fp", "fn", "tp")},
        }
        for result in results
    ]
    write_csv(root / "run_metrics_45.csv", run_rows)
    seed_rows = []
    for method in METHODS:
        for seed in REQUIRED_SEEDS:
            subset = [
                result
                for result in results
                if result["method"] == method and result["tcn_seed"] == seed
            ]
            seed_rows.append(
                {
                    "method": method,
                    "tcn_seed": seed,
                    **{
                        key: float(
                            np.mean([result["test"][key] for result in subset])
                        )
                        for key in METRIC_KEYS
                    },
                }
            )
    write_csv(root / "seed_macro_over_3folds.csv", seed_rows)
    primary: dict[str, Any] = {}
    summary_rows = []
    for method in METHODS:
        method_rows = [row for row in seed_rows if row["method"] == method]
        primary[method] = {
            key: mean_std(row[key] for row in method_rows) for key in METRIC_KEYS
        }
        for key in METRIC_KEYS:
            summary_rows.append(
                {"method": method, "metric": key, **primary[method][key]}
            )
    write_csv(root / "method_summary_5seed_mean_std.csv", summary_rows)
    comparisons = {
        "GRU_V1_C_minus_RAW": ("GRU_V1_C", "RAW"),
        "GRU_V15_C_minus_RAW": ("GRU_V15_C", "RAW"),
        "GRU_V15_C_minus_GRU_V1_C": ("GRU_V15_C", "GRU_V1_C"),
    }
    delta_summaries = {}
    delta_rows_by_comparison = {}
    for name, (left, right) in comparisons.items():
        rows, summary = paired_deltas(seed_rows, left, right)
        delta_rows_by_comparison[name] = rows
        delta_summaries[name] = summary
        write_csv(root / f"paired_delta_{name}_by_seed.csv", rows)
        write_csv(
            root / f"paired_delta_{name}_summary.csv",
            [{"metric": key, **value} for key, value in summary.items()],
        )
    subject_rows = []
    subject_json: dict[str, Any] = {method: {} for method in METHODS}
    for method in METHODS:
        for subject in SUBJECTS:
            per_seed = []
            for seed in REQUIRED_SEEDS:
                subset = [
                    result
                    for result in results
                    if result["method"] == method and result["tcn_seed"] == seed
                ]
                per_seed.append(
                    {
                        key: float(
                            np.mean(
                                [
                                    result["test_by_subject"][subject][key]
                                    for result in subset
                                ]
                            )
                        )
                        for key in METRIC_KEYS
                    }
                )
            stats = {
                key: mean_std(item[key] for item in per_seed)
                for key in METRIC_KEYS
            }
            subject_json[method][subject] = stats
            subject_rows.append(
                {
                    "method": method,
                    "subject_id": subject,
                    **{
                        f"{key}_mean": stats[key]["mean"] for key in METRIC_KEYS
                    },
                    **{f"{key}_std": stats[key]["std"] for key in METRIC_KEYS},
                }
            )
    write_csv(root / "subject_metrics_5seed_mean_std.csv", subject_rows)
    v15_vs_v1 = delta_summaries["GRU_V15_C_minus_GRU_V1_C"]
    v15_vs_v1_rows = delta_rows_by_comparison[
        "GRU_V15_C_minus_GRU_V1_C"
    ]
    success_checks = {
        "sensitivity_mean_delta_at_least_0.010": v15_vs_v1["sensitivity"]["mean"]
        >= 0.010,
        "auprc_mean_delta_at_least_minus_0.005": v15_vs_v1["auprc"]["mean"]
        >= -0.005,
        "specificity_mean_delta_at_least_minus_0.010": v15_vs_v1["specificity"][
            "mean"
        ]
        >= -0.010,
        "precision_mean_delta_at_least_minus_0.010": v15_vs_v1["precision"][
            "mean"
        ]
        >= -0.010,
        "sensitivity_positive_in_at_least_4_of_5_seeds": sum(
            row["sensitivity"] > 0 for row in v15_vs_v1_rows
        )
        >= 4,
    }
    final = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metrics": primary,
        "paired_deltas": delta_summaries,
        "pre_registered_success": {
            "checks": success_checks,
            "all_passed": all(success_checks.values()),
        },
        "subject_metrics": subject_json,
        "definition": (
            "within each seed macro-average 3 folds; mean +/- population SD "
            "across 5 paired seeds"
        ),
        "strict_global_test_barrier": True,
        "barrier_schema": barrier["barrier_schema"],
        "barrier_id": barrier["barrier_id"],
        "test_data_manifest_sha256": barrier["test_data_manifest"]["sha256"],
        "run_count": len(results),
        "methods": list(METHODS),
    }
    atomic_json_dump(final, root / "summary.json")
    atomic_json_dump(
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "run_count": len(results),
            "methods": list(METHODS),
            "nbm_seeds": list(REQUIRED_SEEDS),
            "tcn_seeds": list(REQUIRED_SEEDS),
        },
        root / "DONE.json",
    )
    print(json.dumps(primary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    validate_global_contract(args)
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    if args.stage == "seal":
        run_seal(args)
        return
    if args.stage == "aggregate":
        run_aggregate(args)
        return
    require_job_args(args)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    if args.stage == "train":
        run_train(args, device)
    elif args.stage == "evaluate":
        run_evaluate(args, device)
    else:
        raise ValueError(f"unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
