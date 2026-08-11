#!/usr/bin/env python3
"""Strict RAW / GRU baseline-mask / stronger-local-mask three-arm experiment.

This classifier worker consumes two independently frozen GRU-v1 NBM sources
created by ``run_daphnet_gru_mask_strength_nbm300_fold.py``:

* GRU_BASE_C: 40/40/20 clean/Gaussian/mask, local mask length 4..8 samples;
* GRU_MASK8_12_C: identical training, local mask length 8..12 samples.

Roles 4/5 are reserved for Scaler/NBM fit and NBM early stopping/calibration;
roles 6/7 train the classifier; roles 2/3 select its checkpoint and threshold.
Permanent-test roles 0/1 are materialized only after all 45 classifier jobs are
sealed by one strict-test barrier.  The adapter deliberately reuses the mature
three-arm implementation without modifying the previous v1/v1.5 experiment.
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

import scripts.run_daphnet_gru_v15_three_arm as base
from cnbr_fog.resume import atomic_json_dump, canonical_fingerprint
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    ROLES,
    SUBJECTS,
    RobustScaler,
    load_fold_rows,
    write_csv,
)
from scripts.run_daphnet_residual_calibration_abcd import sha256_file
from scripts.run_daphnet_gru_mask_strength_nbm300_fold import (
    VARIANTS as NBM_VARIANTS,
    architecture_config as nbm_architecture_config,
    checkpoint_name as nbm_checkpoint_name,
    protocol_contract as nbm_protocol_contract,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    GRUReconstructionNBM,
    prepare_nbm_windows,
    reconstruct as reconstruct_gru,
)

FOLDS = (0, 1, 2)
METHODS = ("RAW", "GRU_BASE_C", "GRU_MASK8_12_C")
REQUIRED_SEEDS = (0, 52, 161, 5216, 52161)
METRIC_KEYS = base.METRIC_KEYS
SOURCE_GRU_BASE = "gru_v1_mask4_8"
SOURCE_GRU_MASK8_12 = "gru_v1_mask8_12"
GRU_ARCHITECTURE_NAME = "gru_reconstruction_nbm_v1"
GRU_PARAMETER_COUNT = 31_513
PARAMETERIZED_NBM_WORKER = (
    REPO_ROOT / "scripts" / "run_daphnet_gru_mask_strength_nbm300_fold.py"
)

CRITICAL_CODE_PATHS = (
    Path(__file__).resolve(),
    PARAMETERIZED_NBM_WORKER,
    REPO_ROOT / "scripts" / "run_daphnet_gru_v15_three_arm.py",
    REPO_ROOT / "scripts" / "run_daphnet_nbm300_c_vs_raw_ablation.py",
    REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py",
    REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_centered_residual_tcn.py",
    REPO_ROOT / "scripts" / "run_daphnet_s01_nonfog_gru_reconstruction_tcnm.py",
    REPO_ROOT / "cnbr_fog" / "data.py",
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
    "GRU_BASE_C": {
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
    "GRU_MASK8_12_C": {
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

_ORIGINAL_LOAD_SOURCE_METADATA = base.load_source_metadata
_ORIGINAL_LOAD_ROLE4_SCALER_METADATA = base.load_role4_scaler_metadata
_ORIGINAL_PAIRED_INITIALIZATION = base.paired_initialization
_ORIGINAL_SEALED_JOB = base.sealed_job


def critical_code_sha256() -> dict[str, str]:
    missing = [str(path) for path in CRITICAL_CODE_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "critical experiment code is incomplete: " + ", ".join(missing)
        )
    return {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in CRITICAL_CODE_PATHS
    }


def feature_contract(method: str) -> dict[str, Any]:
    if method not in FEATURE_CONTRACTS:
        raise ValueError(f"unsupported method: {method}")
    contract = dict(FEATURE_CONTRACTS[method])
    return {**contract, "sha256": base.stable_json_hash(contract)}


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
    parser.add_argument("--gru-base-source-root", type=Path, required=True)
    parser.add_argument("--gru-mask8-12-source-root", type=Path, required=True)
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
    nbm_seeds = base.parse_csv_ints(args.nbm_seeds)
    tcn_seeds = base.parse_csv_ints(args.tcn_seeds)
    required = base.parse_csv_ints(args.required_seeds)
    if nbm_seeds != REQUIRED_SEEDS or tcn_seeds != REQUIRED_SEEDS:
        raise ValueError(f"this experiment requires paired seeds {REQUIRED_SEEDS}")
    if required != REQUIRED_SEEDS:
        raise ValueError(f"--required-seeds must equal {REQUIRED_SEEDS}")
    if (args.sampling_rate_hz, args.window_samples, args.stride_samples) != (
        64,
        128,
        64,
    ):
        raise ValueError("this experiment is frozen to 64 Hz, window=128, stride=64")
    if (args.tcn_max_epochs, args.tcn_patience) != (5, 2):
        raise ValueError("the paired TCN experiment requires max_epoch=5, patience=2")
    if (args.required_nbm_max_epochs, args.required_nbm_patience) != (300, 20):
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
    if method == "GRU_BASE_C":
        return SOURCE_GRU_BASE, args.gru_base_source_root.resolve(), True
    if method == "GRU_MASK8_12_C":
        return SOURCE_GRU_MASK8_12, args.gru_mask8_12_source_root.resolve(), True
    if method == "RAW":
        # RAW uses only the baseline source's role-4 scaler artifact.
        return SOURCE_GRU_BASE, args.gru_base_source_root.resolve(), False
    raise ValueError(f"unsupported method: {method}")


def checkpoint_name(source_kind: str) -> str:
    variants = {
        SOURCE_GRU_BASE: "BASE",
        SOURCE_GRU_MASK8_12: "MASK8_12",
    }
    if source_kind not in variants:
        raise ValueError(f"unsupported source kind: {source_kind}")
    return nbm_checkpoint_name(variants[source_kind])


def _source_hashes(artifact: dict[str, Any], frozen: dict[str, Any]) -> dict[str, str]:
    sigma_hash = base.stable_json_hash(
        {"sigma": [float(value) for value in frozen["calibration"]["sigma"]]}
    )
    source_bundle = {
        "scaler_json_sha256": artifact["scaler_json_sha256"],
        "config_json_sha256": artifact["config_json_sha256"],
        "frozen_json_sha256": artifact["frozen_json_sha256"],
        "done_nbm_json_sha256": artifact["done_nbm_json_sha256"],
        "nbm_checkpoint_sha256": artifact["nbm_checkpoint_sha256"],
        "calibration_sigma_sha256": sigma_hash,
        "initial_model_state_sha256": artifact["initial_model_state_sha256"],
        "scientific_data_sha256": artifact["scientific_data_sha256"],
    }
    return {
        "calibration_sigma_sha256": sigma_hash,
        "source_bundle_sha256": base.stable_json_hash(source_bundle),
    }


def load_source_metadata(
    source_root: Path,
    fold: int,
    source_kind: str,
    seed: int,
    scientific_data_sha256: str,
) -> tuple[RobustScaler, dict[str, Any], dict[str, Any]]:
    # The reused loader resolves its checkpoint-name helper in the base module.
    # Temporarily bind the local variant-aware mapping so this public function
    # is also safe when called directly by validation tests before main().
    previous_checkpoint_name = base.checkpoint_name
    base.checkpoint_name = checkpoint_name
    try:
        scaler, artifact, frozen = _ORIGINAL_LOAD_SOURCE_METADATA(
            source_root, fold, source_kind, seed, scientific_data_sha256
        )
    finally:
        base.checkpoint_name = previous_checkpoint_name
    variants = {
        SOURCE_GRU_BASE: "BASE",
        SOURCE_GRU_MASK8_12: "MASK8_12",
    }
    variant = variants[source_kind]
    if variant not in NBM_VARIANTS or frozen.get("variant") != variant:
        raise AssertionError(f"{source_kind} frozen variant identity mismatch")
    config_path = source_root / f"fold_{fold}" / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"parameterized NBM config missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_protocol = nbm_protocol_contract(variant, scientific_data_sha256)
    if config.get("protocol") != expected_protocol:
        raise AssertionError(f"{source_kind} protocol config changed")
    expected_protocol_sha = canonical_fingerprint(expected_protocol)
    if config.get("fold") != fold or config.get("seed") != seed:
        raise AssertionError(f"{source_kind} config fold/seed mismatch")
    done_path = source_root / f"fold_{fold}" / "DONE_NBM.json"
    checkpoint_path = Path(artifact["nbm_checkpoint"])
    done = json.loads(done_path.read_text(encoding="utf-8"))
    scaler_payload = json.loads(
        Path(artifact["scaler_json"]).read_text(encoding="utf-8")
    )
    checkpoint_payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    expected_initial = frozen["training"].get("initial_model_state_sha256")
    if not expected_initial:
        raise AssertionError(f"{source_kind} lacks initial model-state identity")
    if done.get("variant") != variant or checkpoint_payload.get("variant") != variant:
        raise AssertionError(f"{source_kind} DONE/checkpoint variant mismatch")
    if any(
        value != expected_protocol_sha
        for value in (
            config.get("protocol_sha256"),
            scaler_payload.get("protocol_sha256"),
            frozen.get("protocol_sha256"),
            done.get("protocol_sha256"),
        )
    ):
        raise AssertionError(f"{source_kind} protocol hash mismatch")
    if any(
        value != expected_initial
        for value in (
            done.get("initial_model_state_sha256"),
            checkpoint_payload.get("initial_model_state_sha256"),
        )
    ):
        raise AssertionError(f"{source_kind} initial model-state hash mismatch")
    if checkpoint_payload.get("architecture") != nbm_architecture_config():
        raise AssertionError(f"{source_kind} checkpoint architecture changed")
    frozen["_verified_source_config"] = config
    artifact["config_json"] = str(config_path.resolve())
    artifact["config_json_sha256"] = sha256_file(config_path)
    artifact["protocol_sha256"] = expected_protocol_sha
    artifact["initial_model_state_sha256"] = str(expected_initial)
    artifact.update(_source_hashes(artifact, frozen))
    return scaler, artifact, frozen


def load_role4_scaler_metadata(
    source_root: Path,
    fold: int,
    seed: int,
    scientific_data_sha256: str,
) -> tuple[RobustScaler, dict[str, Any], dict[str, Any]]:
    scaler, artifact, contract = _ORIGINAL_LOAD_ROLE4_SCALER_METADATA(
        source_root, fold, seed, scientific_data_sha256
    )
    artifact["source_kind"] = SOURCE_GRU_BASE
    artifact["calibration_sigma_sha256"] = None
    artifact["source_bundle_sha256"] = base.stable_json_hash(
        {
            "scaler_json_sha256": artifact["scaler_json_sha256"],
            "scientific_data_sha256": scientific_data_sha256,
            "uses_nbm": False,
        }
    )
    return scaler, artifact, contract


def validate_source_contract(
    frozen: dict[str, Any],
    source_kind: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if source_kind not in (SOURCE_GRU_BASE, SOURCE_GRU_MASK8_12):
        raise ValueError(f"unsupported source kind: {source_kind}")
    training = frozen["training"]
    architecture = training["architecture"]
    augmentation = training["augmentation"]
    config = frozen.get("_verified_source_config")
    if not isinstance(config, dict):
        raise AssertionError("verified parameterized-NBM config is unavailable")
    protocol = config["protocol"]
    expected_mask = (4, 8) if source_kind == SOURCE_GRU_BASE else (8, 12)
    checks = {
        "fit_role": int(frozen["nbm_train_role"]) == 4,
        "validation_calibration_role": int(
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
        "loss": str(protocol["loss"]) == "SmoothL1(beta=1.0)",
        "optimizer": str(protocol["optimizer"])
        == "AdamW(lr=0.001,weight_decay=0.0001)",
        "clean_probability": augmentation.get("clean_probability") == 0.40,
        "gaussian_probability": augmentation.get("gaussian_probability") == 0.40,
        "mask_probability": augmentation.get("mask_probability") == 0.20,
        "gaussian_std": augmentation.get("gaussian_std") == 0.04,
        "mask_minimum": augmentation.get("mask_minimum_samples")
        == expected_mask[0],
        "mask_maximum": augmentation.get("mask_maximum_samples")
        == expected_mask[1],
        "mask_all_channels": augmentation.get("mask_all_channels") is True,
        "architecture_name": architecture.get("name") == GRU_ARCHITECTURE_NAME,
        "parameter_count": int(architecture.get("parameter_count", -1))
        == GRU_PARAMETER_COUNT,
        "encoder_hidden": architecture.get("encoder_gru", {}).get("hidden_size")
        == 64,
        "bottleneck": architecture.get("latent_shape") == ["B", 16],
        "decoder_hidden": architecture.get("decoder_gru", {}).get("hidden_size")
        == 64,
        "zero_input": architecture.get("decoder_gru", {}).get("input")
        == "128-step all-zero sequence",
        "skip_free": architecture.get("skip_connections") is False,
        "architecture_exact": architecture == nbm_architecture_config(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(
            f"frozen {source_kind} source violates the experiment contract: {failed}"
        )
    return {
        "source_kind": source_kind,
        "all_checks_passed": True,
        "architecture": architecture,
        "augmentation": augmentation,
        "mask_span_samples": list(expected_mask),
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
    if source_kind not in (SOURCE_GRU_BASE, SOURCE_GRU_MASK8_12):
        raise ValueError(f"unsupported source kind: {source_kind}")
    payload = torch.load(
        Path(artifact["nbm_checkpoint"]), map_location=device, weights_only=False
    )
    model = GRUReconstructionNBM(channels=9, hidden=64, bottleneck=16).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    sigma = np.asarray(frozen["calibration"]["sigma"], dtype=np.float32)
    bias = np.asarray(frozen["calibration"]["bias"], dtype=np.float32)
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
        values = base.raw_features(scaler, raw, window_samples)
        contract = feature_contract(method)
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
    if method not in ("GRU_BASE_C", "GRU_MASK8_12_C") or frozen is None:
        raise AssertionError("a named residual arm requires frozen NBM metadata")
    model, sigma = load_reconstruction_model(
        artifact["source_kind"], artifact, frozen, device
    )
    scaled = prepare_nbm_windows(scaler, raw, center=True)
    reconstruction = reconstruct_gru(model, scaled, device)
    error_bct = np.ascontiguousarray(
        (scaled - reconstruction).astype(np.float32, copy=False).transpose(0, 2, 1)
    )
    values, clip_stats = base.build_scheme_c_features(
        error_bct, labels, sigma, window_samples
    )
    del model, reconstruction, error_bct
    if device.type == "cuda":
        torch.cuda.empty_cache()
    contract = feature_contract(method)
    return values, {
        "formula": contract["formula"],
        "shape": contract["shape"],
        "feature_contract_sha256": contract["sha256"],
        "uses_nbm": True,
        "subtracts_role5_bias": False,
        "sigma_estimation_centered_by_role5_bias": True,
        "uses_role5_sigma": True,
        "source_kind": artifact["source_kind"],
        "clip_statistics": clip_stats,
    }


def paired_initialization(
    seed: int, method: str
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    return _ORIGINAL_PAIRED_INITIALIZATION(
        seed, "RAW" if method == "RAW" else "GRU_V1_C"
    )


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
    expected_job = job_id(fold, method, seed)
    expected_source, _, expected_uses_nbm = source_for_method(args, method)
    expected_contract = feature_contract(method)
    expected = {
        "job_id": expected_job,
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
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise AssertionError(f"frozen classifier mismatch {expected_job}: {key}")
    if frozen.get("feature_contract") != expected_contract:
        raise AssertionError(f"feature contract changed: {expected_job}")
    checkpoint_sha = sha256_file(checkpoint)
    if frozen.get("checkpoint_sha256") != checkpoint_sha:
        raise AssertionError(f"TCN checkpoint changed: {expected_job}")
    expected_done = {
        "status": "frozen",
        "job_id": expected_job,
        "fold": fold,
        "method": method,
        "source_kind": expected_source,
        "uses_nbm": expected_uses_nbm,
        "nbm_seed": seed,
        "tcn_seed": seed,
        "checkpoint_sha256": checkpoint_sha,
        "frozen_validation_sha256": sha256_file(frozen_path),
        "scientific_data_sha256": scientific_data_sha256,
        "feature_contract_sha256": expected_contract["sha256"],
        "test_roles_accessed": False,
    }
    for key, value in expected_done.items():
        if done.get(key) != value:
            raise AssertionError(f"DONE_TRAIN mismatch {expected_job}: {key}")
    artifact = frozen["role4_scaler_artifact"]
    if method == "RAW":
        source_root = Path(artifact["scaler_json"]).resolve().parent.parent
        _, current_artifact, _ = load_role4_scaler_metadata(
            source_root, fold, seed, scientific_data_sha256
        )
    else:
        source_root = Path(artifact["frozen_json"]).resolve().parent.parent
        _, current_artifact, current_frozen = load_source_metadata(
            source_root, fold, expected_source, seed, scientific_data_sha256
        )
        validate_source_contract(current_frozen, expected_source, seed, args)
    for key in (
        "scaler_sha256",
        "scaler_json_sha256",
        "config_json_sha256",
        "protocol_sha256",
        "frozen_json_sha256",
        "done_nbm_json_sha256",
        "nbm_checkpoint_sha256",
        "calibration_sigma_sha256",
        "initial_model_state_sha256",
        "source_bundle_sha256",
        "scientific_data_sha256",
    ):
        if artifact.get(key) != current_artifact.get(key):
            raise AssertionError(f"upstream source changed {expected_job}: {key}")
    return frozen


def sealed_job(args: argparse.Namespace) -> dict[str, Any]:
    sealed = _ORIGINAL_SEALED_JOB(args)
    directory = job_dir(args.output_root.resolve(), args.fold, args.method, args.tcn_seed)
    frozen = json.loads(
        (directory / "frozen_validation.json").read_text(encoding="utf-8")
    )
    artifact = frozen["role4_scaler_artifact"]
    for key in ("calibration_sigma_sha256", "source_bundle_sha256"):
        if sealed.get(key) != artifact.get(key):
            raise AssertionError(f"sealed upstream field changed: {key}")
    return sealed


def run_seal(args: argparse.Namespace) -> None:
    validate_global_contract(args)
    root = args.output_root.resolve()
    code_sha = critical_code_sha256()
    scientific_data = processed_nbm_scientific_manifest(args.data_dir.resolve())
    entries: list[dict[str, Any]] = []
    for fold, method, seed in expected_jobs():
        directory = job_dir(root, fold, method, seed)
        frozen_path = directory / "frozen_validation.json"
        frozen = validate_frozen_training_job(
            args, directory, fold, method, seed, scientific_data["sha256"]
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
                "config_json_sha256": artifact.get("config_json_sha256"),
                "protocol_sha256": artifact.get("protocol_sha256"),
                "nbm_frozen_sha256": artifact["frozen_json_sha256"],
                "done_nbm_sha256": artifact["done_nbm_json_sha256"],
                "nbm_checkpoint_sha256": artifact["nbm_checkpoint_sha256"],
                "calibration_sigma_sha256": artifact[
                    "calibration_sigma_sha256"
                ],
                "initial_model_state_sha256": artifact.get(
                    "initial_model_state_sha256"
                ),
                "source_bundle_sha256": artifact["source_bundle_sha256"],
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
        for seed in REQUIRED_SEEDS:
            paired = [entry for entry in fold_entries if entry["tcn_seed"] == seed]
            by_method = {entry["method"]: entry for entry in paired}
            if set(by_method) != set(METHODS) or len(paired) != 3:
                raise AssertionError(f"fold {fold}, seed {seed} lacks three arms")
            if len({entry["pair_id"] for entry in paired}) != 1:
                raise AssertionError(f"fold {fold}, seed {seed} initialization mismatch")
            if any(entry["nbm_seed"] != seed for entry in paired):
                raise AssertionError(f"fold {fold}, seed {seed} NBM seed mismatch")
            if by_method["RAW"]["uses_nbm"] is not False:
                raise AssertionError("RAW unexpectedly uses an NBM")
            if by_method["RAW"]["nbm_checkpoint_sha256"] is not None:
                raise AssertionError("RAW unexpectedly binds an NBM checkpoint")
            if by_method["RAW"]["calibration_sigma_sha256"] is not None:
                raise AssertionError("RAW unexpectedly binds role-5 sigma")
            for residual in ("GRU_BASE_C", "GRU_MASK8_12_C"):
                if by_method[residual]["uses_nbm"] is not True:
                    raise AssertionError(f"{residual} does not use its NBM")
                for key in (
                    "nbm_checkpoint_sha256",
                    "nbm_frozen_sha256",
                    "calibration_sigma_sha256",
                    "config_json_sha256",
                    "protocol_sha256",
                    "initial_model_state_sha256",
                    "source_bundle_sha256",
                ):
                    if by_method[residual][key] is None:
                        raise AssertionError(f"{residual} lacks {key}")
            if (
                by_method["GRU_BASE_C"]["selected_state_sha256"]
                != by_method["GRU_MASK8_12_C"]["selected_state_sha256"]
            ):
                raise AssertionError("residual arms do not share TCN initialization")
            if (
                by_method["GRU_BASE_C"]["initial_model_state_sha256"]
                != by_method["GRU_MASK8_12_C"]["initial_model_state_sha256"]
            ):
                raise AssertionError("residual arms do not share NBM initialization")
            if (
                by_method["RAW"]["selected_state_sha256"]
                == by_method["GRU_BASE_C"]["selected_state_sha256"]
            ):
                raise AssertionError("RAW and 27-channel hashes unexpectedly match")
    if any(entry["tcn_max_epochs"] != 5 for entry in entries):
        raise AssertionError("TCN maximum epoch mismatch")
    if any(entry["tcn_patience"] != 2 for entry in entries):
        raise AssertionError("TCN patience mismatch")
    rows_by_fold = {
        fold: load_fold_rows(args.data_dir.resolve(), fold) for fold in FOLDS
    }
    source_audit = base.audit_protocol_dynamic(
        args.data_dir.resolve(), rows_by_fold, 64, 128, 64
    )
    source_audit["experiment_code_sha256"] = code_sha
    source_audit["scientific_data_manifest"] = scientific_data
    test_manifest = base.build_test_data_manifest(args.data_dir.resolve(), rows_by_fold)
    barrier = {
        "barrier_schema": "strict_test_barrier.v2",
        "status": "all_RAW_GRU_BASE_C_GRU_MASK8_12_C_classifiers_frozen",
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
    barrier["barrier_id"] = base.stable_json_hash(
        base.barrier_identity_payload(barrier)
    )
    atomic_json_dump(barrier, root / "TRAINING_BARRIER.json")
    atomic_json_dump(
        {
            "experiment": "strict_GRU_mask8_12_vs_mask4_8_vs_RAW",
            "methods": list(METHODS),
            "single_variable": "local all-channel mask length 4..8 -> 8..12 samples",
            "fixed_augmentation": {
                "clean_probability": 0.40,
                "gaussian_probability": 0.40,
                "mask_probability": 0.20,
                "gaussian_std": 0.04,
            },
            "nbm_training": "max300/pat20, SmoothL1, AdamW lr1e-3",
            "tcn_training": "max5/pat2, weighted BCE, AdamW lr1e-3",
            "paired_seeds": list(REQUIRED_SEEDS),
            "sampling_rate_hz": 64,
            "window_samples": 128,
            "stride_samples": 64,
            "roles": {str(key): value for key, value in ROLES.items()},
            "threshold": "roles 2/3 max balanced accuracy; ties F1 then threshold",
            "barrier_schema": barrier["barrier_schema"],
            "barrier_id": barrier["barrier_id"],
            "full_scientific_data_sha256": scientific_data["sha256"],
            "full_test_data_sha256": test_manifest["sha256"],
            "adaptive_benchmark_notice": (
                "roles 0/1 have been inspected in prior architecture studies; "
                "this run is exploratory and requires a new external holdout "
                "for an unbiased confirmatory claim"
            ),
            "pre_registered_success": {
                "comparison": "GRU_MASK8_12_C minus GRU_BASE_C",
                "sensitivity_mean_delta_min": 0.010,
                "sensitivity_positive_seed_count_min": 4,
                "auprc_mean_delta_min": -0.005,
                "precision_mean_delta_min": -0.010,
                "specificity_mean_delta_min": -0.010,
            },
        },
        root / "experiment_config.json",
    )
    print(f"GLOBAL TRAINING BARRIER SEALED jobs={len(entries)}", flush=True)


def mean_std(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "n": int(len(array)),
    }


def paired_deltas(
    seed_rows: list[dict[str, Any]], left: str, right: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for seed in REQUIRED_SEEDS:
        left_row = next(
            row for row in seed_rows if row["method"] == left and row["tcn_seed"] == seed
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
    return rows, {
        key: mean_std(row[key] for row in rows) for key in METRIC_KEYS
    }


def run_aggregate(args: argparse.Namespace) -> None:
    validate_global_contract(args)
    root = args.output_root.resolve()
    barrier = base.load_and_validate_barrier(root / "TRAINING_BARRIER.json")
    if barrier.get("barrier_schema") != "strict_test_barrier.v2":
        raise RuntimeError("strict_test_barrier.v2 is required")
    if barrier.get("source_audit", {}).get("experiment_code_sha256") != critical_code_sha256():
        raise AssertionError("experiment code changed after the global seal")
    scientific_data = processed_nbm_scientific_manifest(args.data_dir.resolve())
    if (
        barrier.get("source_audit", {})
        .get("scientific_data_manifest", {})
        .get("sha256")
        != scientific_data["sha256"]
    ):
        raise AssertionError("scientific dataset changed after seal")
    current_test = base.build_test_data_manifest(
        args.data_dir.resolve(),
        {fold: load_fold_rows(args.data_dir.resolve(), fold) for fold in FOLDS},
    )
    if current_test["sha256"] != barrier["test_data_manifest"]["sha256"]:
        raise AssertionError("permanent-test data changed after seal")
    barrier_jobs = {entry["job_id"]: entry for entry in barrier["jobs"]}
    results = []
    for fold, method, seed in expected_jobs():
        target = job_id(fold, method, seed)
        if target not in barrier_jobs:
            raise AssertionError(f"job absent from barrier: {target}")
        directory = job_dir(root, fold, method, seed)
        validate_frozen_training_job(
            args, directory, fold, method, seed, scientific_data["sha256"]
        )
        sealed = {
            **barrier_jobs[target],
            "barrier_schema": barrier["barrier_schema"],
            "barrier_id": barrier["barrier_id"],
            "test_data_manifest_sha256": barrier["test_data_manifest"]["sha256"],
        }
        result = base.validate_completed_test_artifacts(directory, sealed)
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
                        key: float(np.mean([result["test"][key] for result in subset]))
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
            summary_rows.append({"method": method, "metric": key, **primary[method][key]})
    write_csv(root / "method_summary_5seed_mean_std.csv", summary_rows)
    comparisons = {
        "GRU_BASE_C_minus_RAW": ("GRU_BASE_C", "RAW"),
        "GRU_MASK8_12_C_minus_RAW": ("GRU_MASK8_12_C", "RAW"),
        "GRU_MASK8_12_C_minus_GRU_BASE_C": ("GRU_MASK8_12_C", "GRU_BASE_C"),
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
                                [result["test_by_subject"][subject][key] for result in subset]
                            )
                        )
                        for key in METRIC_KEYS
                    }
                )
            stats = {
                key: mean_std(item[key] for item in per_seed) for key in METRIC_KEYS
            }
            subject_json[method][subject] = stats
            subject_rows.append(
                {
                    "method": method,
                    "subject_id": subject,
                    **{f"{key}_mean": stats[key]["mean"] for key in METRIC_KEYS},
                    **{f"{key}_std": stats[key]["std"] for key in METRIC_KEYS},
                }
            )
    write_csv(root / "subject_metrics_5seed_mean_std.csv", subject_rows)
    comparison_name = "GRU_MASK8_12_C_minus_GRU_BASE_C"
    candidate = delta_summaries[comparison_name]
    candidate_rows = delta_rows_by_comparison[comparison_name]
    checks = {
        "sensitivity_mean_delta_at_least_0.010": candidate["sensitivity"]["mean"]
        >= 0.010,
        "sensitivity_positive_in_at_least_4_of_5_seeds": sum(
            row["sensitivity"] > 0 for row in candidate_rows
        )
        >= 4,
        "auprc_mean_delta_at_least_minus_0.005": candidate["auprc"]["mean"]
        >= -0.005,
        "precision_mean_delta_at_least_minus_0.010": candidate["precision"]["mean"]
        >= -0.010,
        "specificity_mean_delta_at_least_minus_0.010": candidate["specificity"]["mean"]
        >= -0.010,
    }
    final = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metrics": primary,
        "paired_deltas": delta_summaries,
        "pre_registered_success": {"checks": checks, "all_passed": all(checks.values())},
        "subject_metrics": subject_json,
        "definition": "within each seed macro-average 3 folds; population SD over 5 paired seeds",
        "strict_global_test_barrier": True,
        "barrier_schema": barrier["barrier_schema"],
        "barrier_id": barrier["barrier_id"],
        "full_scientific_data_sha256": scientific_data["sha256"],
        "full_test_data_sha256": current_test["sha256"],
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


def _install_base_adapters() -> None:
    """Redirect reused train/evaluate functions to this experiment contract."""
    bindings = {
        "FOLDS": FOLDS,
        "METHODS": METHODS,
        "REQUIRED_SEEDS": REQUIRED_SEEDS,
        "critical_code_sha256": critical_code_sha256,
        "feature_contract": feature_contract,
        "validate_global_contract": validate_global_contract,
        "require_job_args": require_job_args,
        "expected_jobs": expected_jobs,
        "job_id": job_id,
        "job_dir": job_dir,
        "source_for_method": source_for_method,
        "checkpoint_name": checkpoint_name,
        "load_source_metadata": load_source_metadata,
        "load_role4_scaler_metadata": load_role4_scaler_metadata,
        "validate_source_contract": validate_source_contract,
        "load_reconstruction_model": load_reconstruction_model,
        "make_method_features": make_method_features,
        "paired_initialization": paired_initialization,
        "validate_frozen_training_job": validate_frozen_training_job,
        "sealed_job": sealed_job,
    }
    for name, value in bindings.items():
        setattr(base, name, value)


def main() -> None:
    args = parse_args()
    validate_global_contract(args)
    _install_base_adapters()
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    if args.stage == "seal":
        run_seal(args)
        return
    if args.stage == "aggregate":
        run_aggregate(args)
        return
    require_job_args(args)
    device = base.resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    if args.stage == "train":
        base.run_train(args, device)
    elif args.stage == "evaluate":
        base.run_evaluate(args, device)
    else:
        raise ValueError(f"unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
