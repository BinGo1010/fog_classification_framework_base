#!/usr/bin/env python3
"""Strict Persistence-C versus multivariate Linear-AR(8)-C experiment."""

from __future__ import annotations

import argparse
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
from cnbr_fog.resume import atomic_json_dump
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts.run_daphnet_nbm300_c_vs_raw_ablation import build_scheme_c_features
from scripts.run_daphnet_persistence_linear_ar_nbm_fold import (
    MultivariateLinearAR8,
    architecture_config as nbm_architecture_config,
    augmentation_config as nbm_augmentation_config,
    checkpoint_name as nbm_checkpoint_name,
    linear_ar_reconstruct,
    persistence_reconstruct,
    protocol_contract as nbm_protocol_contract,
    source_code_sha256 as nbm_source_code_sha256,
)
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    ROLES,
    SUBJECTS,
    RobustScaler,
    load_fold_rows,
    write_csv,
)
from scripts.run_daphnet_residual_calibration_abcd import sha256_file
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import prepare_nbm_windows

FOLDS = (0, 1, 2)
METHODS = ("PERSISTENCE_C", "LINEAR_AR_C")
REQUIRED_SEEDS = (0, 52, 161, 5216, 52161)
METRIC_KEYS = base.METRIC_KEYS
SOURCE_PERSISTENCE = "persistence_lag1"
SOURCE_LINEAR_AR = "multivariate_linear_ar8"
SOURCE_VARIANTS = {
    SOURCE_PERSISTENCE: "PERSISTENCE",
    SOURCE_LINEAR_AR: "LINEAR_AR",
}
NBM_WORKER = REPO_ROOT / "scripts" / "run_daphnet_persistence_linear_ar_nbm_fold.py"
CRITICAL_CODE_PATHS = (
    Path(__file__).resolve(),
    NBM_WORKER,
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
    method: {
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
    }
    for method in METHODS
}
_BASE_PAIRED_INITIALIZATION = base.paired_initialization


def critical_code_sha256() -> dict[str, str]:
    missing = [str(path) for path in CRITICAL_CODE_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"critical experiment code missing: {missing}")
    return {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in CRITICAL_CODE_PATHS
    }


def feature_contract(method: str) -> dict[str, Any]:
    contract = dict(FEATURE_CONTRACTS[method])
    return {**contract, "sha256": base.stable_json_hash(contract)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("train", "seal", "evaluate", "aggregate"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed_NBM",
    )
    parser.add_argument("--persistence-source-root", type=Path, required=True)
    parser.add_argument("--linear-ar-source-root", type=Path, required=True)
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
    parser.add_argument("--required-linear-ar-max-epochs", type=int, default=300)
    parser.add_argument("--required-linear-ar-patience", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_global_contract(args: argparse.Namespace) -> tuple[int, ...]:
    required = base.parse_csv_ints(args.required_seeds)
    if (
        base.parse_csv_ints(args.nbm_seeds) != REQUIRED_SEEDS
        or base.parse_csv_ints(args.tcn_seeds) != REQUIRED_SEEDS
        or required != REQUIRED_SEEDS
    ):
        raise ValueError(f"this experiment requires paired seeds {REQUIRED_SEEDS}")
    if (args.sampling_rate_hz, args.window_samples, args.stride_samples) != (64, 128, 64):
        raise ValueError("experiment is frozen to 64 Hz/window128/stride64")
    if (args.tcn_max_epochs, args.tcn_patience) != (5, 2):
        raise ValueError("TCN training is frozen to max5/pat2")
    if (args.required_linear_ar_max_epochs, args.required_linear_ar_patience) != (300, 20):
        raise ValueError("Linear-AR training is frozen to max300/pat20")
    return required


def require_job_args(args: argparse.Namespace) -> None:
    if any(value is None for value in (args.fold, args.method, args.nbm_seed, args.tcn_seed)):
        raise ValueError("fold/method/nbm-seed/tcn-seed are required")
    if args.nbm_seed != args.tcn_seed or args.nbm_seed not in REQUIRED_SEEDS:
        raise ValueError("NBM and TCN seeds must be the same registered seed")


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
    if method == "PERSISTENCE_C":
        return SOURCE_PERSISTENCE, args.persistence_source_root.resolve(), True
    if method == "LINEAR_AR_C":
        return SOURCE_LINEAR_AR, args.linear_ar_source_root.resolve(), True
    raise ValueError(f"unsupported method: {method}")


def checkpoint_name(source_kind: str) -> str:
    return nbm_checkpoint_name(SOURCE_VARIANTS[source_kind])


def load_source_metadata(
    source_root: Path,
    fold: int,
    source_kind: str,
    seed: int,
    scientific_data_sha256: str,
) -> tuple[RobustScaler, dict[str, Any], dict[str, Any]]:
    variant = SOURCE_VARIANTS[source_kind]
    fold_dir = source_root / f"fold_{fold}"
    paths = {
        "config": fold_dir / "config.json",
        "scaler": fold_dir / "scaler_role4.json",
        "frozen": fold_dir / "nbm_frozen.json",
        "done": fold_dir / "DONE_NBM.json",
        "checkpoint": fold_dir / "checkpoints" / checkpoint_name(source_kind),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"frozen {source_kind} source missing: {missing}")
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    scaler_file = json.loads(paths["scaler"].read_text(encoding="utf-8"))
    frozen = json.loads(paths["frozen"].read_text(encoding="utf-8"))
    done = json.loads(paths["done"].read_text(encoding="utf-8"))
    expected_protocol = nbm_protocol_contract(variant, scientific_data_sha256)
    expected_source_code = nbm_source_code_sha256()
    checks = {
        "config_identity": config.get("fold") == fold and config.get("seed") == seed,
        "config_variant": config.get("variant") == variant,
        "protocol": config.get("protocol") == expected_protocol,
        "scaler_identity": scaler_file.get("fold") == fold and scaler_file.get("seed") == seed,
        "scaler_role": scaler_file.get("scaler_fit_role") == 4,
        "scaler_dataset": scaler_file.get("scientific_data_sha256") == scientific_data_sha256,
        "frozen_variant": frozen.get("variant") == variant,
        "frozen_dataset": frozen.get("scientific_data_sha256") == scientific_data_sha256,
        "done": done.get("status") == "frozen",
        "done_identity": done.get("fold") == fold and done.get("seed") == seed,
        "done_variant": done.get("variant") == variant,
        "config_source_code": config.get("source_code_sha256") == expected_source_code,
        "frozen_source_code": frozen.get("source_code_sha256") == expected_source_code,
        "done_source_code": done.get("source_code_sha256") == expected_source_code,
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"{source_kind} source contract failed: {failed}")
    if scaler_file["scaler"] != frozen["scaler"]:
        raise AssertionError("role4 scaler/frozen scaler mismatch")
    hashes = {
        "config_json_sha256": sha256_file(paths["config"]),
        "scaler_json_sha256": sha256_file(paths["scaler"]),
        "frozen_json_sha256": sha256_file(paths["frozen"]),
        "done_nbm_json_sha256": sha256_file(paths["done"]),
        "nbm_checkpoint_sha256": sha256_file(paths["checkpoint"]),
    }
    done_hash_names = {
        "config_json_sha256": "config_sha256",
        "scaler_json_sha256": "scaler_role4_sha256",
        "frozen_json_sha256": "nbm_frozen_sha256",
        "nbm_checkpoint_sha256": "checkpoint_sha256",
    }
    for artifact_key, done_key in done_hash_names.items():
        if done.get(done_key) != hashes[artifact_key]:
            raise AssertionError(f"{source_kind} DONE hash mismatch: {done_key}")
    payload = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    if (
        payload.get("variant") != variant
        or payload.get("fold") != fold
        or payload.get("seed") != seed
    ):
        raise AssertionError("checkpoint identity mismatch")
    if payload.get("architecture") != nbm_architecture_config(variant):
        raise AssertionError("checkpoint architecture changed")
    if payload.get("source_code_sha256") != expected_source_code:
        raise AssertionError("checkpoint NBM source code changed")
    scaler_payload = frozen["scaler"]
    scaler = RobustScaler(
        median=np.asarray(scaler_payload["median"], dtype=np.float32),
        iqr=np.asarray(scaler_payload["iqr"], dtype=np.float32),
        epsilon=float(scaler_payload.get("epsilon", 1e-6)),
    )
    sigma_hash = base.stable_json_hash(
        {"sigma": [float(value) for value in frozen["calibration"]["sigma"]]}
    )
    artifact = {
        "fold": fold,
        "source_kind": source_kind,
        "scaler_fit_role": 4,
        "scaler_unique_raw_points": int(frozen["scaler_unique_raw_points"]),
        "scaler_sha256": base.stable_json_hash(scaler_payload),
        "scaler_json": str(paths["scaler"].resolve()),
        "frozen_json": str(paths["frozen"].resolve()),
        "done_nbm_json": str(paths["done"].resolve()),
        "nbm_checkpoint": str(paths["checkpoint"].resolve()),
        "scientific_data_sha256": scientific_data_sha256,
        "protocol_sha256": str(frozen["protocol_sha256"]),
        "calibration_sigma_sha256": sigma_hash,
        **hashes,
    }
    artifact["source_bundle_sha256"] = base.stable_json_hash(
        {key: value for key, value in artifact.items() if key.endswith("sha256")}
    )
    return scaler, artifact, frozen


def validate_source_contract(
    frozen: dict[str, Any],
    source_kind: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    variant = SOURCE_VARIANTS[source_kind]
    training = frozen["training"]
    architecture = training["architecture"]
    checks = {
        "variant": frozen.get("variant") == variant,
        "calibration_role": frozen.get("nbm_calibration_role") == 5,
        "role5_clean": frozen.get("validation_mask_or_noise") is False,
        "best_restored": frozen.get("best_checkpoint_restored_before_calibration") is True,
        "training_seed": training.get("seed") == seed,
        "architecture": architecture == nbm_architecture_config(variant),
        "augmentation": training.get("augmentation") == nbm_augmentation_config(variant),
    }
    if variant == "PERSISTENCE":
        checks.update(
            {
                "parameter_free": architecture.get("parameter_count") == 0,
                "no_fit_role": frozen.get("nbm_train_role") is None,
                "epochs": training.get("maximum_epochs") == 0,
            }
        )
    else:
        checks.update(
            {
                "fit_role": frozen.get("nbm_train_role") == 4,
                "earlystop_role": frozen.get("nbm_earlystop_role") == 5,
                "parameter_count": architecture.get("parameter_count") == 657,
                "order": architecture.get("order") == 8,
                "cross_channel": architecture.get("cross_channel") is True,
                "maximum_epochs": training.get("maximum_epochs") == args.required_linear_ar_max_epochs,
                "patience": training.get("patience") == args.required_linear_ar_patience,
            }
        )
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"{source_kind} frozen contract failed: {failed}")
    return {
        "source_kind": source_kind,
        "all_checks_passed": True,
        "architecture": architecture,
        "augmentation": training["augmentation"],
        "seed": seed,
        "checkpoint_rule": nbm_protocol_contract(
            variant, frozen["scientific_data_sha256"]
        )["checkpoint_rule"],
    }


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
    if frozen is None:
        raise AssertionError("both methods require a frozen NBM source")
    scaled = prepare_nbm_windows(scaler, raw, center=True)
    if method == "PERSISTENCE_C":
        reconstruction = persistence_reconstruct(scaled)
    elif method == "LINEAR_AR_C":
        payload = torch.load(artifact["nbm_checkpoint"], map_location=device, weights_only=False)
        model = MultivariateLinearAR8().to(device)
        model.load_state_dict(payload["model_state"])
        reconstruction = linear_ar_reconstruct(model, scaled, device)
        del model
    else:
        raise ValueError(f"unsupported method: {method}")
    sigma = np.asarray(frozen["calibration"]["sigma"], dtype=np.float32)
    error_bct = np.ascontiguousarray((scaled - reconstruction).transpose(0, 2, 1))
    values, clip_stats = build_scheme_c_features(
        error_bct, labels, sigma, window_samples
    )
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
    # Both methods have the same 27-channel scheme-C representation.
    return _BASE_PAIRED_INITIALIZATION(seed, "GRU_V1_C")


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
            raise FileNotFoundError(f"incomplete classifier job: {path}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    expected_job = job_id(fold, method, seed)
    source_kind, _, _ = source_for_method(args, method)
    expected_contract = feature_contract(method)
    expected = {
        "job_id": expected_job,
        "fold": fold,
        "method": method,
        "source_kind": source_kind,
        "uses_nbm": True,
        "nbm_seed": seed,
        "tcn_seed": seed,
        "test_roles_accessed": False,
        "scientific_data_sha256": scientific_data_sha256,
        "experiment_code_sha256": critical_code_sha256(),
        "feature_contract": expected_contract,
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise AssertionError(f"frozen classifier mismatch {expected_job}: {key}")
    if frozen.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise AssertionError("TCN checkpoint changed")
    done_expected = {
        "status": "frozen",
        "job_id": expected_job,
        "fold": fold,
        "method": method,
        "source_kind": source_kind,
        "uses_nbm": True,
        "nbm_seed": seed,
        "tcn_seed": seed,
        "checkpoint_sha256": sha256_file(checkpoint),
        "frozen_validation_sha256": sha256_file(frozen_path),
        "scientific_data_sha256": scientific_data_sha256,
        "feature_contract_sha256": expected_contract["sha256"],
        "test_roles_accessed": False,
    }
    for key, value in done_expected.items():
        if done.get(key) != value:
            raise AssertionError(f"DONE_TRAIN mismatch {expected_job}: {key}")
    artifact = frozen["role4_scaler_artifact"]
    source_root = Path(artifact["frozen_json"]).resolve().parent.parent
    _, current_artifact, current_frozen = load_source_metadata(
        source_root, fold, source_kind, seed, scientific_data_sha256
    )
    validate_source_contract(current_frozen, source_kind, seed, args)
    for key in (
        "scaler_sha256",
        "scaler_json_sha256",
        "config_json_sha256",
        "protocol_sha256",
        "frozen_json_sha256",
        "done_nbm_json_sha256",
        "nbm_checkpoint_sha256",
        "calibration_sigma_sha256",
        "source_bundle_sha256",
        "scientific_data_sha256",
    ):
        if artifact.get(key) != current_artifact.get(key):
            raise AssertionError(f"upstream source changed {expected_job}: {key}")
    return frozen


def sealed_job(args: argparse.Namespace) -> dict[str, Any]:
    barrier_path = args.output_root.resolve() / "TRAINING_BARRIER.json"
    if not barrier_path.is_file():
        raise FileNotFoundError("TRAINING_BARRIER.json missing; roles 0/1 forbidden")
    barrier = base.load_and_validate_barrier(barrier_path)
    if barrier.get("barrier_schema") != "strict_test_barrier.v2":
        raise RuntimeError("strict_test_barrier.v2 is required")
    target = job_id(args.fold, args.method, args.tcn_seed)
    matches = [entry for entry in barrier["jobs"] if entry["job_id"] == target]
    if len(matches) != 1:
        raise AssertionError(f"job not uniquely sealed: {target}")
    sealed = dict(matches[0])
    source_kind, _, _ = source_for_method(args, args.method)
    if sealed["source_kind"] != source_kind or sealed["uses_nbm"] is not True:
        raise AssertionError("requested method differs from sealed source")
    sealed.update(
        {
            "barrier_schema": barrier["barrier_schema"],
            "barrier_id": barrier["barrier_id"],
            "test_data_manifest_sha256": barrier["test_data_manifest"]["sha256"],
        }
    )
    return sealed


def run_seal(args: argparse.Namespace) -> None:
    validate_global_contract(args)
    root = args.output_root.resolve()
    scientific_data = processed_nbm_scientific_manifest(args.data_dir.resolve())
    entries = []
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
                "nbm_seed": seed,
                "source_kind": frozen["source_kind"],
                "uses_nbm": True,
                "threshold": frozen["threshold"],
                "checkpoint_sha256": frozen["checkpoint_sha256"],
                "frozen_validation_sha256": sha256_file(frozen_path),
                "pair_id": frozen["initialization"]["pair_id"],
                "selected_state_sha256": frozen["initialization"]["selected_state_sha256"],
                "scaler_sha256": artifact["scaler_sha256"],
                "scaler_json_sha256": artifact["scaler_json_sha256"],
                "nbm_frozen_sha256": artifact["frozen_json_sha256"],
                "done_nbm_sha256": artifact["done_nbm_json_sha256"],
                "nbm_checkpoint_sha256": artifact["nbm_checkpoint_sha256"],
                "calibration_sigma_sha256": artifact["calibration_sigma_sha256"],
                "source_bundle_sha256": artifact["source_bundle_sha256"],
                "nbm_architecture_name": frozen["nbm_contract"]["architecture"]["name"],
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
            raise AssertionError(f"fold {fold} scalers differ across methods/seeds")
        if len({entry["pos_weight"] for entry in fold_entries}) != 1:
            raise AssertionError(f"fold {fold} pos_weight differs")
        for seed in REQUIRED_SEEDS:
            paired = [entry for entry in fold_entries if entry["tcn_seed"] == seed]
            if len(paired) != 2 or {entry["method"] for entry in paired} != set(METHODS):
                raise AssertionError(f"fold {fold}, seed {seed} lacks two methods")
            if len({entry["pair_id"] for entry in paired}) != 1:
                raise AssertionError("TCN initialization pairing failed")
            if len({entry["selected_state_sha256"] for entry in paired}) != 1:
                raise AssertionError("27-channel TCN state pairing failed")
    if any(entry["tcn_max_epochs"] != 5 or entry["tcn_patience"] != 2 for entry in entries):
        raise AssertionError("TCN training contract mismatch")
    rows_by_fold = {
        fold: load_fold_rows(args.data_dir.resolve(), fold) for fold in FOLDS
    }
    source_audit = base.audit_protocol_dynamic(
        args.data_dir.resolve(), rows_by_fold, 64, 128, 64
    )
    source_audit["experiment_code_sha256"] = critical_code_sha256()
    source_audit["scientific_data_manifest"] = scientific_data
    test_manifest = base.build_test_data_manifest(args.data_dir.resolve(), rows_by_fold)
    barrier = {
        "barrier_schema": "strict_test_barrier.v2",
        "status": "all_PERSISTENCE_C_and_LINEAR_AR_C_classifiers_frozen",
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
    barrier["barrier_id"] = base.stable_json_hash(base.barrier_identity_payload(barrier))
    atomic_json_dump(barrier, root / "TRAINING_BARRIER.json")
    atomic_json_dump(
        {
            "experiment": "strict_Persistence_vs_multivariate_Linear_AR8_schemeC_TCN",
            "methods": list(METHODS),
            "persistence": "Xhat[0]=X[0], Xhat[t]=X[t-1], parameter-free",
            "linear_ar": "joint 9-channel causal AR(8), 657 params, denoising max300/pat20",
            "linear_ar_augmentation": "40% clean/40% Gaussian std=.04/20% all-axis mask4..8",
            "scheme_c": "[r,abs(r),delta] [B,27,128]",
            "tcn_training": "max5/pat2, validation PR-AUC checkpoint",
            "paired_seeds": list(REQUIRED_SEEDS),
            "roles": {str(key): value for key, value in ROLES.items()},
            "threshold": "roles2/3 max balanced accuracy; ties F1 then higher threshold",
            "barrier_schema": barrier["barrier_schema"],
            "barrier_id": barrier["barrier_id"],
            "full_scientific_data_sha256": scientific_data["sha256"],
            "full_test_data_sha256": test_manifest["sha256"],
            "adaptive_benchmark_notice": (
                "roles 0/1 were inspected in prior studies; this is exploratory"
            ),
        },
        root / "experiment_config.json",
    )
    print(f"GLOBAL TRAINING BARRIER SEALED jobs={len(entries)}", flush=True)


def mean_std(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0)), "n": len(array)}


def run_aggregate(args: argparse.Namespace) -> None:
    validate_global_contract(args)
    root = args.output_root.resolve()
    barrier = base.load_and_validate_barrier(root / "TRAINING_BARRIER.json")
    if barrier.get("barrier_schema") != "strict_test_barrier.v2":
        raise RuntimeError("strict_test_barrier.v2 is required")
    if barrier["source_audit"].get("experiment_code_sha256") != critical_code_sha256():
        raise AssertionError("experiment code changed after seal")
    scientific_data = processed_nbm_scientific_manifest(args.data_dir.resolve())
    if barrier["source_audit"]["scientific_data_manifest"]["sha256"] != scientific_data["sha256"]:
        raise AssertionError("scientific dataset changed")
    current_test = base.build_test_data_manifest(
        args.data_dir.resolve(),
        {fold: load_fold_rows(args.data_dir.resolve(), fold) for fold in FOLDS},
    )
    if current_test["sha256"] != barrier["test_data_manifest"]["sha256"]:
        raise AssertionError("test data changed")
    barrier_jobs = {entry["job_id"]: entry for entry in barrier["jobs"]}
    results = []
    for fold, method, seed in expected_jobs():
        target = job_id(fold, method, seed)
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
        if result["method"] != method or int(result["fold"]) != fold or int(result["tcn_seed"]) != seed:
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
    write_csv(root / "run_metrics_30.csv", run_rows)
    seed_rows = []
    for method in METHODS:
        for seed in REQUIRED_SEEDS:
            subset = [r for r in results if r["method"] == method and r["tcn_seed"] == seed]
            seed_rows.append(
                {
                    "method": method,
                    "tcn_seed": seed,
                    **{key: float(np.mean([r["test"][key] for r in subset])) for key in METRIC_KEYS},
                }
            )
    write_csv(root / "seed_macro_over_3folds.csv", seed_rows)
    primary = {
        method: {
            key: mean_std(row[key] for row in seed_rows if row["method"] == method)
            for key in METRIC_KEYS
        }
        for method in METHODS
    }
    write_csv(
        root / "method_summary_5seed_mean_std.csv",
        [
            {"method": method, "metric": key, **primary[method][key]}
            for method in METHODS
            for key in METRIC_KEYS
        ],
    )
    delta_rows = []
    for seed in REQUIRED_SEEDS:
        left = next(row for row in seed_rows if row["method"] == "LINEAR_AR_C" and row["tcn_seed"] == seed)
        right = next(row for row in seed_rows if row["method"] == "PERSISTENCE_C" and row["tcn_seed"] == seed)
        delta_rows.append({"tcn_seed": seed, **{key: left[key] - right[key] for key in METRIC_KEYS}})
    delta_summary = {key: mean_std(row[key] for row in delta_rows) for key in METRIC_KEYS}
    write_csv(root / "paired_delta_LINEAR_AR_C_minus_PERSISTENCE_C_by_seed.csv", delta_rows)
    write_csv(
        root / "paired_delta_LINEAR_AR_C_minus_PERSISTENCE_C_summary.csv",
        [{"metric": key, **value} for key, value in delta_summary.items()],
    )
    subject_rows = []
    subject_json: dict[str, Any] = {method: {} for method in METHODS}
    for method in METHODS:
        for subject in SUBJECTS:
            per_seed = []
            for seed in REQUIRED_SEEDS:
                subset = [r for r in results if r["method"] == method and r["tcn_seed"] == seed]
                per_seed.append(
                    {
                        key: float(np.mean([r["test_by_subject"][subject][key] for r in subset]))
                        for key in METRIC_KEYS
                    }
                )
            stats = {key: mean_std(item[key] for item in per_seed) for key in METRIC_KEYS}
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
    final = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metrics": primary,
        "paired_delta_LINEAR_AR_C_minus_PERSISTENCE_C": delta_summary,
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


def _install_adapters() -> None:
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
        "validate_source_contract": validate_source_contract,
        "make_method_features": make_method_features,
        "paired_initialization": paired_initialization,
        "validate_frozen_training_job": validate_frozen_training_job,
        "sealed_job": sealed_job,
    }
    for name, value in bindings.items():
        setattr(base, name, value)


def run_train(args: argparse.Namespace, device: torch.device) -> None:
    """Run the shared TCN trainer, then replace its RAW-specific audit label."""

    base.run_train(args, device)
    directory = job_dir(
        args.output_root.resolve(), args.fold, args.method, args.tcn_seed
    )
    frozen_path = directory / "frozen_validation.json"
    done_path = directory / "DONE_TRAIN.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    expected_policy = (
        "Each method uses only its own clean role-5 residual calibration after "
        "the NBM is frozen; role 5 is never used to fit the TCN classifier."
    )
    changed = frozen.pop("raw_ablation_role5_policy", None) is not None
    if frozen.get("role5_policy") != expected_policy:
        frozen["role5_policy"] = expected_policy
        changed = True
    if changed:
        atomic_json_dump(frozen, frozen_path)
        done["frozen_validation_sha256"] = sha256_file(frozen_path)
        atomic_json_dump(done, done_path)


def main() -> None:
    args = parse_args()
    validate_global_contract(args)
    _install_adapters()
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
        run_train(args, device)
    else:
        base.run_evaluate(args, device)


if __name__ == "__main__":
    main()
