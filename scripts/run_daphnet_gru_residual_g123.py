#!/usr/bin/env python3
"""Strict G1/G2/G3 residual experiment using frozen paired GRU-v1 NBMs.

G1: q=clip((e-b)/(sigma+eps), -12, 12); r=q-mean_t(q)
G2: r=clip((e-b)/(sigma+eps), -12, 12)
G3: r=asinh((e-b)/(sigma+eps))

Every group uses F=[r, abs(r), delta_t(r)] in [B,27,128].  GRU-NBMs are
frozen; roles 6/7 train the TCN, roles 2/3 select its checkpoint and threshold,
and roles 0/1 remain inaccessible until all 27 jobs are globally sealed.
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

import scripts.run_daphnet_residual_calibration_abcd as core
from cnbr_fog.data import DaphnetDataset
from cnbr_fog.evaluation import binary_metrics
from scripts.run_daphnet_nbm300_c_vs_raw_ablation import (
    audit_protocol_dynamic,
    barrier_identity_payload,
    build_test_data_manifest,
    load_and_validate_barrier,
    stable_json_hash,
    validate_completed_test_artifacts,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    GRUReconstructionNBM,
    prepare_nbm_windows,
    reconstruct,
)


GROUPS = ("G1", "G2", "G3")
FOLDS = (0, 1, 2)
REQUIRED_SEEDS = (0, 52, 161)
EPSILON = 1e-6
CLIP_LIMIT = 12.0

GROUP_CONFIG: dict[str, dict[str, Any]] = {
    "G1": {
        "name": "bias_scale_clip_then_window_center",
        "uses_b": True,
        "uses_sigma": True,
        "nonlinearity": "hard_clip",
        "clip": [-12.0, 12.0],
        "residual_window_centering": True,
        "formula": "q=clip((e-b)/(sigma+1e-6),-12,12); r=q-mean_t(q)",
    },
    "G2": {
        "name": "bias_scale_clip_without_second_center",
        "uses_b": True,
        "uses_sigma": True,
        "nonlinearity": "hard_clip",
        "clip": [-12.0, 12.0],
        "residual_window_centering": False,
        "formula": "r=clip((e-b)/(sigma+1e-6),-12,12)",
    },
    "G3": {
        "name": "bias_scale_asinh_without_second_center",
        "uses_b": True,
        "uses_sigma": True,
        "nonlinearity": "asinh",
        "clip": None,
        "residual_window_centering": False,
        "formula": "r=asinh((e-b)/(sigma+1e-6)); no hard clipping",
    },
}


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
    parser.add_argument(
        "--nbm-source-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_gru_nbm300_C_vs_raw_tcn_ep10pat2_seedset_0_52_161"
        / "nbm_source"
        / "seed_0",
        help=(
            "For train/evaluate, point to one paired seed directory containing fold_0..2. "
            "The launcher supplies seed_0, seed_52 or seed_161 automatically."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_gru_nbm300_residual_G1_G2_G3_tcn_ep10pat2_seedset_0_52_161",
    )
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--tcn-seed", type=int)
    parser.add_argument("--tcn-seeds", default="0,52,161")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tcn-max-epochs", type=int, default=10)
    parser.add_argument("--tcn-patience", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_contract(args: argparse.Namespace) -> tuple[int, ...]:
    seeds = core.parse_seed_list(args.tcn_seeds)
    groups = core.parse_group_list(args.groups)
    if seeds != REQUIRED_SEEDS:
        raise ValueError(f"this experiment requires exact paired seeds {REQUIRED_SEEDS}")
    if groups != GROUPS:
        raise ValueError(f"this experiment requires exact groups {GROUPS}")
    if args.tcn_max_epochs != 10 or args.tcn_patience != 2:
        raise ValueError("this controlled experiment requires TCN max_epoch=10, patience=2")
    if args.tcn_seed is not None and args.tcn_seed not in seeds:
        raise ValueError(f"--tcn-seed must be one of {seeds}")
    return seeds


def no_hard_clip_statistics() -> dict[str, Any]:
    return {
        "applicable": False,
        "reason": "G3 uses asinh compression and performs no hard clipping",
        "transform": "asinh",
    }


def load_frozen_gru_nbm(
    source_root: Path,
    fold: int,
    device: torch.device,
) -> tuple[GRUReconstructionNBM, core.RobustScaler, np.ndarray, np.ndarray, dict[str, Any]]:
    fold_dir = source_root.resolve() / f"fold_{fold}"
    checkpoint = fold_dir / "checkpoints" / "gru_nbm_best.pt"
    frozen_path = fold_dir / "nbm_frozen.json"
    done_path = fold_dir / "DONE_NBM.json"
    for path in (checkpoint, frozen_path, done_path):
        if not path.is_file():
            raise FileNotFoundError(f"frozen paired GRU-NBM artifact missing: {path}")

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    training = frozen["training"]
    architecture = training["architecture"]
    if architecture.get("name") != "gru_reconstruction_nbm_v1":
        raise AssertionError("unexpected frozen GRU-NBM architecture")
    if int(architecture.get("parameter_count", -1)) != 31_513:
        raise AssertionError("unexpected frozen GRU-NBM parameter count")
    if int(training.get("maximum_epochs", -1)) != 300 or int(training.get("patience", -1)) != 20:
        raise AssertionError("frozen GRU-NBM must use max_epoch=300 and patience=20")
    seed = int(training["seed"])
    if done.get("status") != "frozen" or int(done.get("fold", -1)) != fold:
        raise AssertionError("invalid GRU-NBM DONE_NBM identity")
    if int(done.get("seed", -1)) != seed:
        raise AssertionError("GRU-NBM frozen/DONE seed mismatch")
    checkpoint_sha256 = core.sha256_file(checkpoint)
    if done.get("checkpoint_sha256") != checkpoint_sha256:
        raise AssertionError("GRU-NBM checkpoint hash differs from DONE_NBM")

    scaler_payload = frozen["scaler"]
    scaler = core.RobustScaler(
        median=np.asarray(scaler_payload["median"], dtype=np.float32),
        iqr=np.asarray(scaler_payload["iqr"], dtype=np.float32),
        epsilon=float(scaler_payload.get("epsilon", EPSILON)),
    )
    calibration = frozen["calibration"]
    bias = np.asarray(calibration["bias"], dtype=np.float32)
    sigma = np.asarray(calibration["sigma"], dtype=np.float32)
    if (
        bias.shape != (9,)
        or sigma.shape != (9,)
        or not np.all(np.isfinite(bias))
        or not np.all(np.isfinite(sigma))
        or np.any(sigma < 0.05)
    ):
        raise AssertionError("invalid role-5 b/sigma calibration")

    model = GRUReconstructionNBM(channels=9, hidden=64, bottleneck=16).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if int(payload.get("seed", -1)) != seed:
        raise AssertionError("GRU-NBM checkpoint seed mismatch")
    if int(payload.get("epoch", -1)) != int(training["best_epoch"]):
        raise AssertionError("GRU-NBM checkpoint best epoch mismatch")
    if not np.isclose(
        float(payload.get("validation_huber", np.nan)),
        float(training["best_validation_huber"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("GRU-NBM checkpoint validation loss mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    manifest = {
        "fold": fold,
        "seed": seed,
        "architecture": architecture["name"],
        "parameter_count": int(architecture["parameter_count"]),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "frozen_json": str(frozen_path.resolve()),
        "frozen_json_sha256": core.sha256_file(frozen_path),
        "done_nbm_sha256": core.sha256_file(done_path),
        "scaler_role4_sha256": stable_json_hash(scaler.as_dict()),
        "best_epoch": int(training["best_epoch"]),
        "best_validation_loss": float(training["best_validation_huber"]),
        "best_validation_metric": "role5_validation_SmoothL1",
        "calibration_role": 5,
        "scaler_role": 4,
        "scientific_data_sha256": frozen.get("scientific_data_sha256"),
    }
    return model, scaler, bias, sigma, manifest


def reconstruction_error_gru(
    model: GRUReconstructionNBM,
    scaler: core.RobustScaler,
    raw: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    scaled = prepare_nbm_windows(scaler, raw, center=True)
    reconstructed = reconstruct(model, scaled, device)
    error = (scaled - reconstructed).astype(np.float32, copy=False)
    if error.ndim != 3 or error.shape[1:] != (128, 9):
        raise AssertionError(f"unexpected GRU reconstruction error shape: {error.shape}")
    return np.ascontiguousarray(error.transpose(0, 2, 1))


def build_g123_features(
    error_bct: np.ndarray,
    labels: np.ndarray,
    group: str,
    bias: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return F=[r,abs(r),delta(r)] in [N,T,27] plus transform diagnostics."""
    error = np.asarray(error_bct, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int8)
    if error.ndim != 3 or error.shape[1:] != (9, 128):
        raise ValueError(f"expected error [N,9,128], got {error.shape}")
    if labels.shape != (len(error),):
        raise ValueError("label/residual batch mismatch")
    if bias.shape != (9,) or sigma.shape != (9,):
        raise ValueError("b and sigma must both have shape [9]")
    standardized = (error - bias[None, :, None]) / (
        sigma[None, :, None] + EPSILON
    )
    if group in ("G1", "G2"):
        clipped = np.clip(standardized, -CLIP_LIMIT, CLIP_LIMIT).astype(np.float32)
        r = (
            clipped - clipped.mean(axis=2, keepdims=True)
            if group == "G1"
            else clipped
        )
        transform_stats = core.clip_statistics(standardized, labels)
    elif group == "G3":
        r = np.arcsinh(standardized).astype(np.float32)
        transform_stats = no_hard_clip_statistics()
    else:
        raise ValueError(f"unknown group: {group}")

    if not np.all(np.isfinite(r)):
        raise FloatingPointError(f"{group} residual contains NaN or infinity")
    if group == "G1":
        maximum_mean = float(np.max(np.abs(np.mean(r, axis=2, dtype=np.float64))))
        maximum_signal = float(np.max(np.abs(r)))
        tolerance = max(
            1e-5,
            64.0 * float(np.finfo(np.float32).eps) * max(1.0, maximum_signal),
        )
        if maximum_mean > tolerance:
            raise AssertionError(
                f"G1 residual centering failed: max_mean={maximum_mean}, tolerance={tolerance}"
            )
    absolute = np.abs(r).astype(np.float32, copy=False)
    delta = np.diff(r, axis=2, prepend=r[:, :, :1]).astype(np.float32, copy=False)
    if not np.all(delta[:, :, 0] == 0):
        raise AssertionError("delta first sample must be exactly zero")
    features = np.concatenate([r, absolute, delta], axis=1)
    if features.shape[1:] != (27, 128):
        raise AssertionError(f"unexpected G1/G2/G3 feature shape: {features.shape}")
    return np.ascontiguousarray(features.transpose(0, 2, 1)), transform_stats


def patch_core() -> None:
    """Install the GRU/G1-G3 contract while reusing the audited A-D trainer."""
    core.GROUPS = GROUPS
    core.GROUP_CONFIG = GROUP_CONFIG
    core.load_frozen_nbm = load_frozen_gru_nbm
    core.reconstruction_error = reconstruction_error_gru
    core.build_abcd_features = build_g123_features
    core.no_clip_statistics = no_hard_clip_statistics
    core.CLIP_STATISTICS_SEED_MODE = "paired_nbm"
    core.CLIP_STATISTICS_EQUIVALENT_GROUP_PAIRS = (("G1", "G2"),)


def expected_jobs(seeds: tuple[int, ...]) -> list[tuple[int, str, int]]:
    return [(fold, group, seed) for fold in FOLDS for group in GROUPS for seed in seeds]


def run_seal(args: argparse.Namespace, seeds: tuple[int, ...]) -> None:
    root = args.output_root.resolve()
    entries: list[dict[str, Any]] = []
    for fold, group, seed in expected_jobs(seeds):
        directory = core.job_directory(root, fold, group, seed)
        frozen_path = directory / "frozen_validation.json"
        checkpoint = directory / "checkpoints" / "tcn.pt"
        done_path = directory / "DONE_TRAIN.json"
        for path in (frozen_path, checkpoint, done_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"training job incomplete: {core.job_id(fold, group, seed)}: {path}"
                )
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen.get("test_roles_accessed") is not False:
            raise AssertionError(f"premature test access: {frozen['job_id']}")
        if frozen.get("group_config") != GROUP_CONFIG[group]:
            raise AssertionError(f"feature contract mismatch: {frozen['job_id']}")
        if int(frozen["tcn_seed"]) != seed or int(frozen["nbm"]["seed"]) != seed:
            raise AssertionError(f"NBM/TCN seed pairing failed: {frozen['job_id']}")
        checkpoint_sha256 = core.sha256_file(checkpoint)
        if checkpoint_sha256 != frozen["checkpoint_sha256"]:
            raise AssertionError(f"TCN checkpoint changed: {frozen['job_id']}")
        training = frozen["training"]
        if (
            int(training["maximum_epochs"]) != args.tcn_max_epochs
            or int(training["patience"]) != args.tcn_patience
        ):
            raise AssertionError(f"TCN budget mismatch: {frozen['job_id']}")
        entries.append(
            {
                "job_id": frozen["job_id"],
                "fold": fold,
                "group": group,
                "method": group,
                "tcn_seed": seed,
                "nbm_seed": int(frozen["nbm"]["seed"]),
                "threshold": float(frozen["threshold"]),
                "checkpoint_sha256": checkpoint_sha256,
                "frozen_validation_sha256": core.sha256_file(frozen_path),
                "tcn_initial_state_sha256": training["initial_state_sha256"],
                "nbm_checkpoint_sha256": frozen["nbm"]["checkpoint_sha256"],
                "nbm_frozen_sha256": frozen["nbm"]["frozen_json_sha256"],
                "done_nbm_sha256": frozen["nbm"]["done_nbm_sha256"],
                "scaler_sha256": frozen["nbm"]["scaler_role4_sha256"],
                "scientific_data_sha256": frozen["nbm"].get("scientific_data_sha256"),
                "pos_weight": float(training["pos_weight"]),
                "tcn_max_epochs": int(training["maximum_epochs"]),
                "tcn_patience": int(training["patience"]),
            }
        )

    for fold in FOLDS:
        fold_entries = [entry for entry in entries if entry["fold"] == fold]
        if len({entry["pos_weight"] for entry in fold_entries}) != 1:
            raise AssertionError(f"fold {fold} pos_weight mismatch")
        for seed in seeds:
            paired = [
                entry
                for entry in fold_entries
                if entry["tcn_seed"] == seed and entry["nbm_seed"] == seed
            ]
            if len(paired) != len(GROUPS):
                raise AssertionError(f"fold {fold}, seed {seed}: incomplete G1/G2/G3 pair")
            for key in (
                "tcn_initial_state_sha256",
                "nbm_checkpoint_sha256",
                "nbm_frozen_sha256",
                "done_nbm_sha256",
                "scaler_sha256",
                "pos_weight",
            ):
                if len({entry[key] for entry in paired}) != 1:
                    raise AssertionError(f"fold {fold}, seed {seed} mismatch: {key}")

    rows_by_fold = {
        fold: core.load_fold_rows(args.data_dir.resolve(), fold) for fold in FOLDS
    }
    source_audit = audit_protocol_dynamic(
        args.data_dir.resolve(), rows_by_fold, 64, 128, 64
    )
    test_manifest = build_test_data_manifest(args.data_dir.resolve(), rows_by_fold)
    barrier = {
        "barrier_schema": "strict_test_barrier.v2",
        "status": "all_G1_G2_G3_classifiers_and_thresholds_frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "folds": list(FOLDS),
        "methods": list(GROUPS),
        "groups": list(GROUPS),
        "nbm_seeds": list(seeds),
        "tcn_seeds": list(seeds),
        "job_count": len(entries),
        "strict_test_gate": "roles 0/1 may be accessed only after this global barrier",
        "source_audit": source_audit,
        "test_data_manifest": test_manifest,
        "jobs": entries,
    }
    barrier["barrier_id"] = stable_json_hash(barrier_identity_payload(barrier))
    core.write_json(root / "TRAINING_BARRIER.json", barrier)
    core.write_json(
        root / "experiment_config.json",
        {
            "experiment": "frozen_GRU_v1_NBM_residual_G1_G2_G3",
            "groups": GROUP_CONFIG,
            "input": "F=[r,abs(r),delta_t(r)] in [B,27,128]",
            "nbm_retrained": False,
            "nbm": "paired frozen GRU-v1; role4/5; max300/pat20",
            "paired_seeds": list(seeds),
            "tcn": "same RepresentationTCNM; max10/pat2; AdamW lr1e-3; weighted BCE",
            "roles": {str(key): value for key, value in core.ROLES.items()},
            "threshold": "roles 2/3 balanced accuracy; ties FoG F1 then higher threshold",
            "barrier_schema": barrier["barrier_schema"],
            "barrier_id": barrier["barrier_id"],
            "test_data_manifest_sha256": test_manifest["sha256"],
            "primary_summary": (
                "within each seed macro-average 3 folds, then mean and population SD "
                "across seeds 0,52,161"
            ),
        },
    )
    print(f"GLOBAL TRAINING BARRIER SEALED jobs={len(entries)}", flush=True)


def sealed_job(args: argparse.Namespace) -> dict[str, Any]:
    barrier_path = args.output_root.resolve() / "TRAINING_BARRIER.json"
    if not barrier_path.is_file():
        raise FileNotFoundError("TRAINING_BARRIER.json missing; test access forbidden")
    barrier = load_and_validate_barrier(barrier_path)
    target = core.job_id(args.fold, args.group, args.tcn_seed)
    matches = [entry for entry in barrier["jobs"] if entry["job_id"] == target]
    if len(matches) != 1:
        raise AssertionError(f"job not sealed: {target}")
    sealed = dict(matches[0])
    sealed["barrier_schema"] = barrier["barrier_schema"]
    sealed["barrier_id"] = barrier["barrier_id"]
    sealed["test_data_manifest_sha256"] = barrier["test_data_manifest"]["sha256"]
    return sealed


def current_test_manifest(args: argparse.Namespace) -> dict[str, Any]:
    rows = {fold: core.load_fold_rows(args.data_dir.resolve(), fold) for fold in FOLDS}
    return build_test_data_manifest(args.data_dir.resolve(), rows)


def run_evaluate(args: argparse.Namespace, device: torch.device) -> None:
    core.require_job_args(args)
    sealed = sealed_job(args)
    directory = core.job_directory(
        args.output_root.resolve(), args.fold, args.group, args.tcn_seed
    )
    manifest = current_test_manifest(args)
    if manifest["sha256"] != sealed["test_data_manifest_sha256"]:
        raise AssertionError("permanent-test data changed after global seal")
    done_path = directory / "DONE_TEST.json"
    if done_path.exists() and not args.overwrite:
        validate_completed_test_artifacts(directory, sealed)
        print(f"SKIP completed test job: {done_path}", flush=True)
        return

    checkpoint = directory / "checkpoints" / "tcn.pt"
    frozen_path = directory / "frozen_validation.json"
    if core.sha256_file(checkpoint) != sealed["checkpoint_sha256"]:
        raise AssertionError("sealed TCN checkpoint changed")
    if core.sha256_file(frozen_path) != sealed["frozen_validation_sha256"]:
        raise AssertionError("sealed validation artifact changed")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if float(frozen["threshold"]) != float(sealed["threshold"]):
        raise AssertionError("sealed threshold changed")

    # Roles 0/1 are first materialized only after the global seal above.
    records, rows = core.load_records_and_rows(args.data_dir.resolve(), args.fold)
    test_rows = rows.take_role(0, 1)
    nbm, scaler, bias, sigma, nbm_manifest = load_frozen_gru_nbm(
        args.nbm_source_root, args.fold, device
    )
    for manifest_key, sealed_key in (
        ("checkpoint_sha256", "nbm_checkpoint_sha256"),
        ("frozen_json_sha256", "nbm_frozen_sha256"),
        ("done_nbm_sha256", "done_nbm_sha256"),
        ("scaler_role4_sha256", "scaler_sha256"),
    ):
        if nbm_manifest[manifest_key] != sealed[sealed_key]:
            raise AssertionError(f"sealed NBM artifact changed: {manifest_key}")
    if int(nbm_manifest["seed"]) != int(args.tcn_seed):
        raise AssertionError("test NBM/TCN seed pairing failed")
    error = reconstruction_error_gru(
        nbm, scaler, core.raw_windows(records, test_rows), device
    )
    test_x, test_transform = build_g123_features(
        error, test_rows.label, args.group, bias, sigma
    )
    del nbm, error
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = core.RepresentationTCNM(27).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    test_true, test_prob = core.classifier_predict(
        model, test_x, test_rows.label, device
    )
    threshold = float(sealed["threshold"])
    test_metrics = binary_metrics(test_true, test_prob, threshold)
    test_pred = (test_prob >= threshold).astype(np.int8)
    by_subject = {}
    for subject in core.SUBJECTS:
        mask = test_rows.subject_id == subject
        by_subject[subject] = binary_metrics(
            test_true[mask], test_prob[mask], threshold
        )
    split_transform = {
        **frozen["clip_statistics"],
        "roles_0_1_test": test_transform,
    }
    combined_transform = core.combine_clip_statistics(split_transform.values())
    result = {
        "job_id": sealed["job_id"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "group": args.group,
        "group_config": GROUP_CONFIG[args.group],
        "tcn_seed": args.tcn_seed,
        "nbm_seed": nbm_manifest["seed"],
        "threshold": threshold,
        "threshold_source_roles": [2, 3],
        "strict_global_test_barrier_verified": True,
        "test_roles": [0, 1],
        "barrier_schema": sealed["barrier_schema"],
        "barrier_id": sealed["barrier_id"],
        "test_data_manifest_sha256": sealed["test_data_manifest_sha256"],
        "test": test_metrics,
        "test_by_subject": by_subject,
        "clip_statistics": {
            **split_transform,
            "all_classifier_roles_6_7_2_3_0_1": combined_transform,
        },
        "test_feature_diagnostics": core.residual_diagnostics(test_x, test_true),
        "nbm": nbm_manifest,
        "tcn_checkpoint_sha256": sealed["checkpoint_sha256"],
        "nbm_checkpoint_sha256": sealed["nbm_checkpoint_sha256"],
        "nbm_frozen_sha256": sealed["nbm_frozen_sha256"],
        "done_nbm_sha256": sealed["done_nbm_sha256"],
        "scaler_sha256": sealed["scaler_sha256"],
        "frozen_validation_sha256": sealed["frozen_validation_sha256"],
        "scientific_data_sha256": sealed.get("scientific_data_sha256"),
    }
    metrics_path = directory / "metrics.json"
    predictions_path = directory / "test_predictions.csv"
    probabilities_path = directory / "test_probabilities.npz"
    core.write_json(metrics_path, result)
    core.write_csv(
        predictions_path,
        [
            {
                "fold": args.fold,
                "group": args.group,
                "tcn_seed": args.tcn_seed,
                "nbm_seed": nbm_manifest["seed"],
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
        probabilities_path,
        y_true=test_true,
        y_prob=test_prob,
        y_pred=test_pred,
        subject_id=test_rows.subject_id,
        window_id=test_rows.window_id,
        threshold=np.asarray(threshold),
    )
    done_payload = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": sealed["job_id"],
        "test": test_metrics,
        "threshold": threshold,
        "barrier_id": sealed["barrier_id"],
        "test_data_manifest_sha256": sealed["test_data_manifest_sha256"],
        "tcn_checkpoint_sha256": sealed["checkpoint_sha256"],
        "nbm_checkpoint_sha256": sealed["nbm_checkpoint_sha256"],
        "nbm_frozen_sha256": sealed["nbm_frozen_sha256"],
        "done_nbm_sha256": sealed["done_nbm_sha256"],
        "scaler_sha256": sealed["scaler_sha256"],
        "frozen_validation_sha256": sealed["frozen_validation_sha256"],
        "scientific_data_sha256": sealed.get("scientific_data_sha256"),
        "metrics_sha256": core.sha256_file(metrics_path),
        "predictions_sha256": core.sha256_file(predictions_path),
        "probabilities_sha256": core.sha256_file(probabilities_path),
    }
    core.write_json(done_path, done_payload)
    validate_completed_test_artifacts(directory, sealed)
    print(
        f"TEST COMPLETE {sealed['job_id']} acc={test_metrics['accuracy']:.6f} "
        f"recall={test_metrics['sensitivity']:.6f} "
        f"specificity={test_metrics['specificity']:.6f} "
        f"pr_auc={test_metrics['auprc']:.6f}",
        flush=True,
    )


def run_aggregate(args: argparse.Namespace, seeds: tuple[int, ...]) -> None:
    barrier_path = args.output_root.resolve() / "TRAINING_BARRIER.json"
    if not barrier_path.is_file():
        raise FileNotFoundError("cannot aggregate without TRAINING_BARRIER.json")
    load_and_validate_barrier(barrier_path)
    manifest = current_test_manifest(args)
    for fold, group, seed in expected_jobs(seeds):
        shadow = argparse.Namespace(**vars(args))
        shadow.fold, shadow.group, shadow.tcn_seed = fold, group, seed
        sealed = sealed_job(shadow)
        if manifest["sha256"] != sealed["test_data_manifest_sha256"]:
            raise AssertionError("permanent-test data changed before aggregation")
        directory = core.job_directory(args.output_root.resolve(), fold, group, seed)
        validate_completed_test_artifacts(directory, sealed)
    core.run_aggregate(args)


def main() -> None:
    patch_core()
    args = parse_args()
    seeds = parse_contract(args)
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    if args.stage == "seal":
        run_seal(args, seeds)
        return
    if args.stage == "aggregate":
        run_aggregate(args, seeds)
        return
    device = core.resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    if args.stage == "train":
        core.run_train(args, device)
    else:
        run_evaluate(args, device)


if __name__ == "__main__":
    main()
