#!/usr/bin/env python3
"""Strict A-D residual-calibration experiment using frozen Conv-TCN NBMs.

Stages
------
train
    Train one TCN for one (fold, group, seed) using roles 6/7 and select the
    checkpoint/threshold using roles 2/3.  Roles 0/1 are never requested.
seal
    Verify that every A-D classifier for every fold and seed is frozen, then
    write the global TRAINING_BARRIER.json.
evaluate
    After the barrier exists, evaluate one frozen job on roles 0/1.
aggregate
    Verify every test job and produce fold/seed/subject and clip-rate tables.

The NBM is never retrained here.  Every group and TCN seed in one outer fold
loads the same frozen NBM checkpoint, scaler, b and sigma from the preceding
Conv-TCN NBM experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

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
    classifier_loader,
    classifier_predict,
    load_fold_rows,
    raw_windows,
    residual_diagnostics,
    set_seed,
    write_csv,
    write_json,
)
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    ConvTCNAutoencoderNBM,
    RepresentationTCNM,
    atomic_torch_save,
    centered_scaled_bct,
    reconstruct,
    validation_classifier_loss,
)

GROUPS = ("A", "B", "C", "D")
FOLDS = (0, 1, 2)
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
EPSILON = 1e-6
CLIP_LIMIT = 12.0
CHANNEL_NAMES = (
    "ankle_acc_forward",
    "ankle_acc_vertical",
    "ankle_acc_lateral",
    "thigh_acc_forward",
    "thigh_acc_vertical",
    "thigh_acc_lateral",
    "trunk_acc_forward",
    "trunk_acc_vertical",
    "trunk_acc_lateral",
)

GROUP_CONFIG: dict[str, dict[str, Any]] = {
    "A": {
        "name": "location_scale_calibration",
        "uses_b": True,
        "uses_sigma": True,
        "clip": [-12.0, 12.0],
        "window_centering": False,
        "formula": "r=clip((e-b)/(sigma+1e-6),-12,12)",
    },
    "B": {
        "name": "location_scale_calibration_plus_window_centering",
        "uses_b": True,
        "uses_sigma": True,
        "clip": [-12.0, 12.0],
        "window_centering": True,
        "formula": "q=clip((e-b)/(sigma+1e-6),-12,12); r=q-mean_t(q)",
    },
    "C": {
        "name": "scale_calibration_plus_window_centering",
        "uses_b": False,
        "uses_sigma": True,
        "clip": [-12.0, 12.0],
        "window_centering": True,
        "formula": "q=clip(e/(sigma+1e-6),-12,12); r=q-mean_t(q)",
    },
    "D": {
        "name": "window_centering_only",
        "uses_b": False,
        "uses_sigma": False,
        "clip": None,
        "window_centering": True,
        "formula": "r=e-mean_t(e); no clipping",
    },
}


def parse_seed_list(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"invalid unique seed list: {value}")
    return seeds


def parse_group_list(value: str) -> tuple[str, ...]:
    groups = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not groups or len(groups) != len(set(groups)):
        raise ValueError(f"invalid unique group list: {value}")
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown:
        raise ValueError(f"unknown residual-calibration groups: {unknown}")
    return groups


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
        / "daphnet_processed_NBM_conv_tcn_residual_repr_compare_seed20260807",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_residual_calibration_ABCD_3seed_seed20260807",
    )
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument(
        "--groups",
        default=",".join(GROUPS),
        help="Comma-separated group set required by seal/aggregate (default: A,B,C,D).",
    )
    parser.add_argument("--tcn-seed", type=int)
    parser.add_argument("--tcn-seeds", default="20260807,20260808,20260809")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tcn-max-epochs", type=int, default=30)
    parser.add_argument("--tcn-patience", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    return device


def require_job_args(args: argparse.Namespace) -> None:
    if args.fold is None or args.group is None or args.tcn_seed is None:
        raise ValueError(f"--fold, --group and --tcn-seed are required for {args.stage}")
    if args.group not in parse_group_list(args.groups):
        raise ValueError(f"job group {args.group} is not enabled by --groups={args.groups}")


def job_id(fold: int, group: str, seed: int) -> str:
    return f"fold{fold}_group{group}_seed{seed}"


def job_directory(root: Path, fold: int, group: str, seed: int) -> Path:
    return root / "runs" / f"fold_{fold}" / f"group_{group}" / f"seed_{seed}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def load_records_and_rows(data_dir: Path, fold: int) -> tuple[dict[str, Any], Any]:
    dataset = DaphnetDataset.load(data_dir.resolve())
    records = {record.record_id: record for record in dataset.records}
    rows = load_fold_rows(data_dir.resolve(), fold)
    expected = np.isin(rows.role, [1, 3, 7]).astype(np.int8)
    if not np.array_equal(rows.label, expected):
        raise AssertionError(f"role/label mismatch in fold {fold}")
    return records, rows


def load_frozen_nbm(
    source_root: Path,
    fold: int,
    device: torch.device,
) -> tuple[ConvTCNAutoencoderNBM, RobustScaler, np.ndarray, np.ndarray, dict[str, Any]]:
    fold_dir = source_root.resolve() / f"fold_{fold}"
    checkpoint = fold_dir / "checkpoints" / "conv_tcn_nbm_best.pt"
    frozen_path = fold_dir / "nbm_frozen.json"
    if not checkpoint.exists() or not frozen_path.exists():
        raise FileNotFoundError(f"frozen NBM artifacts missing under {fold_dir}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    scaler_payload = frozen["scaler"]
    scaler = RobustScaler(
        median=np.asarray(scaler_payload["median"], dtype=np.float32),
        iqr=np.asarray(scaler_payload["iqr"], dtype=np.float32),
        epsilon=float(scaler_payload.get("epsilon", EPSILON)),
    )
    calibration = frozen["calibration"]
    bias = np.asarray(calibration["bias"], dtype=np.float32)
    sigma = np.asarray(calibration["sigma"], dtype=np.float32)
    if bias.shape != (9,) or sigma.shape != (9,) or np.any(sigma < 0.05):
        raise AssertionError("invalid frozen C1 calibration")
    dropout = float(frozen["training"]["architecture"].get("dropout", 0.10))
    model = ConvTCNAutoencoderNBM(dropout=dropout).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()
    manifest = {
        "fold": fold,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "frozen_json": str(frozen_path.resolve()),
        "frozen_json_sha256": sha256_file(frozen_path),
        "best_epoch": int(frozen["training"]["best_epoch"]),
        "best_validation_huber": float(frozen["training"]["best_validation_huber"]),
        "calibration_role": 5,
        "scaler_role": 4,
    }
    return model, scaler, bias, sigma, manifest


def reconstruction_error(
    model: ConvTCNAutoencoderNBM,
    scaler: RobustScaler,
    raw: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x = centered_scaled_bct(scaler, raw)
    x_hat = reconstruct(model, x, device)
    error = (x - x_hat).astype(np.float32, copy=False)
    if error.ndim != 3 or error.shape[1:] != (9, 128):
        raise AssertionError(f"unexpected reconstruction error shape: {error.shape}")
    return np.ascontiguousarray(error)


def _clip_count(mask: np.ndarray) -> dict[str, Any]:
    points = int(mask.size)
    clipped = int(np.count_nonzero(mask))
    return {
        "points": points,
        "clipped": clipped,
        "rate": float(clipped / points) if points else None,
    }


def clip_statistics(unclipped: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    mask = np.abs(unclipped) > CLIP_LIMIT
    if mask.shape[0] != labels.shape[0] or mask.shape[1:] != (9, 128):
        raise ValueError("clip-statistic input mismatch")
    by_class = {}
    for value, name in ((0, "nonfog"), (1, "fog")):
        by_class[name] = _clip_count(mask[labels == value])
    per_channel = []
    for channel in range(9):
        channel_entry = {
            "channel": channel,
            "channel_name": CHANNEL_NAMES[channel],
            "overall": _clip_count(mask[:, channel, :]),
            "nonfog": _clip_count(mask[labels == 0, channel, :]),
            "fog": _clip_count(mask[labels == 1, channel, :]),
        }
        per_channel.append(channel_entry)
    return {
        "applicable": True,
        "definition": "abs(unclipped_calibrated_residual)>12 before np.clip",
        "overall": _clip_count(mask),
        "nonfog": by_class["nonfog"],
        "fog": by_class["fog"],
        "per_channel": per_channel,
    }


def no_clip_statistics() -> dict[str, Any]:
    return {
        "applicable": False,
        "reason": "group D performs no clipping",
    }


def combine_clip_statistics(stats: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(stats)
    if not items or not all(item.get("applicable", False) for item in items):
        return no_clip_statistics()

    def combine_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        records = list(records)
        points = sum(int(record["points"]) for record in records)
        clipped = sum(int(record["clipped"]) for record in records)
        return {
            "points": points,
            "clipped": clipped,
            "rate": float(clipped / points) if points else None,
        }

    per_channel = []
    for channel in range(9):
        per_channel.append(
            {
                "channel": channel,
                "channel_name": CHANNEL_NAMES[channel],
                "overall": combine_records(item["per_channel"][channel]["overall"] for item in items),
                "nonfog": combine_records(item["per_channel"][channel]["nonfog"] for item in items),
                "fog": combine_records(item["per_channel"][channel]["fog"] for item in items),
            }
        )
    return {
        "applicable": True,
        "definition": items[0]["definition"],
        "overall": combine_records(item["overall"] for item in items),
        "nonfog": combine_records(item["nonfog"] for item in items),
        "fog": combine_records(item["fog"] for item in items),
        "per_channel": per_channel,
    }


def build_abcd_features(
    error_bct: np.ndarray,
    labels: np.ndarray,
    group: str,
    bias: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return F=[r,abs(r),delta(r)] in [N,T,27] plus clip statistics."""
    error = np.asarray(error_bct, dtype=np.float32)
    if error.ndim != 3 or error.shape[1:] != (9, 128):
        raise ValueError(f"expected error [N,9,128], got {error.shape}")
    if group in ("A", "B"):
        unclipped = (error - bias[None, :, None]) / (sigma[None, :, None] + EPSILON)
        clipped = np.clip(unclipped, -CLIP_LIMIT, CLIP_LIMIT).astype(np.float32)
        r = clipped if group == "A" else clipped - clipped.mean(axis=2, keepdims=True)
        clip_stats = clip_statistics(unclipped, labels)
    elif group == "C":
        unclipped = error / (sigma[None, :, None] + EPSILON)
        clipped = np.clip(unclipped, -CLIP_LIMIT, CLIP_LIMIT).astype(np.float32)
        r = clipped - clipped.mean(axis=2, keepdims=True)
        clip_stats = clip_statistics(unclipped, labels)
    elif group == "D":
        r = error - error.mean(axis=2, keepdims=True)
        clip_stats = no_clip_statistics()
    else:
        raise ValueError(f"unknown group: {group}")
    r = np.asarray(r, dtype=np.float32)
    if GROUP_CONFIG[group]["window_centering"]:
        maximum_mean = float(np.max(np.abs(r.mean(axis=2))))
        if maximum_mean > 2e-6:
            raise AssertionError(f"group {group} centering failed: {maximum_mean}")
    absolute = np.abs(r).astype(np.float32, copy=False)
    delta = np.diff(r, axis=2, prepend=r[:, :, :1]).astype(np.float32, copy=False)
    if not np.all(delta[:, :, 0] == 0):
        raise AssertionError("delta first sample is not zero")
    features_bct = np.concatenate([r, absolute, delta], axis=1)
    if features_bct.shape[1:] != (27, 128):
        raise AssertionError(f"unexpected A-D feature shape: {features_bct.shape}")
    return np.ascontiguousarray(features_bct.transpose(0, 2, 1)), clip_stats


def train_tcn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    output_dir: Path,
    device: torch.device,
    seed: int,
    num_workers: int,
    max_epochs: int,
    patience: int,
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(seed)
    model = RepresentationTCNM(27).to(device)
    initial_hash = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_pos = int(np.sum(train_y == 1))
    n_neg = int(np.sum(train_y == 0))
    if not n_pos or not n_neg:
        raise ValueError("roles 6/7 must contain both classes")
    pos_weight = n_neg / n_pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    loader = classifier_loader(train_x, train_y, True, seed, num_workers)
    checkpoint = output_dir / "checkpoints" / "tcn.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_pr = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite TCN gradient")
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
        train_bce = total / count
        validation_bce = validation_classifier_loss(
            model, validation_x, validation_y, criterion, device
        )
        val_true, val_prob = classifier_predict(model, validation_x, validation_y, device)
        validation_pr_auc = float(average_precision_score(val_true, val_prob))
        improved = validation_pr_auc > best_pr + 1e-10
        history.append(
            {
                "epoch": epoch,
                "train_weighted_bce": train_bce,
                "validation_weighted_bce": validation_bce,
                "validation_pr_auc": validation_pr_auc,
                "improved": improved,
            }
        )
        if improved:
            best_pr = validation_pr_auc
            best_epoch = epoch
            stale = 0
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_pr_auc": validation_pr_auc,
                    "seed": seed,
                    "input_channels": 27,
                },
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"TCN epoch={epoch:02d} train={train_bce:.7f} val={validation_bce:.7f} "
            f"val_pr={validation_pr_auc:.7f} stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    write_csv(output_dir / "logs" / "tcn_history.csv", history)
    return model, {
        "seed": seed,
        "input_channels": 27,
        "initial_state_sha256": initial_hash,
        "maximum_epochs": max_epochs,
        "patience": patience,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_pr_auc": best_pr,
        "pos_weight": pos_weight,
        "n_nonfog_role6": n_neg,
        "n_fog_role7": n_pos,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer": "AdamW(lr=0.001,weight_decay=0.0001)",
        "loss": "BCEWithLogitsLoss(pos_weight=N_role6/N_role7)",
        "history": history,
    }


def run_train(args: argparse.Namespace, device: torch.device) -> None:
    require_job_args(args)
    output_root = args.output_root.resolve()
    output_dir = job_directory(output_root, args.fold, args.group, args.tcn_seed)
    done = output_dir / "DONE_TRAIN.json"
    if done.exists() and not args.overwrite:
        print(f"SKIP frozen training job: {done}", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    records, rows = load_records_and_rows(args.data_dir.resolve(), args.fold)
    role67 = rows.take_role(6, 7)
    role23 = rows.take_role(2, 3)
    nbm, scaler, bias, sigma, nbm_manifest = load_frozen_nbm(
        args.nbm_source_root, args.fold, device
    )
    train_error = reconstruction_error(
        nbm, scaler, raw_windows(records, role67), device
    )
    validation_error = reconstruction_error(
        nbm, scaler, raw_windows(records, role23), device
    )
    train_x, train_clip = build_abcd_features(
        train_error, role67.label, args.group, bias, sigma
    )
    validation_x, validation_clip = build_abcd_features(
        validation_error, role23.label, args.group, bias, sigma
    )
    del nbm, train_error, validation_error
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model, training = train_tcn(
        train_x,
        role67.label,
        validation_x,
        role23.label,
        output_dir,
        device,
        args.tcn_seed,
        args.num_workers,
        args.tcn_max_epochs,
        args.tcn_patience,
    )
    val_true, val_prob = classifier_predict(model, validation_x, role23.label, device)
    threshold, validation_metrics = choose_document_threshold(val_true, val_prob)
    checkpoint = output_dir / "checkpoints" / "tcn.pt"
    frozen_payload = {
        "job_id": job_id(args.fold, args.group, args.tcn_seed),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "group": args.group,
        "group_config": GROUP_CONFIG[args.group],
        "tcn_seed": args.tcn_seed,
        "input_shape": ["B", 27, 128],
        "feature_formula": "F=concat(r,abs(r),delta_t(r)); delta_r[0]=0",
        "nbm": nbm_manifest,
        "c1_calibration": {
            "bias": bias.astype(float).tolist(),
            "sigma": sigma.astype(float).tolist(),
            "epsilon": EPSILON,
            "sigma_floor": 0.05,
        },
        "roles": {
            "classifier_train": [6, 7],
            "classifier_validation": [2, 3],
            "permanent_test_not_accessed": [0, 1],
        },
        "test_roles_accessed": False,
        "training": {key: value for key, value in training.items() if key != "history"},
        "threshold": threshold,
        "threshold_source_roles": [2, 3],
        "threshold_rule": "max balanced accuracy; ties FoG F1 then higher threshold; 0.05..0.95 step 0.01",
        "validation": validation_metrics,
        "clip_statistics": {
            "roles_6_7_train": train_clip,
            "roles_2_3_validation": validation_clip,
        },
        "feature_diagnostics": {
            "roles_6_7_train": residual_diagnostics(train_x, role67.label),
            "roles_2_3_validation": residual_diagnostics(validation_x, role23.label),
        },
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    write_json(output_dir / "frozen_validation.json", frozen_payload)
    write_json(
        done,
        {
            "status": "frozen",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": frozen_payload["job_id"],
            "threshold": threshold,
            "checkpoint_sha256": frozen_payload["checkpoint_sha256"],
            "test_roles_accessed": False,
        },
    )
    print(
        f"TRAIN FROZEN {frozen_payload['job_id']} threshold={threshold:.2f} "
        f"best_epoch={training['best_epoch']}",
        flush=True,
    )


def expected_jobs(
    groups: tuple[str, ...], seeds: tuple[int, ...]
) -> list[tuple[int, str, int]]:
    return [(fold, group, seed) for fold in FOLDS for group in groups for seed in seeds]


def run_seal(args: argparse.Namespace) -> None:
    seeds = parse_seed_list(args.tcn_seeds)
    groups = parse_group_list(args.groups)
    output_root = args.output_root.resolve()
    entries = []
    for fold, group, seed in expected_jobs(groups, seeds):
        directory = job_directory(output_root, fold, group, seed)
        done = directory / "DONE_TRAIN.json"
        frozen_path = directory / "frozen_validation.json"
        checkpoint = directory / "checkpoints" / "tcn.pt"
        if not done.exists() or not frozen_path.exists() or not checkpoint.exists():
            raise FileNotFoundError(f"training job incomplete: {job_id(fold, group, seed)}")
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        current_hash = sha256_file(checkpoint)
        if frozen.get("test_roles_accessed") is not False:
            raise AssertionError(f"test role accessed before barrier: {frozen['job_id']}")
        if current_hash != frozen["checkpoint_sha256"]:
            raise AssertionError(f"checkpoint changed before barrier: {frozen['job_id']}")
        entries.append(
            {
                "job_id": frozen["job_id"],
                "fold": fold,
                "group": group,
                "tcn_seed": seed,
                "threshold": frozen["threshold"],
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": current_hash,
                "tcn_initial_state_sha256": frozen["training"]["initial_state_sha256"],
                "nbm_checkpoint_sha256": frozen["nbm"]["checkpoint_sha256"],
                "nbm_frozen_json_sha256": frozen["nbm"]["frozen_json_sha256"],
                "pos_weight": frozen["training"]["pos_weight"],
            }
        )
    for fold in FOLDS:
        fold_entries = [entry for entry in entries if entry["fold"] == fold]
        if len({entry["nbm_checkpoint_sha256"] for entry in fold_entries}) != 1:
            raise AssertionError(f"fold {fold} did not share one NBM checkpoint")
        if len({entry["nbm_frozen_json_sha256"] for entry in fold_entries}) != 1:
            raise AssertionError(f"fold {fold} did not share one calibration file")
        if len({entry["pos_weight"] for entry in fold_entries}) != 1:
            raise AssertionError(f"fold {fold} pos_weight mismatch")
        for seed in seeds:
            paired = [entry for entry in fold_entries if entry["tcn_seed"] == seed]
            if len({entry["tcn_initial_state_sha256"] for entry in paired}) != 1:
                raise AssertionError(f"fold {fold} seed {seed} TCN initializations differ")
    rows_by_fold = {
        fold: load_fold_rows(args.data_dir.resolve(), fold) for fold in FOLDS
    }
    dataset = DaphnetDataset.load(args.data_dir.resolve())
    records = {record.record_id: record for record in dataset.records}
    source_audit = audit_protocol(args.data_dir.resolve(), rows_by_fold, records)
    barrier = {
        "status": "all_classifiers_and_thresholds_frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "groups": list(groups),
        "folds": list(FOLDS),
        "tcn_seeds": list(seeds),
        "job_count": len(entries),
        "strict_test_gate": "roles 0/1 may be accessed only after this barrier",
        "nbm_source_root": str(args.nbm_source_root.resolve()),
        "source_audit": source_audit,
        "jobs": entries,
    }
    write_json(output_root / "TRAINING_BARRIER.json", barrier)
    write_json(
        output_root / "experiment_config.json",
        {
            "experiment": "residual_calibration_window_centering_" + "".join(groups),
            "groups": {group: GROUP_CONFIG[group] for group in groups},
            "input": "F=[r,abs(r),delta_t(r)] in [B,27,128]",
            "fixed_nbm_per_fold": True,
            "nbm_retrained": False,
            "tcn_seeds": list(seeds),
            "primary_summary": (
                "for each seed average the three folds, then report mean±population-std "
                "across the three seed-level macro values"
            ),
            "roles": {str(key): value for key, value in ROLES.items()},
            "threshold": "roles 2/3 balanced accuracy; ties FoG F1 then higher threshold",
            "test_gate": (
                f"all {len(entries)} checkpoints and thresholds frozen before any "
                "roles 0/1 access"
            ),
        },
    )
    print(f"TRAINING BARRIER SEALED jobs={len(entries)}", flush=True)


def barrier_job(args: argparse.Namespace) -> dict[str, Any]:
    barrier_path = args.output_root.resolve() / "TRAINING_BARRIER.json"
    if not barrier_path.exists():
        raise FileNotFoundError("TRAINING_BARRIER.json missing; test access is forbidden")
    barrier = json.loads(barrier_path.read_text(encoding="utf-8"))
    target = job_id(args.fold, args.group, args.tcn_seed)
    matches = [entry for entry in barrier["jobs"] if entry["job_id"] == target]
    if len(matches) != 1:
        raise AssertionError(f"job not sealed in barrier: {target}")
    return matches[0]


def run_evaluate(args: argparse.Namespace, device: torch.device) -> None:
    require_job_args(args)
    output_root = args.output_root.resolve()
    output_dir = job_directory(output_root, args.fold, args.group, args.tcn_seed)
    done = output_dir / "DONE_TEST.json"
    if done.exists() and not args.overwrite:
        print(f"SKIP completed test job: {done}", flush=True)
        return
    sealed = barrier_job(args)
    checkpoint = output_dir / "checkpoints" / "tcn.pt"
    if sha256_file(checkpoint) != sealed["checkpoint_sha256"]:
        raise AssertionError("sealed classifier checkpoint changed")
    frozen = json.loads((output_dir / "frozen_validation.json").read_text(encoding="utf-8"))
    if float(frozen["threshold"]) != float(sealed["threshold"]):
        raise AssertionError("sealed threshold changed")

    # Permanent test access starts only below this line, after global sealing.
    records, rows = load_records_and_rows(args.data_dir.resolve(), args.fold)
    test_rows = rows.take_role(0, 1)
    nbm, scaler, bias, sigma, nbm_manifest = load_frozen_nbm(
        args.nbm_source_root, args.fold, device
    )
    if nbm_manifest["checkpoint_sha256"] != sealed["nbm_checkpoint_sha256"]:
        raise AssertionError("sealed NBM checkpoint changed")
    test_error = reconstruction_error(
        nbm, scaler, raw_windows(records, test_rows), device
    )
    test_x, test_clip = build_abcd_features(
        test_error, test_rows.label, args.group, bias, sigma
    )
    model = RepresentationTCNM(27).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    test_true, test_prob = classifier_predict(model, test_x, test_rows.label, device)
    threshold = float(sealed["threshold"])
    test_metrics = binary_metrics(test_true, test_prob, threshold)
    test_pred = (test_prob >= threshold).astype(np.int8)
    subject_metrics = {}
    for subject in SUBJECTS:
        mask = test_rows.subject_id == subject
        subject_metrics[subject] = binary_metrics(
            test_true[mask], test_prob[mask], threshold
        )
    split_clip = {
        **frozen["clip_statistics"],
        "roles_0_1_test": test_clip,
    }
    all_clip = combine_clip_statistics(split_clip.values())
    result = {
        "job_id": sealed["job_id"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "group": args.group,
        "group_config": GROUP_CONFIG[args.group],
        "tcn_seed": args.tcn_seed,
        "threshold": threshold,
        "threshold_source_roles": [2, 3],
        "strict_global_test_barrier_verified": True,
        "test_roles": [0, 1],
        "test": test_metrics,
        "test_by_subject": subject_metrics,
        "clip_statistics": {
            **split_clip,
            "all_classifier_roles_6_7_2_3_0_1": all_clip,
        },
        "test_feature_diagnostics": residual_diagnostics(test_x, test_true),
        "nbm": nbm_manifest,
        "tcn_checkpoint_sha256": sealed["checkpoint_sha256"],
    }
    write_json(output_dir / "metrics.json", result)
    write_csv(
        output_dir / "test_predictions.csv",
        [
            {
                "fold": args.fold,
                "group": args.group,
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
        output_dir / "test_probabilities.npz",
        y_true=test_true,
        y_prob=test_prob,
        y_pred=test_pred,
        subject_id=test_rows.subject_id,
        window_id=test_rows.window_id,
        threshold=np.asarray(threshold),
    )
    write_json(
        done,
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": sealed["job_id"],
            "test": test_metrics,
        },
    )
    print(
        f"TEST COMPLETE {sealed['job_id']} acc={test_metrics['accuracy']:.6f} "
        f"recall={test_metrics['sensitivity']:.6f} "
        f"specificity={test_metrics['specificity']:.6f} "
        f"pr_auc={test_metrics['auprc']:.6f}",
        flush=True,
    )


def mean_std(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "n": int(len(array)),
    }


def clip_rate_row(group: str, fold: Any, split: str, stats: dict[str, Any]) -> dict[str, Any]:
    if not stats.get("applicable", False):
        return {
            "group": group,
            "fold": fold,
            "split": split,
            "applicable": False,
            "overall_clip_rate": None,
            "nonfog_clip_rate": None,
            "fog_clip_rate": None,
        }
    return {
        "group": group,
        "fold": fold,
        "split": split,
        "applicable": True,
        "overall_clip_rate": stats["overall"]["rate"],
        "nonfog_clip_rate": stats["nonfog"]["rate"],
        "fog_clip_rate": stats["fog"]["rate"],
        "overall_clipped": stats["overall"]["clipped"],
        "overall_points": stats["overall"]["points"],
        "nonfog_clipped": stats["nonfog"]["clipped"],
        "nonfog_points": stats["nonfog"]["points"],
        "fog_clipped": stats["fog"]["clipped"],
        "fog_points": stats["fog"]["points"],
    }


def run_aggregate(args: argparse.Namespace) -> None:
    seeds = parse_seed_list(args.tcn_seeds)
    groups = parse_group_list(args.groups)
    output_root = args.output_root.resolve()
    barrier_path = output_root / "TRAINING_BARRIER.json"
    if not barrier_path.exists():
        raise FileNotFoundError("cannot aggregate without TRAINING_BARRIER.json")
    results = []
    for fold, group, seed in expected_jobs(groups, seeds):
        directory = job_directory(output_root, fold, group, seed)
        metrics_path = directory / "metrics.json"
        done = directory / "DONE_TEST.json"
        if not metrics_path.exists() or not done.exists():
            raise FileNotFoundError(f"test job incomplete: {job_id(fold, group, seed)}")
        results.append(json.loads(metrics_path.read_text(encoding="utf-8")))

    run_rows = []
    for result in results:
        run_rows.append(
            {
                "fold": result["fold"],
                "group": result["group"],
                "tcn_seed": result["tcn_seed"],
                "threshold": result["threshold"],
                **{key: result["test"][key] for key in METRIC_KEYS},
                **{key: result["test"][key] for key in ("tn", "fp", "fn", "tp")},
            }
        )
    write_csv(output_root / f"run_metrics_{len(results)}.csv", run_rows)

    seed_macro_rows = []
    for group in groups:
        for seed in seeds:
            subset = [
                result for result in results
                if result["group"] == group and result["tcn_seed"] == seed
            ]
            if len(subset) != 3:
                raise AssertionError(f"expected 3 folds for {group}, seed {seed}")
            seed_macro_rows.append(
                {
                    "group": group,
                    "tcn_seed": seed,
                    **{
                        key: float(np.mean([result["test"][key] for result in subset]))
                        for key in METRIC_KEYS
                    },
                }
            )
    write_csv(output_root / "seed_macro_over_3folds.csv", seed_macro_rows)

    summary_rows = []
    summary_json: dict[str, Any] = {}
    for group in groups:
        group_rows = [row for row in seed_macro_rows if row["group"] == group]
        summary_json[group] = {}
        for key in METRIC_KEYS:
            stats = mean_std(row[key] for row in group_rows)
            summary_json[group][key] = stats
            summary_rows.append(
                {
                    "group": group,
                    "metric": key,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "n_seeds": stats["n"],
                }
            )
    write_csv(output_root / "group_summary_3seed_mean_std.csv", summary_rows)

    subject_rows = []
    subject_summary_json: dict[str, Any] = {}
    for group in groups:
        subject_summary_json[group] = {}
        for subject in SUBJECTS:
            seed_subject_rows = []
            for seed in seeds:
                subset = [
                    result for result in results
                    if result["group"] == group and result["tcn_seed"] == seed
                ]
                seed_subject_rows.append(
                    {
                        key: float(
                            np.mean(
                                [result["test_by_subject"][subject][key] for result in subset]
                            )
                        )
                        for key in METRIC_KEYS
                    }
                )
            subject_summary_json[group][subject] = {}
            row: dict[str, Any] = {"group": group, "subject_id": subject}
            for key in METRIC_KEYS:
                stats = mean_std(item[key] for item in seed_subject_rows)
                subject_summary_json[group][subject][key] = stats
                row[f"{key}_mean"] = stats["mean"]
                row[f"{key}_std"] = stats["std"]
            subject_rows.append(row)
    write_csv(output_root / "subject_metrics_3seed_mean_std.csv", subject_rows)

    clip_by_fold_group: dict[tuple[str, int], dict[str, Any]] = {}
    for group in groups:
        for fold in FOLDS:
            subset = [
                result for result in results
                if result["group"] == group and result["fold"] == fold
            ]
            reference = subset[0]["clip_statistics"]
            if any(result["clip_statistics"] != reference for result in subset[1:]):
                raise AssertionError(f"clip statistics depend on TCN seed for {group}, fold {fold}")
            clip_by_fold_group[(group, fold)] = reference
    for fold in FOLDS:
        if {"A", "B"}.issubset(groups) and (
            clip_by_fold_group[("A", fold)] != clip_by_fold_group[("B", fold)]
        ):
            raise AssertionError(f"A/B pre-centering clip rates differ in fold {fold}")

    split_names = (
        "roles_6_7_train",
        "roles_2_3_validation",
        "roles_0_1_test",
        "all_classifier_roles_6_7_2_3_0_1",
    )
    clip_rows = []
    channel_rows = []
    for (group, fold), payload in clip_by_fold_group.items():
        for split in split_names:
            stats = payload[split]
            clip_rows.append(clip_rate_row(group, fold, split, stats))
            if stats.get("applicable", False):
                for channel in stats["per_channel"]:
                    channel_rows.append(
                        {
                            "group": group,
                            "fold": fold,
                            "split": split,
                            "channel": channel["channel"],
                            "channel_name": channel["channel_name"],
                            "overall_clip_rate": channel["overall"]["rate"],
                            "nonfog_clip_rate": channel["nonfog"]["rate"],
                            "fog_clip_rate": channel["fog"]["rate"],
                        }
                    )
    clip_group_summary: dict[str, Any] = {}
    for group in groups:
        clip_group_summary[group] = {}
        for split in split_names:
            combined = combine_clip_statistics(
                clip_by_fold_group[(group, fold)][split] for fold in FOLDS
            )
            clip_group_summary[group][split] = combined
            clip_rows.append(clip_rate_row(group, "all_folds", split, combined))
    write_csv(output_root / "clip_rates_by_fold_split.csv", clip_rows)
    write_csv(output_root / "clip_rates_per_channel.csv", channel_rows)

    write_json(
        output_root / "summary.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "primary_group_metrics": summary_json,
            "primary_definition": (
                "within each TCN seed macro-average 3 folds; report mean and population "
                "standard deviation across 3 seeds"
            ),
            "subject_metrics": subject_summary_json,
            "clip_rates": clip_group_summary,
            "run_count": len(results),
            "groups": {group: GROUP_CONFIG[group] for group in groups},
            "tcn_seeds": list(seeds),
            "strict_global_test_barrier": True,
        },
    )
    write_json(
        output_root / "DONE.json",
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "run_count": len(results),
            "groups": list(groups),
            "tcn_seeds": list(seeds),
        },
    )
    print(json.dumps(summary_json, ensure_ascii=False, indent=2), flush=True)


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
    elif args.stage == "evaluate":
        run_evaluate(args, device)


if __name__ == "__main__":
    main()
