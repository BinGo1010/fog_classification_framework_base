#!/usr/bin/env python3
"""Strict RAW+TCN loss ablation using ordinary unweighted BCE.

The experiment reuses the exact role-4 RobustScaler artifacts from the
completed five-seed RAW+TCN experiment.  The protocol is held fixed:
three processed_NBM folds, roles 6/7 for classifier training, roles 2/3 for
checkpoint and threshold selection, the same TCN architecture, seeds and
initialization rule, AdamW,
batch size, epoch budget, patience, AP checkpoint rule, and balanced-accuracy
threshold rule.  The sole intended change is

    BCEWithLogitsLoss(pos_weight=N_nonfog/N_fog)

to

    BCEWithLogitsLoss().

All 15 classifiers and validation-selected thresholds are frozen before the
permanent role-0/1 test features are generated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.evaluation import binary_metrics
from scripts.run_daphnet_nbm300_c_vs_raw_ablation import (
    SUBJECTS,
    audit_protocol_dynamic,
    build_test_data_manifest,
    load_records_rows,
    load_scaler_only,
    make_features,
    paired_initialization,
    raw_windows_dynamic,
    sha256_file,
    stable_json_hash,
)
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    choose_document_threshold,
    classifier_loader,
    classifier_predict,
    residual_diagnostics,
    save_figure_bundle,
    set_seed,
    write_csv,
    write_json,
)
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    RepresentationTCNM,
)


FOLDS = (0, 1, 2)
SEEDS = (0, 52, 161, 5216, 52161)
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "auprc",
    "auroc",
)
DEFAULT_DATA = (
    REPO_ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed_NBM"
)
DEFAULT_SOURCE = (
    REPO_ROOT
    / "outputs"
    / "daphnet_tcn_nbm300_C_vs_raw_tcn_ep5pat2_seedset_0_52_161_5216_52161"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "daphnet_raw_tcn_unweighted_bce_ep5pat2_seedset_0_52_161_5216_52161"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weighted-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    return device


def job_dir(root: Path, fold: int, seed: int) -> Path:
    return root / "runs" / f"fold_{fold}" / "method_RAW" / f"seed_{seed}"


def job_id(fold: int, seed: int) -> str:
    return f"fold{fold}_methodRAW_unweightedBCE_seed{seed}"


def source_dir(source_root: Path, seed: int) -> Path:
    return source_root / "nbm_source" / f"seed_{seed}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_history(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    converted: list[dict[str, Any]] = []
    for row in rows:
        converted.append(
            {
                "epoch": int(row["epoch"]),
                "train_bce": float(row["train_bce"]),
                "validation_bce": float(row["validation_bce"]),
                "validation_ap": float(row["validation_ap"]),
                "improved": str(row["improved"]).lower() == "true",
            }
        )
    return converted


@torch.no_grad()
def validation_loss(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch_x, batch_y in classifier_loader(x, y, False, 0, 0):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        loss = criterion(model(batch_x), batch_y)
        total += float(loss) * len(batch_x)
        count += len(batch_x)
    return total / count


def train_unweighted_tcn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    directory: Path,
    device: torch.device,
    seed: int,
    num_workers: int,
    max_epochs: int,
    patience: int,
    initial_state: dict[str, torch.Tensor],
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(seed)
    model = RepresentationTCNM(9).to(device)
    model.load_state_dict(initial_state)
    # Match the original paired experiment's reset after loading the state.
    set_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    n_pos = int(np.sum(train_y == 1))
    n_neg = int(np.sum(train_y == 0))
    if not n_pos or not n_neg:
        raise ValueError("roles 6/7 must contain both classes")
    loader = classifier_loader(train_x, train_y, True, seed, num_workers)
    checkpoint = directory / "checkpoints" / "tcn.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_ap = -math.inf
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
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite RAW TCN gradient")
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
        train_bce = total / count
        val_bce = validation_loss(
            model, validation_x, validation_y, criterion, device
        )
        val_true, val_prob = classifier_predict(
            model, validation_x, validation_y, device
        )
        validation_ap = float(average_precision_score(val_true, val_prob))
        improved = validation_ap > best_ap + 1e-10
        history.append(
            {
                "epoch": epoch,
                "train_bce": train_bce,
                "validation_bce": val_bce,
                "validation_ap": validation_ap,
                "improved": improved,
            }
        )
        if improved:
            best_ap = validation_ap
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_ap": validation_ap,
                    "seed": seed,
                    "input_channels": 9,
                    "loss": "BCEWithLogitsLoss()",
                },
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"TRAIN fold_dir={directory.parent.parent.name} seed={seed} "
            f"epoch={epoch}/{max_epochs} train_bce={train_bce:.7f} "
            f"val_bce={val_bce:.7f} val_ap={validation_ap:.7f} "
            f"stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    write_csv(directory / "logs" / "tcn_history.csv", history)
    return model, {
        "seed": seed,
        "input_channels": 9,
        "maximum_epochs": max_epochs,
        "patience": patience,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_ap": best_ap,
        "n_nonfog_role6": n_neg,
        "n_fog_role7": n_pos,
        "class_ratio_nonfog_to_fog": n_neg / n_pos,
        "effective_pos_weight": 1.0,
        "optimizer": "AdamW(lr=0.001,weight_decay=0.0001)",
        "loss": "BCEWithLogitsLoss()",
        "gradient_clip_norm": 1.0,
        "batch_size": 128,
        "checkpoint_rule": "highest roles-2/3 validation average precision",
        "history": history,
    }


def plot_training(directory: Path, history: list[dict[str, Any]]) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    axes[0].plot(epochs, [row["train_bce"] for row in history], label="Train")
    axes[0].plot(
        epochs, [row["validation_bce"] for row in history], label="Validation"
    )
    axes[0].set(xlabel="Epoch", ylabel="Unweighted BCE", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_ap"] for row in history])
    axes[1].set(xlabel="Epoch", ylabel="Validation AP", title="Model selection")
    for axis in axes:
        axis.grid(alpha=0.25)
    save_figure_bundle(fig, directory / "tcn_training_validation")
    plt.close(fig)


def plot_confusion(directory: Path, confusion: list[list[int]]) -> None:
    cm = np.asarray(confusion, dtype=np.int64)
    fig, axis = plt.subplots(figsize=(3.4, 3.2), constrained_layout=True)
    image = axis.imshow(cm, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(cm[row, column]), ha="center", va="center")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Non-FoG", "FoG"],
        yticklabels=["Non-FoG", "FoG"],
        xlabel="Predicted",
        ylabel="True",
        title="RAW+TCN, unweighted BCE",
    )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    save_figure_bundle(fig, directory / "test_confusion_matrix")
    plt.close(fig)


def prepare_train_fold(
    args: argparse.Namespace,
    device: torch.device,
    fold: int,
) -> dict[str, Any]:
    """Materialize seed-invariant RAW train/validation features once per fold."""

    records, rows = load_records_rows(args.data_dir, fold)
    role67 = rows.take_role(6, 7)
    role23 = rows.take_role(2, 3)
    scaler, scaler_manifest, _ = load_scaler_only(
        source_dir(args.weighted_source, SEEDS[0]), fold, "gru"
    )
    train_x, train_feature = make_features(
        "RAW",
        scaler,
        raw_windows_dynamic(records, role67, 128),
        role67.label,
        device,
        source_dir(args.weighted_source, SEEDS[0]),
        fold,
        "gru",
        128,
    )
    validation_x, validation_feature = make_features(
        "RAW",
        scaler,
        raw_windows_dynamic(records, role23, 128),
        role23.label,
        device,
        source_dir(args.weighted_source, SEEDS[0]),
        fold,
        "gru",
        128,
    )
    return {
        "role67": role67,
        "role23": role23,
        "train_x": train_x,
        "validation_x": validation_x,
        "train_feature": train_feature,
        "validation_feature": validation_feature,
        "scaler_sha256": scaler_manifest["scaler_sha256"],
    }


def train_job(
    args: argparse.Namespace,
    device: torch.device,
    fold: int,
    seed: int,
    fold_cache: dict[str, Any],
) -> None:
    directory = job_dir(args.output_root, fold, seed)
    done = directory / "DONE_TRAIN.json"
    if done.exists() and not args.overwrite:
        print(f"SKIP TRAIN fold={fold} seed={seed}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    role67 = fold_cache["role67"]
    role23 = fold_cache["role23"]
    scaler, scaler_manifest, _ = load_scaler_only(
        source_dir(args.weighted_source, seed), fold, "gru"
    )
    if scaler_manifest["scaler_sha256"] != fold_cache["scaler_sha256"]:
        raise AssertionError("role-4 scaler differs across seeds")
    original_frozen_path = (
        args.weighted_source
        / "runs"
        / f"fold_{fold}"
        / "method_RAW"
        / f"seed_{seed}"
        / "frozen_validation.json"
    )
    original_frozen = read_json(original_frozen_path)
    if (
        scaler_manifest["scaler_sha256"]
        != original_frozen["role4_scaler_artifact"]["scaler_sha256"]
    ):
        raise AssertionError("reused scaler differs from weighted-BCE experiment")
    train_x = fold_cache["train_x"]
    validation_x = fold_cache["validation_x"]
    train_feature = fold_cache["train_feature"]
    validation_feature = fold_cache["validation_feature"]
    initial_state, initialization = paired_initialization(seed, "RAW")
    archived_initial_hash = original_frozen["initialization"][
        "raw_9ch_state_sha256"
    ]
    current_initial_hash = initialization["raw_9ch_state_sha256"]
    bitwise_initialization_match = current_initial_hash == archived_initial_hash
    if not bitwise_initialization_match:
        print(
            "AUDIT NOTE: same seed and initialization rule, but the current runtime "
            f"does not reproduce the archived initial-state hash for fold={fold}, "
            f"seed={seed}: current={current_initial_hash}, archived={archived_initial_hash}",
            flush=True,
        )
    model, training = train_unweighted_tcn(
        train_x,
        role67.label,
        validation_x,
        role23.label,
        directory,
        device,
        seed,
        args.num_workers,
        args.max_epochs,
        args.patience,
        initial_state,
    )
    val_true, val_prob = classifier_predict(
        model, validation_x, role23.label, device
    )
    threshold, validation_metrics = choose_document_threshold(val_true, val_prob)
    checkpoint = directory / "checkpoints" / "tcn.pt"
    frozen = {
        "job_id": job_id(fold, seed),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fold": fold,
        "method": "RAW",
        "loss_ablation": "unweighted_bce",
        "seed": seed,
        "input_shape": ["B", 9, 128],
        "feature": train_feature,
        "validation_feature": validation_feature,
        "role4_scaler_artifact": scaler_manifest,
        "weighted_bce_reference": str(original_frozen_path.resolve()),
        "weighted_bce_reference_sha256": sha256_file(original_frozen_path),
        "fairness_checks": {
            "same_roles": True,
            "same_scaler": True,
            "same_raw_features": True,
            "same_tcn_architecture": True,
            "same_seed_and_initialization_rule": True,
            "bitwise_same_initial_state_as_archived_weighted_run": (
                bitwise_initialization_match
            ),
            "current_initial_state_sha256": current_initial_hash,
            "archived_initial_state_sha256": archived_initial_hash,
            "same_optimizer": True,
            "same_epoch_budget_and_patience": True,
            "same_checkpoint_rule": True,
            "same_threshold_rule": True,
            "intended_algorithmic_change": (
                "remove pos_weight from BCEWithLogitsLoss"
            ),
            "comparison_caveat": (
                "archived weighted-BCE initial weights are not bitwise reproduced "
                "by the current local PyTorch runtime"
            ),
        },
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
    }
    write_json(directory / "frozen_validation.json", frozen)
    plot_training(directory, training["history"])
    write_json(
        done,
        {
            "status": "frozen",
            "job_id": frozen["job_id"],
            "checkpoint_sha256": frozen["checkpoint_sha256"],
            "threshold": float(threshold),
            "test_roles_accessed": False,
        },
    )
    print(
        f"TRAIN FROZEN fold={fold} seed={seed} best_epoch={training['best_epoch']} "
        f"val_ap={training['best_validation_ap']:.6f} threshold={threshold:.2f}",
        flush=True,
    )


def seal(args: argparse.Namespace) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for fold in FOLDS:
        for seed in SEEDS:
            directory = job_dir(args.output_root, fold, seed)
            frozen = read_json(directory / "frozen_validation.json")
            checkpoint = directory / "checkpoints" / "tcn.pt"
            if frozen["test_roles_accessed"] is not False:
                raise AssertionError("test roles were accessed before the barrier")
            if sha256_file(checkpoint) != frozen["checkpoint_sha256"]:
                raise AssertionError("checkpoint changed before the barrier")
            entries.append(
                {
                    "job_id": frozen["job_id"],
                    "fold": fold,
                    "method": "RAW",
                    "seed": seed,
                    "loss": "BCEWithLogitsLoss()",
                    "effective_pos_weight": 1.0,
                    "threshold": frozen["threshold"],
                    "checkpoint_sha256": frozen["checkpoint_sha256"],
                    "scaler_sha256": frozen["role4_scaler_artifact"][
                        "scaler_sha256"
                    ],
                    "frozen_validation_sha256": sha256_file(
                        directory / "frozen_validation.json"
                    ),
                }
            )
    rows_by_fold = {
        fold: load_records_rows(args.data_dir, fold)[1] for fold in FOLDS
    }
    source_audit = audit_protocol_dynamic(args.data_dir, rows_by_fold, 64, 128, 64)
    test_manifest = build_test_data_manifest(args.data_dir, rows_by_fold)
    barrier = {
        "barrier_schema": "raw_tcn_unweighted_bce.strict_test_barrier.v1",
        "status": "all_15_classifiers_and_validation_thresholds_frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "methods": ["RAW"],
        "job_count": len(entries),
        "source_audit": source_audit,
        "test_data_manifest": test_manifest,
        "jobs": entries,
    }
    barrier["barrier_id"] = stable_json_hash(barrier)
    write_json(args.output_root / "TRAINING_BARRIER.json", barrier)
    write_json(
        args.output_root / "experiment_config.json",
        {
            "experiment": "strict_RAW_TCN_unweighted_BCE_loss_ablation",
            "data": str(args.data_dir.resolve()),
            "input": "role4 RobustScaler + window-axis centering + RAW [B,9,128]",
            "loss": "BCEWithLogitsLoss()",
            "effective_pos_weight": 1.0,
            "weighted_bce_reference": str(args.weighted_source.resolve()),
            "folds": list(FOLDS),
            "seeds": list(SEEDS),
            "sampling_rate_hz": 64,
            "window_samples": 128,
            "stride_samples": 64,
            "roles": {
                "0": "permanent_test_nonfog",
                "1": "permanent_test_fog",
                "2": "classifier_validation_nonfog",
                "3": "classifier_validation_fog",
                "4": "scaler_train_clean",
                "6": "classifier_train_nonfog",
                "7": "classifier_train_fog",
            },
            "optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)",
            "batch_size": 128,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "gradient_clip_norm": 1.0,
            "checkpoint_rule": "maximum roles-2/3 validation average precision",
            "threshold_rule": (
                "roles 2/3 maximum balanced accuracy; ties FoG F1 then higher "
                "threshold; 0.05..0.95 step 0.01"
            ),
            "strict_test_barrier_jobs": len(entries),
            "barrier_id": barrier["barrier_id"],
            "test_data_manifest_sha256": test_manifest["sha256"],
        },
    )
    print(f"GLOBAL TRAINING BARRIER SEALED jobs={len(entries)}", flush=True)
    return barrier


def prepare_test_fold(
    args: argparse.Namespace,
    device: torch.device,
    fold: int,
) -> dict[str, Any]:
    """Materialize the frozen permanent-test RAW features once per fold."""

    records, rows = load_records_rows(args.data_dir, fold)
    test_rows = rows.take_role(0, 1)
    scaler, scaler_manifest, _ = load_scaler_only(
        source_dir(args.weighted_source, SEEDS[0]), fold, "gru"
    )
    test_x, test_feature = make_features(
        "RAW",
        scaler,
        raw_windows_dynamic(records, test_rows, 128),
        test_rows.label,
        device,
        source_dir(args.weighted_source, SEEDS[0]),
        fold,
        "gru",
        128,
    )
    return {
        "test_rows": test_rows,
        "test_x": test_x,
        "test_feature": test_feature,
        "scaler_sha256": scaler_manifest["scaler_sha256"],
    }


def test_job(
    args: argparse.Namespace,
    device: torch.device,
    barrier: dict[str, Any],
    fold: int,
    seed: int,
    fold_cache: dict[str, Any],
) -> dict[str, Any]:
    directory = job_dir(args.output_root, fold, seed)
    done = directory / "DONE_TEST.json"
    if done.exists() and not args.overwrite:
        print(f"SKIP TEST fold={fold} seed={seed}", flush=True)
        return read_json(directory / "metrics.json")
    sealed = next(
        entry
        for entry in barrier["jobs"]
        if int(entry["fold"]) == fold and int(entry["seed"]) == seed
    )
    checkpoint = directory / "checkpoints" / "tcn.pt"
    if sha256_file(checkpoint) != sealed["checkpoint_sha256"]:
        raise AssertionError("sealed checkpoint changed")
    test_rows = fold_cache["test_rows"]
    scaler, scaler_manifest, _ = load_scaler_only(
        source_dir(args.weighted_source, seed), fold, "gru"
    )
    if scaler_manifest["scaler_sha256"] != sealed["scaler_sha256"]:
        raise AssertionError("sealed scaler changed")
    if scaler_manifest["scaler_sha256"] != fold_cache["scaler_sha256"]:
        raise AssertionError("test scaler differs across seeds")
    test_x = fold_cache["test_x"]
    test_feature = fold_cache["test_feature"]
    model = RepresentationTCNM(9).to(device)
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
        "fold": fold,
        "method": "RAW",
        "loss_ablation": "unweighted_bce",
        "tcn_seed": seed,
        "threshold": threshold,
        "threshold_source_roles": [2, 3],
        "strict_global_test_barrier_verified": True,
        "barrier_id": barrier["barrier_id"],
        "test_data_manifest_sha256": barrier["test_data_manifest"]["sha256"],
        "test_roles": [0, 1],
        "test": metrics,
        "test_by_subject": by_subject,
        "test_feature": test_feature,
        "test_feature_diagnostics": residual_diagnostics(test_x, test_true),
        "tcn_checkpoint_sha256": sealed["checkpoint_sha256"],
        "scaler_sha256": sealed["scaler_sha256"],
    }
    write_json(directory / "metrics.json", result)
    write_csv(
        directory / "test_predictions.csv",
        [
            {
                "fold": fold,
                "method": "RAW",
                "tcn_seed": seed,
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
    plot_confusion(directory, metrics["confusion_matrix"])
    write_json(
        done,
        {
            "status": "complete",
            "job_id": sealed["job_id"],
            "barrier_id": barrier["barrier_id"],
            "test": metrics,
        },
    )
    print(
        f"TEST fold={fold} seed={seed} acc={metrics['accuracy']:.6f} "
        f"recall={metrics['sensitivity']:.6f} spec={metrics['specificity']:.6f} "
        f"ap={metrics['auprc']:.6f}",
        flush=True,
    )
    return result


def mean_std(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "n": int(len(array)),
    }


def aggregate(args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    run_rows = [
        {
            "fold": result["fold"],
            "method": "RAW",
            "loss": "unweighted_bce",
            "tcn_seed": result["tcn_seed"],
            "threshold": result["threshold"],
            **{metric: result["test"][metric] for metric in METRICS},
            **{key: result["test"][key] for key in ("tn", "fp", "fn", "tp")},
        }
        for result in results
    ]
    write_csv(args.output_root / "run_metrics_15.csv", run_rows)
    seed_rows = []
    for seed in SEEDS:
        subset = [result for result in results if result["tcn_seed"] == seed]
        seed_rows.append(
            {
                "method": "RAW",
                "loss": "unweighted_bce",
                "tcn_seed": seed,
                **{
                    metric: float(
                        np.mean([result["test"][metric] for result in subset])
                    )
                    for metric in METRICS
                },
            }
        )
    write_csv(args.output_root / "seed_macro_over_3folds.csv", seed_rows)
    summary = {
        metric: mean_std(row[metric] for row in seed_rows) for metric in METRICS
    }
    write_csv(
        args.output_root / "method_summary_5seed_mean_std.csv",
        [
            {
                "method": "RAW",
                "loss": "unweighted_bce",
                "metric": metric,
                **summary[metric],
            }
            for metric in METRICS
        ],
    )
    subject_rows = []
    for subject in SUBJECTS:
        per_seed = []
        for seed in SEEDS:
            subset = [result for result in results if result["tcn_seed"] == seed]
            per_seed.append(
                {
                    metric: float(
                        np.mean(
                            [
                                result["test_by_subject"][subject][metric]
                                for result in subset
                            ]
                        )
                    )
                    for metric in METRICS
                }
            )
        stats = {
            metric: mean_std(item[metric] for item in per_seed)
            for metric in METRICS
        }
        subject_rows.append(
            {
                "method": "RAW",
                "loss": "unweighted_bce",
                "subject_id": subject,
                **{
                    f"{metric}_mean": stats[metric]["mean"] for metric in METRICS
                },
                **{
                    f"{metric}_std": stats[metric]["std"] for metric in METRICS
                },
            }
        )
    write_csv(
        args.output_root / "subject_metrics_5seed_mean_std.csv", subject_rows
    )
    final = {
        "method": "RAW+TCN",
        "loss": "BCEWithLogitsLoss()",
        "effective_pos_weight": 1.0,
        "fold_count": len(FOLDS),
        "seeds": list(SEEDS),
        "aggregation": (
            "mean over 3 folds within each seed, then mean and population SD "
            "over 5 seeds"
        ),
        "primary_metrics": summary,
    }
    write_json(args.output_root / "summary.json", final)
    write_json(
        args.output_root / "DONE.json",
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "run_count": len(results),
            "seeds": list(SEEDS),
        },
    )
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.weighted_source = args.weighted_source.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.output_root / "DONE.json").exists() and not args.overwrite:
        print(f"EXPERIMENT ALREADY COMPLETE: {args.output_root}", flush=True)
        return
    if args.max_epochs != 5 or args.patience != 2:
        raise ValueError("strict ablation requires max_epochs=5 and patience=2")
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    print(
        f"START device={device} output={args.output_root} seeds={SEEDS}", flush=True
    )
    # Phase 1: no role-0/1 feature construction or inference.
    for fold in FOLDS:
        fold_cache = prepare_train_fold(args, device, fold)
        for seed in SEEDS:
            train_job(args, device, fold, seed, fold_cache)
        del fold_cache
    barrier = seal(args)
    # Phase 2: permanent test only after all checkpoints and thresholds freeze.
    results = []
    for fold in FOLDS:
        fold_cache = prepare_test_fold(args, device, fold)
        for seed in SEEDS:
            results.append(
                test_job(args, device, barrier, fold, seed, fold_cache)
            )
        del fold_cache
    aggregate(args, results)


if __name__ == "__main__":
    main()
