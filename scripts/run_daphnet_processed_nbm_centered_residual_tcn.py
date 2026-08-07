#!/usr/bin/env python
"""Three-fold role-isolated centered GRU-NBM residual -> TCN experiment.

The authoritative role assignments are read from ``processed_NBM/split_indices``.
Subjects S04 and S10 are excluded.  For each outer fold:

* role 4 alone fits the RobustScaler and GRU reconstruction NBM;
* role 5 alone early-stops the NBM and estimates residual bias/scale;
* roles 6/7 alone train the TCN classifier;
* roles 2/3 alone early-stop the classifier and select the threshold;
* roles 0/1 are evaluated only after every trainable/tunable quantity is frozen.

Both the scaled NBM input and clipped standardized residual are centered per
window and per channel over the 128-sample time dimension.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.evaluation import binary_metrics
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    GRUReconstructionNBM,
    ResidualTCNM,
    RobustScaler,
    calibrate,
    choose_document_threshold,
    classifier_loader,
    classifier_predict,
    normalized_residual,
    prepare_nbm_windows,
    residual_diagnostics,
    set_seed,
    train_nbm,
    write_csv,
    write_json,
)

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)

FS = 64
WINDOW = 128
STRIDE = 64
SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
ROLES = {
    0: "permanent_test_nonfog",
    1: "permanent_test_fog",
    2: "external_validation_nonfog",
    3: "external_validation_fog",
    4: "nbm_train_clean",
    5: "nbm_earlystop_clean",
    6: "classifier_train_clean",
    7: "classifier_train_fog",
}
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


@dataclass(frozen=True)
class RoleRows:
    subject_id: np.ndarray
    record_id: np.ndarray
    start: np.ndarray
    end: np.ndarray
    role: np.ndarray
    label: np.ndarray
    window_id: np.ndarray

    def __len__(self) -> int:
        return int(self.start.size)

    def take_role(self, *roles: int) -> "RoleRows":
        mask = np.isin(self.role, np.asarray(roles, dtype=np.int8))
        return RoleRows(*(getattr(self, name)[mask] for name in self.__dataclass_fields__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT
        / "dataset"
        / "1.Daphnet Freezing of Gait Dataset"
        / "processed_NBM",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "daphnet_processed_NBM_centered_residual_tcn_roles8_seed20260807",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-max-epochs", type=int, default=50)
    parser.add_argument("--nbm-patience", type=int, default=8)
    parser.add_argument("--tcn-max-epochs", type=int, default=30)
    parser.add_argument("--tcn-patience", type=int, default=6)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Re-export existing fold plots without loading data or retraining.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fold_rows(data_dir: Path, fold: int) -> RoleRows:
    columns: dict[str, list[np.ndarray]] = {
        name: [] for name in RoleRows.__dataclass_fields__
    }
    for subject in SUBJECTS:
        path = data_dir / "split_indices" / f"{subject}_outer{fold}_nbm_indices.npz"
        with np.load(path, allow_pickle=False) as payload:
            count = len(payload["start_index"])
            columns["subject_id"].append(np.full(count, subject, dtype="U3"))
            columns["record_id"].append(payload["record_id"].astype("U32"))
            columns["start"].append(payload["start_index"].astype(np.int32))
            columns["end"].append(payload["end_index_exclusive"].astype(np.int32))
            columns["role"].append(payload["role_code"].astype(np.int8))
            columns["label"].append(payload["y_binary"].astype(np.int8))
            columns["window_id"].append(payload["window_id"].astype("U96"))
    return RoleRows(*(np.concatenate(columns[name]) for name in RoleRows.__dataclass_fields__))


def audit_protocol(data_dir: Path, rows_by_fold: dict[int, RoleRows], records: dict[str, Any]) -> dict:
    quality = json.loads((data_dir / "nbm_quality_report.json").read_text(encoding="utf-8"))
    if not quality.get("overall_pass", False):
        raise AssertionError("processed_NBM quality report does not pass")
    fixed_tests: dict[str, bool] = {}
    for subject in SUBJECTS:
        test_sets = []
        for fold in (0, 1, 2):
            rows = rows_by_fold[fold]
            mask = (rows.subject_id == subject) & np.isin(rows.role, [0, 1])
            test_sets.append(set(rows.window_id[mask].tolist()))
        fixed_tests[subject] = test_sets[0] == test_sets[1] == test_sets[2]
        if not fixed_tests[subject]:
            raise AssertionError(f"permanent test is not fixed for {subject}")

    fold_reports = []
    for fold, rows in rows_by_fold.items():
        expected = np.isin(rows.role, [1, 3, 7]).astype(np.int8)
        if not np.array_equal(rows.label, expected):
            raise AssertionError(f"role/label mismatch in fold {fold}")
        if set(np.unique(rows.role).tolist()) != set(ROLES):
            raise AssertionError(f"missing role in fold {fold}")
        if len(set(rows.window_id.tolist())) != len(rows):
            raise AssertionError(f"duplicate window_id in fold {fold}")
        if np.any(rows.end - rows.start != WINDOW):
            raise AssertionError(f"non-128 sample window in fold {fold}")
        # Rebuild per-record raw-point role bitmasks. Repeated support within one
        # role is allowed; any raw point shared by different roles is forbidden.
        point_masks = {
            record_id: np.zeros(len(record.x), dtype=np.uint8)
            for record_id, record in records.items()
            if record.subject_id in SUBJECTS
        }
        for record_id, start, end, role in zip(rows.record_id, rows.start, rows.end, rows.role):
            bit = np.uint8(1 << int(role))
            point_masks[str(record_id)][int(start) : int(end)] |= bit
        overlap_points = 0
        for mask in point_masks.values():
            overlap_points += int(np.count_nonzero((mask & (mask - 1)) != 0))
        if overlap_points:
            raise AssertionError(f"fold {fold} has {overlap_points} cross-role raw points")
        counts = {str(role): int(np.sum(rows.role == role)) for role in ROLES}
        fold_reports.append({"fold": fold, "role_counts": counts, "cross_role_overlap_points": 0})
    return {
        "source_quality_overall_pass": True,
        "fixed_permanent_test_by_subject": fixed_tests,
        "folds": fold_reports,
    }


def raw_windows(records: dict[str, Any], rows: RoleRows) -> np.ndarray:
    values = np.empty((len(rows), WINDOW, 9), dtype=np.float32)
    for index, (record_id, start, end) in enumerate(zip(rows.record_id, rows.start, rows.end)):
        values[index] = records[str(record_id)].x[int(start) : int(end)]
    return values


def fit_scaler_unique_role4_points(records: dict[str, Any], rows: RoleRows) -> tuple[RobustScaler, int]:
    if not np.all(rows.role == 4) or not np.all(rows.label == 0):
        raise AssertionError("scaler input is not exclusively role 4")
    masks: dict[str, np.ndarray] = {}
    for record_id, start, end in zip(rows.record_id, rows.start, rows.end):
        record_id = str(record_id)
        masks.setdefault(record_id, np.zeros(len(records[record_id].x), dtype=bool))
        masks[record_id][int(start) : int(end)] = True
    chunks = [records[record_id].x[mask] for record_id, mask in masks.items() if np.any(mask)]
    values = np.concatenate(chunks, axis=0).astype(np.float64, copy=False)
    median = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
    iqr = q75 - q25
    if np.any(iqr <= 1e-6):
        raise ValueError(f"degenerate scaler IQR channels: {np.flatnonzero(iqr <= 1e-6)}")
    return RobustScaler(median.astype(np.float32), iqr.astype(np.float32)), int(len(values))


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
    with torch.no_grad():
        for batch_x, batch_y in classifier_loader(x, y, False, 0, 0):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            loss = criterion(model(batch_x), batch_y)
            total += float(loss) * len(batch_x)
            count += len(batch_x)
    return total / count


def train_tcn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    fold_dir: Path,
    device: torch.device,
    seed: int,
    num_workers: int,
    max_epochs: int,
    patience: int,
) -> tuple[nn.Module, dict]:
    set_seed(seed)
    model = ResidualTCNM().to(device)
    n_pos = int(np.sum(train_y == 1))
    n_neg = int(np.sum(train_y == 0))
    if not n_pos or not n_neg:
        raise ValueError("roles 6/7 must contain both classes")
    pos_weight = n_neg / n_pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = classifier_loader(train_x, train_y, True, seed, num_workers)
    checkpoint = fold_dir / "checkpoints" / "tcn.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_pr = -math.inf
    best_epoch = 0
    stale = 0
    history = []
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
        val_bce = validation_loss(model, val_x, val_y, criterion, device)
        val_true, val_prob = classifier_predict(model, val_x, val_y, device)
        val_pr = float(average_precision_score(val_true, val_prob))
        improved = val_pr > best_pr + 1e-10
        history.append(
            {
                "epoch": epoch,
                "train_weighted_bce": train_bce,
                "validation_weighted_bce": val_bce,
                "validation_pr_auc": val_pr,
                "improved": improved,
            }
        )
        if improved:
            best_pr = val_pr
            best_epoch = epoch
            stale = 0
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "validation_pr_auc": val_pr},
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"TCN fold={fold_dir.name} epoch={epoch:02d} train={train_bce:.7f} "
            f"val={val_bce:.7f} val_pr={val_pr:.7f} stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    write_csv(fold_dir / "logs" / "tcn_history.csv", history)
    return model, {
        "maximum_epochs": max_epochs,
        "patience": patience,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_pr_auc": best_pr,
        "pos_weight": pos_weight,
        "n_nonfog_role6": n_neg,
        "n_fog_role7": n_pos,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "history": history,
    }


def save_figure_bundle(fig: Any, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def plot_fold(fold_dir: Path, nbm_run: dict, tcn_run: dict, confusion: list[list[int]]) -> None:
    nh = nbm_run["history"]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    epochs = [row["epoch"] for row in nh]
    ax.plot(epochs, [row["train_huber"] for row in nh], label="Role 4 train")
    ax.plot(epochs, [row["validation_huber"] for row in nh], label="Role 5 validation")
    ax.axvline(nbm_run["summary"]["best_epoch"], color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Epoch", ylabel="SmoothL1 loss", title="GRU-NBM training")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure_bundle(fig, fold_dir / "nbm_training_validation_loss")
    plt.close(fig)

    th = tcn_run["history"]
    epochs = [row["epoch"] for row in th]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(epochs, [row["train_weighted_bce"] for row in th], label="Roles 6/7 train")
    axes[0].plot(epochs, [row["validation_weighted_bce"] for row in th], label="Roles 2/3 validation")
    axes[0].set(xlabel="Epoch", ylabel="Weighted BCE", title="TCN loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_pr_auc"] for row in th])
    axes[1].axvline(tcn_run["best_epoch"], color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Epoch", ylabel="PR-AUC", title="TCN validation model selection")
    axes[1].grid(alpha=0.25)
    save_figure_bundle(fig, fold_dir / "tcn_training_validation")
    plt.close(fig)

    cm = np.asarray(confusion, dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > 0.5 * cm.max() else "black"
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                color=color,
            )
    ax.set_xticks([0, 1], ["non-FoG", "FoG"])
    ax.set_yticks([0, 1], ["non-FoG", "FoG"])
    ax.set(xlabel="Predicted", ylabel="True", title="Permanent test confusion matrix")
    save_figure_bundle(fig, fold_dir / "test_confusion_matrix")
    plt.close(fig)


def metric_summary(rows: list[dict], keys: tuple[str, ...] = METRIC_KEYS) -> dict:
    output = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if row.get(key) is not None], dtype=float)
        output[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
            "n_folds": int(len(values)),
        }
    return output


def render_existing_plots(output_dir: Path, folds: list[int]) -> None:
    for fold in folds:
        fold_dir = output_dir / f"fold_{fold}"
        metrics = json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8"))
        with (fold_dir / "logs" / "nbm_history.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            nbm_history = list(csv.DictReader(handle))
        for row in nbm_history:
            row["epoch"] = int(row["epoch"])
            row["train_huber"] = float(row["train_huber"])
            row["validation_huber"] = float(row["validation_huber"])
        with (fold_dir / "logs" / "tcn_history.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            tcn_history = list(csv.DictReader(handle))
        for row in tcn_history:
            row["epoch"] = int(row["epoch"])
            row["train_weighted_bce"] = float(row["train_weighted_bce"])
            row["validation_weighted_bce"] = float(row["validation_weighted_bce"])
            row["validation_pr_auc"] = float(row["validation_pr_auc"])
        plot_fold(
            fold_dir,
            {"history": nbm_history, "summary": metrics["nbm_training"]},
            {**metrics["tcn_training"], "history": tcn_history},
            metrics["test"]["confusion_matrix"],
        )
        print(f"RENDERED fold={fold}", flush=True)


def run_fold(
    fold: int,
    rows: RoleRows,
    records: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed + fold
    role4 = rows.take_role(4)
    role5 = rows.take_role(5)
    role67 = rows.take_role(6, 7)
    role23 = rows.take_role(2, 3)

    scaler, scaler_unique_points = fit_scaler_unique_role4_points(records, role4)
    role4_x = prepare_nbm_windows(scaler, raw_windows(records, role4), center=True)
    role5_x = prepare_nbm_windows(scaler, raw_windows(records, role5), center=True)
    nbm, nbm_run = train_nbm(
        "nbm",
        role4_x,
        role5_x,
        fold_dir,
        device,
        seed,
        args.num_workers,
        max_epochs=args.nbm_max_epochs,
        patience=args.nbm_patience,
        bottleneck=16,
    )
    bias, sigma, calibration = calibrate(nbm, role5_x, device)
    write_json(
        fold_dir / "nbm_frozen.json",
        {
            "scaler": scaler.as_dict(),
            "scaler_fit_role": 4,
            "scaler_unique_raw_points": scaler_unique_points,
            "nbm_train_role": 4,
            "nbm_earlystop_and_calibration_role": 5,
            "calibration": calibration,
            "training": nbm_run["summary"],
        },
    )

    # Only after NBM/scaler/b/sigma are frozen are classifier train and validation
    # representations generated.
    train_residual, _, _ = normalized_residual(
        nbm, scaler, bias, sigma, raw_windows(records, role67), device, center_windows=True
    )
    val_residual, _, _ = normalized_residual(
        nbm, scaler, bias, sigma, raw_windows(records, role23), device, center_windows=True
    )
    train_y = role67.label
    val_y = role23.label
    if not np.array_equal(train_y, np.isin(role67.role, [7]).astype(np.int8)):
        raise AssertionError("classifier labels do not come exactly from roles 6/7")
    tcn, tcn_run = train_tcn(
        train_residual,
        train_y,
        val_residual,
        val_y,
        fold_dir,
        device,
        seed + 100,
        args.num_workers,
        args.tcn_max_epochs,
        args.tcn_patience,
    )
    val_true, val_prob = classifier_predict(tcn, val_residual, val_y, device)
    threshold, val_metrics = choose_document_threshold(val_true, val_prob)
    # Test representation and probabilities are created only after the threshold
    # and best classifier checkpoint have been fixed from roles 2/3.
    test_rows = rows.take_role(0, 1)
    test_residual, _, _ = normalized_residual(
        nbm, scaler, bias, sigma, raw_windows(records, test_rows), device, center_windows=True
    )
    test_true, test_prob = classifier_predict(tcn, test_residual, test_rows.label, device)
    test_metrics = binary_metrics(test_true, test_prob, threshold)
    test_pred = (test_prob >= threshold).astype(np.int8)

    subject_metrics = {}
    for subject in SUBJECTS:
        mask = test_rows.subject_id == subject
        subject_metrics[subject] = binary_metrics(test_true[mask], test_prob[mask], threshold)
    diagnostics = {
        "classifier_train_roles_6_7": residual_diagnostics(train_residual, train_y),
        "classifier_validation_roles_2_3": residual_diagnostics(val_residual, val_y),
        "permanent_test_roles_0_1": residual_diagnostics(test_residual, test_true),
    }
    result = {
        "fold": fold,
        "seed": seed,
        "threshold": threshold,
        "threshold_source_roles": [2, 3],
        "validation": val_metrics,
        "test": test_metrics,
        "test_by_subject": subject_metrics,
        "nbm_training": nbm_run["summary"],
        "tcn_training": {key: value for key, value in tcn_run.items() if key != "history"},
        "role_counts": {str(role): int(np.sum(rows.role == role)) for role in ROLES},
        "residual_diagnostics": diagnostics,
    }
    write_json(fold_dir / "metrics.json", result)
    write_csv(
        fold_dir / "predictions.csv",
        [
            {
                "fold": fold,
                "subject_id": str(test_rows.subject_id[i]),
                "record_id": str(test_rows.record_id[i]),
                "window_id": str(test_rows.window_id[i]),
                "start_index": int(test_rows.start[i]),
                "end_index_exclusive": int(test_rows.end[i]),
                "role_code": int(test_rows.role[i]),
                "y_true": int(test_true[i]),
                "fog_probability": float(test_prob[i]),
                "threshold": threshold,
                "y_pred": int(test_pred[i]),
            }
            for i in range(len(test_rows))
        ],
    )
    write_csv(
        fold_dir / "test_by_subject.csv",
        [
            {"subject_id": subject, **{key: value for key, value in metrics.items() if key != "confusion_matrix"}}
            for subject, metrics in subject_metrics.items()
        ],
    )
    write_csv(
        fold_dir / "test_confusion_matrix.csv",
        [
            {"true\\pred": "non-FoG", "non-FoG": test_metrics["tn"], "FoG": test_metrics["fp"]},
            {"true\\pred": "FoG", "non-FoG": test_metrics["fn"], "FoG": test_metrics["tp"]},
        ],
    )
    plot_fold(fold_dir, nbm_run, tcn_run, test_metrics["confusion_matrix"])
    np.savez_compressed(
        fold_dir / "test_probabilities.npz",
        y_true=test_true,
        y_prob=test_prob,
        y_pred=test_pred,
        subject_id=test_rows.subject_id,
        window_id=test_rows.window_id,
        threshold=np.asarray(threshold),
    )
    print(
        f"FOLD {fold} COMPLETE threshold={threshold:.2f} acc={test_metrics['accuracy']:.6f} "
        f"recall={test_metrics['sensitivity']:.6f} specificity={test_metrics['specificity']:.6f} "
        f"pr_auc={test_metrics['auprc']:.6f} cm={test_metrics['confusion_matrix']}",
        flush=True,
    )
    return result


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    done_path = output_dir / "DONE.json"
    if args.render_only:
        folds = [args.fold] if args.fold is not None else [0, 1, 2]
        render_existing_plots(output_dir, folds)
        return
    if done_path.exists() and not args.overwrite:
        raise FileExistsError(f"completed output exists: {done_path}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dataset = DaphnetDataset.load(data_dir)
    records = {record.record_id: record for record in dataset.records}
    rows_by_fold = {fold: load_fold_rows(data_dir, fold) for fold in (0, 1, 2)}
    audit = audit_protocol(data_dir, rows_by_fold, records)
    write_json(output_dir / "preflight_audit.json", audit)

    config = {
        "experiment": "processed_NBM_centered_GRU_reconstruction_residual_TCN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "included_subjects": list(SUBJECTS),
        "excluded_subjects": ["S04", "S10"],
        "outer_folds": [0, 1, 2] if args.fold is None else [args.fold],
        "seed": args.seed,
        "device": str(device),
        "manifest_sha256": sha256(data_dir / "manifest.csv"),
        "protocol_sha256": sha256(data_dir / "nbm_protocol.json"),
        "sampling_rate_hz": FS,
        "window_samples": WINDOW,
        "window_seconds": WINDOW / FS,
        "stride_samples": STRIDE,
        "stride_seconds": STRIDE / FS,
        "roles": {str(key): value for key, value in ROLES.items()},
        "scaler": {
            "type": "per-channel median/IQR",
            "fit_data": "union of unique raw points covered by role-4 windows only",
            "epsilon": 1e-6,
        },
        "centering": {
            "nbm": "after RobustScaler, subtract each window/channel mean over 128 time samples",
            "tcn": "after residual clipping, subtract each residual window/channel mean; no second clipping",
        },
        "nbm": {
            "architecture": "GRU(9,64)->Linear(64,16)->Linear(16,64)->zero-input GRU(9,64)->Linear(64,9)",
            "loss": "SmoothL1(beta=1.0)",
            "augmentation": "40% clean, 40% Gaussian(std=0.04), 20% all-axis time mask(length 4..8)",
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)",
            "batch_size": 128,
            "max_epochs": args.nbm_max_epochs,
            "patience": args.nbm_patience,
            "gradient_clip": 1.0,
            "fit_role": 4,
            "earlystop_role": 5,
            "restore_best": True,
        },
        "residual": {
            "formula": "clip((X_scaled_centered-Xhat-b)/(sigma+1e-6),-12,12)",
            "b": "role-5 per-channel median reconstruction error",
            "sigma": "role-5 1.4826*MAD, per-channel floor 0.05",
        },
        "tcn": {
            "architecture": "causal residual TCN: channels 9/32/64/64/128, kernel 3, dilations 1/2/4/8, two convolutions per block, GAP, dropout 0.3, linear logit",
            "train_roles": [6, 7],
            "validation_roles": [2, 3],
            "loss": "BCEWithLogitsLoss(pos_weight=N_role6/N_role7)",
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
            "batch_size": 128,
            "max_epochs": args.tcn_max_epochs,
            "patience": args.tcn_patience,
            "monitor": "role-2/3 PR-AUC",
            "gradient_clip": 1.0,
        },
        "threshold": "role-2/3 balanced accuracy over 0.05..0.95 step 0.01; ties FoG F1 then higher threshold",
        "test_roles": [0, 1],
        "test_use": "single final inference after NBM, TCN checkpoint, and threshold are frozen",
        "no_internal_nbm_crossfit": True,
        "role4_or_5_used_for_classifier_training": False,
    }
    write_json(output_dir / "config.json", config)
    print(f"PREFLIGHT PASS device={device} folds={config['outer_folds']}", flush=True)
    if args.dry_run:
        write_json(output_dir / "DRY_RUN.json", {"status": "complete"})
        return

    folds = [args.fold] if args.fold is not None else [0, 1, 2]
    results = [run_fold(fold, rows_by_fold[fold], records, output_dir, args, device) for fold in folds]
    fold_metrics = [{"fold": result["fold"], "threshold": result["threshold"], **result["test"]} for result in results]
    aggregate = metric_summary([result["test"] for result in results])
    subject_aggregate = {
        subject: metric_summary([result["test_by_subject"][subject] for result in results])
        for subject in SUBJECTS
    }
    summary = {
        "fold_results": results,
        "aggregate_test_metrics_mean_std_across_folds": aggregate,
        "subject_test_metrics_mean_std_across_folds": subject_aggregate,
        "important_note": "Roles 0/1 are a fixed test set repeated across folds; aggregate values are model-fold mean/std, not a pooled sample estimate.",
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "fold_metrics.csv", fold_metrics)
    write_csv(
        output_dir / "subject_metrics_mean.csv",
        [
            {
                "subject_id": subject,
                **{f"{key}_mean": subject_aggregate[subject][key]["mean"] for key in METRIC_KEYS},
                **{f"{key}_std": subject_aggregate[subject][key]["std"] for key in METRIC_KEYS},
            }
            for subject in SUBJECTS
        ],
    )
    write_json(
        done_path,
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "aggregate_test_metrics": aggregate,
        },
    )
    print("ALL COMPLETE", json.dumps(aggregate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
