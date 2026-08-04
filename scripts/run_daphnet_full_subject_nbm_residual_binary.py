from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_daphnet_nbm_routeA_A1b_generalization_repair as a1b
import run_daphnet_nbm_tcdae_three_rounds as base
from cnbr_fog.data import DaphnetDataset, Record


EXPERIMENT = "daphnet_full_subject_nbm_residual_binary_v1"
SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
A1_PASS = ("S01", "S05", "S07", "S08", "S09")
DIFFICULT = ("S02", "S03", "S06")
METHODS = ("B0", "B1", "B2", "B3")
METHOD_NAMES = {"B0": "Raw-TCN", "B1": "R-TCN", "B2": "R5-TCN", "B3": "Raw+R5-TCN"}
METHOD_CHANNELS = {"B0": 9, "B1": 9, "B2": 27, "B3": 36}
METHOD_DIRS = {"B0": "B0_raw_tcn", "B1": "B1_residual_tcn", "B2": "B2_r5_tcn", "B3": "B3_raw_r5_tcn"}
SEEDS = (20260802, 20260803, 20260804)
NBM_SEED = 20260802
FS = 64
WINDOW = 128
STRIDE = 64
LABEL_SAMPLES = 32
GUARD = 5 * FS
INNER_K = 3
CLASSIFICATION_METRICS = ("pr_auc", "roc_auc", "fog_f1", "recall", "precision", "specificity", "balanced_accuracy", "mcc")
FIGURE_NAMES = (
    "macro_metric_comparison.png", "subject_pr_auc_heatmap.png", "subject_fog_f1_heatmap.png",
    "paired_pr_auc_change.png", "paired_f1_change.png", "method_rank_by_subject.png",
    "pooled_pr_curve.png", "pooled_roc_curve.png", "subject_pr_curves.png",
    "subject_roc_curves.png", "pooled_confusion_matrix.png",
    "reconstruction_quality_vs_residual_gain.png", "fog_ratio_vs_ap.png",
    "false_alarm_by_subject.png", "best_worst_case_probabilities.png",
)


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, (np.floating, float)) and not math.isfinite(float(value)):
            return None
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass
class SubjectWindows:
    subject: str
    records: list[Record]
    raw: np.ndarray
    label: np.ndarray
    strict_clean: np.ndarray
    record_index: np.ndarray
    record_id: np.ndarray
    start: np.ndarray

    @property
    def keys(self) -> np.ndarray:
        return np.asarray([f"{record}:{int(start)}" for record, start in zip(self.record_id, self.start)], dtype="U64")


@dataclass
class RobustScaler:
    center: np.ndarray
    scale: np.ndarray

    def transform(self, raw: np.ndarray) -> np.ndarray:
        scaled = (np.asarray(raw, dtype=np.float32) - self.center) / (self.scale + 1e-6)
        return np.ascontiguousarray(scaled - scaled.mean(axis=1, keepdims=True), dtype=np.float32)


def build_subject_windows(dataset: DaphnetDataset, subject: str) -> SubjectWindows:
    records = [record for record in dataset.records if record.subject_id == subject and np.any(record.valid)]
    raw: list[np.ndarray] = []
    label: list[int] = []
    clean: list[bool] = []
    record_index: list[int] = []
    record_id: list[str] = []
    starts: list[int] = []
    for rec_index, record in enumerate(records):
        for start in range(0, len(record.y) - WINDOW + 1, STRIDE):
            end = start + WINDOW
            if not record.valid[start:end].all():
                continue
            guard_start = max(0, start - GUARD)
            guard_end = min(len(record.y), end + GUARD)
            raw.append(record.x[start:end].astype(np.float32))
            label.append(int(np.mean(record.y[end - LABEL_SAMPLES:end]) >= 0.5))
            clean.append(bool(not np.any(record.y[guard_start:guard_end])))
            record_index.append(rec_index)
            record_id.append(record.record_id)
            starts.append(start)
    if not raw:
        raise ValueError(f"{subject} has no valid windows")
    return SubjectWindows(subject, records, np.stack(raw), np.asarray(label, dtype=np.int8),
                          np.asarray(clean, dtype=bool), np.asarray(record_index, dtype=np.int16),
                          np.asarray(record_id, dtype="U32"), np.asarray(starts, dtype=np.int64))


def outer_folds(item: SubjectWindows) -> list[dict[str, Any]]:
    record_ids = [record.record_id for record in item.records]
    folds: list[dict[str, Any]] = []
    if len(record_ids) > 1:
        for record_id in record_ids:
            test = np.flatnonzero(item.record_id == record_id)
            train = np.flatnonzero(item.record_id != record_id)
            folds.append({"fold_id": record_id, "mode": "leave_one_record_out", "train": train, "test": test,
                          "test_record_or_block": record_id})
    else:
        # Kept for protocol completeness; no included Daphnet subject enters this branch.
        record = item.records[0]
        boundaries = np.linspace(0, len(record.y), 6, dtype=int)
        centers = item.start + WINDOW // 2
        for block in range(5):
            low, high = boundaries[block], boundaries[block + 1]
            test = np.flatnonzero((centers >= low) & (centers < high))
            purge_low, purge_high = max(0, low - GUARD), min(len(record.y), high + GUARD)
            train = np.flatnonzero((centers < purge_low) | (centers >= purge_high))
            folds.append({"fold_id": f"block{block}", "mode": "five_chronological_blocks", "train": train,
                          "test": test, "test_record_or_block": f"block{block}"})
    return folds


def assign_records_to_inner(item: SubjectWindows, train: np.ndarray) -> np.ndarray | None:
    records = sorted(set(item.record_id[train].tolist()))
    if len(records) < INNER_K:
        return None
    counts = {record: (int(np.sum(item.record_id[train] == record)),
                       int(np.sum(item.label[train][item.record_id[train] == record]))) for record in records}
    best: tuple[float, tuple[int, ...]] | None = None
    for assignment in itertools.product(range(INNER_K), repeat=len(records)):
        if set(assignment) != set(range(INNER_K)):
            continue
        total = np.zeros(INNER_K)
        positive = np.zeros(INNER_K)
        for record, fold in zip(records, assignment):
            total[fold] += counts[record][0]
            positive[fold] += counts[record][1]
        score = float(np.std(total) / max(np.mean(total), 1.0) + np.std(positive) / max(np.mean(positive), 1.0)
                      + 5.0 * np.sum(positive == 0))
        candidate = (score, assignment)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise AssertionError("no inner record assignment")
    output = np.full(len(item.label), -1, dtype=np.int8)
    for record, fold in zip(records, best[1]):
        output[train[item.record_id[train] == record]] = fold
    return output


def assign_time_inner(item: SubjectWindows, train: np.ndarray) -> np.ndarray:
    output = np.full(len(item.label), -1, dtype=np.int8)
    for record_id in sorted(set(item.record_id[train].tolist())):
        indices = train[item.record_id[train] == record_id]
        starts = item.start[indices]
        low = int(np.min(starts))
        high = int(np.max(starts) + WINDOW)
        boundaries = np.linspace(low, high, INNER_K + 1)
        centers = starts + WINDOW / 2.0
        for fold in range(INNER_K):
            left, right = boundaries[fold], boundaries[fold + 1]
            mask = (centers >= left) & (centers < right if fold < INNER_K - 1 else centers <= right)
            if fold > 0:
                mask &= centers >= left + GUARD
            if fold < INNER_K - 1:
                mask &= centers < right - GUARD
            output[indices[mask]] = fold
    if any(np.sum(output[train] == fold) == 0 for fold in range(INNER_K)):
        raise ValueError(f"{item.subject} time-block inner split is empty")
    return output


def make_inner_folds(item: SubjectWindows, train: np.ndarray) -> tuple[np.ndarray, str, int]:
    assignment = assign_records_to_inner(item, train)
    mode = "whole_record" if assignment is not None else "purged_chronological_block"
    assignment = assignment if assignment is not None else assign_time_inner(item, train)
    candidates = []
    for fold in range(INNER_K):
        indices = train[assignment[train] == fold]
        positives = int(np.sum(item.label[indices]))
        negatives = len(indices) - positives
        valid = positives > 0 and negatives > 0
        candidates.append((int(valid), min(positives, negatives), len(indices), -fold, fold))
    validation_fold = max(candidates)[-1]
    if max(candidates)[0] == 0:
        raise ValueError(f"{item.subject} outer training has no two-class inner validation fold")
    return assignment, mode, validation_fold


def fit_outer_scaler(item: SubjectWindows, train: np.ndarray) -> RobustScaler:
    masks: dict[int, np.ndarray] = {}
    for index in train[item.strict_clean[train]]:
        rec_idx = int(item.record_index[index])
        masks.setdefault(rec_idx, np.zeros(len(item.records[rec_idx].y), dtype=bool))
        masks[rec_idx][int(item.start[index]):int(item.start[index]) + WINDOW] = True
    chunks = [item.records[rec_idx].x[mask] for rec_idx, mask in masks.items() if np.any(mask)]
    if not chunks:
        raise ValueError(f"{item.subject} outer fold has no pure Non-FoG scaler points")
    values = np.concatenate(chunks).astype(np.float64)
    center = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25, 75], axis=0)
    scale = q75 - q25
    if np.any(scale <= 1e-6):
        raise ValueError("degenerate outer scaler")
    return RobustScaler(center.astype(np.float32), scale.astype(np.float32))


def nbm_train_validation(item: SubjectWindows, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.asarray(candidates, dtype=np.int64)
    calibration: list[int] = []
    training: list[int] = []
    for record_id in sorted(set(item.record_id[candidates].tolist())):
        values = candidates[item.record_id[candidates] == record_id]
        values = values[np.argsort(item.start[values])]
        split = max(1, int(math.floor(0.8 * len(values))))
        boundary = int(item.start[values[min(split, len(values) - 1)]])
        training.extend(values[item.start[values] + WINDOW <= boundary - GUARD].tolist())
        calibration.extend(values[item.start[values] >= boundary].tolist())
    if len(training) < 16 or len(calibration) < 4:
        order = candidates[np.argsort(item.start[candidates])]
        split = max(16, int(0.8 * len(order)))
        split = min(split, len(order) - 4)
        training = order[:split].tolist()
        calibration = order[split:].tolist()
    return np.asarray(training, dtype=np.int64), np.asarray(calibration, dtype=np.int64)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = 2 * dilation
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(8, channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(8, channels), nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.network(x))


class FixedTCNClassifier(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.input = nn.Sequential(nn.Conv1d(in_channels, 64, 5, padding=2, bias=False),
                                   nn.GroupNorm(8, 64), nn.GELU())
        self.blocks = nn.Sequential(*(TCNBlock(64, dilation, 0.2) for dilation in (1, 2, 4, 8)))
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.input(x))
        pooled = torch.cat((features.mean(dim=-1), features.amax(dim=-1)), dim=1)
        return self.head(pooled).squeeze(1)

    def architecture_config(self) -> dict[str, Any]:
        return {"name": "fixed_tcn", "in_channels": self.in_channels, "hidden_channels": 64,
                "kernel_size": 5, "dilations": [1, 2, 4, 8], "dropout": 0.2,
                "normalization": "GroupNorm(8)", "activation": "GELU",
                "pooling": ["global_average", "global_max"],
                "parameter_count": sum(parameter.numel() for parameter in self.parameters())}


def pair_loader(inputs: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    x = torch.from_numpy(np.ascontiguousarray(inputs.transpose(0, 2, 1))).float()
    y = torch.from_numpy(np.asarray(targets, dtype=np.float32))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0, drop_last=False)


@torch.no_grad()
def predict_nbm(model: nn.Module, inputs: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    dummy = np.zeros(len(inputs), dtype=np.float32)
    for batch, _ in pair_loader(inputs, dummy, 128, False, 0):
        prediction, _ = model(batch.to(device))
        values.append(prediction.transpose(1, 2).cpu().numpy().astype(np.float32))
    return np.concatenate(values)


def train_nbm(inputs: np.ndarray, item: SubjectWindows, candidate_indices: np.ndarray, scaler: RobustScaler,
              run_dir: Path, seed: int, device: torch.device, max_epochs: int, patience: int) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = run_dir / "nbm_best.pt"
    log_path = run_dir / "training_log_nbm.csv"
    model = a1b.ContextM3(WINDOW).to(device)
    if checkpoint.exists() and log_path.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        return model, dict(payload["training"])
    run_dir.mkdir(parents=True, exist_ok=True)
    train_indices, val_indices = nbm_train_validation(item, candidate_indices)
    train_x = scaler.transform(item.raw[train_indices])
    val_x = scaler.transform(item.raw[val_indices])
    seed_everything(seed)
    model = a1b.ContextM3(WINDOW).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    batches = a1b.pair_loader(train_x, train_x, shuffle=True, seed=seed, workers=0)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    bad = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in batches:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(batch_x)
            loss = a1b.structural_loss("L4", prediction, batch_y)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient):
                raise FloatingPointError("non-finite NBM gradient")
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
        train_loss = total / count
        validation = a1b.evaluate_loss(model, val_x, val_x, "L4", device)
        improved = validation < best_loss - 1e-8
        if improved:
            best_loss = validation
            best_epoch = epoch
            best_state = base.clone_state(model)
            bad = 0
        else:
            bad += 1
        last_epoch = epoch
        if epoch == 1 or epoch % 10 == 0 or improved:
            history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation,
                            "improved": improved, "bad_epochs": bad})
        if bad >= patience:
            break
    if best_state is None:
        raise AssertionError("NBM produced no checkpoint")
    training = {"seed": seed, "best_epoch": best_epoch, "last_epoch": last_epoch,
                "best_validation_loss": best_loss, "elapsed_seconds": time.perf_counter() - started,
                "train_windows": len(train_x), "validation_windows": len(val_x), "loss": "L4"}
    torch.save({"model_state": best_state, "training": training,
                "train_window_keys": item.keys[train_indices].tolist(),
                "validation_window_keys": item.keys[val_indices].tolist()}, checkpoint)
    write_csv(log_path, history)
    model.load_state_dict(best_state)
    return model, training


@torch.no_grad()
def predict_classifier(model: nn.Module, inputs: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    dummy = np.zeros(len(inputs), dtype=np.float32)
    for batch, _ in pair_loader(inputs, dummy, 256, False, 0):
        output.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def safe_pr_auc(y: np.ndarray, probability: np.ndarray) -> float:
    return float(average_precision_score(y, probability)) if np.any(y == 1) else math.nan


def safe_roc_auc(y: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else math.nan


def select_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, probability)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(f1 == np.nanmax(f1))
    return float(thresholds[best[-1]])


def binary_metrics(y: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    precision_value = tp / max(tp + fp, 1)
    recall_value = tp / max(tp + fn, 1)
    f1_value = 2 * precision_value * recall_value / max(precision_value + recall_value, 1e-12)
    specificity = tn / max(tn + fp, 1)
    denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0))
    mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0.0
    return {"pr_auc": safe_pr_auc(y, probability), "roc_auc": safe_roc_auc(y, probability),
            "fog_f1": float(f1_value), "recall": float(recall_value), "precision": float(precision_value),
            "specificity": float(specificity), "balanced_accuracy": float((recall_value + specificity) / 2.0),
            "mcc": float(mcc), "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp), "prevalence": float(np.mean(y)), "windows": len(y)}


def train_classifier(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
                     run_dir: Path, seed: int, device: torch.device, max_epochs: int, patience: int,
                     pos_weight: float) -> tuple[nn.Module, dict[str, Any], np.ndarray]:
    checkpoint = run_dir / "tcn_best.pt"
    if checkpoint.exists() and (run_dir / "training_log_tcn.csv").exists():
        model = FixedTCNClassifier(train_x.shape[2]).to(device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        val_probability = predict_classifier(model, val_x, device)
        return model, dict(payload["training"]), val_probability
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    model = FixedTCNClassifier(train_x.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    batches = pair_loader(train_x, train_y, 128, True, seed)
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    bad = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in batches:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient):
                raise FloatingPointError("non-finite TCN gradient")
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
        probability = predict_classifier(model, val_x, device)
        score = safe_pr_auc(val_y, probability)
        improved = score > best_score + 1e-8
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = base.clone_state(model)
            bad = 0
        else:
            bad += 1
        last_epoch = epoch
        history.append({"epoch": epoch, "train_bce": total / count, "validation_pr_auc": score,
                        "improved": improved, "bad_epochs": bad})
        if bad >= patience:
            break
    if best_state is None:
        raise AssertionError("classifier produced no checkpoint")
    training = {"seed": seed, "best_epoch": best_epoch, "last_epoch": last_epoch,
                "best_validation_pr_auc": best_score, "pos_weight": pos_weight,
                "elapsed_seconds": time.perf_counter() - started,
                "train_windows": len(train_x), "validation_windows": len(val_x),
                "architecture": model.architecture_config()}
    torch.save({"model_state": best_state, "training": training}, checkpoint)
    torch.save({"model_state": base.clone_state(model), "training": training}, run_dir / "tcn_last.pt")
    write_csv(run_dir / "training_log_tcn.csv", history)
    model.load_state_dict(best_state)
    return model, training, predict_classifier(model, val_x, device)


def representation_arrays(x: np.ndarray, reconstruction: np.ndarray) -> dict[str, np.ndarray]:
    residual = x - reconstruction
    delta = np.diff(residual, axis=1, prepend=residual[:, :1, :])
    r5 = np.concatenate((residual, np.abs(residual), delta), axis=2).astype(np.float32)
    return {"B0": np.ascontiguousarray(x.astype(np.float32)),
            "B1": np.ascontiguousarray(residual.astype(np.float32)),
            "B2": np.ascontiguousarray(r5),
            "B3": np.ascontiguousarray(np.concatenate((x, r5), axis=2).astype(np.float32))}


def nrmse_summary(actual: np.ndarray, prediction: np.ndarray) -> float:
    if len(actual) == 0:
        return math.nan
    rmse = np.sqrt(np.mean(np.square(actual - prediction), axis=1))
    scale = np.std(actual, axis=1) + 1e-6
    return float(np.median(rmse / scale))


def event_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    detected = 0
    total_events = 0
    latencies: list[float] = []
    alarm_durations: list[float] = []
    false_alarm_episodes = 0
    nonfog_windows = 0
    for record_id in sorted({str(row["record_id"]) for row in rows}):
        record = sorted((row for row in rows if str(row["record_id"]) == record_id), key=lambda row: int(row["window_start"]))
        if not record:
            continue
        starts = np.asarray([int(row["window_start"]) for row in record])
        truth = np.asarray([int(row["y_true"]) for row in record])
        prediction = np.asarray([int(row["y_pred"]) for row in record])

        def episodes(mask: np.ndarray) -> list[np.ndarray]:
            selected = np.flatnonzero(mask)
            if not len(selected):
                return []
            groups = np.split(selected, np.flatnonzero(np.diff(starts[selected]) > STRIDE) + 1)
            return [group for group in groups if len(group)]

        truth_events = episodes(truth == 1)
        total_events += len(truth_events)
        for group in truth_events:
            hits = group[prediction[group] == 1]
            if len(hits):
                detected += 1
                latencies.append(float((starts[hits[0]] - starts[group[0]]) / FS))
        nonfog_windows += int(np.sum(truth == 0))
        false_alarm_episodes += len(episodes((truth == 0) & (prediction == 1)))
        for group in episodes(prediction == 1):
            alarm_durations.append(float((starts[group[-1]] - starts[group[0]]) / FS + WINDOW / FS))
    nonfog_minutes = nonfog_windows * STRIDE / FS / 60.0
    return {"event_sensitivity": detected / max(total_events, 1), "detected_events": detected,
            "total_events": total_events, "false_alarm_episodes": false_alarm_episodes,
            "false_alarms_per_minute": false_alarm_episodes / max(nonfog_minutes, 1e-12),
            "median_detection_latency_seconds": float(np.median(latencies)) if latencies else math.nan,
            "average_alarm_duration_seconds": float(np.mean(alarm_durations)) if alarm_durations else 0.0}


def hardlink_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def scaler_payload(scaler: RobustScaler) -> dict[str, Any]:
    return {"center": scaler.center.tolist(), "scale": scaler.scale.tolist(),
            "fit": "unique raw points covered by outer-training strict clean Non-FoG windows",
            "test_statistics_used": False}


def prepare_fold_representations(item: SubjectWindows, fold: dict[str, Any], fold_dir: Path,
                                 device: torch.device, nbm_epochs: int, nbm_patience: int) -> dict[str, Any]:
    cache = fold_dir / "representations.npz"
    metadata_path = fold_dir / "representation_metadata.json"
    if cache.exists() and metadata_path.exists():
        arrays = dict(np.load(cache, allow_pickle=False))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {"arrays": arrays, "metadata": metadata}
    train = np.asarray(fold["train"], dtype=np.int64)
    test = np.asarray(fold["test"], dtype=np.int64)
    inner, inner_mode, validation_fold = make_inner_folds(item, train)
    scaler = fit_outer_scaler(item, train)
    scaled_train = scaler.transform(item.raw[train])
    scaled_test = scaler.transform(item.raw[test])
    recon_oof = np.full_like(scaled_train, np.nan, dtype=np.float32)
    train_position = {int(index): position for position, index in enumerate(train)}
    oof_manifests: list[dict[str, Any]] = []
    for held in range(INNER_K):
        held_indices = train[inner[train] == held]
        candidate = train[(inner[train] >= 0) & (inner[train] != held) & item.strict_clean[train]]
        model_dir = fold_dir / "nbm_oof_models" / f"inner{held}"
        model, training = train_nbm(scaled_train, item, candidate, scaler, model_dir,
                                    NBM_SEED + stable_int(f"{item.subject}:{fold['fold_id']}:{held}") % 100000,
                                    device, nbm_epochs, nbm_patience)
        held_x = scaler.transform(item.raw[held_indices])
        prediction = predict_nbm(model, held_x, device)
        for local, raw_index in enumerate(held_indices):
            recon_oof[train_position[int(raw_index)]] = prediction[local]
        checkpoint_payload = torch.load(model_dir / "nbm_best.pt", map_location="cpu", weights_only=False)
        train_keys = set(checkpoint_payload["train_window_keys"])
        validation_keys = set(checkpoint_payload["validation_window_keys"])
        held_keys = set(item.keys[held_indices].tolist())
        if held_keys & (train_keys | validation_keys):
            raise AssertionError("OOF held windows leaked into NBM model selection")
        manifest = {"held_inner_fold": held, "held_windows": len(held_indices),
                    "nbm_candidate_clean_windows": len(candidate), "training": training,
                    "held_window_keys": sorted(held_keys), "train_window_keys": sorted(train_keys),
                    "validation_window_keys": sorted(validation_keys), "overlap_count": 0}
        write_json(model_dir / "split_manifest.json", manifest)
        oof_manifests.append(manifest)
        del model
    eligible_positions = np.flatnonzero(inner[train] >= 0)
    if not np.isfinite(recon_oof[eligible_positions]).all():
        raise AssertionError("OOF reconstruction contains missing values")
    final_candidate = train[item.strict_clean[train]]
    final_dir = fold_dir / "final_nbm_models"
    final_model, final_training = train_nbm(scaled_train, item, final_candidate, scaler, final_dir,
                                            NBM_SEED + stable_int(f"{item.subject}:{fold['fold_id']}:final") % 100000,
                                            device, nbm_epochs, nbm_patience)
    test_reconstruction = predict_nbm(final_model, scaled_test, device)
    clean_test = item.strict_clean[test]
    reconstruction_nrmse = nrmse_summary(scaled_test[clean_test], test_reconstruction[clean_test])
    arrays: dict[str, np.ndarray] = {
        "outer_train_indices": train, "outer_test_indices": test,
        "inner_fold": inner[train], "validation_inner_fold": np.asarray([validation_fold], dtype=np.int8),
        "train_x": scaled_train, "train_y": item.label[train], "train_reconstruction_oof": recon_oof,
        "test_x": scaled_test, "test_y": item.label[test], "test_reconstruction": test_reconstruction,
        "train_record_id": item.record_id[train], "test_record_id": item.record_id[test],
        "train_start": item.start[train], "test_start": item.start[test],
    }
    temporary = cache.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, cache)
    with (fold_dir / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)
    write_json(fold_dir / "scaler.json", scaler_payload(scaler))
    metadata = {"subject_id": item.subject, "fold_id": fold["fold_id"], "outer_mode": fold["mode"],
                "test_record_or_block": fold["test_record_or_block"], "outer_train_windows": len(train),
                "outer_test_windows": len(test), "inner_mode": inner_mode,
                "classifier_validation_inner_fold": validation_fold, "inner_oof": oof_manifests,
                "final_nbm_training": final_training, "test_clean_reconstruction_nrmse": reconstruction_nrmse,
                "nbm_seed_fixed": NBM_SEED, "test_used_for_scaler_or_training": False}
    write_json(metadata_path, metadata)
    split_rows = []
    for role, indices in (("outer_train", train), ("outer_test", test)):
        for index in indices:
            split_rows.append({"subject_id": item.subject, "fold_id": fold["fold_id"], "role": role,
                               "window_key": item.keys[index], "record_id": item.record_id[index],
                               "window_start": int(item.start[index]), "label": int(item.label[index]),
                               "strict_clean_nonfog": bool(item.strict_clean[index]),
                               "inner_fold": int(inner[index]) if role == "outer_train" else "NOT_APPLICABLE"})
    write_csv(fold_dir / "split_manifest.csv", split_rows)
    return {"arrays": arrays, "metadata": metadata}


def run_outer_fold(item: SubjectWindows, fold: dict[str, Any], root: Path, device: torch.device,
                   nbm_epochs: int, nbm_patience: int, classifier_epochs: int,
                   classifier_patience: int) -> list[dict[str, Any]]:
    fold_dir = root / "splits" / "outer_folds" / item.subject / str(fold["fold_id"])
    prepared = prepare_fold_representations(item, fold, fold_dir, device, nbm_epochs, nbm_patience)
    arrays = prepared["arrays"]
    train_x = arrays["train_x"]
    test_x = arrays["test_x"]
    train_representations = representation_arrays(train_x, arrays["train_reconstruction_oof"])
    test_representations = representation_arrays(test_x, arrays["test_reconstruction"])
    inner = arrays["inner_fold"]
    validation_fold = int(arrays["validation_inner_fold"][0])
    train_mask = (inner >= 0) & (inner != validation_fold)
    val_mask = inner == validation_fold
    eligible = inner >= 0
    train_y = arrays["train_y"].astype(int)
    test_y = arrays["test_y"].astype(int)
    positives = int(np.sum(train_y[eligible]))
    negatives = int(np.sum(1 - train_y[eligible]))
    pos_weight = negatives / max(positives, 1)
    summaries: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in SEEDS:
            run_dir = root / METHOD_DIRS[method] / item.subject / str(fold["fold_id"]) / f"seed{seed}"
            metrics_path = run_dir / "run_metrics.json"
            predictions_path = run_dir / "test_predictions.csv"
            if metrics_path.exists() and predictions_path.exists():
                summaries.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                continue
            model, training, val_probability = train_classifier(
                train_representations[method][train_mask], train_y[train_mask],
                train_representations[method][val_mask], train_y[val_mask], run_dir,
                seed, device, classifier_epochs, classifier_patience, pos_weight,
            )
            threshold = select_threshold(train_y[val_mask], val_probability)
            test_probability = predict_classifier(model, test_representations[method], device)
            test_prediction = (test_probability >= threshold).astype(int)
            fixed_prediction = (test_probability >= 0.5).astype(int)
            metrics = binary_metrics(test_y, test_probability, test_prediction)
            fixed_metrics = binary_metrics(test_y, test_probability, fixed_prediction)
            val_prediction = (val_probability >= threshold).astype(int)
            val_rows = [{"subject_id": item.subject, "fold_id": fold["fold_id"], "method": method,
                         "seed": seed, "record_id": arrays["train_record_id"][np.flatnonzero(val_mask)[row]],
                         "window_start": int(arrays["train_start"][np.flatnonzero(val_mask)[row]]),
                         "y_true": int(train_y[val_mask][row]), "y_prob": float(val_probability[row]),
                         "y_pred": int(val_prediction[row]), "threshold": threshold}
                        for row in range(np.sum(val_mask))]
            test_rows = [{"subject_id": item.subject, "fold_id": fold["fold_id"], "method": method,
                          "seed": seed, "record_id": arrays["test_record_id"][row],
                          "block_id": fold["test_record_or_block"], "window_start": int(arrays["test_start"][row]),
                          "y_true": int(test_y[row]), "y_prob": float(test_probability[row]),
                          "y_pred": int(test_prediction[row]), "y_pred_0p5": int(fixed_prediction[row]),
                          "threshold": threshold}
                         for row in range(len(test_y))]
            events = event_metrics(test_rows)
            result = {"stage": "outer_fold_test", "subject_id": item.subject, "fold_id": fold["fold_id"],
                      "test_record_or_block": fold["test_record_or_block"], "method": method,
                      "method_name": METHOD_NAMES[method], "seed": seed, "threshold": threshold,
                      "threshold_selection": "inner validation F1", "test_used_for_threshold": False,
                      "pos_weight": pos_weight, "nbm_seed": NBM_SEED, "training": training,
                      "metrics": metrics, "fixed_0p5_metrics": fixed_metrics, "event_metrics": events,
                      "test_clean_reconstruction_nrmse": prepared["metadata"]["test_clean_reconstruction_nrmse"]}
            write_json(run_dir / "config.json", {"experiment": EXPERIMENT, "method": method,
                                                  "method_name": METHOD_NAMES[method], "input_channels": METHOD_CHANNELS[method],
                                                  "classifier": model.architecture_config(), "seed": seed,
                                                  "outer_fold": fold["fold_id"], "test_record": fold["test_record_or_block"],
                                                  "test_used_for_selection": False})
            write_json(run_dir / "split_manifest.json", {"common_manifest": str(fold_dir / "split_manifest.csv"),
                                                          "classifier_train_inner_folds": sorted(set(inner[train_mask].tolist())),
                                                          "classifier_validation_inner_fold": validation_fold,
                                                          "outer_test": fold["test_record_or_block"]})
            shutil.copy2(fold_dir / "scaler.pkl", run_dir / "scaler.pkl")
            hardlink_or_copy(fold_dir / "final_nbm_models" / "nbm_best.pt", run_dir / "nbm_best.pt")
            shutil.copy2(fold_dir / "final_nbm_models" / "training_log_nbm.csv", run_dir / "training_log_nbm.csv")
            write_csv(run_dir / "validation_predictions.csv", val_rows)
            write_csv(predictions_path, test_rows)
            write_csv(run_dir / "window_metrics.csv", test_rows)
            write_csv(run_dir / "event_metrics.csv", [{"subject_id": item.subject, "fold_id": fold["fold_id"],
                                                         "method": method, "seed": seed, **events}])
            write_json(metrics_path, result)
            summaries.append(result)
            del model
            print(f"CLASSIFIER DONE {item.subject}/{fold['fold_id']} {method} seed={seed} AP={metrics['pr_auc']}", flush=True)
    return summaries


def bootstrap_ci(values: np.ndarray, statistic: str, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    draws = values[indices]
    estimates = np.mean(draws, axis=1) if statistic == "mean" else np.median(draws, axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def paired_rank_biserial(difference: np.ndarray) -> float:
    difference = np.asarray(difference, dtype=float)
    difference = difference[np.isfinite(difference) & (difference != 0)]
    if not len(difference):
        return 0.0
    ranks = pd.Series(np.abs(difference)).rank(method="average").to_numpy()
    return float((np.sum(ranks[difference > 0]) - np.sum(ranks[difference < 0])) / np.sum(ranks))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def aggregate_results(root: Path, bootstrap_samples: int) -> dict[str, Any]:
    for directory in ("metrics", "predictions", "tables", "figures", "reports"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    subject_seed_rows: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for method in METHODS:
            for seed in SEEDS:
                paths = sorted((root / METHOD_DIRS[method] / subject).glob(f"*/seed{seed}/test_predictions.csv"))
                if not paths:
                    raise FileNotFoundError(f"missing predictions {subject} {method} seed={seed}")
                rows = [row for path in paths for row in read_csv(path)]
                keys = [(row["record_id"], int(row["window_start"])) for row in rows]
                if len(keys) != len(set(keys)):
                    raise AssertionError(f"duplicate outer predictions {subject} {method} seed={seed}")
                y = np.asarray([int(row["y_true"]) for row in rows])
                probability = np.asarray([float(row["y_prob"]) for row in rows])
                prediction = np.asarray([int(row["y_pred"]) for row in rows])
                metrics = binary_metrics(y, probability, prediction)
                events = event_metrics(rows)
                subject_seed_rows.append({"subject_id": subject, "method": method,
                                          "method_name": METHOD_NAMES[method], "seed": seed, **metrics, **events})
                for row in rows:
                    all_prediction_rows.append({**row, "subject_id": subject, "method": method, "seed": seed,
                                                "window_start": int(row["window_start"]), "y_true": int(row["y_true"]),
                                                "y_prob": float(row["y_prob"]), "y_pred": int(row["y_pred"])})
                write_csv(root / "predictions" / subject / f"{method}_seed{seed}.csv", rows)
    write_csv(root / "metrics" / "subject_seed_metrics.csv", subject_seed_rows)
    frame = pd.DataFrame(subject_seed_rows)
    numeric = list(CLASSIFICATION_METRICS) + ["tn", "fp", "fn", "tp", "prevalence", "windows",
                "event_sensitivity", "detected_events", "total_events", "false_alarm_episodes",
                "false_alarms_per_minute", "median_detection_latency_seconds", "average_alarm_duration_seconds"]
    subject_median = frame.groupby(["subject_id", "method", "method_name"], as_index=False)[numeric].median()
    subject_median.to_csv(root / "tables" / "subject_level_main_results.csv", index=False, encoding="utf-8-sig")
    macro_rows: list[dict[str, Any]] = []
    for method in METHODS:
        values = subject_median[subject_median.method == method]
        row: dict[str, Any] = {"method": method, "method_name": METHOD_NAMES[method], "subjects": len(values)}
        for metric in CLASSIFICATION_METRICS:
            array = values[metric].to_numpy(float)
            ci_low, ci_high = bootstrap_ci(array, "mean", bootstrap_samples,
                                           20260804 + stable_int(method + metric) % 100000)
            row[f"macro_{metric}"] = float(np.nanmean(array))
            row[f"median_{metric}"] = float(np.nanmedian(array))
            row[f"iqr_{metric}"] = float(np.nanpercentile(array, 75) - np.nanpercentile(array, 25))
            row[f"macro_{metric}_ci_low"] = ci_low
            row[f"macro_{metric}_ci_high"] = ci_high
        macro_rows.append(row)
    write_csv(root / "tables" / "all_subject_summary.csv", macro_rows)

    subset_rows: list[dict[str, Any]] = []
    for subset_name, subjects in (("all_8", SUBJECTS), ("A1_pass_5", A1_PASS), ("difficult_3", DIFFICULT)):
        for method in METHODS:
            values = subject_median[(subject_median.method == method) & subject_median.subject_id.isin(subjects)]
            subset_rows.append({"subset": subset_name, "method": method, "subjects": len(values),
                                **{f"macro_{metric}": float(values[metric].mean()) for metric in CLASSIFICATION_METRICS}})
    write_csv(root / "tables" / "subset_sensitivity_analysis.csv", subset_rows)

    comparisons = (("B1", "B0"), ("B2", "B1"), ("B2", "B0"), ("B3", "B2"), ("B3", "B0"))
    statistical_rows: list[dict[str, Any]] = []
    for metric in ("pr_auc", "fog_f1"):
        metric_rows: list[dict[str, Any]] = []
        for better, reference in comparisons:
            left = subject_median[subject_median.method == better].set_index("subject_id")[metric]
            right = subject_median[subject_median.method == reference].set_index("subject_id")[metric]
            difference = (left - right).reindex(SUBJECTS).to_numpy(float)
            if np.allclose(difference, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                test = wilcoxon(difference, zero_method="wilcox", alternative="two-sided", method="auto")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            ci_low, ci_high = bootstrap_ci(difference, "median", bootstrap_samples,
                                           20260804 + stable_int(metric + better + reference) % 100000)
            metric_rows.append({"metric": metric, "comparison": f"{better}-{reference}",
                                "better_method": better, "reference_method": reference,
                                "median_difference": float(np.median(difference)),
                                "mean_difference": float(np.mean(difference)),
                                "bootstrap_median_ci_low": ci_low, "bootstrap_median_ci_high": ci_high,
                                "improved_subjects": int(np.sum(difference > 0)), "subjects": len(difference),
                                "wilcoxon_statistic": statistic, "p_value": p_value,
                                "paired_rank_biserial": paired_rank_biserial(difference)})
        adjusted = holm_adjust([row["p_value"] for row in metric_rows])
        for row, value in zip(metric_rows, adjusted):
            row["holm_p_value"] = value
        statistical_rows.extend(metric_rows)
    write_csv(root / "tables" / "paired_statistical_comparisons.csv", statistical_rows)

    direction_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        values = subject_median[subject_median.subject_id == subject].set_index("method")
        pr = values["pr_auc"]
        best = str(pr.idxmax())
        direction_rows.append({"subject_id": subject, "B1_minus_B0_pr_auc": pr.B1 - pr.B0,
                               "B2_minus_B1_pr_auc": pr.B2 - pr.B1, "B2_minus_B0_pr_auc": pr.B2 - pr.B0,
                               "B3_minus_B2_pr_auc": pr.B3 - pr.B2, "best_method": best,
                               "best_pr_auc": pr[best]})
    write_csv(root / "tables" / "subject_improvement_directions.csv", direction_rows)

    predictions = pd.DataFrame(all_prediction_rows)
    key_columns = ["subject_id", "method", "record_id", "window_start"]
    pooled = predictions.groupby(key_columns, as_index=False).agg(y_true=("y_true", "first"),
                                                                  y_prob=("y_prob", "median"),
                                                                  y_pred=("y_pred", lambda x: int(np.median(x) >= 0.5)))
    pooled.to_csv(root / "predictions" / "seed_median_pooled_predictions.csv", index=False, encoding="utf-8-sig")
    pooled_rows = []
    for method in METHODS:
        values = pooled[pooled.method == method]
        pooled_rows.append({"method": method, **binary_metrics(values.y_true.to_numpy(), values.y_prob.to_numpy(),
                                                                values.y_pred.to_numpy())})
    write_csv(root / "tables" / "pooled_window_metrics.csv", pooled_rows)

    reconstruction_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        paths = sorted((root / "splits" / "outer_folds" / subject).glob("*/representation_metadata.json"))
        values = [json.loads(path.read_text(encoding="utf-8"))["test_clean_reconstruction_nrmse"] for path in paths]
        reconstruction_rows.append({"subject_id": subject, "median_test_clean_nrmse": float(np.nanmedian(values)),
                                    "B2_minus_B0_pr_auc": next(row["B2_minus_B0_pr_auc"] for row in direction_rows
                                                               if row["subject_id"] == subject)})
    write_csv(root / "tables" / "reconstruction_quality_vs_residual_gain.csv", reconstruction_rows)

    success = {
        "H1_B1_effective": bool(next(r for r in macro_rows if r["method"] == "B1")["macro_pr_auc"] >
                                next(r for r in macro_rows if r["method"] == "B0")["macro_pr_auc"]
                                and next(r for r in statistical_rows if r["metric"] == "pr_auc" and r["comparison"] == "B1-B0")["improved_subjects"] >= 5
                                and next(r for r in macro_rows if r["method"] == "B1")["macro_fog_f1"] >=
                                next(r for r in macro_rows if r["method"] == "B0")["macro_fog_f1"]),
        "H2_B2_effective": bool(next(r for r in macro_rows if r["method"] == "B2")["macro_pr_auc"] >
                                next(r for r in macro_rows if r["method"] == "B1")["macro_pr_auc"]
                                and next(r for r in statistical_rows if r["metric"] == "pr_auc" and r["comparison"] == "B2-B1")["improved_subjects"] >= 5
                                and next(r for r in statistical_rows if r["metric"] == "pr_auc" and r["comparison"] == "B2-B1")["bootstrap_median_ci_low"] > 0
                                and next(r for r in statistical_rows if r["metric"] == "pr_auc" and r["comparison"] == "B2-B1")["holm_p_value"] < 0.05),
        "H3_B3_effective": bool(next(r for r in macro_rows if r["method"] == "B3")["macro_pr_auc"] >
                                max(next(r for r in macro_rows if r["method"] == "B2")["macro_pr_auc"],
                                    next(r for r in macro_rows if r["method"] == "B0")["macro_pr_auc"])
                                and next(r for r in statistical_rows if r["metric"] == "pr_auc" and r["comparison"] == "B3-B0")["improved_subjects"] >= 5),
    }
    result = {"experiment": EXPERIMENT, "completed_utc": datetime.now(timezone.utc).isoformat(),
              "subjects": list(SUBJECTS), "methods": list(METHODS), "classifier_seeds": list(SEEDS),
              "nbm_seed_fixed": NBM_SEED, "macro_results": macro_rows, "subset_results": subset_rows,
              "paired_statistics": statistical_rows, "success_criteria": success,
              "test_data_used_for_selection": False}
    write_json(root / "FINAL_RESULTS.json", result)
    make_figures(root, subject_median, macro_rows, pooled, pd.DataFrame(direction_rows),
                 pd.DataFrame(reconstruction_rows))
    write_report(root, result, subject_median, pd.DataFrame(direction_rows))
    return result


def make_figures(root: Path, subject: pd.DataFrame, macro: list[dict[str, Any]], pooled: pd.DataFrame,
                 direction: pd.DataFrame, reconstruction: pd.DataFrame) -> None:
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    macro_frame = pd.DataFrame(macro)
    metrics = ["pr_auc", "roc_auc", "fog_f1", "balanced_accuracy", "mcc"]
    long = macro_frame.melt(id_vars="method", value_vars=[f"macro_{m}" for m in metrics],
                            var_name="metric", value_name="value")
    long["metric"] = long.metric.str.replace("macro_", "", regex=False)
    fig, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics)); width = 0.18
    for offset, method in enumerate(METHODS):
        values = [float(macro_frame.loc[macro_frame.method == method, f"macro_{metric}"].iloc[0]) for metric in metrics]
        axis.bar(x + (offset - 1.5) * width, values, width, label=method)
    axis.set_xticks(x, metrics); axis.set_ylim(0, 1); axis.legend(); fig.tight_layout()
    fig.savefig(figure_dir / FIGURE_NAMES[0], dpi=180); plt.close(fig)

    for metric, name in (("pr_auc", FIGURE_NAMES[1]), ("fog_f1", FIGURE_NAMES[2])):
        pivot = subject.pivot(index="subject_id", columns="method", values=metric).reindex(index=SUBJECTS, columns=METHODS)
        fig, axis = plt.subplots(figsize=(7, 5)); image = axis.imshow(pivot.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
        axis.set_xticks(np.arange(len(METHODS)), METHODS); axis.set_yticks(np.arange(len(SUBJECTS)), SUBJECTS)
        for row in range(len(SUBJECTS)):
            for column in range(len(METHODS)):
                axis.text(column, row, f"{pivot.iloc[row, column]:.3f}", ha="center", va="center", color="white")
        fig.colorbar(image, ax=axis); fig.tight_layout(); fig.savefig(figure_dir / name, dpi=180); plt.close(fig)

    for metric, name in (("pr_auc", FIGURE_NAMES[3]), ("fog_f1", FIGURE_NAMES[4])):
        pivot = subject.pivot(index="subject_id", columns="method", values=metric).reindex(index=SUBJECTS, columns=METHODS)
        plt.figure(figsize=(9, 5))
        for subject_id, row in pivot.iterrows():
            plt.plot(METHODS, row.values, marker="o", alpha=.75, label=subject_id)
        plt.ylabel(metric); plt.ylim(0, 1); plt.legend(ncol=4, fontsize=8); plt.tight_layout()
        plt.savefig(figure_dir / name, dpi=180); plt.close()

    ranks = subject.pivot(index="subject_id", columns="method", values="pr_auc").rank(axis=1, ascending=False)
    rank_values = ranks.reindex(index=SUBJECTS, columns=METHODS)
    fig, axis = plt.subplots(figsize=(7, 5)); image = axis.imshow(rank_values, cmap="viridis_r", vmin=1, vmax=4, aspect="auto")
    axis.set_xticks(np.arange(len(METHODS)), METHODS); axis.set_yticks(np.arange(len(SUBJECTS)), SUBJECTS)
    for row in range(len(SUBJECTS)):
        for column in range(len(METHODS)):
            axis.text(column, row, f"{rank_values.iloc[row, column]:.0f}", ha="center", va="center")
    fig.colorbar(image, ax=axis); fig.tight_layout(); fig.savefig(figure_dir / FIGURE_NAMES[5], dpi=180); plt.close(fig)

    plt.figure(figsize=(7, 6))
    for method in METHODS:
        values = pooled[pooled.method == method]
        precision, recall, _ = precision_recall_curve(values.y_true, values.y_prob)
        plt.plot(recall, precision, label=f"{method} AP={average_precision_score(values.y_true, values.y_prob):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.tight_layout()
    plt.savefig(figure_dir / FIGURE_NAMES[6], dpi=180); plt.close()

    plt.figure(figsize=(7, 6))
    for method in METHODS:
        values = pooled[pooled.method == method]
        fpr, tpr, _ = roc_curve(values.y_true, values.y_prob)
        plt.plot(fpr, tpr, label=f"{method} AUC={roc_auc_score(values.y_true, values.y_prob):.3f}")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.tight_layout()
    plt.savefig(figure_dir / FIGURE_NAMES[7], dpi=180); plt.close()

    for curve_type, name in (("pr", FIGURE_NAMES[8]), ("roc", FIGURE_NAMES[9])):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
        for axis, subject_id in zip(axes.flat, SUBJECTS):
            for method in METHODS:
                values = pooled[(pooled.subject_id == subject_id) & (pooled.method == method)]
                if curve_type == "pr":
                    y_axis, x_axis, _ = precision_recall_curve(values.y_true, values.y_prob)
                    axis.plot(x_axis, y_axis, label=method)
                else:
                    x_axis, y_axis, _ = roc_curve(values.y_true, values.y_prob)
                    axis.plot(x_axis, y_axis, label=method)
            axis.set_title(subject_id); axis.legend(fontsize=7)
        fig.savefig(figure_dir / name, dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for axis, method in zip(axes.flat, METHODS):
        values = pooled[pooled.method == method]
        matrix = confusion_matrix(values.y_true, values.y_pred, labels=[0, 1])
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set_xticks([0, 1], ["pred NF", "pred FoG"]); axis.set_yticks([0, 1], ["true NF", "true FoG"]); axis.set_title(method)
    fig.savefig(figure_dir / FIGURE_NAMES[10], dpi=180); plt.close(fig)

    plt.figure(figsize=(7, 5)); plt.scatter(reconstruction.median_test_clean_nrmse, reconstruction.B2_minus_B0_pr_auc)
    if len(reconstruction) >= 2:
        coefficient = np.polyfit(reconstruction.median_test_clean_nrmse, reconstruction.B2_minus_B0_pr_auc, 1)
        x_line = np.linspace(reconstruction.median_test_clean_nrmse.min(), reconstruction.median_test_clean_nrmse.max(), 100)
        plt.plot(x_line, np.polyval(coefficient, x_line))
    for _, row in reconstruction.iterrows(): plt.text(row.median_test_clean_nrmse, row.B2_minus_B0_pr_auc, row.subject_id)
    plt.tight_layout(); plt.savefig(figure_dir / FIGURE_NAMES[11], dpi=180); plt.close()

    b2 = subject[subject.method == "B2"].copy()
    plt.figure(figsize=(7, 5)); plt.scatter(b2.prevalence, b2.pr_auc, s=90)
    for _, row in b2.iterrows(): plt.text(row.prevalence, row.pr_auc, row.subject_id)
    plt.tight_layout(); plt.savefig(figure_dir / FIGURE_NAMES[12], dpi=180); plt.close()

    fig, axis = plt.subplots(figsize=(10, 5)); x = np.arange(len(SUBJECTS)); width = .18
    for offset, method in enumerate(METHODS):
        values = subject[subject.method == method].set_index("subject_id").reindex(SUBJECTS).false_alarms_per_minute
        axis.bar(x + (offset - 1.5) * width, values, width, label=method)
    axis.set_xticks(x, SUBJECTS); axis.legend(); fig.tight_layout(); fig.savefig(figure_dir / FIGURE_NAMES[13], dpi=180); plt.close(fig)

    ordered = direction.sort_values("B2_minus_B0_pr_auc")
    chosen = [ordered.iloc[0].subject_id, ordered.iloc[-1].subject_id]
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)
    for axis, subject_id in zip(axes, chosen):
        subset = pooled[(pooled.subject_id == subject_id) & pooled.method.isin(["B0", "B2"])].sort_values(["record_id", "window_start"])
        for method in ("B0", "B2"):
            values = subset[subset.method == method].reset_index(drop=True)
            axis.plot(values.y_prob.to_numpy()[:500], label=method, alpha=.85)
        truth = subset[subset.method == "B0"].reset_index(drop=True).y_true.to_numpy()[:500]
        axis.fill_between(np.arange(len(truth)), 0, truth, alpha=.15, color="red", label="FoG truth")
        axis.set_title(subject_id); axis.legend()
    fig.savefig(figure_dir / FIGURE_NAMES[14], dpi=180); plt.close(fig)


def write_report(root: Path, result: dict[str, Any], subject: pd.DataFrame, direction: pd.DataFrame) -> None:
    macro = {row["method"]: row for row in result["macro_results"]}
    best_method = max(METHODS, key=lambda method: macro[method]["macro_pr_auc"])
    lines = ["# Daphnet 全被试 NBM 残差二分类实验报告", "",
             f"生成时间（UTC）：{result['completed_utc']}", "", "## 核心结论", "",
             f"- 全8被试宏平均 PR-AUC 最佳方法：**{best_method} {METHOD_NAMES[best_method]}**，"
             f"PR-AUC={macro[best_method]['macro_pr_auc']:.4f}。",
             f"- H1（B1有效）：**{'PASS' if result['success_criteria']['H1_B1_effective'] else 'FAIL'}**。",
             f"- H2（B2有效）：**{'PASS' if result['success_criteria']['H2_B2_effective'] else 'FAIL'}**。",
             f"- H3（B3有效）：**{'PASS' if result['success_criteria']['H3_B3_effective'] else 'FAIL'}**。",
             "", "## 全被试宏平均", "",
             "| 方法 | PR-AUC | ROC-AUC | FoG F1 | Recall | Specificity | BAcc | MCC |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        row = macro[method]
        lines.append(f"| {method} | {row['macro_pr_auc']:.4f} | {row['macro_roc_auc']:.4f} | "
                     f"{row['macro_fog_f1']:.4f} | {row['macro_recall']:.4f} | {row['macro_specificity']:.4f} | "
                     f"{row['macro_balanced_accuracy']:.4f} | {row['macro_mcc']:.4f} |")
    lines += ["", "## 被试级最佳方法", "", "| 被试 | 最佳方法 | PR-AUC | B2-B0 |",
              "|---|---|---:|---:|"]
    for _, row in direction.iterrows():
        lines.append(f"| {row.subject_id} | {row.best_method} | {row.best_pr_auc:.4f} | {row.B2_minus_B0_pr_auc:+.4f} |")
    lines += ["", "## 方法边界", "",
              "- 外层测试记录不参与Scaler、NBM/TCN训练、early stopping、阈值或类别权重选择。",
              "- TCN训练用残差全部来自3折OOF NBM；外层测试残差来自仅用外层训练Non-FoG拟合的最终NBM。",
              "- NBM固定种子20260802以冻结表征；三个报告种子仅改变TCN初始化和batch顺序。",
              "- 统计单位为8名被试，重叠窗口不作为独立统计重复。",
              "- S03_seg003有效率为0，预先排除；其余完整有效记录全部进入外层留一。",
              "", "## 结果索引", "",
              "- `tables/subject_level_main_results.csv`：被试级主结果。",
              "- `tables/all_subject_summary.csv`：宏平均、IQR和bootstrap CI。",
              "- `tables/paired_statistical_comparisons.csv`：配对Wilcoxon、Holm校正与效应量。",
              "- `predictions/seed_median_pooled_predictions.csv`：种子中位外层预测。",
              "- `figures/`：模板要求的15张图片。",
              "- `FINAL_RESULTS.json`：完整机器可读结果。"]
    path = root / "reports" / "full_subject_binary_classification_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / EXPERIMENT / "full_subject_binary_experiment")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "daphnet_full_subject_nbm_residual_binary.yaml")
    parser.add_argument("--template", type=Path, default=Path(r"C:\Users\bin\Downloads\Daphnet_full_subject_NBM_residual_binary_classification_experiment_outline.md"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nbm-max-epochs", type=int, default=2000)
    parser.add_argument("--nbm-patience", type=int, default=100)
    parser.add_argument("--classifier-max-epochs", type=int, default=50)
    parser.add_argument("--classifier-patience", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--only-fold", default="", help="SUBJECT/FOLD for one resumable outer fold")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    protocol = {"experiment": EXPERIMENT, "created_utc": datetime.now(timezone.utc).isoformat(),
                "template": str(args.template.resolve()), "config": str(args.config.resolve()),
                "subjects": list(SUBJECTS), "outer": "leave-one-complete-valid-record-out",
                "inner": "3-fold record-first purged OOF", "nbm_seed_fixed": NBM_SEED,
                "classifier_seeds": list(SEEDS), "test_used_for_selection": False,
                "invalid_record_exclusion": {"S03_seg003": "valid_fraction=0"}}
    write_json(root / "splits" / "frozen_protocol.json", protocol)
    if not args.finalize_only:
        dataset = DaphnetDataset.load(args.data_dir.resolve())
        items = {subject: build_subject_windows(dataset, subject) for subject in SUBJECTS}
        split_summary: list[dict[str, Any]] = []
        all_folds: list[tuple[str, dict[str, Any]]] = []
        for subject, item in items.items():
            for fold in outer_folds(item):
                all_folds.append((subject, fold))
                split_summary.append({"subject_id": subject, "fold_id": fold["fold_id"], "mode": fold["mode"],
                                      "train_windows": len(fold["train"]), "test_windows": len(fold["test"]),
                                      "test_positive_windows": int(np.sum(item.label[fold["test"]])),
                                      "test_negative_windows": int(len(fold["test"]) - np.sum(item.label[fold["test"]]))})
        write_csv(root / "splits" / "outer_folds" / "outer_fold_summary.csv", split_summary)
        selected = all_folds
        if args.only_fold:
            subject_wanted, fold_wanted = args.only_fold.split("/", 1)
            selected = [(subject, fold) for subject, fold in all_folds
                        if subject == subject_wanted and str(fold["fold_id"]) == fold_wanted]
            if len(selected) != 1:
                raise ValueError(f"unknown --only-fold {args.only_fold}")
        if args.smoke:
            selected = selected[:1]
        for position, (subject, fold) in enumerate(selected, 1):
            print(f"OUTER {position}/{len(selected)} {subject}/{fold['fold_id']} device={device}", flush=True)
            run_outer_fold(items[subject], fold, root, device,
                           min(args.nbm_max_epochs, 3) if args.smoke else args.nbm_max_epochs,
                           min(args.nbm_patience, 2) if args.smoke else args.nbm_patience,
                           min(args.classifier_max_epochs, 2) if args.smoke else args.classifier_max_epochs,
                           min(args.classifier_patience, 1) if args.smoke else args.classifier_patience)
    if args.smoke or args.only_fold:
        print(f"PARTIAL COMPLETE {root}", flush=True)
        return
    result = aggregate_results(root, args.bootstrap_samples)
    print(f"COMPLETE {root} best={max(METHODS, key=lambda m: next(r for r in result['macro_results'] if r['method']==m)['macro_pr_auc'])}", flush=True)


if __name__ == "__main__":
    main()
