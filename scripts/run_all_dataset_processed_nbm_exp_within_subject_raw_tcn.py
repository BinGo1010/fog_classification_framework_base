#!/usr/bin/env python
"""One within-subject RAW+TCN job for processed_NBM_Exp.

The runner deliberately separates training from permanent-test evaluation.
Roles 0/1 cannot be evaluated until the launcher has sealed every classifier,
validation-selected checkpoint, and decision threshold in a global barrier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.evaluation import binary_metrics
from cnbr_fog.resume import (
    atomic_json_dump,
    atomic_npz_save,
    atomic_torch_save,
    canonical_fingerprint,
    sha256_file,
)
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    RepresentationTCNM,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import RobustScaler, set_seed


SUBJECTS = tuple(f"P{index:02d}" for index in range(1, 9))
FOLDS = (0, 1, 2)
SEEDS = (0, 52, 161, 5216, 52161)
ROLES = tuple(range(8))
WINDOW_SAMPLES = 128
SAMPLING_RATE_HZ = 64
INPUT_CHANNELS = 30
METRIC_KEYS = (
    "sensitivity",
    "precision",
    "specificity",
    "pr_auc",
    "event_sensitivity",
    "false_alarms_per_hour",
)
THRESHOLDS = tuple(float(value) for value in np.round(np.arange(0.05, 0.951, 0.01), 2))
EVENT_METRIC_VERSION = "coverage_aware.v2"
EXPERIMENT_SCHEMA = "all_dataset_within_subject_raw_tcn.v1"
BARRIER_SCHEMA = "all_dataset_within_subject_raw_tcn_barrier.v1"


@dataclass(frozen=True)
class RoleRows:
    subject_id: np.ndarray
    record_index: np.ndarray
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_seed_list(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid unique seed list: {text}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "evaluate", "seal", "aggregate"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subject", choices=SUBJECTS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tcn-max-epochs", type=int, default=5)
    parser.add_argument("--tcn-patience", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    return device


def run_dir(output_root: Path, subject: str, fold: int, seed: int) -> Path:
    return output_root / "runs" / subject / f"fold_{fold}" / f"seed_{seed}"


def require_job_args(args: argparse.Namespace) -> tuple[str, int, int]:
    if args.subject is None or args.fold is None or args.seed is None:
        raise ValueError(f"stage={args.stage} requires --subject, --fold, and --seed")
    return args.subject, int(args.fold), int(args.seed)


def load_plan(output_root: Path) -> dict[str, Any]:
    path = output_root / "EXPERIMENT_PLAN.json"
    if not path.is_file():
        raise FileNotFoundError(f"launcher plan missing: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != EXPERIMENT_SCHEMA:
        raise AssertionError(f"unexpected experiment plan schema: {plan.get('schema')}")
    return plan


def validate_plan_args(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    expected = {
        "data_dir": str(args.data_dir.resolve()),
        "batch_size": int(args.batch_size),
        "tcn_max_epochs": int(args.tcn_max_epochs),
        "tcn_patience": int(args.tcn_patience),
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise AssertionError(f"plan/worker mismatch for {key}: {plan.get(key)!r} != {value!r}")


def load_subject_rows(data_dir: Path, dataset: Any, subject: str, fold: int) -> RoleRows:
    path = data_dir / "split_indices" / f"{subject}_outer{fold}_nbm_indices.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    record_lookup = {record.record_id: index for index, record in enumerate(dataset.records)}
    with np.load(path, allow_pickle=False) as payload:
        record_id = payload["record_id"].astype("U96")
        rows = RoleRows(
            subject_id=np.full(len(record_id), subject, dtype="U3"),
            record_index=np.asarray([record_lookup[str(value)] for value in record_id], dtype=np.int32),
            record_id=record_id,
            start=payload["start_index"].astype(np.int32),
            end=payload["end_index_exclusive"].astype(np.int32),
            role=payload["role_code"].astype(np.int8),
            label=payload["y_binary"].astype(np.int8),
            window_id=payload["window_id"].astype("U160"),
        )
    audit_rows(dataset, rows, subject, fold)
    return rows


def audit_rows(dataset: Any, rows: RoleRows, subject: str, fold: int) -> None:
    if len(rows) == 0 or set(np.unique(rows.role).tolist()) != set(ROLES):
        raise AssertionError(f"{subject}/fold{fold}: all eight roles are required")
    expected_labels = np.isin(rows.role, (1, 3, 7)).astype(np.int8)
    if not np.array_equal(rows.label, expected_labels):
        raise AssertionError(f"{subject}/fold{fold}: role/label contract failed")
    if np.any(rows.end - rows.start != WINDOW_SAMPLES):
        raise AssertionError(f"{subject}/fold{fold}: non-{WINDOW_SAMPLES}-sample window")
    if len(set(rows.window_id.tolist())) != len(rows):
        raise AssertionError(f"{subject}/fold{fold}: duplicate window_id")
    point_masks: dict[int, np.ndarray] = {}
    for rec_idx, start, end, role in zip(rows.record_index, rows.start, rows.end, rows.role):
        record = dataset.records[int(rec_idx)]
        if record.subject_id != subject:
            raise AssertionError(f"foreign record {record.record_id} in {subject}")
        mask = point_masks.setdefault(int(rec_idx), np.zeros(len(record.y), dtype=np.uint8))
        mask[int(start) : int(end)] |= np.uint8(1 << int(role))
    overlap = sum(int(np.count_nonzero((mask & (mask - 1)) != 0)) for mask in point_masks.values())
    if overlap:
        raise AssertionError(f"{subject}/fold{fold}: {overlap} cross-role raw sample points")


def raw_windows(dataset: Any, rows: RoleRows) -> np.ndarray:
    values = np.empty((len(rows), WINDOW_SAMPLES, dataset.n_channels), dtype=np.float32)
    for index, (rec_idx, start, end) in enumerate(zip(rows.record_index, rows.start, rows.end)):
        values[index] = dataset.records[int(rec_idx)].x[int(start) : int(end)]
    return values


def fit_scaler_unique_role4_points(dataset: Any, rows: RoleRows) -> tuple[RobustScaler, int]:
    if not np.all(rows.role == 4) or not np.all(rows.label == 0):
        raise AssertionError("RobustScaler support must be role-4 Non-FoG only")
    masks: dict[int, np.ndarray] = {}
    for rec_idx, start, end in zip(rows.record_index, rows.start, rows.end):
        mask = masks.setdefault(int(rec_idx), np.zeros(len(dataset.records[int(rec_idx)].y), dtype=bool))
        mask[int(start) : int(end)] = True
    chunks = [dataset.records[index].x[mask] for index, mask in masks.items() if np.any(mask)]
    values = np.concatenate(chunks).astype(np.float64, copy=False)
    median = np.median(values, axis=0)
    q25, q75 = np.percentile(values, (25.0, 75.0), axis=0)
    iqr = q75 - q25
    if np.any(iqr <= 1e-6):
        raise ValueError(f"degenerate RobustScaler axes: {np.flatnonzero(iqr <= 1e-6).tolist()}")
    return RobustScaler(median.astype(np.float32), iqr.astype(np.float32)), int(len(values))


def raw_features(scaler: RobustScaler, raw: np.ndarray) -> np.ndarray:
    values = scaler.transform(raw)
    values = values - values.mean(axis=1, keepdims=True)
    maximum_axis_mean = float(np.max(np.abs(values.mean(axis=1))))
    if maximum_axis_mean > 5e-5:
        raise AssertionError(f"per-window/per-axis centering failed: {maximum_axis_mean}")
    return np.ascontiguousarray(values.transpose(0, 2, 1), dtype=np.float32)


def loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int, workers: int) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(
            torch.from_numpy(np.ascontiguousarray(x)).float(),
            torch.from_numpy(np.asarray(y, dtype=np.float32)),
        ),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for batch_x, batch_y in loader(x, y, batch_size, False, 0, 0):
        probabilities.append(torch.sigmoid(model(batch_x.to(device, non_blocking=True))).cpu().numpy())
        labels.append(batch_y.numpy())
    return np.concatenate(labels).astype(np.int8), np.concatenate(probabilities).astype(np.float64)


@torch.no_grad()
def validation_loss(model: nn.Module, x: np.ndarray, y: np.ndarray, criterion: nn.Module, device: torch.device, batch_size: int) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch_x, batch_y in loader(x, y, batch_size, False, 0, 0):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        loss = criterion(model(batch_x), batch_y)
        total += float(loss) * len(batch_x)
        count += len(batch_x)
    return total / count


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, Any]]:
    best_key = (-math.inf, -math.inf, -math.inf)
    best_threshold = 0.5
    best_metrics: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        metrics = binary_metrics(y_true, y_prob, threshold)
        key = (float(metrics["balanced_accuracy"] or 0.0), float(metrics["f1"] or 0.0), threshold)
        if key > best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


def initial_state_hash(seed: int) -> str:
    set_seed(seed)
    model = RepresentationTCNM(INPUT_CHANNELS).cpu()
    payload = {
        name: canonical_fingerprint({"shape": list(tensor.shape), "bytes": tensor.detach().numpy().tobytes().hex()})
        for name, tensor in model.state_dict().items()
    }
    return canonical_fingerprint(payload)


def train_tcn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    destination: Path,
    device: torch.device,
    seed: int,
    batch_size: int,
    workers: int,
    maximum_epochs: int,
    patience: int,
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(seed)
    model = RepresentationTCNM(INPUT_CHANNELS).to(device)
    n_nonfog = int(np.sum(train_y == 0))
    n_fog = int(np.sum(train_y == 1))
    if min(n_nonfog, n_fog) == 0:
        raise ValueError("roles 6/7 must contain both classes")
    pos_weight = n_nonfog / n_fog
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = loader(train_x, train_y, batch_size, True, seed, workers)
    checkpoint = destination / "checkpoints" / "tcn.pt"
    best_pr_auc = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite TCN gradient")
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
        train_bce = total / count
        val_bce = validation_loss(model, validation_x, validation_y, criterion, device, batch_size)
        val_true, val_prob = predict(model, validation_x, validation_y, device, batch_size)
        val_pr_auc = float(average_precision_score(val_true, val_prob))
        improved = val_pr_auc > best_pr_auc + 1e-10
        history.append({
            "epoch": epoch,
            "train_weighted_bce": train_bce,
            "validation_weighted_bce": val_bce,
            "validation_pr_auc": val_pr_auc,
            "improved": improved,
        })
        if improved:
            best_pr_auc = val_pr_auc
            best_epoch = epoch
            stale = 0
            atomic_torch_save({
                "schema": EXPERIMENT_SCHEMA,
                "model_state": model.state_dict(),
                "seed": seed,
                "input_channels": INPUT_CHANNELS,
                "epoch": epoch,
                "validation_pr_auc": val_pr_auc,
            }, checkpoint)
        else:
            stale += 1
        print(
            f"TCN epoch={epoch:03d} train_bce={train_bce:.7f} val_bce={val_bce:.7f} "
            f"val_pr_auc={val_pr_auc:.7f} stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    return model, {
        "maximum_epochs": maximum_epochs,
        "patience": patience,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_pr_auc": best_pr_auc,
        "n_nonfog_role6": n_nonfog,
        "n_fog_role7": n_fog,
        "pos_weight": pos_weight,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "history": history,
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def training_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input": "RobustScaler(role4 unique raw samples) -> per-window/per-axis centering -> [B,30,128]",
        "architecture": "RepresentationTCNM 30->32->64->64->128; causal k3; dilations 1/2/4/8; GAP; dropout; 1 logit",
        "train_roles": [6, 7],
        "validation_roles": [2, 3],
        "test_roles": [0, 1],
        "loss": "BCEWithLogitsLoss(pos_weight=N_role6/N_role7)",
        "optimizer": {"name": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4},
        "batch_size": int(args.batch_size),
        "maximum_epochs": int(args.tcn_max_epochs),
        "early_stopping_patience": int(args.tcn_patience),
        "checkpoint_monitor": "roles2/3 PR-AUC maximum",
        "gradient_clip_global_norm": 1.0,
        "threshold": "roles2/3 grid 0.05..0.95 step0.01; max balanced accuracy; ties F1 then higher threshold",
    }


def run_train(args: argparse.Namespace) -> None:
    subject, fold, seed = require_job_args(args)
    destination = run_dir(args.output_root.resolve(), subject, fold, seed)
    done_path = destination / "DONE_TRAIN.json"
    plan = load_plan(args.output_root.resolve())
    validate_plan_args(args, plan)
    if done_path.exists() and not args.overwrite:
        frozen_path = destination / "FROZEN_TRAIN.json"
        checkpoint_path = destination / "checkpoints" / "tcn.pt"
        scaler_path = destination / "scaler_role4.json"
        if not all(path.is_file() for path in (frozen_path, checkpoint_path, scaler_path)):
            raise FileNotFoundError(f"incomplete completed-train marker under {destination}")
        done = json.loads(done_path.read_text(encoding="utf-8"))
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if (
            done.get("frozen_sha256") != sha256_file(frozen_path)
            or done.get("frozen_id") != frozen.get("frozen_id")
            or frozen.get("checkpoint_sha256") != sha256_file(checkpoint_path)
            or frozen.get("scaler_sha256") != sha256_file(scaler_path)
            or frozen.get("data_scientific_sha256") != plan["data_scientific_sha256"]
            or frozen.get("code_sha256") != plan["code_sha256"]
        ):
            raise AssertionError(f"completed train job failed artifact validation: {destination}")
        print(f"SKIP validated completed train job: {destination}", flush=True)
        return
    dataset = DaphnetDataset.load(args.data_dir.resolve())
    if dataset.sampling_rate_hz != SAMPLING_RATE_HZ or dataset.n_channels != INPUT_CHANNELS:
        raise AssertionError(f"expected 64 Hz/30 channels, got {dataset.sampling_rate_hz}/{dataset.n_channels}")
    rows = load_subject_rows(args.data_dir.resolve(), dataset, subject, fold)
    role4 = rows.take_role(4)
    role67 = rows.take_role(6, 7)
    role23 = rows.take_role(2, 3)
    scaler, scaler_points = fit_scaler_unique_role4_points(dataset, role4)
    train_x = raw_features(scaler, raw_windows(dataset, role67))
    validation_x = raw_features(scaler, raw_windows(dataset, role23))
    device = resolve_device(args.device)
    model, training = train_tcn(
        train_x, role67.label, validation_x, role23.label, destination, device, seed,
        args.batch_size, args.num_workers, args.tcn_max_epochs, args.tcn_patience,
    )
    val_true, val_prob = predict(model, validation_x, role23.label, device, args.batch_size)
    threshold, validation_metrics = choose_threshold(val_true, val_prob)
    scaler_path = destination / "scaler_role4.json"
    atomic_json_dump({
        "schema": EXPERIMENT_SCHEMA,
        "subject": subject,
        "fold": fold,
        "seed": seed,
        "fit_role": 4,
        "unique_raw_points": scaler_points,
        "scaler": scaler.as_dict(),
    }, scaler_path)
    history_path = destination / "tcn_history.csv"
    write_csv(history_path, training["history"])
    checkpoint_path = destination / "checkpoints" / "tcn.pt"
    frozen = {
        "schema": EXPERIMENT_SCHEMA,
        "status": "frozen_before_permanent_test",
        "created_utc": utc_now(),
        "subject": subject,
        "fold": fold,
        "seed": seed,
        "data_scientific_sha256": plan["data_scientific_sha256"],
        "code_sha256": plan["code_sha256"],
        "initial_state_sha256": initial_state_hash(seed),
        "scaler_sha256": sha256_file(scaler_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "history_sha256": sha256_file(history_path),
        "threshold": threshold,
        "threshold_source_roles": [2, 3],
        "validation_metrics": validation_metrics,
        "training": {key: value for key, value in training.items() if key != "history"},
        "training_contract": training_contract(args),
        "role_counts": {str(role): int(np.sum(rows.role == role)) for role in ROLES},
        "permanent_test_materialized": False,
    }
    frozen["frozen_id"] = canonical_fingerprint(frozen)
    frozen_path = destination / "FROZEN_TRAIN.json"
    atomic_json_dump(frozen, frozen_path)
    atomic_json_dump({
        "schema": EXPERIMENT_SCHEMA,
        "status": "train_complete",
        "subject": subject,
        "fold": fold,
        "seed": seed,
        "frozen_id": frozen["frozen_id"],
        "frozen_sha256": sha256_file(frozen_path),
    }, done_path)
    print(
        f"TRAIN COMPLETE subject={subject} fold={fold} seed={seed} best_epoch={training['best_epoch']} "
        f"threshold={threshold:.2f} val_pr_auc={validation_metrics['auprc']:.6f}",
        flush=True,
    )


def _boolean_runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(values, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def nonfog_false_alarm_metrics(
    dataset: Any,
    rows: RoleRows,
    y_pred: np.ndarray,
    merge_gap_seconds: float = 1.0,
) -> dict[str, Any]:
    """Count false alarms from positive decisions in true Non-FoG support.

    One detector result is emitted per stride.  Its decision timestamp is the
    window end.  A single positive role-0 decision starts an alarm; subsequent
    positive decisions in the same record are merged when their decision-time
    separation is <= ``merge_gap_seconds``.  Records are never bridged.
    """

    if len(rows) != len(y_pred):
        raise ValueError("rows and y_pred length mismatch")
    maximum_gap = int(round(float(merge_gap_seconds) * dataset.sampling_rate_hz))
    by_record: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for rec_idx, start, end, role, label, prediction in zip(
        rows.record_index, rows.start, rows.end, rows.role, rows.label, y_pred
    ):
        if int(role) not in (0, 1) or int(label) != int(role):
            raise AssertionError("false-alarm scoring requires pure permanent-test roles0/1")
        by_record[int(rec_idx)].append(
            (int(start), int(end), int(role), int(prediction))
        )

    false_positive_decisions = 0
    false_alarm_events = 0
    evaluated_nonfog_samples = 0
    for rec_idx, record_rows in by_record.items():
        record = dataset.records[rec_idx]
        coverage = np.zeros(len(record.y), dtype=bool)
        positive_decision_samples: list[int] = []
        for start, end, role, prediction in record_rows:
            if role != 0:
                continue
            coverage[start:end] = True
            if prediction == 1:
                positive_decision_samples.append(end)
        evaluated_nonfog_samples += int(
            np.sum(coverage & record.valid & (record.y == 0))
        )
        decision_times = sorted(set(positive_decision_samples))
        false_positive_decisions += len(decision_times)
        previous: int | None = None
        for decision_sample in decision_times:
            if previous is None or decision_sample - previous > maximum_gap:
                false_alarm_events += 1
            previous = decision_sample

    evaluated_nonfog_hours = evaluated_nonfog_samples / (
        dataset.sampling_rate_hz * 3600.0
    )
    return {
        "false_alarm_metric_version": "nonfog_positive_decision_FA.v1",
        "false_alarm_merge_gap_seconds": float(merge_gap_seconds),
        "false_alarm_minimum_positive_decisions": 1,
        "false_alarm_decision_timestamp": "window end_index_exclusive",
        "false_alarm_same_record_only": True,
        "false_positive_decisions": int(false_positive_decisions),
        "false_alarm_events": int(false_alarm_events),
        "false_alarm_events_per_hour": (
            false_alarm_events / evaluated_nonfog_hours
            if evaluated_nonfog_hours
            else None
        ),
        "evaluated_nonfog_hours": evaluated_nonfog_hours,
    }


def event_metrics(dataset: Any, rows: RoleRows, y_pred: np.ndarray, minimum_positive_windows: int = 2, merge_gap_seconds: float = 0.5) -> dict[str, Any]:
    by_record: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rec_idx, start, end, pred in zip(rows.record_index, rows.start, rows.end, y_pred):
        by_record[int(rec_idx)].append((int(start), int(end), int(pred)))
    true_total = true_detected = predicted_total = matched_predictions = 0
    nonfog_seconds = 0.0
    delays: list[float] = []
    merge_gap = int(round(merge_gap_seconds * dataset.sampling_rate_hz))
    for rec_idx, record_rows in by_record.items():
        record = dataset.records[rec_idx]
        record_rows.sort(key=lambda item: item[0])
        coverage = np.zeros(len(record.y), dtype=bool)
        for start, end, _ in record_rows:
            coverage[start:end] = True
        nonfog_seconds += float(np.sum(coverage & record.valid & (record.y == 0))) / dataset.sampling_rate_hz
        predicted: list[tuple[int, int, int]] = []
        active: list[tuple[int, int, int]] = []

        def finish_active() -> None:
            if len(active) >= minimum_positive_windows:
                interval = (active[0][0], active[-1][1], active[minimum_positive_windows - 1][1])
                if predicted and interval[0] - predicted[-1][1] <= merge_gap:
                    predicted[-1] = (predicted[-1][0], interval[1], predicted[-1][2])
                else:
                    predicted.append(interval)
            active.clear()

        for start, end, prediction in record_rows:
            if prediction != 1:
                finish_active()
            else:
                if active and start - active[-1][1] > merge_gap:
                    finish_active()
                active.append((start, end, prediction))
        finish_active()
        true_intervals = [
            interval for interval in _boolean_runs(record.y == 1)
            if any(max(interval[0], start) < min(interval[1], end) for start, end, _ in record_rows)
        ]
        true_total += len(true_intervals)
        predicted_total += len(predicted)
        matched: set[int] = set()
        for true_start, true_end in true_intervals:
            matches = [
                index for index, (pred_start, pred_end, _) in enumerate(predicted)
                if max(true_start, pred_start) < min(true_end, pred_end)
            ]
            if not matches:
                continue
            match = min(matches, key=lambda index: predicted[index][0])
            matched.update(matches)
            true_detected += 1
            delays.append(max(0.0, (predicted[match][2] - true_start) / dataset.sampling_rate_hz))
        matched_predictions += len(matched)
    false_alarm = nonfog_false_alarm_metrics(
        dataset, rows, y_pred, merge_gap_seconds=1.0
    )
    return {
        "event_metric_version": EVENT_METRIC_VERSION,
        "minimum_positive_windows": minimum_positive_windows,
        "merge_gap_seconds": merge_gap_seconds,
        "evaluable_true_events": true_total,
        "detected_true_events": true_detected,
        "predicted_events": predicted_total,
        "event_sensitivity": true_detected / true_total if true_total else None,
        "median_detection_delay_sec": float(np.median(delays)) if delays else None,
        "mean_detection_delay_sec": float(np.mean(delays)) if delays else None,
        **false_alarm,
    }


def load_and_validate_barrier(output_root: Path, subject: str, fold: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    barrier_path = output_root / "TRAINING_BARRIER.json"
    if not barrier_path.is_file():
        raise FileNotFoundError("permanent test is locked until TRAINING_BARRIER.json exists")
    barrier = json.loads(barrier_path.read_text(encoding="utf-8"))
    if barrier.get("schema") != BARRIER_SCHEMA or barrier.get("status") != "sealed":
        raise AssertionError("invalid or unsealed training barrier")
    key = f"{subject}/fold_{fold}/seed_{seed}"
    sealed = barrier.get("jobs", {}).get(key)
    if sealed is None:
        raise KeyError(f"job absent from training barrier: {key}")
    destination = run_dir(output_root, subject, fold, seed)
    frozen_path = destination / "FROZEN_TRAIN.json"
    if sha256_file(frozen_path) != sealed["frozen_sha256"]:
        raise AssertionError(f"frozen training artifact changed after seal: {key}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["frozen_id"] != sealed["frozen_id"]:
        raise AssertionError(f"frozen id changed after seal: {key}")
    for name, path in (
        ("checkpoint_sha256", destination / "checkpoints" / "tcn.pt"),
        ("scaler_sha256", destination / "scaler_role4.json"),
    ):
        if sha256_file(path) != sealed[name]:
            raise AssertionError(f"{name} changed after seal: {key}")
    return barrier, frozen


def run_evaluate(args: argparse.Namespace) -> None:
    subject, fold, seed = require_job_args(args)
    output_root = args.output_root.resolve()
    destination = run_dir(output_root, subject, fold, seed)
    barrier, frozen = load_and_validate_barrier(output_root, subject, fold, seed)
    current_scientific = processed_nbm_scientific_manifest(args.data_dir.resolve())["sha256"]
    if current_scientific != barrier["data_scientific_sha256"]:
        raise AssertionError("dataset changed after the training barrier was sealed")
    done_path = destination / "DONE_TEST.json"
    if done_path.exists() and not args.overwrite:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        metrics_path = destination / "metrics.json"
        predictions_path = destination / "test_predictions.csv"
        probabilities_path = destination / "test_probabilities.npz"
        if (
            done.get("barrier_id") == barrier["barrier_id"]
            and all(path.is_file() for path in (metrics_path, predictions_path, probabilities_path))
            and done.get("metrics_sha256") == sha256_file(metrics_path)
            and done.get("predictions_sha256") == sha256_file(predictions_path)
            and done.get("probabilities_sha256") == sha256_file(probabilities_path)
        ):
            print(f"SKIP completed evaluate job: {destination}", flush=True)
            return
        raise AssertionError("existing test result failed barrier/artifact validation")
    dataset = DaphnetDataset.load(args.data_dir.resolve())
    rows = load_subject_rows(args.data_dir.resolve(), dataset, subject, fold)
    test_rows = rows.take_role(0, 1)
    scaler_payload = json.loads((destination / "scaler_role4.json").read_text(encoding="utf-8"))
    scaler = RobustScaler(
        np.asarray(scaler_payload["scaler"]["median"], dtype=np.float32),
        np.asarray(scaler_payload["scaler"]["iqr"], dtype=np.float32),
        float(scaler_payload["scaler"]["epsilon"]),
    )
    test_x = raw_features(scaler, raw_windows(dataset, test_rows))
    device = resolve_device(args.device)
    model = RepresentationTCNM(INPUT_CHANNELS).to(device)
    checkpoint = torch.load(destination / "checkpoints" / "tcn.pt", map_location=device, weights_only=False)
    if checkpoint.get("seed") != seed or checkpoint.get("input_channels") != INPUT_CHANNELS:
        raise AssertionError("checkpoint identity mismatch")
    model.load_state_dict(checkpoint["model_state"])
    test_true, test_prob = predict(model, test_x, test_rows.label, device, args.batch_size)
    threshold = float(frozen["threshold"])
    test_pred = (test_prob >= threshold).astype(np.int8)
    metrics = binary_metrics(test_true, test_prob, threshold)
    events = event_metrics(dataset, test_rows, test_pred)
    metrics.update(events)
    metrics["pr_auc"] = metrics["auprc"]
    metrics["false_alarms_per_hour"] = metrics["false_alarm_events_per_hour"]
    metrics.update({
        "schema": EXPERIMENT_SCHEMA,
        "subject": subject,
        "fold": fold,
        "seed": seed,
        "barrier_id": barrier["barrier_id"],
        "frozen_id": frozen["frozen_id"],
    })
    metrics_path = destination / "metrics.json"
    predictions_path = destination / "test_predictions.csv"
    probabilities_path = destination / "test_probabilities.npz"
    atomic_json_dump(metrics, metrics_path)
    write_csv(predictions_path, ({
        "subject_id": subject,
        "fold": fold,
        "seed": seed,
        "record_id": str(test_rows.record_id[index]),
        "start_index": int(test_rows.start[index]),
        "end_index_exclusive": int(test_rows.end[index]),
        "role_code": int(test_rows.role[index]),
        "window_id": str(test_rows.window_id[index]),
        "y_true": int(test_true[index]),
        "probability": float(test_prob[index]),
        "threshold": threshold,
        "y_pred": int(test_pred[index]),
    } for index in range(len(test_rows))))
    atomic_npz_save(
        probabilities_path,
        y_true=test_true,
        probability=test_prob.astype(np.float32),
        y_pred=test_pred,
        threshold=np.asarray(threshold),
        window_id=test_rows.window_id,
    )
    atomic_json_dump({
        "schema": EXPERIMENT_SCHEMA,
        "status": "test_complete",
        "subject": subject,
        "fold": fold,
        "seed": seed,
        "barrier_id": barrier["barrier_id"],
        "metrics_sha256": sha256_file(metrics_path),
        "predictions_sha256": sha256_file(predictions_path),
        "probabilities_sha256": sha256_file(probabilities_path),
    }, done_path)
    print(
        f"TEST COMPLETE subject={subject} fold={fold} seed={seed} "
        f"sens={metrics['sensitivity']:.6f} precision={metrics['precision']:.6f} "
        f"spec={metrics['specificity']:.6f} pr_auc={metrics['auprc']:.6f} "
        f"event_sens={metrics['event_sensitivity']} fa_per_hour={metrics['false_alarm_events_per_hour']:.6f}",
        flush=True,
    )


def run_seal(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    plan = load_plan(output_root)
    validate_plan_args(args, plan)
    current_scientific = processed_nbm_scientific_manifest(args.data_dir.resolve())["sha256"]
    if current_scientific != plan["data_scientific_sha256"]:
        raise AssertionError("dataset changed after the experiment plan was created")
    seeds = parse_seed_list(args.seeds)
    jobs: dict[str, Any] = {}
    for subject in SUBJECTS:
        for fold in FOLDS:
            for seed in seeds:
                destination = run_dir(output_root, subject, fold, seed)
                frozen_path = destination / "FROZEN_TRAIN.json"
                done_path = destination / "DONE_TRAIN.json"
                if not frozen_path.is_file() or not done_path.is_file():
                    raise FileNotFoundError(f"training job incomplete: {destination}")
                frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
                done = json.loads(done_path.read_text(encoding="utf-8"))
                if (frozen["subject"], frozen["fold"], frozen["seed"]) != (subject, fold, seed):
                    raise AssertionError(f"frozen job identity mismatch: {destination}")
                if done.get("frozen_id") != frozen.get("frozen_id"):
                    raise AssertionError(f"DONE/FROZEN mismatch: {destination}")
                if done.get("frozen_sha256") != sha256_file(frozen_path):
                    raise AssertionError(f"FROZEN changed after DONE: {destination}")
                key = f"{subject}/fold_{fold}/seed_{seed}"
                jobs[key] = {
                    "frozen_id": frozen["frozen_id"],
                    "frozen_sha256": sha256_file(frozen_path),
                    "checkpoint_sha256": frozen["checkpoint_sha256"],
                    "scaler_sha256": frozen["scaler_sha256"],
                    "threshold": frozen["threshold"],
                }
    core = {
        "schema": BARRIER_SCHEMA,
        "status": "sealed",
        "created_utc": utc_now(),
        "plan_id": plan["plan_id"],
        "data_scientific_sha256": plan["data_scientific_sha256"],
        "subjects": list(SUBJECTS),
        "folds": list(FOLDS),
        "seeds": list(seeds),
        "job_count": len(jobs),
        "jobs": jobs,
    }
    core["barrier_id"] = canonical_fingerprint({key: value for key, value in core.items() if key != "created_utc"})
    atomic_json_dump(core, output_root / "TRAINING_BARRIER.json")
    print(f"SEALED {len(jobs)} jobs barrier_id={core['barrier_id']}", flush=True)


def mean_std(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0)), "n": int(len(array))}


def run_aggregate(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    seeds = parse_seed_list(args.seeds)
    barrier = json.loads((output_root / "TRAINING_BARRIER.json").read_text(encoding="utf-8"))
    if barrier.get("schema") != BARRIER_SCHEMA or barrier.get("status") != "sealed":
        raise AssertionError("strict training barrier missing")
    run_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for fold in FOLDS:
            for seed in seeds:
                destination = run_dir(output_root, subject, fold, seed)
                done_path = destination / "DONE_TEST.json"
                metrics_path = destination / "metrics.json"
                predictions_path = destination / "test_predictions.csv"
                probabilities_path = destination / "test_probabilities.npz"
                if not done_path.is_file() or not metrics_path.is_file():
                    raise FileNotFoundError(f"test job incomplete: {destination}")
                done = json.loads(done_path.read_text(encoding="utf-8"))
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if done.get("barrier_id") != barrier["barrier_id"] or metrics.get("barrier_id") != barrier["barrier_id"]:
                    raise AssertionError(f"barrier mismatch: {destination}")
                if done.get("metrics_sha256") != sha256_file(metrics_path):
                    raise AssertionError(f"metrics changed after DONE: {destination}")
                if done.get("predictions_sha256") != sha256_file(predictions_path):
                    raise AssertionError(f"predictions changed after DONE: {destination}")
                if done.get("probabilities_sha256") != sha256_file(probabilities_path):
                    raise AssertionError(f"probabilities changed after DONE: {destination}")
                run_rows.append({
                    "subject": subject,
                    "fold": fold,
                    "seed": seed,
                    "threshold": metrics["threshold"],
                    **{key: metrics[key] for key in METRIC_KEYS},
                    "tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"], "tp": metrics["tp"],
                    "evaluable_true_events": metrics["evaluable_true_events"],
                    "detected_true_events": metrics["detected_true_events"],
                    "false_alarm_events": metrics["false_alarm_events"],
                    "evaluated_nonfog_hours": metrics["evaluated_nonfog_hours"],
                })
    subject_seed_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for seed in seeds:
            selected = [row for row in run_rows if row["subject"] == subject and row["seed"] == seed]
            subject_seed_rows.append({
                "subject": subject,
                "seed": seed,
                **{key: float(np.mean([row[key] for row in selected])) for key in METRIC_KEYS},
            })
    subject_summary_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        selected = [row for row in subject_seed_rows if row["subject"] == subject]
        summary_row: dict[str, Any] = {"subject": subject}
        for key in METRIC_KEYS:
            summary = mean_std(row[key] for row in selected)
            summary_row[f"{key}_mean"] = summary["mean"]
            summary_row[f"{key}_std"] = summary["std"]
        subject_summary_rows.append(summary_row)
    overall_seed_rows = []
    for seed in seeds:
        selected = [row for row in subject_seed_rows if row["seed"] == seed]
        overall_seed_rows.append({"seed": seed, **{key: float(np.mean([row[key] for row in selected])) for key in METRIC_KEYS}})
    overall = {key: mean_std(row[key] for row in overall_seed_rows) for key in METRIC_KEYS}
    write_csv(output_root / "run_metrics.csv", run_rows)
    write_csv(output_root / "subject_seed_metrics.csv", subject_seed_rows)
    write_csv(output_root / "subject_summary.csv", subject_summary_rows)
    write_csv(output_root / "overall_seed_metrics.csv", overall_seed_rows)
    summary = {
        "schema": EXPERIMENT_SCHEMA,
        "aggregation": "per subject/seed: macro mean of 3 folds; subject: mean+population SD over 5 seeds; overall: subject-macro per seed then mean+population SD over seeds",
        "event_metric": {
            "version": EVENT_METRIC_VERSION,
            "minimum_positive_windows": 2,
            "merge_gap_seconds": 0.5,
            "false_alarm_denominator": "union coverage of evaluated valid Non-FoG samples",
        },
        "subjects": subject_summary_rows,
        "overall": overall,
    }
    atomic_json_dump(summary, output_root / "summary.json")
    atomic_json_dump({
        "schema": EXPERIMENT_SCHEMA,
        "status": "complete",
        "completed_utc": utc_now(),
        "run_count": len(run_rows),
        "barrier_id": barrier["barrier_id"],
        "summary_sha256": sha256_file(output_root / "summary.json"),
    }, output_root / "DONE.json")
    print(json.dumps(overall, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_root = args.output_root.resolve()
    if args.stage == "train":
        run_train(args)
    elif args.stage == "evaluate":
        run_evaluate(args)
    elif args.stage == "seal":
        run_seal(args)
    else:
        run_aggregate(args)


if __name__ == "__main__":
    main()
