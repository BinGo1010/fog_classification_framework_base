#!/usr/bin/env python
"""Run a transparent within-S01 GRU-H200 residual TCN-M experiment.

The experiment is deliberately small and exploratory.  It uses three
chronologically isolated partitions of S01, fits every learned quantity on the
training partition only (except early stopping and the decision threshold,
which use validation), and evaluates the test partition once at the end.

Signal and decision timing
--------------------------

* 9 acceleration channels from three tri-axial sensors at 64 Hz;
* 2 s context -> probabilistic GRU forecast of the next 2 s;
* standardized innovation ``clip((target - mean) / sigma, -12, 12)``;
* one 2 s innovation block is classified by TCN-M;
* windows advance by 1 s, so decisions update once per second;
* the endpoint state is labelled from the final 0.5 s of each target block.

All window coordinates, split membership, training histories, predictions,
checkpoints, scaler parameters, architecture metadata, and metrics are saved
under the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Required before importing torch when deterministic CUDA GEMMs are requested.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import sklearn
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cnbr_fog.data import (  # noqa: E402
    DaphnetDataset,
    Record,
    RobustChannelScaler,
    SequenceWindowDataset,
    WindowTable,
)
from cnbr_fog.evaluation import binary_metrics, choose_threshold  # noqa: E402
from cnbr_fog.nbm import GRUNBM, gaussian_nll_sigma, parameter_count  # noqa: E402
from cnbr_fog.resume import (  # noqa: E402
    atomic_json_dump,
    atomic_npz_save,
    atomic_torch_save,
    canonical_fingerprint,
    dataset_fingerprint,
    sha256_file,
)
from cnbr_fog.rf125_classifiers import (  # noqa: E402
    DEFAULT_DILATIONS,
    build_rf125_classifier,
)
from run_cnbr_fog_loso import event_metrics  # noqa: E402


EXPERIMENT_VERSION = "daphnet_s01_gru_h200_tcnm.v1"
SUBJECT_ID = "S01"
SAMPLING_RATE_HZ = 64
CONTEXT_SAMPLES = 128
TARGET_SAMPLES = 128
STRIDE_SAMPLES = 64
LABEL_SAMPLES = 32
NORMAL_GUARD_SAMPLES = 32
TRAIN_VALIDATION_CUT_SAMPLE = 50_944
TRAIN_VALIDATION_CUT_RECORD = "S01_seg001"
TRAIN_RECORD = "S01_seg000"
TEST_RECORD = "S01_seg002"
RESIDUAL_CLIP = 12.0
ROBUST_CLIP = 12.0

EXPECTED_CHANNEL_NAMES = (
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


@dataclass(frozen=True)
class SplitBundle:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Within-S01 GRU-H200 standardized-residual TCN-M experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "dataset"
            / "1.Daphnet Freezing of Gait Dataset"
            / "processed"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "daphnet_s01_gru_h200_tcnm_seed42",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--normal-epochs", type=int, default=8)
    parser.add_argument("--normal-patience", type=int, default=3)
    parser.add_argument("--normal-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-epochs", type=int, default=12)
    parser.add_argument("--classifier-patience", type=int, default=4)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--nbm-hidden", type=int, default=48)
    parser.add_argument("--nbm-dropout", type=float, default=0.1)
    parser.add_argument("--classifier-hidden", type=int, default=48)
    parser.add_argument("--classifier-dropout", type=float, default=0.15)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory if it has no DONE.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the protocol and write split/config artifacts without training.",
    )
    parser.add_argument(
        "--normal-only",
        action="store_true",
        help=(
            "Train only the GRU normal-behaviour model, write its loss plots, "
            "and skip residual extraction and TCN-M training."
        ),
    )
    parser.add_argument(
        "--pretrained-normal-dir",
        type=Path,
        default=None,
        help=(
            "Reuse a completed GRU-only artifact directory and train only the "
            "residual classifier stages; the GRU weights are never updated."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if getattr(args, "normal_only", False) and getattr(
        args, "pretrained_normal_dir", None
    ) is not None:
        raise ValueError("--normal-only and --pretrained-normal-dir are mutually exclusive")
    positive_ints = {
        "batch_size": args.batch_size,
        "normal_epochs": args.normal_epochs,
        "normal_patience": args.normal_patience,
        "classifier_epochs": args.classifier_epochs,
        "classifier_patience": args.classifier_patience,
        "nbm_hidden": args.nbm_hidden,
        "classifier_hidden": args.classifier_hidden,
    }
    for name, value in positive_ints.items():
        if int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    for name in ("normal_lr", "classifier_lr"):
        if not math.isfinite(float(getattr(args, name))) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative")
    for name in ("nbm_dropout", "classifier_dropout"):
        if not 0 <= float(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1)")


def resolve_device(specification: str) -> torch.device:
    if str(specification).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(specification)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable: {specification}")
    return device


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not bool(deterministic)
        torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fieldnames = list(rows[0].keys())
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_s01_dataset(data_dir: Path) -> DaphnetDataset:
    full = DaphnetDataset.load(data_dir)
    if full.sampling_rate_hz != SAMPLING_RATE_HZ:
        raise ValueError(
            f"Expected {SAMPLING_RATE_HZ} Hz, got {full.sampling_rate_hz} Hz"
        )
    if tuple(full.channel_names) != EXPECTED_CHANNEL_NAMES:
        raise ValueError(f"Unexpected channels: {full.channel_names}")
    records = [record for record in full.records if record.subject_id == SUBJECT_ID]
    expected_records = {TRAIN_RECORD, TRAIN_VALIDATION_CUT_RECORD, TEST_RECORD}
    actual_records = {record.record_id for record in records}
    if actual_records != expected_records:
        raise ValueError(
            f"S01 records changed: expected {sorted(expected_records)}, "
            f"got {sorted(actual_records)}"
        )
    return DaphnetDataset(
        root=data_dir,
        records=records,
        sampling_rate_hz=full.sampling_rate_hz,
        channel_names=full.channel_names,
    )


def endpoint_relabel(dataset: DaphnetDataset, windows: WindowTable) -> WindowTable:
    labels = np.empty(len(windows), dtype=np.int8)
    fractions = np.empty(len(windows), dtype=np.float32)
    for record_index, record in enumerate(dataset.records):
        rows = np.flatnonzero(windows.record_index == record_index)
        if not len(rows):
            continue
        ends = windows.target_end[rows].astype(np.int64)
        starts = ends - LABEL_SAMPLES
        prefix = np.r_[0, np.cumsum(record.y == 1, dtype=np.int64)]
        counts = prefix[ends] - prefix[starts]
        fraction = counts.astype(np.float64) / float(LABEL_SAMPLES)
        labels[rows] = (fraction >= 0.5).astype(np.int8)
        fractions[rows] = fraction.astype(np.float32)
    return WindowTable(
        record_index=windows.record_index.copy(),
        start=windows.start.copy(),
        target_start=windows.target_start.copy(),
        target_end=windows.target_end.copy(),
        label=labels,
        fog_fraction=fractions,
        clean_normal=windows.clean_normal.copy(),
    )


def event_scoring_windows(windows: WindowTable) -> WindowTable:
    """Map each 1 Hz decision to one complete second of monitoring exposure.

    Classification labels still describe the final 0.5 seconds before the
    endpoint.  Event false-alarm rates need a different support: consecutive
    1 Hz decisions monitor the timeline continuously, so each decision is
    represented by the preceding one-second interval.  Using the sparse label
    intervals here would halve the non-FOG exposure and double FA/h.
    """

    return WindowTable(
        record_index=windows.record_index.copy(),
        start=windows.start.copy(),
        target_start=(windows.target_end - STRIDE_SAMPLES).astype(np.int32),
        target_end=windows.target_end.copy(),
        label=windows.label.copy(),
        fog_fraction=windows.fog_fraction.copy(),
        clean_normal=windows.clean_normal.copy(),
    )


def record_lookup(dataset: DaphnetDataset) -> dict[str, int]:
    return {record.record_id: index for index, record in enumerate(dataset.records)}


def make_split(dataset: DaphnetDataset, windows: WindowTable) -> SplitBundle:
    lookup = record_lookup(dataset)
    train_record_index = lookup[TRAIN_RECORD]
    cut_record_index = lookup[TRAIN_VALIDATION_CUT_RECORD]
    test_record_index = lookup[TEST_RECORD]

    train = np.flatnonzero(
        (windows.record_index == train_record_index)
        | (
            (windows.record_index == cut_record_index)
            & (windows.target_end <= TRAIN_VALIDATION_CUT_SAMPLE)
        )
    )
    validation = np.flatnonzero(
        (windows.record_index == cut_record_index)
        & (windows.start >= TRAIN_VALIDATION_CUT_SAMPLE)
    )
    test = np.flatnonzero(windows.record_index == test_record_index)
    split = SplitBundle(train=train, validation=validation, test=test)
    validate_split(dataset, windows, split)
    return split


def validate_split(
    dataset: DaphnetDataset,
    windows: WindowTable,
    split: SplitBundle,
) -> None:
    groups = split.as_dict()
    for name, indices in groups.items():
        if not len(indices):
            raise ValueError(f"{name} split is empty")
        counts = np.bincount(windows.label[indices], minlength=2)
        if np.any(counts == 0):
            raise ValueError(f"{name} split lacks a class: {counts.tolist()}")
    sets = {name: set(indices.tolist()) for name, indices in groups.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if sets[left] & sets[right]:
            raise AssertionError(f"{left}/{right} window indices overlap")

    # The same raw point must never contribute to a training and validation
    # context/target.  The test set is a distinct contiguous record.
    lookup = record_lookup(dataset)
    cut_record_index = lookup[TRAIN_VALIDATION_CUT_RECORD]
    cut_train = split.train[windows.record_index[split.train] == cut_record_index]
    cut_validation = split.validation[
        windows.record_index[split.validation] == cut_record_index
    ]
    if int(windows.target_end[cut_train].max()) > int(
        windows.start[cut_validation].min()
    ):
        raise AssertionError("Training and validation raw supports overlap")
    if int(windows.target_end[cut_train].max()) != TRAIN_VALIDATION_CUT_SAMPLE:
        raise AssertionError("Training support does not end at the declared cut")
    if int(windows.start[cut_validation].min()) != TRAIN_VALIDATION_CUT_SAMPLE:
        raise AssertionError("Validation support does not start at the declared cut")


def training_ranges(dataset: DaphnetDataset) -> list[tuple[Record, int, int]]:
    by_name = {record.record_id: record for record in dataset.records}
    return [
        (by_name[TRAIN_RECORD], 0, len(by_name[TRAIN_RECORD].y)),
        (by_name[TRAIN_VALIDATION_CUT_RECORD], 0, TRAIN_VALIDATION_CUT_SAMPLE),
    ]


def fit_training_scaler(dataset: DaphnetDataset) -> tuple[RobustChannelScaler, dict]:
    chunks: list[np.ndarray] = []
    retained_points = 0
    for record, start, end in training_ranges(dataset):
        mask = record.valid[start:end] & (record.y[start:end] == 0)
        retained_points += int(mask.sum())
        chunks.append(record.x[start:end][mask])
    values = np.concatenate(chunks).astype(np.float64, copy=False)
    center = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1e-6, scale, fallback)
    scale = np.where(scale > 1e-6, scale, 1.0)
    scaler = RobustChannelScaler(
        center=center.astype(np.float32),
        scale=scale.astype(np.float32),
        clip=ROBUST_CLIP,
    )
    metadata = {
        **scaler.as_dict(),
        "fit_subject": SUBJECT_ID,
        "fit_split": "train_only",
        "fit_class": "valid_non_fog_samples_only",
        "fit_points": retained_points,
        "scale_definition": "IQR/1.349; per-channel std fallback; 1.0 final fallback",
    }
    return scaler, metadata


def normal_support_indices(
    dataset: DaphnetDataset,
    windows: WindowTable,
    split_name: str,
    indices: np.ndarray,
) -> np.ndarray:
    """Return clean-normal rows whose guard also remains inside its split."""

    mask = windows.clean_normal[indices].copy()
    cut_record_index = record_lookup(dataset)[TRAIN_VALIDATION_CUT_RECORD]
    in_cut_record = windows.record_index[indices] == cut_record_index
    if split_name == "train":
        mask &= (~in_cut_record) | (
            windows.target_end[indices]
            <= TRAIN_VALIDATION_CUT_SAMPLE - NORMAL_GUARD_SAMPLES
        )
    elif split_name == "validation":
        mask &= (~in_cut_record) | (
            windows.start[indices]
            >= TRAIN_VALIDATION_CUT_SAMPLE + NORMAL_GUARD_SAMPLES
        )
    elif split_name != "test":
        raise ValueError(f"Unknown split {split_name!r}")
    return indices[mask]


def boolean_runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(values, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def point_ranges(dataset: DaphnetDataset) -> dict[str, list[tuple[Record, int, int]]]:
    by_name = {record.record_id: record for record in dataset.records}
    cut_record = by_name[TRAIN_VALIDATION_CUT_RECORD]
    return {
        "train": [
            (by_name[TRAIN_RECORD], 0, len(by_name[TRAIN_RECORD].y)),
            (cut_record, 0, TRAIN_VALIDATION_CUT_SAMPLE),
        ],
        "validation": [
            (cut_record, TRAIN_VALIDATION_CUT_SAMPLE, len(cut_record.y))
        ],
        "test": [
            (by_name[TEST_RECORD], 0, len(by_name[TEST_RECORD].y))
        ],
    }


def point_statistics(dataset: DaphnetDataset) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for split, ranges in point_ranges(dataset).items():
        labels = np.concatenate([record.y[start:end] for record, start, end in ranges])
        valid = np.concatenate(
            [record.valid[start:end] for record, start, end in ranges]
        )
        n_fog = int(np.sum((labels == 1) & valid))
        n_normal = int(np.sum((labels == 0) & valid))
        events = sum(
            len(boolean_runs(record.y[start:end] == 1))
            for record, start, end in ranges
        )
        result[split] = {
            "raw_points": int(len(labels)),
            "valid_points": int(valid.sum()),
            "non_fog_points": n_normal,
            "fog_points": n_fog,
            "fog_percent": 100.0 * n_fog / max(n_normal + n_fog, 1),
            "duration_seconds": len(labels) / SAMPLING_RATE_HZ,
            "fog_events": int(events),
            "record_ranges": [
                {
                    "record_id": record.record_id,
                    "start_inclusive": int(start),
                    "end_exclusive": int(end),
                }
                for record, start, end in ranges
            ],
        }
    return result


def window_statistics(
    dataset: DaphnetDataset,
    windows: WindowTable,
    split: SplitBundle,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, indices in split.as_dict().items():
        counts = np.bincount(windows.label[indices], minlength=2).astype(int)
        result[name] = {
            "windows": int(len(indices)),
            "non_fog_windows": int(counts[0]),
            "fog_windows": int(counts[1]),
            "fog_percent": 100.0 * float(counts[1]) / max(int(counts.sum()), 1),
            "clean_normal_windows": int(
                len(normal_support_indices(dataset, windows, name, indices))
            ),
            "first_window_index": int(indices[0]),
            "last_window_index": int(indices[-1]),
        }
    return result


def scaler_clip_statistics(
    dataset: DaphnetDataset,
    scaler: RobustChannelScaler,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, ranges in point_ranges(dataset).items():
        x = np.concatenate([record.x[start:end] for record, start, end in ranges])
        z = (x.astype(np.float64) - scaler.center) / scaler.scale
        clipped = np.abs(z) > scaler.clip
        result[name] = {
            "cells": int(z.size),
            "clipped_cells": int(clipped.sum()),
            "clipped_cell_fraction": float(clipped.mean()),
            "per_channel_clipped_fraction": clipped.mean(axis=0).tolist(),
        }
    return result


def split_manifest_rows(
    dataset: DaphnetDataset,
    windows: WindowTable,
    split: SplitBundle,
) -> list[dict[str, Any]]:
    memberships = {
        int(index): name
        for name, indices in split.as_dict().items()
        for index in indices
    }
    normal_memberships = {
        int(index)
        for name, indices in split.as_dict().items()
        for index in normal_support_indices(dataset, windows, name, indices)
    }
    rows: list[dict[str, Any]] = []
    for window_index in sorted(memberships):
        record = dataset.records[int(windows.record_index[window_index])]
        start = int(windows.start[window_index])
        target_start = int(windows.target_start[window_index])
        target_end = int(windows.target_end[window_index])
        rows.append(
            {
                "split": memberships[window_index],
                "window_index": window_index,
                "subject_id": record.subject_id,
                "record_id": record.record_id,
                "context_start": start,
                "context_end_exclusive": target_start,
                "target_start": target_start,
                "target_end_exclusive": target_end,
                "decision_time_sec": target_end / SAMPLING_RATE_HZ,
                "label_start": target_end - LABEL_SAMPLES,
                "label_end_exclusive": target_end,
                "label_fog_fraction": float(windows.fog_fraction[window_index]),
                "y_true": int(windows.label[window_index]),
                "clean_normal_for_nbm": window_index in normal_memberships,
            }
        )
    return rows


def sequence_loader(
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    scaler: RobustChannelScaler,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        SequenceWindowDataset(dataset.records, windows, indices, scaler),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )


def normal_epoch(
    model: GRUNBM,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_windows = 0
    for sequence, _, _ in loader:
        sequence = sequence.to(device, non_blocking=True)
        context = sequence[:, :, :CONTEXT_SAMPLES]
        target = sequence[:, :, CONTEXT_SAMPLES:]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device.type, enabled=bool(amp and device.type == "cuda")
            ):
                mean, sigma = model(context)
                loss = gaussian_nll_sigma(target, mean, sigma)
            if training:
                assert grad_scaler is not None
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
        batch = int(sequence.shape[0])
        total_loss += float(loss.detach()) * batch
        total_windows += batch
    return total_loss / max(total_windows, 1)


def train_nbm(
    args: argparse.Namespace,
    dataset: DaphnetDataset,
    windows: WindowTable,
    split: SplitBundle,
    scaler: RobustChannelScaler,
    output_dir: Path,
    protocol_fingerprint: str,
    device: torch.device,
) -> tuple[GRUNBM, dict[str, Any]]:
    normal_train = normal_support_indices(
        dataset, windows, "train", split.train
    )
    normal_validation = normal_support_indices(
        dataset, windows, "validation", split.validation
    )
    if not len(normal_train) or not len(normal_validation):
        raise RuntimeError("Clean-normal NBM train/validation support is empty")

    set_seed(args.seed, args.deterministic)
    model = GRUNBM(
        in_channels=dataset.n_channels,
        horizon=TARGET_SAMPLES,
        hidden_channels=args.nbm_hidden,
        num_layers=1,
        dropout=args.nbm_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.normal_lr, weight_decay=args.weight_decay
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(args.amp and device.type == "cuda")
    )
    validation_loader = sequence_loader(
        dataset,
        windows,
        normal_validation,
        scaler,
        args.batch_size,
        False,
        args.num_workers,
        device.type == "cuda",
    )
    checkpoint_path = output_dir / "nbm_best.pt"
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, args.normal_epochs + 1):
        if bad_epochs >= args.normal_patience:
            break
        train_loader = sequence_loader(
            dataset,
            windows,
            normal_train,
            scaler,
            args.batch_size,
            True,
            args.num_workers,
            device.type == "cuda",
            seed=args.seed + epoch,
        )
        train_loss = normal_epoch(
            model,
            train_loader,
            device,
            args.amp,
            optimizer,
            grad_scaler,
        )
        with torch.no_grad():
            validation_loss = normal_epoch(
                model, validation_loader, device, args.amp
            )
        improved = validation_loss < best_loss - 1e-5
        history.append(
            {
                "epoch": epoch,
                "shuffle_seed": args.seed + epoch,
                "train_gaussian_nll": train_loss,
                "validation_gaussian_nll": validation_loss,
                "improved": improved,
            }
        )
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            bad_epochs = 0
            atomic_torch_save(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "protocol_fingerprint": protocol_fingerprint,
                    "seed": args.seed,
                    "epoch": epoch,
                    "validation_gaussian_nll": validation_loss,
                    "model_config": model.model_config(),
                    "model_state": model.state_dict(),
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
        print(
            f"[GRU] epoch={epoch:02d} train_nll={train_loss:.6f} "
            f"val_nll={validation_loss:.6f}{' *' if improved else ''}",
            flush=True,
        )

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    elapsed = time.perf_counter() - started
    training = {
        "seed": args.seed,
        "model_config": model.model_config(),
        "parameter_count": int(parameter_count(model)),
        "optimizer": "AdamW",
        "learning_rate": args.normal_lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "gradient_clip_norm": 5.0,
        "loss": "heteroscedastic Gaussian NLL without constant",
        "train_windows": int(len(normal_train)),
        "validation_windows": int(len(normal_validation)),
        "maximum_epochs": args.normal_epochs,
        "patience": args.normal_patience,
        "best_epoch": best_epoch,
        "best_validation_gaussian_nll": best_loss,
        "epochs_completed": len(history),
        "elapsed_seconds": elapsed,
        "history": history,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    atomic_json_dump(training, output_dir / "nbm_training.json")
    write_csv(output_dir / "nbm_training_history.csv", history)
    return model, training


def plot_nbm_losses(
    output_dir: Path,
    training: Mapping[str, Any],
) -> list[Path]:
    """Write separate train/validation Gaussian-NLL curves for the GRU."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    history = list(training["history"])
    if not history:
        raise ValueError("Cannot plot an empty GRU training history")
    epochs = np.asarray([row["epoch"] for row in history], dtype=np.int64)
    best_epoch = int(training["best_epoch"])
    specifications = (
        (
            "train_gaussian_nll",
            "GRU training Gaussian NLL",
            "Training Gaussian NLL",
            "gru_training_loss.png",
            "#2166ac",
        ),
        (
            "validation_gaussian_nll",
            "GRU validation Gaussian NLL",
            "Validation Gaussian NLL",
            "gru_validation_loss.png",
            "#b2182b",
        ),
    )
    paths: list[Path] = []
    for field, title, ylabel, filename, color in specifications:
        values = np.asarray([row[field] for row in history], dtype=np.float64)
        figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        axis.plot(epochs, values, color=color, linewidth=2.0, marker="o", markersize=3)
        axis.axvline(
            best_epoch,
            color="#444444",
            linestyle="--",
            linewidth=1.2,
            label=f"validation-selected epoch = {best_epoch}",
        )
        if field == "validation_gaussian_nll":
            best_index = int(np.argmin(values))
            axis.scatter(
                [epochs[best_index]],
                [values[best_index]],
                color="#111111",
                s=45,
                zorder=3,
                label=f"minimum = {values[best_index]:.6f}",
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_xlim(float(epochs.min()), float(epochs.max()))
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
        path = output_dir / filename
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)
    return paths


def load_pretrained_nbm(
    args: argparse.Namespace,
    dataset: DaphnetDataset,
    scaler: RobustChannelScaler,
    output_dir: Path,
    protocol_fingerprint: str,
    device: torch.device,
) -> tuple[GRUNBM, dict[str, Any]]:
    """Load and audit a completed GRU-only checkpoint without updating it."""

    source_dir = Path(args.pretrained_normal_dir).resolve()
    required = {
        name: source_dir / name
        for name in (
            "DONE.json",
            "config.json",
            "scaler.json",
            "nbm_best.pt",
            "nbm_training.json",
        )
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Pretrained GRU directory is incomplete ({missing}): {source_dir}"
        )
    source_done = json.loads(required["DONE.json"].read_text(encoding="utf-8"))
    source_config = json.loads(required["config.json"].read_text(encoding="utf-8"))
    source_scaler = json.loads(required["scaler.json"].read_text(encoding="utf-8"))
    source_training = json.loads(
        required["nbm_training.json"].read_text(encoding="utf-8")
    )
    if source_done.get("status") != "complete":
        raise ValueError("Pretrained GRU artifact is not marked complete")
    if source_done.get("execution_scope") != "gru_only":
        raise ValueError("Pretrained artifact must have execution_scope=gru_only")
    if source_config.get("dataset_fingerprint_sha256") != dataset_fingerprint(
        args.data_dir
    ):
        raise ValueError("Pretrained GRU dataset fingerprint does not match")
    if source_config.get("subject") != SUBJECT_ID:
        raise ValueError("Pretrained GRU subject does not match S01")
    if not np.allclose(
        np.asarray(source_scaler["center"], dtype=np.float64),
        np.asarray(scaler.center, dtype=np.float64),
        rtol=0.0,
        atol=1e-8,
    ) or not np.allclose(
        np.asarray(source_scaler["scale"], dtype=np.float64),
        np.asarray(scaler.scale, dtype=np.float64),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError("Pretrained GRU scaler does not match the current train split")

    checkpoint_hash = sha256_file(required["nbm_best.pt"])
    if checkpoint_hash != source_training.get("checkpoint_sha256"):
        raise ValueError("Pretrained GRU checkpoint hash does not match training.json")
    payload = torch.load(
        required["nbm_best.pt"], map_location=device, weights_only=False
    )
    model = GRUNBM(
        in_channels=dataset.n_channels,
        horizon=TARGET_SAMPLES,
        hidden_channels=args.nbm_hidden,
        num_layers=1,
        dropout=args.nbm_dropout,
    ).to(device)
    if payload.get("model_config") != model.model_config():
        raise ValueError("Pretrained GRU architecture does not match current arguments")
    if int(payload.get("epoch", -1)) != int(source_training["best_epoch"]):
        raise ValueError("Pretrained GRU checkpoint epoch is not the selected best epoch")
    model.load_state_dict(payload["model_state"])
    model.eval()

    continued_payload = {
        **payload,
        "protocol_fingerprint": protocol_fingerprint,
        "source_protocol_fingerprint": payload.get("protocol_fingerprint"),
        "pretrained_source_dir": str(source_dir),
        "pretrained_source_sha256": checkpoint_hash,
    }
    checkpoint_path = output_dir / "nbm_best.pt"
    atomic_torch_save(continued_payload, checkpoint_path)
    training = {
        **source_training,
        "training_action": "reused_frozen_checkpoint_no_optimizer_steps",
        "pretrained_source_dir": str(source_dir),
        "pretrained_source_checkpoint_sha256": checkpoint_hash,
        "source_protocol_fingerprint": source_config.get("protocol_fingerprint"),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    atomic_json_dump(training, output_dir / "nbm_training.json")
    write_csv(output_dir / "nbm_training_history.csv", training["history"])
    return model, training


@torch.no_grad()
def extract_residuals(
    args: argparse.Namespace,
    model: GRUNBM,
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    scaler: RobustChannelScaler,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    loader = sequence_loader(
        dataset,
        windows,
        indices,
        scaler,
        args.batch_size,
        False,
        args.num_workers,
        device.type == "cuda",
    )
    model.eval()
    residuals: list[np.ndarray] = []
    unbounded_residuals: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    window_indices: list[np.ndarray] = []
    for sequence, y, index in loader:
        sequence = sequence.to(device, non_blocking=True)
        context = sequence[:, :, :CONTEXT_SAMPLES]
        target = sequence[:, :, CONTEXT_SAMPLES:]
        with torch.amp.autocast(
            device.type, enabled=bool(args.amp and device.type == "cuda")
        ):
            mean, sigma = model(context)
            error = target - mean
            residual = error / sigma
        residuals.append(
            residual.clamp(-RESIDUAL_CLIP, RESIDUAL_CLIP)
            .float()
            .cpu()
            .numpy()
        )
        unbounded_residuals.append(residual.float().cpu().numpy())
        errors.append(error.float().cpu().numpy())
        sigmas.append(sigma.float().cpu().numpy())
        labels.append(y.numpy())
        window_indices.append(index.numpy())

    z = np.concatenate(unbounded_residuals).astype(np.float32, copy=False)
    clipped = np.concatenate(residuals).astype(np.float32, copy=False)
    error = np.concatenate(errors).astype(np.float32, copy=False)
    sigma = np.concatenate(sigmas).astype(np.float32, copy=False)
    y = np.concatenate(labels).astype(np.int8, copy=False)
    index = np.concatenate(window_indices).astype(np.int64, copy=False)
    if not np.array_equal(index, indices):
        raise AssertionError("Residual extraction changed window order")
    diagnostics: dict[str, Any] = {
        "windows": int(len(y)),
        "class_counts": np.bincount(y, minlength=2).astype(int).tolist(),
        "forecast_rmse_scaled": float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
        "forecast_mae_scaled": float(np.mean(np.abs(error.astype(np.float64)))),
        "mean_predicted_sigma_scaled": float(np.mean(sigma.astype(np.float64))),
        "gaussian_nll_scaled": float(
            np.mean(np.log(sigma.astype(np.float64)) + 0.5 * z.astype(np.float64) ** 2)
        ),
        "residual_unclipped_mean": float(np.mean(z.astype(np.float64))),
        "residual_unclipped_std": float(np.std(z.astype(np.float64))),
        "residual_unclipped_rms": float(
            np.sqrt(np.mean(z.astype(np.float64) ** 2))
        ),
        "residual_clipped_abs_mean": float(
            np.mean(np.abs(clipped.astype(np.float64)))
        ),
        "residual_clipped_rms": float(
            np.sqrt(np.mean(clipped.astype(np.float64) ** 2))
        ),
        "clip_threshold": RESIDUAL_CLIP,
        "clipped_cells": int(np.sum(np.abs(z) > RESIDUAL_CLIP)),
        "clipped_cell_fraction": float(np.mean(np.abs(z) > RESIDUAL_CLIP)),
    }
    for label, display in ((0, "non_fog"), (1, "fog")):
        mask = y == label
        diagnostics[display] = {
            "windows": int(mask.sum()),
            "residual_clipped_abs_mean": float(
                np.mean(np.abs(clipped[mask].astype(np.float64)))
            ),
            "residual_clipped_rms": float(
                np.sqrt(np.mean(clipped[mask].astype(np.float64) ** 2))
            ),
        }
    return {
        "residual": np.ascontiguousarray(clipped),
        "y": y,
        "window_index": index,
    }, diagnostics


def array_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )


def classifier_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_windows = 0
    truths: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device.type, enabled=bool(amp and device.type == "cuda")
            ):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                assert grad_scaler is not None
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
        batch = int(y.numel())
        total_loss += float(loss.detach()) * batch
        total_windows += batch
        truths.append(y.detach().cpu().numpy().astype(np.int8))
        probabilities.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
    return (
        total_loss / max(total_windows, 1),
        np.concatenate(truths),
        np.concatenate(probabilities),
    )


def enrich_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    tn, fp, fn, tp = [int(metrics[key]) for key in ("tn", "fp", "fn", "tp")]
    f1_non_fog = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1_fog = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    metrics.update(
        {
            "macro_f1": 0.5 * (f1_non_fog + f1_fog),
            "fog_f1": f1_fog,
            "fog_recall": metrics.get("sensitivity"),
            "roc_auc": metrics.get("auroc"),
            "pr_auc": metrics.get("auprc"),
        }
    )
    return metrics


def prediction_rows(
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window_index, truth, probability, prediction in zip(
        indices, y_true, y_prob, y_pred
    ):
        record = dataset.records[int(windows.record_index[window_index])]
        target_end = int(windows.target_end[window_index])
        rows.append(
            {
                "window_index": int(window_index),
                "subject_id": record.subject_id,
                "record_id": record.record_id,
                "context_start": int(windows.start[window_index]),
                "target_start": int(windows.target_start[window_index]),
                "target_end_exclusive": target_end,
                "decision_time_sec": target_end / SAMPLING_RATE_HZ,
                "label_start": target_end - LABEL_SAMPLES,
                "label_end_exclusive": target_end,
                "y_true": int(truth),
                "fog_probability": float(probability),
                "y_pred": int(prediction),
            }
        )
    return rows


def train_classifier(
    args: argparse.Namespace,
    features: dict[str, dict[str, np.ndarray]],
    dataset: DaphnetDataset,
    windows: WindowTable,
    output_dir: Path,
    protocol_fingerprint: str,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x_train = features["train"]["residual"]
    y_train = features["train"]["y"]
    x_validation = features["validation"]["residual"]
    y_validation = features["validation"]["y"]
    x_test = features["test"]["residual"]
    y_test = features["test"]["y"]
    counts = np.bincount(y_train, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError(f"Classifier training lacks a class: {counts.tolist()}")

    classifier_seed = args.seed + 10_000
    set_seed(classifier_seed, args.deterministic)
    model = build_rf125_classifier(
        "tcn_m",
        in_channels=dataset.n_channels,
        input_samples=TARGET_SAMPLES,
        hidden_channels=args.classifier_hidden,
        dropout=args.classifier_dropout,
        dilations=DEFAULT_DILATIONS,
    ).to(device)
    architecture = model.architecture_config()
    pos_weight_value = min(math.sqrt(counts[0] / counts[1]), 6.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.classifier_lr, weight_decay=args.weight_decay
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(args.amp and device.type == "cuda")
    )
    validation_loader = array_loader(
        x_validation,
        y_validation,
        args.batch_size,
        False,
        args.num_workers,
        device.type == "cuda",
    )
    test_loader = array_loader(
        x_test,
        y_test,
        args.batch_size,
        False,
        args.num_workers,
        device.type == "cuda",
    )
    checkpoint_path = output_dir / "classifier_best.pt"
    history: list[dict[str, Any]] = []
    best_auprc = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, args.classifier_epochs + 1):
        if bad_epochs >= args.classifier_patience:
            break
        train_loader = array_loader(
            x_train,
            y_train,
            args.batch_size,
            True,
            args.num_workers,
            device.type == "cuda",
            seed=classifier_seed + epoch,
        )
        train_loss, train_true, train_probability = classifier_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.amp,
            optimizer,
            grad_scaler,
        )
        with torch.no_grad():
            validation_loss, validation_true, validation_probability = (
                classifier_epoch(
                    model,
                    validation_loader,
                    criterion,
                    device,
                    args.amp,
                )
            )
        validation_auprc = float(
            average_precision_score(validation_true, validation_probability)
        )
        improved = validation_auprc > best_auprc + 1e-5
        history.append(
            {
                "epoch": epoch,
                "shuffle_seed": classifier_seed + epoch,
                "train_bce": train_loss,
                "train_pr_auc": float(
                    average_precision_score(train_true, train_probability)
                ),
                "validation_bce": validation_loss,
                "validation_pr_auc": validation_auprc,
                "improved": improved,
            }
        )
        if improved:
            best_auprc = validation_auprc
            best_epoch = epoch
            bad_epochs = 0
            atomic_torch_save(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "protocol_fingerprint": protocol_fingerprint,
                    "seed": classifier_seed,
                    "epoch": epoch,
                    "validation_pr_auc": validation_auprc,
                    "architecture": architecture,
                    "model_state": model.state_dict(),
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
        print(
            f"[TCN-M] epoch={epoch:02d} train_bce={train_loss:.6f} "
            f"train_pr={history[-1]['train_pr_auc']:.6f} "
            f"val_pr={validation_auprc:.6f}{' *' if improved else ''}",
            flush=True,
        )

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    with torch.no_grad():
        _, validation_true, validation_probability = classifier_epoch(
            model, validation_loader, criterion, device, args.amp
        )
        _, test_true, test_probability = classifier_epoch(
            model, test_loader, criterion, device, args.amp
        )
    threshold, validation_metrics = choose_threshold(
        validation_true, validation_probability
    )
    validation_metrics = enrich_metrics(validation_metrics)
    test_metrics = enrich_metrics(binary_metrics(test_true, test_probability, threshold))
    validation_prediction = (
        np.asarray(validation_probability) >= threshold
    ).astype(np.int8)
    test_prediction = (np.asarray(test_probability) >= threshold).astype(np.int8)
    event_windows = event_scoring_windows(windows)
    test_metrics.update(
        event_metrics(
            dataset,
            event_windows,
            features["test"]["window_index"],
            test_prediction,
            minimum_positive_windows=1,
            merge_gap_seconds=0.5,
        )
    )
    elapsed = time.perf_counter() - started
    training = {
        "seed": classifier_seed,
        "architecture": architecture,
        "optimizer": "AdamW",
        "learning_rate": args.classifier_lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "gradient_clip_norm": 5.0,
        "loss": "BCEWithLogitsLoss",
        "train_counts_non_fog_fog": counts.astype(int).tolist(),
        "positive_class_weight": pos_weight_value,
        "maximum_epochs": args.classifier_epochs,
        "patience": args.classifier_patience,
        "early_stop_metric": "validation PR-AUC",
        "best_epoch": best_epoch,
        "best_validation_pr_auc": best_auprc,
        "epochs_completed": len(history),
        "threshold_selection": (
            "validation-only grid 0.01..0.99; maximize balanced accuracy, "
            "then F1, then higher threshold"
        ),
        "selected_threshold": threshold,
        "elapsed_seconds": elapsed,
        "history": history,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    metrics = {
        "validation": validation_metrics,
        "test": test_metrics,
        "test_positive_prevalence_baseline_pr_auc": float(np.mean(test_true)),
    }
    atomic_json_dump(training, output_dir / "classifier_training.json")
    atomic_json_dump(metrics, output_dir / "metrics.json")
    write_csv(output_dir / "classifier_training_history.csv", history)
    write_csv(
        output_dir / "validation_predictions.csv",
        prediction_rows(
            dataset,
            windows,
            features["validation"]["window_index"],
            validation_true,
            validation_probability,
            validation_prediction,
        ),
    )
    write_csv(
        output_dir / "test_predictions.csv",
        prediction_rows(
            dataset,
            windows,
            features["test"]["window_index"],
            test_true,
            test_probability,
            test_prediction,
        ),
    )
    atomic_npz_save(
        output_dir / "predictions.npz",
        validation_window_index=features["validation"]["window_index"],
        validation_y_true=validation_true,
        validation_y_probability=np.asarray(validation_probability, dtype=np.float32),
        validation_y_pred=validation_prediction,
        test_window_index=features["test"]["window_index"],
        test_y_true=test_true,
        test_y_probability=np.asarray(test_probability, dtype=np.float32),
        test_y_pred=test_prediction,
    )
    return training, metrics


def plot_classifier_losses(
    output_dir: Path,
    training: Mapping[str, Any],
) -> list[Path]:
    """Write separate TCN-M train and validation BCE curves."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    history = list(training["history"])
    if not history:
        raise ValueError("Cannot plot an empty TCN-M training history")
    epochs = np.asarray([row["epoch"] for row in history], dtype=np.int64)
    selected_epoch = int(training["best_epoch"])
    specifications = (
        (
            "train_bce",
            "TCN-M training BCE loss",
            "Training BCE loss",
            "tcnm_training_loss.png",
            "#2166ac",
        ),
        (
            "validation_bce",
            "TCN-M validation BCE loss",
            "Validation BCE loss",
            "tcnm_validation_loss.png",
            "#b2182b",
        ),
    )
    paths: list[Path] = []
    for field, title, ylabel, filename, color in specifications:
        values = np.asarray([row[field] for row in history], dtype=np.float64)
        figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        axis.plot(epochs, values, color=color, linewidth=2.0, marker="o", markersize=4)
        axis.axvline(
            selected_epoch,
            color="#444444",
            linestyle="--",
            linewidth=1.2,
            label=f"validation-PR-AUC-selected epoch = {selected_epoch}",
        )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_xlim(float(epochs.min()), float(epochs.max()))
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
        path = output_dir / filename
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)
    return paths


def plot_test_confusion_matrix(
    output_dir: Path,
    test_metrics: Mapping[str, Any],
    subject_label: str = "S01",
) -> Path:
    """Write the thresholded test confusion matrix using absolute counts."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    matrix = np.asarray(test_metrics["confusion_matrix"], dtype=np.int64)
    if matrix.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 confusion matrix, got {matrix.shape}")
    figure, axis = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Windows")
    axis.set_xticks([0, 1], labels=["Non-FoG", "FoG"])
    axis.set_yticks([0, 1], labels=["Non-FoG", "FoG"])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(
        f"{subject_label} test confusion matrix\n"
        f"validation-selected threshold = {float(test_metrics['threshold']):.2f}"
    )
    cutoff = float(matrix.max()) / 2.0
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                fontsize=14,
                color="white" if matrix[row, column] > cutoff else "black",
            )
    path = output_dir / "test_confusion_matrix.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def build_protocol(
    args: argparse.Namespace,
    dataset: DaphnetDataset,
    point_stats: dict[str, dict[str, Any]],
    window_stats: dict[str, dict[str, Any]],
    scaler_metadata: dict[str, Any],
    scaler_clip_stats: dict[str, dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    with torch.random.fork_rng(devices=[]):
        nbm = GRUNBM(
            in_channels=dataset.n_channels,
            horizon=TARGET_SAMPLES,
            hidden_channels=args.nbm_hidden,
            num_layers=1,
            dropout=args.nbm_dropout,
        )
        classifier = build_rf125_classifier(
            "tcn_m",
            in_channels=dataset.n_channels,
            input_samples=TARGET_SAMPLES,
            hidden_channels=args.classifier_hidden,
            dropout=args.classifier_dropout,
            dilations=DEFAULT_DILATIONS,
        )
    payload: dict[str, Any] = {
        "experiment_version": EXPERIMENT_VERSION,
        "created_utc": utc_now(),
        "data_dir": str(args.data_dir.resolve()),
        "dataset_fingerprint_sha256": dataset_fingerprint(args.data_dir),
        "subject": SUBJECT_ID,
        "records": [record.record_id for record in dataset.records],
        "sampling_rate_hz": dataset.sampling_rate_hz,
        "channels": list(dataset.channel_names),
        "point_statistics": point_stats,
        "window_statistics": window_stats,
        "split": {
            "strategy": "chronological record/block split with disjoint raw support",
            "train": (
                "all S01_seg000 plus S01_seg001 samples [0,50944); "
                "window support must end no later than sample 50944"
            ),
            "validation": (
                "S01_seg001 samples [50944,end); window support must begin "
                "at or after sample 50944"
            ),
            "test": "all S01_seg002; untouched until final evaluation",
            "cut_record": TRAIN_VALIDATION_CUT_RECORD,
            "cut_sample": TRAIN_VALIDATION_CUT_SAMPLE,
            "cut_time_seconds": TRAIN_VALIDATION_CUT_SAMPLE / SAMPLING_RATE_HZ,
            "cut_selection_disclosure": (
                "Nominal approximately 70 percent chronological cut was moved to "
                "an event-free boundary between FOG events so one event is not "
                "split and train/validation both contain both classes. This is "
                "label-aware exploratory split design; test record was not used."
            ),
        },
        "scaler": {
            **scaler_metadata,
            "input_clip_statistics": scaler_clip_stats,
        },
        "windowing": {
            "context_seconds": CONTEXT_SAMPLES / SAMPLING_RATE_HZ,
            "context_samples": CONTEXT_SAMPLES,
            "target_seconds": TARGET_SAMPLES / SAMPLING_RATE_HZ,
            "target_samples": TARGET_SAMPLES,
            "stride_seconds": STRIDE_SAMPLES / SAMPLING_RATE_HZ,
            "stride_samples": STRIDE_SAMPLES,
            "complete_context_target_support_seconds": (
                CONTEXT_SAMPLES + TARGET_SAMPLES
            )
            / SAMPLING_RATE_HZ,
            "window_crosses_record": False,
            "train_validation_raw_support_overlap": False,
        },
        "label": {
            "definition": (
                "FOG if at least 50 percent of the final 0.5 seconds (32 samples) "
                "before the decision endpoint are sample-level FOG"
            ),
            "label_samples": LABEL_SAMPLES,
            "fog_fraction_threshold": 0.5,
            "reason": (
                "Endpoint-state detection; a FOG episode that ended earlier in the "
                "2-second residual block should not force the current endpoint label."
            ),
        },
        "normal_behaviour_model": {
            "architecture": nbm.model_config(),
            "parameter_count": int(parameter_count(nbm)),
            "training_input": "clean normal train windows only",
            "clean_normal_rule": (
                "context and target plus a 0.5-second pre/post guard contain no FOG"
            ),
            "output": "mean and positive sigma, each [batch,9,128]",
            "loss": "heteroscedastic Gaussian NLL without constant",
            "early_stopping": "validation clean-normal Gaussian NLL",
        },
        "residual": {
            "formula": "clip((target - mean) / sigma, -12, 12)",
            "clip": RESIDUAL_CLIP,
            "classifier_input_shape": ["batch", dataset.n_channels, TARGET_SAMPLES],
            "history_seconds": TARGET_SAMPLES / SAMPLING_RATE_HZ,
        },
        "classifier": {
            "architecture": classifier.architecture_config(),
            "loss": "BCEWithLogitsLoss",
            "positive_weight": "min(sqrt(N_nonFOG/N_FOG), 6)",
            "early_stopping": "validation PR-AUC",
            "threshold": "validation-only maximum balanced accuracy",
            "event_metric_postprocessing": (
                "none beyond threshold; one positive window can form an event. "
                "For event exposure and FA/h, each 1 Hz decision represents the "
                "preceding 1-second interval; endpoint labels remain final-0.5-second."
            ),
        },
        "training": {
            "seed": args.seed,
            "classifier_seed": args.seed + 10_000,
            "execution_scope": (
                "gru_only"
                if getattr(args, "normal_only", False)
                else (
                    "tcn_continuation"
                    if getattr(args, "pretrained_normal_dir", None) is not None
                    else "full_pipeline"
                )
            ),
            "pretrained_normal_dir": (
                str(Path(args.pretrained_normal_dir).resolve())
                if getattr(args, "pretrained_normal_dir", None) is not None
                else None
            ),
            "batch_size": args.batch_size,
            "normal_epochs_max": args.normal_epochs,
            "normal_patience": args.normal_patience,
            "normal_learning_rate": args.normal_lr,
            "classifier_epochs_max": args.classifier_epochs,
            "classifier_patience": args.classifier_patience,
            "classifier_learning_rate": args.classifier_lr,
            "weight_decay": args.weight_decay,
            "deterministic": args.deterministic,
            "amp_requested": args.amp,
        },
        "leakage_controls": [
            "Robust scaler uses valid non-FOG samples from train ranges only.",
            "GRU weights use clean-normal train windows only.",
            "Validation selects GRU epoch, TCN-M epoch, and decision threshold.",
            "No test point/window is used for fitting, early stopping, weighting, or thresholding.",
            "Train/validation context-target supports are raw-sample disjoint.",
        ],
        "known_limitation": (
            "Classifier-training residuals are generated by the GRU fitted on the "
            "same training block. This is not test leakage, but can make train "
            "non-FOG residuals more optimistic than validation/test residuals. A "
            "blocked out-of-fold residual experiment is the appropriate follow-up."
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
    }
    payload["protocol_fingerprint"] = canonical_fingerprint(
        {
            key: value
            for key, value in payload.items()
            if key not in {"created_utc", "environment"}
        }
    )
    return payload


def write_summary(
    output_dir: Path,
    protocol: dict[str, Any],
    nbm_training: dict[str, Any],
    residual_diagnostics: dict[str, Any],
    classifier_training: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    point_stats = protocol["point_statistics"]
    window_stats = protocol["window_statistics"]
    validation = metrics["validation"]
    test = metrics["test"]
    text = f"""# S01 GRU-H200 standardized-residual TCN-M result

## Protocol

- Subject: S01 only; 9 accelerometer channels at 64 Hz.
- Split: train = S01_seg000 + S01_seg001[0:50944], validation =
  S01_seg001[50944:end], test = S01_seg002.
- Context/target/stride: 128/128/64 samples = 2/2/1 seconds.
- Label: at least 50% FOG in the final 32 samples before each endpoint.
- GRU: 1 layer, hidden {protocol['normal_behaviour_model']['architecture']['hidden_channels']},
  direct Gaussian 128-step decoder, {protocol['normal_behaviour_model']['parameter_count']:,} parameters.
- Residual: clip((target - mean) / sigma, -12, 12), shape [B,9,128].
- TCN-M: dilations {protocol['classifier']['architecture']['dilations']}, local RF
  {protocol['classifier']['architecture']['local_receptive_field_samples']} samples,
  {protocol['classifier']['architecture']['parameter_count']:,} parameters.

## Data split

| Split | Raw points | FOG points | FOG events | Windows | FOG windows | Clean-normal NBM windows |
|---|---:|---:|---:|---:|---:|---:|
| Train | {point_stats['train']['raw_points']:,} | {point_stats['train']['fog_points']:,} | {point_stats['train']['fog_events']} | {window_stats['train']['windows']} | {window_stats['train']['fog_windows']} | {window_stats['train']['clean_normal_windows']} |
| Validation | {point_stats['validation']['raw_points']:,} | {point_stats['validation']['fog_points']:,} | {point_stats['validation']['fog_events']} | {window_stats['validation']['windows']} | {window_stats['validation']['fog_windows']} | {window_stats['validation']['clean_normal_windows']} |
| Test | {point_stats['test']['raw_points']:,} | {point_stats['test']['fog_points']:,} | {point_stats['test']['fog_events']} | {window_stats['test']['windows']} | {window_stats['test']['fog_windows']} | {window_stats['test']['clean_normal_windows']} |

## Training

- GRU best epoch: {nbm_training['best_epoch']}; validation Gaussian NLL:
  {nbm_training['best_validation_gaussian_nll']:.6f}.
- TCN-M best epoch: {classifier_training['best_epoch']}; validation PR-AUC:
  {classifier_training['best_validation_pr_auc']:.6f}.
- Validation-selected threshold: {classifier_training['selected_threshold']:.4f}.

## Residual diagnostics

| Split | Forecast RMSE (scaled) | Gaussian NLL | Residual clip fraction |
|---|---:|---:|---:|
| Train | {residual_diagnostics['train']['forecast_rmse_scaled']:.6f} | {residual_diagnostics['train']['gaussian_nll_scaled']:.6f} | {100*residual_diagnostics['train']['clipped_cell_fraction']:.4f}% |
| Validation | {residual_diagnostics['validation']['forecast_rmse_scaled']:.6f} | {residual_diagnostics['validation']['gaussian_nll_scaled']:.6f} | {100*residual_diagnostics['validation']['clipped_cell_fraction']:.4f}% |
| Test | {residual_diagnostics['test']['forecast_rmse_scaled']:.6f} | {residual_diagnostics['test']['gaussian_nll_scaled']:.6f} | {100*residual_diagnostics['test']['clipped_cell_fraction']:.4f}% |

## Classification metrics

| Split | PR-AUC | AUROC | Balanced accuracy | FOG recall | Specificity | Precision | FOG F1 | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | {validation['pr_auc']:.6f} | {validation['roc_auc']:.6f} | {validation['balanced_accuracy']:.6f} | {validation['fog_recall']:.6f} | {validation['specificity']:.6f} | {validation['precision']:.6f} | {validation['fog_f1']:.6f} | {validation['macro_f1']:.6f} |
| Test | {test['pr_auc']:.6f} | {test['roc_auc']:.6f} | {test['balanced_accuracy']:.6f} | {test['fog_recall']:.6f} | {test['specificity']:.6f} | {test['precision']:.6f} | {test['fog_f1']:.6f} | {test['macro_f1']:.6f} |

Test confusion matrix: TN={test['tn']}, FP={test['fp']}, FN={test['fn']}, TP={test['tp']}.

Event sensitivity: {test['event_sensitivity']}; false alarms/hour:
{test['false_alarm_events_per_hour']}; median detection delay:
{test['median_detection_delay_sec']} seconds.

## Interpretation boundary

This is one label-aware exploratory split and one seed from one subject.  It is
not a subject-generalization result.  Training residuals are in-sample with
respect to the GRU; blocked out-of-fold residual generation and repeated seeds
should be added before treating the score as stable evidence.
"""
    path = output_dir / "summary.md"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    done_path = output_dir / "DONE.json"
    if done_path.exists():
        raise FileExistsError(f"Completed output already exists: {done_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is non-empty; pass --overwrite to reuse: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_s01_dataset(args.data_dir)
    base_windows = dataset.make_windows(
        warmup_samples=CONTEXT_SAMPLES,
        target_samples=TARGET_SAMPLES,
        stride_samples=STRIDE_SAMPLES,
        fog_fraction_threshold=0.5,
        normal_guard_samples=NORMAL_GUARD_SAMPLES,
    )
    windows = endpoint_relabel(dataset, base_windows)
    split = make_split(dataset, windows)
    scaler, scaler_metadata = fit_training_scaler(dataset)
    point_stats = point_statistics(dataset)
    window_stats = window_statistics(dataset, windows, split)
    scaler_clip_stats = scaler_clip_statistics(dataset, scaler)
    protocol = build_protocol(
        args,
        dataset,
        point_stats,
        window_stats,
        scaler_metadata,
        scaler_clip_stats,
        device,
    )
    atomic_json_dump(protocol, output_dir / "config.json")
    atomic_json_dump(scaler_metadata, output_dir / "scaler.json")
    write_csv(
        output_dir / "split_manifest.csv",
        split_manifest_rows(dataset, windows, split),
    )
    atomic_npz_save(
        output_dir / "split_indices.npz",
        train_window_index=split.train,
        validation_window_index=split.validation,
        test_window_index=split.test,
    )
    print(
        f"Protocol {protocol['protocol_fingerprint']}\n"
        f"device={device} window_counts="
        f"{ {name: len(index) for name, index in split.as_dict().items()} }",
        flush=True,
    )
    if args.dry_run:
        atomic_json_dump(
            {
                "status": "dry_run_complete",
                "experiment_version": EXPERIMENT_VERSION,
                "protocol_fingerprint": protocol["protocol_fingerprint"],
            },
            output_dir / "DRY_RUN.json",
        )
        return

    if args.pretrained_normal_dir is not None:
        nbm, nbm_training = load_pretrained_nbm(
            args,
            dataset,
            scaler,
            output_dir,
            protocol["protocol_fingerprint"],
            device,
        )
    else:
        nbm, nbm_training = train_nbm(
            args,
            dataset,
            windows,
            split,
            scaler,
            output_dir,
            protocol["protocol_fingerprint"],
            device,
        )
    loss_plot_paths = plot_nbm_losses(output_dir, nbm_training)
    if args.normal_only:
        atomic_json_dump(
            {
                "status": "complete",
                "completed_utc": utc_now(),
                "experiment_version": EXPERIMENT_VERSION,
                "protocol_fingerprint": protocol["protocol_fingerprint"],
                "execution_scope": "gru_only",
                "gru": {
                    "maximum_epochs": nbm_training["maximum_epochs"],
                    "epochs_completed": nbm_training["epochs_completed"],
                    "best_epoch": nbm_training["best_epoch"],
                    "best_validation_gaussian_nll": nbm_training[
                        "best_validation_gaussian_nll"
                    ],
                    "stop_reason": (
                        "early_stopping_patience"
                        if nbm_training["epochs_completed"]
                        < nbm_training["maximum_epochs"]
                        else "maximum_epochs"
                    ),
                },
                "loss_plots": [path.name for path in loss_plot_paths],
                "artifacts": {
                    path.name: sha256_file(path)
                    for path in sorted(output_dir.iterdir())
                    if path.is_file()
                },
            },
            done_path,
        )
        print(
            "COMPLETE GRU_ONLY "
            f"epochs={nbm_training['epochs_completed']} "
            f"best_epoch={nbm_training['best_epoch']} "
            "best_val_nll="
            f"{nbm_training['best_validation_gaussian_nll']:.6f}",
            flush=True,
        )
        return
    features: dict[str, dict[str, np.ndarray]] = {}
    residual_diagnostics: dict[str, Any] = {}
    for name, indices in split.as_dict().items():
        features[name], residual_diagnostics[name] = extract_residuals(
            args, nbm, dataset, windows, indices, scaler, device
        )
    atomic_json_dump(
        residual_diagnostics, output_dir / "residual_diagnostics.json"
    )
    atomic_npz_save(
        output_dir / "residual_cache.npz",
        **{
            f"{split_name}_{field}": values[field]
            for split_name, values in features.items()
            for field in ("residual", "y", "window_index")
        },
    )
    classifier_training, metrics = train_classifier(
        args,
        features,
        dataset,
        windows,
        output_dir,
        protocol["protocol_fingerprint"],
        device,
    )
    plot_classifier_losses(output_dir, classifier_training)
    plot_test_confusion_matrix(output_dir, metrics["test"])
    write_summary(
        output_dir,
        protocol,
        nbm_training,
        residual_diagnostics,
        classifier_training,
        metrics,
    )
    atomic_json_dump(
        {
            "status": "complete",
            "completed_utc": utc_now(),
            "experiment_version": EXPERIMENT_VERSION,
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "artifacts": {
                path.name: sha256_file(path)
                for path in sorted(output_dir.iterdir())
                if path.is_file()
            },
        },
        done_path,
    )
    test = metrics["test"]
    print(
        "COMPLETE "
        f"test_pr_auc={test['pr_auc']:.6f} "
        f"test_auroc={test['roc_auc']:.6f} "
        f"test_balanced_accuracy={test['balanced_accuracy']:.6f} "
        f"test_fog_recall={test['fog_recall']:.6f} "
        f"test_specificity={test['specificity']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
