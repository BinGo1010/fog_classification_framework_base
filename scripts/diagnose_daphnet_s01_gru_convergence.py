#!/usr/bin/env python
"""Diagnose S01 GRU-NBM convergence without evaluating the held-out test run.

This entry point intentionally contains no classifier stage and never creates a
loader for ``S01_seg002`` (R02).  It uses only the frozen train/validation
protocol from ``run_daphnet_s01_gru_h200_tcnm.py`` and records enough evidence
to distinguish four explanations for the earlier eight-epoch curve:

* the epoch budget was too short;
* the amount of clean-normal training support is limiting validation quality;
* Gaussian NLL falls mainly by inflating sigma rather than improving the mean;
* optimization is numerically unstable.

The default grid is 4 clean-normal duration fractions x 5 seeds, with a
maximum of 40 epochs and validation clean-normal NLL early stopping with
patience 6.  Fractions are nested temporal-block subsets.  Smaller subsets are
sampled with replacement to match the full-data number of window exposures and
optimizer updates per epoch, so duration is not confounded with training steps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_daphnet_s01_gru_h200_tcnm as base  # noqa: E402
from cnbr_fog.data import (  # noqa: E402
    DaphnetDataset,
    Record,
    RobustChannelScaler,
    WindowTable,
    valid_signal_mask,
)
from cnbr_fog.nbm import GRUNBM, gaussian_nll_sigma, parameter_count  # noqa: E402
from cnbr_fog.resume import (  # noqa: E402
    atomic_json_dump,
    atomic_npz_save,
    atomic_torch_save,
    canonical_fingerprint,
    sha256_file,
)


EXPERIMENT_VERSION = "daphnet_s01_gru_convergence_diagnostic.v1"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
MIN_DELTA = 1e-4
MIN_EPOCHS = 8
TEMPORAL_BLOCK_SECONDS = 30
SIGMA_MIN = math.exp(-3.0)
SIGMA_MAX = math.exp(1.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/validation-only S01 GRU-NBM convergence diagnostic",
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
        default=(
            REPO_ROOT
            / "outputs"
            / "daphnet_s01_gru_convergence_40ep_pat6"
        ),
    )
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument(
        "--fractions", default=",".join(map(str, DEFAULT_FRACTIONS))
    )
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def parse_int_list(specification: str) -> tuple[int, ...]:
    values: list[int] = []
    for raw in str(specification).split(","):
        if not raw.strip():
            continue
        value = int(raw)
        if value < 0:
            raise ValueError("Seeds must be non-negative")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("At least one seed is required")
    return tuple(values)


def parse_fraction_list(specification: str) -> tuple[float, ...]:
    values: list[float] = []
    for raw in str(specification).split(","):
        if not raw.strip():
            continue
        value = float(raw)
        if not 0.0 < value <= 1.0 or not math.isfinite(value):
            raise ValueError("Training fractions must be finite and in (0,1]")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("At least one training fraction is required")
    if 1.0 not in values:
        raise ValueError("The convergence diagnostic must include fraction 1.0")
    return tuple(sorted(values))


def validate_args(args: argparse.Namespace) -> None:
    if args.max_epochs not in {30, 40}:
        raise ValueError("--max-epochs must be 30 or 40 for this protocol")
    if args.patience not in {5, 6}:
        raise ValueError("--patience must be 5 or 6 for this protocol")
    for name in ("batch_size", "hidden_channels"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer hyperparameters")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0,1)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
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
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def fraction_name(value: float) -> str:
    return f"fraction_{value:g}".replace(".", "p")


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def load_train_validation_dataset(data_dir: Path) -> DaphnetDataset:
    """Load only the two permitted records; never open the R02 array file."""

    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.csv"
    schema_path = data_dir / "schema.json"
    if not manifest_path.exists() or not schema_path.exists():
        raise FileNotFoundError("Processed Daphnet manifest/schema is missing")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    channel_names = tuple(str(item["name"]) for item in schema["channels"])
    if channel_names != base.EXPECTED_CHANNEL_NAMES:
        raise ValueError(f"Unexpected channels: {channel_names}")

    permitted = {base.TRAIN_RECORD, base.TRAIN_VALIDATION_CUT_RECORD}
    records: list[Record] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("record_id") not in permitted:
                continue
            if str(row.get("usable", "true")).strip().lower() not in {
                "1",
                "true",
                "yes",
            }:
                raise ValueError(f"Required record is unusable: {row['record_id']}")
            sampling_rate = int(row["sampling_rate_hz"])
            if sampling_rate != base.SAMPLING_RATE_HZ:
                raise ValueError(f"Unexpected sampling rate: {sampling_rate}")
            record_path = data_dir / row["record_path"]
            with np.load(record_path, allow_pickle=False) as payload:
                if set(payload.files) != {"x", "y_binary"}:
                    raise ValueError(f"Unexpected arrays in {record_path}")
                x = np.asarray(payload["x"], dtype=np.float32)
                y = np.asarray(payload["y_binary"], dtype=np.int8)
            if x.shape != (int(row["n_samples"]), len(channel_names)):
                raise ValueError(f"Manifest shape mismatch for {record_path}")
            if y.shape != (len(x),) or not set(np.unique(y)).issubset({0, 1}):
                raise ValueError(f"Invalid labels in {record_path}")
            records.append(
                Record(
                    record_id=row["record_id"],
                    subject_id=row["subject_id"],
                    run_id=row["run_id"],
                    x=x,
                    y=y,
                    valid=valid_signal_mask(x, sampling_rate),
                )
            )
    if {record.record_id for record in records} != permitted:
        raise ValueError("Unexpected S01 train/validation records")
    records.sort(
        key=lambda record: (
            0 if record.record_id == base.TRAIN_RECORD else 1,
            record.record_id,
        )
    )
    dataset = DaphnetDataset(
        root=data_dir,
        records=records,
        sampling_rate_hz=base.SAMPLING_RATE_HZ,
        channel_names=channel_names,
    )
    if any(record.record_id == base.TEST_RECORD for record in dataset.records):
        raise AssertionError("Held-out test record entered diagnostic dataset")
    return dataset


def prepare_support(
    data_dir: Path,
) -> tuple[
    DaphnetDataset,
    WindowTable,
    np.ndarray,
    np.ndarray,
    RobustChannelScaler,
    dict[str, Any],
]:
    dataset = load_train_validation_dataset(data_dir)
    raw_windows = dataset.make_windows(
        warmup_samples=base.CONTEXT_SAMPLES,
        target_samples=base.TARGET_SAMPLES,
        stride_samples=base.STRIDE_SAMPLES,
        fog_fraction_threshold=0.5,
        normal_guard_samples=base.NORMAL_GUARD_SAMPLES,
    )
    windows = base.endpoint_relabel(dataset, raw_windows)
    lookup = base.record_lookup(dataset)
    train = np.flatnonzero(
        (windows.record_index == lookup[base.TRAIN_RECORD])
        | (
            (windows.record_index == lookup[base.TRAIN_VALIDATION_CUT_RECORD])
            & (windows.target_end <= base.TRAIN_VALIDATION_CUT_SAMPLE)
        )
    )
    validation = np.flatnonzero(
        (windows.record_index == lookup[base.TRAIN_VALIDATION_CUT_RECORD])
        & (windows.start >= base.TRAIN_VALIDATION_CUT_SAMPLE)
    )
    normal_train = base.normal_support_indices(
        dataset, windows, "train", train
    )
    normal_validation = base.normal_support_indices(
        dataset, windows, "validation", validation
    )
    if len(normal_train) != 978 or len(normal_validation) != 295:
        raise AssertionError(
            "Frozen clean-normal support changed: "
            f"{len(normal_train)}/{len(normal_validation)}"
        )
    if int(windows.target_end[train].max()) != base.TRAIN_VALIDATION_CUT_SAMPLE:
        raise AssertionError("Training endpoint changed")
    if int(windows.start[validation].min()) != base.TRAIN_VALIDATION_CUT_SAMPLE:
        raise AssertionError("Validation start changed")
    scaler, scaler_metadata = base.fit_training_scaler(dataset)
    metadata = {
        "all_train_windows": int(len(train)),
        "all_validation_windows": int(len(validation)),
        "clean_normal_train_windows": int(len(normal_train)),
        "clean_normal_validation_windows": int(len(normal_validation)),
        "normal_train_window_index_sha256": array_sha256(normal_train),
        "normal_validation_window_index_sha256": array_sha256(normal_validation),
        "records_in_diagnostic": [record.record_id for record in dataset.records],
        "excluded_test_record": base.TEST_RECORD,
        "test_array_file_opened": False,
        "test_loader_created": False,
        "test_windows_forwarded": 0,
        "test_predictions_computed": False,
    }
    return (
        dataset,
        windows,
        normal_train,
        normal_validation,
        scaler,
        {"support": metadata, "scaler": scaler_metadata},
    )


def temporal_spread_order(count: int) -> list[int]:
    """Return a deterministic centre-out, recursively spread index order."""

    if count <= 0:
        return []
    order: list[int] = []
    level: list[tuple[int, int]] = [(0, int(count))]
    while level:
        following: list[tuple[int, int]] = []
        for left, right in level:
            middle = (left + right - 1) // 2
            order.append(middle)
            if left < middle:
                following.append((left, middle))
            if middle + 1 < right:
                following.append((middle + 1, right))
        level = following
    if sorted(order) != list(range(count)):
        raise AssertionError("Temporal spread order is not a permutation")
    return order


def unique_raw_support_points(
    windows: WindowTable, indices: np.ndarray
) -> tuple[int, dict[int, int]]:
    """Measure the exact union of complete 4-second window intervals."""

    values = np.asarray(indices, dtype=np.int64)
    total = 0
    by_record: dict[int, int] = {}
    for record_index in np.unique(windows.record_index[values]):
        selected = values[windows.record_index[values] == record_index]
        intervals = sorted(
            (
                int(windows.start[index]),
                int(windows.target_end[index]),
            )
            for index in selected
        )
        merged_points = 0
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged_points += current_end - current_start
                current_start, current_end = start, end
        merged_points += current_end - current_start
        by_record[int(record_index)] = int(merged_points)
        total += merged_points
    return int(total), by_record


def duration_fraction_subsets(
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    fractions: Sequence[float],
    block_seconds: int = TEMPORAL_BLOCK_SECONDS,
) -> tuple[dict[float, np.ndarray], dict[str, Any]]:
    """Create nested subsets targeting fractions of unique raw-time support.

    Clean-normal windows are first divided within each contiguous clean run into
    30-second *window-start* blocks.  Every block contains whole 4-second
    windows, so its exact raw support can extend by at most three seconds beyond
    the nominal start-time bin.  Blocks receive a deterministic temporally
    spread priority within each record; a prefix of that priority is selected
    to approximate each requested exact-union duration fraction.  This keeps
    adjacent overlapping windows together and avoids the false 96% duration
    coverage produced by evenly thinning individual windows.
    """

    values = np.asarray(indices, dtype=np.int64)
    if not len(values):
        raise ValueError("Clean-normal training support is empty")
    block_samples = int(block_seconds * dataset.sampling_rate_hz)
    if block_samples <= 0:
        raise ValueError("Temporal block duration must be positive")

    blocks: list[dict[str, Any]] = []
    for record_index in sorted(np.unique(windows.record_index[values]).tolist()):
        record_values = values[windows.record_index[values] == record_index]
        record_values = record_values[
            np.argsort(windows.start[record_values], kind="stable")
        ]
        starts = windows.start[record_values].astype(np.int64)
        split_positions = np.flatnonzero(
            np.diff(starts) != base.STRIDE_SAMPLES
        ) + 1
        clean_runs = np.split(record_values, split_positions)
        for run_index, run_values in enumerate(clean_runs):
            run_starts = windows.start[run_values].astype(np.int64)
            bin_ids = (run_starts - run_starts[0]) // block_samples
            for bin_id in np.unique(bin_ids):
                block_values = run_values[bin_ids == bin_id]
                blocks.append(
                    {
                        "record_index": int(record_index),
                        "record_id": dataset.records[int(record_index)].record_id,
                        "clean_run_index": int(run_index),
                        "start_bin": int(bin_id),
                        "indices": np.asarray(block_values, dtype=np.int64),
                        "first_start": int(windows.start[block_values[0]]),
                    }
                )

    # Priority is proportional within records, so every duration arm remains
    # record-stratified while prefixes remain strictly nested.
    prioritized: list[tuple[float, str, int, dict[str, Any]]] = []
    for record_index in sorted({block["record_index"] for block in blocks}):
        record_blocks = sorted(
            (block for block in blocks if block["record_index"] == record_index),
            key=lambda item: (
                item["clean_run_index"],
                item["start_bin"],
                item["first_start"],
            ),
        )
        order = temporal_spread_order(len(record_blocks))
        for rank, position in enumerate(order):
            prioritized.append(
                (
                    (rank + 0.5) / len(record_blocks),
                    record_blocks[position]["record_id"],
                    rank,
                    record_blocks[position],
                )
            )
    prioritized.sort(key=lambda item: (item[0], item[1], item[2]))
    ordered_blocks = [item[-1] for item in prioritized]

    full_points, full_points_by_record = unique_raw_support_points(windows, values)
    prefix_indices: list[np.ndarray] = []
    prefix_points: list[int] = []
    accumulated: list[np.ndarray] = []
    for block in ordered_blocks:
        accumulated.append(block["indices"])
        selected = np.sort(np.concatenate(accumulated)).astype(np.int64)
        points, _ = unique_raw_support_points(windows, selected)
        prefix_indices.append(selected)
        prefix_points.append(points)

    subsets: dict[float, np.ndarray] = {}
    fraction_audit: dict[str, Any] = {}
    prior_set: set[int] = set()
    prior_block_count = 0
    for fraction in sorted(float(value) for value in fractions):
        if fraction >= 1.0:
            selected = values.copy()
            block_count = len(ordered_blocks)
        else:
            target = fraction * full_points
            candidates = np.asarray(prefix_points, dtype=np.float64)
            block_count = int(np.argmin(np.abs(candidates - target))) + 1
            selected = prefix_indices[block_count - 1].copy()
        selected = np.sort(selected).astype(np.int64)
        selected_set = set(map(int, selected))
        if not prior_set.issubset(selected_set) or block_count < prior_block_count:
            raise AssertionError("Duration-fraction subsets are not nested")
        prior_set = selected_set
        prior_block_count = block_count
        points, points_by_record = unique_raw_support_points(windows, selected)
        duplication = (
            len(selected)
            * (base.CONTEXT_SAMPLES + base.TARGET_SAMPLES)
            / max(points, 1)
        )
        record_windows = {
            dataset.records[int(record_index)].record_id: int(
                np.sum(windows.record_index[selected] == record_index)
            )
            for record_index in sorted(points_by_record)
        }
        record_seconds = {
            dataset.records[int(record_index)].record_id: (
                points_by_record[record_index] / dataset.sampling_rate_hz
            )
            for record_index in sorted(points_by_record)
        }
        subsets[fraction] = selected
        fraction_audit[f"{fraction:g}"] = {
            "requested_duration_fraction": fraction,
            "selected_temporal_blocks": block_count,
            "total_temporal_blocks": len(ordered_blocks),
            "unique_training_windows": int(len(selected)),
            "unique_raw_support_points": points,
            "unique_raw_support_seconds": points / dataset.sampling_rate_hz,
            "achieved_full_duration_fraction": points / full_points,
            "window_support_duplication_factor": float(duplication),
            "windows_by_record": record_windows,
            "unique_raw_seconds_by_record": record_seconds,
            "window_index_sha256": array_sha256(selected),
        }

    audit = {
        "definition": (
            "Nested, record-stratified 30-second window-start blocks; fractions "
            "target exact unions of complete context+target support."
        ),
        "block_seconds": int(block_seconds),
        "full_unique_raw_support_points": full_points,
        "full_unique_raw_support_seconds": full_points / dataset.sampling_rate_hz,
        "full_unique_raw_support_points_by_record_index": full_points_by_record,
        "fractions": fraction_audit,
    }
    return subsets, audit


def matched_epoch_exposure(
    train_indices: np.ndarray,
    reference_windows: int,
    seed: int,
) -> np.ndarray:
    """Return one deterministic, step-matched epoch of training indices."""

    values = np.asarray(train_indices, dtype=np.int64)
    if not len(values) or reference_windows < len(values):
        raise ValueError("Invalid matched-exposure support")
    if len(values) == reference_windows:
        return values.copy()
    rng = np.random.default_rng(int(seed))
    return rng.choice(values, size=int(reference_windows), replace=True).astype(
        np.int64, copy=False
    )


def scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float | int]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


@torch.no_grad()
def evaluate_model(
    model: GRUNBM,
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    scaler: RobustChannelScaler,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    loader = base.sequence_loader(
        dataset,
        windows,
        indices,
        scaler,
        batch_size,
        False,
        num_workers,
        device.type == "cuda",
    )
    model.eval()
    channels = dataset.n_channels
    horizon = base.TARGET_SAMPLES
    total_values = 0
    nll_sum = 0.0
    log_sigma_sum = 0.0
    half_z2_sum = 0.0
    error_squared_sum = 0.0
    error_absolute_sum = 0.0
    sigma_sum = 0.0
    z_sum = 0.0
    z_squared_sum = 0.0
    coverage_1 = 0
    coverage_196 = 0
    sigma_min_count = 0
    sigma_max_count = 0
    per_channel_squared = np.zeros(channels, dtype=np.float64)
    per_horizon_squared = np.zeros(horizon, dtype=np.float64)
    window_count = 0

    for sequence, _, _ in loader:
        sequence = sequence.to(device, non_blocking=True)
        context = sequence[:, :, : base.CONTEXT_SAMPLES]
        target = sequence[:, :, base.CONTEXT_SAMPLES :]
        # Float32 evaluation is intentional: AMP training speed must not change
        # the metric used for early stopping or cross-run comparisons.
        mean, sigma = model(context.float())
        error = target.float() - mean.float()
        z = error / sigma.float()
        nll = torch.log(sigma.float()) + 0.5 * z.square()
        batch_values = int(z.numel())
        total_values += batch_values
        window_count += int(sequence.shape[0])
        nll_sum += float(nll.double().sum().cpu())
        log_sigma_sum += float(torch.log(sigma.float()).double().sum().cpu())
        half_z2_sum += float((0.5 * z.square()).double().sum().cpu())
        error_squared_sum += float(error.square().double().sum().cpu())
        error_absolute_sum += float(error.abs().double().sum().cpu())
        sigma_sum += float(sigma.double().sum().cpu())
        z_sum += float(z.double().sum().cpu())
        z_squared_sum += float(z.square().double().sum().cpu())
        coverage_1 += int((z.abs() <= 1.0).sum().cpu())
        coverage_196 += int((z.abs() <= 1.96).sum().cpu())
        sigma_min_count += int((sigma <= SIGMA_MIN * (1.0 + 1e-6)).sum().cpu())
        sigma_max_count += int((sigma >= SIGMA_MAX * (1.0 - 1e-6)).sum().cpu())
        squared = error.square().double().cpu().numpy()
        per_channel_squared += squared.sum(axis=(0, 2))
        per_horizon_squared += squared.sum(axis=(0, 1))

    if window_count != len(indices) or total_values <= 0:
        raise AssertionError("Evaluation support changed")
    per_channel_rmse = np.sqrt(
        per_channel_squared / (window_count * horizon)
    )
    per_horizon_rmse = np.sqrt(
        per_horizon_squared / (window_count * channels)
    )
    quartile_rmse = [
        float(np.sqrt(np.mean(per_horizon_rmse[start : start + 32] ** 2)))
        for start in range(0, horizon, 32)
    ]
    result = {
        "windows": window_count,
        "gaussian_nll": nll_sum / total_values,
        "mean_log_sigma": log_sigma_sum / total_values,
        "mean_half_standardized_squared_error": half_z2_sum / total_values,
        "forecast_rmse_scaled": math.sqrt(error_squared_sum / total_values),
        "forecast_mae_scaled": error_absolute_sum / total_values,
        "mean_sigma_scaled": sigma_sum / total_values,
        "standardized_residual_mean": z_sum / total_values,
        "standardized_residual_rms": math.sqrt(z_squared_sum / total_values),
        "coverage_abs_z_le_1": coverage_1 / total_values,
        "coverage_abs_z_le_1p96": coverage_196 / total_values,
        "sigma_at_min_bound_fraction": sigma_min_count / total_values,
        "sigma_at_max_bound_fraction": sigma_max_count / total_values,
        "per_channel_rmse_scaled": per_channel_rmse.tolist(),
        "per_horizon_rmse_scaled": per_horizon_rmse.tolist(),
        "quartile_rmse_scaled": quartile_rmse,
    }
    if not all(
        math.isfinite(float(value))
        for value in scalar_metrics(result).values()
    ):
        raise FloatingPointError("Non-finite GRU evaluation metric")
    decomposed = (
        result["mean_log_sigma"]
        + result["mean_half_standardized_squared_error"]
    )
    if abs(result["gaussian_nll"] - decomposed) > 1e-6:
        raise AssertionError("Gaussian NLL decomposition mismatch")
    return result


def train_epoch(
    model: GRUNBM,
    loader,
    optimizer: torch.optim.Optimizer,
    grad_scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    windows = 0
    gradient_norms: list[float] = []
    clipped_steps = 0
    for sequence, _, _ in loader:
        sequence = sequence.to(device, non_blocking=True)
        context = sequence[:, :, : base.CONTEXT_SAMPLES]
        target = sequence[:, :, base.CONTEXT_SAMPLES :]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device.type, enabled=bool(amp and device.type == "cuda")
        ):
            mean, sigma = model(context)
            loss = gaussian_nll_sigma(target, mean, sigma)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite GRU training loss")
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        gradient_norm = float(
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        )
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("Non-finite GRU gradient norm")
        gradient_norms.append(gradient_norm)
        clipped_steps += int(gradient_norm > 5.0)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        batch = int(sequence.shape[0])
        loss_sum += float(loss.detach()) * batch
        windows += batch
    return {
        "optimization_train_nll": loss_sum / max(windows, 1),
        "mean_preclip_gradient_norm": float(np.mean(gradient_norms)),
        "max_preclip_gradient_norm": float(np.max(gradient_norms)),
        "gradient_clip_step_fraction": clipped_steps / max(len(gradient_norms), 1),
        "optimizer_steps": len(gradient_norms),
    }


def collect_scaled_windows(
    dataset: DaphnetDataset,
    windows: WindowTable,
    indices: np.ndarray,
    scaler: RobustChannelScaler,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = base.sequence_loader(
        dataset,
        windows,
        indices,
        scaler,
        batch_size,
        False,
        num_workers,
        False,
    )
    last_values: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for sequence, _, _ in loader:
        array = sequence.numpy().astype(np.float32, copy=False)
        last_values.append(array[:, :, base.CONTEXT_SAMPLES - 1 : base.CONTEXT_SAMPLES])
        targets.append(array[:, :, base.CONTEXT_SAMPLES :])
    return np.concatenate(last_values), np.concatenate(targets)


def gaussian_array_metrics(
    target: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, float]:
    error = target.astype(np.float64) - mean.astype(np.float64)
    sigma64 = np.broadcast_to(sigma, target.shape).astype(np.float64, copy=False)
    z = error / sigma64
    return {
        "gaussian_nll": float(np.mean(np.log(sigma64) + 0.5 * z**2)),
        "forecast_rmse_scaled": float(np.sqrt(np.mean(error**2))),
        "forecast_mae_scaled": float(np.mean(np.abs(error))),
        "mean_sigma_scaled": float(np.mean(sigma64)),
        "standardized_residual_rms": float(np.sqrt(np.mean(z**2))),
        "coverage_abs_z_le_1": float(np.mean(np.abs(z) <= 1.0)),
        "coverage_abs_z_le_1p96": float(np.mean(np.abs(z) <= 1.96)),
    }


def persistence_baseline(
    dataset: DaphnetDataset,
    windows: WindowTable,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    scaler: RobustChannelScaler,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    train_last, train_target = collect_scaled_windows(
        dataset, windows, train_indices, scaler, batch_size, num_workers
    )
    validation_last, validation_target = collect_scaled_windows(
        dataset, windows, validation_indices, scaler, batch_size, num_workers
    )
    train_mean = np.repeat(train_last, base.TARGET_SAMPLES, axis=2)
    validation_mean = np.repeat(validation_last, base.TARGET_SAMPLES, axis=2)
    sigma = np.sqrt(np.mean((train_target - train_mean) ** 2, axis=0, keepdims=True))
    sigma = np.clip(sigma, SIGMA_MIN, SIGMA_MAX).astype(np.float32)
    return {
        "definition": (
            "Repeat the final context sample for 128 steps; per-channel/horizon "
            "sigma is the train-only analytic RMS error clipped to GRU bounds."
        ),
        "sigma_shape": list(sigma.shape),
        "sigma_sha256": array_sha256(sigma),
        "train": gaussian_array_metrics(train_target, train_mean, sigma),
        "validation": gaussian_array_metrics(
            validation_target, validation_mean, sigma
        ),
    }


@torch.no_grad()
def train_calibrated_gru_sigma(
    model: GRUNBM,
    dataset: DaphnetDataset,
    windows: WindowTable,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    scaler: RobustChannelScaler,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate the GRU mean with sigma estimated only from train residuals."""

    def collect(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        loader = base.sequence_loader(
            dataset,
            windows,
            indices,
            scaler,
            batch_size,
            False,
            num_workers,
            device.type == "cuda",
        )
        means: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        model.eval()
        for sequence, _, _ in loader:
            sequence = sequence.to(device, non_blocking=True)
            context = sequence[:, :, : base.CONTEXT_SAMPLES]
            target = sequence[:, :, base.CONTEXT_SAMPLES :]
            mean, _ = model(context.float())
            means.append(mean.float().cpu().numpy())
            targets.append(target.float().cpu().numpy())
        return np.concatenate(means), np.concatenate(targets)

    train_mean, train_target = collect(train_indices)
    validation_mean, validation_target = collect(validation_indices)
    sigma = np.sqrt(
        np.mean((train_target - train_mean) ** 2, axis=0, keepdims=True)
    )
    sigma = np.clip(sigma, SIGMA_MIN, SIGMA_MAX).astype(np.float32)
    return {
        "definition": (
            "Best-checkpoint GRU mean with per-channel/horizon sigma fitted "
            "analytically from that run's unique training residuals only."
        ),
        "sigma_shape": list(sigma.shape),
        "sigma_sha256": array_sha256(sigma),
        "train": gaussian_array_metrics(train_target, train_mean, sigma),
        "validation": gaussian_array_metrics(
            validation_target, validation_mean, sigma
        ),
    }


def flatten_epoch_row(
    epoch: int,
    optimization: Mapping[str, float],
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    improved: bool,
    cumulative_optimizer_steps: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": int(epoch),
        "cumulative_optimizer_steps": int(cumulative_optimizer_steps),
        **optimization,
        "improved": bool(improved),
    }
    for prefix, metrics in (("train_eval", train_metrics), ("validation", validation_metrics)):
        for key, value in scalar_metrics(metrics).items():
            row[f"{prefix}_{key}"] = value
        for quarter, value in enumerate(metrics["quartile_rmse_scaled"], start=1):
            row[f"{prefix}_quartile{quarter}_rmse_scaled"] = value
    return row


def train_one_run(
    args: argparse.Namespace,
    dataset: DaphnetDataset,
    windows: WindowTable,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    scaler: RobustChannelScaler,
    fraction: float,
    fraction_support: Mapping[str, Any],
    reference_train_windows: int,
    seed: int,
    protocol_fingerprint: str,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    run_root = output_dir / "runs" / fraction_name(fraction) / f"seed_{seed}"
    run_root.mkdir(parents=True, exist_ok=False)
    atomic_npz_save(
        run_root / "support.npz",
        train_window_index=train_indices,
        validation_window_index=validation_indices,
    )
    set_seed(seed, args.deterministic)
    model = GRUNBM(
        in_channels=dataset.n_channels,
        horizon=base.TARGET_SAMPLES,
        hidden_channels=args.hidden_channels,
        num_layers=1,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(args.amp and device.type == "cuda")
    )
    best_path = run_root / "best.pt"
    last_path = run_root / "last.pt"
    initial_train = evaluate_model(
        model,
        dataset,
        windows,
        train_indices,
        scaler,
        args.batch_size,
        args.num_workers,
        device,
    )
    initial_validation = evaluate_model(
        model,
        dataset,
        windows,
        validation_indices,
        scaler,
        args.batch_size,
        args.num_workers,
        device,
    )
    history: list[dict[str, Any]] = []
    best_nll = float("inf")
    best_epoch = 0
    bad_epochs = 0
    cumulative_optimizer_steps = 0
    started = time.perf_counter()

    for epoch in range(1, args.max_epochs + 1):
        if epoch > MIN_EPOCHS and bad_epochs >= args.patience:
            break
        epoch_indices = matched_epoch_exposure(
            train_indices,
            reference_train_windows,
            seed=seed * 100_000 + epoch,
        )
        train_loader = base.sequence_loader(
            dataset,
            windows,
            epoch_indices,
            scaler,
            args.batch_size,
            True,
            args.num_workers,
            device.type == "cuda",
            seed=seed + epoch,
        )
        optimization = train_epoch(
            model, train_loader, optimizer, grad_scaler, device, args.amp
        )
        cumulative_optimizer_steps += int(optimization["optimizer_steps"])
        optimization = {
            **optimization,
            "epoch_window_exposures": int(len(epoch_indices)),
            "unique_training_windows": int(len(train_indices)),
        }
        train_metrics = evaluate_model(
            model,
            dataset,
            windows,
            train_indices,
            scaler,
            args.batch_size,
            args.num_workers,
            device,
        )
        validation_metrics = evaluate_model(
            model,
            dataset,
            windows,
            validation_indices,
            scaler,
            args.batch_size,
            args.num_workers,
            device,
        )
        validation_nll = float(validation_metrics["gaussian_nll"])
        if not math.isfinite(validation_nll):
            raise FloatingPointError("Non-finite validation NLL")
        improved = validation_nll < best_nll - MIN_DELTA
        history.append(
            flatten_epoch_row(
                epoch,
                optimization,
                train_metrics,
                validation_metrics,
                improved,
                cumulative_optimizer_steps,
            )
        )
        if improved:
            best_nll = validation_nll
            best_epoch = epoch
            bad_epochs = 0
            atomic_torch_save(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "protocol_fingerprint": protocol_fingerprint,
                    "fraction": fraction,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_clean_normal_nll": validation_nll,
                    "model_config": model.model_config(),
                    "model_state": model.state_dict(),
                },
                best_path,
            )
        else:
            bad_epochs += 1
        atomic_torch_save(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "protocol_fingerprint": protocol_fingerprint,
                "fraction": fraction,
                "seed": seed,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_clean_normal_nll": best_nll,
                "bad_epochs": bad_epochs,
                "model_config": model.model_config(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            last_path,
        )
        print(
            f"[fraction={fraction:g} seed={seed}] epoch={epoch:02d} "
            f"train_eval_nll={train_metrics['gaussian_nll']:.6f} "
            f"val_nll={validation_nll:.6f} "
            f"val_rmse={validation_metrics['forecast_rmse_scaled']:.6f}"
            f"{' *' if improved else ''}",
            flush=True,
        )

    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    best_train = evaluate_model(
        model,
        dataset,
        windows,
        train_indices,
        scaler,
        args.batch_size,
        args.num_workers,
        device,
    )
    best_validation = evaluate_model(
        model,
        dataset,
        windows,
        validation_indices,
        scaler,
        args.batch_size,
        args.num_workers,
        device,
    )
    persistence = persistence_baseline(
        dataset,
        windows,
        train_indices,
        validation_indices,
        scaler,
        args.batch_size,
        args.num_workers,
    )
    calibrated_gru_sigma = train_calibrated_gru_sigma(
        model,
        dataset,
        windows,
        train_indices,
        validation_indices,
        scaler,
        args.batch_size,
        args.num_workers,
        device,
    )
    epoch8_row = next((row for row in history if row["epoch"] == 8), None)
    if epoch8_row is None:
        raise AssertionError("Every diagnostic run must reach epoch 8")
    validation_values = np.asarray(
        [row["validation_gaussian_nll"] for row in history], dtype=np.float64
    )
    slope_count = min(5, len(validation_values))
    last_slope = float(
        np.polyfit(
            np.arange(slope_count, dtype=np.float64),
            validation_values[-slope_count:],
            1,
        )[0]
    )
    last_validation_nll = float(validation_values[-1])
    stop_reason = (
        "validation_patience"
        if bad_epochs >= args.patience
        else "maximum_epochs"
    )
    gradient_rows = [
        row for row in history if math.isfinite(row["mean_preclip_gradient_norm"])
    ]
    summary = {
        "fraction": fraction,
        "seed": seed,
        "train_windows": int(len(train_indices)),
        "fraction_support": dict(fraction_support),
        "matched_window_exposures_per_epoch": int(reference_train_windows),
        "optimizer_steps_per_epoch": int(
            math.ceil(reference_train_windows / args.batch_size)
        ),
        "cumulative_optimizer_steps": int(cumulative_optimizer_steps),
        "validation_windows": int(len(validation_indices)),
        "train_window_index_sha256": array_sha256(train_indices),
        "maximum_epochs": args.max_epochs,
        "minimum_epochs": MIN_EPOCHS,
        "patience": args.patience,
        "min_delta": MIN_DELTA,
        "epochs_completed": len(history),
        "stop_reason": stop_reason,
        "best_epoch": best_epoch,
        "best_validation_clean_normal_nll": best_nll,
        "last_validation_clean_normal_nll": last_validation_nll,
        "epoch8_validation_clean_normal_nll": float(
            epoch8_row["validation_gaussian_nll"]
        ),
        "absolute_validation_nll_improvement_after_epoch8": float(
            epoch8_row["validation_gaussian_nll"] - best_nll
        ),
        "relative_validation_nll_improvement_after_epoch8": float(
            (epoch8_row["validation_gaussian_nll"] - best_nll)
            / max(abs(epoch8_row["validation_gaussian_nll"]), 1e-12)
        ),
        "last5_validation_nll_slope_per_epoch": last_slope,
        "initial": {
            "train": initial_train,
            "validation": initial_validation,
        },
        "epoch8": {
            "train_gaussian_nll": epoch8_row["train_eval_gaussian_nll"],
            "validation_gaussian_nll": epoch8_row["validation_gaussian_nll"],
            "validation_forecast_rmse_scaled": epoch8_row[
                "validation_forecast_rmse_scaled"
            ],
            "validation_mean_sigma_scaled": epoch8_row[
                "validation_mean_sigma_scaled"
            ],
            "validation_coverage_abs_z_le_1p96": epoch8_row[
                "validation_coverage_abs_z_le_1p96"
            ],
        },
        "best": {
            "train": best_train,
            "validation": best_validation,
        },
        "persistence_baseline": persistence,
        "train_calibrated_gru_sigma": calibrated_gru_sigma,
        "gradient_diagnostics": {
            "mean_preclip_gradient_norm": float(
                np.mean([row["mean_preclip_gradient_norm"] for row in gradient_rows])
            ),
            "maximum_preclip_gradient_norm": float(
                np.max([row["max_preclip_gradient_norm"] for row in gradient_rows])
            ),
            "mean_gradient_clip_step_fraction": float(
                np.mean([row["gradient_clip_step_fraction"] for row in gradient_rows])
            ),
            "all_losses_and_gradients_finite": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "history": history,
    }
    atomic_json_dump(summary, run_root / "summary.json")
    write_csv(run_root / "history.csv", history)
    return summary


def clean_normal_point_shift(
    dataset: DaphnetDataset,
    scaler: RobustChannelScaler,
) -> dict[str, Any]:
    by_name = {record.record_id: record for record in dataset.records}
    train_ranges = [
        (by_name[base.TRAIN_RECORD], 0, len(by_name[base.TRAIN_RECORD].y)),
        (
            by_name[base.TRAIN_VALIDATION_CUT_RECORD],
            0,
            base.TRAIN_VALIDATION_CUT_SAMPLE,
        ),
    ]
    validation_record = by_name[base.TRAIN_VALIDATION_CUT_RECORD]
    validation_ranges = [
        (
            validation_record,
            base.TRAIN_VALIDATION_CUT_SAMPLE,
            len(validation_record.y),
        )
    ]

    def values(ranges):
        chunks = []
        for record, start, end in ranges:
            mask = record.valid[start:end] & (record.y[start:end] == 0)
            chunks.append(record.x[start:end][mask])
        raw = np.concatenate(chunks)
        unclipped = (
            raw.astype(np.float64) - scaler.center.astype(np.float64)
        ) / scaler.scale.astype(np.float64)
        return unclipped

    train = values(train_ranges)
    validation = values(validation_ranges)
    train_mean = train.mean(axis=0)
    validation_mean = validation.mean(axis=0)
    pooled_std = np.sqrt(0.5 * (train.var(axis=0) + validation.var(axis=0)))
    smd = (validation_mean - train_mean) / np.maximum(pooled_std, 1e-12)
    train_iqr = np.percentile(train, 75, axis=0) - np.percentile(train, 25, axis=0)
    validation_iqr = np.percentile(validation, 75, axis=0) - np.percentile(
        validation, 25, axis=0
    )
    return {
        "definition": "Unique valid non-FOG points; RobustScaler fitted on train only.",
        "train_points": int(len(train)),
        "validation_points": int(len(validation)),
        "train_mean_scaled": train_mean.tolist(),
        "validation_mean_scaled": validation_mean.tolist(),
        "standardized_mean_difference": smd.tolist(),
        "mean_absolute_standardized_mean_difference": float(np.mean(np.abs(smd))),
        "maximum_absolute_standardized_mean_difference": float(np.max(np.abs(smd))),
        "validation_to_train_iqr_ratio": (
            validation_iqr / np.maximum(train_iqr, 1e-12)
        ).tolist(),
        "train_input_clip_fraction": float(np.mean(np.abs(train) > base.ROBUST_CLIP)),
        "validation_input_clip_fraction": float(
            np.mean(np.abs(validation) > base.ROBUST_CLIP)
        ),
    }


def aggregate_runs(
    summaries: Sequence[Mapping[str, Any]], fractions: Sequence[float]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        rows.append(
            {
                "fraction": summary["fraction"],
                "seed": summary["seed"],
                "train_windows": summary["train_windows"],
                "unique_raw_support_seconds": summary["fraction_support"][
                    "unique_raw_support_seconds"
                ],
                "achieved_duration_fraction": summary["fraction_support"][
                    "achieved_full_duration_fraction"
                ],
                "selected_temporal_blocks": summary["fraction_support"][
                    "selected_temporal_blocks"
                ],
                "matched_window_exposures_per_epoch": summary[
                    "matched_window_exposures_per_epoch"
                ],
                "cumulative_optimizer_steps": summary[
                    "cumulative_optimizer_steps"
                ],
                "epochs_completed": summary["epochs_completed"],
                "stop_reason": summary["stop_reason"],
                "best_epoch": summary["best_epoch"],
                "epoch8_validation_nll": summary[
                    "epoch8_validation_clean_normal_nll"
                ],
                "best_validation_nll": summary[
                    "best_validation_clean_normal_nll"
                ],
                "improvement_after_epoch8": summary[
                    "absolute_validation_nll_improvement_after_epoch8"
                ],
                "best_train_nll": summary["best"]["train"]["gaussian_nll"],
                "best_train_rmse": summary["best"]["train"][
                    "forecast_rmse_scaled"
                ],
                "best_validation_rmse": summary["best"]["validation"][
                    "forecast_rmse_scaled"
                ],
                "best_validation_mean_log_sigma": summary["best"][
                    "validation"
                ]["mean_log_sigma"],
                "best_validation_half_z2": summary["best"]["validation"][
                    "mean_half_standardized_squared_error"
                ],
                "best_validation_mean_sigma": summary["best"]["validation"][
                    "mean_sigma_scaled"
                ],
                "best_validation_z_rms": summary["best"]["validation"][
                    "standardized_residual_rms"
                ],
                "best_validation_coverage_95": summary["best"]["validation"][
                    "coverage_abs_z_le_1p96"
                ],
                "persistence_validation_nll": summary["persistence_baseline"][
                    "validation"
                ]["gaussian_nll"],
                "gru_minus_persistence_validation_nll": (
                    summary["best"]["validation"]["gaussian_nll"]
                    - summary["persistence_baseline"]["validation"][
                        "gaussian_nll"
                    ]
                ),
                "persistence_validation_rmse": summary["persistence_baseline"][
                    "validation"
                ]["forecast_rmse_scaled"],
                "gru_rmse_skill_vs_persistence": (
                    1.0
                    - summary["best"]["validation"]["forecast_rmse_scaled"]
                    / summary["persistence_baseline"]["validation"][
                        "forecast_rmse_scaled"
                    ]
                ),
                "train_calibrated_sigma_validation_nll": summary[
                    "train_calibrated_gru_sigma"
                ]["validation"]["gaussian_nll"],
                "learned_minus_train_calibrated_sigma_validation_nll": (
                    summary["best"]["validation"]["gaussian_nll"]
                    - summary["train_calibrated_gru_sigma"]["validation"][
                        "gaussian_nll"
                    ]
                ),
                "last5_validation_nll_slope": summary[
                    "last5_validation_nll_slope_per_epoch"
                ],
                "gradient_clip_step_fraction": summary["gradient_diagnostics"][
                    "mean_gradient_clip_step_fraction"
                ],
            }
        )

    aggregate: dict[str, Any] = {}
    for fraction in fractions:
        selected = [row for row in rows if float(row["fraction"]) == float(fraction)]
        numeric_keys = (
            "epochs_completed",
            "cumulative_optimizer_steps",
            "best_epoch",
            "epoch8_validation_nll",
            "best_validation_nll",
            "improvement_after_epoch8",
            "best_train_nll",
            "best_train_rmse",
            "best_validation_rmse",
            "best_validation_mean_log_sigma",
            "best_validation_half_z2",
            "best_validation_mean_sigma",
            "best_validation_z_rms",
            "best_validation_coverage_95",
            "persistence_validation_nll",
            "gru_minus_persistence_validation_nll",
            "persistence_validation_rmse",
            "gru_rmse_skill_vs_persistence",
            "train_calibrated_sigma_validation_nll",
            "learned_minus_train_calibrated_sigma_validation_nll",
            "last5_validation_nll_slope",
            "gradient_clip_step_fraction",
        )
        stats: dict[str, Any] = {
            "runs": len(selected),
            "train_windows": selected[0]["train_windows"],
            "unique_raw_support_seconds": selected[0][
                "unique_raw_support_seconds"
            ],
            "achieved_duration_fraction": selected[0][
                "achieved_duration_fraction"
            ],
            "selected_temporal_blocks": selected[0]["selected_temporal_blocks"],
            "matched_window_exposures_per_epoch": selected[0][
                "matched_window_exposures_per_epoch"
            ],
            "patience_stop_count": sum(
                row["stop_reason"] == "validation_patience" for row in selected
            ),
            "maximum_epoch_stop_count": sum(
                row["stop_reason"] == "maximum_epochs" for row in selected
            ),
        }
        for key in numeric_keys:
            values = np.asarray([float(row[key]) for row in selected])
            stats[key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        aggregate[f"{fraction:g}"] = stats
    return rows, aggregate


def evidence_assessment(
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    full = [row for row in rows if float(row["fraction"]) == 1.0]
    median_best_epoch = float(np.median([row["best_epoch"] for row in full]))
    gains_after8 = np.asarray(
        [row["improvement_after_epoch8"] for row in full], dtype=np.float64
    )
    median_after8 = float(np.median(gains_after8))
    best_after8_count = sum(int(row["best_epoch"]) > 8 for row in full)
    required_seed_count = int(math.ceil(0.8 * len(full)))
    patience_stops = sum(
        row["stop_reason"] == "validation_patience" for row in full
    )
    max_stops = len(full) - patience_stops
    full_mean = float(aggregate["1"]["best_validation_nll"]["mean"])
    ordered_fractions = sorted(float(key) for key in aggregate)
    fraction_means = [
        float(aggregate[f"{fraction:g}"]["best_validation_nll"]["mean"])
        for fraction in ordered_fractions
    ]
    monotonic_with_tolerance = all(
        later <= earlier + 0.005
        for earlier, later in zip(fraction_means, fraction_means[1:])
    )
    smaller_fractions = [value for value in ordered_fractions if value < 1.0]
    nearest_fraction = max(smaller_fractions) if smaller_fractions else None
    paired_gains: list[float] = []
    if nearest_fraction is not None:
        lower_by_seed = {
            int(row["seed"]): float(row["best_validation_nll"])
            for row in rows
            if float(row["fraction"]) == nearest_fraction
        }
        full_by_seed = {
            int(row["seed"]): float(row["best_validation_nll"])
            for row in full
        }
        paired_gains = [
            lower_by_seed[seed] - full_by_seed[seed]
            for seed in sorted(set(lower_by_seed) & set(full_by_seed))
        ]
    mean_last_gain = float(np.mean(paired_gains)) if paired_gains else None
    positive_last_gain_count = sum(value > 0.0 for value in paired_gains)
    sigma_ratios = []
    rmse_reductions = []
    for summary_row in full:
        # The aggregate row stores best only; detailed epoch-8 comparisons are
        # inserted by the caller below when evidence.json is assembled.
        sigma_ratios.append(float(summary_row["best_validation_mean_sigma"]))
        rmse_reductions.append(float(summary_row["best_validation_rmse"]))
    seed_std = float(aggregate["1"]["best_validation_nll"]["std"])
    gradient_clip = float(aggregate["1"]["gradient_clip_step_fraction"]["mean"])
    budget_supported = bool(
        best_after8_count >= required_seed_count and median_after8 >= 0.01
    )
    data_quantity_supported = bool(
        paired_gains
        and monotonic_with_tolerance
        and mean_last_gain is not None
        and mean_last_gain >= 0.01
        and positive_last_gain_count >= math.ceil(0.8 * len(paired_gains))
    )
    optimization_instability_supported = bool(
        seed_std > 0.10 or gradient_clip > 0.20
    )
    return {
        "criteria_are_diagnostic_not_formal_hypothesis_tests": True,
        "eight_epoch_budget_insufficient": {
            "supported": budget_supported,
            "full_data_best_epochs_after_8_count": best_after8_count,
            "required_seed_count": required_seed_count,
            "median_full_data_best_epoch": median_best_epoch,
            "median_full_data_validation_nll_improvement_after_epoch8": median_after8,
            "per_seed_validation_nll_improvement_after_epoch8": gains_after8.tolist(),
            "criterion": (
                "at least 80% of full-data seeds best after epoch 8 and median "
                "post-8 validation-NLL gain >=0.01"
            ),
        },
        "convergence_within_40_epochs": {
            "all_runs_patience_stopped": patience_stops == len(full),
            "full_data_patience_stop_count": patience_stops,
            "full_data_maximum_epoch_stop_count": max_stops,
            "interpretation": (
                "Patience stop means the pre-registered validation-NLL selection "
                "rule was reached. It does not prove parameter convergence or "
                "future-mean convergence; a max-epoch stop is right-censored."
            ),
        },
        "more_clean_normal_duration_help": {
            "supported": data_quantity_supported,
            "ordered_requested_fractions": ordered_fractions,
            "mean_best_validation_nll_by_fraction": fraction_means,
            "aggregate_curve_monotonic_with_0p005_tolerance": monotonic_with_tolerance,
            "nearest_lower_fraction": nearest_fraction,
            "full_fraction_mean_best_validation_nll": full_mean,
            "paired_last_increment_nll_gains": paired_gains,
            "mean_last_increment_nll_gain": mean_last_gain,
            "positive_last_increment_seed_count": positive_last_gain_count,
            "criterion": (
                "duration curve monotonic within 0.005, nearest-to-full paired "
                "mean NLL gain >=0.01, and >=80% of seeds favor full duration"
            ),
            "scope": (
                "Conditional diagnostic only: every model seed uses the same fixed "
                "nested temporal-block path and the full-training scaler, while "
                "record/activity composition changes with fraction. A positive "
                "result shows benefit along this one path, not a causal duration "
                "effect and not proof that data beyond 100% are required."
            ),
        },
        "optimization_instability": {
            "supported": optimization_instability_supported,
            "full_data_seed_sd_best_validation_nll": seed_std,
            "full_data_mean_gradient_clip_step_fraction": gradient_clip,
            "all_recorded_losses_and_gradients_finite": True,
            "criterion": "seed SD >0.10 or more than 20% of optimizer steps clip gradients",
            "scope": (
                "This coarse screen detects non-finite values, exploding gradients, "
                "or very large seed dispersion only. A negative flag is not proof "
                "of global optimization stability."
            ),
        },
        "sigma_inflation": {
            "requires_detailed_epoch8_vs_best_check": True,
            "best_validation_mean_sigma_values": sigma_ratios,
            "best_validation_rmse_values": rmse_reductions,
        },
    }


def enrich_sigma_evidence(
    evidence: dict[str, Any], summaries: Sequence[Mapping[str, Any]]
) -> None:
    full = [summary for summary in summaries if float(summary["fraction"]) == 1.0]
    sigma_ratio = np.asarray(
        [
            summary["best"]["validation"]["mean_sigma_scaled"]
            / summary["epoch8"]["validation_mean_sigma_scaled"]
            for summary in full
        ],
        dtype=np.float64,
    )
    rmse_improvement = np.asarray(
        [
            (
                summary["epoch8"]["validation_forecast_rmse_scaled"]
                - summary["best"]["validation"]["forecast_rmse_scaled"]
            )
            / max(summary["epoch8"]["validation_forecast_rmse_scaled"], 1e-12)
            for summary in full
        ],
        dtype=np.float64,
    )
    z_rms = np.asarray(
        [
            summary["best"]["validation"]["standardized_residual_rms"]
            for summary in full
        ],
        dtype=np.float64,
    )
    coverage_95 = np.asarray(
        [
            summary["best"]["validation"]["coverage_abs_z_le_1p96"]
            for summary in full
        ],
        dtype=np.float64,
    )
    learned_minus_fixed = np.asarray(
        [
            summary["best"]["validation"]["gaussian_nll"]
            - summary["train_calibrated_gru_sigma"]["validation"][
                "gaussian_nll"
            ]
            for summary in full
        ],
        dtype=np.float64,
    )
    trend_flag = bool(
        np.mean(sigma_ratio) > 1.25 and np.mean(rmse_improvement) < 0.05
    )
    overcoverage_flag = bool(np.mean(z_rms) < 0.8 or np.mean(coverage_95) > 0.98)
    fixed_sigma_not_worse = bool(np.mean(learned_minus_fixed) >= -0.01)
    supported = bool(trend_flag and overcoverage_flag and fixed_sigma_not_worse)
    evidence["sigma_inflation"] = {
        "supported": supported,
        "sigma_growth_without_mean_gain_flag": trend_flag,
        "overcoverage_flag": overcoverage_flag,
        "train_calibrated_fixed_sigma_not_worse_flag": fixed_sigma_not_worse,
        "mean_best_to_epoch8_sigma_ratio": float(np.mean(sigma_ratio)),
        "mean_epoch8_to_best_rmse_relative_improvement": float(
            np.mean(rmse_improvement)
        ),
        "per_seed_sigma_ratio": sigma_ratio.tolist(),
        "per_seed_rmse_relative_improvement": rmse_improvement.tolist(),
        "mean_best_validation_z_rms": float(np.mean(z_rms)),
        "mean_best_validation_coverage_95": float(np.mean(coverage_95)),
        "mean_learned_minus_train_calibrated_sigma_validation_nll": float(
            np.mean(learned_minus_fixed)
        ),
        "criterion": (
            "sigma rises >25% while RMSE improves <5%, plus z-RMS<0.8 or "
            "95% coverage>0.98, and train-calibrated fixed sigma is not worse "
            "by more than 0.01 NLL"
        ),
    }


def protocol_payload(
    args: argparse.Namespace,
    seeds: Sequence[int],
    fractions: Sequence[float],
    dataset: DaphnetDataset,
    support_metadata: Mapping[str, Any],
    fraction_audit: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    with torch.random.fork_rng(devices=[]):
        model = GRUNBM(
            in_channels=dataset.n_channels,
            horizon=base.TARGET_SAMPLES,
            hidden_channels=args.hidden_channels,
            num_layers=1,
            dropout=args.dropout,
        )
    payload: dict[str, Any] = {
        "experiment_version": EXPERIMENT_VERSION,
        "created_utc": utc_now(),
        "objective": "Explain the earlier apparent failure of GRU-NBM convergence.",
        "subject": base.SUBJECT_ID,
        "data_dir": str(args.data_dir.resolve()),
        "records_used": [record.record_id for record in dataset.records],
        "held_out_test_record": base.TEST_RECORD,
        "test_policy": (
            "The S01_seg002/R02 array file is never opened. No test loader, "
            "window, forecast, residual, metric, checkpoint selection, or report "
            "statistic is created in this diagnostic."
        ),
        "prior_test_disclosure": (
            "R02 was evaluated by the earlier classifier pilot. It is not used in "
            "this convergence diagnostic, but is no longer pristine for a future "
            "publication-level adaptive analysis."
        ),
        "sampling_rate_hz": dataset.sampling_rate_hz,
        "channels": list(dataset.channel_names),
        "context_samples": base.CONTEXT_SAMPLES,
        "target_samples": base.TARGET_SAMPLES,
        "stride_samples": base.STRIDE_SAMPLES,
        "normal_guard_samples": base.NORMAL_GUARD_SAMPLES,
        "support": support_metadata,
        "duration_fraction_support": fraction_audit,
        "model": {
            "config": model.model_config(),
            "parameter_count": int(parameter_count(model)),
        },
        "training": {
            "seeds": list(seeds),
            "clean_normal_training_fractions": list(fractions),
            "fraction_selection": (
                "Nested, record-stratified 30-second window-start blocks target "
                "exact unique context+target duration. Full-train RobustScaler "
                "and validation stay fixed. Smaller arms sample their selected "
                "windows with replacement to match 978 exposures and four "
                "optimizer steps per epoch."
            ),
            "minimum_epochs": MIN_EPOCHS,
            "maximum_epochs": args.max_epochs,
            "patience": args.patience,
            "early_stop_metric": "validation clean-normal Gaussian NLL",
            "min_delta": MIN_DELTA,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "matched_window_exposures_per_epoch": int(
                support_metadata["support"]["clean_normal_train_windows"]
            ),
            "matched_optimizer_steps_per_epoch": int(
                math.ceil(
                    support_metadata["support"]["clean_normal_train_windows"]
                    / args.batch_size
                )
            ),
            "gradient_clip_norm": 5.0,
            "amp_training": args.amp,
            "float32_early_stop_evaluation": True,
            "deterministic": args.deterministic,
            "deterministic_algorithms_strict": args.deterministic,
        },
        "diagnostics": [
            "eval-mode train and validation NLL every epoch",
            "forecast RMSE and MAE",
            "mean sigma and separated NLL terms",
            "standardized-residual RMS and 1/1.96-sigma coverage",
            "sigma-bound saturation",
            "forecast-horizon quartile RMSE",
            "pre-clip gradient norm and clip frequency",
            "train-calibrated persistence baseline",
            "best-GRU mean with train-residual-calibrated fixed sigma",
            "multi-seed data-fraction learning curve",
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
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


def support_manifest_rows(
    dataset: DaphnetDataset,
    windows: WindowTable,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> list[dict[str, Any]]:
    membership = {
        int(index): split
        for split, indices in (("train", train_indices), ("validation", validation_indices))
        for index in indices
    }
    rows: list[dict[str, Any]] = []
    for index in sorted(membership):
        record = dataset.records[int(windows.record_index[index])]
        rows.append(
            {
                "split": membership[index],
                "window_index": index,
                "subject_id": record.subject_id,
                "record_id": record.record_id,
                "context_start": int(windows.start[index]),
                "target_start": int(windows.target_start[index]),
                "target_end_exclusive": int(windows.target_end[index]),
                "clean_normal": bool(windows.clean_normal[index]),
            }
        )
    if any(row["record_id"] == base.TEST_RECORD for row in rows):
        raise AssertionError("Test record entered support manifest")
    return rows


def write_report(
    output_dir: Path,
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    evidence: Mapping[str, Any],
    shift: Mapping[str, Any],
) -> None:
    run_lines = []
    for row in rows:
        run_lines.append(
            "| {fraction:g} | {seed} | {train_windows} | "
            "{unique_raw_support_seconds:.1f} | {epochs_completed} | "
            "{cumulative_optimizer_steps} | "
            "{stop_reason} | {best_epoch} | {epoch8_validation_nll:.6f} | "
            "{best_validation_nll:.6f} | {improvement_after_epoch8:.6f} | "
            "{best_validation_rmse:.6f} | {persistence_validation_nll:.6f} |".format(
                **row
            )
        )
    aggregate_lines = []
    for fraction in sorted(float(key) for key in aggregate):
        item = aggregate[f"{fraction:g}"]
        aggregate_lines.append(
            f"| {fraction:g} | {item['achieved_duration_fraction']:.3f} | "
            f"{item['unique_raw_support_seconds']:.1f} | {item['train_windows']} | "
            f"{item['best_epoch']['mean']:.2f}±{item['best_epoch']['std']:.2f} | "
            f"{item['best_validation_nll']['mean']:.6f}±{item['best_validation_nll']['std']:.6f} | "
            f"{item['improvement_after_epoch8']['mean']:.6f} | "
            f"{item['patience_stop_count']}/{item['runs']} |"
        )
    budget = evidence["eight_epoch_budget_insufficient"]
    convergence = evidence["convergence_within_40_epochs"]
    data = evidence["more_clean_normal_duration_help"]
    instability = evidence["optimization_instability"]
    sigma = evidence["sigma_inflation"]
    data_gain_text = (
        "not applicable"
        if data["mean_last_increment_nll_gain"] is None
        else f"{data['mean_last_increment_nll_gain']:.6f}"
    )
    report = f"""# S01 GRU-NBM convergence diagnostic

## Locked protocol

- Only S01_seg000 and S01_seg001 are present in the diagnostic dataset.
- The S01_seg002/R02 array is not opened, windowed, inferred, scored, or selected.
- Clean-normal support: {protocol['support']['support']['clean_normal_train_windows']}
  training windows and {protocol['support']['support']['clean_normal_validation_windows']}
  validation windows.
- Maximum epochs: {protocol['training']['maximum_epochs']}; patience:
  {protocol['training']['patience']}; early stopping: validation clean-normal NLL.
- Every arm receives {protocol['training']['matched_window_exposures_per_epoch']}
  window exposures and {protocol['training']['matched_optimizer_steps_per_epoch']}
  optimizer steps per epoch; smaller arms use train-only replacement sampling.
- Fractions: {protocol['training']['clean_normal_training_fractions']}; seeds:
  {protocol['training']['seeds']}.

## Per-run evidence

| Fraction | Seed | Unique windows | Raw seconds | Epochs | Steps | Stop | Best epoch | Epoch-8 val NLL | Best val NLL | Post-8 gain | Best val RMSE | Persistence val NLL |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(run_lines)}

## Data-fraction aggregate

| Requested fraction | Achieved duration | Raw seconds | Unique windows | Best epoch mean±SD | Best val NLL mean±SD | Mean post-8 gain | Patience stops |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(aggregate_lines)}

## Evidence assessment

- Eight-epoch budget insufficient: **{budget['supported']}**. Full-data seeds
  best after epoch 8: {budget['full_data_best_epochs_after_8_count']}/
  {len(protocol['training']['seeds'])}; median best epoch:
  {budget['median_full_data_best_epoch']:.1f}; median post-8 validation-NLL gain:
  {budget['median_full_data_validation_nll_improvement_after_epoch8']:.6f}.
- Reached the validation-NLL patience rule within
  {protocol['training']['maximum_epochs']} epochs: full-data runs
  {convergence['full_data_patience_stop_count']}/{len(protocol['training']['seeds'])};
  maximum-epoch stops {convergence['full_data_maximum_epoch_stop_count']}.
- Conditional duration-path flag: **{data['supported']}**; nearest-to-full paired
  mean validation-NLL gain {data_gain_text}. All model seeds share one fixed
  nested block path and the full-training scaler, so this is not a causal data-
  amount result and does not prove that duration beyond 100% is required.
- Sigma inflation explanation supported: **{sigma['supported']}**; mean best/epoch-8
  sigma ratio {sigma['mean_best_to_epoch8_sigma_ratio']:.4f}, while RMSE relative
  improvement is {sigma['mean_epoch8_to_best_rmse_relative_improvement']:.4f};
  best z-RMS {sigma['mean_best_validation_z_rms']:.4f}, 95% coverage
  {sigma['mean_best_validation_coverage_95']:.4f}.
- Coarse numerical-instability flag: **{instability['supported']}**; full-data
  seed SD {instability['full_data_seed_sd_best_validation_nll']:.6f}, gradient
  clip-step fraction {instability['full_data_mean_gradient_clip_step_fraction']:.4f}.
  A negative flag only excludes the screened failure modes; it is not proof of
  global optimization stability.
- Train/validation unique-point distribution shift: mean absolute standardized
  mean difference {shift['mean_absolute_standardized_mean_difference']:.4f}, maximum
  {shift['maximum_absolute_standardized_mean_difference']:.4f}.

These flags are transparent diagnostic rules, not formal hypothesis tests. The
raw epoch histories and checkpoints remain the authoritative evidence.

## Test-set boundary

This diagnostic did not evaluate R02. The earlier classifier pilot already did,
so R02 is not pristine for future publication-level adaptive model selection;
that limitation is recorded rather than hidden.
"""
    path = output_dir / "report.md"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    seeds = parse_int_list(args.seeds)
    fractions = parse_fraction_list(args.fractions)
    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        dataset,
        windows,
        normal_train,
        normal_validation,
        scaler,
        support_metadata,
    ) = prepare_support(args.data_dir)
    fraction_subsets, fraction_audit = duration_fraction_subsets(
        dataset, windows, normal_train, fractions
    )
    protocol = protocol_payload(
        args,
        seeds,
        fractions,
        dataset,
        support_metadata,
        fraction_audit,
        device,
    )
    atomic_json_dump(protocol, output_dir / "config.json")
    atomic_json_dump(support_metadata["scaler"], output_dir / "scaler.json")
    atomic_json_dump(fraction_audit, output_dir / "duration_fraction_support.json")
    atomic_npz_save(
        output_dir / "locked_support.npz",
        clean_normal_train_window_index=normal_train,
        clean_normal_validation_window_index=normal_validation,
    )
    write_csv(
        output_dir / "locked_support.csv",
        support_manifest_rows(
            dataset, windows, normal_train, normal_validation
        ),
    )
    shift = clean_normal_point_shift(dataset, scaler)
    atomic_json_dump(shift, output_dir / "train_validation_shift.json")
    print(
        f"Protocol {protocol['protocol_fingerprint']} device={device} "
        f"train/val={len(normal_train)}/{len(normal_validation)}; "
        f"test_record={base.TEST_RECORD} excluded",
        flush=True,
    )

    summaries: list[dict[str, Any]] = []
    for fraction in fractions:
        selected_train = fraction_subsets[float(fraction)]
        for seed in seeds:
            summaries.append(
                train_one_run(
                    args,
                    dataset,
                    windows,
                    selected_train,
                    normal_validation,
                    scaler,
                    fraction,
                    fraction_audit["fractions"][f"{fraction:g}"],
                    len(normal_train),
                    seed,
                    protocol["protocol_fingerprint"],
                    output_dir,
                    device,
                )
            )

    rows, aggregate = aggregate_runs(summaries, fractions)
    evidence = evidence_assessment(rows, aggregate)
    enrich_sigma_evidence(evidence, summaries)
    atomic_json_dump(aggregate, output_dir / "aggregate.json")
    atomic_json_dump(evidence, output_dir / "evidence.json")
    write_csv(output_dir / "run_table.csv", rows)
    write_report(output_dir, protocol, rows, aggregate, evidence, shift)
    artifacts = {
        str(path.relative_to(output_dir)).replace("\\", "/"): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "DONE.json"
    }
    atomic_json_dump(
        {
            "status": "complete",
            "completed_utc": utc_now(),
            "experiment_version": EXPERIMENT_VERSION,
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "run_count": len(summaries),
            "test_record_evaluated": False,
            "artifacts": artifacts,
        },
        output_dir / "DONE.json",
    )
    budget = evidence["eight_epoch_budget_insufficient"]
    convergence = evidence["convergence_within_40_epochs"]
    print(
        "COMPLETE "
        f"eight_epoch_budget_insufficient={budget['supported']} "
        f"full_data_median_best_epoch={budget['median_full_data_best_epoch']:.1f} "
        f"full_data_patience_stops={convergence['full_data_patience_stop_count']}/"
        f"{len(seeds)} test_evaluated=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
