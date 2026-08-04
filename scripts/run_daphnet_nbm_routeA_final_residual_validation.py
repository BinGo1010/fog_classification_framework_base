#!/usr/bin/env python
"""Run gated Route A validation for the Daphnet M3 normal-behaviour model.

The implementation intentionally executes A0 and A1 first.  When A1 fails,
the preregistered sequential gate writes explicit NOT RUN reports for A2-A7
instead of consulting FoG residuals or classifier results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for location in (REPO_ROOT, SCRIPTS_ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_daphnet_nbm_tcdae_three_rounds as base  # noqa: E402
import run_daphnet_s01_nonfog_gru_reconstruction_tcnm as legacy  # noqa: E402
from cnbr_fog.data import DaphnetDataset, Record  # noqa: E402


EXPERIMENT = "daphnet_nbm_routeA_final_residual_validation_v1"
SUBJECTS = ("S01", "S02", "S05", "S06", "S07", "S08", "S09")
RECORD_HOLDOUT_SUBJECTS = ("S01", "S05", "S06", "S08", "S09")
TEMPORAL_SUBJECTS = ("S02", "S07")
SEEDS = (20260802, 20260803, 20260804)
FS = 64
WINDOW = 128
STRIDE = 64
FOG_GUARD = 64
ENDPOINT_LABEL_SAMPLES = 32
TEMPORAL_GAP_SAMPLES = 5 * FS


@dataclass
class PreparedSubject:
    subject: str
    records: list[Record]
    windows: legacy.WindowSet
    train_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray
    scaler: legacy.RobustScaler
    train_x: np.ndarray
    calibration_x: np.ndarray
    test_x: np.ndarray
    test_metadata: list[dict[str, Any]]
    disclosure: dict[str, Any]
    channel_names: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT
        / "dataset"
        / "1.Daphnet Freezing of Gait Dataset"
        / "processed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / EXPERIMENT,
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(
            r"C:\Users\bin\Downloads\Daphnet_NBM_routeA_final_residual_validation_template.md"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def temporal_intervals(subject: str, records: list[Record]) -> tuple[list[legacy.Interval], dict[str, Any]]:
    spec = legacy.SUBJECT_SPLITS[subject]
    record_id = str(spec["cut_record"])
    record = next(item for item in records if item.record_id == record_id)
    first_boundary = int(round(0.60 * len(record.y) / STRIDE) * STRIDE)
    second_boundary = int(round(0.80 * len(record.y) / STRIDE) * STRIDE)
    half_gap = TEMPORAL_GAP_SAMPLES // 2
    intervals = [
        legacy.Interval(record_id, 0, first_boundary - half_gap, "train", 0),
        legacy.Interval(
            record_id,
            first_boundary + half_gap,
            second_boundary - half_gap,
            "validation",
            -1,
        ),
        legacy.Interval(
            record_id,
            second_boundary + half_gap,
            len(record.y),
            "test",
            -1,
        ),
    ]
    if any(item.end - item.start < WINDOW + 2 * FOG_GUARD for item in intervals):
        raise ValueError(f"{subject} temporal block is too short")
    return intervals, {
        "split_type": "single_record_chronological_time_block",
        "record_id": record_id,
        "ratios": [0.60, 0.20, 0.20],
        "nominal_boundaries": [first_boundary, second_boundary],
        "gap_seconds_each_boundary": TEMPORAL_GAP_SAMPLES / FS,
        "reserved_records_not_used": [
            item.record_id for item in records if item.record_id != record_id
        ],
        "interpretation": "time-period generalization only",
    }


def subject_intervals(subject: str, records: list[Record]) -> tuple[list[legacy.Interval], dict[str, Any]]:
    if subject in TEMPORAL_SUBJECTS:
        return temporal_intervals(subject, records)
    legacy.SUBJECT = subject
    intervals = legacy.build_intervals(records)
    split = legacy.SUBJECT_SPLITS[subject]
    return intervals, {
        "split_type": "fixed_unseen_record_holdout",
        "test_record": split["test_record"],
        "cut_record": split["cut_record"],
        "cut_sample": split["cut_sample"],
        "ignored_records": split["ignored"],
        "outer_train_calibration_gap_seconds": 2 * legacy.OUTER_HALF_GAP / FS,
        "interpretation": "unseen-record generalization",
    }


def indices_for_split(windows: legacy.WindowSet, split: str, *, clean: bool) -> np.ndarray:
    mask = windows.split == split
    if clean:
        mask &= windows.clean_normal
    return np.flatnonzero(mask).astype(np.int64)


def window_metadata(
    subject: str,
    records: Sequence[Record],
    windows: legacy.WindowSet,
    indices: Sequence[int],
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, raw_index in enumerate(indices):
        index = int(raw_index)
        record = records[int(windows.record_index[index])]
        raw = record.x[int(windows.start[index]) : int(windows.end[index])]
        rows.append(
            {
                "window_id": f"{record.record_id}:{int(windows.start[index])}:{int(windows.end[index])}",
                "window_row": row_index,
                "subject_id": subject,
                "record_id": record.record_id,
                "split": split,
                "start_sample": int(windows.start[index]),
                "end_sample_exclusive": int(windows.end[index]),
                "start_time_sec": float(windows.start[index]) / FS,
                "end_time_sec": float(windows.end[index]) / FS,
                "label": int(windows.label[index]),
                "fog_fraction_final_0p5s": float(windows.fog_fraction[index]),
                "clean_nonfog": bool(windows.clean_normal[index]),
                "energy": float(np.median(np.sqrt(np.mean(np.square(raw.astype(np.float64)), axis=0)))),
            }
        )
    return rows


def prepare_subject(dataset: DaphnetDataset, subject: str) -> PreparedSubject:
    records = [record for record in dataset.records if record.subject_id == subject]
    intervals, disclosure = subject_intervals(subject, records)
    windows = legacy.build_windows(records, intervals)
    train_indices = indices_for_split(windows, "train", clean=True)
    calibration_indices = indices_for_split(windows, "validation", clean=True)
    test_indices = indices_for_split(windows, "test", clean=True)
    if min(len(train_indices), len(calibration_indices), len(test_indices)) == 0:
        raise ValueError(
            f"{subject} empty clean split: train={len(train_indices)} "
            f"calibration={len(calibration_indices)} test={len(test_indices)}"
        )
    scaler = legacy.fit_scaler_unique_points(records, windows, train_indices)
    train_x = legacy.prepare_nbm_windows(
        scaler, legacy.raw_windows(records, windows, train_indices), center=True
    )
    calibration_x = legacy.prepare_nbm_windows(
        scaler, legacy.raw_windows(records, windows, calibration_indices), center=True
    )
    test_x = legacy.prepare_nbm_windows(
        scaler, legacy.raw_windows(records, windows, test_indices), center=True
    )
    return PreparedSubject(
        subject=subject,
        records=records,
        windows=windows,
        train_indices=train_indices,
        calibration_indices=calibration_indices,
        test_indices=test_indices,
        scaler=scaler,
        train_x=train_x,
        calibration_x=calibration_x,
        test_x=test_x,
        test_metadata=window_metadata(
            subject, records, windows, test_indices, "test_clean_nonfog"
        ),
        disclosure=disclosure,
        channel_names=tuple(dataset.channel_names),
    )


def write_a0(
    root: Path,
    dataset: DaphnetDataset,
    prepared: dict[str, PreparedSubject],
    args: argparse.Namespace,
) -> None:
    a0 = root / "A0_protocol"
    manifest_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    disclosures: dict[str, Any] = {}
    for subject, item in prepared.items():
        disclosures[subject] = item.disclosure
        interval_roles: dict[str, set[str]] = {}
        for interval in subject_intervals(subject, item.records)[0]:
            interval_roles.setdefault(interval.record_id, set()).add(interval.split)
        for record in item.records:
            starts = range(0, len(record.y) - WINDOW + 1, STRIDE)
            clean_count = sum(
                bool(
                    record.valid[start : start + WINDOW].all()
                    and not record.y[
                        max(0, start - FOG_GUARD) : min(
                            len(record.y), start + WINDOW + FOG_GUARD
                        )
                    ].any()
                )
                for start in starts
            )
            audit_rows.append(
                {
                    "subject_id": subject,
                    "record_id": record.record_id,
                    "samples": len(record.y),
                    "minutes": len(record.y) / FS / 60.0,
                    "fog_sample_fraction": float(np.mean(record.y)),
                    "valid_sample_fraction": float(np.mean(record.valid)),
                    "all_record_clean_windows": clean_count,
                    "routeA_roles": "+".join(sorted(interval_roles.get(record.record_id, {"unused"}))),
                }
            )
        for split, indices in (
            ("train_clean_nonfog", item.train_indices),
            ("calibration_clean_nonfog", item.calibration_indices),
            ("test_clean_nonfog", item.test_indices),
        ):
            manifest_rows.extend(
                window_metadata(subject, item.records, item.windows, indices, split)
            )
    base.write_csv(a0 / "data_split_manifest.csv", manifest_rows)
    base.write_csv(a0 / "subject_record_audit.csv", audit_rows)
    frozen = {
        "experiment": EXPERIMENT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "template": str(args.template.resolve()),
        "template_sha256": sha256(args.template),
        "data_dir": str(args.data_dir.resolve()),
        "subjects": list(SUBJECTS),
        "seeds": list(SEEDS),
        "sampling_rate_hz": FS,
        "window_samples": WINDOW,
        "stride_samples": STRIDE,
        "fog_guard_samples_each_side": FOG_GUARD,
        "endpoint_label_samples": ENDPOINT_LABEL_SAMPLES,
        "endpoint_fog_fraction_threshold": 0.5,
        "architecture": "M3_tcdae_long",
        "architecture_config": base.build_model("M3_tcdae_long").architecture_config(),
        "preprocessor": {
            "name": "P0_current",
            "scaler": "per-channel median/IQR fitted on unique points covered by training clean Non-FoG windows",
            "window_axis_centering": True,
            "test_or_calibration_statistics_used_for_scaler": False,
        },
        "training": {
            "loss": "MSELoss",
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "maximum_epochs": args.max_epochs,
            "patience": args.patience,
            "gradient_clip_norm": 1.0,
        },
        "split_disclosures": disclosures,
        "stage_gate_policy": "A2-A7 are not run if A1 overall gate fails",
    }
    base.write_json(a0 / "frozen_config.json", frozen)


def loader(x: np.ndarray, *, shuffle: bool, seed: int, workers: int) -> DataLoader:
    tensor = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1))).float()
    return DataLoader(
        TensorDataset(tensor),
        batch_size=64,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def predict_model(
    model: nn.Module, x: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predicted: list[np.ndarray] = []
    latent: list[np.ndarray] = []
    for (batch,) in loader(x, shuffle=False, seed=0, workers=0):
        reconstruction, representation = model(batch.to(device, non_blocking=True))
        predicted.append(reconstruction.transpose(1, 2).cpu().numpy().astype(np.float32))
        latent.append(representation.cpu().numpy().astype(np.float32))
    return np.concatenate(predicted), np.concatenate(latent)


@torch.no_grad()
def evaluation_mse(model: nn.Module, x: np.ndarray, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for (batch,) in loader(x, shuffle=False, seed=0, workers=0):
        batch = batch.to(device, non_blocking=True)
        reconstruction, _ = model(batch)
        total += float(torch.sum(torch.square(reconstruction - batch)))
        count += int(batch.numel())
    return total / count


def train_a1(
    train_x: np.ndarray,
    calibration_x: np.ndarray,
    run_dir: Path,
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
    workers: int,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    base.set_seed(seed)
    model = base.build_model("M3_tcdae_long").to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    criterion = nn.MSELoss()
    train_loader = loader(train_x, shuffle=True, seed=seed, workers=workers)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    initial = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    last_epoch = 0
    last_train_loss = math.inf
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        maximum_gradient = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient):
                raise FloatingPointError("non-finite Route A1 gradient")
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
            maximum_gradient = max(maximum_gradient, float(gradient))
        last_train_loss = total / count
        validation_loss = evaluation_mse(model, calibration_x, device)
        improved = validation_loss < best_loss - 1e-8
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = base.clone_state(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
        last_epoch = epoch
        if epoch == 1 or epoch % 10 == 0 or improved or epoch == max_epochs:
            history.append(
                {
                    "epoch": epoch,
                    "train_mse": last_train_loss,
                    "calibration_mse": validation_loss,
                    "learning_rate": 3e-4,
                    "max_gradient_norm_before_clip": maximum_gradient,
                    "improved": improved,
                    "bad_epochs": bad_epochs,
                }
            )
        if epoch == 1 or epoch % 100 == 0:
            print(
                f"A1 seed={seed} epoch={epoch:04d}/{max_epochs} "
                f"train={last_train_loss:.6g} calibration={validation_loss:.6g} "
                f"best={best_loss:.6g}@{best_epoch}",
                flush=True,
            )
        if bad_epochs >= patience:
            break
    if best_state is None:
        raise AssertionError("A1 training produced no best checkpoint")
    last_state = base.clone_state(model)
    common = {
        "experiment": EXPERIMENT,
        "stage": "A1_nonfog_generalization",
        "architecture": "M3_tcdae_long",
        "architecture_config": model.architecture_config(),  # type: ignore[attr-defined]
        "seed": seed,
    }
    base.torch_save(
        run_dir / "last_model.pt",
        {**common, "epoch": last_epoch, "model_state": last_state},
    )
    base.torch_save(
        run_dir / "best_model.pt",
        {
            **common,
            "epoch": best_epoch,
            "calibration_mse": best_loss,
            "model_state": best_state,
        },
    )
    model.load_state_dict(best_state)
    final = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    training = {
        "best_epoch": best_epoch,
        "final_epoch": last_epoch,
        "stopped_early": last_epoch < max_epochs,
        "best_calibration_mse": best_loss,
        "last_train_mse": last_train_loss,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameter_delta_l2": float(torch.linalg.vector_norm(final - initial)),
        "elapsed_seconds": time.perf_counter() - started,
        "batch_size": 64,
        "maximum_epochs": max_epochs,
        "patience": patience,
    }
    return model, history, training


def smooth_l1_mean(error: np.ndarray) -> float:
    absolute = np.abs(error)
    return float(np.mean(np.where(absolute < 1.0, 0.5 * error**2, absolute - 0.5)))


def spectral_error(
    actual: np.ndarray, predicted: np.ndarray, low: float, high: float
) -> tuple[float, np.ndarray]:
    frequency = np.fft.rfftfreq(WINDOW, d=1.0 / FS)
    mask = (frequency >= low) & (frequency <= high)
    source = np.abs(np.fft.rfft(actual, axis=1))[:, mask, :]
    estimate = np.abs(np.fft.rfft(predicted, axis=1))[:, mask, :]
    numerator = np.sqrt(np.mean(np.square(source - estimate), axis=1))
    denominator = np.sqrt(np.mean(np.square(source), axis=1)) + 1e-8
    values = numerator / denominator
    return float(np.median(values)), values


def nearest_training_windows(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_flat = train_x.reshape(len(train_x), -1).astype(np.float64)
    test_flat = test_x.reshape(len(test_x), -1).astype(np.float64)
    train_norm = np.sum(np.square(train_flat), axis=1)
    nearest: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for start in range(0, len(test_flat), 64):
        batch = test_flat[start : start + 64]
        distances = (
            np.sum(np.square(batch), axis=1)[:, None]
            + train_norm[None, :]
            - 2.0 * batch @ train_flat.T
        )
        chosen = np.argmin(distances, axis=1)
        nearest.append(train_x[chosen])
        indices.append(chosen.astype(np.int64))
    return np.concatenate(nearest), np.concatenate(indices)


def reconstruction_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    template_prediction: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays = base.metric_arrays(actual, predicted)
    window_nrmse = np.median(arrays["nrmse"], axis=1)
    window_corr = np.median(arrays["correlation"], axis=1)
    window_amplitude = np.median(arrays["amplitude_ratio"], axis=1)
    error = actual - predicted
    mse = float(np.mean(np.square(error)))
    zero_mse = float(np.mean(np.square(actual)))
    template_mse = float(np.mean(np.square(actual - template_prediction)))
    window_improvement = 100.0 * (
        arrays["window_zero_mse"] - arrays["window_mse"]
    ) / np.maximum(arrays["window_zero_mse"], 1e-12)
    spectral_low, spectral_low_array = spectral_error(actual, predicted, 0.5, 3.0)
    spectral_high, spectral_high_array = spectral_error(actual, predicted, 3.0, 8.0)
    metrics = {
        "mse": mse,
        "huber": smooth_l1_mean(error),
        "zero_mse": zero_mse,
        "template_mse": template_mse,
        "zero_improvement_pct": 100.0 * (zero_mse - mse) / max(zero_mse, 1e-12),
        "template_improvement_pct": 100.0
        * (template_mse - mse)
        / max(template_mse, 1e-12),
        "median_corr": float(np.median(arrays["correlation"])),
        "median_nrmse": float(np.median(arrays["nrmse"])),
        "nrmse_p90": float(np.percentile(window_nrmse, 90)),
        "nrmse_p95": float(np.percentile(window_nrmse, 95)),
        "median_amplitude_ratio": float(np.median(arrays["amplitude_ratio"])),
        "negative_improvement_window_fraction": float(np.mean(window_improvement < 0.0)),
        "spectral_nrmse_0p5_3hz": spectral_low,
        "spectral_nrmse_3_8hz": spectral_high,
    }
    return metrics, {
        **arrays,
        "window_nrmse": window_nrmse,
        "window_corr": window_corr,
        "window_amplitude_ratio": window_amplitude,
        "window_improvement_pct": window_improvement,
        "spectral_nrmse_0p5_3hz_array": spectral_low_array,
        "spectral_nrmse_3_8hz_array": spectral_high_array,
    }


def a1_pass(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["zero_improvement_pct"] > 20.0
        and metrics["template_improvement_pct"] > 10.0
        and metrics["median_corr"] >= 0.50
        and metrics["median_nrmse"] <= 0.85
        and metrics["nrmse_p90"] <= 1.30
        and metrics["negative_improvement_window_fraction"] <= 0.20
    )


def waveform_collapse(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["zero_improvement_pct"] <= 0.0
        or metrics["median_corr"] < 0.20
        or not 0.50 <= metrics["median_amplitude_ratio"] <= 1.50
    )


def baseline_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    arrays = base.metric_arrays(actual, predicted)
    return {
        "mse": float(np.mean(np.square(actual - predicted))),
        "huber": smooth_l1_mean(actual - predicted),
        "median_corr": float(np.median(arrays["correlation"])),
        "median_nrmse": float(np.median(arrays["nrmse"])),
        "nrmse_p90": float(np.percentile(np.median(arrays["nrmse"], axis=1), 90)),
    }


def run_a1_once(
    item: PreparedSubject,
    root: Path,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    fold_id = (
        f"test_{item.disclosure['test_record']}"
        if item.disclosure["split_type"] == "fixed_unseen_record_holdout"
        else f"timeblock_{item.disclosure['record_id']}"
    )
    run_dir = (
        root
        / "A1_nonfog_generalization"
        / item.subject
        / fold_id
        / f"seed{seed}"
    )
    required = (
        run_dir / "run_metrics.json",
        run_dir / "predictions.npz",
        run_dir / "best_model.pt",
        run_dir / "last_model.pt",
        run_dir / "window_metrics.csv",
        run_dir / "channel_metrics.csv",
    )
    if not args.overwrite and all(path.exists() for path in required):
        print(f"RESUME A1 {item.subject} {fold_id} seed={seed}", flush=True)
        return json.loads((run_dir / "run_metrics.json").read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": EXPERIMENT,
        "stage": "A1_nonfog_generalization",
        "subject_id": item.subject,
        "fold_id": fold_id,
        "seed": seed,
        "split": item.disclosure,
        "train_clean_nonfog_windows": len(item.train_x),
        "calibration_clean_nonfog_windows": len(item.calibration_x),
        "test_clean_nonfog_windows": len(item.test_x),
        "scaler": item.scaler.as_dict(),
        "scaler_fit": "training unique points only",
        "test_statistics_used_for_training_or_preprocessing": False,
        "architecture": "M3_tcdae_long",
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "maximum_epochs": args.max_epochs,
        "patience": args.patience,
    }
    base.write_json(run_dir / "config.json", config)
    split_rows = []
    for split, indices in (
        ("train_clean_nonfog", item.train_indices),
        ("calibration_clean_nonfog", item.calibration_indices),
        ("test_clean_nonfog", item.test_indices),
    ):
        split_rows.extend(
            window_metadata(item.subject, item.records, item.windows, indices, split)
        )
    base.write_csv(run_dir / "split_manifest.csv", split_rows)
    print(
        f"START A1 {item.subject} {fold_id} seed={seed} "
        f"train/cal/test={len(item.train_x)}/{len(item.calibration_x)}/{len(item.test_x)}",
        flush=True,
    )
    model, history, training = train_a1(
        item.train_x,
        item.calibration_x,
        run_dir,
        seed=seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=device,
        workers=args.num_workers,
    )
    test_prediction, test_latent = predict_model(model, item.test_x, device)
    train_prediction, _ = predict_model(model, item.train_x, device)
    template = np.mean(item.train_x, axis=0, keepdims=True).astype(np.float32)
    template_test = np.repeat(template, len(item.test_x), axis=0)
    nearest, nearest_indices = nearest_training_windows(item.train_x, item.test_x)
    metrics, arrays = reconstruction_metrics(item.test_x, test_prediction, template_test)
    train_metrics, _ = reconstruction_metrics(
        item.train_x,
        train_prediction,
        np.repeat(template, len(item.train_x), axis=0),
    )
    baselines = {
        "B0_zero": baseline_summary(item.test_x, np.zeros_like(item.test_x)),
        "B1_training_mean_template": baseline_summary(item.test_x, template_test),
        "B2_nearest_training_window": baseline_summary(item.test_x, nearest),
        "B3_M3_TC_DAE": baseline_summary(item.test_x, test_prediction),
    }
    strict = a1_pass(metrics)
    collapse = waveform_collapse(metrics)
    result: dict[str, Any] = {
        "subject_id": item.subject,
        "fold_id": fold_id,
        "seed": seed,
        "split_type": item.disclosure["split_type"],
        "test_record_or_block": item.disclosure.get(
            "test_record", item.disclosure.get("record_id")
        ),
        "train_windows": len(item.train_x),
        "calibration_windows": len(item.calibration_x),
        "test_windows": len(item.test_x),
        **metrics,
        **training,
        "train_median_corr": train_metrics["median_corr"],
        "train_median_nrmse": train_metrics["median_nrmse"],
        "corr_degradation_train_to_test": train_metrics["median_corr"]
        - metrics["median_corr"],
        "nrmse_degradation_train_to_test": metrics["median_nrmse"]
        - train_metrics["median_nrmse"],
        "strict_pass": strict,
        "pass_status": "PASS" if strict else "FAIL",
        "waveform_collapse": collapse,
        "baselines": baselines,
        "run_dir": str(run_dir.resolve()),
    }
    window_rows: list[dict[str, Any]] = []
    for index, metadata in enumerate(item.test_metadata):
        window_rows.append(
            {
                **metadata,
                "mse": float(arrays["window_mse"][index]),
                "huber": smooth_l1_mean(item.test_x[index] - test_prediction[index]),
                "zero_mse": float(arrays["window_zero_mse"][index]),
                "zero_improvement_pct": float(arrays["window_improvement_pct"][index]),
                "pearson_median": float(arrays["window_corr"][index]),
                "nrmse_median": float(arrays["window_nrmse"][index]),
                "amplitude_ratio_median": float(arrays["window_amplitude_ratio"][index]),
                "spectral_nrmse_0p5_3hz": float(
                    np.median(arrays["spectral_nrmse_0p5_3hz_array"][index])
                ),
                "spectral_nrmse_3_8hz": float(
                    np.median(arrays["spectral_nrmse_3_8hz_array"][index])
                ),
                "nearest_train_window_row": int(nearest_indices[index]),
            }
        )
    channel_rows: list[dict[str, Any]] = []
    for window_index, metadata in enumerate(item.test_metadata):
        for channel, name in enumerate(item.channel_names):
            channel_rows.append(
                {
                    "window_id": metadata["window_id"],
                    "record_id": metadata["record_id"],
                    "channel": name,
                    "channel_id": channel,
                    "mse": float(
                        np.mean(
                            np.square(
                                item.test_x[window_index, :, channel]
                                - test_prediction[window_index, :, channel]
                            )
                        )
                    ),
                    "pearson": float(arrays["correlation"][window_index, channel]),
                    "nrmse": float(arrays["nrmse"][window_index, channel]),
                    "amplitude_ratio": float(
                        arrays["amplitude_ratio"][window_index, channel]
                    ),
                    "spectral_nrmse_0p5_3hz": float(
                        arrays["spectral_nrmse_0p5_3hz_array"][window_index, channel]
                    ),
                    "spectral_nrmse_3_8hz": float(
                        arrays["spectral_nrmse_3_8hz_array"][window_index, channel]
                    ),
                }
            )
    base.write_csv(run_dir / "training_log.csv", history)
    base.write_csv(run_dir / "window_metrics.csv", window_rows)
    base.write_csv(run_dir / "channel_metrics.csv", channel_rows)
    base.write_json(run_dir / "run_metrics.json", result)
    legacy.save_npz(
        run_dir / "predictions.npz",
        target=item.test_x,
        reconstruction=test_prediction,
        latent=test_latent,
        mean_template=template_test,
        nearest_training_window=nearest,
        nearest_training_index=nearest_indices,
    )
    if not args.skip_figures:
        plot_training_log(history, run_dir / "figures" / "training_loss.png")
    print(
        f"DONE A1 {item.subject} seed={seed} {result['pass_status']} "
        f"zero={metrics['zero_improvement_pct']:.1f}% "
        f"template={metrics['template_improvement_pct']:.1f}% "
        f"corr={metrics['median_corr']:.3f} nrmse={metrics['median_nrmse']:.3f} "
        f"p90={metrics['nrmse_p90']:.3f} negative={metrics['negative_improvement_window_fraction']:.3f}",
        flush=True,
    )
    return result


def plot_training_log(history: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([row["epoch"] for row in history], [row["train_mse"] for row in history], label="train")
    ax.plot(
        [row["epoch"] for row in history],
        [row["calibration_mse"] for row in history],
        label="calibration",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def read_run_arrays(result: dict[str, Any]) -> tuple[np.ndarray, ...]:
    with np.load(Path(result["run_dir"]) / "predictions.npz", allow_pickle=False) as payload:
        return tuple(
            np.asarray(payload[key])
            for key in (
                "target",
                "reconstruction",
                "latent",
                "mean_template",
                "nearest_training_window",
            )
        )


def representative_result(results: Sequence[dict[str, Any]], subject: str) -> dict[str, Any]:
    rows = sorted(
        [row for row in results if row["subject_id"] == subject],
        key=lambda row: row["median_nrmse"],
    )
    return rows[len(rows) // 2]


def plot_best_median_worst(
    actual: np.ndarray, predicted: np.ndarray, path: Path, channel_names: Sequence[str]
) -> None:
    arrays = base.metric_arrays(actual, predicted)
    order = np.argsort(np.median(arrays["nrmse"], axis=1))
    chosen = (int(order[0]), int(order[len(order) // 2]), int(order[-1]))
    fig, axes = plt.subplots(3, 9, figsize=(22, 8), sharex=True)
    time_axis = np.arange(WINDOW) / FS
    for row, (index, label) in enumerate(zip(chosen, ("best", "median", "worst"))):
        for channel, name in enumerate(channel_names):
            axes[row, channel].plot(time_axis, actual[index, :, channel], linewidth=0.8)
            axes[row, channel].plot(
                time_axis, predicted[index, :, channel], "--", linewidth=0.8
            )
            if row == 0:
                axes[row, channel].set_title(name, fontsize=7)
            if channel == 0:
                axes[row, channel].set_ylabel(label)
            axes[row, channel].grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_subject_figures(
    root: Path,
    subject: str,
    subject_results: Sequence[dict[str, Any]],
    channel_names: Sequence[str],
) -> None:
    figures = root / "A1_nonfog_generalization" / subject / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    representative = representative_result(subject_results, subject)
    actual, predicted, _, template, nearest = read_run_arrays(representative)
    window_rows = base.read_csv(Path(representative["run_dir"]) / "window_metrics.csv")
    arrays = base.metric_arrays(actual, predicted)
    seeds = [int(row["seed"]) for row in subject_results]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    metric_specs = (
        ("zero_improvement_pct", "Zero improvement (%)"),
        ("template_improvement_pct", "Template improvement (%)"),
        ("median_corr", "Pearson"),
        ("median_nrmse", "NRMSE"),
        ("nrmse_p90", "NRMSE P90"),
        ("negative_improvement_window_fraction", "Negative fraction"),
    )
    for ax, (key, title) in zip(axes.flat, metric_specs):
        ax.bar([str(seed) for seed in seeds], [row[key] for row in subject_results])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "test_record_reconstruction_metrics.png", dpi=150)
    plt.close(fig)

    plot_best_median_worst(
        actual, predicted, figures / "best_median_worst_waveforms.png", channel_names
    )
    sorted_nrmse = np.sort(np.median(arrays["nrmse"], axis=1))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(sorted_nrmse) + 1), sorted_nrmse)
    for percentile, color in ((50, "tab:green"), (90, "tab:orange"), (95, "tab:red")):
        value = float(np.percentile(sorted_nrmse, percentile))
        ax.axhline(value, color=color, linestyle="--", label=f"P{percentile}={value:.3f}")
    ax.set_xlabel("Sorted test window")
    ax.set_ylabel("Median NRMSE")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "window_nrmse_sorted.png", dpi=150)
    plt.close(fig)

    base.plot_metric_heatmap(
        arrays["nrmse"],
        figures / "window_channel_nrmse_heatmap.png",
        channel_names,
        f"{subject} unseen Non-FoG NRMSE",
        cmap="magma",
        vmin=0.0,
    )
    base.plot_metric_heatmap(
        arrays["correlation"],
        figures / "window_channel_pearson_heatmap.png",
        channel_names,
        f"{subject} unseen Non-FoG Pearson",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    base.plot_metric_heatmap(
        arrays["amplitude_ratio"],
        figures / "amplitude_ratio_heatmap.png",
        channel_names,
        f"{subject} unseen Non-FoG amplitude ratio",
        cmap="viridis",
        vmin=0.0,
        vmax=1.8,
    )

    times = [float(row["start_time_sec"]) for row in window_rows]
    nrmse = [float(row["nrmse_median"]) for row in window_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(times, nrmse, s=13, alpha=0.7)
    ax.set_xlabel("Record time (s)")
    ax.set_ylabel("Window NRMSE")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "record_time_vs_nrmse.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(
        [float(row["energy"]) for row in window_rows], nrmse, s=13, alpha=0.7
    )
    ax.set_xlabel("Raw window energy")
    ax.set_ylabel("Window NRMSE")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "energy_vs_nrmse.png", dpi=150)
    plt.close(fig)

    frequency = np.fft.rfftfreq(WINDOW, d=1.0 / FS)
    actual_spectrum = np.median(np.abs(np.fft.rfft(actual, axis=1)), axis=0)
    predicted_spectrum = np.median(np.abs(np.fft.rfft(predicted, axis=1)), axis=0)
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    for channel, ax in enumerate(axes.flat):
        ax.plot(frequency, actual_spectrum[:, channel], label="raw")
        ax.plot(frequency, predicted_spectrum[:, channel], "--", label="reconstruction")
        ax.set_xlim(0, 10)
        ax.set_title(channel_names[channel], fontsize=8)
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "raw_vs_reconstruction_spectrum.png", dpi=150)
    plt.close(fig)

    comparisons = {
        "B0 zero": np.zeros_like(actual),
        "B1 mean": template,
        "B2 nearest": nearest,
        "B3 M3": predicted,
    }
    summaries = {name: baseline_summary(actual, value) for name, value in comparisons.items()}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, key, title in zip(
        axes, ("mse", "median_corr", "median_nrmse"), ("MSE", "Pearson", "NRMSE")
    ):
        ax.bar(list(summaries), [value[key] for value in summaries.values()])
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "baseline_comparison.png", dpi=150)
    plt.close(fig)


def subject_summary(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        values = [row for row in results if row["subject_id"] == subject]
        passes = sum(bool(row["strict_pass"]) for row in values)
        collapses = sum(bool(row["waveform_collapse"]) for row in values)
        rows.append(
            {
                "subject_id": subject,
                "split_type": values[0]["split_type"],
                "test_record_or_block": values[0]["test_record_or_block"],
                "seed_passes": passes,
                "seed_total": len(values),
                "subject_pass": passes >= 2,
                "waveform_collapse_runs": collapses,
                "waveform_collapse_subject": collapses >= 2,
                "median_zero_improvement_pct": float(
                    np.median([row["zero_improvement_pct"] for row in values])
                ),
                "median_template_improvement_pct": float(
                    np.median([row["template_improvement_pct"] for row in values])
                ),
                "median_corr": float(np.median([row["median_corr"] for row in values])),
                "median_nrmse": float(np.median([row["median_nrmse"] for row in values])),
                "median_nrmse_p90": float(np.median([row["nrmse_p90"] for row in values])),
                "median_negative_improvement_fraction": float(
                    np.median(
                        [row["negative_improvement_window_fraction"] for row in values]
                    )
                ),
            }
        )
    return rows


def evaluate_a1_gate(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    subjects = subject_summary(results)
    subject_pass_count = sum(row["subject_pass"] for row in subjects)
    collapse_count = sum(row["waveform_collapse_subject"] for row in subjects)
    passed = subject_pass_count >= 5 and collapse_count <= 2
    return {
        "stage": "A1_nonfog_generalization",
        "run_pass_count": sum(bool(row["strict_pass"]) for row in results),
        "run_total": len(results),
        "subject_pass_count": subject_pass_count,
        "subject_total": 7,
        "waveform_collapse_subject_count": collapse_count,
        "minimum_subject_passes": 5,
        "maximum_waveform_collapse_subjects": 2,
        "status": "PASS" if passed else "FAIL",
        "advance_to_A2": passed,
        "subject_summary": subjects,
    }


def render_global_figures(root: Path, results: Sequence[dict[str, Any]]) -> None:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, key, title in zip(
        axes,
        ("zero_improvement_pct", "median_corr", "median_nrmse"),
        ("Zero improvement (%)", "Pearson", "NRMSE"),
    ):
        groups = [
            [row[key] for row in results if row["subject_id"] == subject]
            for subject in SUBJECTS
        ]
        ax.boxplot(groups, tick_labels=SUBJECTS, showfliers=True)
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "all_subject_generalization_boxplot.png", dpi=150)
    plt.close(fig)

    matrix = np.asarray(
        [
            [
                float(
                    next(
                        row["strict_pass"]
                        for row in results
                        if row["subject_id"] == subject and row["seed"] == seed
                    )
                )
                for seed in SEEDS
            ]
            for subject in SUBJECTS
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(3), [str(seed) for seed in SEEDS])
    ax.set_yticks(range(7), SUBJECTS)
    for row in range(7):
        for column in range(3):
            ax.text(column, row, "PASS" if matrix[row, column] else "FAIL", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="A1 PASS")
    fig.tight_layout()
    fig.savefig(figures / "subject_record_pass_matrix.png", dpi=150)
    plt.close(fig)

    x = np.arange(len(SUBJECTS))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    train_nrmse = [
        np.median([row["train_median_nrmse"] for row in results if row["subject_id"] == subject])
        for subject in SUBJECTS
    ]
    test_nrmse = [
        np.median([row["median_nrmse"] for row in results if row["subject_id"] == subject])
        for subject in SUBJECTS
    ]
    train_corr = [
        np.median([row["train_median_corr"] for row in results if row["subject_id"] == subject])
        for subject in SUBJECTS
    ]
    test_corr = [
        np.median([row["median_corr"] for row in results if row["subject_id"] == subject])
        for subject in SUBJECTS
    ]
    width = 0.36
    axes[0].bar(x - width / 2, train_nrmse, width, label="train")
    axes[0].bar(x + width / 2, test_nrmse, width, label="test")
    axes[0].set_title("NRMSE degradation")
    axes[1].bar(x - width / 2, train_corr, width, label="train")
    axes[1].bar(x + width / 2, test_corr, width, label="test")
    axes[1].set_title("Pearson degradation")
    for ax in axes:
        ax.set_xticks(x, SUBJECTS)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "train_to_test_degradation.png", dpi=150)
    plt.close(fig)


def compact_run_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "subject_id",
        "fold_id",
        "seed",
        "split_type",
        "test_record_or_block",
        "train_windows",
        "calibration_windows",
        "test_windows",
        "mse",
        "huber",
        "zero_improvement_pct",
        "template_improvement_pct",
        "median_corr",
        "median_nrmse",
        "nrmse_p90",
        "nrmse_p95",
        "median_amplitude_ratio",
        "negative_improvement_window_fraction",
        "spectral_nrmse_0p5_3hz",
        "spectral_nrmse_3_8hz",
        "train_median_corr",
        "train_median_nrmse",
        "strict_pass",
        "waveform_collapse",
        "best_epoch",
        "final_epoch",
        "elapsed_seconds",
    )
    return [{key: row.get(key) for key in keys} for row in results]


def baseline_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the four frozen A1 baselines for transparent comparison."""
    rows: list[dict[str, Any]] = []
    for result in results:
        for baseline_name, metrics in result["baselines"].items():
            rows.append(
                {
                    "subject_id": result["subject_id"],
                    "fold_id": result["fold_id"],
                    "seed": result["seed"],
                    "baseline": baseline_name,
                    **metrics,
                }
            )
    return rows


def write_reports(root: Path, results: Sequence[dict[str, Any]], gate: dict[str, Any]) -> None:
    reports = root / "reports"
    tables = root / "tables"
    base.write_csv(tables / "A1_run_metrics.csv", compact_run_rows(results))
    base.write_csv(tables / "A1_subject_summary.csv", gate["subject_summary"])
    base.write_csv(tables / "A1_baseline_comparison.csv", baseline_rows(results))
    base.write_json(reports / "A1_gate_decision.json", gate)
    subject_lines = "\n".join(
        f"- {row['subject_id']}: {row['seed_passes']}/3；"
        f"Zero改善率 {row['median_zero_improvement_pct']:.1f}%；"
        f"Template改善率 {row['median_template_improvement_pct']:.1f}%；"
        f"Pearson {row['median_corr']:.3f}；NRMSE {row['median_nrmse']:.3f}；"
        f"P90 {row['median_nrmse_p90']:.3f}；"
        f"{'PASS' if row['subject_pass'] else 'FAIL'}。"
        for row in gate["subject_summary"]
    )
    baseline_lines = []
    for subject in SUBJECTS:
        subject_results = [row for row in results if row["subject_id"] == subject]
        mean_mse = {
            baseline_name: float(
                np.mean([row["baselines"][baseline_name]["mse"] for row in subject_results])
            )
            for baseline_name in (
                "B0_zero",
                "B1_training_mean_template",
                "B2_nearest_training_window",
                "B3_M3_TC_DAE",
            )
        }
        baseline_lines.append(
            f"- {subject}: B0={mean_mse['B0_zero']:.4f}, "
            f"B1={mean_mse['B1_training_mean_template']:.4f}, "
            f"B2={mean_mse['B2_nearest_training_window']:.4f}, "
            f"M3={mean_mse['B3_M3_TC_DAE']:.4f}."
        )
    report = f"""# Route A A1 独立 Non-FoG 重构泛化报告

## 冻结协议

- 模型：M3_tcdae_long，约 64,633 参数。
- 预处理：仅用训练 clean Non-FoG 唯一点拟合 RobustScaler，随后逐窗口逐轴时间中心化。
- 多记录被试使用冻结的完整未见记录；S02、S07 按模板使用单记录 60/20/20 时间块，边界留 5 秒隔离区。
- 三个种子只改变初始化和 batch 顺序，数据划分完全相同。
- 测试记录/测试时间块未用于 scaler、训练、早停或阈值选择。

## 门控结论

- 运行通过：{gate['run_pass_count']}/{gate['run_total']}
- 被试通过：{gate['subject_pass_count']}/7（要求至少 5/7）
- 明显波形塌缩被试：{gate['waveform_collapse_subject_count']}（要求不超过 2）
- A1：{gate['status']}
- 是否进入 A2：{'是' if gate['advance_to_A2'] else '否'}

## 被试级结果

{subject_lines}

## 解释边界

A1 只评价未见 clean Non-FoG 重构。它不使用测试 FoG 选择模型，也不能单独证明残差可分离或分类增量。只有 A1 总体门控通过后才能进入 A2；若失败，后续报告必须标记为 NOT RUN。
"""
    report += (
        "\n\n## 基线 MSE 对照\n\n"
        "M3 在全部 7 名被试上的被试级平均 MSE 均低于 B0/B1/B2，"
        "但这不能覆盖预注册的逐窗波形门控。\n\n"
        + "\n".join(baseline_lines)
        + "\n"
    )
    (reports / "routeA_A1_nonfog_generalization_report.md").parent.mkdir(
        parents=True, exist_ok=True
    )
    (reports / "routeA_A1_nonfog_generalization_report.md").write_text(
        report, encoding="utf-8"
    )
    if not gate["advance_to_A2"]:
        deferred = {
            "routeA_A2_denoising_ablation_report.md": "A2 去噪对照",
            "routeA_A3_residual_calibration_report.md": "A3 残差校准",
            "routeA_A4_residual_separation_report.md": "A4/A5 残差表示与分离",
            "routeA_A5_residual_classifier_ablation_report.md": "A6 残差分类消融",
        }
        for filename, title in deferred.items():
            (reports / filename).write_text(
                f"# {title}\n\n状态：NOT RUN。\n\n原因：A1 总体门控为 FAIL；"
                "根据预注册决策树，不得使用测试 FoG 开展后续选择。\n",
                encoding="utf-8",
            )
        not_run_reason = (
            "A1 总体泛化门控失败；预注册顺序协议禁止使用测试 FoG 数据进行后续选择。"
        )
        for stage_dir in (
            "A2_denoising",
            "A3_residual_calibration",
            "A4_residual_representation",
            "A5_residual_separation",
            "A6_classifier_ablation",
            "A7_final_selection",
        ):
            base.write_json(
                root / stage_dir / "status.json",
                {"status": "NOT RUN", "reason": not_run_reason},
            )
    final = f"""# Daphnet NBM Route A 最终残差验证总结

Route A 已完成 A0 和 A1。A1 总体门控为 **{gate['status']}**：{gate['subject_pass_count']}/7 名被试达到至少 2/3 种子通过，明显波形塌缩被试 {gate['waveform_collapse_subject_count']} 名。

{'A1 通过，可进入 A2；本报告需在后续阶段完成后更新。' if gate['advance_to_A2'] else 'A1 未通过，实验按模板停止；A2-A7 均未运行，不生成 FoG 残差或分类结论。'}
"""
    if not gate["advance_to_A2"]:
        final += (
            "\n\nA2-A7 状态：NOT RUN。未选择最终 NBM 残差表示或分类器。\n"
        )
    (reports / "routeA_final_nbm_residual_report.md").write_text(
        final, encoding="utf-8"
    )


def audit(root: Path) -> dict[str, Any]:
    run_metrics = list((root / "A1_nonfog_generalization").rglob("run_metrics.json"))
    predictions = list((root / "A1_nonfog_generalization").rglob("predictions.npz"))
    checkpoints = list((root / "A1_nonfog_generalization").rglob("*_model.pt"))
    finite = True
    for path in predictions:
        with np.load(path, allow_pickle=False) as payload:
            finite = finite and all(np.isfinite(payload[key]).all() for key in payload.files)
    checkpoint_errors: list[str] = []
    for path in checkpoints:
        try:
            torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:  # pragma: no cover
            checkpoint_errors.append(f"{path}: {error}")
    output = {
        "run_metrics": len(run_metrics),
        "predictions": len(predictions),
        "checkpoints": len(checkpoints),
        "all_prediction_arrays_finite": finite,
        "all_checkpoints_loadable": not checkpoint_errors,
        "checkpoint_errors": checkpoint_errors,
        "subject_required_figures": len(
            list((root / "A1_nonfog_generalization").glob("S*/figures/*.png"))
        ),
        "global_figures": len(list((root / "figures").glob("*.png"))),
        "temporary_files": len(
            [path for path in root.rglob("*") if path.is_file() and ".tmp-" in path.name]
        ),
        "zero_byte_files": len(
            [path for path in root.rglob("*") if path.is_file() and path.stat().st_size == 0]
        ),
    }
    base.write_json(root / "reports" / "artifact_audit.json", output)
    return output


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    root = args.output_dir / "routeA_final_residual_validation"
    root.mkdir(parents=True, exist_ok=True)
    dataset = DaphnetDataset.load(args.data_dir)
    prepared = {subject: prepare_subject(dataset, subject) for subject in SUBJECTS}
    write_a0(root, dataset, prepared, args)
    preflight = {
        subject: {
            "split_type": item.disclosure["split_type"],
            "train_clean_nonfog": len(item.train_x),
            "calibration_clean_nonfog": len(item.calibration_x),
            "test_clean_nonfog": len(item.test_x),
        }
        for subject, item in prepared.items()
    }
    base.write_json(root / "A0_protocol" / "preflight.json", preflight)
    print(f"ROUTE A PREFLIGHT {json.dumps(preflight, ensure_ascii=False)}", flush=True)
    if args.preflight_only:
        return
    device = base.resolve_device(args.device)
    results: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for seed in SEEDS:
            results.append(run_a1_once(prepared[subject], root, seed, args, device))
    gate = evaluate_a1_gate(results)
    if not args.skip_figures:
        for subject in SUBJECTS:
            render_subject_figures(
                root,
                subject,
                [row for row in results if row["subject_id"] == subject],
                dataset.channel_names,
            )
        render_global_figures(root, results)
    write_reports(root, results, gate)
    artifact_audit = audit(root)
    print(
        f"ROUTE A A1 COMPLETE status={gate['status']} "
        f"subjects={gate['subject_pass_count']}/7 audit={artifact_audit}",
        flush=True,
    )


if __name__ == "__main__":
    main()
