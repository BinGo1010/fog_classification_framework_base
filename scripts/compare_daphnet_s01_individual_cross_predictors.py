#!/usr/bin/env python
"""Compare individual-S01 and cross-subject GRU-NBM outputs on one test set.

Both frozen predictors receive the exact 447 windows from the within-S01
experiment's test record (S01_seg002).  Each predictor uses its own training-
fitted Robust Scaler, as it did during training.  Forecast means and standard
deviations are then inverse-transformed to the common physical unit (g) before
the two models are compared.

No TCN-M classifier is loaded: this script isolates the normal-behaviour
predictor outputs ``mu``, ``sigma``, and standardized residuals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_daphnet_s01_gru_h200_tcnm as core  # noqa: E402
from cnbr_fog.data import RobustChannelScaler, WindowTable  # noqa: E402
from cnbr_fog.nbm import GRUNBM  # noqa: E402
from cnbr_fog.rf125_classifiers import build_rf125_classifier  # noqa: E402
from cnbr_fog.resume import (  # noqa: E402
    atomic_json_dump,
    atomic_npz_save,
    canonical_fingerprint,
    sha256_file,
)


COMPARISON_VERSION = "daphnet_s01_individual_cross_gru_output_comparison.v1"
MODEL_ORDER = ("individual_s01", "cross_subject")
MODEL_LABELS = {
    "individual_s01": "Individual S01",
    "cross_subject": "Cross-subject",
}
GROUP_ORDER = ("all", "non_fog", "fog")
GROUP_LABELS = {"all": "All", "non_fog": "Non-FoG", "fog": "FoG"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two frozen GRU-NBM outputs on S01_seg002",
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
        "--individual-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "daphnet_s01_gru_h200_tcnm_seed42",
    )
    parser.add_argument(
        "--cross-dir",
        type=Path,
        default=(
            REPO_ROOT / "outputs" / "daphnet_loso_s01_gru_h200_tcnm_seed42"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / "daphnet_s01_individual_vs_cross_predictor"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_scaler(path: Path) -> RobustChannelScaler:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    center = np.asarray(payload["center"], dtype=np.float32)
    scale = np.asarray(payload["scale"], dtype=np.float32)
    clip = float(payload["clip"])
    if center.shape != (9,) or scale.shape != (9,):
        raise ValueError(f"Unexpected scaler shape in {path}")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)):
        raise ValueError(f"Non-finite scaler in {path}")
    if np.any(scale <= 0) or not math.isfinite(clip) or clip <= 0:
        raise ValueError(f"Invalid scaler values in {path}")
    return RobustChannelScaler(center=center, scale=scale, clip=clip)


def load_model(path: Path, device: torch.device) -> tuple[GRUNBM, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = dict(payload["model_config"])
    if config.pop("name") != "gru":
        raise ValueError(f"Checkpoint is not a GRU-NBM: {path}")
    model = GRUNBM(**config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload


def load_source_protocol(directory: Path) -> dict[str, Any]:
    with (directory / "config.json").open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    with (directory / "DONE.json").open("r", encoding="utf-8") as handle:
        done = json.load(handle)
    if done.get("status") != "complete":
        raise ValueError(f"Source experiment is not complete: {directory}")
    if done.get("protocol_fingerprint") != protocol.get("protocol_fingerprint"):
        raise ValueError(f"Source protocol/DONE fingerprint mismatch: {directory}")
    recorded_artifacts = done.get("artifacts", {})
    for filename in (
        "config.json",
        "scaler.json",
        "split_indices.npz",
        "nbm_best.pt",
        "classifier_best.pt",
        "classifier_training.json",
    ):
        recorded_sha = recorded_artifacts.get(filename)
        if not recorded_sha:
            raise ValueError(f"Source DONE lacks {filename}: {directory}")
        if sha256_file(directory / filename) != recorded_sha:
            raise ValueError(f"Source artifact hash mismatch for {directory / filename}")
    return protocol


def load_classifier(
    directory: Path, device: torch.device
) -> tuple[torch.nn.Module, float, dict[str, Any]]:
    checkpoint_path = directory / "classifier_best.pt"
    training_path = directory / "classifier_training.json"
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = dict(payload["architecture"])
    model = build_rf125_classifier(
        architecture["canonical_name"],
        in_channels=int(architecture["in_channels"]),
        input_samples=int(architecture["input_samples"]),
        hidden_channels=int(architecture["hidden_channels"]),
        dropout=float(architecture["dropout"]),
        dilations=tuple(int(value) for value in architecture["dilations"]),
        kernel_size=int(architecture["kernel_size"]),
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    with training_path.open("r", encoding="utf-8") as handle:
        training = json.load(handle)
    threshold = float(training["selected_threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Invalid classifier threshold in {training_path}")
    if int(payload["epoch"]) != int(training["best_epoch"]):
        raise ValueError(f"Classifier checkpoint/training epoch mismatch: {directory}")
    return model, threshold, payload


@torch.no_grad()
def infer_classifier(
    model: torch.nn.Module,
    residual: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    values = np.asarray(residual, dtype=np.float32)
    probabilities: list[np.ndarray] = []
    model.eval()
    for offset in range(0, len(values), int(batch_size)):
        batch = torch.from_numpy(values[offset : offset + int(batch_size)]).to(
            device
        )
        with torch.amp.autocast(
            device.type, enabled=bool(amp and device.type == "cuda")
        ):
            logits = model(batch)
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(probabilities).astype(np.float32, copy=False)


def exact_s01_test_support(data_dir: Path) -> tuple[Any, WindowTable, np.ndarray]:
    dataset = core.load_s01_dataset(data_dir)
    base_windows = dataset.make_windows(
        warmup_samples=core.CONTEXT_SAMPLES,
        target_samples=core.TARGET_SAMPLES,
        stride_samples=core.STRIDE_SAMPLES,
        fog_fraction_threshold=0.5,
        normal_guard_samples=core.NORMAL_GUARD_SAMPLES,
    )
    windows = core.endpoint_relabel(dataset, base_windows)
    split = core.make_split(dataset, windows)
    indices = split.test.astype(np.int64, copy=False)
    records = {
        dataset.records[int(windows.record_index[index])].record_id
        for index in indices
    }
    if records != {core.TEST_RECORD} or len(indices) != 447:
        raise AssertionError(
            f"Expected 447 S01_seg002 test windows, got {len(indices)} {records}"
        )
    counts = np.bincount(windows.label[indices], minlength=2).astype(int)
    if counts.tolist() != [423, 24]:
        raise AssertionError(f"Unexpected S01 test class counts: {counts.tolist()}")
    return dataset, windows, indices


def validate_saved_support(individual_dir: Path, indices: np.ndarray) -> None:
    with np.load(individual_dir / "split_indices.npz", allow_pickle=False) as payload:
        saved = np.asarray(payload["test_window_index"], dtype=np.int64)
    if not np.array_equal(saved, indices):
        raise AssertionError("Reconstructed S01 test support differs from saved support")


def raw_test_arrays(
    dataset: Any, windows: WindowTable, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index in indices:
        record = dataset.records[int(windows.record_index[index])]
        start = int(windows.start[index])
        target_start = int(windows.target_start[index])
        target_end = int(windows.target_end[index])
        contexts.append(record.x[start:target_start].T)
        targets.append(record.x[target_start:target_end].T)
    context = np.ascontiguousarray(np.stack(contexts), dtype=np.float32)
    target = np.ascontiguousarray(np.stack(targets), dtype=np.float32)
    labels = np.asarray(windows.label[indices], dtype=np.int8)
    expected = (len(indices), 9, 128)
    if context.shape != expected or target.shape != expected:
        raise AssertionError(
            f"Unexpected raw shapes context={context.shape} target={target.shape}"
        )
    return context, target, labels


@torch.no_grad()
def infer_predictor(
    model: GRUNBM,
    scaler: RobustChannelScaler,
    dataset: Any,
    windows: WindowTable,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp: bool,
) -> dict[str, np.ndarray]:
    loader = core.sequence_loader(
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
    sigmas: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    output_indices: list[np.ndarray] = []
    model.eval()
    for sequence, _, window_index in loader:
        sequence = sequence.to(device, non_blocking=True)
        context = sequence[:, :, : core.CONTEXT_SAMPLES]
        target = sequence[:, :, core.CONTEXT_SAMPLES :]
        with torch.amp.autocast(
            device.type, enabled=bool(amp and device.type == "cuda")
        ):
            mean, sigma = model(context)
            residual = (target - mean) / sigma
        means.append(mean.float().cpu().numpy())
        sigmas.append(sigma.float().cpu().numpy())
        targets.append(target.float().cpu().numpy())
        residuals.append(residual.float().cpu().numpy())
        output_indices.append(window_index.numpy())
    index = np.concatenate(output_indices).astype(np.int64, copy=False)
    if not np.array_equal(index, indices):
        raise AssertionError("Inference changed test window order")
    mean_scaled = np.concatenate(means).astype(np.float32, copy=False)
    sigma_scaled = np.concatenate(sigmas).astype(np.float32, copy=False)
    target_scaled = np.concatenate(targets).astype(np.float32, copy=False)
    residual_unclipped = np.concatenate(residuals).astype(np.float32, copy=False)
    scale = scaler.scale[None, :, None]
    center = scaler.center[None, :, None]
    return {
        "mean_scaled": mean_scaled,
        "sigma_scaled": sigma_scaled,
        "target_scaled_clipped": target_scaled,
        "mean_raw_g": (mean_scaled * scale + center).astype(np.float32),
        "sigma_raw_g": (sigma_scaled * scale).astype(np.float32),
        "target_clipped_raw_g": (target_scaled * scale + center).astype(np.float32),
        "residual_pipeline_unclipped": residual_unclipped,
        "residual_pipeline_clipped": np.clip(
            residual_unclipped, -core.RESIDUAL_CLIP, core.RESIDUAL_CLIP
        ).astype(np.float32),
    }


def group_masks(labels: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(labels), dtype=bool),
        "non_fog": labels == 0,
        "fog": labels == 1,
    }


def scaler_clipping_statistics(
    scaler: RobustChannelScaler,
    context_raw_g: np.ndarray,
    target_raw_g: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, values in (("context", context_raw_g), ("target", target_raw_g)):
        z = (
            values.astype(np.float64) - scaler.center[None, :, None]
        ) / scaler.scale[None, :, None]
        clipped = np.abs(z) > scaler.clip
        result[name] = {
            "cells": int(z.size),
            "clipped_cells": int(clipped.sum()),
            "clipped_fraction": float(clipped.mean()),
            "per_channel_clipped_fraction": clipped.mean(axis=(0, 2)).tolist(),
        }
    return result


def predictor_metrics(
    output: Mapping[str, np.ndarray],
    context_raw_g: np.ndarray,
    target_raw_g: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    target = target_raw_g[mask].astype(np.float64)
    mean = output["mean_raw_g"][mask].astype(np.float64)
    sigma = output["sigma_raw_g"][mask].astype(np.float64)
    pipeline_residual = output["residual_pipeline_unclipped"][mask].astype(
        np.float64
    )
    error = target - mean
    standardized_raw_error = error / sigma
    persistence_mean = context_raw_g[mask, :, -1:].astype(np.float64)
    persistence = target - persistence_mean
    model_mse = float(np.mean(np.square(error)))
    persistence_mse = float(np.mean(np.square(persistence)))
    return {
        "windows": int(mask.sum()),
        "cells": int(error.size),
        "forecast_rmse_raw_g": float(np.sqrt(model_mse)),
        "forecast_mae_raw_g": float(np.mean(np.abs(error))),
        "persistence_rmse_raw_g": float(np.sqrt(persistence_mse)),
        "mse_skill_vs_persistence": float(
            1.0 - model_mse / persistence_mse if persistence_mse > 0 else np.nan
        ),
        "gaussian_nll_raw_no_constant": float(
            np.mean(np.log(sigma) + 0.5 * np.square(standardized_raw_error))
        ),
        "mean_predicted_sigma_raw_g": float(np.mean(sigma)),
        "median_predicted_sigma_raw_g": float(np.median(sigma)),
        "raw_error_within_1sigma": float(
            np.mean(np.abs(standardized_raw_error) <= 1.0)
        ),
        "raw_error_within_1p96sigma": float(
            np.mean(np.abs(standardized_raw_error) <= 1.96)
        ),
        "pipeline_residual_mean": float(np.mean(pipeline_residual)),
        "pipeline_residual_std": float(np.std(pipeline_residual)),
        "pipeline_residual_rms": float(
            np.sqrt(np.mean(np.square(pipeline_residual)))
        ),
        "pipeline_residual_clipped_fraction": float(
            np.mean(np.abs(pipeline_residual) > core.RESIDUAL_CLIP)
        ),
    }


def flat_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64).ravel()
    right64 = np.asarray(right, dtype=np.float64).ravel()
    if np.std(left64) == 0 or np.std(right64) == 0:
        return float("nan")
    return float(np.corrcoef(left64, right64)[0, 1])


def agreement_metrics(
    individual: Mapping[str, np.ndarray],
    cross: Mapping[str, np.ndarray],
    context_raw_g: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    mu_left = individual["mean_raw_g"][mask].astype(np.float64)
    mu_right = cross["mean_raw_g"][mask].astype(np.float64)
    sigma_left = individual["sigma_raw_g"][mask].astype(np.float64)
    sigma_right = cross["sigma_raw_g"][mask].astype(np.float64)
    residual_left = individual["residual_pipeline_unclipped"][mask]
    residual_right = cross["residual_pipeline_unclipped"][mask]
    mu_difference = mu_right - mu_left
    anchor = context_raw_g[mask, :, -1:].astype(np.float64)
    increment_left = mu_left - anchor
    increment_right = mu_right - anchor
    sigma_ratio = sigma_right / sigma_left
    return {
        "windows": int(mask.sum()),
        "mean_output_mae_between_models_g": float(
            np.mean(np.abs(mu_difference))
        ),
        "mean_output_rmse_between_models_g": float(
            np.sqrt(np.mean(np.square(mu_difference)))
        ),
        "mean_output_pearson_r": flat_correlation(mu_left, mu_right),
        "forecast_increment_pearson_r": flat_correlation(
            increment_left, increment_right
        ),
        "sigma_output_median_cross_over_individual": float(
            np.median(sigma_ratio)
        ),
        "sigma_output_mean_cross_over_individual": float(np.mean(sigma_ratio)),
        "sigma_output_pearson_r": flat_correlation(sigma_left, sigma_right),
        "pipeline_residual_pearson_r": flat_correlation(
            residual_left, residual_right
        ),
    }


def per_window_rows(
    dataset: Any,
    windows: WindowTable,
    indices: np.ndarray,
    labels: np.ndarray,
    target_raw_g: np.ndarray,
    outputs: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, np.ndarray]] = {}
    for name in MODEL_ORDER:
        error = target_raw_g.astype(np.float64) - outputs[name]["mean_raw_g"].astype(
            np.float64
        )
        sigma = outputs[name]["sigma_raw_g"].astype(np.float64)
        z = error / sigma
        metrics[name] = {
            "rmse": np.sqrt(np.mean(np.square(error), axis=(1, 2))),
            "mae": np.mean(np.abs(error), axis=(1, 2)),
            "nll": np.mean(np.log(sigma) + 0.5 * np.square(z), axis=(1, 2)),
        }
    for row_index, window_index in enumerate(indices):
        record = dataset.records[int(windows.record_index[window_index])]
        target_end = int(windows.target_end[window_index])
        individual_rmse = float(metrics["individual_s01"]["rmse"][row_index])
        cross_rmse = float(metrics["cross_subject"]["rmse"][row_index])
        rows.append(
            {
                "window_index": int(window_index),
                "record_id": record.record_id,
                "target_end_sample": target_end,
                "decision_time_sec": target_end / core.SAMPLING_RATE_HZ,
                "y_true": int(labels[row_index]),
                "individual_rmse_g": individual_rmse,
                "cross_rmse_g": cross_rmse,
                "cross_minus_individual_rmse_g": cross_rmse - individual_rmse,
                "individual_mae_g": float(
                    metrics["individual_s01"]["mae"][row_index]
                ),
                "cross_mae_g": float(metrics["cross_subject"]["mae"][row_index]),
                "individual_nll_raw": float(
                    metrics["individual_s01"]["nll"][row_index]
                ),
                "cross_nll_raw": float(
                    metrics["cross_subject"]["nll"][row_index]
                ),
            }
        )
    return rows


def paired_window_summary(
    window_rows: list[dict[str, Any]], labels: np.ndarray
) -> dict[str, dict[str, Any]]:
    individual = np.asarray(
        [row["individual_rmse_g"] for row in window_rows], dtype=np.float64
    )
    cross = np.asarray([row["cross_rmse_g"] for row in window_rows], dtype=np.float64)
    result: dict[str, dict[str, Any]] = {}
    for group, mask in group_masks(labels).items():
        delta = cross[mask] - individual[mask]
        result[group] = {
            "windows": int(mask.sum()),
            "individual_lower_rmse_fraction": float(np.mean(delta > 0)),
            "cross_lower_rmse_fraction": float(np.mean(delta < 0)),
            "tie_fraction": float(np.mean(delta == 0)),
            "median_cross_minus_individual_rmse_g": float(np.median(delta)),
            "mean_cross_minus_individual_rmse_g": float(np.mean(delta)),
        }
    return result


def downstream_metrics(
    labels: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result: dict[str, Any] = {
        "scope": (
            "each GRU residual is evaluated by its own trained TCN-M and its "
            "own validation-selected threshold; differences are full-pipeline, "
            "not predictor-only"
        ),
        "models": {},
    }
    predictions: dict[str, np.ndarray] = {}
    for name in MODEL_ORDER:
        probability = np.asarray(probabilities[name], dtype=np.float64)
        threshold = float(thresholds[name])
        prediction = (probability >= threshold).astype(np.int8)
        predictions[name] = prediction
        metrics = core.enrich_metrics(
            core.binary_metrics(labels, probability, threshold)
        )
        metrics["positive_predictions"] = int(prediction.sum())
        metrics["negative_predictions"] = int(len(prediction) - prediction.sum())
        result["models"][name] = metrics
    individual = predictions["individual_s01"]
    cross = predictions["cross_subject"]
    result["paired_outputs"] = {
        "windows": int(len(labels)),
        "binary_agreement_fraction": float(np.mean(individual == cross)),
        "both_non_fog": int(np.sum((individual == 0) & (cross == 0))),
        "individual_non_fog_cross_fog": int(
            np.sum((individual == 0) & (cross == 1))
        ),
        "individual_fog_cross_non_fog": int(
            np.sum((individual == 1) & (cross == 0))
        ),
        "both_fog": int(np.sum((individual == 1) & (cross == 1))),
        "probability_mae_between_pipelines": float(
            np.mean(
                np.abs(
                    np.asarray(probabilities["individual_s01"], dtype=np.float64)
                    - np.asarray(probabilities["cross_subject"], dtype=np.float64)
                )
            )
        ),
        "probability_pearson_r": flat_correlation(
            probabilities["individual_s01"], probabilities["cross_subject"]
        ),
    }
    return result, predictions


def per_channel_rows(
    channel_names: tuple[str, ...],
    target_raw_g: np.ndarray,
    labels: np.ndarray,
    outputs: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, mask in group_masks(labels).items():
        for model_name in MODEL_ORDER:
            error = target_raw_g[mask].astype(np.float64) - outputs[model_name][
                "mean_raw_g"
            ][mask].astype(np.float64)
            sigma = outputs[model_name]["sigma_raw_g"][mask].astype(np.float64)
            for channel, channel_name in enumerate(channel_names):
                channel_error = error[:, channel]
                rows.append(
                    {
                        "group": group,
                        "model": model_name,
                        "channel_index": channel,
                        "channel_name": channel_name,
                        "rmse_raw_g": float(
                            np.sqrt(np.mean(np.square(channel_error)))
                        ),
                        "mae_raw_g": float(np.mean(np.abs(channel_error))),
                        "mean_sigma_raw_g": float(np.mean(sigma[:, channel])),
                    }
                )
    return rows


def per_horizon_rows(
    target_raw_g: np.ndarray,
    labels: np.ndarray,
    outputs: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, mask in group_masks(labels).items():
        for model_name in MODEL_ORDER:
            error = target_raw_g[mask].astype(np.float64) - outputs[model_name][
                "mean_raw_g"
            ][mask].astype(np.float64)
            sigma = outputs[model_name]["sigma_raw_g"][mask].astype(np.float64)
            rmse = np.sqrt(np.mean(np.square(error), axis=(0, 1)))
            mae = np.mean(np.abs(error), axis=(0, 1))
            mean_sigma = np.mean(sigma, axis=(0, 1))
            for step in range(core.TARGET_SAMPLES):
                rows.append(
                    {
                        "group": group,
                        "model": model_name,
                        "horizon_sample": step + 1,
                        "horizon_seconds": (step + 1) / core.SAMPLING_RATE_HZ,
                        "rmse_raw_g": float(rmse[step]),
                        "mae_raw_g": float(mae[step]),
                        "mean_sigma_raw_g": float(mean_sigma[step]),
                    }
                )
    return rows


def plot_comparison(
    path: Path,
    metrics: Mapping[str, Any],
    horizon_rows: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
) -> None:
    colors = {"individual_s01": "#2563eb", "cross_subject": "#e76f51"}
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)

    x = np.arange(len(GROUP_ORDER), dtype=float)
    width = 0.36
    for offset, model_name in enumerate(MODEL_ORDER):
        values = [
            metrics["predictors"][model_name][group]["forecast_rmse_raw_g"]
            for group in GROUP_ORDER
        ]
        axes[0, 0].bar(
            x + (offset - 0.5) * width,
            values,
            width,
            label=MODEL_LABELS[model_name],
            color=colors[model_name],
        )
    axes[0, 0].set_xticks(x, [GROUP_LABELS[group] for group in GROUP_ORDER])
    axes[0, 0].set_ylabel("Forecast RMSE (g)")
    axes[0, 0].set_title("Forecast error on identical S01 windows")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(axis="y", alpha=0.25)

    for model_name in MODEL_ORDER:
        rows = [
            row
            for row in horizon_rows
            if row["group"] == "all" and row["model"] == model_name
        ]
        axes[0, 1].plot(
            [row["horizon_seconds"] for row in rows],
            [row["rmse_raw_g"] for row in rows],
            label=MODEL_LABELS[model_name],
            color=colors[model_name],
            linewidth=2.0,
        )
    axes[0, 1].set_xlabel("Forecast horizon (s)")
    axes[0, 1].set_ylabel("RMSE (g)")
    axes[0, 1].set_title("Error growth across the 2 s target")
    axes[0, 1].grid(alpha=0.25)

    for model_name in MODEL_ORDER:
        rows = [
            row
            for row in horizon_rows
            if row["group"] == "all" and row["model"] == model_name
        ]
        axes[1, 0].plot(
            [row["horizon_seconds"] for row in rows],
            [row["mean_sigma_raw_g"] for row in rows],
            label=MODEL_LABELS[model_name],
            color=colors[model_name],
            linewidth=2.0,
        )
    axes[1, 0].set_xlabel("Forecast horizon (s)")
    axes[1, 0].set_ylabel("Mean predicted sigma (g)")
    axes[1, 0].set_title("Predictive uncertainty output")
    axes[1, 0].grid(alpha=0.25)

    individual_rmse = np.asarray(
        [row["individual_rmse_g"] for row in window_rows], dtype=float
    )
    cross_rmse = np.asarray([row["cross_rmse_g"] for row in window_rows], dtype=float)
    labels = np.asarray([row["y_true"] for row in window_rows], dtype=np.int8)
    for label, display, color in ((0, "Non-FoG", "#64748b"), (1, "FoG", "#dc2626")):
        mask = labels == label
        axes[1, 1].scatter(
            individual_rmse[mask],
            cross_rmse[mask],
            s=20 if label == 0 else 34,
            alpha=0.35 if label == 0 else 0.8,
            color=color,
            label=display,
            edgecolors="none",
        )
    maximum = float(max(individual_rmse.max(), cross_rmse.max()))
    axes[1, 1].plot([0, maximum], [0, maximum], "k--", linewidth=1.2)
    axes[1, 1].set_xlim(0, maximum * 1.02)
    axes[1, 1].set_ylim(0, maximum * 1.02)
    axes[1, 1].set_xlabel("Individual S01 window RMSE (g)")
    axes[1, 1].set_ylabel("Cross-subject window RMSE (g)")
    axes[1, 1].set_title("Paired window comparison (below line: cross better)")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(alpha=0.2)

    figure.suptitle(
        "GRU-NBM predictor outputs on S01_seg002\n"
        "Same 447 windows; each frozen model uses its own training-fitted scaler",
        fontsize=15,
        fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    figure.savefig(temporary, dpi=180)
    plt.close(figure)
    os.replace(temporary, path)


def plot_example_outputs(
    path: Path,
    indices: np.ndarray,
    labels: np.ndarray,
    target_raw_g: np.ndarray,
    channel_names: tuple[str, ...],
    outputs: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    selected_rows = [
        int(np.flatnonzero(labels == label)[0]) for label in (0, 1)
    ]
    selected_channels = (1, 4, 7)
    colors = {"individual_s01": "#2563eb", "cross_subject": "#e76f51"}
    time_seconds = np.arange(1, core.TARGET_SAMPLES + 1) / core.SAMPLING_RATE_HZ
    figure, axes = plt.subplots(
        2, 3, figsize=(15.0, 8.5), sharex=True
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.075,
        top=0.82,
        wspace=0.16,
        hspace=0.26,
    )
    metadata: list[dict[str, Any]] = []
    for row, selected in enumerate(selected_rows):
        label_name = "Non-FoG" if int(labels[selected]) == 0 else "FoG"
        metadata.append(
            {
                "selection_rule": f"chronologically first {label_name} test window",
                "row": selected,
                "window_index": int(indices[selected]),
                "y_true": int(labels[selected]),
            }
        )
        for column, channel in enumerate(selected_channels):
            axis = axes[row, column]
            axis.plot(
                time_seconds,
                target_raw_g[selected, channel],
                color="#111827",
                linewidth=2.0,
                label="Observed target",
                zorder=5,
            )
            for model_name in MODEL_ORDER:
                mean = outputs[model_name]["mean_raw_g"][selected, channel]
                sigma = outputs[model_name]["sigma_raw_g"][selected, channel]
                lower = mean - 1.96 * sigma
                upper = mean + 1.96 * sigma
                axis.fill_between(
                    time_seconds,
                    lower,
                    upper,
                    color=colors[model_name],
                    alpha=0.12,
                    linewidth=0,
                )
                axis.plot(
                    time_seconds,
                    mean,
                    color=colors[model_name],
                    linewidth=1.6,
                    label=f"{MODEL_LABELS[model_name]} mean",
                )
            axis.set_title(channel_names[channel])
            axis.grid(alpha=0.22)
            if column == 0:
                axis.set_ylabel(f"{label_name}\nAcceleration (g)")
            if row == 1:
                axis.set_xlabel("Future time (s)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.905),
    )
    figure.suptitle(
        "Direct GRU-NBM forecasts (shading: model 95% interval)",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    figure.savefig(temporary, dpi=180)
    plt.close(figure)
    os.replace(temporary, path)
    return metadata


def report_text(metrics: Mapping[str, Any]) -> str:
    individual = metrics["predictors"]["individual_s01"]
    cross = metrics["predictors"]["cross_subject"]
    paired = metrics["paired_window_comparison"]
    agreement = metrics["output_agreement"]
    overall_individual = individual["all"]
    overall_cross = cross["all"]
    individual_rmse_advantage = 100.0 * (
        overall_cross["forecast_rmse_raw_g"]
        - overall_individual["forecast_rmse_raw_g"]
    ) / overall_cross["forecast_rmse_raw_g"]
    cross_sigma_increase = 100.0 * (
        overall_cross["mean_predicted_sigma_raw_g"]
        / overall_individual["mean_predicted_sigma_raw_g"]
        - 1.0
    )
    lines = [
        "# S01个体预测器与跨个体预测器输出对比",
        "",
        "## 公平比较协议",
        "",
        "- 完全相同输入：单被试实验测试集 `S01_seg002`，447个窗口。",
        "- 每窗：2秒context预测未来2秒target，步长1秒，9个加速度通道。",
        "- 两个GRU-NBM均冻结；各自使用训练阶段拟合的Robust Scaler。",
        "- 均值和标准差均逆变换为共同物理单位g后比较。",
        "- 主分析隔离比较GRU输出；另附各自TCN-M与验证阈值下的完整管线结果。",
        "",
        "## 预测误差与不确定性",
        "",
        "| 数据 | 模型 | RMSE (g) | MAE (g) | raw NLL | 平均sigma (g) | ±1sigma覆盖率 | ±1.96sigma覆盖率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUP_ORDER:
        for model_name in MODEL_ORDER:
            row = metrics["predictors"][model_name][group]
            lines.append(
                f"| {GROUP_LABELS[group]} | {MODEL_LABELS[model_name]} | "
                f"{row['forecast_rmse_raw_g']:.6f} | "
                f"{row['forecast_mae_raw_g']:.6f} | "
                f"{row['gaussian_nll_raw_no_constant']:.6f} | "
                f"{row['mean_predicted_sigma_raw_g']:.6f} | "
                f"{100*row['raw_error_within_1sigma']:.2f}% | "
                f"{100*row['raw_error_within_1p96sigma']:.2f}% |"
            )
    lines.extend(
        [
            "",
            "## 同窗配对比较",
            "",
            "| 数据 | 个体模型RMSE更低 | 跨个体模型RMSE更低 | RMSE差值中位数 (cross-individual, g) |",
            "|---|---:|---:|---:|",
        ]
    )
    for group in GROUP_ORDER:
        row = paired[group]
        lines.append(
            f"| {GROUP_LABELS[group]} | "
            f"{100*row['individual_lower_rmse_fraction']:.2f}% | "
            f"{100*row['cross_lower_rmse_fraction']:.2f}% | "
            f"{row['median_cross_minus_individual_rmse_g']:.6f} |"
        )
    downstream = metrics["downstream_fog_pipeline"]
    lines.extend(
        [
            "",
            "## 完整FoG管线在同一447窗上的输出",
            "",
            "此表将每个GRU接回其各自的TCN-M和验证集阈值，因而是完整管线对比，"
            "不能把差异只归因于GRU。",
            "",
            "| 管线 | 阈值 | Accuracy | FoG recall | TN | FP | FN | TP | 正判窗 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name in MODEL_ORDER:
        row = downstream["models"][model_name]
        lines.append(
            f"| {MODEL_LABELS[model_name]} | {row['threshold']:.2f} | "
            f"{100*row['accuracy']:.2f}% | {100*row['fog_recall']:.2f}% | "
            f"{row['tn']} | {row['fp']} | {row['fn']} | {row['tp']} | "
            f"{row['positive_predictions']} |"
        )
    paired_output = downstream["paired_outputs"]
    lines.extend(
        [
            "",
            f"两条管线的二值输出一致率为{100*paired_output['binary_agreement_fraction']:.2f}%；"
            f"跨个体管线相较个体管线额外把{paired_output['individual_non_fog_cross_fog']}个窗判为FoG。",
        ]
    )
    lines.extend(
        [
            "",
            "## 两模型输出差异",
            "",
            f"- 全部窗口均值输出相关系数：{agreement['all']['mean_output_pearson_r']:.6f}",
            f"- 去除共同末值锚点后的预测增量相关系数：{agreement['all']['forecast_increment_pearson_r']:.6f}",
            f"- 全部窗口均值输出MAE：{agreement['all']['mean_output_mae_between_models_g']:.6f} g",
            f"- 跨个体/个体sigma中位数比：{agreement['all']['sigma_output_median_cross_over_individual']:.6f}",
            f"- 标准化残差相关系数：{agreement['all']['pipeline_residual_pearson_r']:.6f}",
            "",
            "## 主要解释",
            "",
            f"- 个体模型的整体物理RMSE低{individual_rmse_advantage:.2f}%，均值预测略优。",
            f"- 跨个体模型的平均sigma高{cross_sigma_increase:.2f}%，其95%覆盖率更接近标称值，"
            "说明主要差异来自不确定性头而不是未来均值。",
            f"- 虽然跨个体模型在{100*paired['all']['cross_lower_rmse_fraction']:.2f}%的窗口上RMSE更低，"
            "但少数大误差窗使其聚合RMSE更高；不能只看胜窗比例。",
            "- 原始mu相关系数很高，部分由两个GRU都锚定最后一个context采样点造成；"
            "去掉共同锚点后，预测增量相关性明显下降。",
            "",
            "## 结论边界",
            "",
            "这是一个固定seed、同一重叠滑窗测试集上的配对描述性比较。窗口之间并不独立，"
            "因此未把逐窗差异当作独立样本进行显著性检验。",
            "",
            "![Predictor output comparison](predictor_output_comparison.png)",
            "",
            "以下示例严格按时间顺序选择首个non-FoG窗和首个FoG窗，不按模型表现挑选。",
            "",
            "![Direct predictor outputs](example_forecast_outputs.png)",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    data_dir = args.data_dir.resolve()
    individual_dir = args.individual_dir.resolve()
    cross_dir = args.cross_dir.resolve()
    output_dir = args.output_dir.resolve()
    done_path = output_dir / "DONE.json"
    if done_path.exists() and not args.overwrite:
        raise FileExistsError(f"Completed comparison exists: {done_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is non-empty; pass --overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = core.resolve_device(args.device)

    dataset, windows, indices = exact_s01_test_support(data_dir)
    validate_saved_support(individual_dir, indices)
    context_raw_g, target_raw_g, labels = raw_test_arrays(dataset, windows, indices)

    experiment_dirs = {
        "individual_s01": individual_dir,
        "cross_subject": cross_dir,
    }
    source_protocols = {
        name: load_source_protocol(directory)
        for name, directory in experiment_dirs.items()
    }
    if source_protocols["individual_s01"].get("subject") != "S01":
        raise AssertionError("Individual source protocol is not the S01 experiment")
    cross_split = source_protocols["cross_subject"].get("split", {})
    if cross_split.get("test_subject") != "S01":
        raise AssertionError("Cross-subject source protocol is not the S01 outer fold")
    scalers: dict[str, RobustChannelScaler] = {}
    models: dict[str, GRUNBM] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    classifiers: dict[str, torch.nn.Module] = {}
    thresholds: dict[str, float] = {}
    classifier_checkpoints: dict[str, dict[str, Any]] = {}
    for name in MODEL_ORDER:
        directory = experiment_dirs[name]
        if not (directory / "DONE.json").exists():
            raise FileNotFoundError(f"Incomplete experiment: {directory}")
        scalers[name] = load_scaler(directory / "scaler.json")
        models[name], checkpoints[name] = load_model(
            directory / "nbm_best.pt", device
        )
        if checkpoints[name].get("protocol_fingerprint") != source_protocols[name].get(
            "protocol_fingerprint"
        ):
            raise AssertionError(f"Checkpoint/protocol mismatch for {name}")
        classifiers[name], thresholds[name], classifier_checkpoints[name] = (
            load_classifier(directory, device)
        )
        if classifier_checkpoints[name].get(
            "protocol_fingerprint"
        ) != source_protocols[name].get("protocol_fingerprint"):
            raise AssertionError(f"Classifier/protocol mismatch for {name}")
    configs = [checkpoints[name]["model_config"] for name in MODEL_ORDER]
    if configs[0] != configs[1]:
        raise AssertionError(f"Predictor architectures differ: {configs}")

    outputs: dict[str, dict[str, np.ndarray]] = {}
    for name in MODEL_ORDER:
        print(f"[{name}] infer {len(indices)} windows on {device}", flush=True)
        outputs[name] = infer_predictor(
            models[name],
            scalers[name],
            dataset,
            windows,
            indices,
            args.batch_size,
            args.num_workers,
            device,
            args.amp,
        )

    masks = group_masks(labels)
    metrics: dict[str, Any] = {
        "comparison_version": COMPARISON_VERSION,
        "test_support": {
            "subject": "S01",
            "record": core.TEST_RECORD,
            "windows": int(len(indices)),
            "class_counts_non_fog_fog": np.bincount(
                labels, minlength=2
            ).astype(int).tolist(),
            "context_samples": core.CONTEXT_SAMPLES,
            "target_samples": core.TARGET_SAMPLES,
            "stride_samples": core.STRIDE_SAMPLES,
            "sampling_rate_hz": core.SAMPLING_RATE_HZ,
        },
        "predictor_architecture": configs[0],
        "predictors": {},
        "output_agreement": {},
        "training_provenance": {
            "individual_s01": {
                "training_scope": "earlier S01 train partition",
                "validation_scope": "later S01 validation partition",
                "checkpoint_epoch": int(checkpoints["individual_s01"]["epoch"]),
                "protocol_fingerprint": source_protocols["individual_s01"][
                    "protocol_fingerprint"
                ],
            },
            "cross_subject": {
                "training_scope": cross_split.get("train_subjects"),
                "validation_scope": cross_split.get("validation_subject"),
                "checkpoint_epoch": int(checkpoints["cross_subject"]["epoch"]),
                "protocol_fingerprint": source_protocols["cross_subject"][
                    "protocol_fingerprint"
                ],
            },
        },
        "scaler_policy": (
            "each frozen predictor uses its own training-fitted Robust Scaler; "
            "mu and sigma are inverse-transformed to g for common-unit comparison"
        ),
        "scaler_clipping_on_shared_support": {
            name: scaler_clipping_statistics(
                scalers[name], context_raw_g, target_raw_g
            )
            for name in MODEL_ORDER
        },
    }
    for name in MODEL_ORDER:
        metrics["predictors"][name] = {
            group: predictor_metrics(
                outputs[name], context_raw_g, target_raw_g, mask
            )
            for group, mask in masks.items()
        }
    metrics["output_agreement"] = {
        group: agreement_metrics(
            outputs["individual_s01"],
            outputs["cross_subject"],
            context_raw_g,
            mask,
        )
        for group, mask in masks.items()
    }

    fog_probabilities = {
        name: infer_classifier(
            classifiers[name],
            outputs[name]["residual_pipeline_clipped"],
            batch_size=args.batch_size,
            device=device,
            amp=args.amp,
        )
        for name in MODEL_ORDER
    }
    metrics["downstream_fog_pipeline"], fog_predictions = downstream_metrics(
        labels, fog_probabilities, thresholds
    )

    window_rows = per_window_rows(
        dataset, windows, indices, labels, target_raw_g, outputs
    )
    metrics["paired_window_comparison"] = paired_window_summary(
        window_rows, labels
    )
    channel_rows = per_channel_rows(
        dataset.channel_names, target_raw_g, labels, outputs
    )
    horizon_rows = per_horizon_rows(target_raw_g, labels, outputs)

    config = {
        "comparison_version": COMPARISON_VERSION,
        "created_utc": core.utc_now(),
        "data_dir": str(data_dir),
        "individual_experiment_dir": str(individual_dir),
        "cross_subject_experiment_dir": str(cross_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "amp_requested": args.amp,
        "input_protocol": (
            "exact saved within-S01 test support; raw S01_seg002 windows shared"
        ),
        "individual_checkpoint_sha256": sha256_file(
            individual_dir / "nbm_best.pt"
        ),
        "cross_checkpoint_sha256": sha256_file(cross_dir / "nbm_best.pt"),
        "individual_scaler_sha256": sha256_file(individual_dir / "scaler.json"),
        "cross_scaler_sha256": sha256_file(cross_dir / "scaler.json"),
        "individual_classifier_sha256": sha256_file(
            individual_dir / "classifier_best.pt"
        ),
        "cross_classifier_sha256": sha256_file(cross_dir / "classifier_best.pt"),
        "individual_source_protocol_fingerprint": source_protocols[
            "individual_s01"
        ]["protocol_fingerprint"],
        "cross_source_protocol_fingerprint": source_protocols["cross_subject"][
            "protocol_fingerprint"
        ],
    }
    config["comparison_fingerprint"] = canonical_fingerprint(
        {key: value for key, value in config.items() if key != "created_utc"}
    )

    atomic_json_dump(config, output_dir / "config.json")
    atomic_json_dump(metrics, output_dir / "metrics.json")
    write_csv(output_dir / "per_window_metrics.csv", window_rows)
    write_csv(output_dir / "per_channel_metrics.csv", channel_rows)
    write_csv(output_dir / "per_horizon_metrics.csv", horizon_rows)
    atomic_npz_save(
        output_dir / "predictor_outputs.npz",
        window_index=indices,
        y_true=labels,
        context_raw_g=context_raw_g,
        target_raw_g=target_raw_g,
        individual_mean_raw_g=outputs["individual_s01"]["mean_raw_g"],
        individual_sigma_raw_g=outputs["individual_s01"]["sigma_raw_g"],
        individual_residual_pipeline_unclipped=outputs["individual_s01"][
            "residual_pipeline_unclipped"
        ],
        individual_residual_pipeline_clipped=outputs["individual_s01"][
            "residual_pipeline_clipped"
        ],
        cross_mean_raw_g=outputs["cross_subject"]["mean_raw_g"],
        cross_sigma_raw_g=outputs["cross_subject"]["sigma_raw_g"],
        cross_residual_pipeline_unclipped=outputs["cross_subject"][
            "residual_pipeline_unclipped"
        ],
        cross_residual_pipeline_clipped=outputs["cross_subject"][
            "residual_pipeline_clipped"
        ],
        individual_fog_probability=fog_probabilities["individual_s01"],
        individual_fog_prediction=fog_predictions["individual_s01"],
        cross_fog_probability=fog_probabilities["cross_subject"],
        cross_fog_prediction=fog_predictions["cross_subject"],
    )
    plot_comparison(
        output_dir / "predictor_output_comparison.png",
        metrics,
        horizon_rows,
        window_rows,
    )
    metrics["example_windows"] = plot_example_outputs(
        output_dir / "example_forecast_outputs.png",
        indices,
        labels,
        target_raw_g,
        dataset.channel_names,
        outputs,
    )
    atomic_json_dump(metrics, output_dir / "metrics.json")
    report_path = output_dir / "report.md"
    report_path.write_text(report_text(metrics), encoding="utf-8")
    atomic_json_dump(
        {
            "status": "complete",
            "completed_utc": core.utc_now(),
            "comparison_version": COMPARISON_VERSION,
            "comparison_fingerprint": config["comparison_fingerprint"],
            "artifacts": {
                path.name: sha256_file(path)
                for path in sorted(output_dir.iterdir())
                if path.is_file() and path.name != done_path.name
            },
        },
        done_path,
    )
    print(json.dumps(metrics["predictors"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
