#!/usr/bin/env python3
"""Render a leakage-audited six-panel GRU-NBM residual mechanism case figure."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.resume import atomic_json_dump, sha256_file
from scripts import run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn as worker
from scripts import run_all_dataset_processed_nbm_exp_within_subject_raw_tcn as raw_base
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    RepresentationTCNM,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import RobustScaler


DEFAULT_EXPERIMENT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "all_dataset_processed_NBM_Exp_within_subject_gru_base_mask4_8_C_tcn_"
    "nbm300pat20_ep5pat2_seedset_0_52_161_5216_52161"
)
DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "0.Private" / "processed_NBM_Exp"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "figures" / "private_gru_nbm_residual_mechanism_case"
)
SEED_ORDER = (0, 52, 161, 5216, 52161)
STRIDE_SAMPLES = 64
FIXED_CHANNEL_INDICES = (0, 9, 15)
FIXED_CHANNEL_LABELS = (
    "Lumbar Acc-x",
    "Left ankle Gyro-x",
    "Right ankle Gyro-x",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--context-before-sec", type=float, default=4.5)
    parser.add_argument("--context-after-sec", type=float, default=4.5)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def select_subject(summary: dict[str, Any]) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    candidates = [
        {
            "subject": str(row["subject"]),
            "ap": float(row["pr_auc_mean"]),
        }
        for row in summary["subjects"]
    ]
    median_ap = float(np.median([row["ap"] for row in candidates]))
    for row in candidates:
        row["absolute_distance_to_median_ap"] = abs(row["ap"] - median_ap)
    ordered = sorted(
        candidates,
        key=lambda row: (row["absolute_distance_to_median_ap"], row["subject"]),
    )
    return ordered[0], median_ap, ordered


def select_representative_run(
    run_rows: list[dict[str, str]], subject: str, subject_ap: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for row in run_rows:
        if row["subject"] != subject:
            continue
        seed = int(row["seed"])
        candidate = {
            "subject": subject,
            "fold": int(row["fold"]),
            "seed": seed,
            "ap": float(row["pr_auc"]),
        }
        candidate["absolute_distance_to_subject_ap"] = abs(
            candidate["ap"] - subject_ap
        )
        candidates.append(candidate)
    if len(candidates) != 15:
        raise AssertionError(f"expected 15 runs for {subject}, found {len(candidates)}")
    seed_rank = {seed: index for index, seed in enumerate(SEED_ORDER)}
    ordered = sorted(
        candidates,
        key=lambda row: (
            row["absolute_distance_to_subject_ap"],
            row["fold"],
            seed_rank[row["seed"]],
        ),
    )
    return ordered[0], ordered


def boolean_runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(values, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def predicted_events(
    record_rows: list[tuple[int, int, int]],
    sampling_rate_hz: int,
    minimum_positive_windows: int = 2,
    merge_gap_seconds: float = 0.5,
) -> list[tuple[int, int, int]]:
    """Replicate the frozen coverage_aware.v2 event construction exactly."""

    merge_gap = int(round(merge_gap_seconds * sampling_rate_hz))
    predicted: list[tuple[int, int, int]] = []
    active: list[tuple[int, int, int]] = []

    def finish_active() -> None:
        if len(active) >= minimum_positive_windows:
            interval = (
                active[0][0],
                active[-1][1],
                active[minimum_positive_windows - 1][1],
            )
            if predicted and interval[0] - predicted[-1][1] <= merge_gap:
                predicted[-1] = (
                    predicted[-1][0],
                    interval[1],
                    predicted[-1][2],
                )
            else:
                predicted.append(interval)
        active.clear()

    for start, end, prediction in sorted(record_rows, key=lambda item: item[0]):
        if prediction != 1:
            finish_active()
        else:
            if active and start - active[-1][1] > merge_gap:
                finish_active()
            active.append((start, end, prediction))
    finish_active()
    return predicted


def event_candidates(
    dataset: Any,
    test_rows: Any,
    prediction_by_window: dict[str, int],
) -> list[dict[str, Any]]:
    by_record: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for record_index, start, end, window_id in zip(
        test_rows.record_index,
        test_rows.start,
        test_rows.end,
        test_rows.window_id,
    ):
        key = str(window_id)
        if key not in prediction_by_window:
            raise KeyError(f"saved test prediction missing window: {key}")
        by_record[int(record_index)].append(
            (int(start), int(end), int(prediction_by_window[key]))
        )

    candidates: list[dict[str, Any]] = []
    for record_index, rows in sorted(by_record.items()):
        record = dataset.records[record_index]
        predictions = predicted_events(rows, dataset.sampling_rate_hz)
        true_intervals = [
            interval
            for interval in boolean_runs(record.y == 1)
            if any(
                max(interval[0], start) < min(interval[1], end)
                for start, end, _ in rows
            )
        ]
        for event_index, (start, end) in enumerate(true_intervals):
            matches = [
                {
                    "start_index": pred_start,
                    "end_index_exclusive": pred_end,
                    "detection_index": detection_index,
                }
                for pred_start, pred_end, detection_index in predictions
                if max(start, pred_start) < min(end, pred_end)
            ]
            candidates.append(
                {
                    "record_index": record_index,
                    "record_id": record.record_id,
                    "event_index_within_evaluable_record": event_index,
                    "start_index": start,
                    "end_index_exclusive": end,
                    "duration_samples": end - start,
                    "duration_sec": (end - start) / dataset.sampling_rate_hz,
                    "detected": bool(matches),
                    "matched_predicted_events": matches,
                }
            )
    if not candidates:
        raise RuntimeError("no evaluable permanent-test FoG events")
    median_duration = float(np.median([row["duration_sec"] for row in candidates]))
    for row in candidates:
        row["absolute_distance_to_median_duration_sec"] = abs(
            row["duration_sec"] - median_duration
        )
        row["participant_median_evaluable_event_duration_sec"] = median_duration
    return candidates


def select_event(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    detected = [row for row in candidates if row["detected"]]
    if not detected:
        raise RuntimeError("representative run detected no evaluable FoG event")
    return sorted(
        detected,
        key=lambda row: (
            row["absolute_distance_to_median_duration_sec"],
            row["record_id"],
            row["start_index"],
        ),
    )[0]


def load_frozen_models(
    run_dir: Path, device: torch.device
) -> tuple[Any, Any, RobustScaler, np.ndarray, float, dict[str, Any]]:
    frozen = read_json(run_dir / "FROZEN_TRAIN.json")
    scaler_payload = read_json(run_dir / "scaler_role4.json")["scaler"]
    scaler = RobustScaler(
        np.asarray(scaler_payload["median"], dtype=np.float32),
        np.asarray(scaler_payload["iqr"], dtype=np.float32),
        float(scaler_payload["epsilon"]),
    )
    sigma = np.asarray(
        read_json(run_dir / "calibration_role5.json")["sigma"], dtype=np.float32
    )
    nbm_payload = torch.load(
        run_dir / "checkpoints" / "gru_nbm_best.pt",
        map_location=device,
        weights_only=False,
    )
    nbm = worker.build_nbm_from_checkpoint(nbm_payload, device)
    nbm.eval()
    tcn_payload = torch.load(
        run_dir / "checkpoints" / "tcn.pt",
        map_location=device,
        weights_only=False,
    )
    tcn = RepresentationTCNM(worker.TCN_INPUT_CHANNELS).to(device)
    tcn.load_state_dict(tcn_payload["model_state"], strict=True)
    tcn.eval()
    return nbm, tcn, scaler, sigma, float(frozen["threshold"]), frozen


def residual_components(
    nbm: Any,
    scaler: RobustScaler,
    sigma: np.ndarray,
    raw: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scaled_uncentered = scaler.transform(raw)
    window_mean = scaled_uncentered.mean(axis=1, keepdims=True)
    observed = np.ascontiguousarray(scaled_uncentered - window_mean, dtype=np.float32)
    reconstruction = worker.reconstruct(nbm, observed, device, batch_size)
    q = np.clip(
        (observed - reconstruction) / (sigma[None, None, :] + 1e-6),
        -12.0,
        12.0,
    )
    residual = q - q.mean(axis=1, keepdims=True)
    absolute = np.abs(residual)
    delta = np.diff(residual, axis=1, prepend=residual[:, :1, :])
    reconstruction_scaled = reconstruction + window_mean
    reconstruction_raw = (
        reconstruction_scaled * (scaler.iqr[None, None, :] + scaler.epsilon)
        + scaler.median[None, None, :]
    ).astype(np.float32)
    return (
        observed,
        reconstruction,
        residual.astype(np.float32),
        absolute.astype(np.float32),
        delta.astype(np.float32),
        reconstruction_raw,
    )


def overlap_average(
    window_values: np.ndarray,
    starts: np.ndarray,
    display_start: int,
    display_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    length = display_end - display_start
    channels = window_values.shape[2]
    total = np.zeros((length, channels), dtype=np.float64)
    count = np.zeros(length, dtype=np.int32)
    for values, start in zip(window_values, starts):
        start = int(start)
        end = start + len(values)
        left = max(start, display_start)
        right = min(end, display_end)
        if left >= right:
            continue
        total[left - display_start : right - display_start] += values[
            left - start : right - start
        ]
        count[left - display_start : right - display_start] += 1
    if np.any(count == 0):
        missing = int(np.sum(count == 0))
        raise AssertionError(f"overlap-add visualization has {missing} uncovered samples")
    return (total / count[:, None]).astype(np.float32), count


def shade_fog(ax: Any, intervals: list[tuple[float, float]], selected: tuple[float, float]) -> None:
    for start, end in intervals:
        alpha = 0.19 if (start, end) == selected else 0.10
        ax.axvspan(start, end, color="#D95F5F", alpha=alpha, linewidth=0, zorder=0)


def sensor_group_labels() -> tuple[list[float], list[str]]:
    return (
        [2.5, 8.5, 14.5, 20.5, 26.5],
        ["Lumbar", "Left ankle", "Right ankle", "Left foot", "Right foot"],
    )


def plot_figure(
    output_dir: Path,
    dpi: int,
    metadata: dict[str, Any],
    time: np.ndarray,
    observed_raw: np.ndarray,
    reconstructed_raw: np.ndarray,
    residual: np.ndarray,
    absolute: np.ndarray,
    delta: np.ndarray,
    probability_time: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    prediction_strip: np.ndarray,
    ground_truth: np.ndarray,
    fog_intervals: list[tuple[float, float]],
    selected_interval: tuple[float, float],
    residual_vmax: float,
    delta_vmax: float,
    channel_units: list[str],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(8.3, 13.5))
    grid = GridSpec(
        6,
        1,
        figure=fig,
        height_ratios=(2.55, 1.72, 1.72, 1.72, 1.18, 0.72),
        hspace=0.23,
        left=0.145,
        right=0.92,
        top=0.955,
        bottom=0.055,
    )
    signal_grid = grid[0].subgridspec(3, 1, hspace=0.08)
    signal_axes = [fig.add_subplot(signal_grid[index, 0]) for index in range(3)]
    heat_axes = [fig.add_subplot(grid[index, 0]) for index in (1, 2, 3)]
    probability_ax = fig.add_subplot(grid[4, 0])
    strip_ax = fig.add_subplot(grid[5, 0])
    all_axes = [*signal_axes, *heat_axes, probability_ax, strip_ax]
    for ax in all_axes:
        ax.set_xlim(float(time[0]), float(time[-1]))

    observed_color = "#4A4A4A"
    reconstructed_color = "#2878B5"
    for row, (ax, channel, label) in enumerate(
        zip(signal_axes, FIXED_CHANNEL_INDICES, FIXED_CHANNEL_LABELS)
    ):
        shade_fog(ax, fog_intervals, selected_interval)
        ax.plot(
            time,
            observed_raw[:, channel],
            color=observed_color,
            linewidth=0.75,
            label="Observed",
            zorder=2,
        )
        ax.plot(
            time,
            reconstructed_raw[:, channel],
            color=reconstructed_color,
            linewidth=0.9,
            label="NBM reconstruction",
            zorder=3,
        )
        ax.set_ylabel(f"{label}\n({channel_units[channel]})", fontsize=6.8)
        ax.grid(axis="x", color="#D8D8D8", linewidth=0.4, alpha=0.65)
        ax.tick_params(axis="x", labelbottom=False)
        if row == 0:
            ax.text(
                -0.105,
                1.12,
                "(a)",
                transform=ax.transAxes,
                fontsize=9,
                fontweight="bold",
                va="top",
            )
            ax.legend(
                loc="upper right",
                frameon=False,
                ncol=2,
                fontsize=7,
                handlelength=1.8,
            )

    heatmaps = (
        (residual, "(b)", r"Signed calibrated residual $R$", "RdBu_r", residual_vmax, True),
        (absolute, "(c)", r"Absolute residual $|R|$", "magma", residual_vmax, False),
        (delta, "(d)", r"First-difference residual $\Delta R$", "PuOr_r", delta_vmax, True),
    )
    group_ticks, group_names = sensor_group_labels()
    for ax, (values, panel, title, cmap, vmax, symmetric) in zip(heat_axes, heatmaps):
        if symmetric:
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            image = ax.imshow(
                values.T,
                aspect="auto",
                origin="upper",
                extent=[time[0], time[-1], 29.5, -0.5],
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                rasterized=True,
            )
        else:
            image = ax.imshow(
                values.T,
                aspect="auto",
                origin="upper",
                extent=[time[0], time[-1], 29.5, -0.5],
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
                rasterized=True,
            )
        shade_fog(ax, fog_intervals, selected_interval)
        for boundary in (5.5, 11.5, 17.5, 23.5):
            ax.axhline(boundary, color="white", linewidth=0.8, alpha=0.9)
            ax.axhline(boundary, color="black", linewidth=0.25, alpha=0.75)
        ax.set_yticks(group_ticks, group_names, fontsize=6.2)
        ax.set_ylabel("Sensor group\n(Ax, Ay, Az, Gx, Gy, Gz)", fontsize=6.5)
        ax.tick_params(axis="x", labelbottom=False)
        ax.text(
            -0.105,
            1.08,
            panel,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
        )
        ax.set_title(title, loc="left", fontsize=8, pad=3)
        colorbar = fig.colorbar(image, ax=ax, pad=0.010, fraction=0.022)
        colorbar.ax.tick_params(labelsize=6, width=0.5, length=2)
        colorbar.set_label("Standardized residual", fontsize=6.2)

    shade_fog(probability_ax, fog_intervals, selected_interval)
    probability_ax.plot(
        probability_time,
        probability,
        color="#1B6CA8",
        marker="o",
        markersize=2.8,
        linewidth=1.25,
        label="FoG probability",
    )
    probability_ax.axhline(
        threshold,
        color="#C43C39",
        linestyle="--",
        linewidth=1.0,
        label=rf"Validation threshold $\tau^*={threshold:.2f}$",
    )
    probability_ax.set_ylim(-0.03, 1.03)
    probability_ax.set_ylabel("FoG probability")
    probability_ax.grid(color="#D8D8D8", linewidth=0.4, alpha=0.65)
    probability_ax.tick_params(axis="x", labelbottom=False)
    probability_ax.legend(loc="upper right", frameon=False, fontsize=7)
    probability_ax.text(
        -0.105,
        1.08,
        "(e)",
        transform=probability_ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )

    strips = np.vstack((ground_truth, prediction_strip))
    strip_ax.imshow(
        strips,
        aspect="auto",
        origin="upper",
        extent=[time[0], time[-1], 1.5, -0.5],
        cmap=ListedColormap(["#F1F1F1", "#D95F5F"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    shade_fog(strip_ax, fog_intervals, selected_interval)
    for start, end in fog_intervals:
        strip_ax.axvline(start, color="#7F1D1D", linewidth=0.45, alpha=0.7)
        strip_ax.axvline(end, color="#7F1D1D", linewidth=0.45, alpha=0.7)
    strip_ax.set_yticks([0, 1], ["Ground truth", "Prediction"], fontsize=7)
    strip_ax.set_xlabel("Time relative to selected FoG onset (s)")
    strip_ax.text(
        -0.105,
        1.20,
        "(f)",
        transform=strip_ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )
    strip_ax.legend(
        handles=[
            Patch(facecolor="#F1F1F1", edgecolor="#B0B0B0", label="Non-FoG"),
            Patch(facecolor="#D95F5F", edgecolor="#D95F5F", label="FoG"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, -0.45),
        ncol=2,
        frameon=False,
        fontsize=6.8,
    )

    fig.suptitle(
        metadata.get(
            "figure_title",
            "Representative GRU-NBM residual mechanism case",
        )
        + "\n"
        f"{metadata['selected_subject']}, fold {metadata['selected_fold']}, "
        f"seed {metadata['selected_seed']}; event duration "
        f"{metadata['selected_event']['duration_sec']:.2f} s",
        fontsize=9.2,
        fontweight="bold",
        y=0.991,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / metadata.get(
        "figure_stem", "private_gru_nbm_residual_mechanism_case"
    )
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    experiment_root = args.experiment_root.resolve()
    output_dir = args.output_dir.resolve()
    if args.batch_size <= 0 or args.dpi <= 0:
        raise ValueError("batch size and DPI must be positive")
    if args.context_before_sec < 4.0 or args.context_after_sec < 4.0:
        raise ValueError("the requested figure requires at least 4 s on each side")
    for required in (
        data_dir / "nbm_protocol.json",
        data_dir / "schema.json",
        experiment_root / "summary.json",
        experiment_root / "run_metrics.csv",
        experiment_root / "TRAINING_BARRIER.json",
        experiment_root / "DONE.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    summary = read_json(experiment_root / "summary.json")
    selected_subject, median_subject_ap, subject_candidates = select_subject(summary)
    selected_run, run_candidates = select_representative_run(
        read_csv(experiment_root / "run_metrics.csv"),
        selected_subject["subject"],
        selected_subject["ap"],
    )
    subject = selected_subject["subject"]
    fold = int(selected_run["fold"])
    seed = int(selected_run["seed"])
    run_dir = worker.run_dir(experiment_root, subject, fold, seed)

    dataset = DaphnetDataset.load(data_dir)
    rows = raw_base.load_subject_rows(data_dir, dataset, subject, fold)
    test_rows = rows.take_role(0, 1)
    saved_predictions = read_csv(run_dir / "test_predictions.csv")
    prediction_by_window = {
        row["window_id"]: int(row["y_pred"]) for row in saved_predictions
    }
    probability_by_window = {
        row["window_id"]: float(row["probability"]) for row in saved_predictions
    }
    events = event_candidates(dataset, test_rows, prediction_by_window)
    selected_event = select_event(events)
    record_index = int(selected_event["record_index"])
    record = dataset.records[record_index]
    sampling_rate = int(dataset.sampling_rate_hz)
    display_start = max(
        0,
        int(selected_event["start_index"] - round(args.context_before_sec * sampling_rate)),
    )
    display_end = min(
        len(record.y),
        int(
            selected_event["end_index_exclusive"]
            + round(args.context_after_sec * sampling_rate)
        ),
    )
    achieved_before = (selected_event["start_index"] - display_start) / sampling_rate
    achieved_after = (display_end - selected_event["end_index_exclusive"]) / sampling_rate

    device = worker.resolve_device(args.device)
    nbm, tcn, scaler, sigma, threshold, frozen = load_frozen_models(run_dir, device)

    training_rows = rows.take_role(6, 7)
    training_raw = raw_base.raw_windows(dataset, training_rows)
    _, _, training_r, _, training_delta, _ = residual_components(
        nbm,
        scaler,
        sigma,
        training_raw,
        device,
        args.batch_size,
    )
    residual_vmax = float(np.percentile(np.abs(training_r), 99.0))
    delta_vmax = float(np.percentile(np.abs(training_delta), 99.0))
    if residual_vmax <= 0.0 or delta_vmax <= 0.0:
        raise AssertionError("training-derived residual color limit is non-positive")

    all_starts = np.arange(
        0,
        len(record.y) - worker.WINDOW_SAMPLES + 1,
        STRIDE_SAMPLES,
        dtype=np.int32,
    )
    display_window_mask = (
        (all_starts < display_end)
        & (all_starts + worker.WINDOW_SAMPLES > display_start)
    )
    starts = all_starts[display_window_mask]
    raw_windows = np.stack(
        [record.x[start : start + worker.WINDOW_SAMPLES] for start in starts]
    ).astype(np.float32)
    _, _, residual_windows, absolute_windows, delta_windows, reconstructed_windows = (
        residual_components(
            nbm,
            scaler,
            sigma,
            raw_windows,
            device,
            args.batch_size,
        )
    )
    features = np.concatenate(
        (residual_windows, absolute_windows, delta_windows), axis=2
    ).transpose(0, 2, 1)
    _, continuous_probability = worker.predict(
        tcn,
        np.ascontiguousarray(features, dtype=np.float32),
        np.zeros(len(features), dtype=np.int8),
        device,
        args.batch_size,
    )
    centers = starts + worker.WINDOW_SAMPLES // 2
    center_mask = (centers >= display_start) & (centers < display_end)
    probability_time = (
        centers[center_mask] - selected_event["start_index"]
    ) / sampling_rate
    probability = continuous_probability[center_mask]
    continuous_prediction = (continuous_probability >= threshold).astype(np.int8)

    recomputed_differences: list[float] = []
    for start, value in zip(starts, continuous_probability):
        window_id = f"{record.record_id}:{int(start)}:{int(start + worker.WINDOW_SAMPLES)}"
        if window_id in probability_by_window:
            recomputed_differences.append(abs(value - probability_by_window[window_id]))
    if not recomputed_differences:
        raise AssertionError("display inference does not overlap saved permanent-test windows")
    maximum_probability_difference = max(recomputed_differences)
    if maximum_probability_difference > 2e-5:
        raise AssertionError(
            "recomputed frozen-model probabilities differ from sealed test output: "
            f"{maximum_probability_difference}"
        )

    reconstructed_raw, overlap_counts = overlap_average(
        reconstructed_windows, starts, display_start, display_end
    )
    residual, _ = overlap_average(residual_windows, starts, display_start, display_end)
    absolute, _ = overlap_average(absolute_windows, starts, display_start, display_end)
    delta, _ = overlap_average(delta_windows, starts, display_start, display_end)
    observed_raw = record.x[display_start:display_end]
    ground_truth = record.y[display_start:display_end].astype(np.int8)
    time = (
        np.arange(display_start, display_end) - selected_event["start_index"]
    ) / sampling_rate

    prediction_strip = np.zeros(len(time), dtype=np.int8)
    half_update = STRIDE_SAMPLES // 2
    for center, value in zip(centers, continuous_prediction):
        left = max(display_start, int(center - half_update))
        right = min(display_end, int(center + half_update))
        if left < right:
            prediction_strip[left - display_start : right - display_start] = value

    raw_fog_intervals = [
        (start, end)
        for start, end in boolean_runs(record.y == 1)
        if max(start, display_start) < min(end, display_end)
    ]
    fog_intervals = [
        (
            (max(start, display_start) - selected_event["start_index"]) / sampling_rate,
            (min(end, display_end) - selected_event["start_index"]) / sampling_rate,
        )
        for start, end in raw_fog_intervals
    ]
    selected_interval = (
        0.0,
        selected_event["duration_samples"] / sampling_rate,
    )

    schema = read_json(data_dir / "schema.json")
    channel_units = [str(row["unit"]) for row in schema["channels"]]
    metadata = {
        "schema": "private_gru_nbm_residual_mechanism_case.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_is_visual_signal_independent": True,
        "selection_rule": {
            "subject": (
                "smallest absolute distance between participant mean test AP and "
                "the median across participants; lexical subject ID breaks exact ties"
            ),
            "representative_run": (
                "smallest absolute distance between run test AP and the selected "
                "participant mean test AP; fold then fixed seed order break ties"
            ),
            "event": (
                "among correctly detected evaluable permanent-test FoG events, "
                "smallest absolute distance to that participant's median evaluable "
                "event duration; record ID then event start break ties"
            ),
            "channels": "fixed a priori as indices 0, 9, and 15 from schema.json",
            "color_limits": (
                "99th percentile of absolute role-6/7 training R and delta-R values "
                "from the selected frozen run; no case-derived color scaling"
            ),
        },
        "participant_median_ap": median_subject_ap,
        "selected_subject": subject,
        "selected_subject_ap": selected_subject["ap"],
        "selected_fold": fold,
        "selected_seed": seed,
        "selected_run_ap": selected_run["ap"],
        "selected_event": selected_event,
        "context_before_sec": achieved_before,
        "context_after_sec": achieved_after,
        "record_id": record.record_id,
        "sampling_rate_hz": sampling_rate,
        "window_samples": worker.WINDOW_SAMPLES,
        "stride_samples": STRIDE_SAMPLES,
        "window_seconds": worker.WINDOW_SAMPLES / sampling_rate,
        "stride_seconds": STRIDE_SAMPLES / sampling_rate,
        "probability_display": (
            "unsmoothed frozen-model probabilities at 2-s window centers"
        ),
        "binary_prediction_display": (
            "piecewise-constant 1-s update bins centered at window centers"
        ),
        "continuous_display_inference": (
            "frozen NBM/TCN applied at the canonical 1-s stride throughout the "
            "displayed record context; selection used sealed permanent-test outputs only"
        ),
        "overlap_add_visualization": (
            "window-level NBM reconstructions and residual samples averaged over all "
            "overlapping canonical windows; this is visualization only, not classifier input"
        ),
        "reconstruction_display": (
            "the observed per-window robust-scaled temporal mean was restored before "
            "inverse RobustScaler transformation to physical units"
        ),
        "residual_definition": (
            "q=clip((X-Xhat)/(sigma+1e-6),-12,12); "
            "R=q-mean_time(q); delta-R[0]=0"
        ),
        "scheme_c_uses_bias": False,
        "threshold": threshold,
        "residual_vmax_train_p99": residual_vmax,
        "delta_vmax_train_p99": delta_vmax,
        "training_color_scale_roles": [6, 7],
        "fixed_display_channels": [
            {
                "index": index,
                "name": schema["channels"][index]["name"],
                "label": label,
                "unit": channel_units[index],
            }
            for index, label in zip(FIXED_CHANNEL_INDICES, FIXED_CHANNEL_LABELS)
        ],
        "sensor_order": [
            "lumbar",
            "ankle_l",
            "ankle_r",
            "foot_l",
            "foot_r",
        ],
        "within_sensor_channel_order": ["Acc-x", "Acc-y", "Acc-z", "Gyro-x", "Gyro-y", "Gyro-z"],
        "maximum_recomputed_vs_sealed_probability_difference": maximum_probability_difference,
        "overlap_count_min": int(overlap_counts.min()),
        "overlap_count_max": int(overlap_counts.max()),
        "frozen_id": frozen["frozen_id"],
        "source_artifacts": {
            "experiment_summary": str((experiment_root / "summary.json").resolve()),
            "training_barrier": str((experiment_root / "TRAINING_BARRIER.json").resolve()),
            "run_directory": str(run_dir.resolve()),
            "nbm_checkpoint_sha256": sha256_file(run_dir / "checkpoints" / "gru_nbm_best.pt"),
            "tcn_checkpoint_sha256": sha256_file(run_dir / "checkpoints" / "tcn.pt"),
            "scaler_sha256": sha256_file(run_dir / "scaler_role4.json"),
            "calibration_sha256": sha256_file(run_dir / "calibration_role5.json"),
        },
    }

    plot_figure(
        output_dir,
        args.dpi,
        metadata,
        time,
        observed_raw,
        reconstructed_raw,
        residual,
        absolute,
        delta,
        probability_time,
        probability,
        threshold,
        prediction_strip,
        ground_truth,
        fog_intervals,
        selected_interval,
        residual_vmax,
        delta_vmax,
        channel_units,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(metadata, output_dir / "case_selection_audit.json")
    write_csv(
        output_dir / "subject_selection_candidates.csv",
        ("subject", "ap", "absolute_distance_to_median_ap"),
        subject_candidates,
    )
    write_csv(
        output_dir / "run_selection_candidates.csv",
        ("subject", "fold", "seed", "ap", "absolute_distance_to_subject_ap"),
        run_candidates,
    )
    event_rows = [
        {
            key: value
            for key, value in row.items()
            if key != "matched_predicted_events"
        }
        for row in events
    ]
    write_csv(
        output_dir / "event_selection_candidates.csv",
        (
            "record_index",
            "record_id",
            "event_index_within_evaluable_record",
            "start_index",
            "end_index_exclusive",
            "duration_samples",
            "duration_sec",
            "detected",
            "absolute_distance_to_median_duration_sec",
            "participant_median_evaluable_event_duration_sec",
        ),
        event_rows,
    )
    np.savez_compressed(
        output_dir / "figure_data.npz",
        time_sec=time.astype(np.float32),
        observed_raw=observed_raw.astype(np.float32),
        reconstructed_raw=reconstructed_raw.astype(np.float32),
        residual=residual.astype(np.float32),
        absolute_residual=absolute.astype(np.float32),
        delta_residual=delta.astype(np.float32),
        probability_time_sec=probability_time.astype(np.float32),
        fog_probability=probability.astype(np.float32),
        threshold=np.asarray(threshold, dtype=np.float32),
        ground_truth=ground_truth.astype(np.int8),
        prediction_strip=prediction_strip.astype(np.int8),
        channel_names=np.asarray(dataset.channel_names),
    )
    caption = (
        "**Fig. Y. Representative residual mechanism of the GRU-NBM framework.** "
        f"The case was selected without visual inspection: {subject} had mean test AP "
        f"closest to the participant median, and fold {fold}/seed {seed} had run AP "
        "closest to that participant's mean. Among correctly detected evaluable test "
        "events, the event duration was closest to the participant median. (a) Fixed "
        "lumbar and bilateral-ankle channels show the observed signal and frozen NBM "
        "reconstruction. (b)-(d) Signed calibrated residual, absolute residual, and "
        "first-difference residual. Sensor order is lumbar, left ankle, right ankle, "
        "left foot, and right foot; within each group the order is Acc-x/y/z and "
        "Gyro-x/y/z. Color limits were fixed from the 99th percentile of role-6/7 "
        "training residuals. (e) Unsmoothed FoG probabilities at window centers and "
        "the validation-selected threshold. (f) Ground-truth and predicted labels; "
        "predictions are rendered as 1-s update bins. Red shading denotes annotated "
        "FoG. Window-level reconstruction/residual samples were overlap-averaged only "
        "for visualization; the classifier received each 2-s residual tensor independently."
    )
    (output_dir / "figure_caption.md").write_text(caption + "\n", encoding="utf-8")
    print(
        f"COMPLETE subject={subject} fold={fold} seed={seed} "
        f"record={record.record_id} event={selected_event['start_index']}:"
        f"{selected_event['end_index_exclusive']} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
