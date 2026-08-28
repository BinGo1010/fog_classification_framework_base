#!/usr/bin/env python
"""Compare original epoch and step-trained GRU-NBM experiments.

Outputs event-level FROC curves, normal-role residual distributions, test
reconstruction/residual-direction diagnostics, and channel-scale/false-alarm
analyses for P03, P06, and P08. All permanent-test analyses are post hoc and
use already sealed model artifacts and probabilities.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.resume import atomic_json_dump, sha256_file
from scripts import run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn as model_base
from scripts import run_all_dataset_processed_nbm_exp_within_subject_raw_tcn as data_base
from scripts.summarize_private_raw_tcn_latest_event_metrics import boolean_runs


SUBJECTS = tuple(f"P{index:02d}" for index in range(1, 9))
FOLDS = (0, 1, 2)
SEEDS = (0, 52, 161, 5216, 52161)
DIFFICULT_SUBJECTS = ("P03", "P06", "P08")
ROLE_LABELS = {4: "Train-N", 5: "Val-N", 0: "Test-N", 1: "Test-FoG"}
METHOD_LABELS = {"original": "Original epoch", "latest": "Latest step"}
COLORS = {"original": "#3B6FB6", "latest": "#D95F59"}
THRESHOLDS = np.linspace(0.0, 1.0, 101, dtype=np.float64)
MERGE_GAP_SECONDS = 1.0
EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "0.Private" / "processed_NBM_Exp",
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "all_dataset_processed_NBM_Exp_within_subject_gru_base_mask4_8_C_tcn_"
        "nbm300pat20_ep5pat2_seedset_0_52_161_5216_52161",
    )
    parser.add_argument(
        "--latest-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "processed_NBM_Exp_gru_nbm_step_aug40_40_20_C_tcn_5seed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "processed_NBM_Exp_gru_nbm_original_vs_step_diagnostics",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require_complete(root: Path) -> None:
    for path in (root / "TRAINING_BARRIER.json", root / "DONE.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    barrier = read_json(root / "TRAINING_BARRIER.json")
    if barrier.get("status") != "sealed" or int(barrier.get("job_count", -1)) != 120:
        raise AssertionError(f"experiment is not a sealed 120-job run: {root}")
    for subject in SUBJECTS:
        for fold in FOLDS:
            for seed in SEEDS:
                destination = root / "runs" / subject / f"fold_{fold}" / f"seed_{seed}"
                for name in (
                    "FROZEN_TRAIN.json",
                    "DONE_TEST.json",
                    "test_predictions.csv",
                    "checkpoints/gru_nbm_best.pt",
                    "scaler_role4.json",
                    "calibration_role5.json",
                ):
                    if not (destination / name).is_file():
                        raise FileNotFoundError(destination / name)


def load_scaler(path: Path) -> Any:
    payload = read_json(path)["scaler"]
    return model_base.RobustScaler(
        np.asarray(payload["median"], dtype=np.float32),
        np.asarray(payload["iqr"], dtype=np.float32),
        float(payload["epsilon"]),
    )


def load_nbm(destination: Path, seed: int, device: torch.device) -> torch.nn.Module:
    payload = torch.load(
        destination / "checkpoints" / "gru_nbm_best.pt",
        map_location=device,
        weights_only=False,
    )
    if int(payload["seed"]) != int(seed):
        raise AssertionError(f"checkpoint seed mismatch: {destination}")
    return model_base.build_nbm_from_checkpoint(payload, device)


def residual_arrays(
    model: torch.nn.Module,
    scaler: Any,
    sigma: np.ndarray,
    raw: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = model_base.centered_scaled_ntc(scaler, raw)
    x_hat = model_base.reconstruct(model, x, device, batch_size)
    error = x - x_hat
    q = np.clip(error / (sigma[None, None, :] + EPSILON), -12.0, 12.0)
    residual = q - q.mean(axis=1, keepdims=True)
    return error.astype(np.float32), q.astype(np.float32), residual.astype(np.float32)


def role_summary(
    method: str,
    subject: str,
    fold: int,
    seed: int,
    role: int,
    error: np.ndarray,
    q: np.ndarray,
    residual: np.ndarray,
) -> dict[str, Any]:
    return {
        "method": method,
        "subject": subject,
        "fold": fold,
        "seed": seed,
        "role": role,
        "role_name": ROLE_LABELS[role],
        "n_windows": int(error.shape[0]),
        "reconstruction_mae": float(np.mean(np.abs(error))),
        "reconstruction_abs_median": float(np.median(np.abs(error))),
        "standardized_abs_median": float(np.median(np.abs(q))),
        "residual_abs_median": float(np.median(np.abs(residual))),
        "residual_abs_p90": float(np.percentile(np.abs(residual), 90.0)),
        "q_signed_mean": float(np.mean(q)),
        "q_signed_median": float(np.median(q)),
        "q_positive_fraction": float(np.mean(q > 0.0)),
    }


def prediction_lookup(path: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in read_csv_rows(path):
        mapping[row["window_id"]] = {
            "probability": float(row["probability"]),
            "threshold": float(row["threshold"]),
            "y_pred": int(row["y_pred"]),
            "y_true": int(row["y_true"]),
        }
    return mapping


def collect_residual_diagnostics(
    dataset: Any,
    data_dir: Path,
    roots: dict[str, Path],
    output_dir: Path,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device("cpu")
    role_rows_out: list[dict[str, Any]] = []
    channel_rows_out: list[dict[str, Any]] = []
    for method, root in roots.items():
        for subject in SUBJECTS:
            for fold in FOLDS:
                all_rows = data_base.load_subject_rows(data_dir, dataset, subject, fold)
                role_tables = {role: all_rows.take_role(role) for role in ROLE_LABELS}
                raw_by_role = {
                    role: data_base.raw_windows(dataset, table)
                    for role, table in role_tables.items()
                }
                for seed in SEEDS:
                    destination = root / "runs" / subject / f"fold_{fold}" / f"seed_{seed}"
                    scaler = load_scaler(destination / "scaler_role4.json")
                    calibration = read_json(destination / "calibration_role5.json")
                    sigma = np.asarray(calibration["sigma"], dtype=np.float32)
                    model = load_nbm(destination, seed, device)
                    residual_by_role: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
                    for role in ROLE_LABELS:
                        arrays = residual_arrays(
                            model,
                            scaler,
                            sigma,
                            raw_by_role[role],
                            device,
                            batch_size,
                        )
                        residual_by_role[role] = arrays
                        role_rows_out.append(
                            role_summary(method, subject, fold, seed, role, *arrays)
                        )

                    if subject in DIFFICULT_SUBJECTS:
                        test_rows = all_rows.take_role(0, 1)
                        error, q, residual = residual_arrays(
                            model,
                            scaler,
                            sigma,
                            data_base.raw_windows(dataset, test_rows),
                            device,
                            batch_size,
                        )
                        lookup = prediction_lookup(destination / "test_predictions.csv")
                        predicted = np.asarray(
                            [lookup[str(window_id)]["y_pred"] for window_id in test_rows.window_id],
                            dtype=np.int8,
                        )
                        if not np.array_equal(test_rows.label, np.asarray(
                            [lookup[str(window_id)]["y_true"] for window_id in test_rows.window_id],
                            dtype=np.int8,
                        )):
                            raise AssertionError(f"test prediction order mismatch: {destination}")
                        false_positive = (test_rows.label == 0) & (predicted == 1)
                        true_negative = (test_rows.label == 0) & (predicted == 0)
                        for channel, channel_name in enumerate(dataset.channel_names):
                            fp_values = np.abs(residual[false_positive, :, channel]).reshape(-1)
                            tn_values = np.abs(residual[true_negative, :, channel]).reshape(-1)
                            fp_median = float(np.median(fp_values)) if fp_values.size else math.nan
                            tn_median = float(np.median(tn_values)) if tn_values.size else math.nan
                            ratio = (
                                (fp_median + EPSILON) / (tn_median + EPSILON)
                                if math.isfinite(fp_median) and math.isfinite(tn_median)
                                else math.nan
                            )
                            channel_rows_out.append(
                                {
                                    "method": method,
                                    "subject": subject,
                                    "fold": fold,
                                    "seed": seed,
                                    "channel_index": channel,
                                    "channel_name": channel_name,
                                    "sigma": float(sigma[channel]),
                                    "false_positive_windows": int(np.sum(false_positive)),
                                    "true_negative_windows": int(np.sum(true_negative)),
                                    "fp_residual_abs_median": fp_median,
                                    "tn_residual_abs_median": tn_median,
                                    "fp_to_tn_residual_ratio": ratio,
                                    "fp_q_positive_fraction": (
                                        float(np.mean(q[false_positive, :, channel] > 0.0))
                                        if np.any(false_positive)
                                        else math.nan
                                    ),
                                    "tn_q_positive_fraction": (
                                        float(np.mean(q[true_negative, :, channel] > 0.0))
                                        if np.any(true_negative)
                                        else math.nan
                                    ),
                                }
                            )
                    del model
            print(f"RESIDUAL COMPLETE method={method} subject={subject}", flush=True)

    role_frame = pd.DataFrame(role_rows_out)
    channel_frame = pd.DataFrame(channel_rows_out)
    role_frame.to_csv(output_dir / "residual_role_run_metrics.csv", index=False)
    channel_frame.to_csv(output_dir / "difficult_subject_channel_run_metrics.csv", index=False)
    return role_frame, channel_frame


def merge_count(intervals: Iterable[tuple[int, int]], maximum_gap_samples: int) -> int:
    ordered = sorted((int(start), int(end)) for start, end in intervals)
    if not ordered:
        return 0
    count = 1
    active_end = ordered[0][1]
    for start, end in ordered[1:]:
        if start - active_end <= maximum_gap_samples:
            active_end = max(active_end, end)
        else:
            count += 1
            active_end = end
    return count


def prepare_froc_run(dataset: Any, prediction_path: Path) -> dict[str, Any]:
    frame = pd.read_csv(prediction_path)
    record_lookup = {record.record_id: record for record in dataset.records}
    by_record: dict[str, dict[str, Any]] = {}
    evaluated_nonfog_samples = 0
    total_events = 0
    for record_id, group in frame.groupby("record_id", sort=True):
        record = record_lookup[str(record_id)]
        nonfog_coverage = np.zeros(len(record.y), dtype=bool)
        fog_coverage = np.zeros(len(record.y), dtype=bool)
        nonfog_rows: list[tuple[int, int, float]] = []
        fog_rows: list[tuple[int, int, float]] = []
        for row in group.itertuples(index=False):
            start, end = int(row.start_index), int(row.end_index_exclusive)
            target = nonfog_coverage if int(row.y_true) == 0 else fog_coverage
            target[start:end] = True
            values = (start, end, float(row.probability))
            (nonfog_rows if int(row.y_true) == 0 else fog_rows).append(values)
        evaluated_nonfog_samples += int(
            np.sum(nonfog_coverage & record.valid & (record.y == 0))
        )
        events = [
            (start, end)
            for start, end in boolean_runs(record.y == 1)
            if np.any(fog_coverage[start:end])
        ]
        total_events += len(events)
        by_record[str(record_id)] = {
            "nonfog_rows": nonfog_rows,
            "fog_rows": fog_rows,
            "events": events,
        }
    if total_events <= 0 or evaluated_nonfog_samples <= 0:
        raise AssertionError(f"undefined FROC denominators: {prediction_path}")
    return {
        "by_record": by_record,
        "total_events": total_events,
        "nonfog_hours": evaluated_nonfog_samples / dataset.sampling_rate_hz / 3600.0,
    }


def froc_metrics(prepared: dict[str, Any], threshold: float, maximum_gap: int) -> tuple[float, float]:
    detected = 0
    false_alarms = 0
    for record in prepared["by_record"].values():
        positive_fog = [row for row in record["fog_rows"] if row[2] >= threshold]
        for event_start, event_end in record["events"]:
            detected += int(
                any(max(event_start, start) < min(event_end, end) for start, end, _ in positive_fog)
            )
        positive_nonfog = [
            (start, end)
            for start, end, probability in record["nonfog_rows"]
            if probability >= threshold
        ]
        false_alarms += merge_count(positive_nonfog, maximum_gap)
    return (
        detected / int(prepared["total_events"]),
        false_alarms / float(prepared["nonfog_hours"]),
    )


def collect_froc(
    dataset: Any,
    roots: dict[str, Path],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    maximum_gap = int(round(MERGE_GAP_SECONDS * dataset.sampling_rate_hz))
    for method, root in roots.items():
        for subject in SUBJECTS:
            for fold in FOLDS:
                for seed in SEEDS:
                    destination = root / "runs" / subject / f"fold_{fold}" / f"seed_{seed}"
                    prepared = prepare_froc_run(dataset, destination / "test_predictions.csv")
                    for threshold in THRESHOLDS:
                        event_sensitivity, false_alarms_per_hour = froc_metrics(
                            prepared, float(threshold), maximum_gap
                        )
                        run_rows.append(
                            {
                                "method": method,
                                "subject": subject,
                                "fold": fold,
                                "seed": seed,
                                "threshold": float(threshold),
                                "event_sensitivity": event_sensitivity,
                                "false_alarms_per_hour": false_alarms_per_hour,
                            }
                        )
            print(f"FROC COMPLETE method={method} subject={subject}", flush=True)
    run_frame = pd.DataFrame(run_rows)
    subject_seed = (
        run_frame.groupby(["method", "threshold", "subject", "seed"], as_index=False)[
            ["event_sensitivity", "false_alarms_per_hour"]
        ]
        .mean()
    )
    seed_macro = (
        subject_seed.groupby(["method", "threshold", "seed"], as_index=False)[
            ["event_sensitivity", "false_alarms_per_hour"]
        ]
        .mean()
    )
    summary = (
        seed_macro.groupby(["method", "threshold"], as_index=False)
        .agg(
            event_sensitivity_mean=("event_sensitivity", "mean"),
            event_sensitivity_sd=("event_sensitivity", lambda values: values.std(ddof=0)),
            false_alarms_per_hour_mean=("false_alarms_per_hour", "mean"),
            false_alarms_per_hour_sd=("false_alarms_per_hour", lambda values: values.std(ddof=0)),
        )
    )
    run_frame.to_csv(output_dir / "event_froc_run_metrics.csv", index=False)
    summary.to_csv(output_dir / "event_froc_summary.csv", index=False)
    return run_frame, summary


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)


def draw_box_scatter(
    axis: plt.Axes,
    groups: list[np.ndarray],
    positions: list[float],
    colors: list[str],
    widths: float = 0.34,
) -> None:
    result = axis.boxplot(
        groups,
        positions=positions,
        widths=widths,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1F1F1F", "linewidth": 1.2},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
    )
    for box, color in zip(result["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.42)
        box.set_edgecolor(color)
    rng = np.random.default_rng(20260828)
    for values, position, color in zip(groups, positions, colors):
        jitter = rng.normal(0.0, widths * 0.10, size=len(values))
        axis.scatter(
            position + jitter,
            values,
            s=7,
            alpha=0.20,
            color=color,
            linewidths=0,
            rasterized=True,
        )


def operating_points(roots: dict[str, Path]) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for method, root in roots.items():
        path = root / "latest_event_metrics_1s" / "raw_tcn_subject_seed_metrics_unrounded.csv"
        frame = pd.read_csv(path)
        seed_macro = frame.groupby("seed")[["event_sensitivity", "false_alarms_per_hour"]].mean()
        result[method] = (
            float(seed_macro["false_alarms_per_hour"].mean()),
            float(seed_macro["event_sensitivity"].mean()),
        )
    return result


def plot_froc(summary: pd.DataFrame, roots: dict[str, Path], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.8, 5.2))
    points = operating_points(roots)
    for method in ("original", "latest"):
        values = summary[summary.method == method].sort_values("false_alarms_per_hour_mean")
        x = values.false_alarms_per_hour_mean.to_numpy()
        y = values.event_sensitivity_mean.to_numpy()
        ysd = values.event_sensitivity_sd.to_numpy()
        axis.plot(x, y, color=COLORS[method], linewidth=2.2, label=METHOD_LABELS[method])
        axis.fill_between(
            x,
            np.clip(y - ysd, 0.0, 1.0),
            np.clip(y + ysd, 0.0, 1.0),
            color=COLORS[method],
            alpha=0.14,
            linewidth=0,
        )
        op_x, op_y = points[method]
        axis.scatter(
            [op_x], [op_y], marker="*", s=130, color=COLORS[method],
            edgecolor="white", linewidth=0.8, zorder=5,
        )
    axis.set_xlabel("False alarms per hour")
    axis.set_ylabel("Event sensitivity")
    axis.set_xlim(left=0.0)
    axis.set_ylim(0.0, 1.03)
    axis.legend(frameon=False, loc="lower right")
    axis.text(
        0.02, 0.98,
        "Curves: common threshold sweep\nStars: validation-selected per-run thresholds",
        transform=axis.transAxes, va="top", ha="left", fontsize=8.5, color="#444444",
    )
    style_axis(axis)
    fig.tight_layout()
    save_figure(fig, output_dir, "event_froc_original_vs_latest")


def plot_normal_residuals(role_frame: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 5.0))
    positions: list[float] = []
    groups: list[np.ndarray] = []
    colors: list[str] = []
    centers = np.arange(3, dtype=float)
    for role_index, role in enumerate((4, 5, 0)):
        for method_index, method in enumerate(("original", "latest")):
            positions.append(centers[role_index] + (-0.20 if method_index == 0 else 0.20))
            groups.append(
                role_frame[(role_frame.role == role) & (role_frame.method == method)][
                    "residual_abs_median"
                ].to_numpy()
            )
            colors.append(COLORS[method])
    draw_box_scatter(axis, groups, positions, colors)
    axis.set_xticks(centers, ["Train-N\n(role 4)", "Val-N\n(role 5)", "Test-N\n(role 0)"])
    axis.set_ylabel(r"Run-level median $|r|$")
    handles = [
        plt.Line2D([0], [0], color=COLORS[method], linewidth=6, alpha=0.5)
        for method in ("original", "latest")
    ]
    axis.legend(handles, [METHOD_LABELS[m] for m in ("original", "latest")], frameon=False)
    style_axis(axis)
    fig.tight_layout()
    save_figure(fig, output_dir, "normal_residual_distributions")


def plot_error_direction(role_frame: pd.DataFrame, output_dir: Path) -> None:
    metrics = (
        ("reconstruction_mae", "Reconstruction MAE", None),
        ("residual_abs_median", r"Median $|r|$", None),
        ("q_signed_mean", r"Mean signed $q$", 0.0),
        ("q_positive_fraction", r"Positive fraction of $q$", 0.5),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    for axis, (metric, ylabel, reference) in zip(axes.flat, metrics):
        groups: list[np.ndarray] = []
        positions: list[float] = []
        colors: list[str] = []
        for class_index, role in enumerate((0, 1)):
            for method_index, method in enumerate(("original", "latest")):
                values = role_frame[(role_frame.role == role) & (role_frame.method == method)][metric]
                groups.append(values.to_numpy())
                positions.append(class_index + (-0.20 if method_index == 0 else 0.20))
                colors.append(COLORS[method])
        draw_box_scatter(axis, groups, positions, colors)
        axis.set_xticks([0, 1], ["Non-FoG", "FoG"])
        axis.set_ylabel(ylabel)
        if reference is not None:
            axis.axhline(reference, color="#777777", linestyle="--", linewidth=0.9)
        style_axis(axis)
    handles = [
        plt.Line2D([0], [0], color=COLORS[method], linewidth=6, alpha=0.5)
        for method in ("original", "latest")
    ]
    fig.legend(
        handles,
        [METHOD_LABELS[m] for m in ("original", "latest")],
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_dir, "test_reconstruction_error_and_residual_direction")


def short_channel_labels(channel_names: Iterable[str]) -> list[str]:
    labels = []
    for name in channel_names:
        suffix = str(name).split("_")[-1]
        labels.append(suffix.upper())
    return labels


def latest_subject_event_metrics(root: Path) -> pd.DataFrame:
    return pd.read_csv(
        root / "latest_event_metrics_1s" / "raw_tcn_subject_summary_unrounded.csv"
    )


def plot_channel_analysis(
    dataset: Any,
    channel_frame: pd.DataFrame,
    roots: dict[str, Path],
    output_dir: Path,
) -> pd.DataFrame:
    aggregate = (
        channel_frame.groupby(["method", "subject", "channel_index", "channel_name"], as_index=False)
        .agg(
            sigma_median=("sigma", "median"),
            sigma_mean=("sigma", "mean"),
            fp_residual_abs_median=("fp_residual_abs_median", "median"),
            tn_residual_abs_median=("tn_residual_abs_median", "median"),
            fp_to_tn_residual_ratio=("fp_to_tn_residual_ratio", "median"),
            fp_windows_mean=("false_positive_windows", "mean"),
            tn_windows_mean=("true_negative_windows", "mean"),
            valid_fp_runs=("fp_residual_abs_median", "count"),
        )
    )
    aggregate.to_csv(output_dir / "difficult_subject_channel_summary.csv", index=False)

    event_frames = {}
    for method, root in roots.items():
        frame = latest_subject_event_metrics(root).set_index("subject")
        event_frames[method] = frame

    x = np.arange(dataset.n_channels)
    labels = short_channel_labels(dataset.channel_names)
    fig, axes = plt.subplots(3, 2, figsize=(14.0, 9.2), sharex=True)
    for row_index, subject in enumerate(DIFFICULT_SUBJECTS):
        scale_axis, residual_axis = axes[row_index]
        for method in ("original", "latest"):
            values = aggregate[
                (aggregate.subject == subject) & (aggregate.method == method)
            ].sort_values("channel_index")
            scale_axis.plot(
                x,
                values.sigma_median,
                color=COLORS[method],
                linewidth=1.6,
                marker="o",
                markersize=2.7,
                label=METHOD_LABELS[method],
            )
            ratios = values.fp_to_tn_residual_ratio.to_numpy(dtype=float)
            if np.any(np.isfinite(ratios)):
                residual_axis.plot(
                    x,
                    ratios,
                    color=COLORS[method],
                    linewidth=1.6,
                    marker="o",
                    markersize=2.7,
                    label=METHOD_LABELS[method],
                )
            else:
                residual_axis.text(
                    0.98,
                    0.88 if method == "original" else 0.76,
                    f"{METHOD_LABELS[method]}: no FP windows",
                    transform=residual_axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color=COLORS[method],
                )
        scale_axis.set_ylabel(f"{subject}\nCalibration scale $\\sigma_c$")
        scale_axis.set_yscale("log")
        residual_axis.set_ylabel(f"{subject}\nFP/TN median $|r_c|$ ratio")
        residual_axis.axhline(1.0, color="#666666", linestyle="--", linewidth=0.9)
        old_fa = float(event_frames["original"].loc[subject, "false_alarms_per_hour_mean"])
        new_fa = float(event_frames["latest"].loc[subject, "false_alarms_per_hour_mean"])
        residual_axis.set_title(f"FA/h: original {old_fa:.1f}, latest {new_fa:.1f}", fontsize=9)
        for axis in (scale_axis, residual_axis):
            for boundary in (5.5, 11.5, 17.5, 23.5):
                axis.axvline(boundary, color="#BBBBBB", linewidth=0.7)
            style_axis(axis)
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper right")
    axes[-1, 0].set_xticks(x, labels, rotation=90, fontsize=7)
    axes[-1, 1].set_xticks(x, labels, rotation=90, fontsize=7)
    axes[-1, 0].set_xlabel("Channels grouped as lumbar, left/right ankle, left/right foot")
    axes[-1, 1].set_xlabel("Channels grouped as lumbar, left/right ankle, left/right foot")
    fig.tight_layout()
    save_figure(fig, output_dir, "p03_p06_p08_channel_scale_and_false_alarm")
    return aggregate


def summary_table(role_frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = (
        role_frame.groupby(["method", "role", "role_name"], as_index=False)
        .agg(
            runs=("subject", "size"),
            reconstruction_mae_mean=("reconstruction_mae", "mean"),
            reconstruction_mae_sd=("reconstruction_mae", lambda values: values.std(ddof=0)),
            residual_abs_median_mean=("residual_abs_median", "mean"),
            residual_abs_median_sd=("residual_abs_median", lambda values: values.std(ddof=0)),
            q_signed_mean=("q_signed_mean", "mean"),
            q_positive_fraction_mean=("q_positive_fraction", "mean"),
        )
    )
    summary.to_csv(output_dir / "residual_role_summary.csv", index=False)
    return summary


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.original_root = args.original_root.resolve()
    args.latest_root = args.latest_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    roots = {"original": args.original_root, "latest": args.latest_root}
    for root in roots.values():
        require_complete(root)
    dataset = DaphnetDataset.load(args.data_dir)
    if dataset.sampling_rate_hz != 64 or dataset.n_channels != 30:
        raise AssertionError("expected the 64-Hz, 30-channel processed_NBM_Exp dataset")

    froc_runs, froc_summary = collect_froc(dataset, roots, args.output_dir)
    plot_froc(froc_summary, roots, args.output_dir)
    role_frame, channel_frame = collect_residual_diagnostics(
        dataset, args.data_dir, roots, args.output_dir, args.batch_size
    )
    residual_summary = summary_table(role_frame, args.output_dir)
    plot_normal_residuals(role_frame, args.output_dir)
    plot_error_direction(role_frame, args.output_dir)
    channel_summary = plot_channel_analysis(
        dataset, channel_frame, roots, args.output_dir
    )

    generated = sorted(
        path for path in args.output_dir.iterdir() if path.is_file()
    )
    metadata = {
        "schema": "private_gru_nbm_original_vs_step_diagnostics.v1",
        "data_dir": str(args.data_dir),
        "experiments": {method: str(root) for method, root in roots.items()},
        "subjects": list(SUBJECTS),
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "run_count_per_method": 120,
        "normal_roles": {"Train-N": 4, "Val-N": 5, "Test-N": 0},
        "error_domain": "Robust-scaled and per-window/per-axis centered signal domain",
        "residual_definitions": {
            "error": "e=X-Xhat",
            "pre_centered_standardized_error": "q=clip(e/(sigma+1e-6),-12,12)",
            "scheme_c_signed_residual": "r=q-mean_t(q)",
            "direction": "reported from q because temporal centering forces mean_t(r)=0",
        },
        "froc": {
            "thresholds": "0.00 to 1.00 inclusive, step 0.01",
            "event_sensitivity": "reference FoG event hit by at least one positive pure-FoG test window",
            "false_alarm": "positive pure-Non-FoG test-window supports merged within record at gap <=1 s",
            "aggregation": "3-fold mean within subject/seed, subject macro within seed, mean and population SD over 5 seeds",
            "curve_thresholding": "one common threshold at each curve point",
            "star_thresholding": "validation-selected threshold independently frozen for each run",
        },
        "residual_distribution_unit": "one summary point per subject/fold/seed run",
        "difficult_subjects": list(DIFFICULT_SUBJECTS),
        "tables": {
            "froc_rows": int(len(froc_runs)),
            "froc_summary_rows": int(len(froc_summary)),
            "residual_run_rows": int(len(role_frame)),
            "residual_summary_rows": int(len(residual_summary)),
            "channel_run_rows": int(len(channel_frame)),
            "channel_summary_rows": int(len(channel_summary)),
        },
        "outputs": {
            path.name: sha256_file(path)
            for path in generated
        },
    }
    atomic_json_dump(metadata, args.output_dir / "analysis_contract.json")
    print(f"COMPLETE output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
