"""Render a transparently post-hoc, best-looking GRU-NBM mechanism case.

This script is deliberately separate from the pre-specified representative-case
analysis.  It searches only the performance-perfect P08 frozen runs and ranks
their correctly detected permanent-test FoG events by local visual-mechanism
clarity.  The resulting figure must not be presented as unbiased evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.resume import atomic_json_dump, sha256_file
from scripts import plot_private_gru_nbm_residual_mechanism_case as base
from scripts import run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn as worker
from scripts import run_all_dataset_processed_nbm_exp_within_subject_raw_tcn as raw_base


DEFAULT_EXPERIMENT_ROOT = base.DEFAULT_EXPERIMENT_ROOT
DEFAULT_DATA_DIR = base.DEFAULT_DATA_DIR
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "figures"
    / "private_gru_nbm_residual_mechanism_exploratory_best_case"
)
PERFECT_TOLERANCE = 1e-12


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty candidate table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def perfect_runs(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    required = ("sensitivity", "precision", "specificity", "pr_auc")
    candidates: list[dict[str, Any]] = []
    for row in rows:
        metrics = {name: float(row[name]) for name in required}
        event_sensitivity = float(row["event_sensitivity"])
        false_alarms_per_hour = float(row["false_alarms_per_hour"])
        if (
            all(abs(metrics[name] - 1.0) <= PERFECT_TOLERANCE for name in required)
            and abs(event_sensitivity - 1.0) <= PERFECT_TOLERANCE
            and abs(false_alarms_per_hour) <= PERFECT_TOLERANCE
        ):
            candidates.append(
                {
                    "subject": str(row["subject"]),
                    "fold": int(row["fold"]),
                    "seed": int(row["seed"]),
                    "run_ap": metrics["pr_auc"],
                    "threshold": float(row["threshold"]),
                }
            )
    if not candidates:
        raise RuntimeError("no performance-perfect run is available for exploratory search")
    return sorted(candidates, key=lambda row: (row["subject"], row["fold"], row["seed"]))


def context_arrays(
    dataset: Any,
    event: dict[str, Any],
    nbm: Any,
    tcn: Any,
    scaler: Any,
    sigma: np.ndarray,
    threshold: float,
    device: torch.device,
    batch_size: int,
    context_before_sec: float,
    context_after_sec: float,
) -> dict[str, Any]:
    record = dataset.records[int(event["record_index"])]
    sampling_rate = int(dataset.sampling_rate_hz)
    display_start = max(
        0,
        int(event["start_index"] - round(context_before_sec * sampling_rate)),
    )
    display_end = min(
        len(record.y),
        int(event["end_index_exclusive"] + round(context_after_sec * sampling_rate)),
    )
    all_starts = np.arange(
        0,
        len(record.y) - worker.WINDOW_SAMPLES + 1,
        base.STRIDE_SAMPLES,
        dtype=np.int32,
    )
    overlap = (all_starts < display_end) & (
        all_starts + worker.WINDOW_SAMPLES > display_start
    )
    starts = all_starts[overlap]
    raw_windows = np.stack(
        [record.x[start : start + worker.WINDOW_SAMPLES] for start in starts]
    ).astype(np.float32)
    (
        _,
        _,
        residual_windows,
        absolute_windows,
        delta_windows,
        reconstructed_windows,
    ) = base.residual_components(
        nbm, scaler, sigma, raw_windows, device, batch_size
    )
    features = np.concatenate(
        (residual_windows, absolute_windows, delta_windows), axis=2
    ).transpose(0, 2, 1)
    _, probabilities = worker.predict(
        tcn,
        np.ascontiguousarray(features, dtype=np.float32),
        np.zeros(len(features), dtype=np.int8),
        device,
        batch_size,
    )
    centers = starts + worker.WINDOW_SAMPLES // 2
    center_mask = (centers >= display_start) & (centers < display_end)
    prediction_by_window = (probabilities >= threshold).astype(np.int8)

    reconstructed_raw, overlap_counts = base.overlap_average(
        reconstructed_windows, starts, display_start, display_end
    )
    residual, _ = base.overlap_average(
        residual_windows, starts, display_start, display_end
    )
    absolute, _ = base.overlap_average(
        absolute_windows, starts, display_start, display_end
    )
    delta, _ = base.overlap_average(
        delta_windows, starts, display_start, display_end
    )
    observed_raw = record.x[display_start:display_end]
    ground_truth = record.y[display_start:display_end].astype(np.int8)
    time = (
        np.arange(display_start, display_end) - int(event["start_index"])
    ) / sampling_rate

    prediction_strip = np.zeros(len(time), dtype=np.int8)
    half_update = base.STRIDE_SAMPLES // 2
    for center, value in zip(centers, prediction_by_window):
        left = max(display_start, int(center - half_update))
        right = min(display_end, int(center + half_update))
        if left < right:
            prediction_strip[left - display_start : right - display_start] = value

    raw_fog_intervals = [
        (start, end)
        for start, end in base.boolean_runs(record.y == 1)
        if max(start, display_start) < min(end, display_end)
    ]
    fog_intervals = [
        (
            (max(start, display_start) - int(event["start_index"])) / sampling_rate,
            (min(end, display_end) - int(event["start_index"])) / sampling_rate,
        )
        for start, end in raw_fog_intervals
    ]
    return {
        "record": record,
        "display_start": display_start,
        "display_end": display_end,
        "time": time,
        "observed_raw": observed_raw,
        "reconstructed_raw": reconstructed_raw,
        "residual": residual,
        "absolute": absolute,
        "delta": delta,
        "probability_time": (
            centers[center_mask] - int(event["start_index"])
        ) / sampling_rate,
        "probability": probabilities[center_mask],
        "probability_centers": centers[center_mask],
        "prediction_strip": prediction_strip,
        "ground_truth": ground_truth,
        "fog_intervals": fog_intervals,
        "selected_interval": (
            0.0,
            int(event["duration_samples"]) / sampling_rate,
        ),
        "overlap_counts": overlap_counts,
    }


def score_case(
    arrays: dict[str, Any],
    event: dict[str, Any],
    scaler: Any,
) -> dict[str, float]:
    display_start = int(arrays["display_start"])
    start = int(event["start_index"]) - display_start
    end = int(event["end_index_exclusive"]) - display_start
    selected_mask = np.zeros(len(arrays["time"]), dtype=bool)
    selected_mask[max(0, start) : min(len(selected_mask), end)] = True
    nonfog_mask = arrays["ground_truth"] == 0
    if not selected_mask.any() or not nonfog_mask.any():
        raise ValueError("candidate context lacks selected-event or Non-FoG samples")

    strip = arrays["prediction_strip"]
    selected_sensitivity = float(np.mean(strip[selected_mask] == 1))
    local_specificity = float(np.mean(strip[nonfog_mask] == 0))
    other_fog_samples = int(np.sum((arrays["ground_truth"] == 1) & ~selected_mask))

    centers = arrays["probability_centers"]
    event_center = (centers >= int(event["start_index"])) & (
        centers < int(event["end_index_exclusive"])
    )
    center_labels = arrays["record"].y[centers]
    nonfog_center = center_labels == 0
    probability = arrays["probability"]
    if not event_center.any() or not nonfog_center.any():
        probability_margin = float("-inf")
        probability_hard_margin = float("-inf")
    else:
        probability_margin = float(
            probability[event_center].mean() - probability[nonfog_center].mean()
        )
        probability_hard_margin = float(
            probability[event_center].min() - probability[nonfog_center].max()
        )

    event_residual = float(np.mean(arrays["absolute"][selected_mask]))
    nonfog_residual = float(np.mean(arrays["absolute"][nonfog_mask]))
    residual_contrast = event_residual / max(nonfog_residual, 1e-12)

    channels = np.asarray(base.FIXED_CHANNEL_INDICES, dtype=int)
    scale = np.asarray(scaler.iqr, dtype=np.float64)[channels] + float(scaler.epsilon)
    nonfog_difference = np.abs(
        arrays["observed_raw"][nonfog_mask][:, channels]
        - arrays["reconstructed_raw"][nonfog_mask][:, channels]
    )
    nonfog_reconstruction_nmae = float(np.mean(nonfog_difference / scale[None, :]))
    return {
        "selected_event_sensitivity": selected_sensitivity,
        "local_nonfog_specificity": local_specificity,
        "other_fog_seconds": other_fog_samples / 64.0,
        "probability_margin": probability_margin,
        "probability_hard_margin": probability_hard_margin,
        "residual_contrast": residual_contrast,
        "nonfog_reconstruction_nmae": nonfog_reconstruction_nmae,
    }


def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    # Prefer a single isolated event and exact local binary localization first;
    # use probability/residual clarity and reconstruction proximity as tie-breaks.
    return (
        row["other_fog_seconds"] > 0.0,
        -row["local_nonfog_specificity"],
        -row["selected_event_sensitivity"],
        -row["probability_hard_margin"],
        -row["probability_margin"],
        -row["residual_contrast"],
        row["nonfog_reconstruction_nmae"],
        row["subject"],
        row["fold"],
        row["seed"],
        row["record_id"],
        row["start_index"],
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    experiment_root = args.experiment_root.resolve()
    output_dir = args.output_dir.resolve()
    if args.batch_size <= 0 or args.dpi <= 0:
        raise ValueError("batch size and DPI must be positive")
    if args.context_before_sec < 4.0 or args.context_after_sec < 4.0:
        raise ValueError("at least 4 s of context is required on each side")

    dataset = DaphnetDataset.load(data_dir)
    device = worker.resolve_device(args.device)
    candidates: list[dict[str, Any]] = []
    cached: dict[tuple[str, int, int, str, int], dict[str, Any]] = {}
    run_rows = perfect_runs(read_csv(experiment_root / "run_metrics.csv"))

    for run in run_rows:
        subject = str(run["subject"])
        fold = int(run["fold"])
        seed = int(run["seed"])
        run_dir = worker.run_dir(experiment_root, subject, fold, seed)
        rows = raw_base.load_subject_rows(data_dir, dataset, subject, fold)
        test_rows = rows.take_role(0, 1)
        saved_predictions = read_csv(run_dir / "test_predictions.csv")
        prediction_by_window = {
            row["window_id"]: int(row["y_pred"]) for row in saved_predictions
        }
        events = [
            event
            for event in base.event_candidates(dataset, test_rows, prediction_by_window)
            if event["detected"]
        ]
        if not events:
            continue
        nbm, tcn, scaler, sigma, threshold, frozen = base.load_frozen_models(
            run_dir, device
        )
        for event in events:
            arrays = context_arrays(
                dataset,
                event,
                nbm,
                tcn,
                scaler,
                sigma,
                threshold,
                device,
                args.batch_size,
                args.context_before_sec,
                args.context_after_sec,
            )
            score = score_case(arrays, event, scaler)
            key = (
                subject,
                fold,
                seed,
                str(event["record_id"]),
                int(event["start_index"]),
            )
            cached[key] = {
                "arrays": arrays,
                "event": event,
                "run_dir": run_dir,
                "rows": rows,
                "scaler": scaler,
                "sigma": sigma,
                "threshold": threshold,
                "frozen": frozen,
            }
            candidates.append(
                {
                    "subject": subject,
                    "fold": fold,
                    "seed": seed,
                    "run_ap": float(run["run_ap"]),
                    "record_id": str(event["record_id"]),
                    "start_index": int(event["start_index"]),
                    "end_index_exclusive": int(event["end_index_exclusive"]),
                    "duration_sec": float(event["duration_sec"]),
                    **score,
                }
            )
    if not candidates:
        raise RuntimeError("no correctly detected event was scored")
    ranked = sorted(candidates, key=rank_key)
    for index, row in enumerate(ranked, start=1):
        row["exploratory_rank"] = index
    selected = ranked[0]
    selected_key = (
        selected["subject"],
        int(selected["fold"]),
        int(selected["seed"]),
        selected["record_id"],
        int(selected["start_index"]),
    )
    item = cached[selected_key]
    arrays = item["arrays"]
    event = item["event"]
    run_dir = item["run_dir"]
    rows = item["rows"]
    scaler = item["scaler"]
    sigma = item["sigma"]
    threshold = float(item["threshold"])
    frozen = item["frozen"]

    training_raw = raw_base.raw_windows(dataset, rows.take_role(6, 7))
    _, _, training_r, _, training_delta, _ = base.residual_components(
        base.load_frozen_models(run_dir, device)[0],
        scaler,
        sigma,
        training_raw,
        device,
        args.batch_size,
    )
    residual_vmax = float(np.percentile(np.abs(training_r), 99.0))
    delta_vmax = float(np.percentile(np.abs(training_delta), 99.0))
    schema = read_json(data_dir / "schema.json")
    channel_units = [str(row["unit"]) for row in schema["channels"]]

    metadata = {
        "schema": "private_gru_nbm_residual_mechanism_exploratory_best_case.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "figure_title": "Exploratory best-looking GRU-NBM mechanism case",
        "figure_stem": "private_gru_nbm_residual_mechanism_exploratory_best_case",
        "post_hoc_selected": True,
        "eligible_run_rule": (
            "window sensitivity, precision, specificity, AP, event sensitivity all "
            "equal 1 and FA/h equal 0"
        ),
        "ranking_rule": (
            "isolated event; higher local Non-FoG specificity; higher selected-event "
            "sensitivity; higher hard and mean probability margins; higher residual "
            "contrast; lower fixed-channel Non-FoG normalized reconstruction MAE"
        ),
        "paper_use_warning": (
            "post-hoc visualization only; must not be presented as an unbiased "
            "representative case or used as primary efficacy evidence"
        ),
        "eligible_runs": len(run_rows),
        "scored_events": len(ranked),
        "selected_subject": selected["subject"],
        "selected_fold": int(selected["fold"]),
        "selected_seed": int(selected["seed"]),
        "selected_run_ap": float(selected["run_ap"]),
        "selected_event": event,
        "selected_scores": {
            key: value
            for key, value in selected.items()
            if key
            not in {
                "subject",
                "fold",
                "seed",
                "run_ap",
                "record_id",
                "start_index",
                "end_index_exclusive",
                "duration_sec",
                "exploratory_rank",
            }
        },
        "record_id": selected["record_id"],
        "sampling_rate_hz": int(dataset.sampling_rate_hz),
        "window_samples": worker.WINDOW_SAMPLES,
        "stride_samples": base.STRIDE_SAMPLES,
        "threshold": threshold,
        "residual_vmax_train_p99": residual_vmax,
        "delta_vmax_train_p99": delta_vmax,
        "fixed_display_channels": [
            {
                "index": index,
                "name": schema["channels"][index]["name"],
                "label": label,
                "unit": channel_units[index],
            }
            for index, label in zip(
                base.FIXED_CHANNEL_INDICES, base.FIXED_CHANNEL_LABELS
            )
        ],
        "overlap_add_visualization": (
            "window-level values averaged over overlapping canonical windows for display"
        ),
        "frozen_id": frozen["frozen_id"],
        "source_artifacts": {
            "run_directory": str(run_dir.resolve()),
            "nbm_checkpoint_sha256": sha256_file(
                run_dir / "checkpoints" / "gru_nbm_best.pt"
            ),
            "tcn_checkpoint_sha256": sha256_file(
                run_dir / "checkpoints" / "tcn.pt"
            ),
        },
    }

    base.plot_figure(
        output_dir,
        args.dpi,
        metadata,
        arrays["time"],
        arrays["observed_raw"],
        arrays["reconstructed_raw"],
        arrays["residual"],
        arrays["absolute"],
        arrays["delta"],
        arrays["probability_time"],
        arrays["probability"],
        threshold,
        arrays["prediction_strip"],
        arrays["ground_truth"],
        arrays["fog_intervals"],
        arrays["selected_interval"],
        residual_vmax,
        delta_vmax,
        channel_units,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(metadata, output_dir / "case_selection_audit.json")
    write_csv(output_dir / "exploratory_candidate_ranking.csv", ranked)
    np.savez_compressed(
        output_dir / "figure_data.npz",
        time_sec=arrays["time"].astype(np.float32),
        observed_raw=arrays["observed_raw"].astype(np.float32),
        reconstructed_raw=arrays["reconstructed_raw"].astype(np.float32),
        residual=arrays["residual"].astype(np.float32),
        absolute_residual=arrays["absolute"].astype(np.float32),
        delta_residual=arrays["delta"].astype(np.float32),
        probability_time_sec=arrays["probability_time"].astype(np.float32),
        fog_probability=arrays["probability"].astype(np.float32),
        threshold=np.asarray(threshold, dtype=np.float32),
        ground_truth=arrays["ground_truth"].astype(np.int8),
        prediction_strip=arrays["prediction_strip"].astype(np.int8),
        channel_names=np.asarray(dataset.channel_names),
    )
    caption = (
        "**Exploratory post-hoc GRU-NBM residual mechanism example.** "
        "This case was selected after examining frozen test-run outputs to maximize "
        "local visual-mechanism clarity and is therefore not an unbiased representative "
        "case. The full ranking and selection criteria are supplied with the figure."
    )
    (output_dir / "figure_caption.md").write_text(caption + "\n", encoding="utf-8")
    print(
        f"COMPLETE exploratory_rank=1/{len(ranked)} subject={selected['subject']} "
        f"fold={selected['fold']} seed={selected['seed']} "
        f"event={selected['record_id']}:{selected['start_index']}:"
        f"{selected['end_index_exclusive']} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
