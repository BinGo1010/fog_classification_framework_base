#!/usr/bin/env python3
"""Extract reproducible S01/S02/S03 waist-IMU Z-axis slices for the NBM figure."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    fit_scaler_unique_role4_points,
    load_fold_rows,
    raw_windows,
)

FS = 64
WINDOW_SAMPLES = 128
TRUNK_CHANNELS = (6, 7, 8)
WAIST_Z_CHANNEL = 7  # canonical schema: trunk_acc_vertical
SUBJECTS = ("S01", "S02", "S03")
CONDITIONS = ("Original", "Gaussian noise", "Time mask")


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
        / "figures"
        / "nbm_paper_augmentation_axisless_panels",
    )
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--visualization-gaussian-std", type=float, default=0.30)
    parser.add_argument("--training-gaussian-std", type=float, default=0.04)
    parser.add_argument("--mask-min-samples", type=int, default=4)
    parser.add_argument("--mask-max-samples", type=int, default=8)
    return parser.parse_args()


def parse_window_bounds(window_id: str) -> tuple[int, int]:
    parts = window_id.split(":")
    if len(parts) != 3:
        raise ValueError(f"unexpected window id: {window_id}")
    start, end = int(parts[1]), int(parts[2])
    if end - start != WINDOW_SAMPLES:
        raise ValueError(f"window is not {WINDOW_SAMPLES} samples: {window_id}")
    return start, end


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DaphnetDataset.load(data_dir)
    records = {record.record_id: record for record in dataset.records}
    role4 = load_fold_rows(data_dir, args.fold).take_role(4)
    scaler, scaler_points = fit_scaler_unique_role4_points(records, role4)
    centered = scaler.transform(raw_windows(records, role4))
    centered -= centered.mean(axis=1, keepdims=True)

    trunk = centered[:, :, TRUNK_CHANNELS]
    trunk_energy = np.sqrt(np.mean(np.square(trunk), axis=(1, 2)))
    waist_z_std = np.std(centered[:, :, WAIST_Z_CHANNEL], axis=1)
    requested_subject = np.isin(role4.subject_id, np.asarray(SUBJECTS))
    valid = (
        requested_subject
        & np.isfinite(trunk_energy)
        & np.isfinite(waist_z_std)
        & (waist_z_std > 0.05)
    )
    pooled_candidates = np.flatnonzero(valid)
    if pooled_candidates.size == 0:
        raise RuntimeError("no valid S01/S02/S03 role-4 waist-Z windows are available")
    target_energy = float(np.median(trunk_energy[pooled_candidates]))

    selected: list[int] = []
    candidate_counts: dict[str, int] = {}
    for subject in SUBJECTS:
        candidates = np.flatnonzero((role4.subject_id == subject) & valid)
        if candidates.size == 0:
            raise RuntimeError(f"no valid role-4 waist-Z window for {subject}")
        candidate_counts[subject] = int(candidates.size)
        choice = int(candidates[np.argmin(np.abs(trunk_energy[candidates] - target_energy))])
        selected.append(choice)

    clean_all = centered[selected].astype(np.float32)
    rng = np.random.default_rng(args.seed)
    gaussian_s02 = clean_all[1] + rng.normal(
        0.0,
        args.visualization_gaussian_std,
        size=clean_all[1].shape,
    ).astype(np.float32)
    mask_length = int(rng.integers(args.mask_min_samples, args.mask_max_samples + 1))
    mask_start = int(rng.integers(0, WINDOW_SAMPLES - mask_length + 1))
    mask_end = mask_start + mask_length
    masked_s03 = clean_all[2].copy()
    masked_s03[mask_start:mask_end, :] = 0.0

    displayed = (
        clean_all[0, :, WAIST_Z_CHANNEL],
        gaussian_s02[:, WAIST_Z_CHANNEL],
        masked_s03[:, WAIST_Z_CHANNEL],
    )
    clean_z = clean_all[:, :, WAIST_Z_CHANNEL]
    time = np.arange(WINDOW_SAMPLES, dtype=np.float64) / FS

    source_path = output_dir / "source_data.csv"
    fieldnames = [
        "panel",
        "subject_id",
        "condition",
        "record_id",
        "window_id",
        "sample_index_in_segment",
        "segment_time_s",
        "plot_time_s",
        "trunk_energy_all_axes",
        "clean_waist_z",
        "displayed_waist_z",
        "augmentation_delta_waist_z",
        "mask_active",
    ]
    with source_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for panel, (subject, condition, index) in enumerate(
            zip(SUBJECTS, CONDITIONS, selected), start=1
        ):
            window_id = str(role4.window_id[index])
            window_start, _ = parse_window_bounds(window_id)
            for sample in range(WINDOW_SAMPLES):
                clean_value = float(clean_z[panel - 1, sample])
                displayed_value = float(displayed[panel - 1][sample])
                writer.writerow(
                    {
                        "panel": panel,
                        "subject_id": subject,
                        "condition": condition,
                        "record_id": str(role4.record_id[index]),
                        "window_id": window_id,
                        "sample_index_in_segment": window_start + sample,
                        "segment_time_s": (window_start + sample) / FS,
                        "plot_time_s": float(time[sample]),
                        "trunk_energy_all_axes": float(trunk_energy[index]),
                        "clean_waist_z": clean_value,
                        "displayed_waist_z": displayed_value,
                        "augmentation_delta_waist_z": displayed_value - clean_value,
                        "mask_active": int(panel == 3 and mask_start <= sample < mask_end),
                    }
                )

    windows = []
    for panel, (subject, condition, index) in enumerate(
        zip(SUBJECTS, CONDITIONS, selected), start=1
    ):
        window_id = str(role4.window_id[index])
        start, end = parse_window_bounds(window_id)
        windows.append(
            {
                "panel": panel,
                "subject_id": subject,
                "condition": condition,
                "selected_index_within_role4": index,
                "record_id": str(role4.record_id[index]),
                "window_id": window_id,
                "start_sample": start,
                "end_sample_exclusive": end,
                "start_time_s": start / FS,
                "end_time_s_exclusive": end / FS,
                "trunk_energy_all_axes": float(trunk_energy[index]),
                "waist_z_std": float(waist_z_std[index]),
            }
        )
    windows[2].update(
        {
            "mask_start_sample_in_window": mask_start,
            "mask_end_sample_in_window_exclusive": mask_end,
            "mask_start_segment_time_s": windows[2]["start_time_s"] + mask_start / FS,
            "mask_end_segment_time_s_exclusive": windows[2]["start_time_s"]
            + mask_end / FS,
        }
    )

    metadata = {
        "backend": "Python/numpy",
        "source_dataset": "Daphnet processed_NBM",
        "fold": args.fold,
        "role": 4,
        "role_definition": "clean non-FoG used to train the NBM",
        "sampling_rate_hz": FS,
        "window_samples": WINDOW_SAMPLES,
        "window_duration_s": WINDOW_SAMPLES / FS,
        "displayed_channel": {
            "index": WAIST_Z_CHANNEL,
            "canonical_name": "trunk_acc_vertical",
            "user_label": "waist IMU Z-axis acceleration",
        },
        "scaling": "fold-0 role-4 RobustScaler, followed by per-window/per-channel mean centering",
        "scaler_unique_raw_points": scaler_points,
        "candidate_filter": "finite trunk energy and waist-Z std > 0.05",
        "candidate_counts": candidate_counts,
        "selection_rule": (
            "For each of S01, S02, and S03, select the valid role-4 window nearest "
            "the pooled median three-axis trunk RMS energy across all three subjects."
        ),
        "pooled_target_trunk_energy": target_energy,
        "seed": args.seed,
        "visualization_gaussian_std": args.visualization_gaussian_std,
        "actual_training_gaussian_std": args.training_gaussian_std,
        "mask_length_range_samples": [args.mask_min_samples, args.mask_max_samples],
        "windows": windows,
    }
    (output_dir / "slice_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
