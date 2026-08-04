#!/usr/bin/env python
"""Run the transparent within-S09 GRU-H200 residual TCN-M experiment.

This is the S09 counterpart of ``run_daphnet_s01_gru_h200_tcnm.py``.  Model,
window, optimiser, early-stopping, threshold, and plotting rules are unchanged.
Only the chronological record/range split is adapted to S09's five records.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_daphnet_s01_gru_h200_tcnm as core  # noqa: E402
from cnbr_fog.data import DaphnetDataset, Record, RobustChannelScaler, WindowTable  # noqa: E402
from cnbr_fog.resume import (  # noqa: E402
    atomic_json_dump,
    atomic_npz_save,
    canonical_fingerprint,
    dataset_fingerprint,
    sha256_file,
)


SPLIT_CONFIGS = {
    "S01": {
        "train_records": ("S01_seg000",),
        "cut_record": "S01_seg001",
        "cut_sample": 50_944,
        "test_record": "S01_seg002",
        "ignored_records": (),
    },
    "S02": {
        "train_records": (),
        "cut_record": "S02_seg000",
        "cut_sample": 17_152,
        "test_record": "S02_seg001",
        "ignored_records": (),
    },
    "S03": {
        "train_records": ("S03_seg000",),
        "cut_record": "S03_seg001",
        "cut_sample": 8_576,
        "test_record": "S03_seg002",
        "ignored_records": ("S03_seg003",),
    },
    "S05": {
        "train_records": (
            "S05_seg000",
            "S05_seg001",
            "S05_seg002",
            "S05_seg003",
        ),
        "cut_record": "S05_seg004",
        "cut_sample": 4_736,
        "test_record": "S05_seg005",
        "ignored_records": (),
    },
    "S06": {
        "train_records": ("S06_seg000",),
        "cut_record": "S06_seg001",
        "cut_sample": 6_144,
        "test_record": "S06_seg002",
        "ignored_records": ("S06_seg003", "S06_seg004"),
    },
    "S07": {
        "train_records": (),
        "cut_record": "S07_seg000",
        "cut_sample": 51_968,
        "test_record": "S07_seg001",
        "ignored_records": (),
    },
    "S08": {
        "train_records": ("S08_seg000", "S08_seg001"),
        "cut_record": "S08_seg002",
        "cut_sample": 1_920,
        "test_record": "S08_seg003",
        "ignored_records": (),
    },
    "S09": {
        "train_records": ("S09_seg000", "S09_seg001", "S09_seg002"),
        "cut_record": "S09_seg003",
        "cut_sample": 15_552,
        "test_record": "S09_seg004",
        "ignored_records": (),
    },
}

EXPERIMENT_VERSION = "daphnet_s09_gru_h200_tcnm.v1"
SUBJECT_ID = "S09"
TRAIN_RECORDS = SPLIT_CONFIGS[SUBJECT_ID]["train_records"]
CUT_RECORD = SPLIT_CONFIGS[SUBJECT_ID]["cut_record"]
TEST_RECORD = SPLIT_CONFIGS[SUBJECT_ID]["test_record"]
CUT_SAMPLE = SPLIT_CONFIGS[SUBJECT_ID]["cut_sample"]
IGNORED_RECORDS = SPLIT_CONFIGS[SUBJECT_ID]["ignored_records"]
EXPECTED_RECORDS = (*TRAIN_RECORDS, CUT_RECORD, TEST_RECORD, *IGNORED_RECORDS)


def configure_subject(subject: str) -> None:
    global EXPERIMENT_VERSION
    global SUBJECT_ID, TRAIN_RECORDS, CUT_RECORD, TEST_RECORD, CUT_SAMPLE
    global IGNORED_RECORDS, EXPECTED_RECORDS

    config = SPLIT_CONFIGS[str(subject)]
    SUBJECT_ID = str(subject)
    TRAIN_RECORDS = tuple(config["train_records"])
    CUT_RECORD = str(config["cut_record"])
    TEST_RECORD = str(config["test_record"])
    CUT_SAMPLE = int(config["cut_sample"])
    IGNORED_RECORDS = tuple(config["ignored_records"])
    EXPECTED_RECORDS = tuple(
        sorted(
            (
                *TRAIN_RECORDS,
                CUT_RECORD,
                TEST_RECORD,
                *IGNORED_RECORDS,
            )
        )
    )
    EXPERIMENT_VERSION = f"daphnet_{SUBJECT_ID.lower()}_gru_h200_tcnm.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Within-subject GRU-H200 standardized-residual TCN-M experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--subject", choices=sorted(SPLIT_CONFIGS), default="S09")
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
        default=REPO_ROOT / "outputs" / "daphnet_s09_gru50_tcnm12_seed42",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--normal-epochs", type=int, default=50)
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
    parser.add_argument("--context-seconds", type=float, default=2.0)
    parser.add_argument("--target-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_geometry(args: argparse.Namespace) -> None:
    values = {
        "context": float(args.context_seconds),
        "target": float(args.target_seconds),
        "stride": float(args.stride_seconds),
    }
    for name, seconds in values.items():
        samples = seconds * core.SAMPLING_RATE_HZ
        if seconds <= 0 or not float(samples).is_integer():
            raise ValueError(
                f"--{name}-seconds must be positive and resolve to whole samples"
            )
    core.CONTEXT_SAMPLES = int(values["context"] * core.SAMPLING_RATE_HZ)
    core.TARGET_SAMPLES = int(values["target"] * core.SAMPLING_RATE_HZ)
    core.STRIDE_SAMPLES = int(values["stride"] * core.SAMPLING_RATE_HZ)
    if core.LABEL_SAMPLES > core.TARGET_SAMPLES:
        raise ValueError("The final 0.5-second label support exceeds the target")
    if CUT_SAMPLE % core.STRIDE_SAMPLES:
        raise ValueError(
            f"Cut sample {CUT_SAMPLE} is not aligned to stride {core.STRIDE_SAMPLES}"
        )


def load_dataset(data_dir: Path) -> DaphnetDataset:
    full = DaphnetDataset.load(data_dir)
    if full.sampling_rate_hz != core.SAMPLING_RATE_HZ:
        raise ValueError(
            f"Expected {core.SAMPLING_RATE_HZ} Hz, got {full.sampling_rate_hz} Hz"
        )
    if tuple(full.channel_names) != core.EXPECTED_CHANNEL_NAMES:
        raise ValueError(f"Unexpected channels: {full.channel_names}")
    records = [record for record in full.records if record.subject_id == SUBJECT_ID]
    actual_records = tuple(record.record_id for record in records)
    if actual_records != EXPECTED_RECORDS:
        raise ValueError(
            f"{SUBJECT_ID} records changed: expected {EXPECTED_RECORDS}, got "
            f"{actual_records}"
        )
    return DaphnetDataset(
        root=data_dir,
        records=records,
        sampling_rate_hz=full.sampling_rate_hz,
        channel_names=full.channel_names,
    )


def point_ranges(dataset: DaphnetDataset) -> dict[str, list[tuple[Record, int, int]]]:
    by_name = {record.record_id: record for record in dataset.records}
    cut_record = by_name[CUT_RECORD]
    return {
        "train": [
            *( (by_name[name], 0, len(by_name[name].y)) for name in TRAIN_RECORDS ),
            (cut_record, 0, CUT_SAMPLE),
        ],
        "validation": [(cut_record, CUT_SAMPLE, len(cut_record.y))],
        "test": [(by_name[TEST_RECORD], 0, len(by_name[TEST_RECORD].y))],
    }


def make_split(dataset: DaphnetDataset, windows: WindowTable) -> core.SplitBundle:
    lookup = core.record_lookup(dataset)
    train_record_indices = np.asarray(
        [lookup[name] for name in TRAIN_RECORDS], dtype=np.int64
    )
    cut_record_index = lookup[CUT_RECORD]
    test_record_index = lookup[TEST_RECORD]
    train = np.flatnonzero(
        np.isin(windows.record_index, train_record_indices)
        | (
            (windows.record_index == cut_record_index)
            & (windows.target_end <= CUT_SAMPLE)
        )
    )
    validation = np.flatnonzero(
        (windows.record_index == cut_record_index) & (windows.start >= CUT_SAMPLE)
    )
    test = np.flatnonzero(windows.record_index == test_record_index)
    split = core.SplitBundle(train=train, validation=validation, test=test)
    groups = split.as_dict()
    for name, indices in groups.items():
        if not len(indices):
            raise ValueError(f"{name} split is empty")
        counts = np.bincount(windows.label[indices], minlength=2)
        if np.any(counts == 0):
            raise ValueError(f"{name} split lacks a class: {counts.tolist()}")
    sets = {name: set(indices.tolist()) for name, indices in groups.items()}
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if sets[left] & sets[right]:
            raise AssertionError(f"{left}/{right} window indices overlap")
    cut_train = train[windows.record_index[train] == cut_record_index]
    cut_validation = validation[
        windows.record_index[validation] == cut_record_index
    ]
    if int(windows.target_end[cut_train].max()) != CUT_SAMPLE:
        raise AssertionError("Training support does not end at the declared cut")
    if int(windows.start[cut_validation].min()) != CUT_SAMPLE:
        raise AssertionError("Validation support does not start at the declared cut")
    return split


def fit_training_scaler(
    dataset: DaphnetDataset,
) -> tuple[RobustChannelScaler, dict[str, Any]]:
    chunks: list[np.ndarray] = []
    retained_points = 0
    for record, start, end in point_ranges(dataset)["train"]:
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
        clip=core.ROBUST_CLIP,
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
    mask = windows.clean_normal[indices].copy()
    cut_record_index = core.record_lookup(dataset)[CUT_RECORD]
    in_cut_record = windows.record_index[indices] == cut_record_index
    if split_name == "train":
        mask &= (~in_cut_record) | (
            windows.target_end[indices]
            <= CUT_SAMPLE - core.NORMAL_GUARD_SAMPLES
        )
    elif split_name == "validation":
        mask &= (~in_cut_record) | (
            windows.start[indices] >= CUT_SAMPLE + core.NORMAL_GUARD_SAMPLES
        )
    elif split_name != "test":
        raise ValueError(f"Unknown split {split_name!r}")
    return indices[mask]


def point_statistics(dataset: DaphnetDataset) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for split, ranges in point_ranges(dataset).items():
        labels = np.concatenate([record.y[start:end] for record, start, end in ranges])
        valid = np.concatenate(
            [record.valid[start:end] for record, start, end in ranges]
        )
        n_fog = int(np.sum((labels == 1) & valid))
        n_normal = int(np.sum((labels == 0) & valid))
        result[split] = {
            "raw_points": int(len(labels)),
            "valid_points": int(valid.sum()),
            "non_fog_points": n_normal,
            "fog_points": n_fog,
            "fog_percent": 100.0 * n_fog / max(n_normal + n_fog, 1),
            "duration_seconds": len(labels) / core.SAMPLING_RATE_HZ,
            "fog_events": int(
                sum(
                    len(core.boolean_runs(record.y[start:end] == 1))
                    for record, start, end in ranges
                )
            ),
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
    split: core.SplitBundle,
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


def build_protocol(
    args: argparse.Namespace,
    dataset: DaphnetDataset,
    point_stats: dict[str, dict[str, Any]],
    window_stats: dict[str, dict[str, Any]],
    scaler_metadata: dict[str, Any],
    clip_stats: dict[str, dict[str, Any]],
    device,
) -> dict[str, Any]:
    train_record_text = (
        ", ".join(TRAIN_RECORDS) if TRAIN_RECORDS else "no complete earlier record"
    )
    payload = core.build_protocol(
        args,
        dataset,
        point_stats,
        window_stats,
        scaler_metadata,
        clip_stats,
        device,
    )
    payload.update(
        {
            "experiment_version": EXPERIMENT_VERSION,
            "subject": SUBJECT_ID,
            "records": [record.record_id for record in dataset.records],
            "split": {
                "strategy": "chronological record/block split with disjoint raw support",
                "train": (
                    f"all of [{train_record_text}] plus {CUT_RECORD} samples "
                    f"[0,{CUT_SAMPLE}); window support must end no later than "
                    f"sample {CUT_SAMPLE}"
                ),
                "validation": (
                    f"{CUT_RECORD} samples [{CUT_SAMPLE},end); window support must "
                    f"begin at or after sample {CUT_SAMPLE}"
                ),
                "test": f"all {TEST_RECORD}; untouched until final evaluation",
                "ignored_post_test_records": list(IGNORED_RECORDS),
                "cut_record": CUT_RECORD,
                "cut_sample": CUT_SAMPLE,
                "cut_time_seconds": CUT_SAMPLE / core.SAMPLING_RATE_HZ,
                "cut_selection_disclosure": (
                    "The nominal 70 percent chronological pre-test cut was rounded "
                    "to the 64-sample window grid and verified to lie between FOG "
                    "events so no event is split and both train/validation retain "
                    "both classes. This is label-aware exploratory split design; "
                    "the final test record was not used."
                ),
            },
            "leakage_controls": [
                f"Robust scaler uses valid non-FoG samples from {SUBJECT_ID} train ranges only.",
                f"GRU weights use clean-normal {SUBJECT_ID} train windows only.",
                f"{SUBJECT_ID} validation selects the GRU epoch, TCN-M epoch, and threshold.",
                f"{TEST_RECORD} is used once for final evaluation only.",
                "Train/validation context-target supports are raw-sample disjoint.",
            ],
            "known_limitation": (
                f"Test support is one record ({TEST_RECORD}), so metrics may have "
                "high sampling uncertainty. Classifier-training residuals are also "
                "in-sample with respect to the GRU."
            ),
        }
    )
    payload["dataset_fingerprint_sha256"] = dataset_fingerprint(args.data_dir)
    payload["protocol_fingerprint"] = canonical_fingerprint(
        {
            key: value
            for key, value in payload.items()
            if key not in {"created_utc", "environment", "protocol_fingerprint"}
        }
    )
    return payload


def write_summary(
    output_dir: Path,
    protocol: dict[str, Any],
    nbm_training: dict[str, Any],
    classifier_training: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    test = metrics["test"]
    text = f"""# {SUBJECT_ID} single-subject result

- Context/target/stride: {core.CONTEXT_SAMPLES}/{core.TARGET_SAMPLES}/{core.STRIDE_SAMPLES} samples
- Train complete records: {', '.join(TRAIN_RECORDS) if TRAIN_RECORDS else '(none)'}
- Train cut range: {CUT_RECORD}[0:{CUT_SAMPLE}]
- Validation: {CUT_RECORD}[{CUT_SAMPLE}:end]
- Test: {TEST_RECORD}
- Ignored post-test records: {', '.join(IGNORED_RECORDS) if IGNORED_RECORDS else '(none)'}
- GRU maximum/best epoch: {nbm_training['maximum_epochs']}/{nbm_training['best_epoch']}
- TCN-M maximum/best epoch: {classifier_training['maximum_epochs']}/{classifier_training['best_epoch']}
- Validation-selected threshold: {classifier_training['selected_threshold']:.4f}

| Test metric | Value |
|---|---:|
| Accuracy | {test['accuracy']:.6f} |
| FoG recall | {test['fog_recall']:.6f} |
| Specificity | {test['specificity']:.6f} |
| PR-AUC | {test['pr_auc']:.6f} |

Test confusion matrix: TN={test['tn']}, FP={test['fp']}, FN={test['fn']}, TP={test['tp']}.
"""
    path = output_dir / "summary.md"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    configure_subject(args.subject)
    configure_geometry(args)
    core.validate_args(args)
    core.EXPERIMENT_VERSION = EXPERIMENT_VERSION
    # The shared trainer resolves this name dynamically for clean-normal guards.
    core.normal_support_indices = normal_support_indices
    device = core.resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    done_path = output_dir / "DONE.json"
    if done_path.exists():
        raise FileExistsError(f"Completed output already exists: {done_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is non-empty; pass --overwrite to reuse: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.data_dir.resolve())
    base_windows = dataset.make_windows(
        warmup_samples=core.CONTEXT_SAMPLES,
        target_samples=core.TARGET_SAMPLES,
        stride_samples=core.STRIDE_SAMPLES,
        fog_fraction_threshold=0.5,
        normal_guard_samples=core.NORMAL_GUARD_SAMPLES,
    )
    windows = core.endpoint_relabel(dataset, base_windows)
    split = make_split(dataset, windows)
    scaler, scaler_metadata = fit_training_scaler(dataset)
    point_stats = point_statistics(dataset)
    window_stats = window_statistics(dataset, windows, split)
    clip_stats = scaler_clip_statistics(dataset, scaler)
    protocol = build_protocol(
        args,
        dataset,
        point_stats,
        window_stats,
        scaler_metadata,
        clip_stats,
        device,
    )
    atomic_json_dump(protocol, output_dir / "config.json")
    atomic_json_dump(scaler_metadata, output_dir / "scaler.json")
    core.write_csv(
        output_dir / "split_manifest.csv",
        core.split_manifest_rows(dataset, windows, split),
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
        f"{ {name: len(value) for name, value in split.as_dict().items()} }",
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

    nbm, nbm_training = core.train_nbm(
        args,
        dataset,
        windows,
        split,
        scaler,
        output_dir,
        protocol["protocol_fingerprint"],
        device,
    )
    core.plot_nbm_losses(output_dir, nbm_training)
    features: dict[str, dict[str, np.ndarray]] = {}
    residual_diagnostics: dict[str, Any] = {}
    for name, indices in split.as_dict().items():
        features[name], residual_diagnostics[name] = core.extract_residuals(
            args, nbm, dataset, windows, indices, scaler, device
        )
    atomic_json_dump(residual_diagnostics, output_dir / "residual_diagnostics.json")
    atomic_npz_save(
        output_dir / "residual_cache.npz",
        **{
            f"{split_name}_{field}": values[field]
            for split_name, values in features.items()
            for field in ("residual", "y", "window_index")
        },
    )
    classifier_training, metrics = core.train_classifier(
        args,
        features,
        dataset,
        windows,
        output_dir,
        protocol["protocol_fingerprint"],
        device,
    )
    core.plot_classifier_losses(output_dir, classifier_training)
    core.plot_test_confusion_matrix(output_dir, metrics["test"], SUBJECT_ID)
    write_summary(output_dir, protocol, nbm_training, classifier_training, metrics)
    atomic_json_dump(
        {
            "status": "complete",
            "completed_utc": core.utc_now(),
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
        f"test_accuracy={test['accuracy']:.6f} "
        f"test_fog_recall={test['fog_recall']:.6f} "
        f"test_specificity={test['specificity']:.6f} "
        f"test_pr_auc={test['pr_auc']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
