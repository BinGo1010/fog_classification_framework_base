#!/usr/bin/env python
"""Run the S01 outer fold of the 9-channel GRU-H200 residual TCN-M experiment.

This keeps the model, window, label, optimiser, epoch, and threshold settings
of ``run_daphnet_s01_gru_h200_tcnm.py``.  Only the data split changes:

* outer test: S01 (all records);
* inner validation: S02 (all records);
* training: S03--S10 (all records).

The subject-disjoint validation subject is used for both early stopping and
decision-threshold selection.  S01 is never used by the scaler, either model,
class weighting, early stopping, or threshold selection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_daphnet_s01_gru_h200_tcnm as core  # noqa: E402
from cnbr_fog.data import DaphnetDataset, Record, RobustChannelScaler, WindowTable  # noqa: E402
from cnbr_fog.resume import (  # noqa: E402
    atomic_json_dump,
    atomic_npz_save,
    canonical_fingerprint,
    sha256_file,
)


EXPERIMENT_VERSION = "daphnet_loso_s01_gru_h200_tcnm.v1"
TEST_SUBJECT = "S01"
VALIDATION_SUBJECT = "S02"
TRAIN_SUBJECTS = tuple(f"S{index:02d}" for index in range(3, 11))
EXPECTED_SUBJECTS = (TEST_SUBJECT, VALIDATION_SUBJECT, *TRAIN_SUBJECTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S01 LOSO fold: GRU-H200 standardized-residual TCN-M",
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
        default=REPO_ROOT / "outputs" / "daphnet_loso_s01_gru_h200_tcnm_seed42",
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
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_dataset(data_dir: Path) -> DaphnetDataset:
    dataset = DaphnetDataset.load(data_dir)
    if dataset.sampling_rate_hz != core.SAMPLING_RATE_HZ:
        raise ValueError(
            f"Expected {core.SAMPLING_RATE_HZ} Hz, got {dataset.sampling_rate_hz} Hz"
        )
    if tuple(dataset.channel_names) != core.EXPECTED_CHANNEL_NAMES:
        raise ValueError(f"Unexpected channels: {dataset.channel_names}")
    if tuple(dataset.subjects) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected subjects {EXPECTED_SUBJECTS}, got {tuple(dataset.subjects)}"
        )
    return dataset


def record_subjects(dataset: DaphnetDataset, windows: WindowTable) -> np.ndarray:
    return np.asarray(
        [dataset.records[int(index)].subject_id for index in windows.record_index],
        dtype="U3",
    )


def make_split(dataset: DaphnetDataset, windows: WindowTable) -> core.SplitBundle:
    subjects = record_subjects(dataset, windows)
    split = core.SplitBundle(
        train=np.flatnonzero(np.isin(subjects, TRAIN_SUBJECTS)),
        validation=np.flatnonzero(subjects == VALIDATION_SUBJECT),
        test=np.flatnonzero(subjects == TEST_SUBJECT),
    )
    memberships = split.as_dict()
    for name, indices in memberships.items():
        if not len(indices):
            raise ValueError(f"{name} split is empty")
        counts = np.bincount(windows.label[indices], minlength=2)
        if np.any(counts == 0):
            raise ValueError(f"{name} split lacks a class: {counts.tolist()}")
    index_sets = {name: set(value.tolist()) for name, value in memberships.items()}
    if index_sets["train"] & index_sets["validation"]:
        raise AssertionError("train/validation overlap")
    if index_sets["train"] & index_sets["test"]:
        raise AssertionError("train/test overlap")
    if index_sets["validation"] & index_sets["test"]:
        raise AssertionError("validation/test overlap")
    actual = {
        name: sorted(set(record_subjects(dataset, windows.take(indices)).tolist()))
        for name, indices in memberships.items()
    }
    expected = {
        "train": list(TRAIN_SUBJECTS),
        "validation": [VALIDATION_SUBJECT],
        "test": [TEST_SUBJECT],
    }
    if actual != expected:
        raise AssertionError(f"Unexpected subject memberships: {actual}")
    return split


def records_for_subjects(
    dataset: DaphnetDataset, subjects: Iterable[str]
) -> list[Record]:
    selected = set(subjects)
    return [record for record in dataset.records if record.subject_id in selected]


def fit_training_scaler(
    dataset: DaphnetDataset,
) -> tuple[RobustChannelScaler, dict[str, Any]]:
    chunks: list[np.ndarray] = []
    retained_points = 0
    for record in records_for_subjects(dataset, TRAIN_SUBJECTS):
        mask = record.valid & (record.y == 0)
        retained_points += int(mask.sum())
        chunks.append(record.x[mask])
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
        "fit_subjects": list(TRAIN_SUBJECTS),
        "excluded_validation_subject": VALIDATION_SUBJECT,
        "excluded_test_subject": TEST_SUBJECT,
        "fit_split": "train_only",
        "fit_class": "valid_non_fog_samples_only",
        "fit_points": retained_points,
        "scale_definition": "IQR/1.349; per-channel std fallback; 1.0 final fallback",
    }
    return scaler, metadata


def subject_groups() -> dict[str, tuple[str, ...]]:
    return {
        "train": TRAIN_SUBJECTS,
        "validation": (VALIDATION_SUBJECT,),
        "test": (TEST_SUBJECT,),
    }


def point_statistics(dataset: DaphnetDataset) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, subjects in subject_groups().items():
        records = records_for_subjects(dataset, subjects)
        labels = np.concatenate([record.y for record in records])
        valid = np.concatenate([record.valid for record in records])
        normal_points = int(np.sum((labels == 0) & valid))
        fog_points = int(np.sum((labels == 1) & valid))
        result[name] = {
            "subjects": list(subjects),
            "records": [record.record_id for record in records],
            "raw_points": int(len(labels)),
            "valid_points": int(valid.sum()),
            "non_fog_points": normal_points,
            "fog_points": fog_points,
            "fog_percent": 100.0 * fog_points / max(normal_points + fog_points, 1),
            "duration_seconds": len(labels) / core.SAMPLING_RATE_HZ,
            "fog_events": int(
                sum(len(core.boolean_runs(record.y == 1)) for record in records)
            ),
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
        clean = core.normal_support_indices(dataset, windows, name, indices)
        result[name] = {
            "subjects": list(subject_groups()[name]),
            "windows": int(len(indices)),
            "non_fog_windows": int(counts[0]),
            "fog_windows": int(counts[1]),
            "fog_percent": 100.0 * int(counts[1]) / max(int(counts.sum()), 1),
            "clean_normal_windows": int(len(clean)),
            "first_window_index": int(indices[0]),
            "last_window_index": int(indices[-1]),
        }
    return result


def scaler_clip_statistics(
    dataset: DaphnetDataset, scaler: RobustChannelScaler
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, subjects in subject_groups().items():
        x = np.concatenate(
            [record.x for record in records_for_subjects(dataset, subjects)]
        )
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
    scaler_clip_stats: dict[str, dict[str, Any]],
    device,
) -> dict[str, Any]:
    payload = core.build_protocol(
        args,
        dataset,
        point_stats,
        window_stats,
        scaler_metadata,
        scaler_clip_stats,
        device,
    )
    payload.update(
        {
            "experiment_version": EXPERIMENT_VERSION,
            "outer_fold": TEST_SUBJECT,
            "subject": None,
            "subjects": list(dataset.subjects),
            "records": [record.record_id for record in dataset.records],
            "split": {
                "strategy": "nested subject-disjoint split for one LOSO outer fold",
                "train_subjects": list(TRAIN_SUBJECTS),
                "validation_subject": VALIDATION_SUBJECT,
                "test_subject": TEST_SUBJECT,
                "validation_selection": (
                    "fixed next cyclic subject after S01 with both window classes"
                ),
                "raw_support_overlap": False,
            },
            "leakage_controls": [
                "Robust scaler uses valid non-FOG points from S03-S10 only.",
                "GRU weights use clean-normal S03-S10 windows only.",
                "S02 selects the GRU epoch, TCN-M epoch, and decision threshold.",
                "S01 is used once for final evaluation only.",
                "Train, validation, and test subjects are disjoint.",
            ],
            "known_limitation": (
                "Classifier-training residuals are generated by the GRU fitted on "
                "the same S03-S10 windows. This is not test leakage, but can make "
                "training non-FOG residuals optimistic; cross-fitted training "
                "residuals are a stricter follow-up. One fixed seed is reported."
            ),
        }
    )
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
    text = f"""# S01 LOSO fold result

- Train subjects: {', '.join(TRAIN_SUBJECTS)}
- Validation subject: {VALIDATION_SUBJECT}
- Test subject: {TEST_SUBJECT}
- Test windows: {protocol['window_statistics']['test']['windows']:,}
- GRU best epoch: {nbm_training['best_epoch']}
- TCN-M best epoch: {classifier_training['best_epoch']}
- Validation-selected threshold: {classifier_training['selected_threshold']:.4f}

| Overall test metric | Value |
|---|---:|
| Accuracy | {test['accuracy']:.6f} |
| FoG recall | {test['fog_recall']:.6f} |
| Specificity | {test['specificity']:.6f} |
| PR-AUC | {test['pr_auc']:.6f} |

Test confusion matrix: TN={test['tn']}, FP={test['fp']},
FN={test['fn']}, TP={test['tp']}.
"""
    path = output_dir / "summary.md"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    core.validate_args(args)
    core.EXPERIMENT_VERSION = EXPERIMENT_VERSION
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
        f"device={device} subjects={subject_groups()} "
        f"window_counts={ {name: len(value) for name, value in split.as_dict().items()} }",
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
    core.plot_test_confusion_matrix(output_dir, metrics["test"])
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
        f"COMPLETE test_accuracy={test['accuracy']:.6f} "
        f"test_fog_recall={test['fog_recall']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
