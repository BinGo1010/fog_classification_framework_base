#!/usr/bin/env python
"""Run the within-subject raw-target TCN-M ablation.

This removes both the GRU normal-behaviour predictor and the standardized
forecast-residual transform.  To keep the comparison controlled, windowing,
splits, labels, scaler, TCN-M architecture/training, validation-only epoch and
threshold selection, and test evaluation match the GRU-residual experiment.
The classifier sees the Robust-scaled 1-second target block directly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

import run_daphnet_s01_gru_h200_tcnm as core
import run_daphnet_s09_gru_h200_tcnm as subject_core
from cnbr_fog.resume import (
    atomic_json_dump,
    atomic_npz_save,
    canonical_fingerprint,
    dataset_fingerprint,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_VERSION = "daphnet_single_subject_raw_target_tcnm.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Within-subject raw-target TCN-M ablation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--subject", choices=sorted(subject_core.SPLIT_CONFIGS), required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    # Retained only so the shared argument validator/protocol schema is explicit.
    # No GRU model is instantiated or trained by this script.
    parser.add_argument("--normal-epochs", type=int, default=50)
    parser.add_argument("--normal-patience", type=int, default=6)
    parser.add_argument("--normal-lr", type=float, default=1e-3)
    parser.add_argument("--nbm-hidden", type=int, default=48)
    parser.add_argument("--nbm-dropout", type=float, default=0.1)
    parser.add_argument("--classifier-epochs", type=int, default=12)
    parser.add_argument("--classifier-patience", type=int, default=4)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--classifier-hidden", type=int, default=48)
    parser.add_argument("--classifier-dropout", type=float, default=0.15)
    parser.add_argument("--context-seconds", type=float, default=2.0)
    parser.add_argument("--target-seconds", type=float, default=1.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def direct_target_features(
    args: argparse.Namespace,
    dataset,
    windows,
    indices: np.ndarray,
    scaler,
    device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract Robust-scaled target blocks in deterministic window order."""
    loader = core.sequence_loader(
        dataset,
        windows,
        indices,
        scaler,
        args.batch_size,
        False,
        args.num_workers,
        device.type == "cuda",
    )
    targets: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    window_indices: list[np.ndarray] = []
    for sequence, y, index in loader:
        targets.append(sequence[:, :, core.CONTEXT_SAMPLES:].numpy())
        labels.append(y.numpy())
        window_indices.append(index.numpy())
    x = np.ascontiguousarray(np.concatenate(targets).astype(np.float32, copy=False))
    y = np.concatenate(labels).astype(np.int8, copy=False)
    index = np.concatenate(window_indices).astype(np.int64, copy=False)
    if not np.array_equal(index, indices):
        raise AssertionError("Raw-target extraction changed window order")
    if x.shape[1:] != (dataset.n_channels, core.TARGET_SAMPLES):
        raise AssertionError(f"Unexpected classifier input shape: {x.shape}")
    diagnostics = {
        "windows": int(len(y)),
        "class_counts": np.bincount(y, minlength=2).astype(int).tolist(),
        "shape": list(x.shape),
        "input": "Robust-scaled target raw signal",
        "absolute_mean": float(np.mean(np.abs(x.astype(np.float64)))),
        "rms": float(np.sqrt(np.mean(x.astype(np.float64) ** 2))),
        "cells_at_scaler_clip": int(np.sum(np.abs(x) >= float(scaler.clip))),
        "fraction_at_scaler_clip": float(np.mean(np.abs(x) >= float(scaler.clip))),
    }
    return {"residual": x, "y": y, "window_index": index}, diagnostics


def ablation_protocol(args, dataset, point_stats, window_stats, scaler_metadata, clip_stats, device):
    payload = subject_core.build_protocol(
        args, dataset, point_stats, window_stats, scaler_metadata, clip_stats, device
    )
    payload["experiment_version"] = EXPERIMENT_VERSION
    payload.pop("normal_behaviour_model", None)
    payload.pop("residual", None)
    payload["ablation"] = {
        "removed_modules": ["GRU normal-behaviour predictor", "standardized residual"],
        "classifier_input": "Robust-scaled raw target signal",
        "classifier_input_shape": ["batch", dataset.n_channels, core.TARGET_SAMPLES],
        "context_usage": (
            "The 2-second context is retained only to preserve identical window endpoints, "
            "split membership, and test decisions; it is not passed to TCN-M."
        ),
        "controlled_factors": [
            "subject-specific chronological split",
            "window endpoints and endpoint labels",
            "train-only Robust scaler",
            "TCN-M architecture and optimizer",
            "validation-only epoch and threshold selection",
            "random seed",
        ],
    }
    payload["classifier"]["input"] = "Robust-scaled raw target [batch,9,64]"
    payload["training"]["execution_scope"] = "raw_target_tcnm_ablation"
    for key in ("normal_epochs_max", "normal_patience", "normal_learning_rate"):
        payload["training"].pop(key, None)
    payload["leakage_controls"] = [
        f"Robust scaler uses valid non-FoG samples from {subject_core.SUBJECT_ID} train ranges only.",
        f"{subject_core.SUBJECT_ID} validation selects the TCN-M epoch and threshold.",
        f"{subject_core.TEST_RECORD} is used once for final evaluation only.",
        "Train/validation context-target supports are raw-sample disjoint.",
    ]
    payload["known_limitation"] = (
        f"Test support is one record ({subject_core.TEST_RECORD}), so metrics may have high "
        "sampling uncertainty; overlapping 0.5-second decisions are not independent."
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


def write_summary(output_dir: Path, protocol, diagnostics, training, metrics) -> None:
    test = metrics["test"]
    validation = metrics["validation"]
    text = f"""# {subject_core.SUBJECT_ID} raw-target TCN-M ablation

- Removed: GRU predictor and standardized forecast residual.
- Classifier input: Robust-scaled raw target, shape [B, 9, {core.TARGET_SAMPLES}].
- Context/target/stride: {core.CONTEXT_SAMPLES}/{core.TARGET_SAMPLES}/{core.STRIDE_SAMPLES} samples = {core.CONTEXT_SAMPLES/64:g}/{core.TARGET_SAMPLES/64:g}/{core.STRIDE_SAMPLES/64:g} seconds.
- Context is retained for identical endpoints/splits only and is not input to TCN-M.
- TCN-M maximum/best/completed epoch: {training['maximum_epochs']}/{training['best_epoch']}/{training['epochs_completed']}.
- Validation PR-AUC at selected epoch: {training['best_validation_pr_auc']:.6f}.
- Validation-selected classification threshold: {training['selected_threshold']:.4f}.

| Split | Windows | Non-FoG/FoG | Scaled input RMS |
|---|---:|---:|---:|
| Train | {diagnostics['train']['windows']} | {diagnostics['train']['class_counts'][0]}/{diagnostics['train']['class_counts'][1]} | {diagnostics['train']['rms']:.6f} |
| Validation | {diagnostics['validation']['windows']} | {diagnostics['validation']['class_counts'][0]}/{diagnostics['validation']['class_counts'][1]} | {diagnostics['validation']['rms']:.6f} |
| Test | {diagnostics['test']['windows']} | {diagnostics['test']['class_counts'][0]}/{diagnostics['test']['class_counts'][1]} | {diagnostics['test']['rms']:.6f} |

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | {validation['accuracy']:.6f} | {test['accuracy']:.6f} |
| FoG recall | {validation['fog_recall']:.6f} | {test['fog_recall']:.6f} |
| Specificity | {validation['specificity']:.6f} | {test['specificity']:.6f} |
| PR-AUC | {validation['pr_auc']:.6f} | {test['pr_auc']:.6f} |

Test confusion matrix: TN={test['tn']}, FP={test['fp']}, FN={test['fn']}, TP={test['tp']}.
"""
    path = output_dir / "summary.md"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    subject_core.configure_subject(args.subject)
    subject_core.configure_geometry(args)
    core.validate_args(args)
    core.EXPERIMENT_VERSION = EXPERIMENT_VERSION
    core.normal_support_indices = subject_core.normal_support_indices
    device = core.resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    done_path = output_dir / "DONE.json"
    if done_path.exists():
        raise FileExistsError(f"Completed output already exists: {done_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = subject_core.load_dataset(args.data_dir.resolve())
    base_windows = dataset.make_windows(
        warmup_samples=core.CONTEXT_SAMPLES,
        target_samples=core.TARGET_SAMPLES,
        stride_samples=core.STRIDE_SAMPLES,
        fog_fraction_threshold=0.5,
        normal_guard_samples=core.NORMAL_GUARD_SAMPLES,
    )
    windows = core.endpoint_relabel(dataset, base_windows)
    split = subject_core.make_split(dataset, windows)
    scaler, scaler_metadata = subject_core.fit_training_scaler(dataset)
    point_stats = subject_core.point_statistics(dataset)
    window_stats = subject_core.window_statistics(dataset, windows, split)
    clip_stats = subject_core.scaler_clip_statistics(dataset, scaler)
    protocol = ablation_protocol(
        args, dataset, point_stats, window_stats, scaler_metadata, clip_stats, device
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
        f"subject={subject_core.SUBJECT_ID} device={device} window_counts="
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

    features: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, Any] = {}
    for name, indices in split.as_dict().items():
        features[name], diagnostics[name] = direct_target_features(
            args, dataset, windows, indices, scaler, device
        )
    atomic_json_dump(diagnostics, output_dir / "direct_input_diagnostics.json")
    atomic_npz_save(
        output_dir / "raw_target_cache.npz",
        **{
            f"{split_name}_{field}": values[field]
            for split_name, values in features.items()
            for field in ("residual", "y", "window_index")
        },
    )
    training, metrics = core.train_classifier(
        args,
        features,
        dataset,
        windows,
        output_dir,
        protocol["protocol_fingerprint"],
        device,
    )
    core.plot_classifier_losses(output_dir, training)
    core.plot_test_confusion_matrix(output_dir, metrics["test"], subject_core.SUBJECT_ID)
    write_summary(output_dir, protocol, diagnostics, training, metrics)
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
