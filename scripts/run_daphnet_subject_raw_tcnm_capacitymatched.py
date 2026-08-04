#!/usr/bin/env python
"""Capacity-matched Raw-TCN ablation for one Daphnet subject.

The complete spectral NBM front-end is removed. Robust-scaled 2 s raw IMU
windows [B,9,128] are sent directly to the same four-block TCN-M used by the
spectral residual experiment. There is no FFT, NBM, residual, or crossfit.
"""

from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch

import run_daphnet_s01_spectral_gru_nbm_tcnm as model_core
import run_daphnet_subject_spectral_gru_nbm_blocked_crossfit as protocol
from cnbr_fog.resume import atomic_json_dump, atomic_npz_save, dataset_fingerprint, sha256_file


EXPERIMENT_VERSION = "daphnet_subject_raw_tcnm_capacitymatched.v1"


def parse_args() -> argparse.Namespace:
    local_data = (
        model_core.REPO_ROOT
        / "dataset"
        / "1.Daphnet Freezing of Gait Dataset"
        / "processed"
    )
    cloud_data = Path(
        r"E:\fog_cloud\dataset\1.Daphnet Freezing of Gait Dataset\processed"
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Within-subject capacity-matched Raw-TCN ablation",
    )
    parser.add_argument("--subject", choices=protocol.SUBJECTS, required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=local_data if local_data.exists() else cloud_data
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--classifier-epochs", type=int, default=50)
    parser.add_argument("--classifier-patience", type=int, default=10)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--window-axis-centering",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After Robust scaling, subtract each 2 s window's temporal mean "
            "independently for all nine axes before transposing to [9,128]."
        ),
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.output_dir is None:
        centering_suffix = "_window_axis_centered" if args.window_axis_centering else ""
        args.output_dir = (
            model_core.REPO_ROOT
            / "outputs"
            / f"daphnet_{args.subject.lower()}_raw_tcnm_capacitymatched{centering_suffix}_seed{args.seed}"
        )
    # Kept for compatibility with the shared classifier trainer; unused here.
    args.nbm_epochs = 0
    args.nbm_patience = 0
    args.nbm_lr = 0.0
    args.noise_std = 0.0
    args.time_mask_probability = 0.0
    args.channel_mask_probability = 0.0
    return args


def main() -> None:
    args = parse_args()
    subject = args.subject
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = model_core.resolve_device(args.device)
    model_core.set_seed(args.seed, args.deterministic)

    dataset = protocol.load_subject(args.data_dir, subject)
    windows = model_core.make_windows(dataset)
    split = protocol.make_subject_split(dataset, windows, subject)
    scaler, scaler_fit_points = protocol.fit_subject_scaler(dataset, subject)
    window_statistics = protocol.stats(windows, split)

    features: dict[str, dict[str, np.ndarray]] = {}
    for name, indices in split.as_dict().items():
        time_channel = protocol.base.extract_windows(
            dataset, windows, indices, scaler
        )
        if args.window_axis_centering:
            time_channel = time_channel - time_channel.mean(
                axis=1, keepdims=True, dtype=np.float32
            )
            time_channel = np.ascontiguousarray(time_channel, dtype=np.float32)
            maximum_axis_mean = float(
                np.max(np.abs(time_channel.mean(axis=1)))
            )
            if maximum_axis_mean > 5e-5:
                raise AssertionError(
                    f"Window-axis centering numerical error: {maximum_axis_mean}"
                )
        channel_time = np.ascontiguousarray(time_channel.transpose(0, 2, 1))
        if channel_time.shape[1:] != (9, 128):
            raise AssertionError(f"Unexpected {name} input shape {channel_time.shape}")
        features[name] = {
            "residual": channel_time,
            "y": windows.label[indices].copy(),
            "window_index": indices.copy(),
        }
    print(
        f"[{subject} Raw-TCN] device={device} windows={window_statistics} "
        f"input_shapes={ {name: list(value['residual'].shape) for name, value in features.items()} }",
        flush=True,
    )

    classifier_training, metrics, predictions = model_core.train_classifier(
        args,
        {name: values["residual"] for name, values in features.items()},
        {name: values["y"] for name, values in features.items()},
        output_dir,
        device,
    )
    atomic_npz_save(output_dir / "predictions.npz", **predictions)
    atomic_npz_save(
        output_dir / "split_indices.npz",
        train=split.train,
        validation=split.validation,
        test=split.test,
    )
    model_core.write_csv(
        output_dir / "tcnm_history.csv", classifier_training["history"]
    )
    model_core.plot_history(
        classifier_training["history"],
        "train_bce",
        "validation_bce",
        f"{subject} capacity-matched Raw-TCN loss",
        output_dir / "tcnm_loss.png",
    )
    model_core.plot_confusion(
        metrics["test"], output_dir / "test_confusion_matrix.png", subject
    )

    split_config = protocol.subject_config(subject)
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject": subject,
        "seed": args.seed,
        "data_dir": str(args.data_dir.resolve()),
        "dataset_fingerprint_sha256": dataset_fingerprint(args.data_dir),
        "split": {
            **split_config,
            "strategy": "same predefined chronological raw-support-disjoint split as full NBM experiment",
            "cut_is_label_aware": True,
            "ignored_records_are_not_used": True,
        },
        "windowing": {
            "sampling_rate_hz": 64,
            "seconds": 2.0,
            "samples": 128,
            "stride_seconds": 1.0,
            "stride_samples": 64,
            "label": "FOG if at least 50% of all 128 samples are FOG",
            "statistics": window_statistics,
            "per_window_per_axis_temporal_centering": {
                "enabled": bool(args.window_axis_centering),
                "ordering": "after train-fitted Robust scaling and before [B,T,C] to [B,C,T] transpose",
                "formula": "x_centered[b,t,c] = x_scaled[b,t,c] - mean_t(x_scaled[b,:,c])",
                "fit_parameters": None,
            },
        },
        "scaler": {
            **scaler.as_dict(),
            "fit_split": "train only",
            "fit_class": "valid Non-FoG sample points only",
            "fit_points": scaler_fit_points,
            "definition": "median and IQR/1.349 with train standard-deviation fallback",
        },
        "input": {
            "shape": ["batch", 9, 128],
            "content": (
                "Robust-scaled, per-window per-axis temporally centered raw IMU samples"
                if args.window_axis_centering
                else "Robust-scaled raw IMU samples"
            ),
            "axis": "time",
            "fft": False,
            "log_power_spectrum": False,
            "nbm": False,
            "residual": False,
            "crossfit": False,
            "augmentation": False,
        },
        "classifier": {
            "name": "capacity-matched TCN-M",
            "hidden_channels": 48,
            "residual_blocks": 4,
            "dilations": [1, 2, 4, 8],
            "kernel_size": 3,
            "convolutions_per_block": 2,
            "local_receptive_field_samples": 61,
            "local_receptive_field_seconds": 61 / 64,
            "global_pooling": "mean and max over the full 2 s input",
            "normalization": "GroupNorm",
            "activation": "GELU",
            "dropout": 0.2,
            "parameter_count": classifier_training["parameter_count"],
            "training": classifier_training,
        },
        "fairness_controls": [
            "The outer split indices, scaler rule, windows, labels, TCN parameterization, optimizer, class weight, early stopping, and threshold rule match the full NBM experiment.",
            "Only the complete spectral NBM front-end and its crossfit are removed.",
            "The TCN is reinitialized; no full-model weights are reused.",
            "Validation selects TCN epoch and threshold; test is evaluated after freezing both.",
        ],
        "arguments": vars(args)
        | {"data_dir": str(args.data_dir), "output_dir": str(output_dir)},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
    }
    atomic_json_dump(config, output_dir / "config.json")

    validation = metrics["validation"]
    test = metrics["test"]
    summary = f"""# {subject} capacity-matched Raw-TCN ablation

| Split | Accuracy | FoG Recall | Specificity | PR-AUC | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Validation | {validation['accuracy']:.6f} | {validation['fog_recall']:.6f} | {validation['specificity']:.6f} | {validation['pr_auc']:.6f} | {validation['f1']:.6f} | {validation['balanced_accuracy']:.6f} |
| Test | {test['accuracy']:.6f} | {test['fog_recall']:.6f} | {test['specificity']:.6f} | {test['pr_auc']:.6f} | {test['f1']:.6f} | {test['balanced_accuracy']:.6f} |

- Input: Robust-scaled raw IMU `[B,9,128]`; no FFT, NBM, residual, augmentation, or crossfit.
- Per-window per-axis temporal centering: {'enabled' if args.window_axis_centering else 'disabled'} after Robust scaling.
- Train/validation/test windows: {window_statistics['train']['windows']}/{window_statistics['validation']['windows']}/{window_statistics['test']['windows']}.
- TCN parameters: {classifier_training['parameter_count']:,}; best epoch: {classifier_training['best_epoch']}.
- Validation-selected threshold: {classifier_training['selected_threshold']:.4f}.
- Test confusion: TN={test['tn']}, FP={test['fp']}, FN={test['fn']}, TP={test['tp']}.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    atomic_json_dump(
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_version": EXPERIMENT_VERSION,
            "subject": subject,
            "main_test_metrics": {
                key: test[key]
                for key in (
                    "accuracy",
                    "fog_recall",
                    "specificity",
                    "pr_auc",
                    "f1",
                    "balanced_accuracy",
                )
            },
            "artifacts": {
                path.name: sha256_file(path)
                for path in sorted(output_dir.iterdir())
                if path.is_file()
            },
        },
        output_dir / "DONE.json",
    )
    print(
        f"COMPLETE RAW {subject} accuracy={test['accuracy']:.6f} "
        f"recall={test['fog_recall']:.6f} specificity={test['specificity']:.6f} "
        f"pr_auc={test['pr_auc']:.6f} confusion={test['confusion_matrix']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
