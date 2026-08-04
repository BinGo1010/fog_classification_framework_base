#!/usr/bin/env python
"""Run the S01 spectral blocked-crossfit protocol for one Daphnet subject."""

from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch

import run_daphnet_s01_spectral_gru_nbm_blocked_crossfit as cross
import run_daphnet_s01_spectral_gru_nbm_tcnm as base
from cnbr_fog.data import DaphnetDataset, RobustChannelScaler
from cnbr_fog.resume import atomic_json_dump, atomic_npz_save, dataset_fingerprint, sha256_file
from run_daphnet_s09_gru_h200_tcnm import SPLIT_CONFIGS


EXPERIMENT_VERSION = "daphnet_subject_spectral_gru_nbm_blocked_crossfit.v1"
SUBJECTS = tuple(subject for subject in sorted(SPLIT_CONFIGS) if subject not in {"S04", "S10"})


def parse_args() -> argparse.Namespace:
    local_data = (
        base.REPO_ROOT
        / "dataset"
        / "1.Daphnet Freezing of Gait Dataset"
        / "processed"
    )
    cloud_data = Path(
        r"E:\fog_cloud\dataset\1.Daphnet Freezing of Gait Dataset\processed"
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Within-subject spectral GRU-NBM blocked crossfit + TCN-M",
    )
    parser.add_argument("--subject", choices=SUBJECTS, required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=local_data if local_data.exists() else cloud_data
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-epochs", type=int, default=50)
    parser.add_argument("--nbm-patience", type=int, default=10)
    parser.add_argument("--nbm-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-epochs", type=int, default=50)
    parser.add_argument("--classifier-patience", type=int, default=10)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--time-mask-probability", type=float, default=0.30)
    parser.add_argument("--channel-mask-probability", type=float, default=0.20)
    parser.add_argument(
        "--window-axis-centering",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After Robust scaling, subtract each 2 s window's temporal mean "
            "independently for all nine axes before corruption or FFT."
        ),
    )
    parser.add_argument(
        "--spectral-robust-standardization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fit per-axis/per-frequency median and IQR scale on each NBM's "
            "clean-normal gradient-training spectra and reconstruct in that space."
        ),
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and print split diagnostics without fitting models.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        centering_suffix = "_window_axis_centered" if args.window_axis_centering else ""
        spectral_suffix = (
            "_spectral_robust" if args.spectral_robust_standardization else ""
        )
        args.output_dir = (
            base.REPO_ROOT
            / "outputs"
            / f"daphnet_{args.subject.lower()}_spectral_gru_nbm_blocked5{centering_suffix}{spectral_suffix}_tcnm_seed{args.seed}"
        )
    return args


def subject_config(subject: str) -> dict[str, Any]:
    return dict(SPLIT_CONFIGS[subject])


def load_subject(root: Path, subject: str) -> DaphnetDataset:
    full = DaphnetDataset.load(root)
    records = [record for record in full.records if record.subject_id == subject]
    expected = {
        *subject_config(subject)["train_records"],
        subject_config(subject)["cut_record"],
        subject_config(subject)["test_record"],
        *subject_config(subject)["ignored_records"],
    }
    if {record.record_id for record in records} != expected:
        raise ValueError(f"Unexpected {subject} records")
    if full.sampling_rate_hz != base.FS or tuple(full.channel_names) != base.EXPECTED_CHANNEL_NAMES:
        raise ValueError("Unexpected sampling rate or channel schema")
    return DaphnetDataset(
        root=root,
        records=records,
        sampling_rate_hz=full.sampling_rate_hz,
        channel_names=full.channel_names,
    )


def make_subject_split(
    dataset: DaphnetDataset, windows: base.Windows, subject: str
) -> base.Split:
    config = subject_config(subject)
    lookup = {record.record_id: index for index, record in enumerate(dataset.records)}
    train_record_indices = np.asarray(
        [lookup[name] for name in config["train_records"]], dtype=np.int64
    )
    cut_record_index = lookup[config["cut_record"]]
    test_record_index = lookup[config["test_record"]]
    train = np.flatnonzero(
        np.isin(windows.record_index, train_record_indices)
        | (
            (windows.record_index == cut_record_index)
            & (windows.end <= int(config["cut_sample"]))
        )
    )
    validation = np.flatnonzero(
        (windows.record_index == cut_record_index)
        & (windows.start >= int(config["cut_sample"]))
    )
    test = np.flatnonzero(windows.record_index == test_record_index)
    split = base.Split(train, validation, test)
    for name, indices in split.as_dict().items():
        counts = np.bincount(windows.label[indices], minlength=2)
        if not len(indices) or np.any(counts == 0):
            raise ValueError(f"{subject} {name} lacks a class: {counts.tolist()}")
    cut_train = train[windows.record_index[train] == cut_record_index]
    if int(windows.end[cut_train].max()) != int(config["cut_sample"]):
        raise AssertionError("Training raw support does not end at cut")
    if int(windows.start[validation].min()) != int(config["cut_sample"]):
        raise AssertionError("Validation raw support does not start at cut")
    return split


def training_ranges(
    dataset: DaphnetDataset, subject: str
) -> list[tuple[Any, int, int]]:
    config = subject_config(subject)
    by_id = {record.record_id: record for record in dataset.records}
    return [
        *[
            (by_id[record_id], 0, len(by_id[record_id].y))
            for record_id in config["train_records"]
        ],
        (by_id[config["cut_record"]], 0, int(config["cut_sample"])),
    ]


def fit_subject_scaler(
    dataset: DaphnetDataset, subject: str
) -> tuple[RobustChannelScaler, int]:
    chunks = []
    for record, start, end in training_ranges(dataset, subject):
        mask = record.valid[start:end] & (record.y[start:end] == 0)
        chunks.append(record.x[start:end][mask])
    values = np.concatenate(chunks).astype(np.float64, copy=False)
    center = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1e-6, scale, fallback)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return (
        RobustChannelScaler(center.astype(np.float32), scale.astype(np.float32), 12.0),
        int(len(values)),
    )


def clean_normal_indices(
    dataset: DaphnetDataset,
    windows: base.Windows,
    split: base.Split,
    subject: str,
) -> tuple[np.ndarray, np.ndarray]:
    config = subject_config(subject)
    lookup = {record.record_id: index for index, record in enumerate(dataset.records)}
    cut_record_index = lookup[config["cut_record"]]
    cut_sample = int(config["cut_sample"])
    train = split.train[windows.clean_normal[split.train]]
    validation = split.validation[windows.clean_normal[split.validation]]
    train = train[
        (windows.record_index[train] != cut_record_index)
        | (windows.end[train] <= cut_sample - base.NORMAL_GUARD_SAMPLES)
    ]
    validation = validation[
        (windows.record_index[validation] != cut_record_index)
        | (windows.start[validation] >= cut_sample + base.NORMAL_GUARD_SAMPLES)
    ]
    if not len(train) or not len(validation):
        raise ValueError(f"{subject} lacks clean-normal NBM support")
    return train, validation


def stats(windows: base.Windows, split: base.Split) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, indices in split.as_dict().items():
        counts = np.bincount(windows.label[indices], minlength=2)
        result[name] = {
            "windows": int(len(indices)),
            "non_fog_windows": int(counts[0]),
            "fog_windows": int(counts[1]),
            "fog_percent": float(100.0 * counts[1] / counts.sum()),
        }
    return result


def run() -> None:
    args = parse_args()
    subject = args.subject
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = base.resolve_device(args.device)
    base.set_seed(args.seed, args.deterministic)

    dataset = load_subject(args.data_dir, subject)
    windows = base.make_windows(dataset)
    split = make_subject_split(dataset, windows, subject)
    scaler, scaler_fit_points = fit_subject_scaler(dataset, subject)
    normal_train, normal_validation = clean_normal_indices(
        dataset, windows, split, subject
    )
    window_statistics = stats(windows, split)
    print(
        f"[{subject}] device={device} windows={window_statistics} "
        f"clean-normal train/validation={len(normal_train)}/{len(normal_validation)}",
        flush=True,
    )
    if args.dry_run:
        atomic_json_dump(
            {
                "subject": subject,
                "window_statistics": window_statistics,
                "clean_normal_train": int(len(normal_train)),
                "clean_normal_validation": int(len(normal_validation)),
                "window_axis_centering": bool(args.window_axis_centering),
                "spectral_robust_standardization": bool(
                    args.spectral_robust_standardization
                ),
                "split_config": subject_config(subject),
            },
            output_dir / "DRY_RUN.json",
        )
        return

    split_x = {
        name: cross.extract_experiment_windows(args, dataset, windows, indices, scaler)
        for name, indices in split.as_dict().items()
    }
    shared_spectral_scaler = None
    if args.spectral_robust_standardization:
        spectral_scaler_fit_x = cross.extract_experiment_windows(
            args, dataset, windows, normal_train, scaler
        )
        shared_spectral_scaler = base.fit_spectral_robust_scaler(
            spectral_scaler_fit_x, args.batch_size
        )
        atomic_json_dump(
            shared_spectral_scaler.as_dict(), output_dir / "spectral_scaler.json"
        )
    folds = cross.blocked_folds(windows, split, cross.N_FOLDS)
    train_position = {int(index): position for position, index in enumerate(split.train)}
    oof_shape = (len(split.train), dataset.n_channels, base.N_FREQ)
    oof_observed = np.full(oof_shape, np.nan, dtype=np.float32)
    oof_reconstruction = np.full(oof_shape, np.nan, dtype=np.float32)
    oof_residual = np.full(oof_shape, np.nan, dtype=np.float32)
    fold_assignment = np.full(len(split.train), -1, dtype=np.int8)
    fold_rows: list[dict[str, Any]] = []
    audit_arrays: dict[str, np.ndarray] = {}

    for fold_number, heldout in enumerate(folds, start=1):
        fold_dir = output_dir / f"fold_{fold_number:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        candidates = cross.exclude_near_spans(
            windows,
            normal_train,
            cross.contiguous_spans(windows, heldout),
            cross.PURGE_SAMPLES,
        )
        fit_indices, inner_validation_indices = cross.make_inner_split(
            windows, candidates
        )
        cross.assert_no_support_leakage(
            windows, fit_indices, heldout, cross.PURGE_SAMPLES
        )
        fold_args = cross.fold_args(args, args.seed + fold_number)
        fit_x = cross.extract_experiment_windows(
            fold_args, dataset, windows, fit_indices, scaler
        )
        inner_validation_x = cross.extract_experiment_windows(
            fold_args, dataset, windows, inner_validation_indices, scaler
        )
        fold_spectral_scaler = shared_spectral_scaler
        model, training = base.train_gru_nbm(
            fold_args,
            fit_x,
            inner_validation_x,
            fold_dir,
            device,
            spectral_scaler=fold_spectral_scaler,
        )
        observed, reconstruction, residual = base.extract_residuals(
            fold_args,
            model,
            cross.extract_experiment_windows(
                fold_args, dataset, windows, heldout, scaler
            ),
            device,
            spectral_scaler=fold_spectral_scaler,
        )
        positions = np.asarray([train_position[int(index)] for index in heldout])
        oof_observed[positions] = observed
        oof_reconstruction[positions] = reconstruction
        oof_residual[positions] = residual
        fold_assignment[positions] = fold_number
        counts = np.bincount(windows.label[heldout], minlength=2)
        fold_rows.append(
            {
                "fold": fold_number,
                "heldout_windows": int(len(heldout)),
                "heldout_non_fog": int(counts[0]),
                "heldout_fog": int(counts[1]),
                "nbm_candidates_after_outer_purge": int(len(candidates)),
                "nbm_gradient_train_windows": int(len(fit_indices)),
                "nbm_inner_validation_windows": int(len(inner_validation_indices)),
                "best_epoch": int(training["best_epoch"]),
                "best_inner_validation_smooth_l1": float(
                    training["best_validation_smooth_l1"]
                ),
            }
        )
        audit_arrays.update(
            {
                f"fold_{fold_number:02d}_heldout": heldout,
                f"fold_{fold_number:02d}_nbm_gradient_train": fit_indices,
                f"fold_{fold_number:02d}_nbm_inner_validation": inner_validation_indices,
            }
        )
        print(
            f"[{subject} crossfit] fold={fold_number}/5 heldout={len(heldout)} "
            f"NBM_fit/inner_val={len(fit_indices)}/{len(inner_validation_indices)}",
            flush=True,
        )
    if np.isnan(oof_residual).any() or np.any(fold_assignment < 1):
        raise AssertionError("Incomplete OOF features")

    final_dir = output_dir / "final_nbm"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_train_x = cross.extract_experiment_windows(
        args, dataset, windows, normal_train, scaler
    )
    final_validation_x = cross.extract_experiment_windows(
        args, dataset, windows, normal_validation, scaler
    )
    final_spectral_scaler = shared_spectral_scaler
    final_model, final_training = base.train_gru_nbm(
        args,
        final_train_x,
        final_validation_x,
        final_dir,
        device,
        spectral_scaler=final_spectral_scaler,
    )
    validation_observed, validation_reconstruction, validation_residual = (
        base.extract_residuals(
            args,
            final_model,
            split_x["validation"],
            device,
            spectral_scaler=final_spectral_scaler,
        )
    )
    test_observed, test_reconstruction, test_residual = base.extract_residuals(
        args,
        final_model,
        split_x["test"],
        device,
        spectral_scaler=final_spectral_scaler,
    )
    residuals = {
        "train": oof_residual,
        "validation": validation_residual,
        "test": test_residual,
    }
    labels = {
        name: windows.label[indices].copy() for name, indices in split.as_dict().items()
    }
    diagnostics = {
        name: base.residual_diagnostics(residuals[name], labels[name])
        for name in residuals
    }
    atomic_json_dump(diagnostics, output_dir / "residual_diagnostics.json")
    atomic_npz_save(
        output_dir / "spectral_residuals_crossfit.npz",
        train_observed_log_power=oof_observed,
        train_reconstructed_log_power=oof_reconstruction,
        train_signed_residual=oof_residual,
        train_y=labels["train"],
        train_window_index=split.train,
        train_fold_assignment=fold_assignment,
        validation_observed_log_power=validation_observed,
        validation_reconstructed_log_power=validation_reconstruction,
        validation_signed_residual=validation_residual,
        validation_y=labels["validation"],
        validation_window_index=split.validation,
        test_observed_log_power=test_observed,
        test_reconstructed_log_power=test_reconstruction,
        test_signed_residual=test_residual,
        test_y=labels["test"],
        test_window_index=split.test,
    )
    atomic_npz_save(
        output_dir / "crossfit_indices.npz",
        train=split.train,
        validation=split.validation,
        test=split.test,
        final_nbm_train_clean_normal=normal_train,
        final_nbm_validation_clean_normal=normal_validation,
        train_fold_assignment=fold_assignment,
        **audit_arrays,
    )
    base.write_csv(output_dir / "crossfit_folds.csv", fold_rows)

    tcn_training, metrics, predictions = base.train_classifier(
        args, residuals, labels, output_dir, device
    )
    atomic_npz_save(output_dir / "predictions.npz", **predictions)
    base.write_csv(output_dir / "tcnm_history.csv", tcn_training["history"])
    base.plot_history(
        final_training["history"],
        "train_smooth_l1",
        "validation_smooth_l1",
        f"{subject} final GRU-NBM spectral reconstruction loss",
        output_dir / "final_gru_nbm_loss.png",
    )
    base.plot_history(
        tcn_training["history"],
        "train_bce",
        "validation_bce",
        f"{subject} blocked-crossfit TCN-M loss",
        output_dir / "tcnm_loss.png",
    )
    base.plot_confusion(metrics["test"], output_dir / "test_confusion_matrix.png")

    config_values = subject_config(subject)
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject": subject,
        "data_dir": str(args.data_dir.resolve()),
        "dataset_fingerprint_sha256": dataset_fingerprint(args.data_dir),
        "records_loaded": [record.record_id for record in dataset.records],
        "split": {
            **config_values,
            "strategy": "chronological record/block split with disjoint raw support",
            "cut_is_label_aware": True,
            "ignored_records_are_not_used": True,
        },
        "windowing": {
            "sampling_rate_hz": 64,
            "seconds": 2.0,
            "samples": 128,
            "stride_seconds": 1.0,
            "label": "FOG if at least 50% of all 128 samples are FOG",
            "statistics": window_statistics,
            "per_window_per_axis_temporal_centering": {
                "enabled": bool(args.window_axis_centering),
                "ordering": "after train-fitted Robust scaling and before corruption or FFT",
                "formula": "x_centered[b,t,c] = x_scaled[b,t,c] - mean_t(x_scaled[b,:,c])",
                "fit_parameters": None,
            },
        },
        "scaler": {
            **scaler.as_dict(),
            "fit_split": "train only",
            "fit_class": "valid Non-FoG sample points only",
            "fit_points": scaler_fit_points,
        },
        "spectrum": {
            "shape": [9, 65],
            "formula": "log1p(abs(rfft(hann*x))**2 / sum(hann**2))",
            "range_hz": [0.0, 32.0],
            "resolution_hz": 0.5,
            "robust_standardization": {
                "enabled": bool(args.spectral_robust_standardization),
                "fit_scope": (
                    "one shared scaler fitted on all eligible outer-training clean Non-FoG spectra; reused by every crossfit NBM and the final NBM"
                    if args.spectral_robust_standardization
                    else None
                ),
                "formula": "(log_power - per_axis_frequency_median) / (IQR/1.349 with standard-deviation fallback)",
                "clipping": None,
                "residual_error_statistics_fitted": False,
                "final_scaler": (
                    final_spectral_scaler.as_dict()
                    if final_spectral_scaler is not None
                    else None
                ),
            },
        },
        "crossfit": {
            "folds": 5,
            "blocking": "within every training record, five contiguous chronological blocks",
            "purge_samples_each_side": 32,
            "purge_seconds_each_side": 0.5,
            "fold_nbm_early_stopping": "chronological internal 15% post-purge fit-pool subset",
            "folds_detail": fold_rows,
            "oof_windows": int(len(split.train)),
            "oof_missing_windows": 0,
            "validation_test_model": "single final NBM fitted on all eligible training clean-normal windows",
        },
        "gru_nbm": {
            "input_output": (
                "[B,9,65] training-only Robust-standardized log-power spectrum"
                if args.spectral_robust_standardization
                else "[B,9,65] log-power spectrum"
            ),
            "hidden_bottleneck": 64,
            "decoder_output_activation": (
                "identity" if args.spectral_robust_standardization else "softplus"
            ),
            "corruption": {
                "gaussian_noise_std": args.noise_std,
                "time_mask_probability": args.time_mask_probability,
                "time_mask_samples": [4, 12],
                "channel_mask_probability": args.channel_mask_probability,
            },
            "final_training": final_training,
        },
        "residual": {
            "formula": (
                "observed - reconstructed training-only Robust-standardized log power"
                if args.spectral_robust_standardization
                else "observed - reconstructed log power"
            ),
            "error_distribution_standardization": None,
            "clipping": None,
        },
        "tcnm": {
            "input": "one [9,65] spectral residual",
            "frequency_dilations": [1, 2, 4, 8],
            "training": tcn_training,
        },
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
    summary = f"""# {subject} blocked-crossfit spectral GRU-NBM + TCN-M

| Split | Accuracy | FoG Recall | Specificity | PR-AUC | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Validation | {validation['accuracy']:.6f} | {validation['fog_recall']:.6f} | {validation['specificity']:.6f} | {validation['pr_auc']:.6f} | {validation['f1']:.6f} | {validation['balanced_accuracy']:.6f} |
| Test | {test['accuracy']:.6f} | {test['fog_recall']:.6f} | {test['specificity']:.6f} | {test['pr_auc']:.6f} | {test['f1']:.6f} | {test['balanced_accuracy']:.6f} |

- Validation-selected threshold: {tcn_training['selected_threshold']:.4f}.
- Per-window per-axis temporal centering: {'enabled' if args.window_axis_centering else 'disabled'} after Robust scaling and before corruption/FFT.
- Per-axis/per-frequency spectral Robust standardization: {'enabled' if args.spectral_robust_standardization else 'disabled'} with training-only statistics; no residual-error mean/std is fitted.
- Test confusion: TN={test['tn']}, FP={test['fp']}, FN={test['fn']}, TP={test['tp']}.
- Train/validation/test windows: {window_statistics['train']['windows']}/{window_statistics['validation']['windows']}/{window_statistics['test']['windows']}.
- All {len(split.train)} TCN training residuals are blocked OOF with 0.5 s outer purge.
- Final NBM best epoch: {final_training['best_epoch']}; TCN-M best epoch: {tcn_training['best_epoch']}.
- One subject, one chronological outer split, and one seed; ignored records in the predefined split are not used.
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
                str(path.relative_to(output_dir)): sha256_file(path)
                for path in sorted(output_dir.rglob("*"))
                if path.is_file()
            },
        },
        output_dir / "DONE.json",
    )
    print(
        f"COMPLETE {subject} accuracy={test['accuracy']:.6f} "
        f"recall={test['fog_recall']:.6f} specificity={test['specificity']:.6f} "
        f"pr_auc={test['pr_auc']:.6f} confusion={test['confusion_matrix']}",
        flush=True,
    )


if __name__ == "__main__":
    run()
