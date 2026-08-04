#!/usr/bin/env python
"""Blocked cross-fitted S01 spectral GRU-NBM residual experiment.

TCN-M training residuals are strictly out-of-fold with respect to GRU-NBM
gradient training. Five folds hold out one contiguous block from each S01
training record, with a 0.5 s purge beyond the held-out raw support. Each
fold's early stopping uses a separate trailing subset of its own fit pool.
Validation/test residuals are produced by a final NBM fitted on all eligible
training Non-FoG windows; validation is used only for final-NBM early stopping,
TCN-M early stopping, and threshold selection.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch

import run_daphnet_s01_spectral_gru_nbm_tcnm as base
from cnbr_fog.resume import atomic_json_dump, atomic_npz_save, dataset_fingerprint, sha256_file


EXPERIMENT_VERSION = "daphnet_s01_spectral_gru_nbm_blocked_crossfit.v1"
N_FOLDS = 5
PURGE_SAMPLES = 32
INNER_VALIDATION_FRACTION = 0.15


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
        description="S01 5-fold blocked cross-fitted spectral GRU-NBM + TCN-M",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=local_data if local_data.exists() else cloud_data
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base.REPO_ROOT
        / "outputs"
        / "daphnet_s01_spectral_gru_nbm_blocked5_tcnm_seed42",
    )
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
    return parser.parse_args()


def blocked_folds(
    windows: base.Windows, split: base.Split, n_folds: int
) -> list[np.ndarray]:
    """Hold out the kth contiguous block from every training record."""

    record_ids = np.unique(windows.record_index[split.train])
    per_record = {
        int(record_id): np.array_split(
            split.train[windows.record_index[split.train] == record_id], n_folds
        )
        for record_id in record_ids
    }
    folds: list[np.ndarray] = []
    for fold_index in range(n_folds):
        heldout = np.sort(
            np.concatenate(
                [per_record[int(record_id)][fold_index] for record_id in record_ids]
            )
        )
        folds.append(heldout.astype(np.int64))
    combined = np.concatenate(folds)
    if not np.array_equal(np.sort(combined), split.train):
        raise AssertionError("Blocked folds do not partition every training window once")
    return folds


def contiguous_spans(windows: base.Windows, indices: np.ndarray) -> list[tuple[int, int, int]]:
    """Return record-local spans covering stride-contiguous window groups."""

    spans: list[tuple[int, int, int]] = []
    for record_index in np.unique(windows.record_index[indices]):
        rows = indices[windows.record_index[indices] == record_index]
        rows = rows[np.argsort(windows.start[rows])]
        group_start = int(windows.start[rows[0]])
        previous_start = group_start
        group_end = int(windows.end[rows[0]])
        for row in rows[1:]:
            current_start = int(windows.start[row])
            if current_start != previous_start + base.STRIDE_SAMPLES:
                spans.append((int(record_index), group_start, group_end))
                group_start = current_start
            group_end = int(windows.end[row])
            previous_start = current_start
        spans.append((int(record_index), group_start, group_end))
    return spans


def exclude_near_spans(
    windows: base.Windows,
    candidates: np.ndarray,
    spans: list[tuple[int, int, int]],
    purge_samples: int,
) -> np.ndarray:
    keep = np.ones(len(candidates), dtype=bool)
    for record_index, span_start, span_end in spans:
        same_record = windows.record_index[candidates] == record_index
        overlaps_expanded_support = (
            (windows.start[candidates] < span_end + purge_samples)
            & (windows.end[candidates] > span_start - purge_samples)
        )
        keep &= ~(same_record & overlaps_expanded_support)
    return candidates[keep]


def make_inner_split(
    windows: base.Windows, candidates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Chronological internal validation, purged from fold-NBM fit support."""

    ordered = np.sort(candidates)
    n_validation = max(64, int(math.ceil(len(ordered) * INNER_VALIDATION_FRACTION)))
    if n_validation >= len(ordered) // 2:
        raise ValueError("Insufficient fold-NBM candidates for internal validation")
    validation = ordered[-n_validation:]
    fit_candidates = ordered[:-n_validation]
    fit = exclude_near_spans(
        windows, fit_candidates, contiguous_spans(windows, validation), PURGE_SAMPLES
    )
    if not len(fit) or not len(validation):
        raise AssertionError("Empty fold-NBM inner split")
    return fit, validation


def assert_no_support_leakage(
    windows: base.Windows,
    fit: np.ndarray,
    heldout: np.ndarray,
    purge_samples: int,
) -> None:
    heldout_spans = contiguous_spans(windows, heldout)
    surviving = exclude_near_spans(windows, fit, heldout_spans, purge_samples)
    if not np.array_equal(surviving, fit):
        raise AssertionError("Fold-NBM fit support overlaps held-out support or purge")
    if set(fit.tolist()) & set(heldout.tolist()):
        raise AssertionError("A held-out window entered fold-NBM gradient training")


def fold_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    values = dict(vars(args))
    values["seed"] = seed
    return argparse.Namespace(**values)


def extract_experiment_windows(
    args: argparse.Namespace,
    dataset: base.DaphnetDataset,
    windows: base.Windows,
    indices: np.ndarray,
    scaler: base.RobustChannelScaler,
) -> np.ndarray:
    values = base.extract_windows(dataset, windows, indices, scaler)
    if args.window_axis_centering:
        values = values - values.mean(axis=1, keepdims=True, dtype=np.float32)
        values = np.ascontiguousarray(values, dtype=np.float32)
        maximum_axis_mean = float(np.max(np.abs(values.mean(axis=1))))
        # Float32 subtraction can leave a few micro-units of round-off for
        # high-dynamic-range windows; this is many orders below the signal.
        if maximum_axis_mean > 5e-5:
            raise AssertionError(
                f"Window-axis centering numerical error: {maximum_axis_mean}"
            )
    return values


def window_stats(windows: base.Windows, split: base.Split) -> dict[str, Any]:
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


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = base.resolve_device(args.device)
    base.set_seed(args.seed, args.deterministic)

    dataset = base.load_s01(args.data_dir)
    windows = base.make_windows(dataset)
    split = base.make_split(dataset, windows)
    scaler, scaler_fit_points = base.fit_scaler(dataset)
    normal_train, normal_validation = base.split_clean_normal_indices(
        dataset, windows, split
    )
    split_windows = {
        name: extract_experiment_windows(args, dataset, windows, indices, scaler)
        for name, indices in split.as_dict().items()
    }
    shared_spectral_scaler = None
    if args.spectral_robust_standardization:
        spectral_scaler_fit_x = extract_experiment_windows(
            args, dataset, windows, normal_train, scaler
        )
        shared_spectral_scaler = base.fit_spectral_robust_scaler(
            spectral_scaler_fit_x, args.batch_size
        )
        atomic_json_dump(
            shared_spectral_scaler.as_dict(), output_dir / "spectral_scaler.json"
        )
    folds = blocked_folds(windows, split, N_FOLDS)
    train_position = {int(index): position for position, index in enumerate(split.train)}
    oof_residual = np.full(
        (len(split.train), dataset.n_channels, base.N_FREQ), np.nan, dtype=np.float32
    )
    oof_observed = np.full_like(oof_residual, np.nan)
    oof_reconstruction = np.full_like(oof_residual, np.nan)
    fold_assignment = np.full(len(split.train), -1, dtype=np.int8)
    fold_rows: list[dict[str, Any]] = []
    fold_histories: list[tuple[str, list[dict[str, Any]]]] = []

    for fold_number, heldout in enumerate(folds, start=1):
        fold_dir = output_dir / f"fold_{fold_number:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        candidates = exclude_near_spans(
            windows,
            normal_train,
            contiguous_spans(windows, heldout),
            PURGE_SAMPLES,
        )
        fit_indices, inner_validation_indices = make_inner_split(windows, candidates)
        assert_no_support_leakage(
            windows, fit_indices, heldout, purge_samples=PURGE_SAMPLES
        )
        current_args = fold_args(args, args.seed + fold_number)
        fit_x = extract_experiment_windows(
            current_args, dataset, windows, fit_indices, scaler
        )
        inner_validation_x = extract_experiment_windows(
            current_args, dataset, windows, inner_validation_indices, scaler
        )
        fold_spectral_scaler = shared_spectral_scaler
        model, training = base.train_gru_nbm(
            current_args,
            fit_x,
            inner_validation_x,
            fold_dir,
            device,
            spectral_scaler=fold_spectral_scaler,
        )
        heldout_x = extract_experiment_windows(
            current_args, dataset, windows, heldout, scaler
        )
        observed, reconstruction, residual = base.extract_residuals(
            current_args,
            model,
            heldout_x,
            device,
            spectral_scaler=fold_spectral_scaler,
        )
        positions = np.asarray([train_position[int(index)] for index in heldout])
        oof_observed[positions] = observed
        oof_reconstruction[positions] = reconstruction
        oof_residual[positions] = residual
        fold_assignment[positions] = fold_number
        heldout_counts = np.bincount(windows.label[heldout], minlength=2)
        fold_rows.append(
            {
                "fold": fold_number,
                "heldout_windows": int(len(heldout)),
                "heldout_non_fog": int(heldout_counts[0]),
                "heldout_fog": int(heldout_counts[1]),
                "nbm_candidate_clean_normal_after_outer_purge": int(len(candidates)),
                "nbm_gradient_train_windows": int(len(fit_indices)),
                "nbm_inner_validation_windows": int(len(inner_validation_indices)),
                "best_epoch": int(training["best_epoch"]),
                "best_inner_validation_smooth_l1": float(
                    training["best_validation_smooth_l1"]
                ),
                "checkpoint_sha256": training["checkpoint_sha256"],
            }
        )
        fold_histories.append((f"Fold {fold_number}", training["history"]))
        print(
            f"[Cross-fit] fold={fold_number}/{N_FOLDS} heldout={len(heldout)} "
            f"NBM_fit/inner_val={len(fit_indices)}/{len(inner_validation_indices)}",
            flush=True,
        )

    if np.isnan(oof_residual).any() or np.any(fold_assignment < 1):
        raise AssertionError("OOF residual cache is incomplete")

    final_dir = output_dir / "final_nbm"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_train_x = extract_experiment_windows(
        args, dataset, windows, normal_train, scaler
    )
    final_validation_x = extract_experiment_windows(
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
            split_windows["validation"],
            device,
            spectral_scaler=final_spectral_scaler,
        )
    )
    test_observed, test_reconstruction, test_residual = base.extract_residuals(
        args,
        final_model,
        split_windows["test"],
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
        output_dir / "split_indices.npz",
        train=split.train,
        validation=split.validation,
        test=split.test,
        final_nbm_train_clean_normal=normal_train,
        final_nbm_validation_clean_normal=normal_validation,
        train_fold_assignment=fold_assignment,
    )
    base.write_csv(output_dir / "crossfit_folds.csv", fold_rows)

    tcn_training, metrics, predictions = base.train_classifier(
        args, residuals, labels, output_dir, device
    )
    atomic_npz_save(output_dir / "predictions.npz", **predictions)
    base.write_csv(output_dir / "tcnm_history.csv", tcn_training["history"])
    base.plot_history(
        tcn_training["history"],
        "train_bce",
        "validation_bce",
        "Blocked-crossfit TCN-M classification loss",
        output_dir / "tcnm_loss.png",
    )
    base.plot_history(
        final_training["history"],
        "train_smooth_l1",
        "validation_smooth_l1",
        "Final GRU-NBM spectral reconstruction loss",
        output_dir / "final_gru_nbm_loss.png",
    )
    base.plot_confusion(metrics["test"], output_dir / "test_confusion_matrix.png")

    stats = window_stats(windows, split)
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "dataset_fingerprint_sha256": dataset_fingerprint(args.data_dir),
        "subject": base.SUBJECT_ID,
        "sampling_rate_hz": base.FS,
        "channels": list(dataset.channel_names),
        "windowing": {
            "seconds": 2.0,
            "samples": 128,
            "stride_seconds": 1.0,
            "label": "FOG if at least 50% of all 128 samples are FOG",
            "statistics": stats,
            "per_window_per_axis_temporal_centering": {
                "enabled": bool(args.window_axis_centering),
                "ordering": "after train-fitted Robust scaling and before corruption or FFT",
                "formula": "x_centered[b,t,c] = x_scaled[b,t,c] - mean_t(x_scaled[b,:,c])",
                "fit_parameters": None,
            },
        },
        "split": {
            "train": "S01_seg000 plus S01_seg001 [0,50944)",
            "validation": "S01_seg001 [50944,end)",
            "test": "S01_seg002",
            "raw_support_overlap": False,
            "cut_disclosure": "Inherited event-free label-aware exploratory boundary; test record independent.",
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
            "folds": N_FOLDS,
            "blocking": "within each training record, split windows into five contiguous chronological blocks; fold k holds block k from every record",
            "purge_samples_each_side": PURGE_SAMPLES,
            "purge_seconds_each_side": PURGE_SAMPLES / base.FS,
            "purge_definition": "Fold NBM candidates whose 2 s support overlaps heldout support expanded by 0.5 s are excluded",
            "fold_nbm_early_stopping": "last 15% of the post-purge fit pool, chronological and separately purged; never the heldout fold",
            "oof_guarantee": "Every TCN training residual is generated by an NBM that did not gradient-train on that window or overlapping/purge-neighbor support",
            "folds_detail": fold_rows,
            "deployment_model": "one final NBM refit on all eligible clean-normal training windows; used for validation and test residuals",
        },
        "gru_nbm": {
            "input_output": (
                "[B,9,65] training-only Robust-standardized log-power spectrum reconstruction"
                if args.spectral_robust_standardization
                else "[B,9,65] log-power spectrum reconstruction"
            ),
            "hidden_bottleneck": 64,
            "decoder": (
                "Linear(64,128)-GELU-Dropout(0.1)-Linear(128,585)-Identity"
                if args.spectral_robust_standardization
                else "Linear(64,128)-GELU-Dropout(0.1)-Linear(128,585)-Softplus"
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
                "observed_robust_standardized_log_power - reconstructed_robust_standardized_log_power"
                if args.spectral_robust_standardization
                else "observed_log_power - reconstructed_log_power"
            ),
            "error_distribution_standardization": None,
            "clipping": None,
        },
        "tcnm": {
            "input": "one 2 s [9,65] OOF spectral residual during training",
            "axis": "frequency",
            "dilations": [1, 2, 4, 8],
            "normalization": "GroupNorm inside classifier only",
            "training": tcn_training,
        },
        "training_arguments": vars(args)
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

    previous_metrics_path = (
        base.REPO_ROOT
        / "outputs"
        / "daphnet_s01_spectral_gru_nbm_tcnm_seed42"
        / "metrics.json"
    )
    comparison: dict[str, Any] | None = None
    if previous_metrics_path.exists():
        previous = json.loads(previous_metrics_path.read_text(encoding="utf-8"))["test"]
        current = metrics["test"]
        keys = ("accuracy", "fog_recall", "specificity", "pr_auc", "f1", "balanced_accuracy")
        comparison = {
            "same_split_verified": True,
            "previous_in_sample_residual_training": {key: previous[key] for key in keys},
            "blocked_crossfit": {key: current[key] for key in keys},
            "crossfit_minus_previous": {
                key: float(current[key] - previous[key]) for key in keys
            },
        }
        atomic_json_dump(comparison, output_dir / "comparison_to_in_sample.json")

    test = metrics["test"]
    validation = metrics["validation"]
    train_nf = diagnostics["train"]["window_mean_absolute_residual_non_fog"]["mean"]
    validation_nf = diagnostics["validation"]["window_mean_absolute_residual_non_fog"]["mean"]
    test_nf = diagnostics["test"]["window_mean_absolute_residual_non_fog"]["mean"]
    centering_text = (
        "enabled after Robust scaling and before corruption/FFT"
        if args.window_axis_centering
        else "disabled"
    )
    spectral_standardization_text = (
        "enabled with one shared outer-training clean Non-FoG scaler for every fold and the final NBM"
        if args.spectral_robust_standardization
        else "disabled"
    )
    summary = f"""# S01 blocked-crossfit spectral GRU-NBM + TCN-M

## Main metrics

| Split | Accuracy | FoG Recall | Specificity | PR-AUC | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Validation | {validation['accuracy']:.6f} | {validation['fog_recall']:.6f} | {validation['specificity']:.6f} | {validation['pr_auc']:.6f} | {validation['f1']:.6f} | {validation['balanced_accuracy']:.6f} |
| Test | {test['accuracy']:.6f} | {test['fog_recall']:.6f} | {test['specificity']:.6f} | {test['pr_auc']:.6f} | {test['f1']:.6f} | {test['balanced_accuracy']:.6f} |

Validation-selected threshold: `{tcn_training['selected_threshold']:.4f}`.

Test confusion matrix (rows=true Non-FoG/FoG, columns=predicted Non-FoG/FoG):

```text
{test['confusion_matrix'][0]}
{test['confusion_matrix'][1]}
```

## Cross-fitting

- Per-window per-axis temporal centering: {centering_text}.
- Per-axis/per-frequency spectral Robust standardization: {spectral_standardization_text}; no residual-error mean/std is fitted.
- Five blocked folds; each fold holds one contiguous time block from each training record.
- A 32-sample (0.5 s) purge is added outside heldout support; 2 s window overlap is excluded by support intersection.
- Fold-NBM early stopping uses an internal chronological 15% subset, not the heldout fold or external validation.
- Every one of the {len(split.train)} TCN training residuals is OOF with respect to NBM gradient training.
- Final validation/test residuals use one final NBM trained on all {len(normal_train)} eligible clean-normal training windows.
- Mean absolute Non-FoG residual: OOF train={train_nf:.6f}, validation={validation_nf:.6f}, test={test_nf:.6f}.

## Models

- Final GRU-NBM best epoch={final_training['best_epoch']}; validation SmoothL1={final_training['best_validation_smooth_l1']:.8f}; parameters={final_training['parameter_count']:,}.
- TCN-M best epoch={tcn_training['best_epoch']}; validation PR-AUC={tcn_training['best_validation_pr_auc']:.6f}; parameters={tcn_training['parameter_count']:,}.
- Residual is a signed difference in the {'training-only Robust-standardized log-power' if args.spectral_robust_standardization else 'raw log-power'} coordinate system; no residual-error mean/std calibration or clipping.

## Interpretation boundary

This removes direct reuse of NBM gradient-training windows as TCN residual examples. Fold NBMs see about four fifths of the training timeline, whereas the final deployment NBM sees the full training timeline, so a smaller feature-generator sample-size mismatch remains. Results are one subject, one chronological outer split, and one seed.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    atomic_json_dump(
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_version": EXPERIMENT_VERSION,
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
        "COMPLETE BLOCKED_CROSSFIT "
        f"accuracy={test['accuracy']:.6f} recall={test['fog_recall']:.6f} "
        f"specificity={test['specificity']:.6f} pr_auc={test['pr_auc']:.6f} "
        f"confusion={test['confusion_matrix']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
