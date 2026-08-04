#!/usr/bin/env python
"""Train a within-S01 normal-window DAE and residual TCN-M detector.

The referenced framework defines a same-window denoising reconstruction task,
not a context-to-future forecaster.  Consequently, this experiment reconstructs
each arriving two-second target block, calibrates a static residual scale from
clean-normal training reconstruction errors, and passes the standardized
reconstruction residual to the unchanged TCN-M classifier.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cnbr_fog.denoising_autoencoder import (  # noqa: E402
    ChannelZScoreScaler,
    CorruptionConfig,
    TCNDenoisingAutoencoder,
    corrupt_batch,
    dae_combined_loss,
)
from cnbr_fog.nbm_representations import calibrate_fixed_sigma  # noqa: E402
from cnbr_fog.resume import (  # noqa: E402
    atomic_json_dump,
    atomic_npz_save,
    atomic_torch_save,
    canonical_fingerprint,
    dataset_fingerprint,
    sha256_file,
)
from cnbr_fog.rf125_classifiers import (  # noqa: E402
    DEFAULT_DILATIONS,
    build_rf125_classifier,
)
import run_daphnet_s01_gru_h200_tcnm as core  # noqa: E402


EXPERIMENT_VERSION = "daphnet_s01_dae_tcnm.v1"
FRAMEWORK_DOCUMENT = REPO_ROOT / "NonFoG_Denoising_Autoencoder_65Hz_Training_Framework.md"
RESIDUAL_CLIP = 12.0
SCALER_EPSILON = 1e-8
FIXED_SIGMA_EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Within-S01 DAE reconstruction-residual TCN-M experiment",
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
        default=REPO_ROOT / "outputs" / "daphnet_s01_dae_tcnm_seed42",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dae-epochs", type=int, default=100)
    parser.add_argument("--dae-patience", type=int, default=15)
    parser.add_argument("--dae-lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--dae-dropout", type=float, default=0.10)
    parser.add_argument("--classifier-epochs", type=int, default=12)
    parser.add_argument("--classifier-patience", type=int, default=4)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-hidden", type=int, default=48)
    parser.add_argument("--classifier-dropout", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dae-only",
        action="store_true",
        help="Train and evaluate clean-normal DAE loss only; skip residuals and TCN-M.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "batch_size": args.batch_size,
        "dae_epochs": args.dae_epochs,
        "dae_patience": args.dae_patience,
        "latent_dim": args.latent_dim,
        "classifier_epochs": args.classifier_epochs,
        "classifier_patience": args.classifier_patience,
        "classifier_hidden": args.classifier_hidden,
    }
    for name, value in positive_ints.items():
        if int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    for name in ("dae_lr", "classifier_lr"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative")
    for name in ("dae_dropout", "classifier_dropout"):
        if not 0 <= float(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1)")
    if not FRAMEWORK_DOCUMENT.is_file():
        raise FileNotFoundError(FRAMEWORK_DOCUMENT)


def resolve_device(specification: str) -> torch.device:
    return core.resolve_device(specification)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_npy_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def extract_target_windows(
    dataset: core.DaphnetDataset,
    windows: core.WindowTable,
    indices: np.ndarray,
) -> np.ndarray:
    output = np.empty(
        (len(indices), dataset.n_channels, core.TARGET_SAMPLES),
        dtype=np.float32,
    )
    for row, window_index in enumerate(np.asarray(indices, dtype=np.int64)):
        record = dataset.records[int(windows.record_index[window_index])]
        start = int(windows.target_start[window_index])
        end = int(windows.target_end[window_index])
        signal = record.x[start:end]
        if signal.shape != (core.TARGET_SAMPLES, dataset.n_channels):
            raise AssertionError(f"unexpected target shape {signal.shape}")
        if not np.all(record.valid[start:end]):
            raise AssertionError("window includes invalid target samples")
        output[row] = signal.T
    if not np.isfinite(output).all():
        raise ValueError("target windows contain NaN or Inf")
    return output


def array_loader(
    x: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(x)).float()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )


def dae_epoch(
    model: TCNDenoisingAutoencoder,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    corruption: CorruptionConfig | None = None,
    corruption_generator: torch.Generator | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> tuple[dict[str, float], list[int]]:
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ("total", "time", "difference", "frequency")}
    total_windows = 0
    mode_counts = np.zeros(4, dtype=np.int64)
    for (clean,) in loader:
        clean = clean.to(device, non_blocking=True)
        if corruption is None:
            network_input = clean
        else:
            network_input, modes = corrupt_batch(
                clean,
                corruption,
                generator=corruption_generator,
            )
            mode_counts += np.bincount(
                modes.detach().cpu().numpy(), minlength=4
            ).astype(np.int64)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device.type,
                enabled=bool(amp and device.type == "cuda"),
            ):
                reconstruction, _ = model(network_input)
            losses = dae_combined_loss(reconstruction, clean)
            if training:
                assert grad_scaler is not None
                grad_scaler.scale(losses["total"]).backward()
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
        batch = int(clean.shape[0])
        total_windows += batch
        for name, value in losses.items():
            totals[name] += float(value.detach()) * batch
    averages = {
        name: value / max(total_windows, 1) for name, value in totals.items()
    }
    return averages, mode_counts.astype(int).tolist()


def make_device_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def train_dae(
    args: argparse.Namespace,
    train_clean: np.ndarray,
    validation_clean: np.ndarray,
    output_dir: Path,
    protocol_fingerprint: str,
    device: torch.device,
    corruption: CorruptionConfig,
) -> tuple[TCNDenoisingAutoencoder, dict[str, Any]]:
    core.set_seed(args.seed, args.deterministic)
    model = TCNDenoisingAutoencoder(
        in_channels=9,
        input_samples=core.TARGET_SAMPLES,
        latent_dim=args.latent_dim,
        dropout=args.dae_dropout,
        residual_kernel_size=3,
        group_norm_groups=8,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.dae_lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(args.amp and device.type == "cuda")
    )
    validation_loader = array_loader(
        validation_clean,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    best_path = output_dir / "dae_best.pt"
    last_path = output_dir / "dae_last.pt"
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    stopped_early = False
    started = time.perf_counter()

    for epoch in range(1, args.dae_epochs + 1):
        shuffle_seed = args.seed + epoch
        corruption_seed = args.seed + 100_000 + epoch
        train_loader = array_loader(
            train_clean,
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            seed=shuffle_seed,
        )
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        train_loss, corruption_counts = dae_epoch(
            model,
            train_loader,
            device,
            amp=args.amp,
            corruption=corruption,
            corruption_generator=make_device_generator(device, corruption_seed),
            optimizer=optimizer,
            grad_scaler=grad_scaler,
        )
        with torch.no_grad():
            validation_loss, _ = dae_epoch(
                model,
                validation_loader,
                device,
                amp=args.amp,
            )
        scheduler.step(validation_loss["total"])
        improved = validation_loss["total"] < best_loss - 1e-8
        row: dict[str, Any] = {
            "epoch": epoch,
            "shuffle_seed": shuffle_seed,
            "corruption_seed": corruption_seed,
            "learning_rate_used": learning_rate_used,
            "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}_loss": value for key, value in train_loss.items()},
            **{
                f"validation_{key}_loss": value
                for key, value in validation_loss.items()
            },
            "corruption_time_mask_windows": corruption_counts[0],
            "corruption_channel_mask_windows": corruption_counts[1],
            "corruption_gaussian_windows": corruption_counts[2],
            "corruption_clean_windows": corruption_counts[3],
            "improved": improved,
        }
        history.append(row)
        checkpoint = {
            "experiment_version": EXPERIMENT_VERSION,
            "protocol_fingerprint": protocol_fingerprint,
            "seed": args.seed,
            "epoch": epoch,
            "validation_total_loss": validation_loss["total"],
            "architecture": model.architecture_config(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }
        atomic_torch_save(checkpoint, last_path)
        if improved:
            best_loss = validation_loss["total"]
            best_epoch = epoch
            bad_epochs = 0
            atomic_torch_save(checkpoint, best_path)
        else:
            bad_epochs += 1
        print(
            f"[DAE] epoch={epoch:03d} train={train_loss['total']:.6f} "
            f"val={validation_loss['total']:.6f} lr={learning_rate_used:.2e}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if bad_epochs >= args.dae_patience:
            stopped_early = True
            break

    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state"])
    elapsed = time.perf_counter() - started
    last_epoch = len(history)
    epochs_after_best = last_epoch - best_epoch
    if stopped_early:
        convergence_status = "early_stopped_after_full_patience"
    elif best_epoch == last_epoch:
        convergence_status = "maximum_epoch_reached_while_validation_still_improving"
    else:
        convergence_status = "maximum_epoch_reached_without_full_patience"
    training = {
        "seed": args.seed,
        "architecture": model.architecture_config(),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "optimizer": "AdamW",
        "initial_learning_rate": args.dae_lr,
        "weight_decay": args.weight_decay,
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
            "minimum_learning_rate": 1e-5,
        },
        "batch_size": args.batch_size,
        "gradient_clip_norm": 1.0,
        "loss": {
            "formula": "SmoothL1_time + 0.2*SmoothL1_first_difference + 0.1*L1_log_STFT_magnitude",
            "huber_beta": 1.0,
            "difference_weight": 0.2,
            "frequency_weight": 0.1,
            "stft": {
                "n_fft": 64,
                "win_length": 64,
                "hop_length": 16,
                "window": "periodic Hann",
                "center": True,
                "normalized": False,
            },
        },
        "train_windows": int(len(train_clean)),
        "validation_windows": int(len(validation_clean)),
        "validation_input": "clean, corruption disabled",
        "maximum_epochs": args.dae_epochs,
        "early_stopping_patience": args.dae_patience,
        "best_epoch": best_epoch,
        "best_validation_total_loss": best_loss,
        "epochs_completed": last_epoch,
        "epochs_after_best": epochs_after_best,
        "stopped_early": stopped_early,
        "convergence_status": convergence_status,
        "elapsed_seconds": elapsed,
        "history": history,
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
    }
    atomic_json_dump(training, output_dir / "dae_training.json")
    core.write_csv(output_dir / "dae_training_history.csv", history)
    return model, training


@torch.no_grad()
def reconstruct(
    args: argparse.Namespace,
    model: TCNDenoisingAutoencoder,
    values: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = array_loader(
        values,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    reconstructions: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    for (clean,) in loader:
        clean = clean.to(device, non_blocking=True)
        with torch.amp.autocast(
            device.type,
            enabled=bool(args.amp and device.type == "cuda"),
        ):
            reconstruction, latent = model(clean)
        reconstructions.append(reconstruction.float().cpu().numpy())
        latents.append(latent.float().cpu().numpy())
    return (
        np.ascontiguousarray(np.concatenate(reconstructions).astype(np.float32)),
        np.ascontiguousarray(np.concatenate(latents).astype(np.float32)),
    )


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "p01": float(np.percentile(array, 1)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(np.max(array)),
    }


def residual_diagnostics(
    target_scaled: np.ndarray,
    reconstruction_scaled: np.ndarray,
    fixed_sigma: np.ndarray,
    residual_unclipped: np.ndarray,
    residual_clipped: np.ndarray,
    labels: np.ndarray,
    scaler: ChannelZScoreScaler,
) -> dict[str, Any]:
    error = target_scaled.astype(np.float64) - reconstruction_scaled.astype(np.float64)
    physical_scale = (scaler.std.astype(np.float64) + scaler.epsilon)[None, :, None]
    physical_error = error * physical_scale
    result: dict[str, Any] = {
        "windows": int(len(labels)),
        "class_counts_non_fog_fog": np.bincount(labels, minlength=2).astype(int).tolist(),
        "reconstruction_rmse_scaled": float(np.sqrt(np.mean(np.square(error)))),
        "reconstruction_mae_scaled": float(np.mean(np.abs(error))),
        "reconstruction_rmse_physical_g": float(
            np.sqrt(np.mean(np.square(physical_error)))
        ),
        "reconstruction_mae_physical_g": float(np.mean(np.abs(physical_error))),
        "fixed_sigma_scaled": distribution_summary(fixed_sigma),
        "residual_unclipped": distribution_summary(residual_unclipped),
        "residual_unclipped_rms": float(
            np.sqrt(np.mean(np.square(residual_unclipped.astype(np.float64))))
        ),
        "residual_clipped_abs_mean": float(
            np.mean(np.abs(residual_clipped.astype(np.float64)))
        ),
        "residual_clipped_rms": float(
            np.sqrt(np.mean(np.square(residual_clipped.astype(np.float64))))
        ),
        "clip_threshold": RESIDUAL_CLIP,
        "clipped_cells": int(np.sum(np.abs(residual_unclipped) > RESIDUAL_CLIP)),
        "clipped_cell_fraction": float(
            np.mean(np.abs(residual_unclipped) > RESIDUAL_CLIP)
        ),
    }
    for label, name in ((0, "non_fog"), (1, "fog")):
        mask = labels == label
        result[name] = {
            "windows": int(mask.sum()),
            "reconstruction_rmse_scaled": float(
                np.sqrt(np.mean(np.square(error[mask])))
            ),
            "residual_clipped_abs_mean": float(
                np.mean(np.abs(residual_clipped[mask].astype(np.float64)))
            ),
            "residual_clipped_rms": float(
                np.sqrt(
                    np.mean(np.square(residual_clipped[mask].astype(np.float64)))
                )
            ),
        }
    return result


def build_protocol(
    args: argparse.Namespace,
    dataset: core.DaphnetDataset,
    point_stats: dict[str, Any],
    window_stats: dict[str, Any],
    scaler: ChannelZScoreScaler,
    scaler_fit_windows: int,
    corruption: CorruptionConfig,
    device: torch.device,
) -> dict[str, Any]:
    with torch.random.fork_rng(devices=[]):
        dae = TCNDenoisingAutoencoder(
            in_channels=dataset.n_channels,
            input_samples=core.TARGET_SAMPLES,
            latent_dim=args.latent_dim,
            dropout=args.dae_dropout,
            residual_kernel_size=3,
            group_norm_groups=8,
        )
        classifier = None
        if not args.dae_only:
            classifier = build_rf125_classifier(
                "tcn_m",
                in_channels=dataset.n_channels,
                input_samples=core.TARGET_SAMPLES,
                hidden_channels=args.classifier_hidden,
                dropout=args.classifier_dropout,
                dilations=DEFAULT_DILATIONS,
            )
    payload: dict[str, Any] = {
        "experiment_version": EXPERIMENT_VERSION,
        "execution_scope": (
            "dae_only" if args.dae_only else "dae_residual_and_tcnm"
        ),
        "created_utc": utc_now(),
        "framework_document": str(FRAMEWORK_DOCUMENT.resolve()),
        "framework_document_sha256": sha256_file(FRAMEWORK_DOCUMENT),
        "data_dir": str(args.data_dir.resolve()),
        "dataset_fingerprint_sha256": dataset_fingerprint(args.data_dir),
        "subject": core.SUBJECT_ID,
        "records": [record.record_id for record in dataset.records],
        "sampling_rate_hz": dataset.sampling_rate_hz,
        "channels": list(dataset.channel_names),
        "point_statistics": point_stats,
        "window_statistics": window_stats,
        "split": {
            "strategy": "same chronological within-S01 split as GRU baseline",
            "train": "S01_seg000 + S01_seg001 support ending at or before sample 50944",
            "validation": "S01_seg001 support beginning at or after sample 50944",
            "test": "all eligible windows of S01_seg002",
            "raw_train_validation_support_overlap": False,
            "test_used_for_fitting_or_selection": False,
            "single_subject_adaptation": (
                "framework recommendation for an independent validation subject cannot "
                "be satisfied in a within-S01 experiment; validation is a later, "
                "raw-support-disjoint chronological block"
            ),
        },
        "windowing": {
            "source_window_context_samples": core.CONTEXT_SAMPLES,
            "source_window_target_samples": core.TARGET_SAMPLES,
            "dae_input": "target block only; source context is ignored",
            "dae_input_samples": core.TARGET_SAMPLES,
            "dae_input_seconds": core.TARGET_SAMPLES / core.SAMPLING_RATE_HZ,
            "stride_samples": core.STRIDE_SAMPLES,
            "decision_update_seconds": core.STRIDE_SAMPLES / core.SAMPLING_RATE_HZ,
            "label": "at least 50% FOG in final 32 target samples",
            "clean_normal_rule": "context, target and 0.5-second guards contain no FOG",
        },
        "normalization": {
            **scaler.as_dict(),
            "fit_split": "train only",
            "fit_class": "clean-normal target windows only",
            "fit_windows": int(scaler_fit_windows),
            "fit_cells_per_channel_including_window_overlap": int(
                scaler_fit_windows * core.TARGET_SAMPLES
            ),
            "window_overlap_weighting_disclosure": (
                "the document defines statistics over windows; because stride is one "
                "second, raw samples appearing in two adjacent target windows receive "
                "their corresponding repeated window weight"
            ),
        },
        "dae": {
            "task": "same-window clean-target reconstruction from corrupted target during training",
            "architecture": dae.architecture_config(),
            "training_input": "clean-normal training target windows only",
            "validation_input": "clean-normal validation target windows, no corruption",
            "corruption": corruption.as_dict(),
            "loss": "SmoothL1_time + 0.2*SmoothL1_difference + 0.1*L1_log_STFT_magnitude",
            "early_stopping": "clean validation combined loss",
        },
        "document_adaptations_and_underspecified_choices": [
            "Actual Daphnet rate is 64 Hz, so two seconds is 128 rather than 130 samples.",
            "Conv1d(k=4,s=2,p=1) gives exact lengths 128->64->32->16.",
            "Time-mask range remains 7..26 samples, equal to 109.375..406.25 ms at 64 Hz.",
            "STFT 64/64/16 equals 1.0 s window, 0.25 s hop, and 1 Hz bins at 64 Hz.",
            "Unspecified residual-block kernel is 3 with symmetric same padding.",
            "Unspecified GroupNorm maximum group count is 8.",
            "Unspecified decoder dilations are symmetric 4,2,1; interpolation precedes each block.",
            "STFT uses a periodic Hann window, center=True, normalized=False.",
        ],
        "residual": (
            {"executed": False, "reason": "DAE-only training requested"}
            if args.dae_only
            else {
                "executed": True,
                "error": "target_scaled - dae_reconstruction_scaled",
                "sigma": (
                    "sqrt(mean(error^2 over clean-normal DAE-training windows, axis=window) "
                    "+ 1e-6), separately for channel and within-window time position"
                ),
                "sigma_shape": [1, dataset.n_channels, core.TARGET_SAMPLES],
                "formula": "clip((target_scaled - reconstruction_scaled) / fixed_sigma, -12, 12)",
                "classifier_input_shape": ["batch", dataset.n_channels, core.TARGET_SAMPLES],
                "dae_has_native_uncertainty_head": False,
            }
        ),
        "classifier": (
            {"executed": False, "reason": "DAE-only training requested"}
            if classifier is None
            else {
                "executed": True,
                "architecture": classifier.architecture_config(),
                "unchanged_from_gru_baseline": True,
                "loss": "BCEWithLogitsLoss",
                "positive_weight": "min(sqrt(N_nonFOG/N_FOG), 6)",
                "early_stopping": "validation PR-AUC",
                "threshold": "validation-only maximum balanced accuracy",
            }
        ),
        "training": {
            "seed": args.seed,
            "classifier_seed": args.seed + 10_000,
            "batch_size": args.batch_size,
            "dae_epochs_max": args.dae_epochs,
            "dae_patience": args.dae_patience,
            "dae_learning_rate": args.dae_lr,
            "weight_decay": args.weight_decay,
            "classifier_epochs_max": args.classifier_epochs,
            "classifier_patience": args.classifier_patience,
            "classifier_learning_rate": args.classifier_lr,
            "deterministic": args.deterministic,
            "amp_requested": args.amp,
            "downstream_classifier_executed": not args.dae_only,
        },
        "leakage_controls": [
            "Scaler, DAE weights and fixed sigma use clean-normal training windows only.",
            "Validation selects DAE epoch, classifier epoch and probability threshold.",
            "Test does not fit normalization, DAE, sigma, classifier, class weight or threshold.",
            "Train and validation have disjoint raw sample support.",
        ],
        "interpretation_limits": [
            "This DAE is a reconstructor, not a future predictor: it observes the target block it reconstructs.",
            "It emits no conditional sigma; the fixed RMS sigma is an experiment adapter, not part of the document.",
            "Training residuals and fixed sigma are in-sample for the DAE, matching the prior exploratory pipeline but potentially optimistic.",
            "One subject, one chronological split and one seed do not establish subject generalization.",
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
    }
    payload["protocol_fingerprint"] = canonical_fingerprint(
        {
            key: value
            for key, value in payload.items()
            if key not in {"created_utc", "environment"}
        }
    )
    return payload


def classifier_args_compatible(args: argparse.Namespace) -> argparse.Namespace:
    # core.train_classifier reads only these existing attributes.  Keeping one
    # namespace preserves the exact classifier implementation of the baseline.
    return args


def run_classifier(
    args: argparse.Namespace,
    features: dict[str, dict[str, np.ndarray]],
    dataset: core.DaphnetDataset,
    windows: core.WindowTable,
    output_dir: Path,
    protocol_fingerprint: str,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_version = core.EXPERIMENT_VERSION
    core.EXPERIMENT_VERSION = EXPERIMENT_VERSION
    try:
        return core.train_classifier(
            classifier_args_compatible(args),
            features,
            dataset,
            windows,
            output_dir,
            protocol_fingerprint,
            device,
        )
    finally:
        core.EXPERIMENT_VERSION = original_version


def compare_with_gru(
    output_dir: Path,
    metrics: dict[str, Any],
    test_indices: np.ndarray,
) -> dict[str, Any] | None:
    baseline_dir = REPO_ROOT / "outputs" / "daphnet_s01_gru_h200_tcnm_seed42"
    baseline_metrics_path = baseline_dir / "metrics.json"
    baseline_predictions_path = baseline_dir / "predictions.npz"
    if not baseline_metrics_path.is_file() or not baseline_predictions_path.is_file():
        return None
    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    with np.load(baseline_predictions_path, allow_pickle=False) as archive:
        baseline_indices = np.asarray(archive["test_window_index"], dtype=np.int64)
    if not np.array_equal(baseline_indices, test_indices):
        raise AssertionError("GRU baseline test window indices do not match DAE experiment")
    keys = ("accuracy", "fog_recall", "specificity", "balanced_accuracy", "pr_auc", "roc_auc")
    dae_test = metrics["test"]
    gru_test = baseline_metrics["test"]
    comparison = {
        "same_test_windows_verified": True,
        "test_windows": int(len(test_indices)),
        "dae_reconstruction_tcnm": {key: float(dae_test[key]) for key in keys},
        "gru_future_prediction_tcnm": {key: float(gru_test[key]) for key in keys},
        "dae_minus_gru": {
            key: float(dae_test[key] - gru_test[key]) for key in keys
        },
        "dae_confusion_matrix": dae_test["confusion_matrix"],
        "gru_confusion_matrix": gru_test["confusion_matrix"],
        "fairness_warning": (
            "same split/windows/classifier hyperparameters, but not the same inference "
            "task: DAE reconstructs an observed target while GRU forecasts an unseen target"
        ),
        "baseline_dir": str(baseline_dir.resolve()),
    }
    atomic_json_dump(comparison, output_dir / "comparison_to_gru.json")
    return comparison


def write_summary(
    output_dir: Path,
    protocol: dict[str, Any],
    dae_training: dict[str, Any],
    diagnostics: dict[str, Any],
    sigma_diagnostics: dict[str, Any],
    classifier_training: dict[str, Any],
    metrics: dict[str, Any],
    comparison: dict[str, Any] | None,
) -> None:
    test = metrics["test"]
    validation = metrics["validation"]
    train_diag = diagnostics["train"]
    validation_diag = diagnostics["validation"]
    test_diag = diagnostics["test"]
    comparison_text = "未找到可核验的既有 GRU 基线产物。"
    if comparison is not None:
        gru = comparison["gru_future_prediction_tcnm"]
        delta = comparison["dae_minus_gru"]
        comparison_text = (
            f"同一 447 个测试窗口上，GRU+TCN-M accuracy={gru['accuracy']:.6f}、"
            f"recall={gru['fog_recall']:.6f}；DAE 相对变化分别为 "
            f"{delta['accuracy']:+.6f} 和 {delta['fog_recall']:+.6f}。"
        )
    text = f"""# S01 DAE 重构残差 + TCN-M 实验结果

## 主要结果

| 数据 | Accuracy | FoG recall | Specificity | Balanced accuracy | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Validation | {validation['accuracy']:.6f} | {validation['fog_recall']:.6f} | {validation['specificity']:.6f} | {validation['balanced_accuracy']:.6f} | {validation['pr_auc']:.6f} | {validation['roc_auc']:.6f} |
| Test | {test['accuracy']:.6f} | {test['fog_recall']:.6f} | {test['specificity']:.6f} | {test['balanced_accuracy']:.6f} | {test['pr_auc']:.6f} | {test['roc_auc']:.6f} |

测试混淆矩阵（行=真实 Non-FoG/FoG，列=预测 Non-FoG/FoG）：

```text
[[{test['tn']}, {test['fp']}],
 [{test['fn']}, {test['tp']}]]
```

验证集选择阈值：{classifier_training['selected_threshold']:.4f}。{comparison_text}

## DAE 训练

- 输入/输出：同一个已到达的 2 秒 target，形状 `[B,9,128]`；这不是未来预测。
- 正常训练/验证窗口：{dae_training['train_windows']} / {dae_training['validation_windows']}。
- 参数量：{dae_training['parameter_count']:,}；latent={protocol['dae']['architecture']['latent_dim']}。
- 最优 epoch：{dae_training['best_epoch']} / {dae_training['epochs_completed']}；最佳 clean-validation combined loss：{dae_training['best_validation_total_loss']:.6f}。
- 停止状态：`{dae_training['convergence_status']}`。
- TCN-M 最优 epoch：{classifier_training['best_epoch']}；验证 PR-AUC：{classifier_training['best_validation_pr_auc']:.6f}。

## 残差过程

固定尺度仅由 {sigma_diagnostics['calibration_windows']} 个 clean-normal 训练窗口标定：

```text
e = target_z - DAE(target_z)
sigma[c,t] = sqrt(mean_train_normal(e[:,c,t]^2) + 1e-6)
r = clip(e / sigma, -12, 12)
```

| 数据 | 重构 RMSE (z) | 重构 RMSE (g) | 残差 RMS | clip 比例 |
|---|---:|---:|---:|---:|
| Train | {train_diag['reconstruction_rmse_scaled']:.6f} | {train_diag['reconstruction_rmse_physical_g']:.6f} | {train_diag['residual_clipped_rms']:.6f} | {100*train_diag['clipped_cell_fraction']:.4f}% |
| Validation | {validation_diag['reconstruction_rmse_scaled']:.6f} | {validation_diag['reconstruction_rmse_physical_g']:.6f} | {validation_diag['residual_clipped_rms']:.6f} | {100*validation_diag['clipped_cell_fraction']:.4f}% |
| Test | {test_diag['reconstruction_rmse_scaled']:.6f} | {test_diag['reconstruction_rmse_physical_g']:.6f} | {test_diag['residual_clipped_rms']:.6f} | {100*test_diag['clipped_cell_fraction']:.4f}% |

`residual_process.npz` 保留每折的 target、重构、误差、未裁剪残差、裁剪残差、latent、标签和窗口索引。

## 公开边界

文档的 DAE 是异常重构器，不产生未来均值或条件方差。这里的固定 sigma 是为了接回原标准化残差接口而增加的训练集标定步骤。DAE 在推理时看到了它要重构的 target，因此与只看 context 的 GRU 未来预测器不是严格同任务比较；两者只共享数据划分、窗口端点、TCN-M 和阈值选择规则。此外，训练残差和 sigma 对 DAE 是 in-sample，可能偏乐观。
"""
    temporary = output_dir / f".summary.md.tmp-{os.getpid()}"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output_dir / "summary.md")


def write_dae_only_summary(
    output_dir: Path,
    protocol: dict[str, Any],
    dae_training: dict[str, Any],
) -> None:
    history = dae_training["history"]
    first = history[0]
    last = history[-1]
    text = f"""# S01 DAE-only training result

## Scope

- DAE was reinitialized and trained from scratch with seed {dae_training['seed']}.
- Maximum epochs: {dae_training['maximum_epochs']}; early-stopping patience: {dae_training['early_stopping_patience']}.
- Clean-normal train/validation windows: {dae_training['train_windows']} / {dae_training['validation_windows']}.
- Residual calibration and TCN-M classification were not run.
- Protocol fingerprint: `{protocol['protocol_fingerprint']}`.

## Loss result

| Quantity | Value |
|---|---:|
| Epochs completed | {dae_training['epochs_completed']} |
| Best epoch | {dae_training['best_epoch']} |
| Best validation combined loss | {dae_training['best_validation_total_loss']:.9f} |
| Training loss, epoch 1 | {first['train_total_loss']:.9f} |
| Training loss, final epoch | {last['train_total_loss']:.9f} |
| Validation loss, epoch 1 | {first['validation_total_loss']:.9f} |
| Validation loss, final epoch | {last['validation_total_loss']:.9f} |

Stop status: `{dae_training['convergence_status']}`.

The plotted curves contain raw, unsmoothed epoch values. Validation uses clean
Non-FoG windows with all training corruptions disabled.
"""
    temporary = output_dir / f".summary.md.tmp-{os.getpid()}"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output_dir / "summary.md")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    done_path = output_dir / "DONE.json"
    if done_path.exists():
        raise FileExistsError(f"completed output exists: {done_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is non-empty; pass --overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = core.load_s01_dataset(args.data_dir)
    base_windows = dataset.make_windows(
        warmup_samples=core.CONTEXT_SAMPLES,
        target_samples=core.TARGET_SAMPLES,
        stride_samples=core.STRIDE_SAMPLES,
        fog_fraction_threshold=0.5,
        normal_guard_samples=core.NORMAL_GUARD_SAMPLES,
    )
    windows = core.endpoint_relabel(dataset, base_windows)
    split = core.make_split(dataset, windows)
    point_stats = core.point_statistics(dataset)
    window_stats = core.window_statistics(dataset, windows, split)

    normal_train_indices = core.normal_support_indices(
        dataset, windows, "train", split.train
    )
    normal_validation_indices = core.normal_support_indices(
        dataset, windows, "validation", split.validation
    )
    raw_normal_train = extract_target_windows(
        dataset, windows, normal_train_indices
    )
    scaler = ChannelZScoreScaler.fit_channel_time(
        raw_normal_train, epsilon=SCALER_EPSILON
    )
    corruption = CorruptionConfig()
    corruption.validate(channels=dataset.n_channels, samples=core.TARGET_SAMPLES)
    protocol = build_protocol(
        args,
        dataset,
        point_stats,
        window_stats,
        scaler,
        len(normal_train_indices),
        corruption,
        device,
    )
    atomic_json_dump(protocol, output_dir / "config.json")
    atomic_json_dump(
        {
            **scaler.as_dict(),
            "fit_split": "train",
            "fit_class": "clean-normal target windows",
            "fit_window_indices": normal_train_indices.astype(int).tolist(),
        },
        output_dir / "scaler.json",
    )
    atomic_npy_save(output_dir / "scaler_mean.npy", scaler.mean)
    atomic_npy_save(output_dir / "scaler_std.npy", scaler.std)
    manifests = core.split_manifest_rows(dataset, windows, split)
    for row in manifests:
        row["clean_normal_for_dae"] = row.pop("clean_normal_for_nbm")
    core.write_csv(output_dir / "split_manifest.csv", manifests)
    atomic_npz_save(
        output_dir / "split_indices.npz",
        train_window_index=split.train,
        validation_window_index=split.validation,
        test_window_index=split.test,
        dae_train_clean_normal_window_index=normal_train_indices,
        dae_validation_clean_normal_window_index=normal_validation_indices,
    )
    print(
        f"Protocol {protocol['protocol_fingerprint']}\n"
        f"device={device} windows="
        f"{ {name: len(index) for name, index in split.as_dict().items()} } "
        f"dae_normal_train={len(normal_train_indices)} "
        f"dae_normal_validation={len(normal_validation_indices)}",
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

    train_clean = scaler.transform_channel_time(raw_normal_train)
    raw_normal_validation = extract_target_windows(
        dataset, windows, normal_validation_indices
    )
    validation_clean = scaler.transform_channel_time(raw_normal_validation)
    model, dae_training = train_dae(
        args,
        train_clean,
        validation_clean,
        output_dir,
        protocol["protocol_fingerprint"],
        device,
        corruption,
    )

    if args.dae_only:
        from plot_daphnet_s01_dae_losses import (
            read_history,
            save_training_plot,
            save_validation_plot,
        )

        plot_history = read_history(output_dir / "dae_training_history.csv")
        save_training_plot(plot_history, output_dir / "dae_training_loss.png")
        save_validation_plot(plot_history, output_dir / "dae_validation_loss.png")
        write_dae_only_summary(output_dir, protocol, dae_training)
        atomic_json_dump(
            {
                "status": "complete",
                "scope": "dae_only",
                "completed_utc": utc_now(),
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
        print(
            "COMPLETE DAE_ONLY "
            f"epochs={dae_training['epochs_completed']} "
            f"best_epoch={dae_training['best_epoch']} "
            f"best_validation_loss={dae_training['best_validation_total_loss']:.9f}",
            flush=True,
        )
        return

    raw_by_split = {
        name: extract_target_windows(dataset, windows, indices)
        for name, indices in split.as_dict().items()
    }
    scaled_by_split = {
        name: np.ascontiguousarray(scaler.transform_channel_time(values))
        for name, values in raw_by_split.items()
    }

    train_clean_reconstruction, _ = reconstruct(
        args, model, train_clean, device
    )
    calibration_error = train_clean - train_clean_reconstruction
    fixed_sigma = calibrate_fixed_sigma(
        calibration_error, epsilon=FIXED_SIGMA_EPSILON
    )
    atomic_npy_save(output_dir / "fixed_sigma.npy", fixed_sigma)
    sigma_diagnostics = {
        "definition": "channel-by-time RMS of clean-normal training reconstruction errors",
        "calibration_split": "train only",
        "calibration_class": "clean-normal",
        "calibration_windows": int(len(train_clean)),
        "calibration_window_indices": normal_train_indices.astype(int).tolist(),
        "epsilon_inside_square_root": FIXED_SIGMA_EPSILON,
        "shape": list(fixed_sigma.shape),
        "distribution": distribution_summary(fixed_sigma),
        "test_used": False,
        "in_sample_for_dae": True,
    }
    atomic_json_dump(sigma_diagnostics, output_dir / "fixed_sigma.json")

    features: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, Any] = {}
    process_arrays: dict[str, np.ndarray] = {"fixed_sigma": fixed_sigma}
    for name, indices in split.as_dict().items():
        target = scaled_by_split[name]
        reconstruction, latent = reconstruct(args, model, target, device)
        error = target - reconstruction
        residual_unclipped = error / fixed_sigma
        residual = np.clip(
            residual_unclipped, -RESIDUAL_CLIP, RESIDUAL_CLIP
        ).astype(np.float32)
        labels = windows.label[indices].astype(np.int8, copy=True)
        features[name] = {
            "residual": np.ascontiguousarray(residual),
            "y": labels,
            "window_index": indices.astype(np.int64, copy=True),
        }
        diagnostics[name] = residual_diagnostics(
            target,
            reconstruction,
            fixed_sigma,
            residual_unclipped,
            residual,
            labels,
            scaler,
        )
        process_arrays.update(
            {
                f"{name}_target_scaled": target,
                f"{name}_reconstruction_scaled": reconstruction,
                f"{name}_error_scaled": error.astype(np.float32),
                f"{name}_residual_unclipped": residual_unclipped.astype(np.float32),
                f"{name}_residual_clipped": residual,
                f"{name}_latent": latent,
                f"{name}_y": labels,
                f"{name}_window_index": indices.astype(np.int64),
            }
        )
    atomic_json_dump(diagnostics, output_dir / "residual_diagnostics.json")
    atomic_npz_save(output_dir / "residual_process.npz", **process_arrays)

    classifier_training, metrics = run_classifier(
        args,
        features,
        dataset,
        windows,
        output_dir,
        protocol["protocol_fingerprint"],
        device,
    )
    comparison = compare_with_gru(output_dir, metrics, split.test)
    write_summary(
        output_dir,
        protocol,
        dae_training,
        diagnostics,
        sigma_diagnostics,
        classifier_training,
        metrics,
        comparison,
    )
    atomic_json_dump(
        {
            "status": "complete",
            "completed_utc": utc_now(),
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
        f"test_recall={test['fog_recall']:.6f} "
        f"test_specificity={test['specificity']:.6f} "
        f"test_confusion={test['confusion_matrix']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
