#!/usr/bin/env python
"""Continue a completed S01 DAE-only artifact into residual TCN-M training."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cnbr_fog.denoising_autoencoder import (  # noqa: E402
    ChannelZScoreScaler,
    TCNDenoisingAutoencoder,
)
from cnbr_fog.nbm_representations import calibrate_fixed_sigma  # noqa: E402
from cnbr_fog.resume import (  # noqa: E402
    atomic_json_dump,
    atomic_npz_save,
    canonical_fingerprint,
    dataset_fingerprint,
    sha256_file,
)
from cnbr_fog.rf125_classifiers import (  # noqa: E402
    DEFAULT_DILATIONS,
    build_rf125_classifier,
)
import run_daphnet_s01_dae_tcnm as dae_core  # noqa: E402
import run_daphnet_s01_gru_h200_tcnm as core  # noqa: E402


EXPERIMENT_VERSION = "daphnet_s01_pretrained_dae_tcnm.v1"
RESIDUAL_CLIP = 12.0
FIXED_SIGMA_EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a frozen S01 DAE artifact to train residual TCN-M",
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
        "--dae-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / "daphnet_s01_dae_only_max200_seed42"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / "daphnet_s01_dae_max200_best120_tcnm_seed42"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
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
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "batch_size",
        "classifier_epochs",
        "classifier_patience",
        "classifier_hidden",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.classifier_lr <= 0 or args.weight_decay < 0:
        raise ValueError("invalid optimizer values")
    if not 0 <= args.classifier_dropout < 1:
        raise ValueError("--classifier-dropout must be in [0,1)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_done_artifacts(directory: Path) -> dict[str, Any]:
    done_path = directory / "DONE.json"
    if not done_path.is_file():
        raise FileNotFoundError(f"missing completed DAE artifact: {done_path}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("status") != "complete" or done.get("scope") != "dae_only":
        raise ValueError("source artifact is not a completed DAE-only run")
    for name, expected in done.get("artifacts", {}).items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source artifact hash mismatch: {path}")
    return done


def load_frozen_dae(
    dae_dir: Path,
    device: torch.device,
) -> tuple[
    TCNDenoisingAutoencoder,
    ChannelZScoreScaler,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    done = verify_done_artifacts(dae_dir)
    config = json.loads((dae_dir / "config.json").read_text(encoding="utf-8"))
    training = json.loads(
        (dae_dir / "dae_training.json").read_text(encoding="utf-8")
    )
    scaler_json = json.loads((dae_dir / "scaler.json").read_text(encoding="utf-8"))
    if config.get("execution_scope") != "dae_only":
        raise ValueError("source config does not declare DAE-only execution")
    if done.get("protocol_fingerprint") != config.get("protocol_fingerprint"):
        raise ValueError("source DONE/config protocol fingerprints differ")
    if training.get("convergence_status") != "early_stopped_after_full_patience":
        raise ValueError("source DAE did not complete the declared early-stop rule")
    if training.get("best_checkpoint_sha256") != sha256_file(
        dae_dir / "dae_best.pt"
    ):
        raise ValueError("source best-checkpoint hash differs from training metadata")

    mean = np.load(dae_dir / "scaler_mean.npy", allow_pickle=False).astype(np.float32)
    std = np.load(dae_dir / "scaler_std.npy", allow_pickle=False).astype(np.float32)
    if not np.array_equal(mean, np.asarray(scaler_json["mean"], dtype=np.float32)):
        raise ValueError("scaler mean JSON/NPY mismatch")
    if not np.array_equal(std, np.asarray(scaler_json["std"], dtype=np.float32)):
        raise ValueError("scaler std JSON/NPY mismatch")
    scaler = ChannelZScoreScaler(
        mean=mean,
        std=std,
        epsilon=float(scaler_json["epsilon"]),
    )

    architecture = training["architecture"]
    if architecture["input_shape"][-2:] != [9, core.TARGET_SAMPLES]:
        raise ValueError("source DAE input shape differs from this experiment")
    model = TCNDenoisingAutoencoder(
        in_channels=9,
        input_samples=core.TARGET_SAMPLES,
        latent_dim=int(architecture["latent_dim"]),
        dropout=float(architecture["dropout"]),
        residual_kernel_size=int(architecture["residual_kernel_size"]),
        group_norm_groups=int(architecture["maximum_group_norm_groups"]),
    ).to(device)
    checkpoint = torch.load(
        dae_dir / "dae_best.pt", map_location=device, weights_only=False
    )
    if checkpoint.get("epoch") != training.get("best_epoch"):
        raise ValueError("source checkpoint epoch differs from training best epoch")
    if checkpoint.get("protocol_fingerprint") != config.get(
        "protocol_fingerprint"
    ):
        raise ValueError("source checkpoint/config protocol fingerprints differ")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, scaler, config, training, done


def verify_source_split(
    dae_dir: Path,
    split: core.SplitBundle,
    normal_train: np.ndarray,
    normal_validation: np.ndarray,
) -> None:
    with np.load(dae_dir / "split_indices.npz", allow_pickle=False) as archive:
        expected = {
            "train_window_index": split.train,
            "validation_window_index": split.validation,
            "test_window_index": split.test,
            "dae_train_clean_normal_window_index": normal_train,
            "dae_validation_clean_normal_window_index": normal_validation,
        }
        for key, values in expected.items():
            if key not in archive or not np.array_equal(archive[key], values):
                raise ValueError(f"source/current split mismatch: {key}")


def build_protocol(
    args: argparse.Namespace,
    dataset: core.DaphnetDataset,
    windows: core.WindowTable,
    split: core.SplitBundle,
    source_config: dict[str, Any],
    source_training: dict[str, Any],
    source_done: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    with torch.random.fork_rng(devices=[]):
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
        "created_utc": utc_now(),
        "data_dir": str(args.data_dir.resolve()),
        "dataset_fingerprint_sha256": dataset_fingerprint(args.data_dir),
        "subject": core.SUBJECT_ID,
        "sampling_rate_hz": core.SAMPLING_RATE_HZ,
        "channels": list(dataset.channel_names),
        "point_statistics": core.point_statistics(dataset),
        "window_statistics": core.window_statistics(dataset, windows, split),
        "source_dae": {
            "directory": str(args.dae_dir.resolve()),
            "source_protocol_fingerprint": source_config["protocol_fingerprint"],
            "source_done_sha256": sha256_file(args.dae_dir / "DONE.json"),
            "best_checkpoint_sha256": sha256_file(args.dae_dir / "dae_best.pt"),
            "best_epoch": source_training["best_epoch"],
            "best_validation_total_loss": source_training[
                "best_validation_total_loss"
            ],
            "epochs_completed": source_training["epochs_completed"],
            "convergence_status": source_training["convergence_status"],
            "weights_frozen": True,
            "retrained_in_this_run": False,
            "all_declared_source_artifacts_hash_verified": True,
            "verified_source_artifact_count": len(source_done["artifacts"]),
        },
        "split": {
            "same_as_source_dae_and_original_s01_experiment": True,
            "counts": {
                name: int(len(indices))
                for name, indices in split.as_dict().items()
            },
            "test_used_for_fitting_or_selection": False,
        },
        "normalization": {
            "source": "frozen source-DAE training-fold z-score",
            "mean_sha256": sha256_file(args.dae_dir / "scaler_mean.npy"),
            "std_sha256": sha256_file(args.dae_dir / "scaler_std.npy"),
            "refit": False,
        },
        "residual": {
            "error": "target_scaled - frozen_dae_reconstruction_scaled",
            "sigma": (
                "sqrt(mean(error^2 over 978 clean-normal training windows, "
                "axis=window) + 1e-6), separately by channel and time position"
            ),
            "sigma_shape": [1, dataset.n_channels, core.TARGET_SAMPLES],
            "formula": "clip(error / fixed_sigma, -12, 12)",
            "fixed_sigma_fit_split": "clean-normal train only",
            "test_used_to_calibrate_sigma": False,
        },
        "classifier": {
            "architecture": classifier.architecture_config(),
            "loss": "BCEWithLogitsLoss",
            "positive_weight": "min(sqrt(N_nonFOG/N_FOG), 6)",
            "early_stopping": "validation PR-AUC, patience 4",
            "threshold": "validation-only maximum balanced accuracy",
        },
        "training": {
            "seed": args.seed,
            "classifier_seed": args.seed + 10_000,
            "batch_size": args.batch_size,
            "classifier_epochs_max": args.classifier_epochs,
            "classifier_patience": args.classifier_patience,
            "classifier_learning_rate": args.classifier_lr,
            "weight_decay": args.weight_decay,
            "deterministic_requested": args.deterministic,
            "amp_requested": args.amp,
        },
        "leakage_controls": [
            "DAE checkpoint and scaler are frozen, hash-verified source artifacts.",
            "Fixed sigma uses only source-matched clean-normal training windows.",
            "TCN-M weights use training residuals and labels only.",
            "Validation selects classifier epoch and decision threshold.",
            "Test is used only for the final frozen-pipeline evaluation.",
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "device": str(device),
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


def train_classifier(
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
            args,
            features,
            dataset,
            windows,
            output_dir,
            protocol_fingerprint,
            device,
        )
    finally:
        core.EXPERIMENT_VERSION = original_version


def write_summary(
    output_dir: Path,
    protocol: dict[str, Any],
    sigma_diagnostics: dict[str, Any],
    residual_diagnostics: dict[str, Any],
    classifier_training: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    validation = metrics["validation"]
    test = metrics["test"]
    text = f"""# S01 frozen DAE residual + TCN-M result

The DAE was loaded from the completed DAE-only run and remained frozen.

## Source DAE

- Best epoch: {protocol['source_dae']['best_epoch']}.
- Best clean-validation loss: {protocol['source_dae']['best_validation_total_loss']:.9f}.
- Convergence status: `{protocol['source_dae']['convergence_status']}`.
- Clean-normal fixed-sigma calibration windows: {sigma_diagnostics['calibration_windows']}.

## Classification

| Split | Accuracy | FoG recall | Specificity | PR-AUC |
|---|---:|---:|---:|---:|
| Validation | {validation['accuracy']:.6f} | {validation['fog_recall']:.6f} | {validation['specificity']:.6f} | {validation['pr_auc']:.6f} |
| Test | {test['accuracy']:.6f} | {test['fog_recall']:.6f} | {test['specificity']:.6f} | {test['pr_auc']:.6f} |

Validation-selected threshold: {classifier_training['selected_threshold']:.4f}.
Test confusion matrix: `[[{test['tn']}, {test['fp']}], [{test['fn']}, {test['tp']}]]`.

Test residual RMS: {residual_diagnostics['test']['residual_clipped_rms']:.6f};
Non-FoG/FoG residual RMS: {residual_diagnostics['test']['non_fog']['residual_clipped_rms']:.6f} /
{residual_diagnostics['test']['fog']['residual_clipped_rms']:.6f}.
"""
    temporary = output_dir / f".summary.md.tmp-{os.getpid()}"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output_dir / "summary.md")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = core.resolve_device(args.device)
    dae_dir = args.dae_dir.resolve()
    output_dir = args.output_dir.resolve()
    done_path = output_dir / "DONE.json"
    if done_path.exists():
        raise FileExistsError(f"completed output exists: {done_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is non-empty; pass --overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    model, scaler, source_config, source_training, source_done = load_frozen_dae(
        dae_dir, device
    )
    current_dataset_fingerprint = dataset_fingerprint(args.data_dir)
    if current_dataset_fingerprint != source_config.get(
        "dataset_fingerprint_sha256"
    ):
        raise ValueError("source DAE and current dataset fingerprints differ")
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
    normal_train = core.normal_support_indices(
        dataset, windows, "train", split.train
    )
    normal_validation = core.normal_support_indices(
        dataset, windows, "validation", split.validation
    )
    verify_source_split(
        dae_dir, split, normal_train, normal_validation
    )
    protocol = build_protocol(
        args,
        dataset,
        windows,
        split,
        source_config,
        source_training,
        source_done,
        device,
    )
    atomic_json_dump(protocol, output_dir / "config.json")
    manifests = core.split_manifest_rows(dataset, windows, split)
    for row in manifests:
        row["clean_normal_for_dae"] = row.pop("clean_normal_for_nbm")
    core.write_csv(output_dir / "split_manifest.csv", manifests)
    atomic_npz_save(
        output_dir / "split_indices.npz",
        train_window_index=split.train,
        validation_window_index=split.validation,
        test_window_index=split.test,
        dae_train_clean_normal_window_index=normal_train,
        dae_validation_clean_normal_window_index=normal_validation,
    )
    print(
        f"Protocol {protocol['protocol_fingerprint']}\n"
        f"frozen_dae_epoch={source_training['best_epoch']} "
        f"device={device} windows="
        f"{ {name: len(index) for name, index in split.as_dict().items()} }",
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

    raw_normal_train = dae_core.extract_target_windows(
        dataset, windows, normal_train
    )
    scaled_normal_train = scaler.transform_channel_time(raw_normal_train)
    train_normal_reconstruction, _ = dae_core.reconstruct(
        args, model, scaled_normal_train, device
    )
    calibration_error = scaled_normal_train - train_normal_reconstruction
    fixed_sigma = calibrate_fixed_sigma(
        calibration_error, epsilon=FIXED_SIGMA_EPSILON
    )
    dae_core.atomic_npy_save(output_dir / "fixed_sigma.npy", fixed_sigma)
    sigma_diagnostics = {
        "definition": "channel-by-time RMS of frozen-DAE clean-normal training reconstruction errors",
        "calibration_split": "train only",
        "calibration_class": "clean-normal",
        "calibration_windows": int(len(normal_train)),
        "calibration_window_indices": normal_train.astype(int).tolist(),
        "epsilon_inside_square_root": FIXED_SIGMA_EPSILON,
        "shape": list(fixed_sigma.shape),
        "distribution": dae_core.distribution_summary(fixed_sigma),
        "source_dae_best_epoch": source_training["best_epoch"],
        "test_used": False,
        "in_sample_for_dae": True,
    }
    atomic_json_dump(sigma_diagnostics, output_dir / "fixed_sigma.json")

    features: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, Any] = {}
    process_arrays: dict[str, np.ndarray] = {"fixed_sigma": fixed_sigma}
    for name, indices in split.as_dict().items():
        raw = dae_core.extract_target_windows(dataset, windows, indices)
        target = np.ascontiguousarray(scaler.transform_channel_time(raw))
        reconstruction, latent = dae_core.reconstruct(
            args, model, target, device
        )
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
        diagnostics[name] = dae_core.residual_diagnostics(
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

    classifier_training, metrics = train_classifier(
        args,
        features,
        dataset,
        windows,
        output_dir,
        protocol["protocol_fingerprint"],
        device,
    )
    write_summary(
        output_dir,
        protocol,
        sigma_diagnostics,
        diagnostics,
        classifier_training,
        metrics,
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
        f"accuracy={test['accuracy']:.6f} "
        f"recall={test['fog_recall']:.6f} "
        f"specificity={test['specificity']:.6f} "
        f"pr_auc={test['pr_auc']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
