#!/usr/bin/env python3
"""Train one strict GRU-v1 NBM fold for the local-mask strength comparison.

The two supported variants are identical except for the inclusive length of
the contiguous, all-axis time mask: BASE uses 4..8 samples and MASK8_12 uses
8..12 samples.  This worker only accesses roles 4 and 5 and never trains or
evaluates the downstream classifier.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.resume import atomic_json_dump, atomic_torch_save, canonical_fingerprint
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    ROLES,
    SUBJECTS,
    audit_protocol,
    fit_scaler_unique_role4_points,
    load_fold_rows,
    raw_windows,
    save_figure_bundle,
    write_csv,
)
from scripts.run_daphnet_residual_calibration_abcd import (
    sha256_file,
    state_dict_sha256,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    GRUReconstructionNBM,
    RobustScaler,
    make_loader,
    prepare_nbm_windows,
    set_seed,
)

FOLDS = (0, 1, 2)
REQUIRED_SEEDS = (0, 52, 161, 5216, 52161)
CHANNELS = 9
WINDOW_SAMPLES = 128
HIDDEN = 64
BOTTLENECK = 16
PARAMETER_COUNT = 31_513
ARCHITECTURE_NAME = "gru_reconstruction_nbm_v1"
VARIANTS: dict[str, tuple[int, int]] = {
    "BASE": (4, 8),
    "MASK8_12": (8, 12),
}


def parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid unique integer list: {value}")
    return values


def checkpoint_name(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    return f"gru_nbm_{variant.lower()}_best.pt"


def architecture_config() -> dict[str, Any]:
    model = GRUReconstructionNBM(
        channels=CHANNELS, hidden=HIDDEN, bottleneck=BOTTLENECK
    )
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != PARAMETER_COUNT:
        raise RuntimeError(f"GRU-v1 parameter contract changed: {count}")
    return {
        "name": ARCHITECTURE_NAME,
        "input_shape": ["B", 128, 9],
        "encoder_gru": {
            "input_size": 9,
            "hidden_size": 64,
            "layers": 1,
            "bidirectional": False,
            "summary": "last hidden state",
        },
        "bottleneck": "Linear(64,16)",
        "latent_shape": ["B", 16],
        "decoder_initial_state": "Linear(16,64)",
        "decoder_gru": {
            "input_size": 9,
            "hidden_size": 64,
            "layers": 1,
            "bidirectional": False,
            "input": "128-step all-zero sequence",
        },
        "output": "Linear(64,9), no output activation",
        "skip_connections": False,
        "teacher_forcing": False,
        "normalization": None,
        "output_shape": ["B", 128, 9],
        "parameter_count": PARAMETER_COUNT,
    }


def augmentation_config(variant: str) -> dict[str, Any]:
    minimum, maximum = VARIANTS[variant]
    return {
        "clean_probability": 0.40,
        "gaussian_probability": 0.40,
        "mask_probability": 0.20,
        "gaussian_std": 0.04,
        "gaussian_scope": "whole window and all samples/channels",
        "mask_minimum_samples": minimum,
        "mask_maximum_samples": maximum,
        "mask_interval": "inclusive",
        "mask_length_sampling": "discrete_uniform_inclusive",
        "mask_contiguous": True,
        "mask_all_channels": True,
        "mask_replacement_value": 0.0,
        "augmentation_roles": [4],
        "validation_augmentation": False,
    }


def protocol_contract(variant: str, scientific_data_sha256: str) -> dict[str, Any]:
    """Return the path-independent contract bound into every artifact."""

    return {
        "schema": "gru_mask_strength_nbm.v1",
        "variant": variant,
        "scientific_data_sha256": scientific_data_sha256,
        "sampling_rate_hz": 64,
        "window_samples": 128,
        "stride_samples": 64,
        "fit_role": 4,
        "validation_and_calibration_role": 5,
        "input_preprocessing": "role4 RobustScaler then per-window/per-axis centering",
        "architecture": architecture_config(),
        "augmentation": augmentation_config(variant),
        "loss": "SmoothL1(beta=1.0)",
        "optimizer": "AdamW(lr=0.001,weight_decay=0.0001)",
        "scheduler": "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)",
        "batch_size": 128,
        "maximum_epochs": 300,
        "patience": 20,
        "gradient_clip": 1.0,
        "checkpoint_rule": "lowest clean role-5 validation SmoothL1",
        "calibration": "post-best-restore role-5 per-axis median bias and MAD scale",
    }


def corrupt_local_mask(
    clean: torch.Tensor,
    generator: torch.Generator,
    *,
    mask_minimum_samples: int,
    mask_maximum_samples: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Apply the frozen 40/40/20 corruption with a parameterized local mask."""

    if clean.ndim != 3 or tuple(clean.shape[1:]) != (128, 9):
        raise ValueError(f"expected [B,128,9], got {tuple(clean.shape)}")
    if not (1 <= mask_minimum_samples <= mask_maximum_samples <= clean.shape[1]):
        raise ValueError("invalid inclusive mask-length bounds")
    output = clean.clone()
    modes = torch.rand(clean.shape[0], device=clean.device, generator=generator)
    gaussian = (modes >= 0.4) & (modes < 0.8)
    if torch.any(gaussian):
        noise = torch.randn(
            output[gaussian].shape,
            device=output.device,
            dtype=output.dtype,
            generator=generator,
        )
        output[gaussian] += noise * 0.04
    masked_indices = torch.nonzero(modes >= 0.8, as_tuple=False).flatten().tolist()
    for index in masked_indices:
        length = int(
            torch.randint(
                mask_minimum_samples,
                mask_maximum_samples + 1,
                (1,),
                device=clean.device,
                generator=generator,
            )
        )
        start = int(
            torch.randint(
                0,
                clean.shape[1] - length + 1,
                (1,),
                device=clean.device,
                generator=generator,
            )
        )
        output[index, start : start + length, :] = 0.0
    counts = np.asarray(
        [int((modes < 0.4).sum()), int(gaussian.sum()), len(masked_indices)],
        dtype=np.int64,
    )
    return output, counts


@torch.no_grad()
def reconstruct_gru_mask_strength(
    model: GRUReconstructionNBM,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    if x.ndim != 3 or tuple(x.shape[1:]) != (128, 9) or len(x) == 0:
        raise ValueError(f"expected non-empty [N,128,9], got {x.shape}")
    model.eval()
    output: list[np.ndarray] = []
    for (batch,) in make_loader(x, batch_size, False, 0, 0):
        output.append(
            model(batch.to(device, non_blocking=True))
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    return np.concatenate(output, axis=0)


def train_gru_mask_strength_nbm(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    output_dir: Path,
    device: torch.device,
    seed: int,
    num_workers: int,
    variant: str,
    max_epochs: int = 300,
    patience: int = 20,
) -> tuple[GRUReconstructionNBM, dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    if max_epochs != 300 or patience != 20:
        raise ValueError("this experiment freezes NBM training to max300/pat20")
    for name, values in (("role4", train_x), ("role5", validation_x)):
        if values.ndim != 3 or tuple(values.shape[1:]) != (128, 9) or not len(values):
            raise ValueError(f"invalid non-empty {name} array: {values.shape}")

    set_seed(seed)
    model = GRUReconstructionNBM(channels=9, hidden=64, bottleneck=16).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise RuntimeError("GRU-v1 parameter contract changed")
    # Captured before optimizer creation/steps.  BASE and MASK8_12 must have
    # exactly the same value for a paired fold/seed.
    initial_model_state_sha256 = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    train_loader = make_loader(train_x, 128, True, seed, num_workers)
    validation_loader = make_loader(validation_x, 128, False, seed, num_workers)
    criterion = nn.SmoothL1Loss(beta=1.0)
    augmentation_generator = torch.Generator(device=device).manual_seed(seed + 1000)
    mask_minimum, mask_maximum = VARIANTS[variant]
    checkpoint = output_dir / "checkpoints" / checkpoint_name(variant)
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        mode_counts = np.zeros(3, dtype=np.int64)
        for (clean,) in train_loader:
            clean = clean.to(device, non_blocking=True)
            network_input, counts = corrupt_local_mask(
                clean,
                augmentation_generator,
                mask_minimum_samples=mask_minimum,
                mask_maximum_samples=mask_maximum,
            )
            mode_counts += counts
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(network_input), clean)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite GRU-v1 NBM gradient")
            optimizer.step()
            total_loss += float(loss.detach()) * len(clean)
            total_n += len(clean)

        model.eval()
        validation_total = 0.0
        validation_n = 0
        with torch.no_grad():
            for (clean,) in validation_loader:
                clean = clean.to(device, non_blocking=True)
                # Role 5 is intentionally clean: no corruption call is allowed here.
                loss = criterion(model(clean), clean)
                validation_total += float(loss) * len(clean)
                validation_n += len(clean)
        train_loss = total_loss / total_n
        validation_loss = validation_total / validation_n
        scheduler.step(validation_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        improved = validation_loss < best_loss - 1e-10
        history.append(
            {
                "epoch": epoch,
                "train_huber": train_loss,
                "validation_huber": validation_loss,
                "learning_rate": learning_rate,
                "clean_windows": int(mode_counts[0]),
                "gaussian_windows": int(mode_counts[1]),
                "masked_windows": int(mode_counts[2]),
                "improved": improved,
            }
        )
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_huber": validation_loss,
                    "seed": seed,
                    "variant": variant,
                    "initial_model_state_sha256": initial_model_state_sha256,
                    "architecture": architecture_config(),
                    "augmentation": augmentation_config(variant),
                },
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"GRU-v1 {variant} epoch={epoch:03d} train={train_loss:.7f} "
            f"val={validation_loss:.7f} lr={learning_rate:.2e} "
            f"stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload["variant"] != variant or payload["architecture"] != architecture_config():
        raise RuntimeError("best checkpoint contract mismatch")
    if payload.get("initial_model_state_sha256") != initial_model_state_sha256:
        raise RuntimeError("best checkpoint initial-state identity mismatch")
    model.load_state_dict(payload["model_state"])
    history_path = output_dir / "logs" / f"gru_nbm_{variant.lower()}_history.csv"
    write_csv(history_path, history)
    summary = {
        "model_id": f"gru_nbm_{variant.lower()}_best",
        "seed": seed,
        "variant": variant,
        "fit_windows": int(len(train_x)),
        "calibration_validation_windows": int(len(validation_x)),
        "maximum_epochs": max_epochs,
        "patience": patience,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_huber": best_loss,
        "parameter_count": PARAMETER_COUNT,
        "initial_model_state_sha256": initial_model_state_sha256,
        "history_file": str(history_path.relative_to(output_dir)),
        "checkpoint_file": str(checkpoint.relative_to(output_dir)),
    }
    return model, {"summary": summary, "history": history}


def calibrate_gru_mask_strength(
    model: GRUReconstructionNBM,
    role5_x: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reconstruction = reconstruct_gru_mask_strength(model, role5_x, device)
    error = role5_x - reconstruction
    bias = np.median(error, axis=(0, 1)).astype(np.float32)
    sigma_raw = 1.4826 * np.median(
        np.abs(error - bias[None, None, :]), axis=(0, 1)
    )
    sigma = np.maximum(sigma_raw, 0.05).astype(np.float32)
    return bias, sigma, {
        "bias": bias.astype(float).tolist(),
        "sigma_raw": sigma_raw.astype(float).tolist(),
        "sigma": sigma.astype(float).tolist(),
        "sigma_floor": 0.05,
        "floor_applied_channels": np.flatnonzero(sigma_raw < 0.05).astype(int).tolist(),
        "calibration_windows": int(len(role5_x)),
    }


def write_role4_scaler_artifact(
    fold_dir: Path,
    *,
    fold: int,
    seed: int,
    scaler: RobustScaler,
    unique_raw_points: int,
    scientific_data_sha256: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    payload = {
        "fold": fold,
        "seed": seed,
        "scaler_fit_role": 4,
        "scaler_unique_raw_points": unique_raw_points,
        "scaler": scaler.as_dict(),
        "scientific_data_sha256": scientific_data_sha256,
        "protocol_sha256": protocol_sha256,
    }
    atomic_json_dump(payload, fold_dir / "scaler_role4.json")
    return payload


def _artifact_hashes(fold_dir: Path, checkpoint: Path) -> dict[str, str]:
    return {
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_sha256": sha256_file(fold_dir / "config.json"),
        "scaler_role4_sha256": sha256_file(fold_dir / "scaler_role4.json"),
        "nbm_frozen_sha256": sha256_file(fold_dir / "nbm_frozen.json"),
    }


def validate_existing_nbm(
    fold_dir: Path,
    args: argparse.Namespace,
    scientific_data_sha256: str,
) -> None:
    variant = args.variant
    checkpoint = fold_dir / "checkpoints" / checkpoint_name(variant)
    required = {
        "done": fold_dir / "DONE_NBM.json",
        "config": fold_dir / "config.json",
        "scaler": fold_dir / "scaler_role4.json",
        "frozen": fold_dir / "nbm_frozen.json",
        "checkpoint": checkpoint,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete NBM artifacts: {missing}")
    done = json.loads(required["done"].read_text(encoding="utf-8"))
    config = json.loads(required["config"].read_text(encoding="utf-8"))
    scaler = json.loads(required["scaler"].read_text(encoding="utf-8"))
    frozen = json.loads(required["frozen"].read_text(encoding="utf-8"))
    contract = protocol_contract(variant, scientific_data_sha256)
    expected = {
        "status": "frozen",
        "fold": args.fold,
        "seed": args.seed,
        "variant": variant,
        "maximum_epochs": 300,
        "patience": 20,
        "parameter_count": PARAMETER_COUNT,
        "scientific_data_sha256": scientific_data_sha256,
        "protocol_sha256": canonical_fingerprint(contract),
    }
    for key, value in expected.items():
        if done.get(key) != value:
            raise AssertionError(f"stale DONE_NBM {key}: {done.get(key)!r} != {value!r}")
    for key, digest in _artifact_hashes(fold_dir, checkpoint).items():
        if done.get(key) != digest:
            raise AssertionError(f"DONE_NBM artifact hash mismatch: {key}")
    if config.get("protocol") != contract:
        raise AssertionError("config protocol contract mismatch")
    if (
        config.get("fold") != args.fold
        or config.get("seed") != args.seed
        or config.get("variant") != variant
        or config.get("protocol_sha256") != expected["protocol_sha256"]
    ):
        raise AssertionError("config fold/seed/variant identity mismatch")
    if scaler.get("scaler_fit_role") != 4 or scaler.get("scaler") != frozen.get("scaler"):
        raise AssertionError("role-4 scaler contract mismatch")
    if (
        scaler.get("fold") != args.fold
        or scaler.get("seed") != args.seed
        or scaler.get("scientific_data_sha256") != scientific_data_sha256
        or scaler.get("protocol_sha256") != expected["protocol_sha256"]
    ):
        raise AssertionError("role-4 scaler identity/hash mismatch")
    if frozen.get("variant") != variant or frozen.get("protocol_sha256") != expected["protocol_sha256"]:
        raise AssertionError("frozen variant/protocol mismatch")
    if (
        frozen.get("scientific_data_sha256") != scientific_data_sha256
        or frozen.get("nbm_train_role") != 4
        or frozen.get("nbm_earlystop_and_calibration_role") != 5
        or frozen.get("validation_mask_or_noise") is not False
        or frozen.get("best_checkpoint_restored_before_calibration") is not True
        or frozen.get("classifier_or_test_roles_accessed") is not False
    ):
        raise AssertionError("frozen role/isolation contract mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("seed") != args.seed or payload.get("variant") != variant:
        raise AssertionError("checkpoint seed/variant mismatch")
    initial_hash = frozen["training"].get("initial_model_state_sha256")
    if not initial_hash or payload.get("initial_model_state_sha256") != initial_hash:
        raise AssertionError("checkpoint/frozen initial-state identity mismatch")
    if done.get("initial_model_state_sha256") != initial_hash:
        raise AssertionError("DONE/frozen initial-state identity mismatch")
    if payload.get("architecture") != architecture_config():
        raise AssertionError("checkpoint architecture mismatch")
    if int(payload.get("epoch", -1)) != int(frozen["training"]["best_epoch"]):
        raise AssertionError("checkpoint best epoch mismatch")
    if done.get("best_epoch") != frozen["training"]["best_epoch"]:
        raise AssertionError("DONE/frozen best epoch mismatch")
    if not np.isclose(
        float(payload.get("validation_huber", np.nan)),
        float(frozen["training"]["best_validation_huber"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("checkpoint/frozen best validation loss mismatch")
    if not np.isclose(
        float(done.get("best_validation_huber", np.nan)),
        float(frozen["training"]["best_validation_huber"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("DONE/frozen best validation loss mismatch")


def plot_training(fold_dir: Path, variant: str, run_payload: dict[str, Any]) -> None:
    history = run_payload["history"]
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax.plot(epochs, [row["train_huber"] for row in history], label="Role 4 train")
    ax.plot(epochs, [row["validation_huber"] for row in history], label="Role 5 clean")
    ax.axvline(run_payload["summary"]["best_epoch"], color="black", linestyle="--")
    ax.set(xlabel="Epoch", ylabel="SmoothL1", title=f"GRU-v1 NBM {variant}")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure_bundle(fig, fold_dir / f"gru_nbm_{variant.lower()}_training_validation")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed_NBM",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--fold", type=int, choices=FOLDS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--required-seeds", default="0,52,161,5216,52161")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-max-epochs", type=int, default=300)
    parser.add_argument("--nbm-patience", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    return device


def run(args: argparse.Namespace, device: torch.device) -> None:
    required_seeds = parse_csv_ints(args.required_seeds)
    if required_seeds != REQUIRED_SEEDS or args.seed not in REQUIRED_SEEDS:
        raise ValueError(f"required exact seed set is {REQUIRED_SEEDS}")
    if args.nbm_max_epochs != 300 or args.nbm_patience != 20:
        raise ValueError("this experiment requires max300/pat20")
    data_dir = args.data_dir.resolve()
    scientific_data = processed_nbm_scientific_manifest(data_dir)
    contract = protocol_contract(args.variant, scientific_data["sha256"])
    protocol_sha256 = canonical_fingerprint(contract)
    fold_dir = args.output_root.resolve() / f"fold_{args.fold}"
    done_path = fold_dir / "DONE_NBM.json"
    if done_path.exists() and not args.overwrite:
        validate_existing_nbm(fold_dir, args, scientific_data["sha256"])
        print(f"SKIP completed GRU mask-strength source: {done_path}", flush=True)
        return

    fold_dir.mkdir(parents=True, exist_ok=True)
    dataset = DaphnetDataset.load(data_dir)
    if dataset.sampling_rate_hz != 64 or dataset.n_channels != 9:
        raise AssertionError("this protocol requires the 64-Hz, 9-channel dataset")
    records = {record.record_id: record for record in dataset.records}
    rows_by_fold = {fold: load_fold_rows(data_dir, fold) for fold in FOLDS}
    source_audit = audit_protocol(data_dir, rows_by_fold, records)
    rows = rows_by_fold[args.fold]
    role4 = rows.take_role(4)
    role5 = rows.take_role(5)
    scaler, unique_points = fit_scaler_unique_role4_points(records, role4)
    write_role4_scaler_artifact(
        fold_dir,
        fold=args.fold,
        seed=args.seed,
        scaler=scaler,
        unique_raw_points=unique_points,
        scientific_data_sha256=scientific_data["sha256"],
        protocol_sha256=protocol_sha256,
    )
    role4_x = prepare_nbm_windows(scaler, raw_windows(records, role4), center=True)
    role5_x = prepare_nbm_windows(scaler, raw_windows(records, role5), center=True)
    config = {
        "experiment": "GRU_v1_local_mask_strength_NBM300_source",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "seed": args.seed,
        "seed_policy": "exact seed; no fold offset",
        "required_seeds": list(REQUIRED_SEEDS),
        "variant": args.variant,
        "protocol": contract,
        "protocol_sha256": protocol_sha256,
        "roles": {str(key): value for key, value in ROLES.items()},
        "subjects": list(SUBJECTS),
        "excluded_subjects": ["S04", "S10"],
        "role_counts": {str(role): int(np.sum(rows.role == role)) for role in ROLES},
        "source_audit": source_audit,
        "classifier_or_test_roles_accessed": False,
    }
    atomic_json_dump(config, fold_dir / "config.json")
    model, run_payload = train_gru_mask_strength_nbm(
        role4_x,
        role5_x,
        fold_dir,
        device,
        args.seed,
        args.num_workers,
        args.variant,
        args.nbm_max_epochs,
        args.nbm_patience,
    )
    _bias, _sigma, calibration = calibrate_gru_mask_strength(model, role5_x, device)
    plot_training(fold_dir, args.variant, run_payload)
    frozen = {
        "variant": args.variant,
        "protocol_sha256": protocol_sha256,
        "scientific_data_sha256": scientific_data["sha256"],
        "scaler": scaler.as_dict(),
        "scaler_fit_role": 4,
        "scaler_unique_raw_points": unique_points,
        "nbm_train_role": 4,
        "nbm_earlystop_and_calibration_role": 5,
        "validation_mask_or_noise": False,
        "best_checkpoint_restored_before_calibration": True,
        "training": {
            **run_payload["summary"],
            "architecture": architecture_config(),
            "augmentation": augmentation_config(args.variant),
        },
        "calibration": calibration,
        "classifier_or_test_roles_accessed": False,
    }
    atomic_json_dump(frozen, fold_dir / "nbm_frozen.json")
    checkpoint = fold_dir / "checkpoints" / checkpoint_name(args.variant)
    hashes = _artifact_hashes(fold_dir, checkpoint)
    summary = run_payload["summary"]
    done = {
        "status": "frozen",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "seed": args.seed,
        "seed_policy": "exact; no fold offset",
        "variant": args.variant,
        "checkpoint": str(checkpoint.resolve()),
        **hashes,
        "best_epoch": summary["best_epoch"],
        "epochs_completed": summary["epochs_completed"],
        "best_validation_huber": summary["best_validation_huber"],
        "maximum_epochs": 300,
        "patience": 20,
        "parameter_count": PARAMETER_COUNT,
        "initial_model_state_sha256": summary["initial_model_state_sha256"],
        "protocol_sha256": protocol_sha256,
        "scientific_data_sha256": scientific_data["sha256"],
        "classifier_or_test_roles_accessed": False,
    }
    atomic_json_dump(done, done_path)
    validate_existing_nbm(fold_dir, args, scientific_data["sha256"])
    print(json.dumps(done, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    run(args, device)


if __name__ == "__main__":
    main()
