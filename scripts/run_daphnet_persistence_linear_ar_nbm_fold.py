#!/usr/bin/env python3
"""Freeze one Persistence or multivariate Linear-AR(8) NBM source.

Both sources use the role-4-only RobustScaler and per-window/per-axis
centering.  LINEAR_AR is a 657-parameter causal denoising AR model trained on
role 4 with the frozen 40/40/20 corruption and a 4..8-sample all-axis mask.
PERSISTENCE is parameter-free and deterministic.  Clean role 5 is reserved
for early stopping (LINEAR_AR only) and post-freeze MAD residual calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.resume import atomic_json_dump, atomic_torch_save, canonical_fingerprint
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts.run_daphnet_gru_mask_strength_nbm300_fold import corrupt_local_mask
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    ROLES,
    SUBJECTS,
    audit_protocol,
    fit_scaler_unique_role4_points,
    load_fold_rows,
    raw_windows,
    write_csv,
)
from scripts.run_daphnet_residual_calibration_abcd import (
    sha256_file,
    state_dict_sha256,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    RobustScaler,
    make_loader,
    prepare_nbm_windows,
    set_seed,
)

FOLDS = (0, 1, 2)
REQUIRED_SEEDS = (0, 52, 161, 5216, 52161)
VARIANTS = ("PERSISTENCE", "LINEAR_AR")
WINDOW_SAMPLES = 128
CHANNELS = 9
AR_ORDER = 8
LINEAR_AR_PARAMETER_COUNT = 9 * (9 * 8) + 9
SOURCE_CODE_PATHS = (
    Path(__file__).resolve(),
    REPO_ROOT / "scripts" / "run_daphnet_gru_mask_strength_nbm300_fold.py",
    REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_centered_residual_tcn.py",
    REPO_ROOT / "scripts" / "run_daphnet_s01_nonfog_gru_reconstruction_tcnm.py",
    REPO_ROOT / "cnbr_fog" / "data.py",
    REPO_ROOT / "cnbr_fog" / "resume.py",
    REPO_ROOT / "cnbr_fog" / "scientific_fingerprint.py",
)


class MultivariateLinearAR8(nn.Module):
    """Causal 9-channel VAR(8), trained as a linear denoising predictor."""

    def __init__(self) -> None:
        super().__init__()
        # lag_weight[output_channel, lag, input_channel]
        self.lag_weight = nn.Parameter(torch.empty(CHANNELS, AR_ORDER, CHANNELS))
        self.bias = nn.Parameter(torch.empty(CHANNELS))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lag_weight.reshape(CHANNELS, -1))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (WINDOW_SAMPLES, CHANNELS):
            raise ValueError(f"expected [B,128,9], got {tuple(x.shape)}")
        # Left padding followed by a sliding view creates, for every t, the
        # strictly past samples only.  The flip makes lag index 0 equal t-1.
        padded = F.pad(x.transpose(1, 2), (AR_ORDER, 0))
        lagged = (
            padded.unfold(2, AR_ORDER, 1)[:, :, :WINDOW_SAMPLES, :]
            .flip(-1)
            .permute(0, 2, 3, 1)
        )
        predictions = self.bias + torch.einsum(
            "btli,oli->bto", lagged, self.lag_weight
        )
        return torch.cat((x[:, :1, :], predictions[:, 1:, :]), dim=1)


def source_code_sha256() -> dict[str, str]:
    missing = [str(path) for path in SOURCE_CODE_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"critical NBM source code missing: {missing}")
    return {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in SOURCE_CODE_PATHS
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid unique integer list: {value}")
    return values


def checkpoint_name(variant: str) -> str:
    return {
        "PERSISTENCE": "persistence_nbm_frozen.pt",
        "LINEAR_AR": "linear_ar8_nbm_best.pt",
    }[variant]


def architecture_config(variant: str) -> dict[str, Any]:
    if variant == "PERSISTENCE":
        return {
            "name": "persistence_nbm_lag1",
            "input_shape": ["B", 128, 9],
            "formula": "Xhat[:,0,:]=X[:,0,:]; Xhat[:,t,:]=X[:,t-1,:] for t>=1",
            "causal": True,
            "trainable": False,
            "parameter_count": 0,
            "output_shape": ["B", 128, 9],
        }
    if variant == "LINEAR_AR":
        model = MultivariateLinearAR8()
        count = sum(parameter.numel() for parameter in model.parameters())
        if count != LINEAR_AR_PARAMETER_COUNT:
            raise RuntimeError(f"Linear-AR parameter contract changed: {count}")
        return {
            "name": "multivariate_linear_ar8_denoising_nbm",
            "input_shape": ["B", 128, 9],
            "order": 8,
            "cross_channel": True,
            "coefficient_shape": [9, 8, 9],
            "bias_shape": [9],
            "first_sample": "identity copy Xhat[:,0,:]=X[:,0,:]",
            "remaining_samples": "bias + sum_lag A_lag @ X[t-lag]",
            "causal": True,
            "hidden_layers": 0,
            "activation": None,
            "parameter_count": LINEAR_AR_PARAMETER_COUNT,
            "output_shape": ["B", 128, 9],
        }
    raise ValueError(f"unsupported variant: {variant}")


def augmentation_config(variant: str) -> dict[str, Any]:
    if variant == "PERSISTENCE":
        return {
            "applicable": False,
            "reason": "parameter-free deterministic baseline; no fitting step",
            "inference_input_corruption": False,
        }
    if variant == "LINEAR_AR":
        return {
            "applicable": True,
            "clean_probability": 0.40,
            "gaussian_probability": 0.40,
            "mask_probability": 0.20,
            "gaussian_std": 0.04,
            "gaussian_scope": "whole window and all samples/channels",
            "mask_minimum_samples": 4,
            "mask_maximum_samples": 8,
            "mask_interval": "inclusive",
            "mask_length_sampling": "discrete_uniform_inclusive",
            "mask_contiguous": True,
            "mask_all_channels": True,
            "mask_replacement_value": 0.0,
            "augmentation_roles": [4],
            "validation_augmentation": False,
            "training_target": "uncorrupted clean role-4 window",
        }
    raise ValueError(f"unsupported variant: {variant}")


def protocol_contract(variant: str, scientific_data_sha256: str) -> dict[str, Any]:
    common: dict[str, Any] = {
        "schema": "persistence_linear_ar_nbm.v1",
        "variant": variant,
        "scientific_data_sha256": scientific_data_sha256,
        "sampling_rate_hz": 64,
        "window_samples": 128,
        "stride_samples": 64,
        "scaler_fit_role": 4,
        "nbm_fit_role": 4 if variant == "LINEAR_AR" else None,
        "validation_and_calibration_role": 5,
        "input_preprocessing": "role4 RobustScaler then per-window/per-axis centering",
        "architecture": architecture_config(variant),
        "augmentation": augmentation_config(variant),
        "calibration": "clean role-5 per-axis median bias and MAD scale after freeze/best restore",
    }
    if variant == "LINEAR_AR":
        common.update(
            {
                "loss": "SmoothL1(beta=1.0)",
                "optimizer": "AdamW(lr=0.001,weight_decay=0.0001)",
                "scheduler": "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)",
                "batch_size": 128,
                "maximum_epochs": 300,
                "patience": 20,
                "gradient_clip": 1.0,
                "checkpoint_rule": "lowest clean role-5 validation SmoothL1",
            }
        )
    else:
        common.update(
            {
                "loss": None,
                "optimizer": None,
                "scheduler": None,
                "batch_size": None,
                "maximum_epochs": 0,
                "patience": 0,
                "gradient_clip": None,
                "checkpoint_rule": "deterministic parameter-free freeze",
            }
        )
    return common


def persistence_reconstruct(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3 or tuple(values.shape[1:]) != (128, 9) or not len(values):
        raise ValueError(f"expected non-empty [N,128,9], got {values.shape}")
    output = np.empty_like(values)
    output[:, 0, :] = values[:, 0, :]
    output[:, 1:, :] = values[:, :-1, :]
    return np.ascontiguousarray(output)


@torch.no_grad()
def linear_ar_reconstruct(
    model: MultivariateLinearAR8,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    if x.ndim != 3 or tuple(x.shape[1:]) != (128, 9) or not len(x):
        raise ValueError(f"expected non-empty [N,128,9], got {x.shape}")
    model.eval()
    output = []
    for (batch,) in make_loader(x, batch_size, False, 0, 0):
        output.append(model(batch.to(device)).cpu().numpy().astype(np.float32))
    return np.ascontiguousarray(np.concatenate(output, axis=0))


def train_linear_ar(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    fold_dir: Path,
    device: torch.device,
    fold: int,
    seed: int,
    num_workers: int,
    source_code: dict[str, str],
) -> tuple[MultivariateLinearAR8, dict[str, Any]]:
    set_seed(seed)
    model = MultivariateLinearAR8().to(device)
    initial_state_sha256 = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    criterion = nn.SmoothL1Loss(beta=1.0)
    train_loader = make_loader(train_x, 128, True, seed, num_workers)
    validation_loader = make_loader(validation_x, 128, False, seed, num_workers)
    augmentation_generator = torch.Generator(device=device).manual_seed(seed + 1000)
    checkpoint = fold_dir / "checkpoints" / checkpoint_name("LINEAR_AR")
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, 301):
        model.train()
        total = 0.0
        count = 0
        mode_counts = np.zeros(3, dtype=np.int64)
        for (clean,) in train_loader:
            clean = clean.to(device, non_blocking=True)
            corrupted, counts = corrupt_local_mask(
                clean,
                augmentation_generator,
                mask_minimum_samples=4,
                mask_maximum_samples=8,
            )
            mode_counts += counts
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(corrupted), clean)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite Linear-AR gradient")
            optimizer.step()
            total += float(loss.detach()) * len(clean)
            count += len(clean)
        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.no_grad():
            for (clean,) in validation_loader:
                clean = clean.to(device, non_blocking=True)
                loss = criterion(model(clean), clean)
                validation_total += float(loss) * len(clean)
                validation_count += len(clean)
        train_loss = total / count
        validation_loss = validation_total / validation_count
        scheduler.step(validation_loss)
        improved = validation_loss < best_loss - 1e-10
        history.append(
            {
                "epoch": epoch,
                "train_huber": train_loss,
                "validation_huber": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
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
                    "variant": "LINEAR_AR",
                    "model_state": model.state_dict(),
                    "fold": fold,
                    "epoch": epoch,
                    "validation_huber": validation_loss,
                    "seed": seed,
                    "initial_model_state_sha256": initial_state_sha256,
                    "architecture": architecture_config("LINEAR_AR"),
                    "augmentation": augmentation_config("LINEAR_AR"),
                    "source_code_sha256": source_code,
                },
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"Linear-AR fold epoch={epoch:03d} train={train_loss:.7f} "
            f"val={validation_loss:.7f} stale={stale}/20",
            flush=True,
        )
        if stale >= 20:
            break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    write_csv(fold_dir / "logs" / "linear_ar8_history.csv", history)
    return model, {
        "seed": seed,
        "maximum_epochs": 300,
        "patience": 20,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_huber": best_loss,
        "parameter_count": LINEAR_AR_PARAMETER_COUNT,
        "initial_model_state_sha256": initial_state_sha256,
        "history_file": "logs/linear_ar8_history.csv",
        "checkpoint_file": f"checkpoints/{checkpoint_name('LINEAR_AR')}",
        "architecture": architecture_config("LINEAR_AR"),
        "augmentation": augmentation_config("LINEAR_AR"),
    }


def calibrate(error: np.ndarray) -> dict[str, Any]:
    bias = np.median(error, axis=(0, 1)).astype(np.float32)
    sigma_raw = 1.4826 * np.median(
        np.abs(error - bias[None, None, :]), axis=(0, 1)
    )
    sigma = np.maximum(sigma_raw, 0.05).astype(np.float32)
    return {
        "bias": bias.astype(float).tolist(),
        "sigma_raw": sigma_raw.astype(float).tolist(),
        "sigma": sigma.astype(float).tolist(),
        "sigma_floor": 0.05,
        "floor_applied_channels": np.flatnonzero(sigma_raw < 0.05).astype(int).tolist(),
        "calibration_windows": int(len(error)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed_NBM",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--fold", type=int, choices=FOLDS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--required-seeds", default="0,52,161,5216,52161")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--linear-ar-max-epochs", type=int, default=300)
    parser.add_argument("--linear-ar-patience", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    return device


def _file_hashes(fold_dir: Path, checkpoint: Path) -> dict[str, str]:
    return {
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_sha256": sha256_file(fold_dir / "config.json"),
        "scaler_role4_sha256": sha256_file(fold_dir / "scaler_role4.json"),
        "nbm_frozen_sha256": sha256_file(fold_dir / "nbm_frozen.json"),
    }


def validate_existing(
    fold_dir: Path,
    args: argparse.Namespace,
    scientific_data_sha256: str,
) -> None:
    checkpoint = fold_dir / "checkpoints" / checkpoint_name(args.variant)
    paths = {
        "done": fold_dir / "DONE_NBM.json",
        "config": fold_dir / "config.json",
        "scaler": fold_dir / "scaler_role4.json",
        "frozen": fold_dir / "nbm_frozen.json",
        "checkpoint": checkpoint,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete NBM artifacts: {missing}")
    done = json.loads(paths["done"].read_text(encoding="utf-8"))
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    scaler = json.loads(paths["scaler"].read_text(encoding="utf-8"))
    frozen = json.loads(paths["frozen"].read_text(encoding="utf-8"))
    expected_protocol = protocol_contract(args.variant, scientific_data_sha256)
    current_source_code = source_code_sha256()
    expected = {
        "status": "frozen",
        "fold": args.fold,
        "seed": args.seed,
        "variant": args.variant,
        "scientific_data_sha256": scientific_data_sha256,
        "protocol_sha256": canonical_fingerprint(expected_protocol),
    }
    for key, value in expected.items():
        if done.get(key) != value:
            raise AssertionError(f"stale DONE_NBM {key}")
    for key, value in _file_hashes(fold_dir, checkpoint).items():
        if done.get(key) != value:
            raise AssertionError(f"DONE_NBM artifact hash mismatch: {key}")
    if config.get("protocol") != expected_protocol:
        raise AssertionError("NBM protocol changed")
    if (
        config.get("source_code_sha256") != current_source_code
        or frozen.get("source_code_sha256") != current_source_code
        or done.get("source_code_sha256") != current_source_code
    ):
        raise AssertionError("NBM source code changed")
    if scaler.get("scaler_fit_role") != 4 or scaler.get("scaler") != frozen.get("scaler"):
        raise AssertionError("role-4 scaler mismatch")
    if frozen.get("variant") != args.variant:
        raise AssertionError("frozen variant mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("variant") != args.variant
        or payload.get("fold") != args.fold
        or payload.get("seed") != args.seed
    ):
        raise AssertionError("checkpoint identity mismatch")
    if payload.get("architecture") != architecture_config(args.variant):
        raise AssertionError("checkpoint architecture mismatch")
    if payload.get("source_code_sha256") != current_source_code:
        raise AssertionError("checkpoint source code mismatch")


def run(args: argparse.Namespace, device: torch.device) -> None:
    if parse_csv_ints(args.required_seeds) != REQUIRED_SEEDS or args.seed not in REQUIRED_SEEDS:
        raise ValueError(f"required exact seed set is {REQUIRED_SEEDS}")
    if (args.linear_ar_max_epochs, args.linear_ar_patience) != (300, 20):
        raise ValueError("Linear-AR training is frozen to max300/pat20")
    data_dir = args.data_dir.resolve()
    scientific_data = processed_nbm_scientific_manifest(data_dir)
    source_code = source_code_sha256()
    protocol = protocol_contract(args.variant, scientific_data["sha256"])
    protocol_sha256 = canonical_fingerprint(protocol)
    fold_dir = args.output_root.resolve() / f"fold_{args.fold}"
    done_path = fold_dir / "DONE_NBM.json"
    if done_path.exists() and not args.overwrite:
        validate_existing(fold_dir, args, scientific_data["sha256"])
        print(f"SKIP frozen {args.variant} source: {done_path}", flush=True)
        return
    fold_dir.mkdir(parents=True, exist_ok=True)
    dataset = DaphnetDataset.load(data_dir)
    if dataset.sampling_rate_hz != 64 or dataset.n_channels != 9:
        raise AssertionError("this experiment requires the 64-Hz, 9-channel dataset")
    records = {record.record_id: record for record in dataset.records}
    rows_by_fold = {fold: load_fold_rows(data_dir, fold) for fold in FOLDS}
    source_audit = audit_protocol(data_dir, rows_by_fold, records)
    rows = rows_by_fold[args.fold]
    role4 = rows.take_role(4)
    role5 = rows.take_role(5)
    scaler, unique_points = fit_scaler_unique_role4_points(records, role4)
    scaler_payload = {
        "fold": args.fold,
        "seed": args.seed,
        "variant": args.variant,
        "scaler_fit_role": 4,
        "scaler_unique_raw_points": unique_points,
        "scaler": scaler.as_dict(),
        "scientific_data_sha256": scientific_data["sha256"],
        "protocol_sha256": protocol_sha256,
    }
    atomic_json_dump(scaler_payload, fold_dir / "scaler_role4.json")
    role4_x = prepare_nbm_windows(scaler, raw_windows(records, role4), center=True)
    role5_x = prepare_nbm_windows(scaler, raw_windows(records, role5), center=True)
    config = {
        "experiment": "Persistence_vs_multivariate_Linear_AR8_NBM_source",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "seed": args.seed,
        "variant": args.variant,
        "required_seeds": list(REQUIRED_SEEDS),
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "roles": {str(key): value for key, value in ROLES.items()},
        "subjects": list(SUBJECTS),
        "excluded_subjects": ["S04", "S10"],
        "role_counts": {str(role): int(np.sum(rows.role == role)) for role in ROLES},
        "source_audit": source_audit,
        "source_code_sha256": source_code,
        "classifier_or_test_roles_accessed": False,
    }
    atomic_json_dump(config, fold_dir / "config.json")
    checkpoint = fold_dir / "checkpoints" / checkpoint_name(args.variant)
    if args.variant == "PERSISTENCE":
        reconstruction = persistence_reconstruct(role5_x)
        training = {
            "seed": args.seed,
            "maximum_epochs": 0,
            "patience": 0,
            "epochs_completed": 0,
            "best_epoch": 0,
            "best_validation_huber": float(
                nn.functional.smooth_l1_loss(
                    torch.from_numpy(reconstruction), torch.from_numpy(role5_x), beta=1.0
                )
            ),
            "parameter_count": 0,
            "initial_model_state_sha256": None,
            "architecture": architecture_config(args.variant),
            "augmentation": augmentation_config(args.variant),
        }
        atomic_torch_save(
            {
                "variant": args.variant,
                "seed": args.seed,
                "fold": args.fold,
                "architecture": architecture_config(args.variant),
                "parameter_count": 0,
                "source_code_sha256": source_code,
            },
            checkpoint,
        )
    else:
        model, training = train_linear_ar(
            role4_x,
            role5_x,
            fold_dir,
            device,
            args.fold,
            args.seed,
            args.num_workers,
            source_code,
        )
        reconstruction = linear_ar_reconstruct(model, role5_x, device)
    calibration = calibrate(role5_x - reconstruction)
    frozen = {
        "variant": args.variant,
        "protocol_sha256": protocol_sha256,
        "scientific_data_sha256": scientific_data["sha256"],
        "scaler": scaler.as_dict(),
        "scaler_fit_role": 4,
        "scaler_unique_raw_points": unique_points,
        "nbm_train_role": 4 if args.variant == "LINEAR_AR" else None,
        "nbm_earlystop_role": 5 if args.variant == "LINEAR_AR" else None,
        "nbm_calibration_role": 5,
        "validation_mask_or_noise": False,
        "best_checkpoint_restored_before_calibration": True,
        "training": training,
        "calibration": calibration,
        "source_code_sha256": source_code,
        "classifier_or_test_roles_accessed": False,
    }
    atomic_json_dump(frozen, fold_dir / "nbm_frozen.json")
    hashes = _file_hashes(fold_dir, checkpoint)
    done = {
        "status": "frozen",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "seed": args.seed,
        "variant": args.variant,
        "checkpoint": str(checkpoint.resolve()),
        **hashes,
        "parameter_count": architecture_config(args.variant)["parameter_count"],
        "best_epoch": training["best_epoch"],
        "epochs_completed": training["epochs_completed"],
        "protocol_sha256": protocol_sha256,
        "scientific_data_sha256": scientific_data["sha256"],
        "source_code_sha256": source_code,
        "classifier_or_test_roles_accessed": False,
    }
    atomic_json_dump(done, done_path)
    validate_existing(fold_dir, args, scientific_data["sha256"])
    print(json.dumps(done, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    run(args, device)


if __name__ == "__main__":
    main()
