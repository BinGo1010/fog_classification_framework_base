#!/usr/bin/env python
"""Within-subject GRU-BASE-NBM + scheme-C TCN on processed_NBM_Exp.

For every P01-P08 subject/fold/seed job:
role 4 fits the RobustScaler and GRU-NBM, role 5 early-stops the NBM and
calibrates residual scale, roles 6/7 train the TCN, roles 2/3 select the TCN
checkpoint and threshold, and roles 0/1 remain locked behind a global barrier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.evaluation import binary_metrics
from cnbr_fog.resume import (
    atomic_json_dump,
    atomic_npz_save,
    atomic_torch_save,
    canonical_fingerprint,
    sha256_file,
)
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts import run_all_dataset_processed_nbm_exp_within_subject_raw_tcn as raw_base
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import RepresentationTCNM
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    GRUReconstructionNBM,
    RobustScaler,
    set_seed,
)


SUBJECTS = raw_base.SUBJECTS
FOLDS = raw_base.FOLDS
SEEDS = raw_base.SEEDS
ROLES = raw_base.ROLES
WINDOW_SAMPLES = 128
SAMPLING_RATE_HZ = 64
RAW_CHANNELS = 30
TCN_INPUT_CHANNELS = 90
HIDDEN = 64
BOTTLENECK = 16
NBM_PARAMETER_COUNT = 40_942
TCN_PARAMETER_COUNT = 143_649
NBM_VARIANT = "GRU_BASE_MASK4_8"
METRIC_KEYS = raw_base.METRIC_KEYS
EXPERIMENT_SCHEMA = "all_dataset_within_subject_gru_nbm_tcn.v1"
BARRIER_SCHEMA = "all_dataset_within_subject_gru_nbm_tcn_barrier.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_seed_list(text: str) -> tuple[int, ...]:
    return raw_base.parse_seed_list(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "seal", "evaluate", "aggregate"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subject", choices=SUBJECTS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--nbm-max-epochs", type=int, default=300)
    parser.add_argument("--nbm-patience", type=int, default=20)
    parser.add_argument("--tcn-max-epochs", type=int, default=5)
    parser.add_argument("--tcn-patience", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_dir(root: Path, subject: str, fold: int, seed: int) -> Path:
    return root / "runs" / subject / f"fold_{fold}" / f"seed_{seed}"


def require_job_args(args: argparse.Namespace) -> tuple[str, int, int]:
    if args.subject is None or args.fold is None or args.seed is None:
        raise ValueError(f"stage={args.stage} requires --subject, --fold and --seed")
    return str(args.subject), int(args.fold), int(args.seed)


def resolve_device(spec: str) -> torch.device:
    return raw_base.resolve_device(spec)


def architecture_config() -> dict[str, Any]:
    model = GRUReconstructionNBM(channels=RAW_CHANNELS, hidden=HIDDEN, bottleneck=BOTTLENECK)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != NBM_PARAMETER_COUNT:
        raise RuntimeError(f"30-channel GRU-NBM parameter contract changed: {count}")
    return {
        "name": "gru_reconstruction_nbm_v1_30channel",
        "input_shape": ["B", 128, 30],
        "encoder": "unidirectional GRU(input=30,hidden=64,layers=1), last hidden state",
        "bottleneck": "Linear(64,16), latent [B,16]",
        "decoder_initial_state": "Linear(16,64)",
        "decoder": "unidirectional GRU(input=30,hidden=64,layers=1), 128-step all-zero input",
        "output": "Linear(64,30), no activation",
        "skip_connections": False,
        "teacher_forcing": False,
        "parameter_count": NBM_PARAMETER_COUNT,
    }


def augmentation_config() -> dict[str, Any]:
    return {
        "clean_probability": 0.40,
        "gaussian_probability": 0.40,
        "mask_probability": 0.20,
        "gaussian_std": 0.04,
        "mask_minimum_samples": 4,
        "mask_maximum_samples": 8,
        "mask_length_sampling": "discrete_uniform_inclusive",
        "mask_contiguous": True,
        "mask_all_channels": True,
        "mask_replacement_value": 0.0,
        "augmentation_roles": [4],
        "validation_augmentation": False,
    }


def load_plan(root: Path) -> dict[str, Any]:
    path = root / "EXPERIMENT_PLAN.json"
    if not path.is_file():
        raise FileNotFoundError(f"launcher plan missing: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != EXPERIMENT_SCHEMA:
        raise AssertionError(f"unexpected plan schema: {plan.get('schema')}")
    return plan


def validate_plan_args(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    expected = {
        "data_dir": str(args.data_dir.resolve()),
        "batch_size": int(args.batch_size),
        "nbm_max_epochs": int(args.nbm_max_epochs),
        "nbm_patience": int(args.nbm_patience),
        "tcn_max_epochs": int(args.tcn_max_epochs),
        "tcn_patience": int(args.tcn_patience),
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise AssertionError(f"plan/worker mismatch for {key}: {plan.get(key)!r} != {value!r}")


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def centered_scaled_ntc(scaler: RobustScaler, raw: np.ndarray) -> np.ndarray:
    values = scaler.transform(raw)
    values = values - values.mean(axis=1, keepdims=True)
    maximum = float(np.max(np.abs(values.mean(axis=1))))
    if maximum > 5e-5:
        raise AssertionError(f"per-window/per-axis centering failed: {maximum}")
    return np.ascontiguousarray(values, dtype=np.float32)


def nbm_loader(x: np.ndarray, batch_size: int, shuffle: bool, seed: int, workers: int) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(x)).float()),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def corrupt_gru_base(clean: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, np.ndarray]:
    if clean.ndim != 3 or tuple(clean.shape[1:]) != (WINDOW_SAMPLES, RAW_CHANNELS):
        raise ValueError(f"expected [B,128,30], got {tuple(clean.shape)}")
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
        output[gaussian] += 0.04 * noise
    masked_indices = torch.nonzero(modes >= 0.8, as_tuple=False).flatten().tolist()
    for index in masked_indices:
        length = int(torch.randint(4, 9, (1,), device=clean.device, generator=generator))
        start = int(
            torch.randint(0, WINDOW_SAMPLES - length + 1, (1,), device=clean.device, generator=generator)
        )
        output[index, start : start + length, :] = 0.0
    counts = np.asarray(
        [int((modes < 0.4).sum()), int(gaussian.sum()), len(masked_indices)], dtype=np.int64
    )
    return output, counts


@torch.no_grad()
def reconstruct(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for (batch,) in nbm_loader(x, batch_size, False, 0, 0):
        outputs.append(model(batch.to(device, non_blocking=True)).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def train_nbm(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    destination: Path,
    device: torch.device,
    seed: int,
    batch_size: int,
    workers: int,
    maximum_epochs: int,
    patience: int,
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(seed)
    model = GRUReconstructionNBM(
        channels=RAW_CHANNELS, hidden=HIDDEN, bottleneck=BOTTLENECK
    ).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != NBM_PARAMETER_COUNT:
        raise RuntimeError("GRU-NBM parameter contract changed")
    initial_state = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    criterion = nn.SmoothL1Loss(beta=1.0)
    train_batches = nbm_loader(train_x, batch_size, True, seed, workers)
    validation_batches = nbm_loader(validation_x, batch_size, False, seed, workers)
    augmentation_generator = torch.Generator(device=device).manual_seed(seed + 1000)
    checkpoint = destination / "checkpoints" / "gru_nbm_best.pt"
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        mode_counts = np.zeros(3, dtype=np.int64)
        for (clean,) in train_batches:
            clean = clean.to(device, non_blocking=True)
            network_input, counts = corrupt_gru_base(clean, augmentation_generator)
            mode_counts += counts
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(network_input), clean)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite GRU-NBM gradient")
            optimizer.step()
            train_total += float(loss.detach()) * len(clean)
            train_count += len(clean)
        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.no_grad():
            for (clean,) in validation_batches:
                clean = clean.to(device, non_blocking=True)
                loss = criterion(model(clean), clean)
                validation_total += float(loss) * len(clean)
                validation_count += len(clean)
        train_loss = train_total / train_count
        validation_loss = validation_total / validation_count
        scheduler.step(validation_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        improved = validation_loss < best_loss - 1e-10
        history.append({
            "epoch": epoch,
            "train_huber": train_loss,
            "validation_huber": validation_loss,
            "learning_rate": learning_rate,
            "clean_windows": int(mode_counts[0]),
            "gaussian_windows": int(mode_counts[1]),
            "masked_windows": int(mode_counts[2]),
            "improved": improved,
        })
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            atomic_torch_save({
                "schema": EXPERIMENT_SCHEMA,
                "model_state": model.state_dict(),
                "seed": seed,
                "epoch": epoch,
                "validation_huber": validation_loss,
                "initial_model_state_sha256": initial_state,
                "architecture": architecture_config(),
                "augmentation": augmentation_config(),
            }, checkpoint)
        else:
            stale += 1
        print(
            f"GRU-NBM epoch={epoch:03d} train={train_loss:.7f} val={validation_loss:.7f} "
            f"lr={learning_rate:.2e} stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("architecture") != architecture_config():
        raise AssertionError("GRU-NBM checkpoint architecture mismatch")
    model.load_state_dict(payload["model_state"])
    return model, {
        "maximum_epochs": maximum_epochs,
        "patience": patience,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_huber": best_loss,
        "initial_model_state_sha256": initial_state,
        "parameter_count": NBM_PARAMETER_COUNT,
        "history": history,
    }


def calibrate(model: nn.Module, role5_x: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reconstruction = reconstruct(model, role5_x, device, batch_size)
    error = role5_x - reconstruction
    bias = np.median(error, axis=(0, 1)).astype(np.float32)
    sigma_raw = 1.4826 * np.median(np.abs(error - bias[None, None, :]), axis=(0, 1))
    sigma = np.maximum(sigma_raw, 0.05).astype(np.float32)
    return bias, sigma, {
        "bias": bias.astype(float).tolist(),
        "sigma_raw": sigma_raw.astype(float).tolist(),
        "sigma": sigma.astype(float).tolist(),
        "sigma_floor": 0.05,
        "floor_applied_channels": np.flatnonzero(sigma_raw < 0.05).astype(int).tolist(),
        "calibration_windows": int(len(role5_x)),
        "scheme_c_uses_bias": False,
        "bias_usage": "bias centers role5 error only while estimating MAD sigma",
    }


def scheme_c_features(
    model: nn.Module,
    scaler: RobustScaler,
    sigma: np.ndarray,
    raw: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    x = centered_scaled_ntc(scaler, raw)
    x_hat = reconstruct(model, x, device, batch_size)
    q = np.clip((x - x_hat) / (sigma[None, None, :] + 1e-6), -12.0, 12.0)
    r = q - q.mean(axis=1, keepdims=True)
    absolute = np.abs(r)
    delta = np.diff(r, axis=1, prepend=r[:, :1, :])
    features = np.concatenate((r, absolute, delta), axis=2).astype(np.float32, copy=False)
    if features.shape[1:] != (WINDOW_SAMPLES, TCN_INPUT_CHANNELS):
        raise AssertionError(f"unexpected scheme-C shape: {features.shape}")
    return np.ascontiguousarray(features.transpose(0, 2, 1), dtype=np.float32)


def tcn_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int, workers: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(np.ascontiguousarray(x)).float(),
            torch.from_numpy(np.asarray(y, dtype=np.float32)),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for batch_x, batch_y in tcn_loader(x, y, batch_size, False, 0, 0):
        probabilities.append(torch.sigmoid(model(batch_x.to(device, non_blocking=True))).cpu().numpy())
        labels.append(batch_y.numpy())
    return np.concatenate(labels).astype(np.int8), np.concatenate(probabilities).astype(np.float64)


def validation_tcn_loss(model: nn.Module, x: np.ndarray, y: np.ndarray, criterion: nn.Module, device: torch.device, batch_size: int) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch_x, batch_y in tcn_loader(x, y, batch_size, False, 0, 0):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            loss = criterion(model(batch_x), batch_y)
            total += float(loss) * len(batch_x)
            count += len(batch_x)
    return total / count


def train_tcn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    destination: Path,
    device: torch.device,
    seed: int,
    batch_size: int,
    workers: int,
    maximum_epochs: int,
    patience: int,
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(seed)
    model = RepresentationTCNM(TCN_INPUT_CHANNELS).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != TCN_PARAMETER_COUNT:
        raise RuntimeError(f"90-channel TCN parameter contract changed: {parameter_count}")
    initial_state = state_dict_sha256(model.state_dict())
    n_nonfog = int(np.sum(train_y == 0))
    n_fog = int(np.sum(train_y == 1))
    if min(n_nonfog, n_fog) == 0:
        raise ValueError("roles6/7 must contain both classes")
    pos_weight = n_nonfog / n_fog
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batches = tcn_loader(train_x, train_y, batch_size, True, seed, workers)
    checkpoint = destination / "checkpoints" / "tcn.pt"
    best_pr_auc = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in batches:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite TCN gradient")
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
        train_bce = total / count
        validation_bce = validation_tcn_loss(
            model, validation_x, validation_y, criterion, device, batch_size
        )
        val_true, val_prob = predict(model, validation_x, validation_y, device, batch_size)
        validation_pr_auc = float(average_precision_score(val_true, val_prob))
        improved = validation_pr_auc > best_pr_auc + 1e-10
        history.append({
            "epoch": epoch,
            "train_weighted_bce": train_bce,
            "validation_weighted_bce": validation_bce,
            "validation_pr_auc": validation_pr_auc,
            "improved": improved,
        })
        if improved:
            best_pr_auc = validation_pr_auc
            best_epoch = epoch
            stale = 0
            atomic_torch_save({
                "schema": EXPERIMENT_SCHEMA,
                "model_state": model.state_dict(),
                "seed": seed,
                "epoch": epoch,
                "validation_pr_auc": validation_pr_auc,
                "input_channels": TCN_INPUT_CHANNELS,
                "initial_model_state_sha256": initial_state,
            }, checkpoint)
        else:
            stale += 1
        print(
            f"TCN epoch={epoch:03d} train={train_bce:.7f} val={validation_bce:.7f} "
            f"val_pr_auc={validation_pr_auc:.7f} stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    return model, {
        "maximum_epochs": maximum_epochs,
        "patience": patience,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_pr_auc": best_pr_auc,
        "n_nonfog_role6": n_nonfog,
        "n_fog_role7": n_fog,
        "pos_weight": pos_weight,
        "initial_model_state_sha256": initial_state,
        "parameter_count": parameter_count,
        "history": history,
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    raw_base.write_csv(path, rows)


def training_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "scaler": "per-axis median/IQR fitted on unique role4 raw samples",
        "nbm_preprocessing": "RobustScaler then per-window/per-axis time centering",
        "nbm": architecture_config(),
        "augmentation": augmentation_config(),
        "nbm_loss": "SmoothL1(beta=1.0), corrupted input predicts clean target",
        "nbm_optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)",
        "nbm_scheduler": "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)",
        "nbm_maximum_epochs": args.nbm_max_epochs,
        "nbm_patience": args.nbm_patience,
        "nbm_checkpoint": "minimum clean role5 SmoothL1",
        "calibration": "after restoring best NBM, role5 b=median(e), sigma=max(1.4826*MAD(e-b),0.05)",
        "scheme_c": "e=X-Xhat; q=clip(e/(sigma+1e-6),-12,12); r=q-mean_t(q); [r,abs(r),delta(r)]",
        "scheme_c_uses_bias_b": False,
        "tcn_input_shape": ["B", 90, 128],
        "tcn": "RepresentationTCNM 90->32->64->64->128; dilations1/2/4/8; GAP; one logit",
        "classifier_train_roles": [6, 7],
        "classifier_validation_roles": [2, 3],
        "classifier_test_roles": [0, 1],
        "tcn_loss": "BCEWithLogitsLoss(pos_weight=N_role6/N_role7)",
        "tcn_optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)",
        "tcn_maximum_epochs": args.tcn_max_epochs,
        "tcn_patience": args.tcn_patience,
        "tcn_checkpoint": "maximum roles2/3 PR-AUC",
        "batch_size": args.batch_size,
        "gradient_clip": 1.0,
        "threshold": "roles2/3 grid 0.05..0.95 step0.01; max balanced accuracy; ties F1 then higher threshold",
    }


def validate_completed_train(destination: Path, plan: dict[str, Any]) -> bool:
    done_path = destination / "DONE_TRAIN.json"
    if not done_path.is_file():
        return False
    frozen_path = destination / "FROZEN_TRAIN.json"
    paths = {
        "nbm_checkpoint_sha256": destination / "checkpoints" / "gru_nbm_best.pt",
        "tcn_checkpoint_sha256": destination / "checkpoints" / "tcn.pt",
        "scaler_sha256": destination / "scaler_role4.json",
        "calibration_sha256": destination / "calibration_role5.json",
    }
    if not frozen_path.is_file() or not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"incomplete completed training job: {destination}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    valid = (
        done.get("frozen_sha256") == sha256_file(frozen_path)
        and done.get("frozen_id") == frozen.get("frozen_id")
        and frozen.get("data_scientific_sha256") == plan["data_scientific_sha256"]
        and frozen.get("code_sha256") == plan["code_sha256"]
        and all(frozen.get(name) == sha256_file(path) for name, path in paths.items())
    )
    if not valid:
        raise AssertionError(f"completed train job failed artifact validation: {destination}")
    return True


def run_train(args: argparse.Namespace) -> None:
    subject, fold, seed = require_job_args(args)
    output_root = args.output_root.resolve()
    destination = run_dir(output_root, subject, fold, seed)
    plan = load_plan(output_root)
    validate_plan_args(args, plan)
    if not args.overwrite and validate_completed_train(destination, plan):
        print(f"SKIP validated completed train job: {destination}", flush=True)
        return
    dataset = DaphnetDataset.load(args.data_dir.resolve())
    if dataset.sampling_rate_hz != SAMPLING_RATE_HZ or dataset.n_channels != RAW_CHANNELS:
        raise AssertionError(f"expected 64Hz/30 channels, got {dataset.sampling_rate_hz}/{dataset.n_channels}")
    rows = raw_base.load_subject_rows(args.data_dir.resolve(), dataset, subject, fold)
    role4 = rows.take_role(4)
    role5 = rows.take_role(5)
    role67 = rows.take_role(6, 7)
    role23 = rows.take_role(2, 3)
    scaler, scaler_points = raw_base.fit_scaler_unique_role4_points(dataset, role4)
    role4_x = centered_scaled_ntc(scaler, raw_base.raw_windows(dataset, role4))
    role5_x = centered_scaled_ntc(scaler, raw_base.raw_windows(dataset, role5))
    device = resolve_device(args.device)
    nbm, nbm_training = train_nbm(
        role4_x, role5_x, destination, device, seed, args.batch_size,
        args.num_workers, args.nbm_max_epochs, args.nbm_patience,
    )
    bias, sigma, calibration = calibrate(nbm, role5_x, device, args.batch_size)
    scaler_path = destination / "scaler_role4.json"
    calibration_path = destination / "calibration_role5.json"
    atomic_json_dump({
        "schema": EXPERIMENT_SCHEMA,
        "subject": subject,
        "fold": fold,
        "seed": seed,
        "fit_role": 4,
        "unique_raw_points": scaler_points,
        "scaler": scaler.as_dict(),
    }, scaler_path)
    atomic_json_dump({
        "schema": EXPERIMENT_SCHEMA,
        "subject": subject,
        "fold": fold,
        "seed": seed,
        "source_role": 5,
        **calibration,
    }, calibration_path)
    train_x = scheme_c_features(
        nbm, scaler, sigma, raw_base.raw_windows(dataset, role67), device, args.batch_size
    )
    validation_x = scheme_c_features(
        nbm, scaler, sigma, raw_base.raw_windows(dataset, role23), device, args.batch_size
    )
    tcn, tcn_training = train_tcn(
        train_x, role67.label, validation_x, role23.label, destination, device, seed,
        args.batch_size, args.num_workers, args.tcn_max_epochs, args.tcn_patience,
    )
    val_true, val_prob = predict(tcn, validation_x, role23.label, device, args.batch_size)
    threshold, validation_metrics = raw_base.choose_threshold(val_true, val_prob)
    nbm_history_path = destination / "nbm_history.csv"
    tcn_history_path = destination / "tcn_history.csv"
    write_csv(nbm_history_path, nbm_training["history"])
    write_csv(tcn_history_path, tcn_training["history"])
    nbm_checkpoint = destination / "checkpoints" / "gru_nbm_best.pt"
    tcn_checkpoint = destination / "checkpoints" / "tcn.pt"
    frozen = {
        "schema": EXPERIMENT_SCHEMA,
        "status": "frozen_before_permanent_test",
        "created_utc": utc_now(),
        "subject": subject,
        "fold": fold,
        "seed": seed,
        "nbm_variant": NBM_VARIANT,
        "data_scientific_sha256": plan["data_scientific_sha256"],
        "code_sha256": plan["code_sha256"],
        "nbm_checkpoint_sha256": sha256_file(nbm_checkpoint),
        "tcn_checkpoint_sha256": sha256_file(tcn_checkpoint),
        "scaler_sha256": sha256_file(scaler_path),
        "calibration_sha256": sha256_file(calibration_path),
        "nbm_history_sha256": sha256_file(nbm_history_path),
        "tcn_history_sha256": sha256_file(tcn_history_path),
        "threshold": threshold,
        "threshold_source_roles": [2, 3],
        "validation_metrics": validation_metrics,
        "nbm_training": {key: value for key, value in nbm_training.items() if key != "history"},
        "tcn_training": {key: value for key, value in tcn_training.items() if key != "history"},
        "calibration": calibration,
        "training_contract": training_contract(args),
        "role_counts": {str(role): int(np.sum(rows.role == role)) for role in ROLES},
        "permanent_test_materialized": False,
    }
    frozen["frozen_id"] = canonical_fingerprint(frozen)
    frozen_path = destination / "FROZEN_TRAIN.json"
    atomic_json_dump(frozen, frozen_path)
    atomic_json_dump({
        "schema": EXPERIMENT_SCHEMA,
        "status": "train_complete",
        "subject": subject,
        "fold": fold,
        "seed": seed,
        "frozen_id": frozen["frozen_id"],
        "frozen_sha256": sha256_file(frozen_path),
    }, destination / "DONE_TRAIN.json")
    print(
        f"TRAIN COMPLETE subject={subject} fold={fold} seed={seed} "
        f"nbm_best={nbm_training['best_epoch']} tcn_best={tcn_training['best_epoch']} "
        f"threshold={threshold:.2f} val_pr_auc={validation_metrics['auprc']:.6f}",
        flush=True,
    )


def load_and_validate_barrier(root: Path, subject: str, fold: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    barrier_path = root / "TRAINING_BARRIER.json"
    if not barrier_path.is_file():
        raise FileNotFoundError("permanent test is locked until TRAINING_BARRIER.json exists")
    barrier = json.loads(barrier_path.read_text(encoding="utf-8"))
    if barrier.get("schema") != BARRIER_SCHEMA or barrier.get("status") != "sealed":
        raise AssertionError("invalid or unsealed training barrier")
    key = f"{subject}/fold_{fold}/seed_{seed}"
    sealed = barrier.get("jobs", {}).get(key)
    if sealed is None:
        raise KeyError(f"job absent from barrier: {key}")
    destination = run_dir(root, subject, fold, seed)
    frozen_path = destination / "FROZEN_TRAIN.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if sha256_file(frozen_path) != sealed["frozen_sha256"] or frozen["frozen_id"] != sealed["frozen_id"]:
        raise AssertionError(f"frozen artifact changed after seal: {key}")
    paths = {
        "nbm_checkpoint_sha256": destination / "checkpoints" / "gru_nbm_best.pt",
        "tcn_checkpoint_sha256": destination / "checkpoints" / "tcn.pt",
        "scaler_sha256": destination / "scaler_role4.json",
        "calibration_sha256": destination / "calibration_role5.json",
    }
    for name, path in paths.items():
        if sha256_file(path) != sealed[name]:
            raise AssertionError(f"{name} changed after seal: {key}")
    return barrier, frozen


def run_evaluate(args: argparse.Namespace) -> None:
    subject, fold, seed = require_job_args(args)
    root = args.output_root.resolve()
    destination = run_dir(root, subject, fold, seed)
    barrier, frozen = load_and_validate_barrier(root, subject, fold, seed)
    current_data = processed_nbm_scientific_manifest(args.data_dir.resolve())["sha256"]
    if current_data != barrier["data_scientific_sha256"]:
        raise AssertionError("dataset changed after training barrier")
    done_path = destination / "DONE_TEST.json"
    if done_path.is_file() and not args.overwrite:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        artifacts = {
            "metrics_sha256": destination / "metrics.json",
            "predictions_sha256": destination / "test_predictions.csv",
            "probabilities_sha256": destination / "test_probabilities.npz",
        }
        if (
            done.get("barrier_id") == barrier["barrier_id"]
            and all(path.is_file() and done.get(name) == sha256_file(path) for name, path in artifacts.items())
        ):
            print(f"SKIP validated completed evaluate job: {destination}", flush=True)
            return
        raise AssertionError("existing test result failed barrier/artifact validation")
    dataset = DaphnetDataset.load(args.data_dir.resolve())
    rows = raw_base.load_subject_rows(args.data_dir.resolve(), dataset, subject, fold)
    test_rows = rows.take_role(0, 1)
    scaler_payload = json.loads((destination / "scaler_role4.json").read_text(encoding="utf-8"))
    calibration_payload = json.loads((destination / "calibration_role5.json").read_text(encoding="utf-8"))
    scaler = RobustScaler(
        np.asarray(scaler_payload["scaler"]["median"], dtype=np.float32),
        np.asarray(scaler_payload["scaler"]["iqr"], dtype=np.float32),
        float(scaler_payload["scaler"]["epsilon"]),
    )
    sigma = np.asarray(calibration_payload["sigma"], dtype=np.float32)
    device = resolve_device(args.device)
    nbm = GRUReconstructionNBM(channels=RAW_CHANNELS, hidden=HIDDEN, bottleneck=BOTTLENECK).to(device)
    nbm_payload = torch.load(destination / "checkpoints" / "gru_nbm_best.pt", map_location=device, weights_only=False)
    if nbm_payload.get("seed") != seed or nbm_payload.get("architecture") != architecture_config():
        raise AssertionError("NBM checkpoint identity mismatch")
    nbm.load_state_dict(nbm_payload["model_state"])
    test_x = scheme_c_features(
        nbm, scaler, sigma, raw_base.raw_windows(dataset, test_rows), device, args.batch_size
    )
    tcn = RepresentationTCNM(TCN_INPUT_CHANNELS).to(device)
    tcn_payload = torch.load(destination / "checkpoints" / "tcn.pt", map_location=device, weights_only=False)
    if tcn_payload.get("seed") != seed or tcn_payload.get("input_channels") != TCN_INPUT_CHANNELS:
        raise AssertionError("TCN checkpoint identity mismatch")
    tcn.load_state_dict(tcn_payload["model_state"])
    y_true, probability = predict(tcn, test_x, test_rows.label, device, args.batch_size)
    threshold = float(frozen["threshold"])
    y_pred = (probability >= threshold).astype(np.int8)
    metrics = binary_metrics(y_true, probability, threshold)
    metrics.update(raw_base.event_metrics(dataset, test_rows, y_pred))
    metrics["pr_auc"] = metrics["auprc"]
    metrics["false_alarms_per_hour"] = metrics["false_alarm_events_per_hour"]
    metrics.update({
        "schema": EXPERIMENT_SCHEMA,
        "subject": subject,
        "fold": fold,
        "seed": seed,
        "barrier_id": barrier["barrier_id"],
        "frozen_id": frozen["frozen_id"],
        "nbm_variant": NBM_VARIANT,
    })
    metrics_path = destination / "metrics.json"
    predictions_path = destination / "test_predictions.csv"
    probabilities_path = destination / "test_probabilities.npz"
    atomic_json_dump(metrics, metrics_path)
    write_csv(predictions_path, ({
        "subject_id": subject,
        "fold": fold,
        "seed": seed,
        "record_id": str(test_rows.record_id[index]),
        "start_index": int(test_rows.start[index]),
        "end_index_exclusive": int(test_rows.end[index]),
        "role_code": int(test_rows.role[index]),
        "window_id": str(test_rows.window_id[index]),
        "y_true": int(y_true[index]),
        "probability": float(probability[index]),
        "threshold": threshold,
        "y_pred": int(y_pred[index]),
    } for index in range(len(test_rows))))
    atomic_npz_save(
        probabilities_path,
        y_true=y_true,
        probability=probability.astype(np.float32),
        y_pred=y_pred,
        threshold=np.asarray(threshold),
        window_id=test_rows.window_id,
    )
    atomic_json_dump({
        "schema": EXPERIMENT_SCHEMA,
        "status": "test_complete",
        "subject": subject,
        "fold": fold,
        "seed": seed,
        "barrier_id": barrier["barrier_id"],
        "metrics_sha256": sha256_file(metrics_path),
        "predictions_sha256": sha256_file(predictions_path),
        "probabilities_sha256": sha256_file(probabilities_path),
    }, done_path)
    print(
        f"TEST COMPLETE subject={subject} fold={fold} seed={seed} "
        f"sens={metrics['sensitivity']:.6f} precision={metrics['precision']:.6f} "
        f"spec={metrics['specificity']:.6f} pr_auc={metrics['pr_auc']:.6f} "
        f"event_sens={metrics['event_sensitivity']} fa_h={metrics['false_alarms_per_hour']:.6f}",
        flush=True,
    )


def run_seal(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    plan = load_plan(root)
    validate_plan_args(args, plan)
    current_data = processed_nbm_scientific_manifest(args.data_dir.resolve())["sha256"]
    if current_data != plan["data_scientific_sha256"]:
        raise AssertionError("dataset changed after experiment plan creation")
    seeds = parse_seed_list(args.seeds)
    jobs: dict[str, Any] = {}
    for subject in SUBJECTS:
        for fold in FOLDS:
            for seed in seeds:
                destination = run_dir(root, subject, fold, seed)
                if not validate_completed_train(destination, plan):
                    raise FileNotFoundError(f"training job incomplete: {destination}")
                frozen_path = destination / "FROZEN_TRAIN.json"
                frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
                if (frozen["subject"], frozen["fold"], frozen["seed"]) != (subject, fold, seed):
                    raise AssertionError(f"frozen job identity mismatch: {destination}")
                key = f"{subject}/fold_{fold}/seed_{seed}"
                jobs[key] = {
                    name: frozen[name] for name in (
                        "frozen_id", "nbm_checkpoint_sha256", "tcn_checkpoint_sha256",
                        "scaler_sha256", "calibration_sha256", "threshold",
                    )
                }
                jobs[key]["frozen_sha256"] = sha256_file(frozen_path)
    core = {
        "schema": BARRIER_SCHEMA,
        "status": "sealed",
        "created_utc": utc_now(),
        "plan_id": plan["plan_id"],
        "data_scientific_sha256": plan["data_scientific_sha256"],
        "subjects": list(SUBJECTS),
        "folds": list(FOLDS),
        "seeds": list(seeds),
        "job_count": len(jobs),
        "jobs": jobs,
    }
    core["barrier_id"] = canonical_fingerprint({key: value for key, value in core.items() if key != "created_utc"})
    atomic_json_dump(core, root / "TRAINING_BARRIER.json")
    print(f"SEALED {len(jobs)} jobs barrier_id={core['barrier_id']}", flush=True)


def mean_std(values: Iterable[float]) -> dict[str, Any]:
    return raw_base.mean_std(values)


def run_aggregate(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    seeds = parse_seed_list(args.seeds)
    barrier = json.loads((root / "TRAINING_BARRIER.json").read_text(encoding="utf-8"))
    if barrier.get("schema") != BARRIER_SCHEMA or barrier.get("status") != "sealed":
        raise AssertionError("strict training barrier missing")
    run_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for fold in FOLDS:
            for seed in seeds:
                destination = run_dir(root, subject, fold, seed)
                done_path = destination / "DONE_TEST.json"
                metrics_path = destination / "metrics.json"
                predictions_path = destination / "test_predictions.csv"
                probabilities_path = destination / "test_probabilities.npz"
                if not all(path.is_file() for path in (done_path, metrics_path, predictions_path, probabilities_path)):
                    raise FileNotFoundError(f"test job incomplete: {destination}")
                done = json.loads(done_path.read_text(encoding="utf-8"))
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if done.get("barrier_id") != barrier["barrier_id"] or metrics.get("barrier_id") != barrier["barrier_id"]:
                    raise AssertionError(f"barrier mismatch: {destination}")
                for name, path in (
                    ("metrics_sha256", metrics_path),
                    ("predictions_sha256", predictions_path),
                    ("probabilities_sha256", probabilities_path),
                ):
                    if done.get(name) != sha256_file(path):
                        raise AssertionError(f"{name} mismatch: {destination}")
                run_rows.append({
                    "subject": subject,
                    "fold": fold,
                    "seed": seed,
                    "threshold": metrics["threshold"],
                    **{key: metrics[key] for key in METRIC_KEYS},
                    "tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"], "tp": metrics["tp"],
                    "evaluable_true_events": metrics["evaluable_true_events"],
                    "detected_true_events": metrics["detected_true_events"],
                    "false_alarm_events": metrics["false_alarm_events"],
                    "evaluated_nonfog_hours": metrics["evaluated_nonfog_hours"],
                })
    subject_seed_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for seed in seeds:
            selected = [row for row in run_rows if row["subject"] == subject and row["seed"] == seed]
            subject_seed_rows.append({
                "subject": subject,
                "seed": seed,
                **{key: float(np.mean([row[key] for row in selected])) for key in METRIC_KEYS},
            })
    subject_summary_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        selected = [row for row in subject_seed_rows if row["subject"] == subject]
        row: dict[str, Any] = {"subject": subject}
        for key in METRIC_KEYS:
            summary = mean_std(item[key] for item in selected)
            row[f"{key}_mean"] = summary["mean"]
            row[f"{key}_std"] = summary["std"]
        subject_summary_rows.append(row)
    overall_seed_rows = []
    for seed in seeds:
        selected = [row for row in subject_seed_rows if row["seed"] == seed]
        overall_seed_rows.append({
            "seed": seed,
            **{key: float(np.mean([row[key] for row in selected])) for key in METRIC_KEYS},
        })
    overall = {key: mean_std(row[key] for row in overall_seed_rows) for key in METRIC_KEYS}
    write_csv(root / "run_metrics.csv", run_rows)
    write_csv(root / "subject_seed_metrics.csv", subject_seed_rows)
    write_csv(root / "subject_summary.csv", subject_summary_rows)
    write_csv(root / "overall_seed_metrics.csv", overall_seed_rows)
    summary = {
        "schema": EXPERIMENT_SCHEMA,
        "model": "GRU BASE Mask4-8 NBM + scheme-C 90-channel TCN",
        "aggregation": "subject/seed macro mean of 3 folds; subject mean+population SD over 5 seeds; overall subject-macro per seed then mean+population SD",
        "event_metric": {
            "version": raw_base.EVENT_METRIC_VERSION,
            "minimum_positive_windows": 2,
            "merge_gap_seconds": 0.5,
            "false_alarm_denominator": "union coverage of evaluated valid Non-FoG samples",
        },
        "subjects": subject_summary_rows,
        "overall": overall,
    }
    atomic_json_dump(summary, root / "summary.json")
    atomic_json_dump({
        "schema": EXPERIMENT_SCHEMA,
        "status": "complete",
        "completed_utc": utc_now(),
        "run_count": len(run_rows),
        "barrier_id": barrier["barrier_id"],
        "summary_sha256": sha256_file(root / "summary.json"),
    }, root / "DONE.json")
    print(json.dumps(overall, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_root = args.output_root.resolve()
    if args.stage == "train":
        run_train(args)
    elif args.stage == "seal":
        run_seal(args)
    elif args.stage == "evaluate":
        run_evaluate(args)
    else:
        run_aggregate(args)


if __name__ == "__main__":
    main()
