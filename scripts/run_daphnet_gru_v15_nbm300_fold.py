#!/usr/bin/env python3
"""Train one exact-seed asymmetric GRU-v1.5 reconstruction NBM fold.

This worker implements the deliberately small change from the retained GRU-v1
baseline: the encoder remains a one-layer 64-unit unidirectional GRU and the
global bottleneck remains 16-dimensional, while the skip-free decoder is
expanded to 96 units.  The worker is restricted to the canonical 64-Hz
``processed_NBM`` protocol and materializes only role 4 (fit) and role 5
(clean early stopping and post-restore MAD calibration).
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
from cnbr_fog.resume import atomic_json_dump, atomic_torch_save
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
from scripts.run_daphnet_residual_calibration_abcd import sha256_file
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    RobustScaler,
    corrupt,
    make_loader,
    prepare_nbm_windows,
    set_seed,
)

FOLDS = (0, 1, 2)
CHANNELS = 9
WINDOW_SAMPLES = 128
ENCODER_HIDDEN = 64
BOTTLENECK_DIM = 16
DECODER_HIDDEN = 96
PARAMETER_COUNT = 48_761
ARCHITECTURE_NAME = "gru_reconstruction_nbm_v15_decoder96"


def validate_existing_nbm(
    fold_dir: Path,
    args: argparse.Namespace,
    scientific_data_sha256: str,
) -> None:
    done_path = fold_dir / "DONE_NBM.json"
    frozen_path = fold_dir / "nbm_frozen.json"
    scaler_path = fold_dir / "scaler_role4.json"
    checkpoint = fold_dir / "checkpoints" / "gru_v15_nbm_best.pt"
    for path in (done_path, frozen_path, scaler_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete GRU-v1.5 NBM artifacts: {path}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    training = frozen["training"]
    expected = {
        "status": "frozen",
        "fold": args.fold,
        "seed": args.seed,
        "maximum_epochs": 300,
        "patience": 20,
        "parameter_count": PARAMETER_COUNT,
        "scientific_data_sha256": scientific_data_sha256,
    }
    for key, value in expected.items():
        if done.get(key) != value:
            raise AssertionError(
                f"stale GRU-v1.5 DONE_NBM {key}: {done.get(key)!r}"
            )
    if training.get("seed") != args.seed:
        raise AssertionError("GRU-v1.5 frozen training seed mismatch")
    if frozen.get("scientific_data_sha256") != scientific_data_sha256:
        raise AssertionError("GRU-v1.5 frozen scientific dataset changed")
    if scaler.get("fold") != args.fold or scaler.get("seed") != args.seed:
        raise AssertionError("GRU-v1.5 role-4 scaler identity mismatch")
    if scaler.get("scientific_data_sha256") != scientific_data_sha256:
        raise AssertionError("GRU-v1.5 role-4 scaler scientific dataset changed")
    if scaler.get("scaler_fit_role") != 4 or scaler.get("scaler") != frozen["scaler"]:
        raise AssertionError("GRU-v1.5 role-4 scaler/frozen scaler mismatch")
    actual_checkpoint_sha256 = sha256_file(checkpoint)
    if done.get("checkpoint_sha256") != actual_checkpoint_sha256:
        raise AssertionError("GRU-v1.5 checkpoint hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("seed") != args.seed:
        raise AssertionError("GRU-v1.5 checkpoint seed mismatch")
    if payload.get("architecture") != training["architecture"]:
        raise AssertionError("GRU-v1.5 checkpoint architecture mismatch")
    if int(payload.get("epoch", -1)) != int(training["best_epoch"]):
        raise AssertionError("GRU-v1.5 checkpoint best epoch mismatch")
    if not np.isclose(
        float(payload.get("validation_huber", np.nan)),
        float(training["best_validation_huber"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("GRU-v1.5 checkpoint validation loss mismatch")


def parse_csv_ints(value: str) -> tuple[int, ...]:
    """Parse a non-empty list of unique exact seeds."""

    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid unique integer list: {value}")
    return values


class GRUV15Decoder96NBM(nn.Module):
    """One-layer asymmetric, skip-free GRU denoising autoencoder."""

    def __init__(
        self,
        *,
        channels: int = CHANNELS,
        window_samples: int = WINDOW_SAMPLES,
        encoder_hidden: int = ENCODER_HIDDEN,
        bottleneck: int = BOTTLENECK_DIM,
        decoder_hidden: int = DECODER_HIDDEN,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.window_samples = int(window_samples)
        self.encoder_hidden = int(encoder_hidden)
        self.bottleneck = int(bottleneck)
        self.decoder_hidden = int(decoder_hidden)
        if (
            self.channels != CHANNELS
            or self.window_samples != WINDOW_SAMPLES
            or self.encoder_hidden != ENCODER_HIDDEN
            or self.bottleneck != BOTTLENECK_DIM
            or self.decoder_hidden != DECODER_HIDDEN
        ):
            raise ValueError(
                "GRU-v1.5 is frozen to channels=9, samples=128, "
                "encoder_hidden=64, bottleneck=16, decoder_hidden=96"
            )

        self.encoder = nn.GRU(
            self.channels,
            self.encoder_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.to_bottleneck = nn.Linear(self.encoder_hidden, self.bottleneck)
        self.to_decoder_hidden = nn.Linear(self.bottleneck, self.decoder_hidden)
        self.decoder = nn.GRU(
            self.channels,
            self.decoder_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.output = nn.Linear(self.decoder_hidden, self.channels)
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count != PARAMETER_COUNT:
            raise RuntimeError(
                f"GRU-v1.5 parameter contract changed: "
                f"{parameter_count} != {PARAMETER_COUNT}"
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (
            self.window_samples,
            self.channels,
        ):
            raise ValueError(
                f"expected [B,{self.window_samples},{self.channels}], "
                f"got {tuple(x.shape)}"
            )
        _, hidden = self.encoder(x)
        z = self.to_bottleneck(hidden[-1])
        if tuple(z.shape[1:]) != (self.bottleneck,):
            raise RuntimeError(f"unexpected bottleneck shape: {tuple(z.shape)}")
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.bottleneck:
            raise ValueError(f"expected [B,{self.bottleneck}], got {tuple(z.shape)}")
        initial = self.to_decoder_hidden(z).unsqueeze(0)
        decoder_input = z.new_zeros(
            (z.shape[0], self.window_samples, self.channels)
        )
        decoded, _ = self.decoder(decoder_input, initial)
        return self.output(decoded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction = self.decode(self.encode(x))
        if reconstruction.shape != x.shape:
            raise RuntimeError(
                f"reconstruction {tuple(reconstruction.shape)} != "
                f"input {tuple(x.shape)}"
            )
        return reconstruction

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": ARCHITECTURE_NAME,
            "input_shape": ["B", 128, 9],
            "encoder": {
                "type": "unidirectional GRU",
                "layers": 1,
                "input_size": self.channels,
                "hidden": self.encoder_hidden,
                "dropout": 0.0,
                "summary": "last hidden state [B,64]",
            },
            "bottleneck_projection": "Linear(64,16)",
            "encoder_gru": {
                "input_size": self.channels,
                "hidden_size": self.encoder_hidden,
                "layers": 1,
                "bidirectional": False,
            },
            "latent_shape": ["B", self.bottleneck],
            "bottleneck_shape": ["B", self.bottleneck],
            "decoder_conditioning": {
                "initial_state": "Linear(16,96)",
                "per_step_input": "128-step all-zero sequence [B,128,9]",
                "raw_or_encoder_token_connection": False,
            },
            "decoder": {
                "type": "unidirectional GRU",
                "layers": 1,
                "input_size": self.channels,
                "hidden": self.decoder_hidden,
                "dropout": 0.0,
            },
            "decoder_gru": {
                "input_size": self.channels,
                "hidden_size": self.decoder_hidden,
                "layers": 1,
                "bidirectional": False,
                "input": "all-zero sequence",
            },
            "output": "Linear(96,9), no output activation",
            "normalization": None,
            "fourier_or_positional_code": False,
            "time_code": False,
            "encoder_decoder_skip_connections": False,
            "skip_connections": False,
            "teacher_forcing": False,
            "output_activation": None,
            "output_shape": ["B", 128, 9],
            "parameter_count": PARAMETER_COUNT,
        }


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    return device


def write_role4_scaler_artifact(
    fold_dir: Path,
    *,
    fold: int,
    seed: int,
    scaler: RobustScaler,
    unique_raw_points: int,
    scientific_data_sha256: str,
) -> dict[str, Any]:
    """Atomically freeze role-4 scaling before role-5 calibration exists."""

    payload = {
        "fold": int(fold),
        "seed": int(seed),
        "scaler_fit_role": 4,
        "scaler_unique_raw_points": int(unique_raw_points),
        "scaler": scaler.as_dict(),
        "scientific_data_sha256": str(scientific_data_sha256),
    }
    atomic_json_dump(payload, fold_dir / "scaler_role4.json")
    return payload


@torch.no_grad()
def reconstruct_gru_v15(
    model: GRUV15Decoder96NBM,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """Reconstruct centered/scaled windows without augmentation."""

    if x.ndim != 3 or tuple(x.shape[1:]) != (WINDOW_SAMPLES, CHANNELS):
        raise ValueError(f"expected [N,128,9], got {tuple(x.shape)}")
    if len(x) == 0:
        raise ValueError("cannot reconstruct an empty window collection")
    model.eval()
    outputs: list[np.ndarray] = []
    for (batch,) in make_loader(x, batch_size, False, 0, 0):
        prediction = model(batch.to(device, non_blocking=True))
        outputs.append(prediction.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def train_gru_v15_nbm(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    output_dir: Path,
    device: torch.device,
    seed: int,
    num_workers: int,
    max_epochs: int,
    patience: int,
) -> tuple[GRUV15Decoder96NBM, dict[str, Any]]:
    """Fit on role 4 and select the lowest clean role-5 SmoothL1 epoch."""

    if max_epochs <= 0 or patience <= 0:
        raise ValueError("NBM max_epochs and patience must be positive")
    expected_tail = (WINDOW_SAMPLES, CHANNELS)
    if tuple(train_x.shape[1:]) != expected_tail or not len(train_x):
        raise ValueError(f"invalid non-empty role-4 array: {train_x.shape}")
    if tuple(validation_x.shape[1:]) != expected_tail or not len(validation_x):
        raise ValueError(f"invalid non-empty role-5 array: {validation_x.shape}")

    set_seed(seed)
    model = GRUV15Decoder96NBM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    train_loader = make_loader(train_x, 128, True, seed, num_workers)
    validation_loader = make_loader(validation_x, 128, False, seed, num_workers)
    criterion = nn.SmoothL1Loss(beta=1.0)
    augmentation_generator = torch.Generator(device=device).manual_seed(seed + 1000)
    checkpoint = output_dir / "checkpoints" / "gru_v15_nbm_best.pt"
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
            network_input, counts = corrupt(clean, augmentation_generator)
            mode_counts += counts
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(network_input), clean)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite GRU-v1.5 NBM gradient")
            optimizer.step()
            total_loss += float(loss.detach()) * len(clean)
            total_n += len(clean)

        model.eval()
        validation_total = 0.0
        validation_n = 0
        with torch.no_grad():
            for (clean,) in validation_loader:
                clean = clean.to(device, non_blocking=True)
                # Role 5 is always evaluated clean: no corrupt() call here.
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
                    "architecture": model.architecture_config(),
                },
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"GRU-v1.5 NBM epoch={epoch:03d} train={train_loss:.7f} "
            f"val={validation_loss:.7f} lr={learning_rate:.2e} "
            f"stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload["architecture"] != model.architecture_config():
        raise RuntimeError("GRU-v1.5 checkpoint architecture contract mismatch")
    model.load_state_dict(payload["model_state"])
    history_path = output_dir / "logs" / "gru_v15_nbm_history.csv"
    write_csv(history_path, history)
    summary = {
        "model_id": "gru_v15_nbm_best",
        "seed": seed,
        "fit_windows": int(len(train_x)),
        "calibration_validation_windows": int(len(validation_x)),
        "maximum_epochs": int(max_epochs),
        "patience": int(patience),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_huber": best_loss,
        "parameter_count": PARAMETER_COUNT,
        "history_file": str(history_path.relative_to(output_dir)),
        "checkpoint_file": str(checkpoint.relative_to(output_dir)),
    }
    return model, {"summary": summary, "history": history}


def calibrate_gru_v15(
    model: GRUV15Decoder96NBM,
    role5_x: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute channel-wise role-5 median bias and robust MAD scale."""

    reconstruction = reconstruct_gru_v15(model, role5_x, device)
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
        "floor_applied_channels": np.flatnonzero(sigma_raw < 0.05)
        .astype(int)
        .tolist(),
        "calibration_windows": int(len(role5_x)),
    }


def plot_training(output_dir: Path, run_payload: dict[str, Any]) -> None:
    history = run_payload["history"]
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax.plot(epochs, [row["train_huber"] for row in history], label="Role 4 train")
    ax.plot(
        epochs,
        [row["validation_huber"] for row in history],
        label="Role 5 validation",
    )
    ax.axvline(
        run_payload["summary"]["best_epoch"],
        color="black",
        linestyle="--",
        linewidth=1,
    )
    ax.set(
        xlabel="Epoch",
        ylabel="SmoothL1 loss",
        title="Asymmetric GRU-v1.5 reconstruction NBM",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure_bundle(fig, output_dir / "gru_v15_nbm_training_validation")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT
        / "dataset"
        / "1.Daphnet Freezing of Gait Dataset"
        / "processed_NBM",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / (
            "daphnet_gru_v15_nbm300_C_vs_raw_tcn_ep5pat2_"
            "seedset_0_52_161_5216_52161"
        )
        / "nbm_source"
        / "seed_0",
    )
    parser.add_argument("--fold", type=int, choices=FOLDS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--required-seeds", default="0,52,161,5216,52161")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-max-epochs", type=int, default=300)
    parser.add_argument("--nbm-patience", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace, device: torch.device) -> None:
    required_seeds = parse_csv_ints(args.required_seeds)
    if args.seed not in required_seeds:
        raise ValueError(f"seed must be one of {required_seeds}")
    if args.nbm_max_epochs != 300 or args.nbm_patience != 20:
        raise ValueError("this experiment requires GRU-v1.5 NBM max300/pat20")

    fold_dir = args.output_root.resolve() / f"fold_{args.fold}"
    done_path = fold_dir / "DONE_NBM.json"
    data_dir = args.data_dir.resolve()
    scientific_data = processed_nbm_scientific_manifest(data_dir)
    if done_path.exists() and not args.overwrite:
        validate_existing_nbm(fold_dir, args, scientific_data["sha256"])
        print(f"SKIP completed GRU-v1.5 NBM fold/seed: {done_path}", flush=True)
        return
    fold_dir.mkdir(parents=True, exist_ok=True)
    dataset = DaphnetDataset.load(data_dir)
    if dataset.sampling_rate_hz != 64 or dataset.n_channels != CHANNELS:
        raise AssertionError(
            f"expected 64 Hz/9 channels, got "
            f"{dataset.sampling_rate_hz}/{dataset.n_channels}"
        )
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
    )
    role4_x = prepare_nbm_windows(
        scaler, raw_windows(records, role4), center=True
    )
    role5_x = prepare_nbm_windows(
        scaler, raw_windows(records, role5), center=True
    )

    probe = GRUV15Decoder96NBM()
    architecture = probe.architecture_config()
    del probe
    config = {
        "experiment": "GRU_v15_NBM300_schemeC_source",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "seed": args.seed,
        "seed_policy": "exact seed; no fold offset",
        "required_seeds": list(required_seeds),
        "sampling_rate_hz": 64,
        "window_samples": 128,
        "stride_samples": 64,
        "device": str(device),
        "subjects": list(SUBJECTS),
        "excluded_subjects": ["S04", "S10"],
        "roles": {str(key): value for key, value in ROLES.items()},
        "role_counts": {
            str(role): int(np.sum(rows.role == role)) for role in ROLES
        },
        "scaler": "per-channel median/IQR fitted on unique role-4 raw points only",
        "input_preprocessing": (
            "RobustScaler then per-window/per-axis mean subtraction"
        ),
        "architecture": architecture,
        "training": {
            "fit_role": 4,
            "validation_role": 5,
            "validation_augmentation": False,
            "augmentation": (
                "40% clean, 40% Gaussian(std=0.04), "
                "20% all-axis time mask(4..8)"
            ),
            "loss": "SmoothL1(beta=1.0)",
            "optimizer": "AdamW(lr=0.001,weight_decay=0.0001)",
            "scheduler": "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)",
            "batch_size": 128,
            "maximum_epochs": 300,
            "patience": 20,
            "gradient_clip": 1.0,
            "checkpoint_rule": "lowest clean role-5 validation SmoothL1",
            "checkpoint_write": "atomic temporary file then replace",
            "restore_best": True,
            "post_restore_calibration": "role-5 channel-wise median and MAD",
        },
        "classifier_or_test_roles_accessed": False,
        "source_audit": source_audit,
        "scientific_data_sha256": scientific_data["sha256"],
    }
    atomic_json_dump(config, fold_dir / "config.json")

    model, run_payload = train_gru_v15_nbm(
        role4_x,
        role5_x,
        fold_dir,
        device,
        args.seed,
        args.num_workers,
        args.nbm_max_epochs,
        args.nbm_patience,
    )
    _bias, _sigma, calibration = calibrate_gru_v15(model, role5_x, device)
    plot_training(fold_dir, run_payload)
    summary = run_payload["summary"]
    training = {
        **summary,
        "maximum_epochs": 300,
        "patience": 20,
        "loss": "SmoothL1(beta=1.0)",
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "scheduler": "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)",
        "augmentation": {
            "clean_probability": 0.40,
            "gaussian_probability": 0.40,
            "mask_probability": 0.20,
            "gaussian_std": 0.04,
            "mask_minimum_samples": 4,
            "mask_maximum_samples": 8,
            "mask_all_channels": True,
        },
        "architecture": architecture,
    }
    checkpoint = fold_dir / "checkpoints" / "gru_v15_nbm_best.pt"
    frozen = {
        "scaler": scaler.as_dict(),
        "scaler_fit_role": 4,
        "scaler_unique_raw_points": unique_points,
        "nbm_train_role": 4,
        "nbm_earlystop_and_calibration_role": 5,
        "validation_mask_or_noise": False,
        "best_checkpoint_restored_before_calibration": True,
        "training": training,
        "calibration": calibration,
        "classifier_or_test_roles_accessed": False,
        "scientific_data_sha256": scientific_data["sha256"],
    }
    atomic_json_dump(frozen, fold_dir / "nbm_frozen.json")
    done = {
        "status": "frozen",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "seed": args.seed,
        "seed_policy": "exact",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_epoch": summary["best_epoch"],
        "epochs_completed": summary["epochs_completed"],
        "best_validation_huber": summary["best_validation_huber"],
        "maximum_epochs": 300,
        "patience": 20,
        "parameter_count": PARAMETER_COUNT,
        "classifier_or_test_roles_accessed": False,
        "scientific_data_sha256": scientific_data["sha256"],
    }
    atomic_json_dump(done, done_path)
    print(json.dumps(done, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    run(args, device)


if __name__ == "__main__":
    main()
