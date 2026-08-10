#!/usr/bin/env python3
"""Train one exact-seed patch-Transformer reconstruction NBM fold.

The worker is intentionally limited to the canonical 64-Hz processed_NBM
protocol: [B,9,128] -> 16 patches -> Transformer encoder -> [B,8,64]
bottleneck -> self-attention decoder -> [B,9,128].  It accesses only roles 4
and 5.  Classifier and permanent-test roles are never materialized here.
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
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    ROLES,
    SUBJECTS,
    audit_protocol,
    fit_scaler_unique_role4_points,
    load_fold_rows,
    raw_windows,
    save_figure_bundle,
    write_csv,
    write_json,
)
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    atomic_torch_save,
    centered_scaled_bct,
)
from scripts.run_daphnet_residual_calibration_abcd import sha256_file
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    corrupt,
    make_loader,
    set_seed,
)

FOLDS = (0, 1, 2)
CHANNELS = 9
WINDOW_SAMPLES = 128
PATCH_SIZE = 8
TOKEN_COUNT = WINDOW_SAMPLES // PATCH_SIZE
PATCH_DIM = CHANNELS * PATCH_SIZE
MODEL_DIM = 192
HEADS = 6
FFN_DIM = 576
ENCODER_LAYERS = 4
DECODER_LAYERS = 2
MERGED_TOKENS = TOKEN_COUNT // 2
BOTTLENECK_DIM = 64


def parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid unique integer list: {value}")
    return values


class PatchTransformerNBM(nn.Module):
    """Skip-free patch Transformer denoising autoencoder for [B,9,128]."""

    def __init__(self, dropout: float = 0.10) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.patch_projection = nn.Linear(PATCH_DIM, MODEL_DIM)
        self.encoder_position = nn.Parameter(torch.zeros(1, TOKEN_COUNT, MODEL_DIM))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM,
            nhead=HEADS,
            dim_feedforward=FFN_DIM,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=ENCODER_LAYERS,
            enable_nested_tensor=False,
        )
        self.to_bottleneck = nn.Sequential(
            nn.Linear(2 * MODEL_DIM, 128),
            nn.GELU(),
            nn.Linear(128, BOTTLENECK_DIM),
        )

        self.from_bottleneck = nn.Linear(BOTTLENECK_DIM, MODEL_DIM)
        self.decoder_position = nn.Parameter(torch.zeros(1, TOKEN_COUNT, MODEL_DIM))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM,
            nhead=HEADS,
            dim_feedforward=FFN_DIM,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        # There is no encoder-memory skip in the requested diagram.  Therefore
        # the two decoder blocks are self-attention Transformer blocks operating
        # only on tokens reconstructed from Z.
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=DECODER_LAYERS,
            enable_nested_tensor=False,
        )
        self.patch_output = nn.Linear(MODEL_DIM, PATCH_DIM)
        nn.init.trunc_normal_(self.encoder_position, std=0.02)
        nn.init.trunc_normal_(self.decoder_position, std=0.02)

    @staticmethod
    def patchify(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (CHANNELS, WINDOW_SAMPLES):
            raise ValueError(f"expected [B,9,128], got {tuple(x.shape)}")
        patches = x.unfold(dimension=2, size=PATCH_SIZE, step=PATCH_SIZE)
        return patches.permute(0, 2, 1, 3).reshape(x.shape[0], TOKEN_COUNT, PATCH_DIM)

    @staticmethod
    def fold_patches(patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3 or tuple(patches.shape[1:]) != (TOKEN_COUNT, PATCH_DIM):
            raise ValueError(f"expected [B,16,72], got {tuple(patches.shape)}")
        values = patches.reshape(
            patches.shape[0], TOKEN_COUNT, CHANNELS, PATCH_SIZE
        )
        return values.permute(0, 2, 1, 3).reshape(
            patches.shape[0], CHANNELS, WINDOW_SAMPLES
        )

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_projection(self.patchify(x))
        return self.encoder(tokens + self.encoder_position)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encode_tokens(x)
        merged = tokens.reshape(tokens.shape[0], MERGED_TOKENS, 2 * MODEL_DIM)
        z = self.to_bottleneck(merged)
        if tuple(z.shape[1:]) != (MERGED_TOKENS, BOTTLENECK_DIM):
            raise RuntimeError(f"unexpected bottleneck shape: {tuple(z.shape)}")
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3 or tuple(z.shape[1:]) != (
            MERGED_TOKENS,
            BOTTLENECK_DIM,
        ):
            raise ValueError(f"expected [B,8,64], got {tuple(z.shape)}")
        tokens = self.from_bottleneck(z)
        # Parameter-free nearest-neighbor token upsampling follows the diagram:
        # every latent token becomes two tokens; decoder positional embeddings
        # then distinguish the two temporal positions.
        tokens = torch.repeat_interleave(tokens, repeats=2, dim=1)
        decoded = self.decoder(tokens + self.decoder_position)
        return self.fold_patches(self.patch_output(decoded))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction = self.decode(self.encode(x))
        if reconstruction.shape != x.shape:
            raise RuntimeError(
                f"reconstruction {tuple(reconstruction.shape)} != input {tuple(x.shape)}"
            )
        return reconstruction

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": "transformer_patch_autoencoder_nbm_v1",
            "dropout": self.dropout,
            "input_shape": ["B", 9, 128],
            "patchify": {
                "patch_size": 8,
                "non_overlapping": True,
                "token_shape": ["B", 16, 72],
                "patch_order": "time-major; each token flattens 9 axes x 8 samples",
            },
            "patch_projection": "Linear(72,192)",
            "encoder_position": "learned [1,16,192], trunc_normal std=0.02",
            "encoder": {
                "layers": 4,
                "d_model": 192,
                "heads": 6,
                "ffn": 576,
                "activation": "GELU",
                "dropout": self.dropout,
                "normalization": "PyTorch post-norm TransformerEncoderLayer",
            },
            "pairwise_token_merge": "adjacent pairs concatenate [B,16,192] to [B,8,384]",
            "bottleneck_projection": "Linear(384,128)+GELU+Linear(128,64)",
            "bottleneck_shape": ["B", 8, 64],
            "decoder_projection": "Linear(64,192)",
            "token_upsampling": "parameter-free repeat_interleave 8 to 16",
            "decoder_position": "learned [1,16,192], trunc_normal std=0.02",
            "decoder": {
                "layers": 2,
                "type": "self-attention blocks; no encoder-memory cross-attention",
                "d_model": 192,
                "heads": 6,
                "ffn": 576,
                "activation": "GELU",
                "dropout": self.dropout,
                "normalization": "PyTorch post-norm TransformerEncoderLayer",
            },
            "patch_output": "Linear(192,72), then exact non-overlapping fold",
            "encoder_decoder_skip_connections": False,
            "output_activation": None,
            "output_shape": ["B", 9, 128],
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }


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
        / "daphnet_transformer_nbm300_C_vs_raw_tcn_ep5pat2_seedset_0_52_161_5216_52161"
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
    parser.add_argument("--nbm-dropout", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    return device


@torch.no_grad()
def reconstruct_transformer(
    model: PatchTransformerNBM,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for (batch,) in make_loader(x, batch_size, False, 0, 0):
        prediction = model(batch.to(device, non_blocking=True))
        outputs.append(prediction.cpu().numpy().astype(np.float32))
    if not outputs:
        raise ValueError("cannot reconstruct an empty window collection")
    return np.concatenate(outputs, axis=0)


def train_transformer_nbm(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    output_dir: Path,
    device: torch.device,
    seed: int,
    num_workers: int,
    max_epochs: int,
    patience: int,
    dropout: float,
) -> tuple[PatchTransformerNBM, dict[str, Any]]:
    set_seed(seed)
    model = PatchTransformerNBM(dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    train_loader = make_loader(train_x, 128, True, seed, num_workers)
    validation_loader = make_loader(validation_x, 128, False, seed, num_workers)
    criterion = nn.SmoothL1Loss(beta=1.0)
    augmentation_generator = torch.Generator(device=device).manual_seed(seed + 1000)
    checkpoint = output_dir / "checkpoints" / "transformer_nbm_best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
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
            augmented_ntc, counts = corrupt(
                clean.transpose(1, 2), augmentation_generator
            )
            network_input = augmented_ntc.transpose(1, 2).contiguous()
            mode_counts += counts
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(network_input), clean)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite Transformer-NBM gradient")
            optimizer.step()
            total_loss += float(loss.detach()) * len(clean)
            total_n += len(clean)

        model.eval()
        validation_total = 0.0
        validation_n = 0
        with torch.no_grad():
            for (clean,) in validation_loader:
                clean = clean.to(device, non_blocking=True)
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
            f"Transformer-NBM epoch={epoch:03d} train={train_loss:.7f} "
            f"val={validation_loss:.7f} lr={learning_rate:.2e} "
            f"stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    history_path = output_dir / "logs" / "transformer_nbm_history.csv"
    write_csv(history_path, history)
    summary = {
        "model_id": "transformer_nbm_best",
        "seed": seed,
        "fit_windows": int(len(train_x)),
        "calibration_validation_windows": int(len(validation_x)),
        "maximum_epochs": int(max_epochs),
        "patience": int(patience),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_huber": best_loss,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history_file": str(history_path.relative_to(output_dir)),
        "checkpoint_file": str(checkpoint.relative_to(output_dir)),
    }
    return model, {"summary": summary, "history": history}


def calibrate_transformer(
    model: PatchTransformerNBM,
    role5_x: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reconstruction = reconstruct_transformer(model, role5_x, device)
    error = role5_x - reconstruction
    bias = np.median(error, axis=(0, 2)).astype(np.float32)
    sigma_raw = 1.4826 * np.median(
        np.abs(error - bias[None, :, None]), axis=(0, 2)
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


def plot_training(output_dir: Path, run: dict[str, Any]) -> None:
    history = run["history"]
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax.plot(epochs, [row["train_huber"] for row in history], label="Role 4 train")
    ax.plot(
        epochs,
        [row["validation_huber"] for row in history],
        label="Role 5 validation",
    )
    ax.axvline(run["summary"]["best_epoch"], color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Epoch", ylabel="SmoothL1 loss", title="Patch Transformer NBM")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure_bundle(fig, output_dir / "transformer_nbm_training_validation")
    plt.close(fig)


def run(args: argparse.Namespace, device: torch.device) -> None:
    required_seeds = parse_csv_ints(args.required_seeds)
    if args.seed not in required_seeds:
        raise ValueError(f"seed must be one of {required_seeds}")
    if args.nbm_max_epochs != 300 or args.nbm_patience != 20:
        raise ValueError("this experiment requires Transformer-NBM max300/pat20")
    if args.nbm_dropout != 0.10:
        raise ValueError("this experiment freezes Transformer dropout at 0.10")

    fold_dir = args.output_root.resolve() / f"fold_{args.fold}"
    done_path = fold_dir / "DONE_NBM.json"
    if done_path.exists() and not args.overwrite:
        print(f"SKIP completed Transformer-NBM fold/seed: {done_path}", flush=True)
        return
    fold_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir.resolve()
    dataset = DaphnetDataset.load(data_dir)
    if dataset.sampling_rate_hz != 64 or dataset.n_channels != CHANNELS:
        raise AssertionError(
            f"expected 64 Hz/9 channels, got {dataset.sampling_rate_hz}/{dataset.n_channels}"
        )
    records = {record.record_id: record for record in dataset.records}
    rows_by_fold = {fold: load_fold_rows(data_dir, fold) for fold in FOLDS}
    source_audit = audit_protocol(data_dir, rows_by_fold, records)
    rows = rows_by_fold[args.fold]
    role4 = rows.take_role(4)
    role5 = rows.take_role(5)
    scaler, unique_points = fit_scaler_unique_role4_points(records, role4)
    role4_x = centered_scaled_bct(scaler, raw_windows(records, role4))
    role5_x = centered_scaled_bct(scaler, raw_windows(records, role5))

    model_probe = PatchTransformerNBM(dropout=args.nbm_dropout)
    architecture = model_probe.architecture_config()
    del model_probe
    config = {
        "experiment": "Transformer_NBM300_schemeC_source",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "seed": args.seed,
        "seed_policy": "exact seed; no fold offset",
        "sampling_rate_hz": 64,
        "window_samples": 128,
        "stride_samples": 64,
        "device": str(device),
        "subjects": list(SUBJECTS),
        "excluded_subjects": ["S04", "S10"],
        "roles": {str(key): value for key, value in ROLES.items()},
        "role_counts": {str(role): int(np.sum(rows.role == role)) for role in ROLES},
        "scaler": "per-channel median/IQR fitted on unique role-4 raw points only",
        "input_preprocessing": "RobustScaler then per-window/per-axis mean subtraction",
        "architecture": architecture,
        "training": {
            "fit_role": 4,
            "validation_role": 5,
            "validation_augmentation": False,
            "augmentation": "40% clean, 40% Gaussian(std=0.04), 20% all-axis time mask(4..8)",
            "loss": "SmoothL1(beta=1.0)",
            "optimizer": "AdamW(lr=0.001,weight_decay=0.0001)",
            "scheduler": "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)",
            "batch_size": 128,
            "maximum_epochs": 300,
            "patience": 20,
            "gradient_clip": 1.0,
            "checkpoint_rule": "lowest clean role-5 validation SmoothL1",
            "restore_best": True,
        },
        "classifier_or_test_roles_accessed": False,
        "source_audit": source_audit,
    }
    write_json(fold_dir / "config.json", config)

    model, run_payload = train_transformer_nbm(
        role4_x,
        role5_x,
        fold_dir,
        device,
        args.seed,
        args.num_workers,
        args.nbm_max_epochs,
        args.nbm_patience,
        args.nbm_dropout,
    )
    bias, sigma, calibration = calibrate_transformer(model, role5_x, device)
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
    checkpoint = fold_dir / "checkpoints" / "transformer_nbm_best.pt"
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
    }
    write_json(fold_dir / "nbm_frozen.json", frozen)
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
        "classifier_or_test_roles_accessed": False,
    }
    write_json(done_path, done)
    print(json.dumps(done, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    run(args, device)


if __name__ == "__main__":
    main()
