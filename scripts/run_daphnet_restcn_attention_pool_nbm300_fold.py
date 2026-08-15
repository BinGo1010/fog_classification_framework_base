#!/usr/bin/env python3
"""Train one strict ResTCN + single-query attention-pooling NBM fold.

Only clean Non-FoG role 4 may fit the RobustScaler and update NBM weights.
Clean Non-FoG role 5 is used only for early stopping and, after restoring the
best checkpoint, residual calibration.  The full 2-second encoder sequence is
compressed by one learned-query attention pool into a global ``[B,16]`` latent
before decoding.  No raw input, encoder token, skip connection, or teacher
forcing path reaches the decoder.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
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
from cnbr_fog.resume import atomic_torch_save
from scripts.run_daphnet_conv_tcn_nbm_gaussian200_fold import (
    GRUMatchedAugmentation,
    gru_matched_augmentation,
)
from scripts.run_daphnet_processed_nbm_centered_residual_tcn import (
    ROLES,
    SUBJECTS,
    audit_protocol,
    fit_scaler_unique_role4_points,
    load_fold_rows,
    raw_windows,
    save_figure_bundle,
    set_seed,
    write_csv,
    write_json,
)
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    TCNResidualStack,
    centered_scaled_bct,
    group_count,
    nbm_loader,
)
from scripts.run_daphnet_residual_calibration_abcd import sha256_file

FOLDS = (0, 1, 2)
CHANNELS = 9
WINDOW_SAMPLES = 128
ENCODED_SAMPLES = 32
ENCODED_CHANNELS = 48
BOTTLENECK_DIM = 16
ATTENTION_HEADS = 4
DROPOUT = 0.10
ARCHITECTURE_NAME = "restcn_single_query_attention_pool_global_z16_nbm_v1"
CHECKPOINT_NAME = "restcn_attention_pool_nbm_best.pt"


def parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid unique integer list: {value}")
    return values


def sinusoidal_position_code(length: int, channels: int) -> torch.Tensor:
    """Return a fixed standard sinusoidal code shaped ``[1,length,channels]``."""
    if length <= 0 or channels <= 0 or channels % 2:
        raise ValueError("length must be positive and channels must be positive/even")
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    scale = torch.exp(
        torch.arange(0, channels, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / channels)
    )
    code = torch.zeros(length, channels, dtype=torch.float32)
    code[:, 0::2] = torch.sin(position * scale)
    code[:, 1::2] = torch.cos(position * scale)
    return code.unsqueeze(0)


class ResTCNSingleQueryAttentionPoolNBM(nn.Module):
    """Skip-free normal-template generator with a strict global Z16 bottleneck."""

    def __init__(self, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.dropout_probability = float(dropout)

        # Same-scale residual blocks are deliberately retained.  None of these
        # tensors bypasses the global bottleneck into the decoder.
        self.encoder_stem = nn.Sequential(
            nn.Conv1d(9, 24, kernel_size=7, stride=1, padding=3, bias=False),
            nn.GroupNorm(group_count(24), 24),
            nn.GELU(),
        )
        self.encoder_down32 = nn.Sequential(
            nn.Conv1d(24, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(group_count(32), 32),
            nn.GELU(),
        )
        self.encoder_tcn32 = TCNResidualStack(
            32, (1, 2, 4, 8), self.dropout_probability
        )
        self.encoder_down48 = nn.Sequential(
            nn.Conv1d(32, 48, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(group_count(48), 48),
            nn.GELU(),
        )
        self.encoder_tcn48 = TCNResidualStack(
            48, (1, 2, 4), self.dropout_probability
        )

        self.token_norm = nn.LayerNorm(ENCODED_CHANNELS)
        self.attention_query = nn.Parameter(
            torch.empty(1, 1, ENCODED_CHANNELS)
        )
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=ENCODED_CHANNELS,
            num_heads=ATTENTION_HEADS,
            dropout=self.dropout_probability,
            batch_first=True,
        )
        self.pool_norm = nn.LayerNorm(ENCODED_CHANNELS)
        self.to_bottleneck = nn.Sequential(
            nn.Linear(ENCODED_CHANNELS, BOTTLENECK_DIM),
            nn.LayerNorm(BOTTLENECK_DIM),
            nn.Tanh(),
        )
        nn.init.trunc_normal_(self.attention_query, std=0.02)
        self.register_buffer(
            "encoder_position_code",
            sinusoidal_position_code(ENCODED_SAMPLES, ENCODED_CHANNELS),
            persistent=True,
        )

        # The decoder receives only global z plus fixed time coordinates.
        self.latent_dropout = nn.Dropout(self.dropout_probability)
        self.latent_projection = nn.Linear(BOTTLENECK_DIM, 32)
        frequencies = torch.tensor(
            [0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00],
            dtype=torch.float32,
        )
        time = torch.arange(WINDOW_SAMPLES, dtype=torch.float32) / 64.0
        phase = 2.0 * torch.pi * frequencies[:, None] * time[None, :]
        time_code = torch.cat((torch.sin(phase), torch.cos(phase)), dim=0)
        self.register_buffer("time_code", time_code.unsqueeze(0), persistent=True)
        self.decoder_input = nn.Sequential(
            nn.Conv1d(48, 48, kernel_size=1, bias=False),
            nn.GroupNorm(group_count(48), 48),
            nn.GELU(),
        )
        self.decoder_tcn48 = TCNResidualStack(
            48, (1, 2, 4, 8, 16), self.dropout_probability
        )
        self.decoder_to32 = nn.Sequential(
            nn.Conv1d(48, 32, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(group_count(32), 32),
            nn.GELU(),
        )
        self.output_head = nn.Conv1d(32, 9, kernel_size=1, bias=True)

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (CHANNELS, WINDOW_SAMPLES):
            raise ValueError(f"expected [B,9,128], got {tuple(x.shape)}")
        encoded = self.encoder_tcn32(self.encoder_down32(self.encoder_stem(x)))
        encoded = self.encoder_tcn48(self.encoder_down48(encoded))
        if tuple(encoded.shape[1:]) != (ENCODED_CHANNELS, ENCODED_SAMPLES):
            raise RuntimeError(f"unexpected encoder shape: {tuple(encoded.shape)}")
        return encoded.transpose(1, 2).contiguous()

    def _encode_and_pool(
        self,
        x: torch.Tensor,
        *,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tokens = self.token_norm(
            self.encode_tokens(x) + self.encoder_position_code
        )
        query = self.attention_query.expand(tokens.shape[0], -1, -1)
        pooled, attention = self.attention_pool(
            query,
            tokens,
            tokens,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        pooled = self.pool_norm(pooled[:, 0, :])
        z = self.to_bottleneck(pooled)
        if tuple(z.shape[1:]) != (BOTTLENECK_DIM,):
            raise RuntimeError(f"unexpected bottleneck shape: {tuple(z.shape)}")
        if attention is not None and tuple(attention.shape[1:]) != (
            1,
            ENCODED_SAMPLES,
        ):
            raise RuntimeError(
                f"unexpected attention shape: {tuple(attention.shape)}"
            )
        return z, attention

    def encode_with_attention(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z, attention = self._encode_and_pool(x, need_weights=True)
        if attention is None:
            raise RuntimeError("attention weights were requested but not returned")
        return z, attention

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self._encode_and_pool(x, need_weights=False)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != BOTTLENECK_DIM:
            raise ValueError(f"expected [B,16], got {tuple(z.shape)}")
        latent = self.latent_projection(self.latent_dropout(z))
        latent = latent.unsqueeze(-1).expand(-1, -1, WINDOW_SAMPLES)
        time_code = self.time_code.expand(z.shape[0], -1, -1)
        decoded = self.decoder_input(torch.cat((latent, time_code), dim=1))
        decoded = self.decoder_tcn48(decoded)
        return self.output_head(self.decoder_to32(decoded))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction = self.decode(self.encode(x))
        if reconstruction.shape != x.shape:
            raise RuntimeError(
                f"reconstruction {tuple(reconstruction.shape)} != input {tuple(x.shape)}"
            )
        return reconstruction

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": ARCHITECTURE_NAME,
            "input_shape": ["B", 9, 128],
            "encoder": [
                "Conv1d(9,24,k=7,s=1,p=3)+GroupNorm+GELU",
                "Conv1d(24,32,k=5,s=2,p=2)+GroupNorm+GELU",
                "TCNResidualStack(32,d=1,2,4,8)",
                "Conv1d(32,48,k=5,s=2,p=2)+GroupNorm+GELU",
                "TCNResidualStack(48,d=1,2,4)",
            ],
            "encoder_token_shape": ["B", 32, 48],
            "attention_pool": {
                "type": "single learned query cross-attends to encoder tokens",
                "query_shape": [1, 1, 48],
                "heads": ATTENTION_HEADS,
                "embed_dim": ENCODED_CHANNELS,
                "token_position_code": (
                    "fixed sinusoidal [1,32,48], added before LayerNorm and "
                    "used by both key and value"
                ),
                "position_code_trainable": False,
                "output_shape": ["B", 48],
                "raw_or_encoder_token_residual_bypass": False,
            },
            "bottleneck_projection": "Linear(48,16)+LayerNorm(16)+Tanh",
            "bottleneck_shape": ["B", 16],
            "latent_regularization": (
                f"Dropout(p={self.dropout_probability}) before decoder during fit"
            ),
            "decoder_conditioning": {
                "latent": "Linear(16,32), broadcast to [B,32,128]",
                "time_code": "fixed Fourier [B,16,128]",
                "frequencies_hz": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
                "time_code_trainable": False,
                "raw_or_encoder_temporal_connection": False,
            },
            "decoder": [
                "concat global latent and fixed Fourier code [B,48,128]",
                "Conv1d(48,48,k=1)+GroupNorm+GELU",
                "TCNResidualStack(48,d=1,2,4,8,16)",
                "Conv1d(48,32,k=5,p=2)+GroupNorm+GELU",
                "Conv1d(32,9,k=1), no output activation",
            ],
            "residual_block": (
                "two same-length k=3 convolutions, GroupNorm, GELU, dropout"
            ),
            "causal": False,
            "encoder_decoder_skip_connections": False,
            "input_output_global_residual": False,
            "teacher_forcing": False,
            "output_activation": None,
            "output_shape": ["B", 9, 128],
            "dropout": self.dropout_probability,
            "parameter_count": sum(
                parameter.numel() for parameter in self.parameters()
            ),
        }


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    return device


@torch.no_grad()
def reconstruct_attention_pool_nbm(
    model: ResTCNSingleQueryAttentionPoolNBM,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for (batch,) in nbm_loader(x, False, 0, 0):
        prediction = model(batch.to(device, non_blocking=True))
        outputs.append(prediction.cpu().numpy().astype(np.float32))
    if not outputs:
        raise ValueError("cannot reconstruct an empty window collection")
    return np.concatenate(outputs, axis=0)


def train_attention_pool_nbm(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    output_dir: Path,
    device: torch.device,
    seed: int,
    num_workers: int,
    max_epochs: int,
    patience: int,
    dropout: float,
    augmentation: GRUMatchedAugmentation,
) -> tuple[ResTCNSingleQueryAttentionPoolNBM, dict[str, Any]]:
    augmentation.validate()
    set_seed(seed)
    model = ResTCNSingleQueryAttentionPoolNBM(dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    criterion = nn.SmoothL1Loss(beta=1.0)
    train_loader = nbm_loader(train_x, True, seed, num_workers)
    validation_loader = nbm_loader(validation_x, False, seed, num_workers)
    augmentation_generator = torch.Generator(device=device).manual_seed(seed + 1000)
    checkpoint = output_dir / "checkpoints" / CHECKPOINT_NAME
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        counts = {"clean_windows": 0, "gaussian_windows": 0, "masked_windows": 0}
        for (clean,) in train_loader:
            clean = clean.to(device, non_blocking=True)
            corrupted, batch_counts = gru_matched_augmentation(
                clean, augmentation, augmentation_generator
            )
            for key in counts:
                counts[key] += batch_counts[key]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(corrupted), clean)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite attention-pool NBM gradient")
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
        if total_n == 0 or validation_n == 0:
            raise ValueError("role 4/5 NBM windows must both be non-empty")
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
                **counts,
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
                    "augmentation": asdict(augmentation),
                },
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"ResTCN-AttnPool NBM epoch={epoch:03d} train={train_loss:.7f} "
            f"val={validation_loss:.7f} lr={learning_rate:.2e} "
            f"clean={counts['clean_windows']} gauss={counts['gaussian_windows']} "
            f"mask={counts['masked_windows']} stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break

    if not checkpoint.exists():
        raise RuntimeError("attention-pool NBM never produced a finite checkpoint")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    history_path = output_dir / "logs" / "restcn_attention_pool_nbm_history.csv"
    write_csv(history_path, history)
    summary = {
        "model_id": "restcn_attention_pool_nbm_best",
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


def calibrate_attention_pool_nbm(
    model: ResTCNSingleQueryAttentionPoolNBM,
    role5_x: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reconstruction = reconstruct_attention_pool_nbm(model, role5_x, device)
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
        "floor_applied_channels": np.flatnonzero(sigma_raw < 0.05)
        .astype(int)
        .tolist(),
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
    ax.axvline(
        run["summary"]["best_epoch"], color="black", linestyle="--", linewidth=1
    )
    ax.set(
        xlabel="Epoch",
        ylabel="SmoothL1 loss",
        title="ResTCN single-query attention-pooling NBM",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure_bundle(fig, output_dir / "restcn_attention_pool_nbm_training_validation")
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
            "daphnet_restcn_attnpool_z16_nbm300_C_vs_raw_tcn_ep10pat2_"
            "seedset_0_52_161"
        )
        / "nbm_source"
        / "seed_0",
    )
    parser.add_argument("--fold", type=int, choices=FOLDS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--required-seeds", default="0,52,161")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nbm-max-epochs", type=int, default=300)
    parser.add_argument("--nbm-patience", type=int, default=20)
    parser.add_argument("--nbm-dropout", type=float, default=DROPOUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace, device: torch.device) -> None:
    required_seeds = parse_csv_ints(args.required_seeds)
    if required_seeds != (0, 52, 161):
        raise ValueError("this experiment requires exact seeds 0,52,161")
    if args.seed not in required_seeds:
        raise ValueError(f"seed must be one of {required_seeds}")
    if args.nbm_max_epochs != 300 or args.nbm_patience != 20:
        raise ValueError("this experiment requires NBM max300/pat20")
    if args.nbm_dropout != DROPOUT:
        raise ValueError(f"this experiment freezes NBM dropout at {DROPOUT}")

    fold_dir = args.output_root.resolve() / f"fold_{args.fold}"
    done_path = fold_dir / "DONE_NBM.json"
    if done_path.exists() and not args.overwrite:
        print(f"SKIP completed attention-pool NBM fold/seed: {done_path}", flush=True)
        return
    fold_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir.resolve()
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
    role4_x = centered_scaled_bct(scaler, raw_windows(records, role4))
    role5_x = centered_scaled_bct(scaler, raw_windows(records, role5))

    probe = ResTCNSingleQueryAttentionPoolNBM(dropout=args.nbm_dropout).eval()
    architecture = probe.architecture_config()
    with torch.no_grad():
        probe_x = torch.zeros(2, 9, 128)
        probe_tokens = probe.encode_tokens(probe_x)
        probe_z, probe_attention = probe.encode_with_attention(probe_x)
        probe_y = probe(probe_x)
    if (
        probe_tokens.shape != (2, 32, 48)
        or probe_z.shape != (2, 16)
        or probe_attention.shape != (2, 1, 32)
        or probe_y.shape != probe_x.shape
    ):
        raise AssertionError("attention-pool NBM shape preflight failed")
    if not torch.allclose(
        probe_attention.sum(dim=-1),
        torch.ones(2, 1),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise AssertionError("attention weights do not sum to one in eval mode")
    del probe, probe_x, probe_tokens, probe_z, probe_attention, probe_y

    augmentation = GRUMatchedAugmentation()
    augmentation.validate()
    config = {
        "experiment": "ResTCN_single_query_attention_pool_Z16_NBM300_schemeC_source",
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
            "augmentation": asdict(augmentation),
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

    model, run_payload = train_attention_pool_nbm(
        role4_x,
        role5_x,
        fold_dir,
        device,
        args.seed,
        args.num_workers,
        args.nbm_max_epochs,
        args.nbm_patience,
        args.nbm_dropout,
        augmentation,
    )
    _bias, _sigma, calibration = calibrate_attention_pool_nbm(
        model, role5_x, device
    )
    plot_training(fold_dir, run_payload)
    summary = run_payload["summary"]
    training = {
        **summary,
        "maximum_epochs": 300,
        "patience": 20,
        "loss": "SmoothL1(beta=1.0)",
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "scheduler": "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)",
        "augmentation": asdict(augmentation),
        "architecture": architecture,
    }
    checkpoint = fold_dir / "checkpoints" / CHECKPOINT_NAME
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
        "architecture_name": ARCHITECTURE_NAME,
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
