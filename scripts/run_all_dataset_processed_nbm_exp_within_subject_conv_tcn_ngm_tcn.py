#!/usr/bin/env python
"""Within-subject Conv-TCN-NGM + scheme-C TCN on processed_NBM_Exp.

For each P01-P08 subject/fold/seed job, role 4 fits the RobustScaler and the
30-channel Conv-TCN normal-gait autoencoder.  Clean role 5 selects the NGM
checkpoint and calibrates residual scale.  Roles 6/7 train the unchanged TCN
classifier, roles 2/3 select its checkpoint and threshold, and roles 0/1 stay
locked until all 120 training jobs have been globally sealed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.resume import atomic_torch_save
from scripts import run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn as base
from scripts.run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_r_delta_tcn import (
    EVENT_AGGREGATION,
    EVENT_FALSE_ALARM_DENOMINATOR,
    EVENT_MERGE_GAP_SECONDS,
    EVENT_METRIC_VERSION,
    EVENT_MINIMUM_POSITIVE_WINDOWS,
    final_event_metrics,
)
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    TCNResidualStack,
    group_count,
)


_BASE_AUGMENTATION_CONFIG = base.augmentation_config
_BASE_TRAINING_CONTRACT = base.training_contract


SUBJECTS = base.SUBJECTS
FOLDS = base.FOLDS
SEEDS = base.SEEDS
ROLES = base.ROLES
WINDOW_SAMPLES = base.WINDOW_SAMPLES
SAMPLING_RATE_HZ = base.SAMPLING_RATE_HZ
RAW_CHANNELS = base.RAW_CHANNELS
TCN_INPUT_CHANNELS = base.TCN_INPUT_CHANNELS
TCN_PARAMETER_COUNT = base.TCN_PARAMETER_COUNT
METRIC_KEYS = base.METRIC_KEYS

NBM_VARIANT = "CONV_TCN_NGM_MASK4_8"
NBM_PARAMETER_COUNT = 52_510
NBM_CHECKPOINT_NAME = "conv_tcn_ngm_best.pt"
NBM_DEFAULT_MAX_EPOCHS = 300
NBM_DEFAULT_PATIENCE = 20
EXPERIMENT_SCHEMA = "all_dataset_within_subject_conv_tcn_ngm_tcn.v1"
BARRIER_SCHEMA = "all_dataset_within_subject_conv_tcn_ngm_tcn_barrier.v1"
MODEL_DESCRIPTION = "Conv-TCN NGM Mask4-8 + scheme-C 90-channel TCN"
AGGREGATION_DESCRIPTION = (
    "window metrics: subject/seed macro mean of 3 folds, then subject-macro per "
    "seed and mean+population SD over 5 seeds; event sensitivity: detected "
    "allocation groups / all allocation groups; FA/h: total role-0 false-alarm "
    "runs / total valid Non-FoG union exposure, each pooled within fold, then "
    "3-fold mean per seed and 5-seed mean+population SD"
)


class ConvTCNNGM30(nn.Module):
    """30x128 -> 16x32 -> 30x128 skip-free Conv-TCN autoencoder."""

    def __init__(self, dropout: float = 0.10) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.encoder_stem = nn.Sequential(
            nn.Conv1d(
                RAW_CHANNELS, 32, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.GroupNorm(group_count(32), 32),
            nn.GELU(),
        )
        self.encoder_tcn32 = TCNResidualStack(32, (1, 2), dropout)
        self.encoder_down24 = nn.Sequential(
            nn.Conv1d(32, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(group_count(24), 24),
            nn.GELU(),
        )
        self.encoder_tcn24 = TCNResidualStack(24, (1, 2), dropout)
        self.to_bottleneck = nn.Sequential(
            nn.Conv1d(24, 16, kernel_size=1, bias=False),
            nn.GroupNorm(group_count(16), 16),
            nn.GELU(),
        )
        self.decoder_from_bottleneck = nn.Sequential(
            nn.Conv1d(16, 24, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_count(24), 24),
            nn.GELU(),
        )
        self.decoder_to32 = nn.Sequential(
            nn.Conv1d(24, 32, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(group_count(32), 32),
            nn.GELU(),
        )
        self.decoder_tcn32 = TCNResidualStack(32, (1, 2), dropout)
        self.decoder_to16 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(group_count(16), 16),
            nn.GELU(),
        )
        self.output_head = nn.Conv1d(16, RAW_CHANNELS, kernel_size=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_tcn32(self.encoder_stem(x))
        x = self.encoder_tcn24(self.encoder_down24(x))
        return self.to_bottleneck(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_from_bottleneck(z)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.decoder_tcn32(self.decoder_to32(x))
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        return self.output_head(self.decoder_to16(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = (RAW_CHANNELS, WINDOW_SAMPLES)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"expected [B,{RAW_CHANNELS},{WINDOW_SAMPLES}], got {tuple(x.shape)}")
        z = self.encode(x)
        if tuple(z.shape[1:]) != (16, 32):
            raise RuntimeError(f"unexpected bottleneck shape: {tuple(z.shape)}")
        reconstruction = self.decode(z)
        if reconstruction.shape != x.shape:
            raise RuntimeError(
                f"reconstruction shape {tuple(reconstruction.shape)} != {tuple(x.shape)}"
            )
        return reconstruction


def architecture_config() -> dict[str, Any]:
    model = ConvTCNNGM30()
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != NBM_PARAMETER_COUNT:
        raise RuntimeError(f"30-channel Conv-TCN NGM parameter contract changed: {count}")
    return {
        "name": "conv_tcn_autoencoder_ngm_v1_30channel",
        "input_shape": ["B", RAW_CHANNELS, WINDOW_SAMPLES],
        "encoder": [
            "Conv1d(30,32,k=7,s=2,p=3)+GroupNorm+GELU",
            "TCNResidualBlock(32,d=1) then TCNResidualBlock(32,d=2)",
            "Conv1d(32,24,k=5,s=2,p=2)+GroupNorm+GELU",
            "TCNResidualBlock(24,d=1) then TCNResidualBlock(24,d=2)",
            "Conv1d(24,16,k=1)+GroupNorm+GELU",
        ],
        "bottleneck_shape": ["B", 16, 32],
        "decoder": [
            "Conv1d(16,24,k=3,p=1)+GroupNorm+GELU",
            "linear interpolation x2",
            "Conv1d(24,32,k=5,p=2)+GroupNorm+GELU",
            "TCNResidualBlock(32,d=1) then TCNResidualBlock(32,d=2)",
            "linear interpolation x2",
            "Conv1d(32,16,k=7,p=3)+GroupNorm+GELU",
            "Conv1d(16,30,k=1), no output activation",
        ],
        "residual_block": (
            "two same-length k=3 convolutions, GroupNorm, GELU, dropout"
        ),
        "causal": False,
        "encoder_decoder_skip_connections": False,
        "teacher_forcing": False,
        "dropout": 0.10,
        "output_shape": ["B", RAW_CHANNELS, WINDOW_SAMPLES],
        "parameter_count": NBM_PARAMETER_COUNT,
    }


def augmentation_config() -> dict[str, Any]:
    return _BASE_AUGMENTATION_CONFIG()


@torch.no_grad()
def reconstruct_conv_tcn(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3 or tuple(values.shape[1:]) != (
        WINDOW_SAMPLES,
        RAW_CHANNELS,
    ):
        raise ValueError(
            f"expected [N,{WINDOW_SAMPLES},{RAW_CHANNELS}], got {values.shape}"
        )
    model.eval()
    outputs: list[np.ndarray] = []
    for (batch,) in base.nbm_loader(values, batch_size, False, 0, 0):
        batch_bct = batch.to(device, non_blocking=True).transpose(1, 2)
        prediction = model(batch_bct).transpose(1, 2)
        outputs.append(prediction.cpu().numpy().astype(np.float32))
    return np.ascontiguousarray(np.concatenate(outputs, axis=0))


def train_conv_tcn_ngm(
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
    base.set_seed(seed)
    model = ConvTCNNGM30().to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != NBM_PARAMETER_COUNT:
        raise RuntimeError("Conv-TCN NGM parameter contract changed")
    initial_state = base.state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    criterion = nn.SmoothL1Loss(beta=1.0)
    train_batches = base.nbm_loader(train_x, batch_size, True, seed, workers)
    validation_batches = base.nbm_loader(
        validation_x, batch_size, False, seed, workers
    )
    augmentation_generator = torch.Generator(device=device).manual_seed(seed + 1000)
    checkpoint = destination / "checkpoints" / NBM_CHECKPOINT_NAME
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        mode_counts = np.zeros(3, dtype=np.int64)
        for (clean_ntc,) in train_batches:
            clean_ntc = clean_ntc.to(device, non_blocking=True)
            network_input_ntc, counts = base.corrupt_gru_base(
                clean_ntc, augmentation_generator
            )
            mode_counts += counts
            clean_bct = clean_ntc.transpose(1, 2)
            network_input_bct = network_input_ntc.transpose(1, 2)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(network_input_bct), clean_bct)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite Conv-TCN NGM gradient")
            optimizer.step()
            train_total += float(loss.detach()) * len(clean_ntc)
            train_count += len(clean_ntc)

        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.no_grad():
            for (clean_ntc,) in validation_batches:
                clean_bct = clean_ntc.to(device, non_blocking=True).transpose(1, 2)
                loss = criterion(model(clean_bct), clean_bct)
                validation_total += float(loss) * len(clean_bct)
                validation_count += len(clean_bct)
        train_loss = train_total / train_count
        validation_loss = validation_total / validation_count
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
                    "schema": EXPERIMENT_SCHEMA,
                    "variant": NBM_VARIANT,
                    "model_state": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "validation_huber": validation_loss,
                    "initial_model_state_sha256": initial_state,
                    "architecture": architecture_config(),
                    "augmentation": augmentation_config(),
                },
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"Conv-TCN-NGM epoch={epoch:03d} train={train_loss:.7f} "
            f"val={validation_loss:.7f} lr={learning_rate:.2e} "
            f"stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("variant") != NBM_VARIANT:
        raise AssertionError("Conv-TCN NGM checkpoint variant mismatch")
    if payload.get("architecture") != architecture_config():
        raise AssertionError("Conv-TCN NGM checkpoint architecture mismatch")
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


def build_conv_tcn_from_checkpoint(
    payload: dict[str, Any], device: torch.device
) -> nn.Module:
    if payload.get("variant") != NBM_VARIANT:
        raise AssertionError("Conv-TCN NGM checkpoint variant mismatch")
    if payload.get("architecture") != architecture_config():
        raise AssertionError("Conv-TCN NGM checkpoint architecture mismatch")
    model = ConvTCNNGM30().to(device)
    model.load_state_dict(payload["model_state"])
    return model


def training_contract(args: Any) -> dict[str, Any]:
    contract = _BASE_TRAINING_CONTRACT(args)
    contract.update(
        {
            "nbm": architecture_config(),
            "augmentation": augmentation_config(),
            "nbm_loss": (
                "SmoothL1(beta=1.0), corrupted input predicts clean target"
            ),
            "nbm_optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)",
            "nbm_scheduler": (
                "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)"
            ),
            "event_metric": {
                "version": EVENT_METRIC_VERSION,
                "reference_event": "one permanent-test FoG allocation group",
                "detected": "at least one group window predicted FoG",
                "false_alarm": (
                    "role-0 Non-FoG predictions only; same-record positives "
                    "with gap <=1 second are merged"
                ),
                "exposure": "union coverage of evaluated valid Non-FoG samples",
            },
        }
    )
    return contract


def configure_base() -> None:
    """Inject the Conv-TCN NGM into the strict proven within-subject worker."""

    base.__doc__ = __doc__
    base.NBM_VARIANT = NBM_VARIANT
    base.NBM_PARAMETER_COUNT = NBM_PARAMETER_COUNT
    base.NBM_CHECKPOINT_NAME = NBM_CHECKPOINT_NAME
    base.NBM_DEFAULT_MAX_EPOCHS = NBM_DEFAULT_MAX_EPOCHS
    base.NBM_DEFAULT_PATIENCE = NBM_DEFAULT_PATIENCE
    base.EXPERIMENT_SCHEMA = EXPERIMENT_SCHEMA
    base.BARRIER_SCHEMA = BARRIER_SCHEMA
    base.MODEL_DESCRIPTION = MODEL_DESCRIPTION
    base.EVENT_MINIMUM_POSITIVE_WINDOWS = EVENT_MINIMUM_POSITIVE_WINDOWS
    base.EVENT_MERGE_GAP_SECONDS = EVENT_MERGE_GAP_SECONDS
    base.EVENT_FALSE_ALARM_DENOMINATOR = EVENT_FALSE_ALARM_DENOMINATOR
    base.EVENT_AGGREGATION = EVENT_AGGREGATION
    base.AGGREGATION_DESCRIPTION = AGGREGATION_DESCRIPTION
    base.architecture_config = architecture_config
    base.augmentation_config = augmentation_config
    base.train_nbm = train_conv_tcn_ngm
    base.reconstruct = reconstruct_conv_tcn
    base.build_nbm_from_checkpoint = build_conv_tcn_from_checkpoint
    base.training_contract = training_contract
    base.raw_base.event_metrics = final_event_metrics
    base.raw_base.EVENT_METRIC_VERSION = EVENT_METRIC_VERSION


def main() -> None:
    configure_base()
    base.main()


if __name__ == "__main__":
    main()
