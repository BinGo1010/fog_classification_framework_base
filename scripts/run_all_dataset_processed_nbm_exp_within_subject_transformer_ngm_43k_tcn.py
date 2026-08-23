#!/usr/bin/env python
"""Within-subject 43k Transformer-NGM + scheme-C TCN on processed_NBM_Exp.

The normal-gait model is a skip-free patch Transformer with a single global
16-dimensional bottleneck.  Role 4 fits the RobustScaler and Transformer-NGM;
clean role 5 selects the lowest reconstruction-loss checkpoint and calibrates
the residual MAD scale.  Roles 6/7 train the classifier, roles 2/3 select its
checkpoint and decision threshold, and roles 0/1 remain locked until every
subject/fold/seed job has been globally sealed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import set_seed


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

PATCH_SIZE = 8
TOKEN_COUNT = WINDOW_SAMPLES // PATCH_SIZE
PATCH_DIM = RAW_CHANNELS * PATCH_SIZE
MODEL_DIM = 32
HEADS = 4
FFN_DIM = 64
ENCODER_LAYERS = 2
DECODER_LAYERS = 1
BOTTLENECK_DIM = 16

NBM_VARIANT = "TRANSFORMER_NGM_43K_GLOBAL_Z16"
NBM_PARAMETER_COUNT = 43_360
NBM_CHECKPOINT_NAME = "transformer_ngm_43k_best.pt"
EXPERIMENT_SCHEMA = "all_dataset_within_subject_transformer_ngm_43k_tcn.v1"
BARRIER_SCHEMA = "all_dataset_within_subject_transformer_ngm_43k_tcn_barrier.v1"
MODEL_DESCRIPTION = "43k Transformer-NGM global-Z16 + scheme-C 90-channel TCN"
AGGREGATION_DESCRIPTION = (
    "window metrics: subject/seed macro mean of 3 folds, then subject-macro per "
    "seed and mean+population SD over 5 seeds; event sensitivity: detected "
    "allocation groups / all allocation groups; FA/h: total role-0 false-alarm "
    "runs / total valid Non-FoG union exposure, each pooled within fold, then "
    "3-fold mean per seed and 5-seed mean+population SD"
)


class TinyPatchTransformerNGM(nn.Module):
    """Skip-free 30-channel patch Transformer with global Z=[B,16]."""

    def __init__(self, dropout: float = 0.10) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.patch_projection = nn.Linear(PATCH_DIM, MODEL_DIM)
        self.encoder_position = nn.Parameter(
            torch.zeros(1, TOKEN_COUNT, MODEL_DIM)
        )
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
        self.to_bottleneck = nn.Linear(MODEL_DIM, BOTTLENECK_DIM)
        self.from_bottleneck = nn.Linear(BOTTLENECK_DIM, MODEL_DIM)
        self.decoder_position = nn.Parameter(
            torch.zeros(1, TOKEN_COUNT, MODEL_DIM)
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM,
            nhead=HEADS,
            dim_feedforward=FFN_DIM,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        # Decoder blocks receive only the broadcast global Z plus position.
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=DECODER_LAYERS,
            enable_nested_tensor=False,
        )
        self.patch_output = nn.Linear(MODEL_DIM, PATCH_DIM)
        nn.init.trunc_normal_(self.encoder_position, std=0.02)
        nn.init.trunc_normal_(self.decoder_position, std=0.02)
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count != NBM_PARAMETER_COUNT:
            raise RuntimeError(
                f"Transformer-NGM parameter contract changed: {parameter_count}"
            )

    @staticmethod
    def patchify(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (
            WINDOW_SAMPLES,
            RAW_CHANNELS,
        ):
            raise ValueError(
                f"expected [B,{WINDOW_SAMPLES},{RAW_CHANNELS}], got {tuple(x.shape)}"
            )
        return x.reshape(
            x.shape[0], TOKEN_COUNT, PATCH_SIZE, RAW_CHANNELS
        ).reshape(x.shape[0], TOKEN_COUNT, PATCH_DIM)

    @staticmethod
    def fold_patches(patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3 or tuple(patches.shape[1:]) != (
            TOKEN_COUNT,
            PATCH_DIM,
        ):
            raise ValueError(
                f"expected [B,{TOKEN_COUNT},{PATCH_DIM}], got {tuple(patches.shape)}"
            )
        return patches.reshape(
            patches.shape[0], TOKEN_COUNT, PATCH_SIZE, RAW_CHANNELS
        ).reshape(patches.shape[0], WINDOW_SAMPLES, RAW_CHANNELS)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_projection(self.patchify(x))
        encoded = self.encoder(tokens + self.encoder_position)
        # Parameter-free mean pooling creates one global information bottleneck.
        z = self.to_bottleneck(encoded.mean(dim=1))
        if tuple(z.shape[1:]) != (BOTTLENECK_DIM,):
            raise RuntimeError(f"unexpected bottleneck shape: {tuple(z.shape)}")
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != BOTTLENECK_DIM:
            raise ValueError(f"expected [B,{BOTTLENECK_DIM}], got {tuple(z.shape)}")
        global_token = self.from_bottleneck(z).unsqueeze(1)
        tokens = global_token.expand(-1, TOKEN_COUNT, -1)
        decoded = self.decoder(tokens + self.decoder_position)
        return self.fold_patches(self.patch_output(decoded))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction = self.decode(self.encode(x))
        if reconstruction.shape != x.shape:
            raise RuntimeError(
                f"reconstruction {tuple(reconstruction.shape)} != input {tuple(x.shape)}"
            )
        return reconstruction


def architecture_config() -> dict[str, Any]:
    model = TinyPatchTransformerNGM(dropout=0.10)
    return {
        "name": "tiny_patch_transformer_ngm_global_z16_v1",
        "input_shape": ["B", WINDOW_SAMPLES, RAW_CHANNELS],
        "patchify": {
            "patch_size": PATCH_SIZE,
            "token_count": TOKEN_COUNT,
            "patch_dim": PATCH_DIM,
            "token_shape": ["B", TOKEN_COUNT, PATCH_DIM],
        },
        "patch_projection": f"Linear({PATCH_DIM},{MODEL_DIM})",
        "encoder_position": [1, TOKEN_COUNT, MODEL_DIM],
        "encoder": {
            "layers": ENCODER_LAYERS,
            "d_model": MODEL_DIM,
            "heads": HEADS,
            "ffn": FFN_DIM,
            "activation": "GELU",
            "dropout": 0.10,
            "normalization": "post-norm",
        },
        "token_pooling": "parameter-free temporal mean",
        "bottleneck": f"Linear({MODEL_DIM},{BOTTLENECK_DIM})",
        "bottleneck_shape": ["B", BOTTLENECK_DIM],
        "decoder_conditioning": (
            "Linear(16,32), broadcast to 16 tokens, add learned position"
        ),
        "decoder": {
            "layers": DECODER_LAYERS,
            "type": "self-attention only; no encoder-memory cross-attention",
            "d_model": MODEL_DIM,
            "heads": HEADS,
            "ffn": FFN_DIM,
            "activation": "GELU",
            "dropout": 0.10,
            "normalization": "post-norm",
        },
        "patch_output": f"Linear({MODEL_DIM},{PATCH_DIM}), exact fold",
        "encoder_decoder_skip_connections": False,
        "cross_attention": False,
        "teacher_forcing": False,
        "raw_input_bypass": False,
        "output_activation": None,
        "output_shape": ["B", WINDOW_SAMPLES, RAW_CHANNELS],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
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
        "training_target": "uncorrupted clean role-4 window",
    }


def train_transformer_ngm(
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
    if (maximum_epochs, patience) != (300, 20):
        raise ValueError("Transformer-NGM training is frozen to max300/pat20")
    set_seed(seed)
    model = TinyPatchTransformerNGM(dropout=0.10).to(device)
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
        for (clean,) in train_batches:
            clean = clean.to(device, non_blocking=True)
            network_input, counts = base.corrupt_gru_base(
                clean, augmentation_generator
            )
            mode_counts += counts
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(network_input), clean)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite Transformer-NGM gradient")
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
            f"Transformer-NGM epoch={epoch:03d} train={train_loss:.7f} "
            f"val={validation_loss:.7f} lr={learning_rate:.2e} "
            f"stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("architecture") != architecture_config():
        raise AssertionError("Transformer-NGM checkpoint architecture mismatch")
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


def build_transformer_from_checkpoint(
    payload: dict[str, Any], device: torch.device
) -> nn.Module:
    if payload.get("variant") != NBM_VARIANT:
        raise AssertionError("Transformer-NGM checkpoint variant mismatch")
    if payload.get("architecture") != architecture_config():
        raise AssertionError("Transformer-NGM checkpoint architecture mismatch")
    model = TinyPatchTransformerNGM(dropout=0.10).to(device)
    model.load_state_dict(payload["model_state"])
    return model


def training_contract(args: Any) -> dict[str, Any]:
    return {
        "scaler": "per-axis median/IQR fitted on unique role-4 raw samples",
        "ngm_preprocessing": "RobustScaler then per-window/per-axis time centering",
        "normal_gait_model": architecture_config(),
        "augmentation": augmentation_config(),
        "ngm_loss": "SmoothL1(beta=1.0), corrupted input predicts clean target",
        "ngm_optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)",
        "ngm_scheduler": "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)",
        "ngm_maximum_epochs": args.nbm_max_epochs,
        "ngm_patience": args.nbm_patience,
        "ngm_checkpoint": "minimum clean role5 SmoothL1",
        "calibration": (
            "after restoring best NGM, clean role5 b=median(e), "
            "sigma=max(1.4826*MAD(e-b),0.05)"
        ),
        "scheme_c": (
            "e=X-Xhat; q=clip(e/(sigma+1e-6),-12,12); "
            "r=q-mean_t(q); [r,abs(r),delta(r)]"
        ),
        "scheme_c_uses_bias_b": False,
        "tcn_input_shape": ["B", TCN_INPUT_CHANNELS, WINDOW_SAMPLES],
        "tcn": (
            "RepresentationTCNM 90->32->64->64->128; "
            "dilations1/2/4/8; GAP; one logit"
        ),
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
        "threshold": (
            "roles2/3 grid0.05..0.95 step0.01; max balanced accuracy; "
            "ties F1 then higher threshold"
        ),
        "event_metric": {
            "reference_event": "one permanent-test FoG allocation group",
            "detected": "any group window predicted FoG",
            "false_alarm": (
                "role-0 only; same-record positive decisions <=1 s apart merged"
            ),
            "exposure": "union coverage of evaluated valid Non-FoG samples",
        },
    }


def configure_base() -> None:
    """Inject the 43k Transformer-NGM into the strict shared worker."""

    base.__doc__ = __doc__
    base.NBM_VARIANT = NBM_VARIANT
    base.NBM_PARAMETER_COUNT = NBM_PARAMETER_COUNT
    base.NBM_CHECKPOINT_NAME = NBM_CHECKPOINT_NAME
    base.NBM_DEFAULT_MAX_EPOCHS = 300
    base.NBM_DEFAULT_PATIENCE = 20
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
    base.train_nbm = train_transformer_ngm
    base.build_nbm_from_checkpoint = build_transformer_from_checkpoint
    base.training_contract = training_contract
    base.raw_base.event_metrics = final_event_metrics
    base.raw_base.EVENT_METRIC_VERSION = EVENT_METRIC_VERSION


def main() -> None:
    configure_base()
    base.main()


if __name__ == "__main__":
    main()
