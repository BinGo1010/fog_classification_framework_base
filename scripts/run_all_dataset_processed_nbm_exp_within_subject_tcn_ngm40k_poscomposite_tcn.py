#!/usr/bin/env python
"""Within-subject position-conditioned TCN-NBM + scheme-C TCN on NBM_Exp.

The role allocation, robust scaling, residual calibration, Group-C residual
representation, downstream TCN classifier, validation threshold selection, and
global test barrier are inherited unchanged from the proven strict worker.
Only the NBM changes: a temporal Conv-TCN autoencoder with fixed position
conditioning and a composite reconstruction objective.
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
from scripts import (
    run_all_dataset_processed_nbm_exp_within_subject_conv_tcn_ngm_tcn as worker,
)
from scripts.tcn_ngm_40k_position_composite import (
    TCN_NGM_POSITION_COMPOSITE_30_PARAMETER_COUNT,
    PositionConditionedTCNNGM30,
    architecture_config as position_architecture_config,
)


NBM_VARIANT = "POSITION_CONDITIONED_TCN_NGM40K_COMPOSITE_MASK4_8"
NBM_PARAMETER_COUNT = TCN_NGM_POSITION_COMPOSITE_30_PARAMETER_COUNT
NBM_CHECKPOINT_NAME = "tcn_ngm40k_poscomposite_best.pt"
EXPERIMENT_SCHEMA = "all_dataset_within_subject_tcn_ngm40k_poscomposite_tcn.v1"
BARRIER_SCHEMA = "all_dataset_within_subject_tcn_ngm40k_poscomposite_tcn_barrier.v1"
MODEL_DESCRIPTION = "Position-conditioned 40k Conv-TCN NBM composite loss + scheme-C 90-channel TCN"
NBM_DISPLAY_NAME = "Position-conditioned TCN-NBM40K"
NBM_DEFAULT_MAX_EPOCHS = 400
NBM_DEFAULT_PATIENCE = 40

SMOOTHL1_WEIGHT = 0.70
CORRELATION_WEIGHT = 0.15
FIRST_DIFFERENCE_WEIGHT = 0.15
SMOOTHL1_BETA = 1.0

SUBJECTS = worker.SUBJECTS
FOLDS = worker.FOLDS
SEEDS = worker.SEEDS


def architecture_config() -> dict[str, Any]:
    return position_architecture_config()


def composite_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Smooth-L1 + temporal correlation + first-difference reconstruction loss.

    ``prediction`` and ``target`` are [B,C,T] tensors in the centered
    robust-scaled domain.  Correlation is calculated independently for every
    window and axis over time, then averaged; it cannot be minimized by a flat
    trajectory when the target has temporal variation.
    """

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            f"prediction/target must share [B,C,T], got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    huber = F.smooth_l1_loss(prediction, target, beta=SMOOTHL1_BETA)
    pred_centered = prediction - prediction.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=-1)
    denominator = torch.sqrt(
        pred_centered.square().sum(dim=-1)
        * target_centered.square().sum(dim=-1)
        + 1e-8
    )
    correlation = (numerator / denominator).clamp(min=-1.0, max=1.0)
    correlation_loss = 1.0 - correlation.mean()
    difference_loss = F.smooth_l1_loss(
        torch.diff(prediction, dim=-1), torch.diff(target, dim=-1), beta=SMOOTHL1_BETA
    )
    total = (
        SMOOTHL1_WEIGHT * huber
        + CORRELATION_WEIGHT * correlation_loss
        + FIRST_DIFFERENCE_WEIGHT * difference_loss
    )
    return total, {
        "huber": huber,
        "correlation": correlation_loss,
        "first_difference": difference_loss,
    }


def train_position_composite_nbm(
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
    """Train NBM on role 4 and choose the minimum clean role-5 composite loss."""

    worker.base.set_seed(seed)
    model = PositionConditionedTCNNGM30().to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != NBM_PARAMETER_COUNT:
        raise RuntimeError(f"{NBM_DISPLAY_NAME} parameter contract changed")
    initial_state = worker.base.state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    train_batches = worker.base.nbm_loader(train_x, batch_size, True, seed, workers)
    validation_batches = worker.base.nbm_loader(
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
        train_sums = {"total": 0.0, "huber": 0.0, "correlation": 0.0, "first_difference": 0.0}
        train_count = 0
        mode_counts = np.zeros(3, dtype=np.int64)
        for (clean_ntc,) in train_batches:
            clean_ntc = clean_ntc.to(device, non_blocking=True)
            corrupted_ntc, counts = worker.base.corrupt_gru_base(
                clean_ntc, augmentation_generator
            )
            mode_counts += counts
            clean_bct = clean_ntc.transpose(1, 2)
            corrupted_bct = corrupted_ntc.transpose(1, 2)
            optimizer.zero_grad(set_to_none=True)
            loss, components = composite_reconstruction_loss(
                model(corrupted_bct), clean_bct
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {NBM_DISPLAY_NAME} loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite {NBM_DISPLAY_NAME} gradient")
            optimizer.step()
            count = len(clean_ntc)
            train_sums["total"] += float(loss.detach()) * count
            for name, value in components.items():
                train_sums[name] += float(value.detach()) * count
            train_count += count

        model.eval()
        validation_sums = {"total": 0.0, "huber": 0.0, "correlation": 0.0, "first_difference": 0.0}
        validation_count = 0
        with torch.no_grad():
            for (clean_ntc,) in validation_batches:
                clean_bct = clean_ntc.to(device, non_blocking=True).transpose(1, 2)
                loss, components = composite_reconstruction_loss(model(clean_bct), clean_bct)
                count = len(clean_bct)
                validation_sums["total"] += float(loss) * count
                for name, value in components.items():
                    validation_sums[name] += float(value) * count
                validation_count += count

        train_metrics = {key: value / train_count for key, value in train_sums.items()}
        validation_metrics = {
            key: value / validation_count for key, value in validation_sums.items()
        }
        validation_loss = validation_metrics["total"]
        scheduler.step(validation_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        improved = validation_loss < best_loss - 1e-10
        history.append(
            {
                "epoch": epoch,
                "train_composite": train_metrics["total"],
                "train_huber": train_metrics["huber"],
                "train_correlation": train_metrics["correlation"],
                "train_first_difference": train_metrics["first_difference"],
                "validation_composite": validation_metrics["total"],
                "validation_huber": validation_metrics["huber"],
                "validation_correlation": validation_metrics["correlation"],
                "validation_first_difference": validation_metrics["first_difference"],
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
                    "validation_composite": validation_loss,
                    "validation_huber": validation_metrics["huber"],
                    "validation_correlation": validation_metrics["correlation"],
                    "validation_first_difference": validation_metrics["first_difference"],
                    "initial_model_state_sha256": initial_state,
                    "architecture": architecture_config(),
                    "augmentation": worker.augmentation_config(),
                },
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"{NBM_DISPLAY_NAME} epoch={epoch:03d} "
            f"train={train_metrics['total']:.7f} "
            f"val={validation_loss:.7f} "
            f"(h={validation_metrics['huber']:.5f}, "
            f"corr={validation_metrics['correlation']:.5f}, "
            f"d1={validation_metrics['first_difference']:.5f}) "
            f"lr={learning_rate:.2e} stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("variant") != NBM_VARIANT:
        raise AssertionError(f"{NBM_DISPLAY_NAME} checkpoint variant mismatch")
    if payload.get("architecture") != architecture_config():
        raise AssertionError(f"{NBM_DISPLAY_NAME} checkpoint architecture mismatch")
    model.load_state_dict(payload["model_state"])
    return model, {
        "maximum_epochs": maximum_epochs,
        "patience": patience,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_composite": best_loss,
        "best_validation_huber": float(payload["validation_huber"]),
        "best_validation_correlation": float(payload["validation_correlation"]),
        "best_validation_first_difference": float(payload["validation_first_difference"]),
        "initial_model_state_sha256": initial_state,
        "parameter_count": NBM_PARAMETER_COUNT,
        "history": history,
    }


def training_contract(args: Any) -> dict[str, Any]:
    """Persist every changed NBM choice in the frozen experiment contract."""

    contract = worker._BASE_TRAINING_CONTRACT(args)
    contract.update(
        {
            "nbm": architecture_config(),
            "augmentation": worker.augmentation_config(),
            "nbm_loss": (
                "0.70*SmoothL1(beta=1.0) + 0.15*(1-temporal Pearson correlation) "
                "+ 0.15*SmoothL1(first difference, beta=1.0); corrupted input predicts clean target"
            ),
            "nbm_loss_weights": {
                "smooth_l1": SMOOTHL1_WEIGHT,
                "temporal_correlation": CORRELATION_WEIGHT,
                "first_difference": FIRST_DIFFERENCE_WEIGHT,
            },
            "nbm_optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)",
            "nbm_scheduler": "ReduceLROnPlateau(factor=0.5,patience=5,min_lr=1e-5)",
            "nbm_checkpoint": "minimum clean role5 composite reconstruction loss",
            "event_metric": {
                "version": worker.EVENT_METRIC_VERSION,
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


def configure_worker() -> None:
    """Inject the changed NBM only; retain every downstream strict-protocol hook."""

    worker.__doc__ = __doc__
    worker.ConvTCNNGM30 = PositionConditionedTCNNGM30
    worker.NBM_VARIANT = NBM_VARIANT
    worker.NBM_PARAMETER_COUNT = NBM_PARAMETER_COUNT
    worker.NBM_CHECKPOINT_NAME = NBM_CHECKPOINT_NAME
    worker.EXPERIMENT_SCHEMA = EXPERIMENT_SCHEMA
    worker.BARRIER_SCHEMA = BARRIER_SCHEMA
    worker.MODEL_DESCRIPTION = MODEL_DESCRIPTION
    worker.NBM_DISPLAY_NAME = NBM_DISPLAY_NAME
    worker.NBM_DEFAULT_MAX_EPOCHS = NBM_DEFAULT_MAX_EPOCHS
    worker.NBM_DEFAULT_PATIENCE = NBM_DEFAULT_PATIENCE
    worker.architecture_config = architecture_config
    worker.train_conv_tcn_ngm = train_position_composite_nbm
    worker.training_contract = training_contract
    worker.configure_base()


def main() -> None:
    configure_worker()
    worker.base.main()


if __name__ == "__main__":
    main()
