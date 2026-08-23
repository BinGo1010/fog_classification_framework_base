#!/usr/bin/env python
"""Within-subject Persistence-NGM + scheme-C TCN on processed_NBM_Exp.

Persistence-NGM is the recommended parameter-free lag-one normal-gait model:
the first sample is copied and each later sample is reconstructed from the
immediately preceding sample.  It has no optimizer, Mask augmentation, or
trainable parameters.  Role 4 fits only the RobustScaler; clean role 5
calibrates the residual MAD scale.  Roles 6/7 train the 90-channel scheme-C
TCN, roles 2/3 select its checkpoint and threshold, and roles 0/1 stay locked
until all subject/fold/seed jobs have been globally sealed.
"""

from __future__ import annotations

from pathlib import Path
import sys
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
NBM_VARIANT = "PERSISTENCE_NGM_LAG1"
NBM_PARAMETER_COUNT = 0
NBM_CHECKPOINT_NAME = "persistence_ngm_frozen.pt"
EXPERIMENT_SCHEMA = "all_dataset_within_subject_persistence_ngm_tcn.v1"
BARRIER_SCHEMA = "all_dataset_within_subject_persistence_ngm_tcn_barrier.v1"
MODEL_DESCRIPTION = "Persistence-NGM lag1 + scheme-C 90-channel TCN"
AGGREGATION_DESCRIPTION = (
    "window metrics: subject/seed macro mean of 3 folds, then subject-macro per "
    "seed and mean+population SD over 5 seeds; event sensitivity: detected "
    "allocation groups / all allocation groups; FA/h: total role-0 false-alarm "
    "runs / total valid Non-FoG union exposure, each pooled within fold, then "
    "3-fold mean per seed and 5-seed mean+population SD"
)


class PersistenceNGM(nn.Module):
    """Parameter-free causal lag-one reconstruction for 30-channel windows."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (WINDOW_SAMPLES, RAW_CHANNELS):
            raise ValueError(
                f"expected [B,{WINDOW_SAMPLES},{RAW_CHANNELS}], got {tuple(x.shape)}"
            )
        return torch.cat((x[:, :1, :], x[:, :-1, :]), dim=1)


def architecture_config() -> dict[str, Any]:
    model = PersistenceNGM()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 0:
        raise RuntimeError("Persistence-NGM must remain parameter-free")
    return {
        "name": "persistence_ngm_lag1_30channel",
        "input_shape": ["B", WINDOW_SAMPLES, RAW_CHANNELS],
        "formula": (
            "Xhat[:,0,:]=X[:,0,:]; "
            "Xhat[:,t,:]=X[:,t-1,:] for t=1..127"
        ),
        "effective_context_samples": 1,
        "causal": True,
        "trainable": False,
        "parameter_count": 0,
        "output_shape": ["B", WINDOW_SAMPLES, RAW_CHANNELS],
    }


def augmentation_config() -> dict[str, Any]:
    return {
        "applicable": False,
        "reason": "parameter-free deterministic NGM; no fitting step",
        "mask_augmentation": False,
        "gaussian_augmentation": False,
        "inference_input_corruption": False,
    }


def persistence_reconstruct(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    del model, device, batch_size
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3 or tuple(values.shape[1:]) != (
        WINDOW_SAMPLES,
        RAW_CHANNELS,
    ):
        raise ValueError(
            f"expected [N,{WINDOW_SAMPLES},{RAW_CHANNELS}], got {values.shape}"
        )
    output = np.empty_like(values)
    output[:, 0, :] = values[:, 0, :]
    output[:, 1:, :] = values[:, :-1, :]
    return np.ascontiguousarray(output)


def smooth_l1_mean(target: np.ndarray, prediction: np.ndarray) -> float:
    absolute = np.abs(np.asarray(target) - np.asarray(prediction))
    loss = np.where(absolute < 1.0, 0.5 * absolute**2, absolute - 0.5)
    return float(np.mean(loss, dtype=np.float64))


def freeze_persistence_ngm(
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
    del device, batch_size, workers
    if maximum_epochs != 0 or patience != 0:
        raise ValueError("Persistence-NGM requires maximum_epochs=0 and patience=0")
    if train_x.shape[1:] != (WINDOW_SAMPLES, RAW_CHANNELS):
        raise ValueError(f"unexpected role-4 shape: {train_x.shape}")
    if validation_x.shape[1:] != (WINDOW_SAMPLES, RAW_CHANNELS):
        raise ValueError(f"unexpected role-5 shape: {validation_x.shape}")
    model = PersistenceNGM()
    role4_diagnostic = smooth_l1_mean(
        train_x, persistence_reconstruct(model, train_x, torch.device("cpu"), 0)
    )
    role5_diagnostic = smooth_l1_mean(
        validation_x,
        persistence_reconstruct(model, validation_x, torch.device("cpu"), 0),
    )
    checkpoint = destination / "checkpoints" / NBM_CHECKPOINT_NAME
    atomic_torch_save(
        {
            "schema": EXPERIMENT_SCHEMA,
            "variant": NBM_VARIANT,
            "seed": int(seed),
            "model_state": model.state_dict(),
            "architecture": architecture_config(),
            "parameter_count": 0,
            "fit_performed": False,
            "role4_diagnostic_smooth_l1": role4_diagnostic,
            "role5_diagnostic_smooth_l1": role5_diagnostic,
        },
        checkpoint,
    )
    history = [
        {
            "epoch": 0,
            "train_huber_diagnostic": role4_diagnostic,
            "validation_huber_diagnostic": role5_diagnostic,
            "trainable_parameters": 0,
            "optimizer_steps": 0,
            "frozen": True,
        }
    ]
    return model, {
        "maximum_epochs": 0,
        "patience": 0,
        "epochs_completed": 0,
        "best_epoch": 0,
        "best_validation_huber": role5_diagnostic,
        "role4_diagnostic_huber": role4_diagnostic,
        "initial_model_state_sha256": base.state_dict_sha256(model.state_dict()),
        "parameter_count": 0,
        "fit_performed": False,
        "optimizer_steps": 0,
        "history": history,
    }


def build_persistence_from_checkpoint(
    payload: dict[str, Any], device: torch.device
) -> nn.Module:
    if payload.get("variant") != NBM_VARIANT:
        raise AssertionError("Persistence checkpoint variant mismatch")
    if payload.get("architecture") != architecture_config():
        raise AssertionError("Persistence checkpoint architecture mismatch")
    if payload.get("fit_performed") is not False:
        raise AssertionError("Persistence checkpoint unexpectedly reports fitting")
    model = PersistenceNGM().to(device)
    model.load_state_dict(payload.get("model_state", {}), strict=True)
    return model


def training_contract(args: Any) -> dict[str, Any]:
    return {
        "scaler": "per-axis median/IQR fitted on unique role-4 raw samples",
        "ngm_preprocessing": "RobustScaler then per-window/per-axis time centering",
        "normal_gait_model": architecture_config(),
        "ngm_fit": "none; deterministic parameter-free lag-one persistence",
        "ngm_augmentation": augmentation_config(),
        "ngm_optimizer": None,
        "ngm_loss": None,
        "ngm_maximum_epochs": 0,
        "ngm_patience": 0,
        "calibration": (
            "clean role5 b=median(e), sigma=max(1.4826*MAD(e-b),0.05); "
            "b only centers the MAD estimate"
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
    """Inject Persistence-NGM into the proven strict within-subject worker."""

    base.__doc__ = __doc__
    base.NBM_VARIANT = NBM_VARIANT
    base.NBM_PARAMETER_COUNT = NBM_PARAMETER_COUNT
    base.NBM_CHECKPOINT_NAME = NBM_CHECKPOINT_NAME
    base.NBM_DEFAULT_MAX_EPOCHS = 0
    base.NBM_DEFAULT_PATIENCE = 0
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
    base.train_nbm = freeze_persistence_ngm
    base.reconstruct = persistence_reconstruct
    base.build_nbm_from_checkpoint = build_persistence_from_checkpoint
    base.training_contract = training_contract
    base.raw_base.event_metrics = final_event_metrics
    base.raw_base.EVENT_METRIC_VERSION = EVENT_METRIC_VERSION


def main() -> None:
    configure_base()
    base.main()


if __name__ == "__main__":
    main()
