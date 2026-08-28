#!/usr/bin/env python3
"""Train GRU-BASE NGM from scratch, then classify [r,delta(r)] on NBM_Exp.

For every P01--P08 subject/fold/seed job, role 4 fits the RobustScaler and
trains a fresh GRU BASE Mask4--8 normal-gait model; clean role 5 selects the
NBM checkpoint and calibrates sigma; roles 6/7 train a 60-channel TCN on
[r,delta(r)]; roles 2/3 select the TCN checkpoint and decision threshold.
Roles 0/1 remain inaccessible until all 120 training jobs are globally sealed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import (
    run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_r_delta_tcn
    as residual,
)
from scripts import (
    run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_r_only_tcn
    as residual_base,
)
from scripts import (
    run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn as base,
)
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import (
    RepresentationTCNM,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import set_seed


EXPERIMENT_SCHEMA = "all_dataset_within_subject_gru_nbm_r_delta_scratch.v1"
BARRIER_SCHEMA = "all_dataset_within_subject_gru_nbm_r_delta_scratch_barrier.v1"
MODEL_DESCRIPTION = (
    "fresh GRU BASE Mask4-8 NBM + [r,delta(r)] 60-channel TCN"
)
TCN_INPUT_CHANNELS = 60
TCN_PARAMETER_COUNT = 139_809
REFERENCE_TCN_CHANNELS = 90
EVENT_METRIC_VERSION = "allocation_group_any_window_nonfog_runs.v1"
EVENT_MINIMUM_POSITIVE_WINDOWS = 1
EVENT_MERGE_GAP_SECONDS = 1.0
EVENT_FALSE_ALARM_DENOMINATOR = (
    "role-0 false-alarm runs divided by union coverage of evaluated valid "
    "Non-FoG samples"
)
EVENT_AGGREGATION = "subject_macro"
AGGREGATION_DESCRIPTION = (
    "per subject/seed macro mean of 3 folds; per subject mean+population SD "
    "over 5 seeds; overall subject-macro within each seed, then "
    "mean+population SD over 5 seeds"
)


def train_r_delta_tcn(
    train_x: Any,
    train_y: Any,
    validation_x: Any,
    validation_y: Any,
    destination: Any,
    device: torch.device,
    seed: int,
    batch_size: int,
    workers: int,
    maximum_epochs: int,
    patience: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Initialize 60 channels as the exact [r,delta] subset of a 90ch TCN."""

    set_seed(seed)
    reference = RepresentationTCNM(REFERENCE_TCN_CHANNELS)
    expected_reference_hash = base.state_dict_sha256(reference.state_dict())
    return residual_base.train_tcn(
        train_x,
        train_y,
        validation_x,
        validation_y,
        destination,
        device,
        seed,
        batch_size,
        workers,
        maximum_epochs,
        patience,
        expected_reference_hash,
    )


def training_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "scaler": "per-axis median/IQR fitted on unique role4 raw samples",
        "nbm_preprocessing": (
            "RobustScaler then per-window/per-axis time centering"
        ),
        "nbm": base.architecture_config(),
        "augmentation": base.augmentation_config(),
        "nbm_loss": (
            "SmoothL1(beta=1.0), corrupted role4 input predicts clean target"
        ),
        "nbm_optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)",
        "nbm_scheduler": (
            "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-5)"
        ),
        "nbm_maximum_epochs": args.nbm_max_epochs,
        "nbm_patience": args.nbm_patience,
        "nbm_checkpoint": "minimum clean role5 SmoothL1",
        "calibration": (
            "after restoring best NBM, role5 b=median(e), "
            "sigma=max(1.4826*MAD(e-b),0.05); scheme does not subtract b"
        ),
        "residual": (
            "e=X-Xhat; q=clip(e/(sigma+1e-6),-12,12); "
            "r=q-mean_t(q); delta(r)[0]=0"
        ),
        "tcn_input": "concatenate [r,delta(r)]; abs(r) is absent",
        "tcn_input_shape": ["B", TCN_INPUT_CHANNELS, 128],
        "tcn": (
            "RepresentationTCNM 60->32->64->64->128; "
            "dilations1/2/4/8; GAP; one logit"
        ),
        "tcn_parameter_count": TCN_PARAMETER_COUNT,
        "paired_initialization": (
            "same tensors as a seed-matched 90-channel TCN; first-layer "
            "channels select 0:30 for r and 60:90 for delta(r)"
        ),
        "classifier_train_roles": [6, 7],
        "classifier_validation_roles": [2, 3],
        "classifier_test_roles": [0, 1],
        "tcn_loss": "BCEWithLogitsLoss(pos_weight=N_role6/N_role7)",
        "tcn_optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)",
        "tcn_maximum_epochs": args.tcn_max_epochs,
        "tcn_patience": args.tcn_patience,
        "tcn_checkpoint": "maximum roles2/3 AP",
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
                "role0 true Non-FoG only; within-record positive decision "
                "start gaps <=1s merge across allocation groups"
            ),
            "exposure": "union coverage of valid evaluated Non-FoG samples",
        },
    }


def configure_base() -> None:
    """Install the r+delta representation into the proven strict full worker."""

    residual.configure_base()
    residual_base.EXPERIMENT_SCHEMA = EXPERIMENT_SCHEMA
    residual_base.TCN_CHECKPOINT_NAME = "tcn.pt"

    base.EXPERIMENT_SCHEMA = EXPERIMENT_SCHEMA
    base.BARRIER_SCHEMA = BARRIER_SCHEMA
    base.MODEL_DESCRIPTION = MODEL_DESCRIPTION
    base.TCN_INPUT_CHANNELS = TCN_INPUT_CHANNELS
    base.TCN_PARAMETER_COUNT = TCN_PARAMETER_COUNT
    base.EVENT_MINIMUM_POSITIVE_WINDOWS = EVENT_MINIMUM_POSITIVE_WINDOWS
    base.EVENT_MERGE_GAP_SECONDS = EVENT_MERGE_GAP_SECONDS
    base.EVENT_FALSE_ALARM_DENOMINATOR = EVENT_FALSE_ALARM_DENOMINATOR
    base.EVENT_AGGREGATION = EVENT_AGGREGATION
    base.AGGREGATION_DESCRIPTION = AGGREGATION_DESCRIPTION
    base.scheme_c_features = residual.r_delta_features
    base.train_tcn = train_r_delta_tcn
    base.training_contract = training_contract
    base.raw_base.event_metrics = residual.final_event_metrics
    base.raw_base.EVENT_METRIC_VERSION = EVENT_METRIC_VERSION


def main() -> None:
    configure_base()
    base.main()


if __name__ == "__main__":
    main()
