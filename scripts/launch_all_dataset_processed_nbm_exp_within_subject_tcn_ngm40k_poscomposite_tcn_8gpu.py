#!/usr/bin/env python
"""Launch position-conditioned composite TCN-NBM + TCN jobs on 7 or 8 GPUs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import (
    launch_all_dataset_processed_nbm_exp_within_subject_conv_tcn_ngm_tcn_7gpu as launcher,
)
from scripts.run_all_dataset_processed_nbm_exp_within_subject_tcn_ngm40k_poscomposite_tcn import (
    BARRIER_SCHEMA,
    EXPERIMENT_SCHEMA,
    NBM_PARAMETER_COUNT,
    NBM_VARIANT,
    architecture_config,
)


WORKER = (
    REPO_ROOT
    / "scripts"
    / "run_all_dataset_processed_nbm_exp_within_subject_tcn_ngm40k_poscomposite_tcn.py"
)
DEFAULT_EXPERIMENT = (
    "all_dataset_processed_NBM_Exp_within_subject_tcn_ngm40k_poscomposite_C_tcn_"
    "nbm400pat40_ep30pat8_seedset_0_52_161_5216_52161"
)

NBM_MAX_EPOCHS = 400
NBM_PATIENCE = 40
TCN_MAX_EPOCHS = 30
TCN_PATIENCE = 8


def validate_contract(args: argparse.Namespace) -> tuple[tuple[int, ...], list[str]]:
    """Freeze the expanded training budget for this new NBM variant."""

    seeds = launcher.parse_seed_list(args.seeds)
    if seeds != launcher.SEEDS:
        raise ValueError(f"the five seeds are frozen to {launcher.SEEDS}; received {seeds}")
    if args.nbm_max_epochs != NBM_MAX_EPOCHS or args.nbm_patience != NBM_PATIENCE:
        raise ValueError(
            f"this NBM requires max_epoch={NBM_MAX_EPOCHS} and patience={NBM_PATIENCE}"
        )
    if args.tcn_max_epochs != TCN_MAX_EPOCHS or args.tcn_patience != TCN_PATIENCE:
        raise ValueError(
            f"this TCN requires max_epoch={TCN_MAX_EPOCHS} and patience={TCN_PATIENCE}"
        )
    if args.batch_size != 128:
        raise ValueError("this experiment requires batch_size=128")
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if (
        len(gpu_ids) not in (7, 8)
        or len(set(gpu_ids)) != len(gpu_ids)
        or any(not value.isdigit() for value in gpu_ids)
    ):
        raise ValueError("--gpu-ids must contain seven or eight unique GPU ids")
    return seeds, gpu_ids


def configure_launcher() -> None:
    launcher.__doc__ = __doc__
    launcher.WORKER = WORKER
    launcher.DEFAULT_EXPERIMENT = DEFAULT_EXPERIMENT
    launcher.EXPERIMENT_SCHEMA = EXPERIMENT_SCHEMA
    launcher.BARRIER_SCHEMA = BARRIER_SCHEMA
    launcher.NBM_PARAMETER_COUNT = NBM_PARAMETER_COUNT
    launcher.NBM_VARIANT = NBM_VARIANT
    launcher.architecture_config = architecture_config
    launcher.validate_contract = validate_contract
    launcher.CRITICAL_CODE = (
        WORKER,
        Path(__file__).resolve(),
        REPO_ROOT / "scripts" / "tcn_ngm_40k_position_composite.py",
        REPO_ROOT / "scripts" / "run_all_dataset_processed_nbm_exp_within_subject_conv_tcn_ngm_tcn.py",
        REPO_ROOT / "scripts" / "run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn.py",
        REPO_ROOT / "scripts" / "run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_r_delta_tcn.py",
        REPO_ROOT / "scripts" / "run_all_dataset_processed_nbm_exp_within_subject_raw_tcn.py",
        REPO_ROOT / "scripts" / "run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py",
        REPO_ROOT / "cnbr_fog" / "data.py",
        REPO_ROOT / "cnbr_fog" / "evaluation.py",
        REPO_ROOT / "cnbr_fog" / "resume.py",
        REPO_ROOT / "cnbr_fog" / "scientific_fingerprint.py",
    )


def main() -> None:
    configure_launcher()
    launcher.main()


if __name__ == "__main__":
    main()
