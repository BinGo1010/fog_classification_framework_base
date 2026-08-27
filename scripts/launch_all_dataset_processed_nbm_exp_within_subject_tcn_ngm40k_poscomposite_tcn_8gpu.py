#!/usr/bin/env python
"""Launch position-conditioned composite TCN-NBM + TCN jobs on 7 or 8 GPUs."""

from __future__ import annotations

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
    "nbm300pat20_ep5pat2_seedset_0_52_161_5216_52161"
)


def configure_launcher() -> None:
    launcher.__doc__ = __doc__
    launcher.WORKER = WORKER
    launcher.DEFAULT_EXPERIMENT = DEFAULT_EXPERIMENT
    launcher.EXPERIMENT_SCHEMA = EXPERIMENT_SCHEMA
    launcher.BARRIER_SCHEMA = BARRIER_SCHEMA
    launcher.NBM_PARAMETER_COUNT = NBM_PARAMETER_COUNT
    launcher.NBM_VARIANT = NBM_VARIANT
    launcher.architecture_config = architecture_config
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
