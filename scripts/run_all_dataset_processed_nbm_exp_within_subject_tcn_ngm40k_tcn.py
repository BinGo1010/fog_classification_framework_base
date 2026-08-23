#!/usr/bin/env python
"""Within-subject 40k TCN-NGM + scheme-C TCN on processed_NBM_Exp."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import (
    run_all_dataset_processed_nbm_exp_within_subject_conv_tcn_ngm_tcn as worker,
)
from scripts.tcn_ngm_40k import (
    TCN_NGM_30_PARAMETER_COUNT,
    CapacityMatchedTCNNGM30,
    architecture_config as generic_architecture_config,
)


NBM_VARIANT = "CAPACITY_MATCHED_TCN_NGM40K_MASK4_8"
NBM_PARAMETER_COUNT = TCN_NGM_30_PARAMETER_COUNT
NBM_CHECKPOINT_NAME = "tcn_ngm40k_best.pt"
EXPERIMENT_SCHEMA = "all_dataset_within_subject_tcn_ngm40k_tcn.v1"
BARRIER_SCHEMA = "all_dataset_within_subject_tcn_ngm40k_tcn_barrier.v1"
MODEL_DESCRIPTION = "40k TCN NGM Mask4-8 + scheme-C 90-channel TCN"
NBM_DISPLAY_NAME = "TCN-NGM40K"

SUBJECTS = worker.SUBJECTS
FOLDS = worker.FOLDS
SEEDS = worker.SEEDS


def architecture_config():
    return generic_architecture_config(30)


def configure_worker() -> None:
    worker.__doc__ = __doc__
    worker.ConvTCNNGM30 = CapacityMatchedTCNNGM30
    worker.NBM_VARIANT = NBM_VARIANT
    worker.NBM_PARAMETER_COUNT = NBM_PARAMETER_COUNT
    worker.NBM_CHECKPOINT_NAME = NBM_CHECKPOINT_NAME
    worker.EXPERIMENT_SCHEMA = EXPERIMENT_SCHEMA
    worker.BARRIER_SCHEMA = BARRIER_SCHEMA
    worker.MODEL_DESCRIPTION = MODEL_DESCRIPTION
    worker.NBM_DISPLAY_NAME = NBM_DISPLAY_NAME
    worker.SUBJECTS = SUBJECTS
    worker.base.SUBJECTS = SUBJECTS
    worker.architecture_config = architecture_config
    worker.configure_base()


def main() -> None:
    configure_worker()
    worker.base.main()


if __name__ == "__main__":
    main()
