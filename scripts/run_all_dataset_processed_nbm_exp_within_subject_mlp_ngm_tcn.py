#!/usr/bin/env python
"""Within-subject MLP-NGM + scheme-C TCN on processed_NBM_Exp.

The strict role split, five paired seeds, Group-C residual representation,
classifier, validation-only threshold selection, event metrics, and global
test barrier are inherited unchanged from the validated Conv-TCN-NGM worker.
Only the normal-gait autoencoder backbone is replaced.
"""

from __future__ import annotations

from scripts import (
    run_all_dataset_processed_nbm_exp_within_subject_conv_tcn_ngm_tcn as worker,
)
from scripts.mlp_ngm_30x128 import (
    MLP_NGM_30_PARAMETER_COUNT,
    FactorizedMLPNGM30,
    architecture_config as generic_architecture_config,
)


NBM_VARIANT = "FACTORIZED_MLP_NGM_MASK4_8"
NBM_PARAMETER_COUNT = MLP_NGM_30_PARAMETER_COUNT
NBM_CHECKPOINT_NAME = "mlp_ngm_best.pt"
EXPERIMENT_SCHEMA = "all_dataset_within_subject_mlp_ngm_tcn.v1"
BARRIER_SCHEMA = "all_dataset_within_subject_mlp_ngm_tcn_barrier.v1"
MODEL_DESCRIPTION = "Factorized MLP NGM Mask4-8 + scheme-C 90-channel TCN"
NBM_DISPLAY_NAME = "MLP-NGM"

SUBJECTS = worker.SUBJECTS
FOLDS = worker.FOLDS
SEEDS = worker.SEEDS


def architecture_config():
    return generic_architecture_config(30)


def configure_worker() -> None:
    worker.__doc__ = __doc__
    worker.ConvTCNNGM30 = FactorizedMLPNGM30
    worker.NBM_VARIANT = NBM_VARIANT
    worker.NBM_PARAMETER_COUNT = NBM_PARAMETER_COUNT
    worker.NBM_CHECKPOINT_NAME = NBM_CHECKPOINT_NAME
    worker.EXPERIMENT_SCHEMA = EXPERIMENT_SCHEMA
    worker.BARRIER_SCHEMA = BARRIER_SCHEMA
    worker.MODEL_DESCRIPTION = MODEL_DESCRIPTION
    worker.NBM_DISPLAY_NAME = NBM_DISPLAY_NAME
    worker.architecture_config = architecture_config
    worker.configure_base()


def main() -> None:
    configure_worker()
    worker.base.main()


if __name__ == "__main__":
    main()
