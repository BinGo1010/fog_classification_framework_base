#!/usr/bin/env python3
"""Train one exact-seed 40k TCN-NGM fold on Daphnet processed_NBM."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_daphnet_mlp_ngm300_fold as worker
from scripts.tcn_ngm_40k import (
    TCN_NGM_9_PARAMETER_COUNT,
    CapacityMatchedTCNNGM9,
    architecture_config as generic_architecture_config,
    reconstruct_bct,
)


NBM_VARIANT = "CAPACITY_MATCHED_TCN_NGM40K_MASK4_8"
CHECKPOINT_NAME = "tcn_ngm40k_best.pt"
EXPERIMENT_SCHEMA = "daphnet_tcn_ngm40k_source.v1"
NBM_DISPLAY_NAME = "TCN-NGM40K"
EXPERIMENT_NAME = "TCN_NGM40K_NBM300_schemeC_source"
TRAINING_FIGURE_STEM = "tcn_ngm40k_training_validation"


def architecture():
    return generic_architecture_config(9)


def configure_worker() -> None:
    worker.__doc__ = __doc__
    worker.NBM_VARIANT = NBM_VARIANT
    worker.CHECKPOINT_NAME = CHECKPOINT_NAME
    worker.EXPERIMENT_SCHEMA = EXPERIMENT_SCHEMA
    worker.NBM_MODEL_CLASS = CapacityMatchedTCNNGM9
    worker.NBM_PARAMETER_COUNT = TCN_NGM_9_PARAMETER_COUNT
    worker.NBM_DISPLAY_NAME = NBM_DISPLAY_NAME
    worker.EXPERIMENT_NAME = EXPERIMENT_NAME
    worker.TRAINING_FIGURE_STEM = TRAINING_FIGURE_STEM
    worker.architecture = architecture
    worker.reconstruct_bct = reconstruct_bct


def main() -> None:
    configure_worker()
    worker.main()


if __name__ == "__main__":
    main()
