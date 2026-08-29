#!/usr/bin/env python3
"""Evaluate one frozen Private GRU-NGM/TCN on the extended corruption grid."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import evaluate_private_gru_ngm_robustness as _base


GAUSSIAN_SIGMAS = (
    0.0,
    0.02,
    0.04,
    0.08,
    0.12,
    0.20,
    0.30,
    0.40,
    0.60,
    0.80,
    1.00,
)
MASK_RHOS = (
    0.0,
    0.025,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.60,
    0.80,
    1.00,
)
EXPERIMENT_SCHEMA = "private_gru_ngm_robustness_extended_evaluation.v1"
RESULT_DIR_NAME = "robustness_test_extended"


# The validated evaluation implementation is shared with the original grid.  Only
# the immutable evaluation contract and destination directory differ here.
_base.GAUSSIAN_SIGMAS = GAUSSIAN_SIGMAS
_base.MASK_RHOS = MASK_RHOS
_base.EXPERIMENT_SCHEMA = EXPERIMENT_SCHEMA
_base.RESULT_DIR_NAME = RESULT_DIR_NAME

EVALUATION_BATCH_SIZE = _base.EVALUATION_BATCH_SIZE
METRICS_NAME = _base.METRICS_NAME
completed_evaluation_is_valid = _base.completed_evaluation_is_valid
evaluation_contract_id = _base.evaluation_contract_id
write_csv = _base.write_csv


if __name__ == "__main__":
    _base.main()
