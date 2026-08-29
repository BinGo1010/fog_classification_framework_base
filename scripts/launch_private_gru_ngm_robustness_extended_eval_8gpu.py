#!/usr/bin/env python3
"""Run the extended frozen-pipeline robustness evaluation on eight GPUs."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import evaluate_private_gru_ngm_robustness_extended as worker
from scripts import launch_private_gru_ngm_robustness_eval_8gpu as launcher


launcher.worker = worker
launcher.WORKER = Path(worker.__file__).resolve()
launcher.AGGREGATE_DIR_NAME = "robustness_extended"
launcher.POOL_STAGE = "robustness_eval_extended"


if __name__ == "__main__":
    launcher.main()
