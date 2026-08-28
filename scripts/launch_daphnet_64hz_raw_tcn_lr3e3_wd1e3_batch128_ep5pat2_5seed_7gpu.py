#!/usr/bin/env python3
"""Run Daphnet RAW+TCN with lr3e-3, wd1e-3, batch128, max5/pat2."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import launch_daphnet_64hz_raw_tcn_ep20pat5_5seed_7gpu as base


TCN_MAX_EPOCHS = 5
TCN_PATIENCE = 2
CLASSIFIER_BATCH_SIZE = 128
TCN_LEARNING_RATE = 3e-3
TCN_WEIGHT_DECAY = 1e-3
DEFAULT_EXPERIMENT = (
    "daphnet_64Hz_raw_tcn_lr3e-3_wd1e-3_batch128_ep5pat2_"
    "seedset_0_52_161_5216_52161"
)


def main() -> None:
    base.main(
        tcn_max_epochs=TCN_MAX_EPOCHS,
        tcn_patience=TCN_PATIENCE,
        classifier_batch_size=CLASSIFIER_BATCH_SIZE,
        tcn_learning_rate=TCN_LEARNING_RATE,
        tcn_weight_decay=TCN_WEIGHT_DECAY,
        default_experiment=DEFAULT_EXPERIMENT,
    )


if __name__ == "__main__":
    main()
