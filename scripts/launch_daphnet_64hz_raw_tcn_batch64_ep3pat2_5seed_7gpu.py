#!/usr/bin/env python3
"""Run strict 64-Hz Daphnet RAW+TCN, batch64, max3/pat2, on 7 GPUs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import launch_daphnet_64hz_raw_tcn_ep20pat5_5seed_7gpu as base


TCN_MAX_EPOCHS = 3
TCN_PATIENCE = 2
CLASSIFIER_BATCH_SIZE = 64
DEFAULT_EXPERIMENT = (
    "daphnet_64Hz_raw_tcn_batch64_ep3pat2_"
    "seedset_0_52_161_5216_52161"
)


def main() -> None:
    base.main(
        tcn_max_epochs=TCN_MAX_EPOCHS,
        tcn_patience=TCN_PATIENCE,
        classifier_batch_size=CLASSIFIER_BATCH_SIZE,
        default_experiment=DEFAULT_EXPERIMENT,
    )


if __name__ == "__main__":
    main()
