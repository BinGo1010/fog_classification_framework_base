#!/usr/bin/env python3
"""Run strict 64-Hz Daphnet RAW+TCN256/pat5 on seven GPUs.

This is the epoch-budget variant of the proven RAW-only launcher.  It trains
15 classifiers (3 folds x 5 paired seeds), seals every validation-selected
checkpoint and threshold, and only then evaluates permanent roles 0/1.  It
does not train or infer an NBM; completed role-4 scaler artifacts are reused.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import launch_daphnet_64hz_raw_tcn_ep20pat5_5seed_7gpu as base


TCN_MAX_EPOCHS = 256
TCN_PATIENCE = 5
DEFAULT_EXPERIMENT = (
    "daphnet_64Hz_raw_tcn_ep256pat5_seedset_0_52_161_5216_52161"
)


def main() -> None:
    base.main(
        tcn_max_epochs=TCN_MAX_EPOCHS,
        tcn_patience=TCN_PATIENCE,
        default_experiment=DEFAULT_EXPERIMENT,
    )


if __name__ == "__main__":
    main()
