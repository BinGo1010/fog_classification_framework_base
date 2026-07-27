#!/usr/bin/env python
"""Run the RF125 TCN-M/CNN replacement suite across physical GPUs.

One LOSO fold is the indivisible work unit.  Each GPU worker trains ``tcn_m``
and ``cnn_rf125`` sequentially for that fold, so the paired comparison shares
one device and one fold-local training protocol.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import start_daphnet_residual_classifier_suite_multigpu as scheduler_base

from cnbr_fog.rf125_classifiers import CANONICAL_RF125_CLASSIFIER_NAMES


RUNNER = REPO_ROOT / "scripts" / "run_daphnet_rf125_cnn_replacement.py"
AUDITOR = REPO_ROOT / "scripts" / "audit_daphnet_rf125_cnn_replacement.py"
SCHEDULER_VERSION = "daphnet_rf125_cnn_replacement_multigpu.v1"
LOCK_FILENAME = ".rf125_cnn_replacement_scheduler.lock"


class OutputDirectoryLock(scheduler_base.OutputDirectoryLock):
    """Use an experiment-specific lock without changing lock semantics."""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.path = output_dir / LOCK_FILENAME


@contextmanager
def configured_scheduler() -> Iterator[Any]:
    replacements = {
        "RUNNER": RUNNER,
        "AUDITOR": AUDITOR,
        "CANONICAL_CLASSIFIERS": CANONICAL_RF125_CLASSIFIER_NAMES,
        "SCHEDULER_VERSION": SCHEDULER_VERSION,
        "OutputDirectoryLock": OutputDirectoryLock,
    }
    saved = {
        name: getattr(scheduler_base, name)
        for name in replacements
    }
    try:
        for name, value in replacements.items():
            setattr(scheduler_base, name, value)
        yield scheduler_base
    finally:
        for name, value in saved.items():
            setattr(scheduler_base, name, value)


def main() -> None:
    with configured_scheduler() as configured:
        configured.main()


if __name__ == "__main__":
    main()
