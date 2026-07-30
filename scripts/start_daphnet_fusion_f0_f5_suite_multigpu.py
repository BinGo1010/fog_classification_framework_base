#!/usr/bin/env python
"""Schedule the Transformer-NBM F0--F5 fusion suite on independent GPUs."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import start_daphnet_3imu_nbm_suite_multigpu as scheduler_base


RUNNER_FILENAME = "run_daphnet_fusion_f0_f5_suite.py"
AUDITOR_FILENAME = "audit_daphnet_fusion_f0_f5_suite.py"
SCHEDULER_VERSION = "daphnet_transformer_fusion_f0_f5_multigpu.v1"
DEFAULT_OUTPUT_DIRNAME = (
    "daphnet_transformer_fusion_f0_f5_h4_tcnm_loso_seed42"
)
SCHEDULER_DESCRIPTION = (
    "Run the Daphnet Transformer-NBM F0--F5 four-second TCN-M LOSO "
    "suite on independent GPUs. One GPU completes all six fusion inputs "
    "for its assigned fold before taking another fold."
)


@contextmanager
def configured_scheduler() -> Iterator[Any]:
    replacements = {
        "RUNNER_FILENAME": RUNNER_FILENAME,
        "AUDITOR_FILENAME": AUDITOR_FILENAME,
        "SCHEDULER_VERSION": SCHEDULER_VERSION,
        "DEFAULT_OUTPUT_DIRNAME": DEFAULT_OUTPUT_DIRNAME,
        "SCHEDULER_DESCRIPTION": SCHEDULER_DESCRIPTION,
        "RESERVED_FORWARDED_OPTIONS": (
            scheduler_base.RESERVED_FORWARDED_OPTIONS
            | {
                "--cache-only",
                "--smoke",
                "--debug-interrupt-classifier-after-epoch",
                "--stop-after-completed-tasks",
            }
        ),
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


def main(argv: Sequence[str] | None = None) -> int:
    with configured_scheduler() as configured:
        return configured.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
