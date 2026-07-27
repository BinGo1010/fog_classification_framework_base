#!/usr/bin/env python
"""Schedule the Daphnet NBM-context TCN-M suite across independent GPUs.

This is a thin configuration wrapper around the established 3-IMU multi-GPU
scheduler.  One LOSO fold remains the indivisible unit of work, while the core
runner owns all NBM/context combinations for that fold.
"""

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


RUNNER_FILENAME = "run_daphnet_nbm_context_tcnm_suite.py"
AUDITOR_FILENAME = "audit_daphnet_nbm_context_tcnm_suite.py"
SCHEDULER_VERSION = "daphnet_nbm_context_tcnm_multigpu.v1"
DEFAULT_OUTPUT_DIRNAME = (
    "daphnet_nbm4_context4_h4_tcnm_loso_seed42"
)
SCHEDULER_DESCRIPTION = (
    "Run the Daphnet NBM/context-length residual_h4s TCN-M LOSO suite on "
    "independent GPUs. One GPU completes all NBM and context-length "
    "configurations for its assigned fold before taking another fold."
)


@contextmanager
def configured_scheduler() -> Iterator[Any]:
    """Temporarily specialize the shared scheduler for this experiment."""
    replacements = {
        "RUNNER_FILENAME": RUNNER_FILENAME,
        "AUDITOR_FILENAME": AUDITOR_FILENAME,
        "SCHEDULER_VERSION": SCHEDULER_VERSION,
        "DEFAULT_OUTPUT_DIRNAME": DEFAULT_OUTPUT_DIRNAME,
        "SCHEDULER_DESCRIPTION": SCHEDULER_DESCRIPTION,
        # A development-only subset run must never be forwarded through the
        # production multi-GPU scheduler, whose unit of work is a complete
        # strict LOSO fold.
        "RESERVED_FORWARDED_OPTIONS": (
            scheduler_base.RESERVED_FORWARDED_OPTIONS
            | {"--allow-protocol-subset"}
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
