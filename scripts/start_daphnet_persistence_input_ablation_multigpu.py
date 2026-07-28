#!/usr/bin/env python
"""Run the Persistence input-representation ablation across physical GPUs.

One LOSO fold is the indivisible scheduling unit.  A fold worker owns its GPU
while it prepares one shared frozen representation cache and trains the four
TCN-M readouts sequentially:

* robust-scaled raw IMU on residual-matched support;
* Persistence error ``x - mu``;
* uncertainty-standardised error ``(x - mu) / sigma``;
* the standardised error clipped to ``[-12, 12]``.

With seven GPUs and the canonical eight folds, the first seven folds start
concurrently and the remaining fold is assigned to the first GPU that becomes
idle.  Scheduling, retry/resume, heartbeat status, logging, locking, CPU
initialization/finalization, and optional audit are reused from the established
source-suite scheduler.  Scientific arguments are forwarded unchanged to the
experiment runner; ``--seed 42`` is added only when the caller omits a seed.
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


RUNNER = (
    REPO_ROOT
    / "scripts"
    / "run_daphnet_persistence_input_ablation.py"
)
AUDITOR = (
    REPO_ROOT
    / "scripts"
    / "audit_daphnet_persistence_input_ablation.py"
)
CANONICAL_REPRESENTATIONS = (
    "raw_support_matched",
    "error_x_minus_mu",
    "standardized_error",
    "standardized_error_clip12",
)
SCHEDULER_VERSION = "daphnet_persistence_input_ablation_multigpu.v1"
LOCK_FILENAME = ".persistence_input_ablation_scheduler.lock"
DEFAULT_SEED = 42

_base_parse_args = scheduler_base.parse_args


def parse_args() -> tuple[Any, list[str]]:
    """Parse scheduler controls and make the fixed default seed explicit."""

    args, forwarded = _base_parse_args()
    if not any(
        value == "--seed" or value.startswith("--seed=")
        for value in forwarded
    ):
        forwarded.extend(["--seed", str(DEFAULT_SEED)])
    return args, forwarded


class OutputDirectoryLock(scheduler_base.OutputDirectoryLock):
    """Use a suite-specific lock while preserving proven lock semantics."""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.path = output_dir / LOCK_FILENAME


@contextmanager
def configured_scheduler() -> Iterator[Any]:
    """Temporarily bind the generic fold scheduler to this experiment."""

    replacements = {
        "RUNNER": RUNNER,
        "AUDITOR": AUDITOR,
        "CANONICAL_CLASSIFIERS": CANONICAL_REPRESENTATIONS,
        "SCHEDULER_VERSION": SCHEDULER_VERSION,
        "OutputDirectoryLock": OutputDirectoryLock,
        "parse_args": parse_args,
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
