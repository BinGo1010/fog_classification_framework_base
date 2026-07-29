#!/usr/bin/env python
"""Run the Persistence IMU-combination ablation across physical GPUs.

The LOSO fold is the indivisible scheduling unit.  One worker is pinned to one
physical GPU and trains the seven TCN-M sensor combinations sequentially for
that fold:

* ankle;
* thigh;
* trunk;
* ankle + thigh;
* ankle + trunk;
* thigh + trunk;
* all three sensors.

With seven GPUs and the canonical eight folds, seven folds start concurrently
and the remaining fold is assigned to the first GPU that becomes idle.
Scheduling, retry/resume, heartbeat status, logging, locking, CPU
initialization/finalization, and optional independent audit are reused from the
established fold scheduler.  Scientific arguments are forwarded unchanged to
the experiment runner; ``--seed 42`` is added only when the caller omits it.
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
    / "run_daphnet_persistence_imu_ablation.py"
)
AUDITOR = (
    REPO_ROOT
    / "scripts"
    / "audit_daphnet_persistence_imu_ablation.py"
)
CANONICAL_IMU_VARIANTS = (
    "ankle",
    "thigh",
    "trunk",
    "ankle_thigh",
    "ankle_trunk",
    "thigh_trunk",
    "all_three",
)
SCHEDULER_VERSION = "daphnet_persistence_imu7_multigpu.v1"
LOCK_FILENAME = ".persistence_imu_ablation_scheduler.lock"
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
        # The generic scheduler uses this field only for progress metadata.
        "CANONICAL_CLASSIFIERS": CANONICAL_IMU_VARIANTS,
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
