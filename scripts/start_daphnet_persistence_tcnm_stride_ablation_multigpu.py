#!/usr/bin/env python
"""Run the Persistence TCN-M stride ablation across physical GPUs.

One LOSO fold is the indivisible scheduling unit.  A fold worker owns its GPU
while it loads the shared frozen Persistence residual cache and trains S1, S2,
and S3 sequentially.  With seven GPUs and the canonical eight folds, the first
seven folds start concurrently and the remaining fold is assigned to the first
GPU that becomes idle.

The process scheduler, retry/resume protocol, heartbeat status, log handling,
CPU initialization/finalization, and optional audit are reused from the
source-based residual-classifier launcher.  Scientific arguments are forwarded
unchanged to ``run_daphnet_persistence_tcnm_stride_ablation.py``; an explicit
``--seed 42`` is added only when the caller omits ``--seed``.
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
    / "run_daphnet_persistence_tcnm_stride_ablation.py"
)
AUDITOR = (
    REPO_ROOT
    / "scripts"
    / "audit_daphnet_persistence_tcnm_stride_ablation.py"
)
CANONICAL_STRIDE_VARIANTS = ("s1", "s2", "s3")
SCHEDULER_VERSION = "daphnet_persistence_tcnm_stride3_multigpu.v1"
LOCK_FILENAME = ".persistence_tcnm_stride3_scheduler.lock"
DEFAULT_SEED = 42

_base_parse_args = scheduler_base.parse_args


def parse_args() -> tuple[Any, list[str]]:
    """Parse scheduler controls and make the locked default seed explicit."""

    args, forwarded = _base_parse_args()
    if not any(
        value == "--seed" or value.startswith("--seed=")
        for value in forwarded
    ):
        forwarded.extend(["--seed", str(DEFAULT_SEED)])
    return args, forwarded


class OutputDirectoryLock(scheduler_base.OutputDirectoryLock):
    """Use a suite-specific lock while preserving the proven lock semantics."""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.path = output_dir / LOCK_FILENAME


@contextmanager
def configured_scheduler() -> Iterator[Any]:
    """Temporarily bind the generic source-suite scheduler to this suite."""

    replacements = {
        "RUNNER": RUNNER,
        "AUDITOR": AUDITOR,
        "CANONICAL_CLASSIFIERS": CANONICAL_STRIDE_VARIANTS,
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
