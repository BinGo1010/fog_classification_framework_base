#!/usr/bin/env python
"""Schedule the Transformer horizon-by-fusion ablation on independent GPUs.

One LOSO fold is the indivisible work unit. Each worker trains every requested
Transformer horizon and all nine classifier inputs for that fold before the
GPU is released to another fold.
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


RUNNER_FILENAME = "run_daphnet_transformer_horizon_fusion_ablation.py"
AUDITOR_FILENAME = "audit_daphnet_transformer_horizon_fusion_ablation.py"
SCHEDULER_VERSION = (
    "daphnet_transformer_horizon_fusion_ablation_multigpu.v1"
)
DEFAULT_OUTPUT_DIRNAME = (
    "daphnet_transformer_horizon_fusion_h4_tcnm_loso_seed42"
)
DEFAULT_SEED = 42
HORIZON_VARIANTS = ("H050", "H100", "H200")
INPUT_VARIANTS = (
    "raw4",
    "raw6",
    "raw4_zero",
    "error_h050",
    "raw4_error_h050",
    "error_h100",
    "raw4_error_h100",
    "error_h200",
    "raw4_error_h200",
)
SCHEDULER_DESCRIPTION = (
    "Run the Daphnet Transformer-NBM horizon-by-fusion ablation on "
    "independent GPUs. One GPU completes all horizons and input variants "
    "for its assigned LOSO fold before taking another fold."
)


def _argv_with_default_seed(
    argv: Sequence[str] | None,
) -> list[str]:
    """Return CLI arguments with the locked default seed made explicit."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not any(
        token == "--seed" or token.startswith("--seed=")
        for token in arguments
    ):
        arguments.extend(["--seed", str(DEFAULT_SEED)])
    return arguments


@contextmanager
def configured_scheduler() -> Iterator[Any]:
    """Temporarily specialize the established fold scheduler for this suite."""

    replacements = {
        "RUNNER_FILENAME": RUNNER_FILENAME,
        "AUDITOR_FILENAME": AUDITOR_FILENAME,
        "SCHEDULER_VERSION": SCHEDULER_VERSION,
        "DEFAULT_OUTPUT_DIRNAME": DEFAULT_OUTPUT_DIRNAME,
        "SCHEDULER_DESCRIPTION": SCHEDULER_DESCRIPTION,
        # Production workers must execute the complete protocol for a fold.
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
        return configured.main(_argv_with_default_seed(argv))


if __name__ == "__main__":
    raise SystemExit(main())
