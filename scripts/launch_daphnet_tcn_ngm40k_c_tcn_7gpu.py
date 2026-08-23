#!/usr/bin/env python3
"""Seven-GPU launcher for Daphnet 40k TCN-NGM + Group-C TCN."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import launch_daphnet_mlp_ngm300_c_tcn_7gpu as launcher
from scripts.tcn_ngm_40k import TCN_NGM_9_PARAMETER_COUNT


NBM_WORKER = REPO_ROOT / "scripts" / "run_daphnet_tcn_ngm40k_fold.py"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "daphnet_tcn_ngm40k_FULL_C_tcn_ep5pat2_seedset_0_52_161_5216_52161"
)


def configure_launcher() -> None:
    launcher.__doc__ = __doc__
    launcher.NBM_WORKER = NBM_WORKER
    launcher.NBM_KIND = "tcn_40k"
    launcher.NBM_DISPLAY_NAME = "TCN-NGM40K"
    launcher.NBM_JOB_LABEL = "TCN_NGM40K"
    launcher.NBM_PARAMETER_COUNT = TCN_NGM_9_PARAMETER_COUNT
    launcher.NBM_BACKBONE = (
        "capacity-matched TCN: Conv 9->30, TCN dilations 1/2, "
        "Conv 30->20, TCN dilations 1/2, bottleneck [B,16,32], "
        "mirrored skip-free decoder"
    )
    launcher.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT


def main() -> None:
    configure_launcher()
    launcher.main()


if __name__ == "__main__":
    main()
