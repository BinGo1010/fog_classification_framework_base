from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preflight_fog_suite import check_suite


def test_multimodal_full_suite_preflight_smoke() -> None:
    report = check_suite(
        argparse.Namespace(
            config=REPO_ROOT / "configs" / "multimodal_full_suite.json",
            require_windows=True,
        )
    )

    assert report["errors"] == []
    assert len(report["experiments"]) == 4
    assert len(report["training_outputs"]) == 4

    windows = {tuple(entry["class_names"]): entry for entry in report["unique_windows"]}
    assert ("NORMAL", "FOG") in windows
    assert ("NORMAL", "PRE_FOG", "FOG") in windows

    for entry in windows.values():
        assert entry["exists"] is True
        assert entry["x_shape"] == (12422, 100, 24)
        assert entry["fold_count"] == 12
