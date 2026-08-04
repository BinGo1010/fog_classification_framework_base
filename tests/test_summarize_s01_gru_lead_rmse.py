from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_s01_gru_lead_rmse.py"
SPEC = importlib.util.spec_from_file_location("s01_lead_rmse_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def test_band_rmse_uses_rms_pooling_not_arithmetic_mean() -> None:
    pointwise = np.asarray(
        [
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 8.0],
        ]
    )
    bands = summary.band_rmse_by_seed(pointwise, band_samples=2)
    assert len(bands) == 2
    assert bands[0] == pytest.approx([np.sqrt(12.5), 0.0])
    assert bands[1] == pytest.approx([0.0, np.sqrt(50.0)])


@pytest.mark.parametrize("shape", [(4,), (0, 4), (2, 3)])
def test_band_rmse_rejects_invalid_matrix_or_band(shape: tuple[int, ...]) -> None:
    values = np.zeros(shape)
    with pytest.raises(ValueError):
        summary.band_rmse_by_seed(values, band_samples=2)
