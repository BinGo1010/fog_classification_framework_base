from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_daphnet_nbm_small_overfit",
    ROOT / "scripts" / "run_daphnet_nbm_small_overfit.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_model_matches_preregistered_shape_and_bottleneck() -> None:
    model = runner.current.GRUReconstructionNBM(channels=9, hidden=64, bottleneck=32)
    x = torch.randn(3, 128, 9)
    with torch.no_grad():
        output = model(x)
    assert output.shape == x.shape
    assert model.encoder.input_size == 9
    assert model.encoder.hidden_size == 64
    assert model.to_bottleneck.in_features == 64
    assert model.to_bottleneck.out_features == 32
    assert model.to_decoder.in_features == 32
    assert model.to_decoder.out_features == 64
    assert model.decoder.input_size == 9
    assert model.decoder.hidden_size == 64


def test_perfect_reconstruction_passes_all_levels() -> None:
    rng = np.random.default_rng(20260802)
    actual = rng.normal(size=(8, 128, 9)).astype(np.float32)
    metrics = runner.summarize_metrics(actual, actual.copy())
    assert metrics["nbm_huber"] == 0.0
    assert metrics["improvement_pct"] == 100.0
    assert metrics["median_corr"] > 0.999999
    assert metrics["median_nrmse"] == 0.0
    assert metrics["median_amplitude_ratio"] > 0.999999
    assert all(runner.pass_status(level, metrics) == "PASS" for level in (1, 8, 32, 128))


def test_zero_output_is_not_mistaken_for_memorization() -> None:
    time = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    actual = np.stack([np.sin((channel + 1) * time) for channel in range(9)], axis=1)
    actual = np.repeat(actual[None].astype(np.float32), 8, axis=0)
    metrics = runner.summarize_metrics(actual, np.zeros_like(actual))
    assert abs(metrics["improvement_pct"]) < 1e-8
    assert metrics["median_corr"] == 0.0
    assert metrics["median_amplitude_ratio"] == 0.0
    assert runner.pass_status(8, metrics) == "FAIL"


def test_actual_daphnet_selection_is_nonoverlapping_and_centered() -> None:
    data_dir = ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed"
    if not data_dir.exists():
        return
    dataset = runner.DaphnetDataset.load(data_dir)
    records, windows, candidates = runner.subject_pool(dataset, "S01")
    eligible, energy = runner.eligible_candidates(records, windows, candidates)
    for sample_count in (1, 8, 32):
        selected = runner.select_windows(sample_count, eligible, energy, records, windows)
        assert len(selected) == sample_count
        assert not any(
            runner.overlaps(windows, records, int(candidate), selected[:position])
            for position, candidate in enumerate(selected)
        )
        x, _ = runner.prepare_run_data(records, windows, selected)
        assert x.shape == (sample_count, 128, 9)
        assert np.max(np.abs(x.mean(axis=1))) < 2e-5
