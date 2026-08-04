from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_daphnet_nbm_spectrum_small_overfit",
    ROOT / "scripts" / "run_daphnet_nbm_spectrum_small_overfit.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_spectrum_matches_current_shape_and_is_finite_nonnegative() -> None:
    rng = np.random.default_rng(20260803)
    x = rng.normal(size=(3, 128, 9)).astype(np.float32)
    spectrum = runner.log_power_spectrum(x)
    assert spectrum.shape == (3, 9, 65)
    assert np.isfinite(spectrum).all()
    assert np.all(spectrum >= 0)


def test_reference_and_current_models_reconstruct_same_shape_without_dropout() -> None:
    x = torch.randn(3, 9, 65)
    for name in runner.DEFAULT_MODELS:
        model = runner.build_model(name)
        assert model(x).shape == x.shape
        assert not any(
            isinstance(module, torch.nn.Dropout) and module.p > 0
            for module in model.modules()
        )


def test_perfect_reconstruction_passes_all_levels() -> None:
    rng = np.random.default_rng(20260803)
    for level in (1, 8, 32):
        actual = rng.lognormal(size=(level, 9, 65)).astype(np.float32)
        metrics = runner.summarize_metrics(actual, actual.copy())
        assert metrics["final_mse"] == 0
        assert metrics["final_nmae"] == 0
        assert runner.pass_status(level, metrics) == "Pass"


def test_zero_output_is_not_mistaken_for_memorization() -> None:
    rng = np.random.default_rng(20260803)
    actual = rng.lognormal(size=(8, 9, 65)).astype(np.float32)
    metrics = runner.summarize_metrics(actual, np.zeros_like(actual))
    assert abs(metrics["improvement_vs_zero"]) < 1e-8
    assert metrics["cosine_similarity"] == 0
    assert runner.pass_status(8, metrics) == "Fail"


def test_actual_daphnet_selection_is_guarded_and_nonoverlapping() -> None:
    data_dir = ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed"
    if not data_dir.exists():
        return
    dataset = runner.DaphnetDataset.load(data_dir)
    for subject in runner.DEFAULT_SUBJECTS:
        records, windows, eligible = runner.subject_pool(dataset, subject)
        for level in (1, 8, 32):
            selected = runner.select_windows(level, eligible, records, windows)
            assert len(selected) == level
            assert len({records[int(windows.record_index[index])].record_id for index in selected}) == 1
            for position, index in enumerate(selected):
                record_index = int(windows.record_index[index])
                record = records[record_index]
                start, end = int(windows.start[index]), int(windows.end[index])
                assert not np.any(record.y[start - 2 * runner.FS : end + runner.FS] == 1)
                for prior in selected[:position]:
                    if int(windows.record_index[prior]) == record_index:
                        assert max(start, int(windows.start[prior])) >= min(end, int(windows.end[prior]))
            spectra, _ = runner.prepare_spectra(records, windows, selected)
            assert spectra.shape == (level, 9, 65)
