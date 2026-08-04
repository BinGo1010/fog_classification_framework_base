from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_daphnet_spectrum_nbm_three_rounds",
    ROOT / "scripts" / "run_daphnet_spectrum_nbm_three_rounds.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_power_and_log_spectrum_are_exact_and_finite() -> None:
    rng = np.random.default_rng(20260803)
    raw = rng.normal(size=(4, 128, 9)).astype(np.float32)
    power, log_power = runner.power_and_log_spectrum(raw)
    assert power.shape == (4, 9, 65)
    assert log_power.shape == power.shape
    assert np.isfinite(power).all() and np.all(power >= 0)
    np.testing.assert_allclose(log_power, np.log1p(power), rtol=1e-6, atol=1e-7)


def test_all_models_have_exact_output_shapes_without_dropout() -> None:
    for model_id, bins in (
        ("round1_mlp65", 65),
        ("round2_mlp65", 65),
        ("B0_gru65", 65),
        ("B1_mlp24", 24),
        ("B2_conv24", 24),
    ):
        model = runner.build_model(model_id)
        values = torch.randn(3, 9, bins)
        assert model(values).shape == values.shape
        assert not any(isinstance(module, torch.nn.Dropout) and module.p > 0 for module in model.modules())
    shape_model = runner.build_model("B3_shape_energy_conv24")
    shape, energy = shape_model(torch.softmax(torch.randn(3, 9, 24), dim=-1))
    assert shape.shape == (3, 9, 24)
    assert energy.shape == (3, 9)
    torch.testing.assert_close(shape.sum(dim=-1), torch.ones(3, 9))


def test_shape_energy_conversion_reconstructs_cropped_log_power() -> None:
    rng = np.random.default_rng(20260803)
    full_power = rng.lognormal(size=(5, 9, 65)).astype(np.float32)
    shape, energy = runner.shape_energy_targets(full_power)
    reconstructed = runner.shape_energy_to_log(torch.from_numpy(shape), torch.from_numpy(energy)).numpy()
    expected = np.log1p(full_power[:, :, runner.CROP_MASK])
    np.testing.assert_allclose(reconstructed, expected, rtol=2e-6, atol=2e-6)


def test_balanced_weights_favor_low_energy_and_are_clipped() -> None:
    target = np.stack([np.full((9, 65), value, dtype=np.float32) for value in (0.01, 0.1, 1.0, 10.0)])
    weights = runner.energy_weights(target)
    assert np.all(np.diff(weights) <= 0)
    assert np.all((weights >= 0.5) & (weights <= 3.0))


def test_perfect_prediction_metrics_and_template_skill() -> None:
    rng = np.random.default_rng(20260803)
    actual = rng.lognormal(size=(8, 9, 65)).astype(np.float32)
    metrics = runner.aggregate_metrics(actual, actual.copy(), 1.0, runner.FULL_FREQ)
    assert metrics["mae"] == 0
    assert metrics["nmae_floor"] == 0
    assert metrics["cosine"] > 0.999999
    assert runner.template_skill(actual, actual.copy(), np.median(actual, axis=0)) == 1.0


def test_effect_size_is_finite_with_a_singleton_group() -> None:
    result = runner.effect_size(np.asarray([0.1, 0.2, 0.3]), np.asarray([0.8]))
    assert np.isfinite(result)
    assert result > 0


def test_actual_daphnet_record_splits_and_round1_subsets() -> None:
    data_dir = ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed"
    if not data_dir.exists():
        return
    dataset = runner.small.DaphnetDataset.load(data_dir)
    prepared = {}
    for subject_id in runner.SUBJECTS:
        subject, _ = runner.prepare_subject(dataset, subject_id)
        prepared[subject_id] = subject
        assert not np.any(subject.arrays["train_label"])
        assert not np.any(subject.arrays["validation_label"])
        assert np.any(subject.arrays["test_clean"])
        assert np.any(subject.arrays["test_label"] == 1)
        train_support = {
            (row["record_id"], sample)
            for row in subject.metadata["train"]
            for sample in range(row["start_index"], row["end_index_exclusive"])
        }
        validation_support = {
            (row["record_id"], sample)
            for row in subject.metadata["validation"]
            for sample in range(row["start_index"], row["end_index_exclusive"])
        }
        test_support = {
            (row["record_id"], sample)
            for row in subject.metadata["test"]
            for sample in range(row["start_index"], row["end_index_exclusive"])
        }
        assert train_support.isdisjoint(validation_support)
        assert train_support.isdisjoint(test_support)
        assert validation_support.isdisjoint(test_support)

    runner.validate_splits(prepared)
    for subject_id in runner.ROUND1_SUBJECTS:
        subsets, manifest = runner.round1_subsets(prepared[subject_id])
        assert len(manifest) == 96
        assert len({int(value) for subset in subsets.values() for value in subset}) == 96
        for subset_id in (1, 2, 3):
            rows = [row for row in manifest if row["subset_id"] == subset_id]
            assert len(rows) == 32
            assert [sum(row["energy_quartile"] == f"Q{q}" for row in rows) for q in range(1, 5)] == [8, 8, 8, 8]
