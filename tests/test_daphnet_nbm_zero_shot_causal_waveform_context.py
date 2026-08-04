from pathlib import Path

import numpy as np
import pytest
import torch

from cnbr_fog.data import DaphnetDataset
from scripts import run_daphnet_nbm_routeA_final_residual_validation as a1
from scripts import run_daphnet_nbm_zero_shot_causal_waveform_context as experiment


def test_causal_context_model_shapes_and_parameter_invariance() -> None:
    parameter_counts = []
    for input_samples, latent_samples in ((128, 32), (256, 64), (384, 96)):
        model = experiment.CausalContextM3(input_samples)
        predicted, latent = model(torch.randn(2, 9, input_samples))
        assert predicted.shape == (2, 9, 128)
        assert latent.shape == (2, 48, latent_samples)
        assert model.architecture_config()["future_samples"] == 0
        parameter_counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert parameter_counts == [64633, 64633, 64633]


@pytest.mark.parametrize("loss_name", experiment.LOSSES)
def test_waveform_losses_are_finite_and_differentiable(loss_name: str) -> None:
    predicted = torch.randn(4, 9, 128, requires_grad=True)
    target = torch.randn(4, 9, 128)
    target[0, 0] = 0.0
    loss = experiment.waveform_loss(loss_name, predicted, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_l1_uses_template_epsilon() -> None:
    predicted = torch.ones(1, 9, 128)
    target = torch.zeros_like(predicted)
    observed = experiment.waveform_loss("L1", predicted, target)
    assert float(observed) == pytest.approx(1000.0, rel=1e-5)


def test_causal_arrays_remain_inside_frozen_split_and_use_no_future() -> None:
    data_dir = (
        Path(__file__).resolve().parents[1]
        / "dataset"
        / "1.Daphnet Freezing of Gait Dataset"
        / "processed"
    )
    dataset = DaphnetDataset.load(data_dir)
    item = a1.prepare_subject(dataset, "S02")
    w0 = experiment.causal_context_arrays(item, item.test_indices, 128)
    w2 = experiment.causal_context_arrays(item, item.test_indices, 384)
    assert len(w2[2]) <= len(w0[2])
    assert w2[0].shape[1:] == (384, 9)
    assert w2[1].shape[1:] == (128, 9)
    for row in w2[3]:
        assert row["future_samples_after_target"] == 0
        assert row["input_end_exclusive"] == row["target_end_exclusive"]
        assert row["input_start"] >= row["interval_start"]
        assert row["input_end_exclusive"] <= row["interval_end"]
        assert row["input_within_same_split"] is True


def test_result_rank_prefers_waveform_quality_before_pass_count() -> None:
    better_waveform = [
        {
            "median_corr": 0.7,
            "median_nrmse": 0.8,
            "nrmse_p90": 0.9,
            "median_lagged_pearson": 0.71,
            "median_amplitude_ratio": 0.9,
            "strict_pass": False,
        }
    ]
    more_passes = [
        {
            "median_corr": 0.6,
            "median_nrmse": 0.7,
            "nrmse_p90": 0.8,
            "median_lagged_pearson": 0.61,
            "median_amplitude_ratio": 0.95,
            "strict_pass": True,
        }
    ]
    assert experiment.result_rank(better_waveform) > experiment.result_rank(more_passes)


def test_log_spectrum_loss_uses_phase_independent_amplitude() -> None:
    time = torch.arange(128, dtype=torch.float32) / 64.0
    target = torch.sin(2 * torch.pi * 3 * time)[None, None].repeat(1, 9, 1)
    same = experiment.log_spectrum_loss(target, target)
    different = experiment.log_spectrum_loss(torch.zeros_like(target), target)
    assert float(same) == pytest.approx(0.0, abs=1e-7)
    assert float(different) > float(same)
