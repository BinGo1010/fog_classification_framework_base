from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.tcn_ngm_40k import (
    TCN_NGM_9_PARAMETER_COUNT,
    TCN_NGM_30_PARAMETER_COUNT,
    CapacityMatchedTCNNGM9,
    CapacityMatchedTCNNGM30,
    architecture_config,
    reconstruct_bct,
)


@pytest.mark.parametrize(
    ("channels", "model_class", "expected_parameters"),
    (
        (9, CapacityMatchedTCNNGM9, TCN_NGM_9_PARAMETER_COUNT),
        (30, CapacityMatchedTCNNGM30, TCN_NGM_30_PARAMETER_COUNT),
    ),
)
def test_parameter_and_shape_contract(
    channels: int,
    model_class: type[torch.nn.Module],
    expected_parameters: int,
) -> None:
    model = model_class(dropout=0.10)
    sample = torch.randn(2, channels, 128)

    assert sum(parameter.numel() for parameter in model.parameters()) == expected_parameters
    assert tuple(model.encode(sample).shape) == (2, 16, 32)
    assert tuple(model(sample).shape) == (2, channels, 128)

    config = architecture_config(channels)
    assert config["parameter_count"] == expected_parameters
    assert config["bottleneck_shape"] == ["B", 16, 32]
    assert config["encoder_decoder_skip_connections"] is False
    assert config["input_output_global_residual"] is False


@pytest.mark.parametrize(
    ("channels", "model_class"),
    ((9, CapacityMatchedTCNNGM9), (30, CapacityMatchedTCNNGM30)),
)
def test_finite_training_gradient_and_batched_reconstruction(
    channels: int,
    model_class: type[torch.nn.Module],
) -> None:
    model = model_class(dropout=0.10)
    sample = torch.randn(3, channels, 128)
    loss = torch.nn.functional.smooth_l1_loss(model(sample), sample)
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    values = np.random.default_rng(20260823).normal(
        size=(3, channels, 128)
    ).astype(np.float32)
    reconstructed = reconstruct_bct(
        model,
        values,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert reconstructed.shape == values.shape
    assert reconstructed.dtype == np.float32
    assert np.isfinite(reconstructed).all()
