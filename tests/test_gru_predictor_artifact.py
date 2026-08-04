from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import torch

from cnbr_fog.gru_convergence_models import GRUMeanForecaster
from cnbr_fog.gru_predictor_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    FixedSigmaGRUPredictor,
    build_mean_model,
    load_gru_predictor_artifact,
)


def test_fixed_sigma_wrapper_never_returns_mean_model_unit_adapter() -> None:
    mean_model = GRUMeanForecaster(
        in_channels=9, horizon=4, hidden_channels=8, dropout=0.0
    )
    fixed_sigma = torch.full((1, 9, 4), 2.5)
    predictor = FixedSigmaGRUPredictor(mean_model, fixed_sigma)
    context = torch.randn(3, 9, 16)
    mean, sigma = predictor(context)
    assert mean.shape == sigma.shape == (3, 9, 4)
    assert torch.all(sigma == 2.5)
    residual = predictor.standardized_residual(context, mean + 5.0)
    assert torch.allclose(residual, torch.full_like(residual, 2.0))


def test_artifact_round_trip_has_executable_constructor() -> None:
    kwargs = {
        "in_channels": 9,
        "horizon": 4,
        "hidden_channels": 8,
        "num_layers": 1,
        "dropout": 0.0,
        "decoder": "shared_horizon",
        "decoder_width": 6,
    }
    original = build_mean_model("GRUMeanForecaster", kwargs)
    path = Path(tempfile.gettempdir()) / f"gru-artifact-{uuid.uuid4().hex}.pt"
    try:
        torch.save(
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "model_class": "GRUMeanForecaster",
                "constructor_kwargs": kwargs,
                "model_state": original.state_dict(),
                "fixed_sigma": torch.full((1, 9, 4), 0.75),
            },
            path,
        )
        loaded = load_gru_predictor_artifact(path)
        context = torch.randn(2, 9, 16)
        with torch.no_grad():
            expected = original.forward_mean(context)
            actual, sigma = loaded(context)
        assert torch.allclose(actual, expected)
        assert torch.all(sigma == 0.75)
    finally:
        path.unlink(missing_ok=True)
