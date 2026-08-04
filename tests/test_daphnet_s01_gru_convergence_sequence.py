from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from cnbr_fog.gru_predictor_artifact import build_mean_model


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_daphnet_s01_gru_convergence_sequence.py"
SPEC = importlib.util.spec_from_file_location("s01_convergence_sequence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sequence_runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sequence_runner)


def test_stage_parser_preserves_scientific_order() -> None:
    assert sequence_runner.parse_stages("overfit,horizon,finalize") == (
        "overfit",
        "horizon",
        "finalize",
    )
    with pytest.raises(ValueError, match="scientific execution order"):
        sequence_runner.parse_stages("horizon,overfit")
    with pytest.raises(ValueError, match="Unknown stage"):
        sequence_runner.parse_stages("overfit,unknown")


@pytest.mark.parametrize("horizon", [16, 32, 64, 128])
def test_short_horizons_are_prefixes_of_same_master_target(horizon: int) -> None:
    sequence = torch.arange(2 * 9 * 256).reshape(2, 9, 256)
    context, target = sequence_runner.split_sequence(sequence, horizon)
    assert torch.equal(context, sequence[:, :, :128])
    assert torch.equal(target, sequence[:, :, 128 : 128 + horizon])


def test_context_only_model_does_not_receive_diagnostic_mode_labels() -> None:
    class ContextOnly(nn.Module):
        def forward_mean(self, context: torch.Tensor) -> torch.Tensor:
            return context[:, :, :2]

    context = torch.randn(3, 9, 128)
    mode = torch.tensor([0, 1, 0])
    output = sequence_runner._mean_output(ContextOnly(), context, mode)
    assert torch.equal(output, context[:, :, :2])


def test_external_mode_requires_explicit_opt_in() -> None:
    class HardConditioned(nn.Module):
        uses_external_mode = True

        def forward_mean(
            self, context: torch.Tensor, mode: torch.Tensor
        ) -> torch.Tensor:
            return context[:, :, :1] + mode[:, None, None]

    context = torch.zeros(2, 9, 128)
    mode = torch.tensor([0, 2])
    output = sequence_runner._mean_output(HardConditioned(), context, mode)
    assert output.shape == (2, 9, 1)
    assert torch.all(output[1] == 2)


def test_paired_comparison_uses_lower_metric_as_gain() -> None:
    reference = [
        {"seed": 42, "rmse": 2.0},
        {"seed": 43, "rmse": 1.0},
    ]
    candidate = [
        {"seed": 42, "rmse": 1.0},
        {"seed": 43, "rmse": 1.1},
    ]
    result = sequence_runner._paired_comparison(reference, candidate, "rmse")
    assert result["candidate_minus_reference"] == [-1.0, pytest.approx(0.1)]
    assert result["relative_gain"] == [0.5, pytest.approx(-0.1)]
    assert result["candidate_win_count"] == 1


def test_decoder_encoder_initialization_is_paired() -> None:
    hashes = []
    for decoder in sequence_runner.DECODER_NAMES:
        sequence_runner.diagnostic.set_seed(42, True)
        model = sequence_runner.GRUMeanForecaster(
            in_channels=9,
            horizon=16,
            hidden_channels=8,
            dropout=0.1,
            decoder=decoder,
        )
        hashes.append(sequence_runner._encoder_sha256(model))
    assert len(set(hashes)) == 1


@pytest.mark.parametrize(
    "model",
    [
        sequence_runner.GRUMeanForecaster(
            in_channels=9, horizon=8, hidden_channels=8, decoder="direct"
        ),
        sequence_runner.GRUMeanForecaster(
            in_channels=9, horizon=8, hidden_channels=8, decoder="shared_horizon"
        ),
        sequence_runner.GRUMeanForecaster(
            in_channels=9, horizon=8, hidden_channels=8, decoder="tcn"
        ),
        sequence_runner.GRUMeanForecaster(
            in_channels=9, horizon=8, hidden_channels=8, decoder="gru"
        ),
        sequence_runner.ClusterConditionedGRUMeanForecaster(
            in_channels=9, horizon=8, hidden_channels=8, n_clusters=3
        ),
        sequence_runner.MoEGRUMeanForecaster(
            in_channels=9, horizon=8, hidden_channels=8, n_experts=3
        ),
    ],
)
def test_final_artifact_constructor_spec_strictly_rebuilds(model: nn.Module) -> None:
    spec = sequence_runner._artifact_constructor_spec(model.model_config())
    rebuilt = build_mean_model(spec["model_class"], spec["constructor_kwargs"])
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    assert rebuilt.model_config() == model.model_config()
