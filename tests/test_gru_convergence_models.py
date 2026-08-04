from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from cnbr_fog.gru_convergence_models import (  # noqa: E402
    AutoregressiveGRUMeanDecoder,
    ClusterConditionedGRUMeanForecaster,
    DirectGRUMeanForecaster,
    DirectMeanDecoder,
    GRUMeanForecaster,
    JointDirectGRUForecaster,
    MoEGRUMeanForecaster,
    SharedHorizonMeanDecoder,
    TCNMeanDecoder,
)


CHANNELS = 9
HIDDEN = 48
CONTEXT = 32
HORIZON = 16


def trainable_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


@pytest.mark.parametrize("decoder", ["direct", "shared_horizon", "tcn", "gru"])
def test_decoder_forecasters_preserve_common_output_contract(decoder: str) -> None:
    torch.manual_seed(7)
    model = GRUMeanForecaster(
        CHANNELS,
        HORIZON,
        hidden_channels=HIDDEN,
        dropout=0.0,
        decoder=decoder,
    )
    context = torch.randn(3, CHANNELS, CONTEXT)

    mean_only = model.forward_mean(context)
    mean, sigma = model(context)

    assert mean_only.shape == (3, CHANNELS, HORIZON)
    assert mean.shape == mean_only.shape
    assert sigma.shape == mean_only.shape
    torch.testing.assert_close(mean, mean_only)
    torch.testing.assert_close(sigma, torch.ones_like(sigma))
    assert model.model_config()["persistence_anchor"] == "context_last_sample"


@pytest.mark.parametrize("decoder", ["direct", "shared_horizon", "tcn", "gru"])
def test_every_decoder_applies_the_persistence_anchor_exactly_once(
    decoder: str,
) -> None:
    model = GRUMeanForecaster(
        CHANNELS,
        HORIZON,
        hidden_channels=HIDDEN,
        dropout=0.0,
        decoder=decoder,
    )
    with torch.no_grad():
        model.mean_decoder.output_head.weight.zero_()
        if model.mean_decoder.output_head.bias is not None:
            model.mean_decoder.output_head.bias.zero_()
    context = torch.randn(2, CHANNELS, CONTEXT)

    mean = model.forward_mean(context)
    persistence = context[:, :, -1:].expand_as(mean)

    torch.testing.assert_close(mean, persistence)


def test_decoders_and_models_have_no_target_argument_or_teacher_forcing() -> None:
    decoder_types = (
        DirectMeanDecoder,
        SharedHorizonMeanDecoder,
        TCNMeanDecoder,
        AutoregressiveGRUMeanDecoder,
    )
    for decoder_type in decoder_types:
        parameters = inspect.signature(decoder_type.forward).parameters
        assert tuple(parameters) == ("self", "state", "last")
        assert decoder_type.uses_teacher_forcing is False

    assert tuple(inspect.signature(GRUMeanForecaster.forward_mean).parameters) == (
        "self",
        "context",
    )
    assert tuple(inspect.signature(GRUMeanForecaster.forward).parameters) == (
        "self",
        "context",
    )

    autoregressive = AutoregressiveGRUMeanDecoder(HIDDEN, CHANNELS, HORIZON)
    assert autoregressive.model_config()["uses_teacher_forcing"] is False


def test_canonical_decoders_are_parameter_matched_to_direct() -> None:
    horizon = 128
    direct = DirectMeanDecoder(HIDDEN, CHANNELS, horizon)
    direct_count = trainable_parameters(direct)
    assert direct_count == 56_448

    decoders = (
        SharedHorizonMeanDecoder(HIDDEN, CHANNELS, horizon),
        TCNMeanDecoder(HIDDEN, CHANNELS, horizon),
        AutoregressiveGRUMeanDecoder(HIDDEN, CHANNELS, horizon),
    )
    expected = (56_367, 56_249, 56_589)
    assert tuple(trainable_parameters(decoder) for decoder in decoders) == expected
    for decoder in decoders:
        assert abs(trainable_parameters(decoder) / direct_count - 1.0) < 0.005


def test_encoder_initialisation_is_decoder_independent_and_explicitly_copyable() -> None:
    torch.manual_seed(123)
    direct = GRUMeanForecaster(
        CHANNELS,
        HORIZON,
        hidden_channels=HIDDEN,
        dropout=0.0,
        decoder="direct",
    )
    torch.manual_seed(123)
    tcn = GRUMeanForecaster(
        CHANNELS,
        HORIZON,
        hidden_channels=HIDDEN,
        dropout=0.0,
        decoder="tcn",
    )
    for key, value in direct.encoder.state_dict().items():
        torch.testing.assert_close(value, tcn.encoder.state_dict()[key], rtol=0, atol=0)

    with torch.no_grad():
        for parameter in tcn.encoder.parameters():
            parameter.add_(1.0)
    assert any(
        not torch.equal(value, tcn.encoder.state_dict()[key])
        for key, value in direct.encoder.state_dict().items()
    )

    tcn.copy_encoder_from(direct)
    for key, value in direct.encoder.state_dict().items():
        torch.testing.assert_close(value, tcn.encoder.state_dict()[key], rtol=0, atol=0)

    incompatible = DirectGRUMeanForecaster(
        CHANNELS, HORIZON, hidden_channels=32, dropout=0.0
    )
    with pytest.raises(ValueError, match="encoder configurations differ"):
        incompatible.copy_encoder_from(direct)


def _encoder_has_nonzero_gradient(model: JointDirectGRUForecaster) -> bool:
    return any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in model.encoder.parameters()
    )


@pytest.mark.parametrize("detach", [False, True])
def test_joint_direct_sigma_state_detach_controls_sigma_encoder_gradient(
    detach: bool,
) -> None:
    torch.manual_seed(11)
    model = JointDirectGRUForecaster(
        CHANNELS,
        HORIZON,
        hidden_channels=HIDDEN,
        dropout=0.0,
        sigma_state_detach=detach,
    )
    context = torch.randn(4, CHANNELS, CONTEXT)

    _, sigma = model(context)
    sigma.sum().backward()

    assert _encoder_has_nonzero_gradient(model) is (not detach)
    assert model.log_sigma_head.weight.grad is not None
    assert bool(torch.any(model.log_sigma_head.weight.grad != 0))
    assert model.model_config()["sigma_state_detach"] is detach


def test_joint_direct_mean_gradient_remains_when_sigma_state_is_detached() -> None:
    model = JointDirectGRUForecaster(
        CHANNELS,
        HORIZON,
        hidden_channels=HIDDEN,
        dropout=0.0,
        sigma_state_detach=True,
    )
    context = torch.randn(4, CHANNELS, CONTEXT)
    target = torch.randn(4, CHANNELS, HORIZON)

    mean, _ = model(context)
    torch.mean((target - mean).square()).backward()

    assert _encoder_has_nonzero_gradient(model)


def test_cluster_conditioned_model_is_context_only_and_returns_probabilities() -> None:
    model = ClusterConditionedGRUMeanForecaster(
        CHANNELS,
        HORIZON,
        hidden_channels=HIDDEN,
        dropout=0.0,
        n_clusters=3,
        cluster_embedding_dim=6,
    )
    context = torch.randn(5, CHANNELS, CONTEXT)

    mean, sigma = model(context)
    probabilities = model.cluster_probabilities(context)

    assert mean.shape == sigma.shape == (5, CHANNELS, HORIZON)
    assert probabilities.shape == (5, 3)
    torch.testing.assert_close(
        probabilities.sum(dim=1), torch.ones(5), rtol=1e-6, atol=1e-6
    )
    assert model.model_config()["routing"] == "context_only_softmax"


def test_moe_model_is_context_only_and_softly_mixes_experts() -> None:
    model = MoEGRUMeanForecaster(
        CHANNELS,
        HORIZON,
        hidden_channels=HIDDEN,
        dropout=0.0,
        n_experts=3,
    )
    context = torch.randn(5, CHANNELS, CONTEXT)

    mean, sigma = model(context)
    probabilities = model.routing_probabilities(context)

    assert mean.shape == sigma.shape == (5, CHANNELS, HORIZON)
    assert probabilities.shape == (5, 3)
    torch.testing.assert_close(
        probabilities.sum(dim=1), torch.ones(5), rtol=1e-6, atol=1e-6
    )
    assert len(model.experts) == 3
    assert model.model_config()["routing"] == "context_only_softmax"


def test_model_config_is_serialisable_and_discloses_parameter_matching() -> None:
    model = GRUMeanForecaster(
        CHANNELS,
        128,
        hidden_channels=HIDDEN,
        dropout=0.0,
        decoder="shared_horizon",
    )
    config = model.model_config()

    assert config["decoder"]["name"] == "shared_horizon"
    assert config["decoder"]["width"] == 303
    assert config["direct_decoder_parameter_budget"] == 56_448
    assert abs(config["decoder_to_direct_parameter_ratio"] - 1.0) < 0.005
    assert isinstance(config["encoder"], dict)
