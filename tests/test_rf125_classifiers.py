from __future__ import annotations

import json

import pytest
import torch

from cnbr_fog.models import ResidualTCNClassifier
from cnbr_fog.rf125_classifiers import (
    CANONICAL_RF125_CLASSIFIER_NAMES,
    DEFAULT_DILATIONS,
    RF125_CLASSIFIER_DISPLAY_NAMES,
    build_rf125_classifier,
    convolutional_receptive_field,
    parameter_count,
    parameter_schema_sha256,
    rf125_classifier_config,
)


EXPECTED_PARAMETERS = 89_329
EXPECTED_MACS = 21_348_912
EXPECTED_RESIDUAL_ADDITIONS = 73_728


def test_rf125_pair_has_exact_shared_capacity_and_receptive_field() -> None:
    assert convolutional_receptive_field(DEFAULT_DILATIONS) == 125
    models = {
        name: build_rf125_classifier(name)
        for name in CANONICAL_RF125_CLASSIFIER_NAMES
    }
    assert {parameter_count(model) for model in models.values()} == {
        EXPECTED_PARAMETERS
    }
    assert {
        parameter_schema_sha256(model) for model in models.values()
    } == {parameter_schema_sha256(models["tcn_m"])}

    configs = {
        name: model.architecture_config()
        for name, model in models.items()
    }
    for name, config in configs.items():
        assert config["canonical_name"] == name
        assert config["display_name"] == RF125_CLASSIFIER_DISPLAY_NAMES[name]
        assert config["dilations"] == [1, 2, 4, 8, 8, 8]
        assert config["n_blocks"] == 6
        assert config["convolutions_per_block"] == 2
        assert config["kernel_size"] == 3
        assert config["local_receptive_field_samples"] == 125
        assert config["local_receptive_field_seconds"] == pytest.approx(
            1.953125
        )
        assert config["conv_linear_macs_per_window"] == EXPECTED_MACS
        assert config["padding"] == "symmetric_same_zero"
        assert config["causal"] is False
        json.dumps(config)

    assert configs["tcn_m"]["residual_skip"] is True
    assert configs["tcn_m"]["block_equation"] == "x_plus_Fx"
    assert (
        configs["tcn_m"]["residual_elementwise_additions_per_window"]
        == EXPECTED_RESIDUAL_ADDITIONS
    )
    assert configs["cnn_rf125"]["residual_skip"] is False
    assert configs["cnn_rf125"]["block_equation"] == "Fx"
    assert (
        configs["cnn_rf125"]["residual_elementwise_additions_per_window"]
        == 0
    )


def test_seeded_rf125_pair_has_identical_initial_state() -> None:
    torch.manual_seed(10042)
    tcn = build_rf125_classifier("tcn_m")
    torch.manual_seed(10042)
    cnn = build_rf125_classifier("cnn_rf125")

    assert tuple(tcn.state_dict()) == tuple(cnn.state_dict())
    for name in tcn.state_dict():
        torch.testing.assert_close(
            tcn.state_dict()[name],
            cnn.state_dict()[name],
            rtol=0.0,
            atol=0.0,
        )


def test_tcn_arm_is_functionally_identical_to_existing_tcn_m() -> None:
    torch.manual_seed(20260727)
    matched = build_rf125_classifier("tcn_m", dropout=0.0)
    existing = ResidualTCNClassifier(
        in_channels=9,
        hidden_channels=48,
        dilations=DEFAULT_DILATIONS,
        kernel_size=3,
        dropout=0.0,
    )
    existing.load_state_dict(matched.state_dict(), strict=True)
    matched.eval()
    existing.eval()
    x = torch.randn(3, 9, 256)
    with torch.no_grad():
        torch.testing.assert_close(
            matched(x),
            existing(x),
            rtol=0.0,
            atol=0.0,
        )


def test_only_block_identity_addition_changes_forward_topology() -> None:
    torch.manual_seed(7)
    tcn = build_rf125_classifier("tcn_m", dropout=0.0)
    torch.manual_seed(7)
    cnn = build_rf125_classifier("cnn_rf125", dropout=0.0)
    for model in (tcn, cnn):
        model.eval()
        with torch.no_grad():
            for block in model.blocks:
                for layer in block.net:
                    if isinstance(layer, torch.nn.Conv1d):
                        layer.weight.zero_()

    x = torch.randn(2, 9, 256)
    with torch.no_grad():
        projected = tcn.projection(x)
        tcn_features = tcn.blocks(projected)
        cnn_features = cnn.blocks(cnn.projection(x))
    torch.testing.assert_close(tcn_features, projected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        cnn_features,
        torch.zeros_like(cnn_features),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("name", CANONICAL_RF125_CLASSIFIER_NAMES)
def test_rf125_forward_gradient_and_input_contract(name: str) -> None:
    model = build_rf125_classifier(name)
    x = torch.randn(2, 9, 256, requires_grad=True)
    logits = model(x)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    with pytest.raises(ValueError, match="9 input channels"):
        model(torch.randn(2, 8, 256))
    with pytest.raises(ValueError, match="256 input samples"):
        model(torch.randn(2, 9, 255))


def test_config_preserves_rng_and_reports_trainable_parameters() -> None:
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    configs = {
        name: rf125_classifier_config(name)
        for name in CANONICAL_RF125_CLASSIFIER_NAMES
    }
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    assert {
        config["parameter_count"] for config in configs.values()
    } == {EXPECTED_PARAMETERS}
    assert {
        config["trainable_parameter_count"] for config in configs.values()
    } == {EXPECTED_PARAMETERS}
