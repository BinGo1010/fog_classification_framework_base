from __future__ import annotations

import json

import pytest
import torch

from cnbr_fog.residual_classifiers import (
    CANONICAL_CLASSIFIER_NAMES,
    CLASSIFIER_DISPLAY_NAMES,
    CLASSIFIER_REGISTRY,
    build_residual_classifier,
    canonical_classifier_name,
    classifier_config,
    parameter_count,
)


EXPECTED_PARAMETER_COUNTS = {
    "mlp": 92_241,
    "cnn1d": 85_857,
    "gru": 90_035,
    "transformer": 86_355,
}


def test_registry_uses_stable_canonical_and_display_names() -> None:
    assert CANONICAL_CLASSIFIER_NAMES == ("mlp", "cnn1d", "gru", "transformer")
    assert tuple(CLASSIFIER_REGISTRY) == CANONICAL_CLASSIFIER_NAMES
    assert CLASSIFIER_DISPLAY_NAMES == {
        "mlp": "MLP",
        "cnn1d": "Multi-scale 1D-CNN",
        "gru": "GRU",
        "transformer": "Lightweight Transformer",
    }
    assert canonical_classifier_name("  CNN1D ") == "cnn1d"


@pytest.mark.parametrize("name", CANONICAL_CLASSIFIER_NAMES)
def test_default_model_forward_gradient_and_serializable_config(name: str) -> None:
    torch.manual_seed(20260727)
    model = build_residual_classifier(name)
    x = torch.randn(2, 9, 256, requires_grad=True)

    logits = model(x)
    assert logits.shape == (2,)
    assert logits.dtype == x.dtype
    assert torch.isfinite(logits).all()

    logits.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    parameter_gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert parameter_gradients
    assert all(gradient is not None for gradient in parameter_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)

    config = model.architecture_config()
    assert config["canonical_name"] == name
    assert config["display_name"] == CLASSIFIER_DISPLAY_NAMES[name]
    assert config["in_channels"] == 9
    assert config["input_samples"] == 256
    assert config["dropout"] == pytest.approx(0.15)
    json.dumps(config)
    assert parameter_count(model) == EXPECTED_PARAMETER_COUNTS[name]
    assert parameter_count(model, trainable_only=True) == parameter_count(model)


@pytest.mark.parametrize("name", CANONICAL_CLASSIFIER_NAMES)
def test_seeded_construction_and_evaluation_are_deterministic(name: str) -> None:
    torch.manual_seed(42)
    first = build_residual_classifier(name, dropout=0.0)
    torch.manual_seed(42)
    second = build_residual_classifier(name, dropout=0.0)

    first_parameters = list(first.parameters())
    second_parameters = list(second.parameters())
    assert len(first_parameters) == len(second_parameters)
    for first_parameter, second_parameter in zip(
        first_parameters, second_parameters
    ):
        torch.testing.assert_close(first_parameter, second_parameter)

    x = torch.randn(2, 9, 256)
    first.eval()
    second.eval()
    with torch.no_grad():
        torch.testing.assert_close(first(x), second(x), rtol=0.0, atol=0.0)


def test_classifier_config_preserves_rng_and_reports_parameter_count() -> None:
    torch.manual_seed(1234)
    state_before = torch.random.get_rng_state().clone()
    configs = {
        name: classifier_config(name) for name in CANONICAL_CLASSIFIER_NAMES
    }
    state_after = torch.random.get_rng_state()

    assert torch.equal(state_before, state_after)
    for name, config in configs.items():
        assert config["canonical_name"] == name
        assert config["parameter_count"] == EXPECTED_PARAMETER_COUNTS[name]
        json.dumps(config)


@pytest.mark.parametrize("name", CANONICAL_CLASSIFIER_NAMES)
@pytest.mark.parametrize(
    ("bad_input", "exception", "message"),
    [
        (torch.randn(9, 256), ValueError, "shape"),
        (torch.randn(2, 8, 256), ValueError, "9 input channels"),
        (torch.randn(2, 9, 255), ValueError, "256 input samples"),
        (torch.ones(2, 9, 256, dtype=torch.int64), TypeError, "floating-point"),
        (torch.empty(0, 9, 256), ValueError, "at least one"),
    ],
)
def test_models_reject_invalid_input(
    name: str,
    bad_input: torch.Tensor,
    exception: type[Exception],
    message: str,
) -> None:
    model = build_residual_classifier(name)
    with pytest.raises(exception, match=message):
        model(bad_input)


def test_invalid_model_names_and_hyperparameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown residual classifier"):
        build_residual_classifier("tcn")
    with pytest.raises(TypeError, match="must be a string"):
        build_residual_classifier(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="in_channels"):
        build_residual_classifier("mlp", in_channels=0)
    with pytest.raises(ValueError, match="dropout"):
        build_residual_classifier("gru", dropout=1.0)
    with pytest.raises(ValueError, match="divisible"):
        build_residual_classifier("transformer", model_dim=62, num_heads=4)
    with pytest.raises(ValueError, match="odd"):
        build_residual_classifier("cnn1d", kernel_sizes=(3, 4, 15))


def test_model_specific_architecture_contracts() -> None:
    cnn = build_residual_classifier("cnn1d")
    assert cnn.architecture_config()["kernel_sizes"] == [3, 7, 15]

    gru = build_residual_classifier("gru")
    gru_config = gru.architecture_config()
    assert gru_config["num_layers"] == 2
    assert gru_config["bidirectional"] is False

    transformer = build_residual_classifier("transformer")
    transformer_config = transformer.architecture_config()
    assert transformer_config["num_layers"] == 2
    assert transformer_config["num_heads"] == 4
