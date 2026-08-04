from __future__ import annotations

import pytest
import torch

from cnbr_fog.temporal_conv_autoencoder import TemporalConvAutoencoder


@pytest.mark.parametrize(
    ("variant", "latent_shape"),
    (("base", (2, 32, 16)), ("wide", (2, 64, 16)), ("long", (2, 48, 32))),
)
def test_tcdae_variants_preserve_output_shape_and_temporal_latent(
    variant: str, latent_shape: tuple[int, int, int]
) -> None:
    torch.manual_seed(20260802)
    model = TemporalConvAutoencoder(variant=variant)
    x = torch.randn(2, 9, 128)
    reconstruction, latent = model(x)
    assert reconstruction.shape == x.shape
    assert latent.shape == latent_shape
    config = model.architecture_config()
    assert config["encoder_decoder_long_skip"] is False
    assert config["output_activation"] is None
    assert config["latent_elements"] == latent_shape[1] * latent_shape[2]
    torch.nn.functional.mse_loss(reconstruction, x).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_tcdae_rejects_wrong_layout() -> None:
    model = TemporalConvAutoencoder(variant="base")
    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(2, 128, 9))
