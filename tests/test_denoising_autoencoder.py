from __future__ import annotations

import numpy as np
import torch

from cnbr_fog.denoising_autoencoder import (
    ChannelZScoreScaler,
    CorruptionConfig,
    TCNDenoisingAutoencoder,
    corrupt_batch,
    dae_combined_loss,
)


def _single_mode(mode: str) -> CorruptionConfig:
    probabilities = {
        "time_mask_probability": 0.0,
        "channel_mask_probability": 0.0,
        "gaussian_noise_probability": 0.0,
        "clean_probability": 0.0,
    }
    probabilities[f"{mode}_probability"] = 1.0
    return CorruptionConfig(
        **probabilities,
        time_mask_min_samples=7,
        time_mask_max_samples=7,
    )


def test_dae_64hz_shape_bottleneck_and_loss_are_exact() -> None:
    torch.manual_seed(7)
    model = TCNDenoisingAutoencoder(
        in_channels=9,
        input_samples=128,
        latent_dim=128,
        dropout=0.1,
    )
    x = torch.randn(4, 9, 128)
    reconstruction, latent = model(x)
    assert reconstruction.shape == x.shape
    assert latent.shape == (4, 128)
    assert model.encoder_lengths == (128, 64, 32, 16)
    assert model.architecture_config()["encoder_decoder_skip_connections"] is False
    assert model.architecture_config()["parameter_count"] == 790_249

    losses = dae_combined_loss(reconstruction, x)
    assert set(losses) == {"total", "time", "difference", "frequency"}
    assert torch.isfinite(torch.stack(list(losses.values()))).all()
    losses["total"].backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_identity_reconstruction_has_zero_combined_loss() -> None:
    clean = torch.randn(2, 9, 128)
    losses = dae_combined_loss(clean.clone(), clean)
    assert all(float(value) == 0.0 for value in losses.values())


def test_corruptions_apply_exactly_one_selected_mode_per_window() -> None:
    clean = torch.ones(5, 9, 128)
    generator = torch.Generator().manual_seed(123)

    time_masked, mode = corrupt_batch(
        clean, _single_mode("time_mask"), generator=generator
    )
    assert mode.tolist() == [0] * 5
    assert torch.sum(time_masked == 0, dim=(1, 2)).tolist() == [9 * 7] * 5

    channel_masked, mode = corrupt_batch(
        clean, _single_mode("channel_mask"), generator=generator
    )
    assert mode.tolist() == [1] * 5
    assert torch.sum(channel_masked == 0, dim=(1, 2)).tolist() == [128] * 5

    noisy, mode = corrupt_batch(
        clean, _single_mode("gaussian_noise"), generator=generator
    )
    assert mode.tolist() == [2] * 5
    assert not torch.equal(noisy, clean)

    untouched, mode = corrupt_batch(
        clean, _single_mode("clean"), generator=generator
    )
    assert mode.tolist() == [3] * 5
    assert torch.equal(untouched, clean)


def test_training_window_channel_zscore_round_trip() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(
        loc=np.arange(3)[None, :, None],
        scale=np.asarray([0.5, 1.0, 2.0])[None, :, None],
        size=(20, 3, 64),
    ).astype(np.float32)
    scaler = ChannelZScoreScaler.fit_channel_time(values)
    standardized = scaler.transform_channel_time(values)
    np.testing.assert_allclose(standardized.mean(axis=(0, 2)), 0, atol=2e-6)
    np.testing.assert_allclose(standardized.std(axis=(0, 2)), 1, atol=2e-6)
    np.testing.assert_allclose(
        scaler.inverse_channel_time(standardized), values, atol=5e-7
    )

