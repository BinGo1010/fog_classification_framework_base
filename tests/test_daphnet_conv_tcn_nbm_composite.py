import torch

from scripts.run_daphnet_conv_tcn_nbm_composite_fold import (
    CompositeLossConfig,
    DynamicAugmentationConfig,
    composite_reconstruction_loss,
    dynamic_corruption,
)


def test_composite_loss_is_near_zero_for_identical_nonconstant_signals() -> None:
    generator = torch.Generator().manual_seed(7)
    target = torch.randn(8, 9, 128, generator=generator)
    components = composite_reconstruction_loss(target.clone(), target, CompositeLossConfig())
    assert float(components["smoothl1"]) == 0.0
    assert float(components["first_difference"]) == 0.0
    assert float(components["correlation"]) < 1e-6
    assert float(components["total"]) < 1e-6


def test_correlation_loss_has_finite_backward_for_constant_windows() -> None:
    prediction = torch.zeros(4, 9, 128, requires_grad=True)
    target = torch.zeros(4, 9, 128)
    components = composite_reconstruction_loss(
        prediction, target, CompositeLossConfig()
    )
    components["total"].backward()
    assert torch.isfinite(components["total"])
    assert prediction.grad is not None
    assert torch.all(torch.isfinite(prediction.grad))


def test_dynamic_augmentation_is_mutually_exclusive_and_resampled() -> None:
    clean = torch.ones(2000, 9, 128)
    config = DynamicAugmentationConfig()
    generator = torch.Generator().manual_seed(20260807)
    first, first_modes = dynamic_corruption(clean, config, generator)
    second, second_modes = dynamic_corruption(clean, config, generator)

    assert set(first_modes.unique().tolist()) == {0, 1, 2}
    assert not torch.equal(first_modes, second_modes)
    counts = torch.bincount(first_modes, minlength=3)
    assert abs(int(counts[0]) - 800) < 100
    assert abs(int(counts[1]) - 800) < 100
    assert abs(int(counts[2]) - 400) < 100

    clean_indices = torch.nonzero(first_modes == 0, as_tuple=False).flatten()
    gaussian_indices = torch.nonzero(first_modes == 1, as_tuple=False).flatten()
    mask_indices = torch.nonzero(first_modes == 2, as_tuple=False).flatten()
    assert torch.equal(first[clean_indices], clean[clean_indices])
    assert torch.all(torch.any(first[gaussian_indices] != clean[gaussian_indices], dim=(1, 2)))
    for index in mask_indices[:20].tolist():
        zero_per_channel = torch.sum(first[index] == 0.0, dim=1)
        assert torch.all(zero_per_channel == zero_per_channel[0])
        assert 4 <= int(zero_per_channel[0]) <= 8
