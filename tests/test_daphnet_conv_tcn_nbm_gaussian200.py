import torch

from scripts.run_daphnet_conv_tcn_nbm_gaussian200_fold import (
    GRUMatchedAugmentation,
    gru_matched_augmentation,
)


def test_gru_matched_augmentation_is_exclusive_and_near_40_40_20() -> None:
    clean = torch.ones(10000, 9, 128)
    config = GRUMatchedAugmentation()
    generator = torch.Generator().manual_seed(12345)
    output, counts = gru_matched_augmentation(clean, config, generator)

    assert output.shape == clean.shape
    assert sum(counts.values()) == len(clean)
    assert abs(counts["clean_windows"] / len(clean) - 0.40) < 0.02
    assert abs(counts["gaussian_windows"] / len(clean) - 0.40) < 0.02
    assert abs(counts["masked_windows"] / len(clean) - 0.20) < 0.02


def test_gru_matched_augmentation_is_seed_deterministic() -> None:
    clean = torch.ones(64, 9, 128)
    config = GRUMatchedAugmentation()
    output_a, counts_a = gru_matched_augmentation(
        clean, config, torch.Generator().manual_seed(99)
    )
    output_b, counts_b = gru_matched_augmentation(
        clean, config, torch.Generator().manual_seed(99)
    )
    assert counts_a == counts_b
    torch.testing.assert_close(output_a, output_b, rtol=0.0, atol=0.0)
