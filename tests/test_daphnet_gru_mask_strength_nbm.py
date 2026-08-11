from __future__ import annotations

import argparse
import inspect
import json

import numpy as np
import pytest
import torch

from cnbr_fog.resume import canonical_fingerprint
from scripts.run_daphnet_gru_mask_strength_nbm300_fold import (
    ARCHITECTURE_NAME,
    PARAMETER_COUNT,
    REQUIRED_SEEDS,
    VARIANTS,
    _artifact_hashes,
    architecture_config,
    augmentation_config,
    checkpoint_name,
    corrupt_local_mask,
    protocol_contract,
    train_gru_mask_strength_nbm,
    validate_existing_nbm,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    GRUReconstructionNBM,
    corrupt,
    set_seed,
)
from scripts.run_daphnet_residual_calibration_abcd import state_dict_sha256


def test_architecture_is_exact_original_gru_v1() -> None:
    model = GRUReconstructionNBM(channels=9, hidden=64, bottleneck=16)
    x = torch.zeros(2, 128, 9)
    assert model(x).shape == x.shape
    assert sum(parameter.numel() for parameter in model.parameters()) == 31_513
    assert PARAMETER_COUNT == 31_513
    config = architecture_config()
    assert config["name"] == ARCHITECTURE_NAME == "gru_reconstruction_nbm_v1"
    assert config["encoder_gru"]["hidden_size"] == 64
    assert config["latent_shape"] == ["B", 16]
    assert config["decoder_gru"]["hidden_size"] == 64
    assert config["decoder_gru"]["input"] == "128-step all-zero sequence"
    assert config["skip_connections"] is False
    assert config["teacher_forcing"] is False


def test_variants_differ_only_in_inclusive_mask_length() -> None:
    assert VARIANTS == {"BASE": (4, 8), "MASK8_12": (8, 12)}
    base = augmentation_config("BASE")
    stronger = augmentation_config("MASK8_12")
    changed = {key for key in base if base[key] != stronger[key]}
    assert changed == {"mask_minimum_samples", "mask_maximum_samples"}
    assert base["clean_probability"] == stronger["clean_probability"] == 0.40
    assert base["gaussian_probability"] == stronger["gaussian_probability"] == 0.40
    assert base["mask_probability"] == stronger["mask_probability"] == 0.20
    assert base["gaussian_std"] == stronger["gaussian_std"] == 0.04
    assert base["mask_length_sampling"] == "discrete_uniform_inclusive"
    assert base["mask_replacement_value"] == 0.0
    assert base["augmentation_roles"] == [4]
    assert base["validation_augmentation"] is False
    assert checkpoint_name("BASE") == "gru_nbm_base_best.pt"
    assert checkpoint_name("MASK8_12") == "gru_nbm_mask8_12_best.pt"


def test_base_corruption_is_bit_exact_with_retained_implementation() -> None:
    clean = torch.randn(64, 128, 9)
    generator_a = torch.Generator().manual_seed(1007)
    generator_b = torch.Generator().manual_seed(1007)
    expected, expected_counts = corrupt(clean, generator_a)
    actual, actual_counts = corrupt_local_mask(
        clean,
        generator_b,
        mask_minimum_samples=4,
        mask_maximum_samples=8,
    )
    assert torch.equal(actual, expected)
    assert np.array_equal(actual_counts, expected_counts)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    ((4, 8), (8, 12)),
)
def test_mask_is_contiguous_all_axis_and_within_requested_bounds(
    minimum: int, maximum: int
) -> None:
    # Ones make zeroed mask samples directly observable; Gaussian rows never
    # contain a full, exactly-zero 9-axis sample with meaningful probability.
    clean = torch.ones(256, 128, 9)
    output, counts = corrupt_local_mask(
        clean,
        torch.Generator().manual_seed(71),
        mask_minimum_samples=minimum,
        mask_maximum_samples=maximum,
    )
    observed = []
    for row in output:
        zero_time = torch.all(row == 0.0, dim=1).nonzero().flatten()
        if len(zero_time):
            assert torch.equal(
                zero_time,
                torch.arange(zero_time[0], zero_time[-1] + 1),
            )
            observed.append(len(zero_time))
    assert len(observed) == int(counts[2])
    assert observed and min(observed) >= minimum and max(observed) <= maximum


def test_training_source_uses_atomic_checkpoint_and_clean_role5() -> None:
    source = inspect.getsource(train_gru_mask_strength_nbm)
    assert "atomic_torch_save(" in source
    validation = source.split("with torch.no_grad():", 1)[1]
    assert "corrupt_local_mask(" not in validation.split("train_loss =", 1)[0]
    assert "model.load_state_dict(payload[\"model_state\"])" in source
    assert REQUIRED_SEEDS == (0, 52, 161, 5216, 52161)


def test_paired_variants_start_from_identical_seeded_state() -> None:
    hashes = []
    for _variant in VARIANTS:
        set_seed(5216)
        model = GRUReconstructionNBM(channels=9, hidden=64, bottleneck=16)
        hashes.append(state_dict_sha256(model.state_dict()))
    assert len(set(hashes)) == 1


def test_done_validation_rejects_tampered_artifact(tmp_path) -> None:
    scientific_hash = "a" * 64
    variant = "MASK8_12"
    fold_dir = tmp_path / "fold_1"
    (fold_dir / "checkpoints").mkdir(parents=True)
    contract = protocol_contract(variant, scientific_hash)
    protocol_hash = canonical_fingerprint(contract)
    scaler_payload = {
        "fold": 1,
        "seed": 52,
        "scaler_fit_role": 4,
        "scaler_unique_raw_points": 999,
        "scaler": {"median": [0.0] * 9, "iqr": [1.0] * 9, "epsilon": 1e-6},
        "scientific_data_sha256": scientific_hash,
        "protocol_sha256": protocol_hash,
    }
    config = {
        "fold": 1,
        "seed": 52,
        "variant": variant,
        "protocol": contract,
        "protocol_sha256": protocol_hash,
    }
    model = GRUReconstructionNBM()
    initial_hash = state_dict_sha256(model.state_dict())
    frozen = {
        "variant": variant,
        "protocol_sha256": protocol_hash,
        "scientific_data_sha256": scientific_hash,
        "scaler": scaler_payload["scaler"],
        "nbm_train_role": 4,
        "nbm_earlystop_and_calibration_role": 5,
        "validation_mask_or_noise": False,
        "best_checkpoint_restored_before_calibration": True,
        "classifier_or_test_roles_accessed": False,
        "training": {
            "best_epoch": 7,
            "best_validation_huber": 0.2,
            "initial_model_state_sha256": initial_hash,
        },
    }
    for name, payload in (
        ("config.json", config),
        ("scaler_role4.json", scaler_payload),
        ("nbm_frozen.json", frozen),
    ):
        (fold_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    checkpoint = fold_dir / "checkpoints" / checkpoint_name(variant)
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": 7,
            "validation_huber": 0.2,
            "seed": 52,
            "variant": variant,
            "initial_model_state_sha256": initial_hash,
            "architecture": architecture_config(),
            "augmentation": augmentation_config(variant),
        },
        checkpoint,
    )
    done = {
        "status": "frozen",
        "fold": 1,
        "seed": 52,
        "variant": variant,
        "maximum_epochs": 300,
        "patience": 20,
        "parameter_count": PARAMETER_COUNT,
        "scientific_data_sha256": scientific_hash,
        "protocol_sha256": protocol_hash,
        "initial_model_state_sha256": initial_hash,
        "best_epoch": 7,
        "best_validation_huber": 0.2,
        **_artifact_hashes(fold_dir, checkpoint),
    }
    (fold_dir / "DONE_NBM.json").write_text(json.dumps(done), encoding="utf-8")
    args = argparse.Namespace(fold=1, seed=52, variant=variant)
    validate_existing_nbm(fold_dir, args, scientific_hash)
    with (fold_dir / "config.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(AssertionError, match="artifact hash mismatch"):
        validate_existing_nbm(fold_dir, args, scientific_hash)
