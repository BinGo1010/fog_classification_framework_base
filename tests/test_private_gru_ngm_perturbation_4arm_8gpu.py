from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from scripts import launch_private_gru_ngm_perturbation_4arm_8gpu as launcher
from scripts import run_private_gru_ngm_perturbation_4arm as worker


def test_four_arm_probability_contract_is_nested() -> None:
    assert worker.ARMS == (
        "none",
        "gaussian_only",
        "mask_only",
        "gaussian_mask",
    )
    assert worker.ARM_PROBABILITIES == {
        "none": (1.0, 0.0, 0.0),
        "gaussian_only": (0.6, 0.4, 0.0),
        "mask_only": (0.8, 0.0, 0.2),
        "gaussian_mask": (0.4, 0.4, 0.2),
    }
    for arm in worker.ARMS:
        config = worker.augmentation_config(arm)
        assert config["gaussian_std"] == 0.04
        assert (config["mask_minimum_samples"], config["mask_maximum_samples"]) == (4, 8)
        assert config["mask_contiguous"] is True
        assert config["mask_all_channels"] is True
        assert config["validation_augmentation"] is False
        assert sum(worker.ARM_PROBABILITIES[arm]) == 1.0


def test_corruptions_are_deterministic_shape_safe_and_mode_exclusive() -> None:
    clean = torch.ones(4096, 128, 30)
    results: dict[str, tuple[torch.Tensor, np.ndarray]] = {}
    for arm in worker.ARMS:
        first = worker.corrupt_for_arm(
            clean, arm, torch.Generator().manual_seed(1234)
        )
        second = worker.corrupt_for_arm(
            clean, arm, torch.Generator().manual_seed(1234)
        )
        torch.testing.assert_close(first[0], second[0], rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(first[1], second[1])
        assert first[0].shape == clean.shape
        assert torch.isfinite(first[0]).all()
        assert int(first[1].sum()) == len(clean)
        results[arm] = first

    torch.testing.assert_close(results["none"][0], clean, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(results["none"][1], [4096, 0, 0])
    assert results["gaussian_only"][1][2] == 0
    assert results["mask_only"][1][1] == 0
    assert results["gaussian_mask"][1][1] > 0
    assert results["gaussian_mask"][1][2] > 0

    for arm, expected in worker.ARM_PROBABILITIES.items():
        observed = results[arm][1] / len(clean)
        np.testing.assert_allclose(observed, expected, atol=0.03, rtol=0.0)


def test_mask_is_one_contiguous_all_channel_block_of_length_four_to_eight() -> None:
    clean = torch.ones(128, 128, 30)
    corrupted, counts = worker.corrupt_for_arm(
        clean, "mask_only", torch.Generator().manual_seed(91)
    )
    masked_windows = 0
    for window in corrupted:
        zero_rows = torch.all(window == 0.0, dim=1).cpu().numpy()
        if not zero_rows.any():
            continue
        masked_windows += 1
        indices = np.flatnonzero(zero_rows)
        assert 4 <= len(indices) <= 8
        np.testing.assert_array_equal(indices, np.arange(indices[0], indices[-1] + 1))
        assert torch.all(window[indices] == 0.0)
    assert masked_windows == counts[2]


def test_eight_gpu_launcher_builds_all_480_paired_jobs() -> None:
    args = Namespace(
        data_dir=Path("/data/processed_NBM_Exp"),
        output_root=Path("/runs/four_arm"),
        python="python",
        num_workers=2,
        nbm_batch_size=16,
        maximum_updates=5000,
        validation_frequency=50,
        validation_patience=20,
        learning_rate=3e-4,
        weight_decay=1e-4,
        overwrite=False,
    )
    jobs = launcher.jobs(args, worker.SUBJECTS, worker.FOLDS, worker.SEEDS)
    assert len(jobs) == 8 * 3 * 5 * 4 == 480
    assert [job["command"][job["command"].index("--arm") + 1] for job in jobs[:4]] == list(
        worker.ARMS
    )
    assert all(
        job["command"][job["command"].index("--device") + 1] == "cuda:0"
        for job in jobs
    )
    identities = {job["id"] for job in jobs}
    assert len(identities) == len(jobs)


def test_previous_step_training_settings_are_frozen() -> None:
    args = Namespace(
        nbm_batch_size=16,
        maximum_updates=5000,
        validation_frequency=50,
        validation_patience=20,
        learning_rate=3e-4,
        weight_decay=1e-4,
    )
    launcher.validate_frozen_settings(args)
    contract = worker.training_contract(args, "gaussian_mask")
    assert contract["scheduler"] is None
    assert contract["batch_size"] == 16
    assert contract["maximum_updates"] == 5000
    assert contract["checkpoint"] == "minimum uncorrupted role-5 SmoothL1"
    assert contract["permanent_test_roles_loaded"] is False
