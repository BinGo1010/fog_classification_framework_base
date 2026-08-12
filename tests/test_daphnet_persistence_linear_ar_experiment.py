from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.launch_daphnet_persistence_vs_linear_ar_7gpu as launcher
import scripts.run_daphnet_persistence_linear_ar_nbm_fold as nbm
import scripts.run_daphnet_persistence_vs_linear_ar_tcn as pair


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_persistence_is_exact_recommended_lag_one() -> None:
    x = np.arange(2 * 128 * 9, dtype=np.float32).reshape(2, 128, 9)
    reconstructed = nbm.persistence_reconstruct(x)
    np.testing.assert_array_equal(reconstructed[:, 0, :], x[:, 0, :])
    np.testing.assert_array_equal(reconstructed[:, 1:, :], x[:, :-1, :])


def test_linear_ar8_parameter_count_and_contract() -> None:
    model = nbm.MultivariateLinearAR8()
    assert sum(parameter.numel() for parameter in model.parameters()) == 657
    architecture = nbm.architecture_config("LINEAR_AR")
    assert architecture["order"] == 8
    assert architecture["cross_channel"] is True
    assert architecture["coefficient_shape"] == [9, 8, 9]
    assert architecture["causal"] is True


def test_linear_ar8_is_strictly_causal() -> None:
    torch.manual_seed(12)
    model = nbm.MultivariateLinearAR8().eval()
    x = torch.randn(3, 128, 9)
    reference = model(x)
    changed = x.clone()
    changed[:, 40:, :] += torch.randn_like(changed[:, 40:, :]) * 100
    candidate = model(changed)
    torch.testing.assert_close(candidate[:, :41, :], reference[:, :41, :])
    assert not torch.equal(candidate[:, 41:, :], reference[:, 41:, :])


def test_linear_ar8_first_sample_is_identity() -> None:
    model = nbm.MultivariateLinearAR8().eval()
    x = torch.randn(4, 128, 9)
    torch.testing.assert_close(model(x)[:, 0, :], x[:, 0, :])


def test_only_two_scheme_c_methods_and_job_grid() -> None:
    assert pair.METHODS == ("PERSISTENCE_C", "LINEAR_AR_C")
    assert "RAW" not in pair.METHODS
    assert len(pair.expected_jobs()) == 30
    assert len(set(pair.expected_jobs())) == 30
    for method in pair.METHODS:
        contract = pair.feature_contract(method)
        assert contract["shape"] == ["B", 27, 128]
        assert contract["uses_nbm"] is True
        assert contract["subtracts_role5_bias"] is False


def test_two_tcn_arms_have_identical_initialization() -> None:
    for seed in pair.REQUIRED_SEEDS:
        persistence_state, persistence_info = pair.paired_initialization(
            seed, "PERSISTENCE_C"
        )
        linear_state, linear_info = pair.paired_initialization(seed, "LINEAR_AR_C")
        assert persistence_info == linear_info
        assert persistence_state.keys() == linear_state.keys()
        for key in persistence_state:
            torch.testing.assert_close(
                persistence_state[key], linear_state[key], rtol=0, atol=0
            )


def test_augmentation_is_mask_4_to_8_only_for_linear_ar() -> None:
    persistence = nbm.augmentation_config("PERSISTENCE")
    linear = nbm.augmentation_config("LINEAR_AR")
    assert persistence["applicable"] is False
    assert linear["clean_probability"] == pytest.approx(0.4)
    assert linear["gaussian_probability"] == pytest.approx(0.4)
    assert linear["mask_probability"] == pytest.approx(0.2)
    assert linear["gaussian_std"] == pytest.approx(0.04)
    assert linear["mask_minimum_samples"] == 4
    assert linear["mask_maximum_samples"] == 8
    assert linear["training_target"] == "uncorrupted clean role-4 window"


def test_launcher_builds_30_source_train_and_evaluate_jobs(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launcher",
            "--dry-run",
            "--python",
            "python",
        ],
    )
    args = launcher.parse_args()
    seeds = launcher.validate_contract(args)
    nbm_specs = [
        (variant, fold, seed)
        for variant in launcher.NBM_VARIANTS
        for fold in launcher.FOLDS
        for seed in seeds
    ]
    classifier_specs = [
        (fold, method, seed)
        for fold in launcher.FOLDS
        for method in launcher.METHODS
        for seed in seeds
    ]
    assert len(nbm_specs) == 30
    assert len(classifier_specs) == 30
    assert len(classifier_specs) == 30


def test_launcher_dry_run_reports_frozen_design() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "launch_daphnet_persistence_vs_linear_ar_7gpu.py"),
            "--dry-run",
            "--python",
            "python",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["methods"] == ["PERSISTENCE_C", "LINEAR_AR_C"]
    assert plan["nbm_jobs"] == 30
    assert plan["classifier_train_jobs"] == 30
    assert plan["post_barrier_test_jobs"] == 30
    assert plan["nbm_architectures"]["PERSISTENCE_C"]["parameter_count"] == 0
    assert plan["nbm_architectures"]["LINEAR_AR_C"]["parameter_count"] == 657
    assert "--gpu-ids" not in plan["example_train"]


def test_evaluation_fails_closed_without_global_barrier(tmp_path: Path) -> None:
    class Args:
        output_root = tmp_path

    with pytest.raises(FileNotFoundError, match="TRAINING_BARRIER"):
        pair.sealed_job(Args())


def test_source_kind_mapping_has_no_raw(tmp_path: Path) -> None:
    class Args:
        persistence_source_root = tmp_path / "persistence"
        linear_ar_source_root = tmp_path / "linear"

    assert pair.source_for_method(Args(), "PERSISTENCE_C")[0] == "persistence_lag1"
    assert pair.source_for_method(Args(), "LINEAR_AR_C")[0] == "multivariate_linear_ar8"
    with pytest.raises(ValueError):
        pair.source_for_method(Args(), "RAW")
