from __future__ import annotations

import copy
import inspect
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import launch_daphnet_gru_mask8_12_three_arm_7gpu as launcher
from scripts import run_daphnet_gru_mask8_12_three_arm as experiment
from scripts import run_daphnet_gru_mask_strength_nbm300_fold as nbm_worker


def _launcher_args(tmp_path: Path) -> Namespace:
    return Namespace(
        python="python",
        data_dir=tmp_path / "processed_NBM",
        output_root=tmp_path / "output",
        num_workers=0,
        overwrite=False,
    )


def _command_options(command: list[str]) -> dict[str, str | bool]:
    """Parse this launcher's simple ``--key value`` command contract."""

    options: dict[str, str | bool] = {}
    index = 2  # python executable and worker path
    while index < len(command):
        key = command[index]
        assert key.startswith("--"), command
        if index + 1 == len(command) or command[index + 1].startswith("--"):
            options[key] = True
            index += 1
        else:
            options[key] = command[index + 1]
            index += 2
    return options


def _role4_scaler_payload(*, fold: int = 0, seed: int = 52) -> dict[str, object]:
    return {
        "fold": fold,
        "seed": seed,
        "scaler_fit_role": 4,
        "scaler_unique_raw_points": 12_345,
        "scientific_data_sha256": "a" * 64,
        "scaler": {
            "median": np.linspace(-1.0, 1.0, 9).tolist(),
            "iqr": np.linspace(0.5, 2.5, 9).tolist(),
            "epsilon": 1e-6,
        },
    }


def _valid_frozen_source(variant: str, seed: int = 52) -> dict[str, object]:
    return {
        "variant": variant,
        "nbm_train_role": 4,
        "nbm_earlystop_and_calibration_role": 5,
        "best_checkpoint_restored_before_calibration": True,
        "validation_mask_or_noise": False,
        "training": {
            "maximum_epochs": 300,
            "patience": 20,
            "seed": seed,
            "architecture": nbm_worker.architecture_config(),
            "augmentation": nbm_worker.augmentation_config(variant),
        },
        "_verified_source_config": {
            "protocol": nbm_worker.protocol_contract(variant, "a" * 64)
        },
    }


def test_public_grid_has_three_methods_and_five_exact_paired_seeds() -> None:
    assert launcher.METHODS == experiment.METHODS == (
        "RAW",
        "GRU_BASE_C",
        "GRU_MASK8_12_C",
    )
    assert launcher.REQUIRED_SEEDS == experiment.REQUIRED_SEEDS == (
        0,
        52,
        161,
        5216,
        52161,
    )
    assert experiment.expected_jobs() == [
        (fold, method, seed)
        for fold in (0, 1, 2)
        for method in experiment.METHODS
        for seed in experiment.REQUIRED_SEEDS
    ]


def test_launcher_dry_run_has_30_nbm_45_train_45_evaluate_jobs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        launcher,
        "processed_nbm_scientific_manifest",
        lambda _path: {"sha256": "a" * 64, "files": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launcher",
            "--dry-run",
            "--data-dir",
            str(tmp_path / "processed_NBM"),
            "--output-root",
            str(tmp_path / "output"),
            "--gpu-ids",
            "0,1,2,3,4,5,6",
        ],
    )

    launcher.main()
    plan = json.loads(capsys.readouterr().out)

    assert plan["gpu_ids"] == ["0", "1", "2", "3", "4", "5", "6"]
    assert plan["nbm_jobs"] == 30
    assert plan["classifier_train_jobs"] == 45
    assert plan["post_barrier_test_jobs"] == 45
    assert plan["methods"] == list(experiment.METHODS)
    assert plan["nbm_seeds"] == list(experiment.REQUIRED_SEEDS)
    assert plan["tcn_seeds"] == list(experiment.REQUIRED_SEEDS)
    assert "cnbr_fog/data.py" in plan["code_sha256"]


def test_experiment_code_hash_binds_the_dataset_loader() -> None:
    relative_paths = {
        path.relative_to(experiment.REPO_ROOT).as_posix()
        for path in experiment.CRITICAL_CODE_PATHS
    }
    assert "cnbr_fog/data.py" in relative_paths


def test_two_nbm_commands_differ_only_by_variant_and_source_root(
    tmp_path: Path,
) -> None:
    args = _launcher_args(tmp_path)
    base = launcher.nbm_command(args, "BASE", fold=1, seed=161)
    stronger = launcher.nbm_command(args, "MASK8_12", fold=1, seed=161)
    assert base[:2] == stronger[:2]

    base_options = _command_options(base)
    stronger_options = _command_options(stronger)
    assert base_options.pop("--variant") == "BASE"
    assert stronger_options.pop("--variant") == "MASK8_12"
    assert "gru_mask4_8" in str(base_options.pop("--output-root"))
    assert "gru_mask8_12" in str(stronger_options.pop("--output-root"))
    assert base_options == stronger_options
    assert base_options["--nbm-max-epochs"] == "300"
    assert base_options["--nbm-patience"] == "20"
    assert base_options["--required-seeds"] == "0,52,161,5216,52161"


def test_variant_checkpoint_mapping_is_explicit_and_non_aliasing() -> None:
    base = experiment.checkpoint_name(experiment.SOURCE_GRU_BASE)
    stronger = experiment.checkpoint_name(experiment.SOURCE_GRU_MASK8_12)
    assert base == nbm_worker.checkpoint_name("BASE") == "gru_nbm_base_best.pt"
    assert (
        stronger
        == nbm_worker.checkpoint_name("MASK8_12")
        == "gru_nbm_mask8_12_best.pt"
    )
    assert base != stronger
    with pytest.raises(ValueError, match="unsupported source kind"):
        experiment.checkpoint_name("gru_v1_mask8_12_typo")


def test_both_nbm_variants_have_identical_gru_structure_and_parameter_count() -> None:
    base = nbm_worker.protocol_contract("BASE", "a" * 64)
    stronger = nbm_worker.protocol_contract("MASK8_12", "a" * 64)
    assert base["architecture"] == stronger["architecture"]
    assert base["architecture"]["parameter_count"] == 31_513

    base_augmentation = copy.deepcopy(base["augmentation"])
    stronger_augmentation = copy.deepcopy(stronger["augmentation"])
    assert base_augmentation.pop("mask_minimum_samples") == 4
    assert base_augmentation.pop("mask_maximum_samples") == 8
    assert stronger_augmentation.pop("mask_minimum_samples") == 8
    assert stronger_augmentation.pop("mask_maximum_samples") == 12
    assert base_augmentation == stronger_augmentation


def test_same_seed_gives_both_nbm_variants_the_same_initial_state_hash() -> None:
    hashes: dict[str, str] = {}
    signatures: dict[str, list[tuple[str, tuple[int, ...]]]] = {}
    for variant in ("BASE", "MASK8_12"):
        # The variant is intentionally selected before initialization; only the
        # corruption call may consume it after the shared model is initialized.
        assert variant in nbm_worker.VARIANTS
        nbm_worker.set_seed(5216)
        model = nbm_worker.GRUReconstructionNBM(
            channels=nbm_worker.CHANNELS,
            hidden=nbm_worker.HIDDEN,
            bottleneck=nbm_worker.BOTTLENECK,
        )
        hashes[variant] = nbm_worker.state_dict_sha256(model.state_dict())
        signatures[variant] = [
            (name, tuple(parameter.shape))
            for name, parameter in model.named_parameters()
        ]
        assert sum(parameter.numel() for parameter in model.parameters()) == 31_513

    assert signatures["BASE"] == signatures["MASK8_12"]
    assert hashes["BASE"] == hashes["MASK8_12"]


def test_two_residual_arms_use_bit_identical_paired_tcn_initialization() -> None:
    base_state, base_meta = experiment.paired_initialization(161, "GRU_BASE_C")
    stronger_state, stronger_meta = experiment.paired_initialization(
        161, "GRU_MASK8_12_C"
    )

    assert base_meta["pair_id"] == stronger_meta["pair_id"]
    assert (
        base_meta["selected_state_sha256"]
        == stronger_meta["selected_state_sha256"]
    )
    assert base_state.keys() == stronger_state.keys()
    for name in base_state:
        assert torch.equal(base_state[name], stronger_state[name]), name


def test_raw_reads_role4_scaler_only_and_never_role5_or_nbm(tmp_path: Path) -> None:
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    (fold_dir / "scaler_role4.json").write_text(
        json.dumps(_role4_scaler_payload()), encoding="utf-8"
    )
    # A parse attempt proves RAW leaked into the NBM/role-5 calibration path.
    (fold_dir / "nbm_frozen.json").write_text("invalid-json", encoding="utf-8")

    _scaler, artifact, contract = experiment.load_role4_scaler_metadata(
        tmp_path, fold=0, seed=52, scientific_data_sha256="a" * 64
    )
    args = Namespace(
        gru_base_source_root=tmp_path,
        gru_mask8_12_source_root=tmp_path / "unused",
    )
    source_kind, source_root, uses_nbm = experiment.source_for_method(args, "RAW")

    assert source_kind == experiment.SOURCE_GRU_BASE
    assert source_root == tmp_path.resolve()
    assert uses_nbm is False
    assert artifact["scaler_fit_role"] == 4
    assert artifact["frozen_json"] is None
    assert artifact["nbm_checkpoint"] is None
    assert artifact["calibration_sigma_sha256"] is None
    assert contract["uses_nbm"] is False
    assert contract["uses_role5_calibration"] is False


def test_source_loader_rejects_variant_impersonation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        experiment,
        "_ORIGINAL_LOAD_SOURCE_METADATA",
        lambda *args, **kwargs: (object(), {}, {"variant": "MASK8_12"}),
    )
    with pytest.raises(AssertionError, match="frozen variant identity mismatch"):
        experiment.load_source_metadata(
            tmp_path,
            fold=0,
            source_kind=experiment.SOURCE_GRU_BASE,
            seed=52,
            scientific_data_sha256="a" * 64,
        )


def test_source_contract_rejects_mask_span_impersonation() -> None:
    args = Namespace(required_nbm_max_epochs=300, required_nbm_patience=20)
    frozen = _valid_frozen_source("BASE")
    contract = experiment.validate_source_contract(
        frozen, experiment.SOURCE_GRU_BASE, seed=52, args=args
    )
    assert contract["mask_span_samples"] == [4, 8]

    impersonator = copy.deepcopy(frozen)
    impersonator["training"]["augmentation"]["mask_minimum_samples"] = 8
    impersonator["training"]["augmentation"]["mask_maximum_samples"] = 12
    with pytest.raises(AssertionError, match="mask_minimum"):
        experiment.validate_source_contract(
            impersonator, experiment.SOURCE_GRU_BASE, seed=52, args=args
        )


def test_evaluate_fails_closed_before_global_training_barrier(tmp_path: Path) -> None:
    args = Namespace(
        output_root=tmp_path,
        fold=0,
        method="RAW",
        tcn_seed=0,
        nbm_seed=0,
        gru_base_source_root=tmp_path / "base",
        gru_mask8_12_source_root=tmp_path / "stronger",
        # Fields used by the underlying mature three-arm implementation.
        gru_v1_source_root=tmp_path / "base",
        gru_v15_source_root=tmp_path / "stronger",
    )
    with pytest.raises(FileNotFoundError, match="roles 0/1 forbidden"):
        experiment.sealed_job(args)


def test_aggregate_exposes_three_deltas_and_mask_strength_primary_comparison() -> None:
    source = inspect.getsource(experiment.run_aggregate)
    for comparison in (
        "GRU_BASE_C_minus_RAW",
        "GRU_MASK8_12_C_minus_RAW",
        "GRU_MASK8_12_C_minus_GRU_BASE_C",
    ):
        assert f'"{comparison}"' in source
    assert 'comparison_name = "GRU_MASK8_12_C_minus_GRU_BASE_C"' in source
    for public_field in (
        '"primary_metrics"',
        '"paired_deltas"',
        '"pre_registered_success"',
        '"sensitivity_mean_delta_at_least_0.010"',
        '"auprc_mean_delta_at_least_minus_0.005"',
        '"precision_mean_delta_at_least_minus_0.010"',
        '"specificity_mean_delta_at_least_minus_0.010"',
    ):
        assert public_field in source
