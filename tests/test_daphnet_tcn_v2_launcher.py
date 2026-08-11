from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.launch_daphnet_tcn_v2_nbm300_c_vs_raw_ep5pat2_7gpu import (
    FOLDS,
    METHODS,
    REQUIRED_SEEDS,
    nbm_command,
    pair_command,
    validate_contract,
    verify_existing_output_identity,
)


def launcher_args() -> Namespace:
    return Namespace(
        python="python",
        data_dir=Path("processed_NBM"),
        output_root=Path("tcn_v2_output"),
        num_workers=0,
        overwrite=False,
        nbm_seeds="0,52,161,5216,52161",
        tcn_seeds="0,52,161,5216,52161",
        nbm_max_epochs=300,
        nbm_patience=20,
        nbm_dropout=0.10,
        tcn_max_epochs=5,
        tcn_patience=2,
    )


def test_tcn_v2_launcher_job_grid_and_contract() -> None:
    args = launcher_args()
    assert validate_contract(args) == REQUIRED_SEEDS
    assert len({(fold, seed) for fold in FOLDS for seed in REQUIRED_SEEDS}) == 15
    assert len(
        {
            (fold, method, seed)
            for fold in FOLDS
            for method in METHODS
            for seed in REQUIRED_SEEDS
        }
    ) == 30


def test_launcher_routes_tcn_v2_and_freezes_300_20_and_5_2() -> None:
    args = launcher_args()
    nbm = nbm_command(args, fold=2, seed=52161)
    assert "run_daphnet_tcn_v2_nbm300_fold.py" in " ".join(nbm)
    assert nbm[nbm.index("--nbm-max-epochs") + 1] == "300"
    assert nbm[nbm.index("--nbm-patience") + 1] == "20"
    train = pair_command(args, "train", 2, "FULL_C", 52161)
    assert train[train.index("--nbm-kind") + 1] == "tcn_v2"
    assert train[train.index("--tcn-max-epochs") + 1] == "5"
    assert train[train.index("--tcn-patience") + 1] == "2"
    assert train[train.index("--nbm-seed") + 1] == "52161"
    assert train[train.index("--tcn-seed") + 1] == "52161"


def test_output_identity_rejects_old_conv_tcn_artifacts(tmp_path) -> None:
    (tmp_path / "runs").mkdir()
    config = tmp_path / "experiment_config.json"
    config.write_text(json.dumps({"nbm_kind": "conv_tcn"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-TCN-v2"):
        verify_existing_output_identity(tmp_path, {})


def test_output_identity_rejects_dataset_change(tmp_path) -> None:
    plan = {
        "experiment_id": "tcn_v2",
        "nbm_kind": "tcn_v2",
        "dataset": "/data/processed_NBM_A",
        "sampling_rate_hz": 64,
        "window_samples": 128,
        "stride_samples": 64,
        "nbm_seeds": [0, 52, 161, 5216, 52161],
        "tcn_seeds": [0, 52, 161, 5216, 52161],
        "nbm_training": "frozen",
        "classifier_training": "frozen",
    }
    launch_plan = tmp_path / "logs" / "launch_plan.json"
    launch_plan.parent.mkdir(parents=True)
    launch_plan.write_text(json.dumps(plan), encoding="utf-8")
    requested = {**plan, "dataset": "/data/processed_NBM_B"}
    with pytest.raises(RuntimeError, match="dataset"):
        verify_existing_output_identity(tmp_path, requested)


def test_output_identity_rejects_tcn_v2_architecture_change(tmp_path) -> None:
    plan = {
        "experiment_id": "tcn_v2",
        "nbm_kind": "tcn_v2",
        "dataset": "/data/processed_NBM",
        "sampling_rate_hz": 64,
        "window_samples": 128,
        "stride_samples": 64,
        "nbm_seeds": [0, 52, 161, 5216, 52161],
        "tcn_seeds": [0, 52, 161, 5216, 52161],
        "nbm_architecture_contract": {
            "name": "global_bottleneck_tcn_autoencoder_nbm_v2",
            "parameter_count": 186_065,
            "bottleneck_shape": ["B", 16],
        },
        "nbm_training": "frozen",
        "classifier_training": "frozen",
    }
    launch_plan = tmp_path / "logs" / "launch_plan.json"
    launch_plan.parent.mkdir(parents=True)
    launch_plan.write_text(json.dumps(plan), encoding="utf-8")
    requested = {
        **plan,
        "nbm_architecture_contract": {
            **plan["nbm_architecture_contract"],
            "bottleneck_shape": ["B", 24],
        },
    }
    with pytest.raises(RuntimeError, match="nbm_architecture_contract"):
        verify_existing_output_identity(tmp_path, requested)
