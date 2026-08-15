from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.launch_daphnet_restcn_attention_pool_nbm300_c_vs_raw_7gpu import (
    ARCHITECTURE_NAME,
    FOLDS,
    METHODS,
    NBM_KIND,
    PARAMETER_COUNT,
    REQUIRED_SEEDS,
    architecture_contract,
    nbm_command,
    pair_command,
    validate_contract,
    validate_gpus,
    verify_existing_output_identity,
)


def launcher_args() -> Namespace:
    return Namespace(
        python="python",
        data_dir=Path("processed_NBM"),
        output_root=Path("attention_output"),
        num_workers=0,
        overwrite=False,
        nbm_seeds="0,52,161",
        tcn_seeds="0,52,161",
        nbm_max_epochs=300,
        nbm_patience=20,
        nbm_dropout=0.10,
        tcn_max_epochs=10,
        tcn_patience=2,
    )


def test_launcher_job_grid_and_frozen_contract() -> None:
    args = launcher_args()
    assert validate_contract(args) == REQUIRED_SEEDS
    assert len({(fold, seed) for fold in FOLDS for seed in REQUIRED_SEEDS}) == 9
    assert len(
        {
            (fold, method, seed)
            for fold in FOLDS
            for method in METHODS
            for seed in REQUIRED_SEEDS
        }
    ) == 18
    contract = architecture_contract()
    assert contract["name"] == ARCHITECTURE_NAME
    assert contract["parameter_count"] == PARAMETER_COUNT == 171_905
    assert contract["bottleneck_shape"] == ["B", 16]
    assert contract["encoder_decoder_skip_connections"] is False


def test_launcher_routes_new_kind_and_freezes_300_20_and_10_2() -> None:
    args = launcher_args()
    nbm = nbm_command(args, fold=2, seed=161)
    assert "run_daphnet_restcn_attention_pool_nbm300_fold.py" in " ".join(nbm)
    assert nbm[nbm.index("--nbm-max-epochs") + 1] == "300"
    assert nbm[nbm.index("--nbm-patience") + 1] == "20"

    train = pair_command(args, "train", 2, "FULL_C", 161)
    assert train[train.index("--nbm-kind") + 1] == NBM_KIND
    assert train[train.index("--tcn-max-epochs") + 1] == "10"
    assert train[train.index("--tcn-patience") + 1] == "2"
    assert train[train.index("--nbm-seed") + 1] == "161"
    assert train[train.index("--tcn-seed") + 1] == "161"


def test_gpu_parser_accepts_exactly_the_seven_requested_ids() -> None:
    assert validate_gpus("0,1,2,3,4,5,6", check_hardware=False) == [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    ]
    with pytest.raises(ValueError, match="invalid unique GPU ids"):
        validate_gpus("0,1,1", check_hardware=False)


def test_output_identity_rejects_old_backbone(tmp_path) -> None:
    (tmp_path / "runs").mkdir()
    config = tmp_path / "experiment_config.json"
    config.write_text(json.dumps({"nbm_kind": "tcn_v2"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different NBM"):
        verify_existing_output_identity(tmp_path, {})


def test_output_identity_rejects_architecture_change(tmp_path) -> None:
    plan = {
        "experiment_id": "attention_z16",
        "nbm_kind": NBM_KIND,
        "dataset": "/data/processed_NBM",
        "sampling_rate_hz": 64,
        "window_samples": 128,
        "stride_samples": 64,
        "nbm_seeds": [0, 52, 161],
        "tcn_seeds": [0, 52, 161],
        "nbm_architecture_contract": architecture_contract(),
        "nbm_training": "frozen",
        "classifier_training": "frozen",
    }
    launch_plan = tmp_path / "logs" / "launch_plan.json"
    launch_plan.parent.mkdir(parents=True)
    launch_plan.write_text(json.dumps(plan), encoding="utf-8")
    changed = {
        **plan,
        "nbm_architecture_contract": {
            **plan["nbm_architecture_contract"],
            "attention_heads": 2,
        },
    }
    with pytest.raises(RuntimeError, match="nbm_architecture_contract"):
        verify_existing_output_identity(tmp_path, changed)
