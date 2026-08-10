from __future__ import annotations

import json
import sys

import pytest

from scripts import (
    launch_daphnet_tcn_nbm300_c_vs_raw_tcn_ep5pat2_7gpu as launcher,
)


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_fixed_five_seed_tcn_nbm_ep5pat2_contract(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["launcher"])
    args = launcher.parse_args()

    assert args.data_dir.name == "processed_NBM"
    assert args.output_root.name == (
        "daphnet_tcn_nbm300_C_vs_raw_tcn_ep5pat2_"
        "seedset_0_52_161_5216_52161"
    )
    assert launcher.validate_contract(args) == (0, 52, 161, 5216, 52161)
    assert launcher.NBM_WORKER.name == (
        "run_daphnet_conv_tcn_nbm_gaussian200_fold.py"
    )

    nbm = launcher.nbm_command(args, fold=2, seed=52161)
    assert option_value(nbm, "--fold") == "2"
    assert option_value(nbm, "--seed") == "52161"
    assert option_value(nbm, "--seed-mode") == "exact"
    assert option_value(nbm, "--nbm-max-epochs") == "300"
    assert option_value(nbm, "--nbm-patience") == "20"
    assert option_value(nbm, "--clean-probability") == "0.40"
    assert option_value(nbm, "--gaussian-probability") == "0.40"
    assert option_value(nbm, "--mask-probability") == "0.20"

    raw = launcher.pair_command(
        args, "train", fold=2, method="RAW", seed=52161
    )
    assert option_value(raw, "--fold") == "2"
    assert option_value(raw, "--method") == "RAW"
    assert option_value(raw, "--nbm-kind") == "conv_tcn"
    assert option_value(raw, "--nbm-seed") == "52161"
    assert option_value(raw, "--tcn-seed") == "52161"
    assert option_value(raw, "--required-seeds") == launcher.SEED_TEXT
    assert option_value(raw, "--sampling-rate-hz") == "64"
    assert option_value(raw, "--window-samples") == "128"
    assert option_value(raw, "--stride-samples") == "64"
    assert option_value(raw, "--tcn-max-epochs") == "5"
    assert option_value(raw, "--tcn-patience") == "2"


def test_job_grid_is_three_folds_two_methods_five_paired_seeds() -> None:
    assert launcher.FOLDS == (0, 1, 2)
    assert launcher.METHODS == ("FULL_C", "RAW")
    assert launcher.REQUIRED_SEEDS == (0, 52, 161, 5216, 52161)
    assert len(launcher.FOLDS) * len(launcher.REQUIRED_SEEDS) == 15
    assert (
        len(launcher.FOLDS)
        * len(launcher.METHODS)
        * len(launcher.REQUIRED_SEEDS)
        == 30
    )


def test_rejects_output_root_with_gru_plan(tmp_path) -> None:
    plan_path = tmp_path / "logs" / "launch_plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(
        json.dumps({"nbm_backbone": "GRU(9,64)"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-TCN-NBM"):
        launcher.validate_output_root_identity(tmp_path)


def test_accepts_output_root_with_conv_tcn_artifact(tmp_path) -> None:
    config_path = (
        tmp_path / "nbm_source" / "seed_0" / "fold_0" / "config.json"
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps({"architecture": {"name": "conv_tcn_autoencoder_nbm_v1"}}),
        encoding="utf-8",
    )

    launcher.validate_output_root_identity(tmp_path)
