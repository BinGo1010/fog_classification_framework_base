from __future__ import annotations

import sys

from scripts import launch_daphnet_gru_nbm300_c_vs_raw_ep5pat2_7gpu as launcher


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_fixed_five_seed_ep5pat2_contract(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["launcher"])
    args = launcher.parse_args()

    assert args.data_dir.name == "processed_NBM"
    assert args.output_root.name == (
        "daphnet_tcn_nbm300_C_vs_raw_tcn_ep5pat2_"
        "seedset_0_52_161_5216_52161"
    )
    assert launcher.validate_contract(args) == (0, 52, 161, 5216, 52161)

    nbm = launcher.nbm_command(args, fold=2, seed=52161)
    assert option_value(nbm, "--fold") == "2"
    assert option_value(nbm, "--seed") == "52161"
    assert option_value(nbm, "--nbm-max-epochs") == "300"
    assert option_value(nbm, "--nbm-patience") == "20"

    raw = launcher.pair_command(args, "train", fold=2, method="RAW", seed=52161)
    assert option_value(raw, "--fold") == "2"
    assert option_value(raw, "--method") == "RAW"
    assert option_value(raw, "--nbm-seed") == "52161"
    assert option_value(raw, "--tcn-seed") == "52161"
    assert option_value(raw, "--tcn-max-epochs") == "5"
    assert option_value(raw, "--tcn-patience") == "2"


def test_job_grid_is_three_folds_two_methods_five_paired_seeds() -> None:
    assert launcher.FOLDS == (0, 1, 2)
    assert launcher.METHODS == ("FULL_C", "RAW")
    assert launcher.REQUIRED_SEEDS == (0, 52, 161, 5216, 52161)
    assert (
        len(launcher.FOLDS)
        * len(launcher.METHODS)
        * len(launcher.REQUIRED_SEEDS)
        == 30
    )
