from __future__ import annotations

import sys

import pytest

from scripts import launch_daphnet_64hz_raw_tcn_ep20pat5_5seed_7gpu as launcher


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def default_args(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["launcher"])
    return launcher.parse_args()


def test_fixed_64hz_raw_ep20pat5_five_seed_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = default_args(monkeypatch)
    assert args.data_dir.name == "processed_NBM"
    assert args.output_root.name == (
        "daphnet_64Hz_raw_tcn_ep20pat5_seedset_0_52_161_5216_52161"
    )
    assert launcher.METHODS == ("RAW",)
    assert launcher.validate_contract(args) == (0, 52, 161, 5216, 52161)
    assert launcher.validate_gpus(args.gpu_ids, check_hardware=False) == [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    ]


def test_exactly_15_train_and_15_post_barrier_test_jobs() -> None:
    count = len(launcher.FOLDS) * len(launcher.REQUIRED_SEEDS)
    assert launcher.FOLDS == (0, 1, 2)
    assert count == 15


def test_worker_command_is_raw_only_and_uses_ep20pat5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = default_args(monkeypatch)
    command = launcher.job_command(args, "train", fold=2, seed=52161)
    assert option_value(command, "--method") == "RAW"
    assert option_value(command, "--experiment-methods") == "RAW"
    assert option_value(command, "--sampling-rate-hz") == "64"
    assert option_value(command, "--window-samples") == "128"
    assert option_value(command, "--stride-samples") == "64"
    assert option_value(command, "--tcn-max-epochs") == "20"
    assert option_value(command, "--tcn-patience") == "5"
    assert option_value(command, "--nbm-seed") == "52161"
    assert option_value(command, "--tcn-seed") == "52161"
    assert option_value(command, "--nbm-source-root").endswith(
        "nbm_source\\seed_52161"
    ) or option_value(command, "--nbm-source-root").endswith(
        "nbm_source/seed_52161"
    )


def test_invalid_training_budget_or_seed_list_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = default_args(monkeypatch)
    args.tcn_max_epochs = 19
    with pytest.raises(ValueError, match="max_epoch=20"):
        launcher.validate_contract(args)
    args.tcn_max_epochs = 20
    args.seeds = "0,52,161"
    with pytest.raises(ValueError, match="exact seeds"):
        launcher.validate_contract(args)


def test_dry_run_reports_no_nbm_training_or_inference(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["launcher", "--dry-run"])
    launcher.main()
    output = capsys.readouterr().out
    assert '"classifier_train_jobs": 15' in output
    assert '"post_barrier_test_jobs": 15' in output
    assert '"nbm_trained_or_inferred": false' in output
    assert '"maximum_epochs": 20' in output
    assert '"early_stopping_patience": 5' in output
