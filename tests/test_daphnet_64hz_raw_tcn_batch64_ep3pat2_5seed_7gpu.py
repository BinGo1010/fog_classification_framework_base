from __future__ import annotations

import sys

import pytest

from scripts import (
    launch_daphnet_64hz_raw_tcn_batch64_ep3pat2_5seed_7gpu as launcher,
)


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def default_args(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["launcher"])
    return launcher.base.parse_args(
        tcn_max_epochs=launcher.TCN_MAX_EPOCHS,
        tcn_patience=launcher.TCN_PATIENCE,
        classifier_batch_size=launcher.CLASSIFIER_BATCH_SIZE,
        default_experiment=launcher.DEFAULT_EXPERIMENT,
    )


def test_fixed_batch64_ep3pat2_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = default_args(monkeypatch)
    assert args.output_root.name == (
        "daphnet_64Hz_raw_tcn_batch64_ep3pat2_"
        "seedset_0_52_161_5216_52161"
    )
    assert args.batch_size == 64
    assert args.tcn_max_epochs == 3
    assert args.tcn_patience == 2
    assert launcher.base.validate_contract(args) == (0, 52, 161, 5216, 52161)


def test_15_jobs_and_worker_receives_exact_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = default_args(monkeypatch)
    assert len(launcher.base.FOLDS) * len(launcher.base.REQUIRED_SEEDS) == 15
    command = launcher.base.job_command(args, "train", fold=2, seed=52161)
    assert option_value(command, "--method") == "RAW"
    assert option_value(command, "--batch-size") == "64"
    assert option_value(command, "--tcn-max-epochs") == "3"
    assert option_value(command, "--tcn-patience") == "2"


def test_budget_drift_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    args = default_args(monkeypatch)
    args.tcn_max_epochs = 5
    with pytest.raises(ValueError, match="max_epoch=3"):
        launcher.base.validate_contract(args)


def test_dry_run_records_batch64_ep3pat2(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["launcher", "--dry-run"])
    launcher.main()
    output = capsys.readouterr().out
    assert '"classifier_train_jobs": 15' in output
    assert '"post_barrier_test_jobs": 15' in output
    assert '"nbm_trained_or_inferred": false' in output
    assert '"maximum_epochs": 3' in output
    assert '"early_stopping_patience": 2' in output
    assert '"batch_size": 64' in output
