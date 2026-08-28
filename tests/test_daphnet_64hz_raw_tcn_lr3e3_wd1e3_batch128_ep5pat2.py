from __future__ import annotations

import sys

import pytest

from scripts import (
    launch_daphnet_64hz_raw_tcn_lr3e3_wd1e3_batch128_ep5pat2_5seed_7gpu
    as launcher,
)


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def default_args(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["launcher"])
    return launcher.base.parse_args(
        tcn_max_epochs=launcher.TCN_MAX_EPOCHS,
        tcn_patience=launcher.TCN_PATIENCE,
        classifier_batch_size=launcher.CLASSIFIER_BATCH_SIZE,
        tcn_learning_rate=launcher.TCN_LEARNING_RATE,
        tcn_weight_decay=launcher.TCN_WEIGHT_DECAY,
        default_experiment=launcher.DEFAULT_EXPERIMENT,
    )


def test_exact_hyperparameter_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    args = default_args(monkeypatch)
    assert args.output_root.name == launcher.DEFAULT_EXPERIMENT
    assert args.tcn_max_epochs == 5
    assert args.tcn_patience == 2
    assert args.batch_size == 128
    assert args.tcn_learning_rate == pytest.approx(3e-3)
    assert args.tcn_weight_decay == pytest.approx(1e-3)
    assert launcher.base.validate_contract(args) == (0, 52, 161, 5216, 52161)


def test_15_raw_jobs_and_values_reach_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = default_args(monkeypatch)
    assert len(launcher.base.FOLDS) * len(launcher.base.REQUIRED_SEEDS) == 15
    command = launcher.base.job_command(args, "train", fold=2, seed=52161)
    assert option_value(command, "--method") == "RAW"
    assert option_value(command, "--batch-size") == "128"
    assert option_value(command, "--tcn-max-epochs") == "5"
    assert option_value(command, "--tcn-patience") == "2"
    assert float(option_value(command, "--tcn-learning-rate")) == pytest.approx(3e-3)
    assert float(option_value(command, "--tcn-weight-decay")) == pytest.approx(1e-3)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("tcn_learning_rate", 1e-3, "learning_rate=0.003"),
        ("tcn_weight_decay", 1e-4, "weight_decay=0.001"),
    ),
)
def test_optimizer_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: float,
    message: str,
) -> None:
    args = default_args(monkeypatch)
    setattr(args, field, value)
    with pytest.raises(ValueError, match=message):
        launcher.base.validate_contract(args)


def test_dry_run_records_optimizer_settings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["launcher", "--dry-run"])
    launcher.main()
    output = capsys.readouterr().out
    assert '"classifier_train_jobs": 15' in output
    assert '"maximum_epochs": 5' in output
    assert '"batch_size": 128' in output
    assert '"optimizer": "AdamW(lr=0.003, weight_decay=0.001)"' in output
