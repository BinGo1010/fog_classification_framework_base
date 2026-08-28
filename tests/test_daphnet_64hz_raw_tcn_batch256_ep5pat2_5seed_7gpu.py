from __future__ import annotations

import sys

import numpy as np
import pytest

from scripts import (
    launch_daphnet_64hz_raw_tcn_batch256_ep5pat2_5seed_7gpu as launcher,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import (
    classifier_loader,
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


def test_fixed_batch256_ep5pat2_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = default_args(monkeypatch)
    assert args.output_root.name == (
        "daphnet_64Hz_raw_tcn_batch256_ep5pat2_"
        "seedset_0_52_161_5216_52161"
    )
    assert args.batch_size == 256
    assert args.tcn_max_epochs == 5
    assert args.tcn_patience == 2
    assert launcher.base.validate_contract(args) == (0, 52, 161, 5216, 52161)


def test_exactly_15_raw_jobs_and_batch_reaches_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = default_args(monkeypatch)
    assert len(launcher.base.FOLDS) * len(launcher.base.REQUIRED_SEEDS) == 15
    command = launcher.base.job_command(args, "train", fold=2, seed=52161)
    assert option_value(command, "--method") == "RAW"
    assert option_value(command, "--batch-size") == "256"
    assert option_value(command, "--tcn-max-epochs") == "5"
    assert option_value(command, "--tcn-patience") == "2"


def test_classifier_loader_really_batches_256_samples() -> None:
    x = np.zeros((300, 128, 9), dtype=np.float32)
    y = np.zeros(300, dtype=np.float32)
    loader = classifier_loader(
        x, y, shuffle=False, seed=0, num_workers=0, batch_size=256
    )
    batches = [len(batch_x) for batch_x, _ in loader]
    assert batches == [256, 44]


def test_batch_or_budget_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = default_args(monkeypatch)
    args.batch_size = 128
    with pytest.raises(ValueError, match="batch_size=256"):
        launcher.base.validate_contract(args)
    args = default_args(monkeypatch)
    args.tcn_max_epochs = 256
    with pytest.raises(ValueError, match="max_epoch=5"):
        launcher.base.validate_contract(args)


def test_dry_run_records_batch256_without_nbm(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["launcher", "--dry-run"])
    launcher.main()
    output = capsys.readouterr().out
    assert '"classifier_train_jobs": 15' in output
    assert '"post_barrier_test_jobs": 15' in output
    assert '"nbm_trained_or_inferred": false' in output
    assert '"maximum_epochs": 5' in output
    assert '"early_stopping_patience": 2' in output
    assert '"batch_size": 256' in output
