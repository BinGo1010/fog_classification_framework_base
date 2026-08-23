from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import (
    launch_all_dataset_processed_nbm_exp_within_subject_persistence_ngm_tcn_8gpu
    as launcher,
)
from scripts import (
    run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn as shared,
)
from scripts import (
    run_all_dataset_processed_nbm_exp_within_subject_persistence_ngm_tcn as worker,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import RobustScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "dataset" / "All_dataset" / "processed_NBM_Exp"


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_persistence_ngm_is_exact_lag_one_and_parameter_free() -> None:
    model = worker.PersistenceNGM()
    assert sum(parameter.numel() for parameter in model.parameters()) == 0
    x = torch.arange(2 * 128 * 30, dtype=torch.float32).reshape(2, 128, 30)
    actual = model(x)
    torch.testing.assert_close(actual[:, 0], x[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(actual[:, 1:], x[:, :-1], rtol=0, atol=0)
    architecture = worker.architecture_config()
    assert architecture["effective_context_samples"] == 1
    assert architecture["trainable"] is False
    assert architecture["parameter_count"] == 0


def test_persistence_has_no_mask_noise_optimizer_or_ngm_epochs() -> None:
    augmentation = worker.augmentation_config()
    assert augmentation["applicable"] is False
    assert augmentation["mask_augmentation"] is False
    assert augmentation["gaussian_augmentation"] is False
    assert worker.NBM_PARAMETER_COUNT == 0


def test_freeze_writes_a_parameter_free_checkpoint(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    role4 = rng.normal(size=(3, 128, 30)).astype(np.float32)
    role5 = rng.normal(size=(2, 128, 30)).astype(np.float32)
    model, training = worker.freeze_persistence_ngm(
        role4,
        role5,
        tmp_path,
        torch.device("cpu"),
        seed=52,
        batch_size=128,
        workers=0,
        maximum_epochs=0,
        patience=0,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 0
    assert training["fit_performed"] is False
    assert training["optimizer_steps"] == 0
    assert training["epochs_completed"] == 0
    checkpoint = tmp_path / "checkpoints" / worker.NBM_CHECKPOINT_NAME
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = worker.build_persistence_from_checkpoint(payload, torch.device("cpu"))
    assert sum(parameter.numel() for parameter in restored.parameters()) == 0


def test_scheme_c_produces_90_channel_persistence_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shared, "reconstruct", worker.persistence_reconstruct)
    scaler = RobustScaler(
        median=np.zeros(30, dtype=np.float32),
        iqr=np.ones(30, dtype=np.float32),
    )
    rng = np.random.default_rng(11)
    raw = rng.normal(size=(4, 128, 30)).astype(np.float32)
    features = shared.scheme_c_features(
        worker.PersistenceNGM(),
        scaler,
        np.ones(30, dtype=np.float32),
        raw,
        torch.device("cpu"),
        batch_size=128,
    )
    assert features.shape == (4, 90, 128)
    # First 30 channels are r and remain centered along time.
    assert float(np.max(np.abs(features[:, :30].mean(axis=2)))) < 1e-5
    # The middle block is |r| and the final block is delta(r).
    np.testing.assert_allclose(features[:, 30:60], np.abs(features[:, :30]))
    np.testing.assert_allclose(features[:, 60:90, 0], 0.0)


def test_launcher_has_120_train_and_120_evaluate_jobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launcher",
            "--data-dir",
            str(DATA_DIR),
            "--output-root",
            str(tmp_path / "persistence"),
            "--dry-run",
        ],
    )
    args = launcher.parse_args()
    seeds = launcher.validate_contract(args)
    assert launcher.validate_gpu_ids(args.gpu_ids) == [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
    ]
    train_jobs = launcher.jobs(args, seeds, "train")
    evaluate_jobs = launcher.jobs(args, seeds, "evaluate")
    assert len(train_jobs) == 8 * 3 * 5 == 120
    assert len(evaluate_jobs) == 120
    first = train_jobs[0]["command"]
    last = train_jobs[-1]["command"]
    assert option_value(first, "--subject") == "P01"
    assert option_value(first, "--fold") == "0"
    assert option_value(first, "--seed") == "0"
    assert option_value(last, "--subject") == "P08"
    assert option_value(last, "--fold") == "2"
    assert option_value(last, "--seed") == "52161"
    assert option_value(first, "--nbm-max-epochs") == "0"
    assert option_value(first, "--nbm-patience") == "0"
    assert option_value(first, "--tcn-max-epochs") == "5"
    assert option_value(first, "--tcn-patience") == "2"


def test_final_six_metric_contract_is_frozen() -> None:
    assert worker.METRIC_KEYS == (
        "sensitivity",
        "precision",
        "specificity",
        "pr_auc",
        "event_sensitivity",
        "false_alarms_per_hour",
    )
    assert worker.EVENT_MINIMUM_POSITIVE_WINDOWS == 1
    assert worker.EVENT_MERGE_GAP_SECONDS == 1.0
    assert worker.EVENT_AGGREGATION == "pooled_counts_and_exposure"
    assert "allocation_group" in worker.EVENT_METRIC_VERSION
