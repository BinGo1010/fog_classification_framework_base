from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from scripts import (
    launch_all_dataset_processed_nbm_exp_within_subject_conv_tcn_ngm_tcn_7gpu
    as launcher,
)
from scripts import (
    run_all_dataset_processed_nbm_exp_within_subject_conv_tcn_ngm_tcn
    as worker,
)


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def default_args(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["launcher"])
    return launcher.parse_args()


def test_conv_tcn_ngm_shape_bottleneck_and_parameter_count() -> None:
    model = worker.ConvTCNNGM30()
    with torch.no_grad():
        x = torch.zeros(2, 30, 128)
        z = model.encode(x)
        output = model(x)
    assert tuple(z.shape) == (2, 16, 32)
    assert tuple(output.shape) == (2, 30, 128)
    assert sum(parameter.numel() for parameter in model.parameters()) == 52_510
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_architecture_and_augmentation_contract() -> None:
    architecture = worker.architecture_config()
    augmentation = worker.augmentation_config()
    assert architecture["name"] == "conv_tcn_autoencoder_ngm_v1_30channel"
    assert architecture["input_shape"] == ["B", 30, 128]
    assert architecture["bottleneck_shape"] == ["B", 16, 32]
    assert architecture["output_shape"] == ["B", 30, 128]
    assert architecture["parameter_count"] == 52_510
    assert architecture["encoder_decoder_skip_connections"] is False
    assert augmentation["clean_probability"] == 0.40
    assert augmentation["gaussian_probability"] == 0.40
    assert augmentation["mask_probability"] == 0.20
    assert augmentation["mask_minimum_samples"] == 4
    assert augmentation["mask_maximum_samples"] == 8


def test_one_epoch_cpu_training_and_ntc_reconstruction(tmp_path) -> None:
    generator = np.random.default_rng(7)
    train_x = generator.normal(size=(4, 128, 30)).astype(np.float32)
    validation_x = generator.normal(size=(2, 128, 30)).astype(np.float32)
    model, training = worker.train_conv_tcn_ngm(
        train_x=train_x,
        validation_x=validation_x,
        destination=tmp_path,
        device=torch.device("cpu"),
        seed=0,
        batch_size=2,
        workers=0,
        maximum_epochs=1,
        patience=1,
    )
    reconstruction = worker.reconstruct_conv_tcn(
        model, validation_x, torch.device("cpu"), batch_size=2
    )
    assert reconstruction.shape == validation_x.shape
    assert training["epochs_completed"] == 1
    assert training["best_epoch"] == 1
    assert training["parameter_count"] == 52_510
    assert (tmp_path / "checkpoints" / worker.NBM_CHECKPOINT_NAME).is_file()


def test_launcher_has_120_jobs_and_exact_seven_gpu_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = default_args(monkeypatch)
    seeds, gpu_ids = launcher.validate_contract(args)
    train_jobs = launcher.jobs(args, seeds, "train")
    evaluate_jobs = launcher.jobs(args, seeds, "evaluate")
    assert seeds == (0, 52, 161, 5216, 52161)
    assert gpu_ids == ["0", "1", "2", "3", "4", "5", "6"]
    assert len(train_jobs) == 8 * 3 * 5 == 120
    assert len(evaluate_jobs) == 120
    assert train_jobs[0]["id"] == "P01_fold0_seed0"
    assert train_jobs[-1]["id"] == "P08_fold2_seed52161"


def test_job_command_freezes_training_budget_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = default_args(monkeypatch)
    seeds, _ = launcher.validate_contract(args)
    command = launcher.jobs(args, seeds, "train")[-1]["command"]
    assert option_value(command, "--stage") == "train"
    assert option_value(command, "--subject") == "P08"
    assert option_value(command, "--fold") == "2"
    assert option_value(command, "--seed") == "52161"
    assert option_value(command, "--nbm-max-epochs") == "300"
    assert option_value(command, "--nbm-patience") == "20"
    assert option_value(command, "--tcn-max-epochs") == "5"
    assert option_value(command, "--tcn-patience") == "2"
    assert option_value(command, "--batch-size") == "128"


def test_final_six_metric_event_contract_is_explicit() -> None:
    assert worker.EVENT_MINIMUM_POSITIVE_WINDOWS == 1
    assert worker.EVENT_MERGE_GAP_SECONDS == 1.0
    assert worker.EVENT_AGGREGATION == "pooled_counts_and_exposure"
    assert "allocation_group" in worker.EVENT_METRIC_VERSION
    assert worker.METRIC_KEYS == (
        "sensitivity",
        "precision",
        "specificity",
        "pr_auc",
        "event_sensitivity",
        "false_alarms_per_hour",
    )


def test_eight_gpu_queue_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = default_args(monkeypatch)
    args.gpu_ids = "0,1,2,3,4,5,6,7"
    _, gpu_ids = launcher.validate_contract(args)
    assert gpu_ids == ["0", "1", "2", "3", "4", "5", "6", "7"]


def test_invalid_gpu_count_or_budget_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = default_args(monkeypatch)
    args.gpu_ids = "0,1,2,3,4,5"
    with pytest.raises(ValueError, match="seven unique"):
        launcher.validate_contract(args)
    args.gpu_ids = "0,1,2,3,4,5,6"
    args.nbm_max_epochs = 299
    with pytest.raises(ValueError, match="max_epoch=300"):
        launcher.validate_contract(args)
