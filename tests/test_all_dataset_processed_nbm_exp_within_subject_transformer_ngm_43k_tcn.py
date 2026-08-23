from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import (
    launch_all_dataset_processed_nbm_exp_within_subject_transformer_ngm_43k_tcn_8gpu
    as launcher,
)
from scripts import (
    run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn as shared,
)
from scripts import (
    run_all_dataset_processed_nbm_exp_within_subject_transformer_ngm_43k_tcn
    as worker,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import RobustScaler


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_transformer_ngm_exact_parameter_and_shape_contract() -> None:
    model = worker.TinyPatchTransformerNGM(dropout=0.10).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count == worker.NBM_PARAMETER_COUNT == 43_360
    x = torch.randn(3, 128, 30)
    with torch.no_grad():
        z = model.encode(x)
        y = model(x)
    assert z.shape == (3, 16)
    assert y.shape == x.shape


def test_patchify_fold_is_exact() -> None:
    x = torch.arange(2 * 128 * 30, dtype=torch.float32).reshape(2, 128, 30)
    patches = worker.TinyPatchTransformerNGM.patchify(x)
    restored = worker.TinyPatchTransformerNGM.fold_patches(patches)
    assert patches.shape == (2, 16, 240)
    torch.testing.assert_close(restored, x, rtol=0, atol=0)


def test_architecture_has_global_bottleneck_and_no_bypass() -> None:
    architecture = worker.architecture_config()
    assert architecture["bottleneck_shape"] == ["B", 16]
    assert architecture["encoder"]["layers"] == 2
    assert architecture["encoder"]["d_model"] == 32
    assert architecture["encoder"]["heads"] == 4
    assert architecture["encoder"]["ffn"] == 64
    assert architecture["decoder"]["layers"] == 1
    assert architecture["encoder_decoder_skip_connections"] is False
    assert architecture["cross_attention"] is False
    assert architecture["teacher_forcing"] is False
    assert architecture["raw_input_bypass"] is False
    assert architecture["parameter_count"] == 43_360


def test_transformer_uses_frozen_mask4_8_training_augmentation() -> None:
    augmentation = worker.augmentation_config()
    assert augmentation["clean_probability"] == 0.40
    assert augmentation["gaussian_probability"] == 0.40
    assert augmentation["mask_probability"] == 0.20
    assert augmentation["gaussian_std"] == 0.04
    assert augmentation["mask_minimum_samples"] == 4
    assert augmentation["mask_maximum_samples"] == 8
    assert augmentation["augmentation_roles"] == [4]
    assert augmentation["validation_augmentation"] is False


def test_scheme_c_output_is_90_channels() -> None:
    model = worker.TinyPatchTransformerNGM(dropout=0.0).eval()
    scaler = RobustScaler(
        median=np.zeros(30, dtype=np.float32),
        iqr=np.ones(30, dtype=np.float32),
    )
    raw = np.random.default_rng(9).normal(size=(2, 128, 30)).astype(np.float32)
    features = shared.scheme_c_features(
        model,
        scaler,
        np.ones(30, dtype=np.float32),
        raw,
        torch.device("cpu"),
        batch_size=128,
    )
    assert features.shape == (2, 90, 128)
    assert float(np.max(np.abs(features[:, :30].mean(axis=2)))) < 1e-5
    np.testing.assert_allclose(features[:, 30:60], np.abs(features[:, :30]))
    np.testing.assert_allclose(features[:, 60:90, 0], 0.0)


def test_launcher_grid_and_frozen_training_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["launcher", "--output-root", str(tmp_path / "transformer")],
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
    assert option_value(first, "--nbm-max-epochs") == "300"
    assert option_value(first, "--nbm-patience") == "20"
    assert option_value(first, "--tcn-max-epochs") == "5"
    assert option_value(first, "--tcn-patience") == "2"


def test_six_metrics_and_final_event_rule() -> None:
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
