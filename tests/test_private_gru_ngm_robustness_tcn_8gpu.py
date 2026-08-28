from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from scripts import launch_private_gru_ngm_robustness_tcn_8gpu as launcher
from scripts import train_private_gru_ngm_robustness_tcn as worker


def make_source(
    root: Path,
    arm: str = "none",
    subject: str = "P01",
    fold: int = 0,
    seed: int = 0,
) -> Path:
    directory = root / subject / f"fold_{fold}" / f"seed_{seed}"
    checkpoint = directory / "checkpoints" / "gru_ngm_best.pt"
    checkpoint.parent.mkdir(parents=True)
    model = worker.base.GRUReconstructionNBM(
        channels=worker.base.RAW_CHANNELS,
        hidden=worker.base.HIDDEN,
        bottleneck=worker.base.BOTTLENECK,
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": worker.base.architecture_config(),
            "arm": arm,
            "seed": seed,
            "step": 123,
        },
        checkpoint,
    )
    scaler = {
        "scaler": {
            "median": [0.0] * worker.base.RAW_CHANNELS,
            "iqr": [1.0] * worker.base.RAW_CHANNELS,
            "epsilon": 1e-6,
        },
        "subject": subject,
        "fold": fold,
        "seed": seed,
    }
    (directory / "scaler_role4.json").write_text(
        json.dumps(scaler), encoding="utf-8"
    )
    return directory


def test_private_source_resolver_accepts_arm_root_layout(tmp_path: Path) -> None:
    expected = make_source(
        tmp_path,
        arm="gaussian_mask",
        subject="P03",
        fold=2,
        seed=52161,
    )
    resolved = worker.resolve_source_run_dir(
        tmp_path, "gaussian_mask", "P03", 2, 52161
    )
    assert resolved == expected.resolve()
    manifest = worker.inspect_source_artifacts(
        tmp_path, "gaussian_mask", "P03", 2, 52161
    )
    assert manifest["checkpoint_seed"] == 52161
    assert manifest["checkpoint_step"] == 123
    assert manifest["parameter_count"] == worker.base.NBM_PARAMETER_COUNT
    assert len(manifest["scaler"]["median"]) == 30
    assert len(manifest["checkpoint_sha256"]) == 64


def test_private_source_error_shows_subject_layer(tmp_path: Path) -> None:
    try:
        worker.resolve_source_run_dir(tmp_path, "none", "P01", 0, 0)
    except FileNotFoundError as error:
        message = str(error)
        assert "30-channel" in message
        assert "P01" in message
        assert "fold_0" in message
        assert "seed_0" in message
    else:
        raise AssertionError("missing source checkpoint should fail closed")


def touch_complete_sources(root: Path, arm: str, subject: str) -> None:
    for fold in worker.FOLDS:
        for seed in worker.SEEDS:
            checkpoint = (
                root
                / subject
                / f"fold_{fold}"
                / f"seed_{seed}"
                / "checkpoints"
                / "gru_ngm_best.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()


def test_auto_subject_detection_requires_15_paired_checkpoints(
    tmp_path: Path,
) -> None:
    none_root = tmp_path / "none"
    gaussian_mask_root = tmp_path / "gaussian_mask"
    touch_complete_sources(none_root, "none", "P04")
    touch_complete_sources(gaussian_mask_root, "gaussian_mask", "P04")
    args = Namespace(
        subjects="auto",
        none_ngm_root=none_root,
        gaussian_mask_ngm_root=gaussian_mask_root,
    )
    assert launcher.resolve_subjects(args) == ("P04",)


def test_launcher_builds_30_private_jobs_for_one_subject() -> None:
    args = Namespace(
        python="python",
        data_dir=Path("/data/private/processed_NBM_Exp"),
        output_root=Path("/runs/private_robustness_tcn"),
        num_workers=2,
        tcn_max_epochs=5,
        tcn_patience=2,
        overwrite=False,
    )
    jobs = launcher.jobs(args, ("P01",))
    assert len(jobs) == 30
    for index in range(0, len(jobs), 2):
        first, second = jobs[index : index + 2]
        first_arm = first["command"][first["command"].index("--arm") + 1]
        second_arm = second["command"][second["command"].index("--arm") + 1]
        assert (first_arm, second_arm) == worker.ARMS
        assert first["command"][first["command"].index("--subject") + 1] == "P01"
        assert second["command"][second["command"].index("--subject") + 1] == "P01"
        assert first["command"][first["command"].index("--fold") + 1] == second[
            "command"
        ][second["command"].index("--fold") + 1]
        assert first["command"][first["command"].index("--seed") + 1] == second[
            "command"
        ][second["command"].index("--seed") + 1]


def test_private_scheme_c_is_90_channel_and_temporally_centered() -> None:
    rng = np.random.default_rng(8)
    raw = rng.normal(size=(4, 128, 30)).astype(np.float32)
    scaler = worker.base.RobustScaler(
        median=np.zeros(30, dtype=np.float32),
        iqr=np.ones(30, dtype=np.float32),
        epsilon=1e-6,
    )
    model = worker.base.GRUReconstructionNBM(
        channels=30,
        hidden=worker.base.HIDDEN,
        bottleneck=worker.base.BOTTLENECK,
    )
    features = worker.base.scheme_c_features(
        model,
        scaler,
        np.ones(30, dtype=np.float32),
        raw,
        torch.device("cpu"),
        2,
    )
    assert features.shape == (4, 90, 128)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features[:, :30, :].mean(axis=2), 0.0, atol=2e-6)


def test_private_tcn_training_settings_are_frozen() -> None:
    launcher.validate_settings(Namespace(tcn_max_epochs=5, tcn_patience=2))
    assert worker.TCN_BATCH_SIZE == 128
    assert worker.TCN_MAX_EPOCHS == 5
    assert worker.TCN_PATIENCE == 2
