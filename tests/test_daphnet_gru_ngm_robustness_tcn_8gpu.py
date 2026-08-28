from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from scripts import launch_daphnet_gru_ngm_robustness_tcn_8gpu as launcher
from scripts import train_daphnet_gru_ngm_robustness_tcn as worker


def make_source(root: Path, fold: int = 0, seed: int = 0) -> Path:
    directory = root / f"seed_{seed}" / f"fold_{fold}"
    checkpoint = directory / "checkpoints" / "gru_ngm_best.pt"
    checkpoint.parent.mkdir(parents=True)
    model = worker.GRUReconstructionNBM(channels=9, hidden=64, bottleneck=16)
    torch.save({"model_state": model.state_dict(), "seed": seed, "step": 123}, checkpoint)
    scaler = {
        "scaler": {
            "median": [0.0] * 9,
            "iqr": [1.0] * 9,
            "epsilon": 1e-6,
        },
        "fold": fold,
        "seed": seed,
    }
    (directory / "scaler_role4.json").write_text(
        json.dumps(scaler), encoding="utf-8"
    )
    return directory


def test_source_resolver_and_manifest_accept_standard_seed_fold_layout(tmp_path: Path) -> None:
    expected = make_source(tmp_path, fold=2, seed=52161)
    resolved = worker.resolve_source_fold_dir(tmp_path, fold=2, seed=52161)
    assert resolved == expected.resolve()
    manifest = worker.inspect_source_artifacts(tmp_path, fold=2, seed=52161)
    assert manifest["checkpoint_name"] == "gru_ngm_best.pt"
    assert manifest["checkpoint_seed"] == 52161
    assert manifest["checkpoint_step"] == 123
    assert manifest["parameter_count"] > 0
    assert manifest["scaler"]["median"] == [0.0] * 9
    assert len(manifest["checkpoint_sha256"]) == 64


def test_source_resolver_fails_on_missing_checkpoint(tmp_path: Path) -> None:
    try:
        worker.resolve_source_fold_dir(tmp_path, fold=0, seed=0)
    except FileNotFoundError as error:
        assert "no GRU-NGM checkpoint" in str(error)
    else:
        raise AssertionError("missing source checkpoint should fail closed")


def test_scheme_c_feature_shape_and_temporal_centering() -> None:
    rng = np.random.default_rng(8)
    raw = rng.normal(size=(5, 128, 9)).astype(np.float32)
    labels = np.asarray([0, 1, 0, 1, 0], dtype=np.int8)
    scaler = worker.RobustScaler(
        median=np.zeros(9, dtype=np.float32),
        iqr=np.ones(9, dtype=np.float32),
        epsilon=1e-6,
    )
    model = worker.GRUReconstructionNBM(channels=9, hidden=64, bottleneck=16)
    features, clip = worker.scheme_c_features(
        model,
        scaler,
        np.ones(9, dtype=np.float32),
        raw,
        labels,
        torch.device("cpu"),
    )
    assert features.shape == (5, 128, 27)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features[:, :, :9].mean(axis=1), 0.0, atol=2e-6)
    assert clip["applicable"] is True


def test_launcher_builds_30_jobs_with_adjacent_paired_arms() -> None:
    args = Namespace(
        python="python",
        data_dir=Path("/data/daphnet/processed_NBM"),
        output_root=Path("/runs/robustness_tcn"),
        num_workers=2,
        tcn_max_epochs=5,
        tcn_patience=2,
        overwrite=False,
    )
    jobs = launcher.jobs(args)
    assert len(jobs) == 30
    for index in range(0, len(jobs), 2):
        first, second = jobs[index : index + 2]
        first_arm = first["command"][first["command"].index("--arm") + 1]
        second_arm = second["command"][second["command"].index("--arm") + 1]
        assert (first_arm, second_arm) == worker.ARMS
        assert first["command"][first["command"].index("--fold") + 1] == second[
            "command"
        ][second["command"].index("--fold") + 1]
        assert first["command"][first["command"].index("--seed") + 1] == second[
            "command"
        ][second["command"].index("--seed") + 1]


def test_tcn_training_settings_are_frozen_to_previous_backend() -> None:
    args = Namespace(tcn_max_epochs=5, tcn_patience=2)
    launcher.validate_settings(args)
    assert worker.TCN_BATCH_SIZE == 128
    assert worker.TCN_MAX_EPOCHS == 5
    assert worker.TCN_PATIENCE == 2
