from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts import evaluate_daphnet_raw_tcn_test_noise_snr as worker
from scripts import launch_daphnet_raw_tcn_test_noise_snr_7gpu as launcher


REPO_ROOT = Path(__file__).resolve().parents[1]


def synthetic_raw(n: int = 128) -> np.ndarray:
    time = np.arange(128, dtype=np.float64) / 64.0
    axes = [
        (axis + 1.0) * np.sin(2.0 * np.pi * (0.5 + axis * 0.1) * time)
        + 100.0 * axis
        for axis in range(9)
    ]
    base = np.stack(axes, axis=1)
    return np.repeat(base[None, :, :], n, axis=0).astype(np.float32)


def test_noise_is_deterministic_and_model_seed_independent() -> None:
    raw = synthetic_raw(32)
    first, contract_a = worker.add_gaussian_noise_at_snr(raw, 20, 0)
    second, contract_b = worker.add_gaussian_noise_at_snr(raw, 20, 0)
    assert np.array_equal(first, second)
    assert contract_a == contract_b
    assert contract_a["seed_scope"].endswith("all five TCN model seeds")


def test_noise_realized_snr_is_close_to_requested_level() -> None:
    raw = synthetic_raw(512)
    for snr_db in worker.SNR_LEVELS:
        noisy, contract = worker.add_gaussian_noise_at_snr(raw, snr_db, 1)
        assert noisy.shape == raw.shape
        assert noisy.dtype == np.float32
        assert abs(contract["realized_pooled_snr_db"] - snr_db) < 0.15


def test_noise_is_added_before_scaler_and_does_not_change_clean_array() -> None:
    raw = synthetic_raw(8)
    original = raw.copy()
    noisy, contract = worker.add_gaussian_noise_at_snr(raw, 0, 2)
    assert np.array_equal(raw, original)
    assert not np.array_equal(noisy, raw)
    assert contract["injection_point"].startswith("raw window before")


def test_launcher_builds_sixty_evaluation_jobs_and_no_training_jobs() -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "launch_daphnet_raw_tcn_test_noise_snr_7gpu.py"),
        "--gpu-ids", "0,1,2,3,4,5,6,7",
        "--dry-run",
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "evaluation_jobs=60 training_jobs=0" in result.stdout
    assert result.stdout.count("--stage evaluate") == 60
    assert result.stdout.count("--stage aggregate") == 1


def test_launcher_commands_freeze_expected_source_and_threshold_policy() -> None:
    parser_args = type(
        "Args",
        (),
        {
            "python": sys.executable,
            "data_dir": REPO_ROOT / "data",
            "source_root": REPO_ROOT / "source",
            "scaler_source_root": REPO_ROOT / "scalers",
            "output_root": REPO_ROOT / "output",
            "batch_size": 128,
            "overwrite": False,
        },
    )()
    command = launcher.evaluate_command(parser_args, 2, 52161, 10)
    text = " ".join(command)
    assert "--stage evaluate" in text
    assert "--fold 2" in text
    assert "--seed 52161" in text
    assert "--snr-db 10" in text
    assert "--batch-size 128" in text
    assert "train" not in command


def test_noise_seed_changes_by_fold_and_snr() -> None:
    values = {
        worker.noise_seed(fold, snr)
        for fold in worker.FOLDS
        for snr in worker.SNR_LEVELS
    }
    assert len(values) == 12
