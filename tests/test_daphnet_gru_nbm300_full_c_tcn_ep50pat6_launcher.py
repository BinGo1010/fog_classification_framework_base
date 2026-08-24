from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts import launch_daphnet_gru_nbm300_full_c_tcn_ep50pat6_7gpu as launcher


REPO_ROOT = Path(__file__).resolve().parents[1]


def args_for_commands() -> argparse.Namespace:
    return argparse.Namespace(
        python=sys.executable,
        data_dir=REPO_ROOT
        / "dataset"
        / "1.Daphnet Freezing of Gait Dataset"
        / "processed_NBM",
        output_root=REPO_ROOT / "outputs" / "dry_gru_tcn50",
        reuse_nbm_source_root=None,
        nbm_seeds=launcher.SEED_TEXT,
        tcn_seeds=launcher.SEED_TEXT,
        experiment_methods="FULL_C",
        num_workers=0,
        nbm_max_epochs=300,
        nbm_patience=20,
        tcn_max_epochs=50,
        tcn_patience=6,
        overwrite=False,
    )


def value_after(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_contract_and_exact_job_commands() -> None:
    args = args_for_commands()
    assert launcher.validate_contract(args) == launcher.REQUIRED_SEEDS
    nbm = [
        launcher.shared.nbm_command(args, fold, seed)
        for fold in launcher.FOLDS
        for seed in launcher.REQUIRED_SEEDS
    ]
    train = [
        launcher.pair_command(args, "train", fold, seed)
        for fold in launcher.FOLDS
        for seed in launcher.REQUIRED_SEEDS
    ]
    evaluate = [
        launcher.pair_command(args, "evaluate", fold, seed)
        for fold in launcher.FOLDS
        for seed in launcher.REQUIRED_SEEDS
    ]
    assert len(nbm) == len(train) == len(evaluate) == 15
    for command in nbm:
        assert value_after(command, "--nbm-max-epochs") == "300"
        assert value_after(command, "--nbm-patience") == "20"
        assert value_after(command, "--nbm-hidden") == "64"
        assert value_after(command, "--nbm-bottleneck") == "16"
    for command in (*train, *evaluate):
        assert value_after(command, "--nbm-kind") == "gru"
        assert value_after(command, "--experiment-methods") == "FULL_C"
        assert value_after(command, "--tcn-max-epochs") == "50"
        assert value_after(command, "--tcn-patience") == "6"


def test_dry_run_plan_is_complete_seven_gpu_experiment() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts"
                / "launch_daphnet_gru_nbm300_full_c_tcn_ep50pat6_7gpu.py"
            ),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["gpu_ids"] == ["0", "1", "2", "3", "4", "5", "6"]
    assert plan["seeds"] == [0, 52, 161, 5216, 52161]
    assert plan["methods"] == ["FULL_C"]
    assert plan["nbm_jobs"] == 15
    assert plan["classifier_train_jobs"] == 15
    assert plan["post_barrier_test_jobs"] == 15
    assert "max50/pat6" in plan["classifier_training"]
