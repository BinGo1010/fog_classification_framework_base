from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from scripts import (
    launch_all_dataset_processed_nbm_exp_gru_r_delta_tcn_ep50pat10_7gpu
    as launcher,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_args() -> Namespace:
    return Namespace(
        data_dir=REPO_ROOT / "dataset" / "All_dataset" / "processed_NBM_Exp",
        source_root=REPO_ROOT / "outputs" / launcher.SOURCE_EXPERIMENT,
        output_root=REPO_ROOT / "outputs" / "unused_r_delta_ep50pat10_test",
        gpu_ids="0,1,2,3,4,5,6",
        seeds=",".join(map(str, launcher.shared.SEEDS)),
        python="python",
        phase="full",
        num_workers=0,
        batch_size=128,
        tcn_max_epochs=50,
        tcn_patience=10,
        overwrite=False,
        dry_run=True,
        skip_runtime_init_preflight=True,
    )


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_fixed_contract_and_complete_job_grid() -> None:
    args = make_args()
    seeds, gpu_ids = launcher.validate_contract(args)
    assert seeds == (0, 52, 161, 5216, 52161)
    assert gpu_ids == ["0", "1", "2", "3", "4", "5", "6"]
    train_jobs = launcher.shared.jobs(args, seeds, "train")
    evaluate_jobs = launcher.shared.jobs(args, seeds, "evaluate")
    assert len(train_jobs) == 8 * 3 * 5 == 120
    assert len(evaluate_jobs) == 120
    assert train_jobs[0]["id"] == "P01_fold0_seed0"
    assert train_jobs[-1]["id"] == "P08_fold2_seed52161"
    for job in (*train_jobs, *evaluate_jobs):
        command = job["command"]
        assert option_value(command, "--tcn-max-epochs") == "50"
        assert option_value(command, "--tcn-patience") == "10"
        assert option_value(command, "--batch-size") == "128"


def test_contract_rejects_training_budget_or_gpu_drift() -> None:
    args = make_args()
    args.tcn_max_epochs = 49
    with pytest.raises(ValueError, match="max_epoch=50"):
        launcher.validate_contract(args)
    args = make_args()
    args.tcn_patience = 9
    with pytest.raises(ValueError, match="patience=10"):
        launcher.validate_contract(args)
    args = make_args()
    args.gpu_ids = "0,1,2,3,4,5"
    with pytest.raises(ValueError, match="seven unique"):
        launcher.validate_contract(args)


def test_representation_and_latest_event_contract_are_unchanged() -> None:
    worker = launcher.shared
    assert worker.REPRESENTATION == "r_delta"
    assert worker.TCN_INPUT_CHANNELS == 60
    assert worker.EVENT_METRIC_VERSION == "allocation_group_any_window_nonfog_runs.v1"
