from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from scripts import launch_tcn_ngm40k_two_datasets_8gpu as dual


def dual_args(tmp_path: Path) -> Namespace:
    return Namespace(
        daphnet_data_dir=tmp_path / "daphnet",
        private_data_dir=tmp_path / "private",
        output_root=tmp_path / "output",
        gpu_ids="0,1,2,3,4,5,6,7",
        seeds=dual.SEED_TEXT,
        python="python",
        num_workers=0,
        batch_size=128,
        nbm_max_epochs=300,
        nbm_patience=20,
        tcn_max_epochs=5,
        tcn_patience=2,
        phase="full",
        overwrite=False,
        dry_run=True,
    )


def test_exact_eight_gpu_and_job_contract(tmp_path: Path) -> None:
    args = dual_args(tmp_path)
    private_launcher, daphnet_launcher = dual.configure_launchers()
    private_ns = dual.private_args(args, args.output_root / "private")
    daphnet_ns = dual.daphnet_args(args, args.output_root / "daphnet")

    private_seeds, private_gpus = private_launcher.validate_contract(private_ns)
    daphnet_seeds = daphnet_launcher.validate_contract(daphnet_ns)
    daphnet_gpus = daphnet_launcher.validate_gpus(args.gpu_ids, False)

    assert private_seeds == dual.SEEDS == daphnet_seeds
    assert private_gpus == daphnet_gpus == [str(index) for index in range(8)]
    assert len(private_launcher.jobs(private_ns, dual.SEEDS, "train")) == 120
    assert len(private_launcher.jobs(private_ns, dual.SEEDS, "evaluate")) == 120

    daphnet_nbm_jobs = [
        daphnet_launcher.nbm_command(daphnet_ns, fold, seed)
        for fold in dual.FOLDS
        for seed in dual.SEEDS
    ]
    daphnet_train_jobs = [
        daphnet_launcher.pair_command(daphnet_ns, "train", fold, "FULL_C", seed)
        for fold in dual.FOLDS
        for seed in dual.SEEDS
    ]
    assert len(daphnet_nbm_jobs) == 15
    assert len(daphnet_train_jobs) == 15
    assert "--nbm-kind" in daphnet_train_jobs[0]
    assert daphnet_train_jobs[0][daphnet_train_jobs[0].index("--nbm-kind") + 1] == "tcn_40k"


def test_seed_contract_rejects_any_other_list() -> None:
    assert dual.parse_seeds(dual.SEED_TEXT) == dual.SEEDS
    with pytest.raises(ValueError, match="exact seeds"):
        dual.parse_seeds("0,52,161")


def test_private_job_ids_are_namespaced() -> None:
    jobs = dual.prefixed_jobs(
        "private",
        [{"id": "P01_fold0_seed0", "command": ["python", "worker.py"]}],
    )
    assert jobs == [
        {
            "id": "private_P01_fold0_seed0",
            "command": ["python", "worker.py"],
        }
    ]
