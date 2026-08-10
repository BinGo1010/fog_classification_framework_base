from argparse import Namespace
from pathlib import Path

from scripts.launch_daphnet_transformer_nbm300_c_vs_raw_ep5pat2_7gpu import (
    FOLDS,
    METHODS,
    REQUIRED_SEEDS,
    nbm_command,
    pair_command,
    validate_contract,
)


def launcher_args() -> Namespace:
    return Namespace(
        python="python",
        data_dir=Path("processed_NBM"),
        output_root=Path("transformer_output"),
        num_workers=0,
        overwrite=False,
        nbm_seeds="0,52,161,5216,52161",
        tcn_seeds="0,52,161,5216,52161",
        nbm_max_epochs=300,
        nbm_patience=20,
        nbm_dropout=0.10,
        tcn_max_epochs=5,
        tcn_patience=2,
    )


def test_transformer_launcher_job_grid_and_contract() -> None:
    args = launcher_args()
    assert validate_contract(args) == REQUIRED_SEEDS
    nbm_specs = {(fold, seed) for fold in FOLDS for seed in REQUIRED_SEEDS}
    tcn_specs = {
        (fold, method, seed)
        for fold in FOLDS
        for method in METHODS
        for seed in REQUIRED_SEEDS
    }
    assert len(nbm_specs) == 15
    assert len(tcn_specs) == 30


def test_launcher_routes_transformer_nbm_and_tcn_5_patience_2() -> None:
    args = launcher_args()
    nbm = nbm_command(args, fold=0, seed=52161)
    assert "run_daphnet_transformer_nbm300_fold.py" in " ".join(nbm)
    assert nbm[nbm.index("--nbm-max-epochs") + 1] == "300"
    assert nbm[nbm.index("--nbm-patience") + 1] == "20"
    train = pair_command(args, "train", 0, "FULL_C", 52161)
    assert train[train.index("--nbm-kind") + 1] == "transformer"
    assert train[train.index("--tcn-max-epochs") + 1] == "5"
    assert train[train.index("--tcn-patience") + 1] == "2"
    assert train[train.index("--nbm-seed") + 1] == "52161"
    assert train[train.index("--tcn-seed") + 1] == "52161"

