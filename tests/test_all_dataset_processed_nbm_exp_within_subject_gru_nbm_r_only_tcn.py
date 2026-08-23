from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.scientific_fingerprint import processed_nbm_scientific_manifest
from scripts import launch_all_dataset_processed_nbm_exp_within_subject_gru_nbm_r_only_tcn_8gpu as launch
from scripts import run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_r_only_tcn as worker
from scripts import run_all_dataset_processed_nbm_exp_within_subject_raw_tcn as raw_base
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import RepresentationTCNM
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import RobustScaler, set_seed


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "dataset" / "All_dataset" / "processed_NBM_Exp"
SOURCE = (
    REPO_ROOT
    / "outputs"
    / "all_dataset_processed_NBM_Exp_within_subject_gru_base_mask4_8_C_tcn_nbm300pat20_ep5pat2_seedset_0_52_161_5216_52161"
)


class ZeroReconstructor(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def test_r_only_feature_contract() -> None:
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(5, 128, 30)).astype(np.float32)
    scaler = RobustScaler(np.zeros(30, dtype=np.float32), np.ones(30, dtype=np.float32))
    sigma = np.ones(30, dtype=np.float32)
    feature = worker.r_only_features(
        ZeroReconstructor(), scaler, sigma, raw, torch.device("cpu"), 5
    )
    assert feature.shape == (5, 30, 128)
    assert float(np.max(np.abs(feature.mean(axis=2)))) < 5e-5


def test_paired_initialization_uses_reference_first_30_channels() -> None:
    set_seed(52)
    reference = RepresentationTCNM(90)
    reference_state = {
        name: tensor.detach().cpu().clone() for name, tensor in reference.state_dict().items()
    }
    reference_hash = worker.expanded.state_dict_sha256(reference_state)
    target, hashes = worker.paired_r_only_tcn(52, reference_hash, torch.device("cpu"))
    target_state = target.state_dict()
    for name, target_tensor in target_state.items():
        source_tensor = reference_state[name]
        if target_tensor.shape == source_tensor.shape:
            assert torch.equal(target_tensor.cpu(), source_tensor)
        else:
            assert target_tensor.shape[1] == 30
            assert source_tensor.shape[1] == 90
            assert torch.equal(target_tensor.cpu(), source_tensor[:, :30, :])
    assert hashes["reference_90ch_initial_state_sha256"] == reference_hash
    assert sum(parameter.numel() for parameter in target.parameters()) == 135_969


def test_source_artifacts_and_scientific_identity_are_valid() -> None:
    scientific = processed_nbm_scientific_manifest(DATA)["sha256"]
    source = worker.load_frozen_source(
        SOURCE, "P01", 0, 0, scientific, torch.device("cpu")
    )
    assert source["sigma"].shape == (30,)
    assert source["bundle"]["source_frozen_id"]
    assert source["bundle"]["source_reference_tcn_initial_state_sha256"]


def test_real_role67_and_role23_shapes() -> None:
    dataset = DaphnetDataset.load(DATA)
    rows = raw_base.load_subject_rows(DATA, dataset, "P01", 0)
    assert np.all(rows.take_role(6).label == 0)
    assert np.all(rows.take_role(7).label == 1)
    assert set(rows.take_role(2, 3).label.tolist()) == {0, 1}
    assert raw_base.raw_windows(dataset, rows.take_role(6, 7)).shape[1:] == (128, 30)


def test_launcher_grid_is_120_train_and_120_evaluate(tmp_path: Path) -> None:
    args = Namespace(
        data_dir=DATA,
        source_root=SOURCE,
        output_root=tmp_path,
        python="python",
        num_workers=0,
        batch_size=128,
        tcn_max_epochs=5,
        tcn_patience=2,
        overwrite=False,
    )
    train_jobs = launch.jobs(args, worker.SEEDS, "train")
    evaluate_jobs = launch.jobs(args, worker.SEEDS, "evaluate")
    assert len(train_jobs) == 8 * 3 * 5 == 120
    assert len(evaluate_jobs) == 120
    assert len({job["id"] for job in train_jobs}) == 120
    assert train_jobs[0]["id"] == "P01_fold0_seed0"
    assert train_jobs[-1]["id"] == "P08_fold2_seed52161"


def test_evaluate_fails_closed_without_barrier(tmp_path: Path) -> None:
    try:
        worker.load_and_validate_barrier(tmp_path, "P01", 0, 0)
    except FileNotFoundError as error:
        assert "locked" in str(error)
    else:
        raise AssertionError("roles0/1 must be inaccessible before global seal")


def test_only_r_is_kept_and_six_metrics_remain() -> None:
    assert worker.REPRESENTATION == "r_only"
    assert worker.TCN_INPUT_CHANNELS == 30
    assert worker.METRIC_KEYS == (
        "sensitivity",
        "precision",
        "specificity",
        "pr_auc",
        "event_sensitivity",
        "false_alarms_per_hour",
    )
