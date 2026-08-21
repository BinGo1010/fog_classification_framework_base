from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cnbr_fog.data import DaphnetDataset
from scripts import launch_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn_8gpu as launch
from scripts import run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn as worker
from scripts import run_all_dataset_processed_nbm_exp_within_subject_raw_tcn as raw_base
from scripts.run_daphnet_processed_nbm_conv_tcn_autoencoder_fold import RepresentationTCNM
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import GRUReconstructionNBM


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "dataset" / "All_dataset" / "processed_NBM_Exp"


def test_architecture_parameter_and_shape_contract() -> None:
    nbm = GRUReconstructionNBM(channels=30, hidden=64, bottleneck=16)
    tcn = RepresentationTCNM(90)
    assert sum(parameter.numel() for parameter in nbm.parameters()) == 40_942
    assert sum(parameter.numel() for parameter in tcn.parameters()) == 143_649
    with torch.no_grad():
        reconstruction = nbm(torch.zeros(2, 128, 30))
        logit = tcn(torch.zeros(2, 90, 128))
    assert tuple(reconstruction.shape) == (2, 128, 30)
    assert tuple(logit.shape) == (2,)


def test_base_augmentation_contract_and_mask_bounds() -> None:
    clean = torch.ones(256, 128, 30)
    generator = torch.Generator().manual_seed(1000)
    corrupted, counts = worker.corrupt_gru_base(clean, generator)
    assert int(counts.sum()) == len(clean)
    assert all(int(value) > 0 for value in counts)
    masked_lengths = []
    for row in corrupted:
        zero_time = torch.all(row == 0.0, dim=1).cpu().numpy()
        if np.any(zero_time):
            indices = np.flatnonzero(zero_time)
            assert np.all(np.diff(indices) == 1)
            masked_lengths.append(len(indices))
    assert len(masked_lengths) == int(counts[2])
    assert min(masked_lengths) >= 4
    assert max(masked_lengths) <= 8


class ZeroReconstructor(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def test_scheme_c_is_90_channels_and_axis_centered() -> None:
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(4, 128, 30)).astype(np.float32)
    scaler = worker.RobustScaler(np.zeros(30, dtype=np.float32), np.ones(30, dtype=np.float32))
    sigma = np.ones(30, dtype=np.float32)
    features = worker.scheme_c_features(
        ZeroReconstructor(), scaler, sigma, raw, torch.device("cpu"), 4
    )
    assert features.shape == (4, 90, 128)
    assert float(np.max(np.abs(features[:, :30].mean(axis=2)))) < 5e-5
    assert np.allclose(features[:, 60:90, 0], 0.0)


def test_real_data_subject_role_isolation_and_shapes() -> None:
    dataset = DaphnetDataset.load(DATA)
    rows = raw_base.load_subject_rows(DATA, dataset, "P01", 0)
    assert set(rows.role.tolist()) == set(range(8))
    scaler, count = raw_base.fit_scaler_unique_role4_points(dataset, rows.take_role(4))
    values = worker.centered_scaled_ntc(scaler, raw_base.raw_windows(dataset, rows.take_role(4)))
    assert count > 0
    assert values.shape[1:] == (128, 30)
    assert np.all(rows.take_role(4).label == 0)
    assert np.all(rows.take_role(5).label == 0)
    assert np.all(rows.take_role(6).label == 0)
    assert np.all(rows.take_role(7).label == 1)


def test_launcher_has_120_train_and_120_evaluate_jobs(tmp_path: Path) -> None:
    args = Namespace(
        data_dir=DATA,
        output_root=tmp_path,
        python="python",
        num_workers=0,
        batch_size=128,
        nbm_max_epochs=300,
        nbm_patience=20,
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


def test_evaluate_fails_closed_without_global_barrier(tmp_path: Path) -> None:
    try:
        worker.load_and_validate_barrier(tmp_path, "P01", 0, 0)
    except FileNotFoundError as error:
        assert "locked" in str(error)
    else:
        raise AssertionError("test roles must remain inaccessible before the barrier")


def test_six_metric_contract() -> None:
    assert worker.METRIC_KEYS == (
        "sensitivity",
        "precision",
        "specificity",
        "pr_auc",
        "event_sensitivity",
        "false_alarms_per_hour",
    )
