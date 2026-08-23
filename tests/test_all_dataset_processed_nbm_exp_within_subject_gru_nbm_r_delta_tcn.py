from __future__ import annotations

import csv
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from scripts import launch_all_dataset_processed_nbm_exp_within_subject_gru_nbm_r_delta_tcn_8gpu as launch
from scripts import run_all_dataset_processed_nbm_exp_within_subject_gru_nbm_r_delta_tcn as worker
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


def test_r_delta_feature_contract() -> None:
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(4, 128, 30)).astype(np.float32)
    scaler = RobustScaler(
        np.zeros(30, dtype=np.float32), np.ones(30, dtype=np.float32)
    )
    sigma = np.ones(30, dtype=np.float32)
    values = worker.r_delta_features(
        ZeroReconstructor(), scaler, sigma, raw, torch.device("cpu"), 4
    )
    assert values.shape == (4, 60, 128)
    residual = values[:, :30]
    delta = values[:, 30:]
    expected_delta = np.diff(
        residual, axis=2, prepend=residual[:, :, :1]
    )
    np.testing.assert_array_equal(delta, expected_delta)
    assert np.all(delta[:, :, 0] == 0.0)
    assert float(np.max(np.abs(residual.mean(axis=2)))) < 5e-5


def test_paired_initialization_selects_r_and_delta_channels() -> None:
    set_seed(52)
    reference = RepresentationTCNM(90)
    reference_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in reference.state_dict().items()
    }
    reference_hash = worker.base.expanded.state_dict_sha256(reference_state)
    target, hashes = worker.paired_r_delta_tcn(
        52, reference_hash, torch.device("cpu")
    )
    selected = torch.cat((torch.arange(30), torch.arange(60, 90)))
    for name, target_tensor in target.state_dict().items():
        source_tensor = reference_state[name]
        if target_tensor.shape == source_tensor.shape:
            assert torch.equal(target_tensor.cpu(), source_tensor)
        else:
            assert target_tensor.shape[1] == 60
            assert source_tensor.shape[1] == 90
            assert torch.equal(
                target_tensor.cpu(), source_tensor.index_select(1, selected)
            )
    assert hashes["reference_90ch_initial_state_sha256"] == reference_hash
    assert hashes["selected_reference_input_channels"] == "0:30 and 60:90"
    assert sum(parameter.numel() for parameter in target.parameters()) == 139_809


def test_final_event_and_false_alarm_rules(tmp_path: Path) -> None:
    manifest = tmp_path / "nbm_window_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "window_id",
                "active_for_outer_fold",
                "role_code",
                "allocation_group_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "window_id": "fog-a",
                    "active_for_outer_fold": "True",
                    "role_code": "1",
                    "allocation_group_id": "event-1",
                },
                {
                    "window_id": "fog-b",
                    "active_for_outer_fold": "True",
                    "role_code": "1",
                    "allocation_group_id": "event-1",
                },
            ]
        )
    record = SimpleNamespace(
        y=np.zeros(320, dtype=np.int8),
        valid=np.ones(320, dtype=bool),
    )
    dataset = SimpleNamespace(
        root=tmp_path,
        sampling_rate_hz=64,
        records=[record],
    )
    rows = SimpleNamespace(
        window_id=np.asarray(["fog-a", "fog-b", "nf-a", "nf-b", "nf-c", "nf-d"]),
        record_index=np.zeros(6, dtype=np.int32),
        start=np.asarray([0, 64, 0, 64, 128, 192]),
        end=np.asarray([128, 192, 128, 192, 256, 320]),
        role=np.asarray([1, 1, 0, 0, 0, 0], dtype=np.int8),
        label=np.asarray([1, 1, 0, 0, 0, 0], dtype=np.int8),
    )
    # One FoG window detects event-1.  Non-FoG positives at starts 0/64 merge;
    # the positive at 192 starts a second false-alarm run (>1 s gap).
    prediction = np.asarray([0, 1, 1, 1, 0, 1], dtype=np.int8)
    metrics = worker.final_event_metrics(dataset, rows, prediction)
    assert metrics["evaluable_true_events"] == 1
    assert metrics["detected_true_events"] == 1
    assert metrics["event_sensitivity"] == 1.0
    assert metrics["false_alarm_events"] == 2
    assert metrics["evaluated_nonfog_hours"] == 5.0 / 3600.0
    assert metrics["false_alarm_events_per_hour"] == 1440.0


def test_launcher_grid_and_identity() -> None:
    args = Namespace(
        data_dir=DATA,
        source_root=SOURCE,
        output_root=REPO_ROOT / "outputs" / "unused_r_delta_test_root",
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
    assert train_jobs[0]["id"] == "P01_fold0_seed0"
    assert train_jobs[-1]["id"] == "P08_fold2_seed52161"
    assert all(str(launch.WORKER) in job["command"] for job in train_jobs)
    assert worker.REPRESENTATION == "r_delta"
    assert worker.TCN_INPUT_CHANNELS == 60
    assert worker.EVENT_MINIMUM_POSITIVE_WINDOWS == 1
    assert worker.EVENT_MERGE_GAP_SECONDS == 1.0


def test_training_contract_excludes_absolute_residual() -> None:
    args = Namespace(batch_size=128, tcn_max_epochs=5, tcn_patience=2)
    contract = worker.training_contract(args)
    assert contract["input_shape"] == ["B", 60, 128]
    assert contract["input"] == "concatenate [r,delta(r)] along channels; abs(r) absent"
    assert contract["source_trainable_parameters_updated"] is False
    assert worker.EVENT_AGGREGATION == "pooled_counts_and_exposure"
