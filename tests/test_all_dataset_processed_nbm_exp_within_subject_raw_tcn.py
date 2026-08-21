from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from cnbr_fog.data import DaphnetDataset
from scripts import launch_all_dataset_processed_nbm_exp_within_subject_raw_tcn_8gpu as launch
from scripts import run_all_dataset_processed_nbm_exp_within_subject_raw_tcn as worker


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "dataset" / "All_dataset" / "processed_NBM_Exp"


def test_dataset_and_all_subject_fold_contracts() -> None:
    dataset = DaphnetDataset.load(DATA)
    assert dataset.sampling_rate_hz == 64
    assert dataset.n_channels == 30
    assert tuple(dataset.subjects) == worker.SUBJECTS
    for subject in worker.SUBJECTS:
        permanent = []
        for fold in worker.FOLDS:
            rows = worker.load_subject_rows(DATA, dataset, subject, fold)
            assert set(rows.role.tolist()) == set(worker.ROLES)
            assert np.array_equal(rows.label, np.isin(rows.role, (1, 3, 7)).astype(np.int8))
            permanent.append(set(rows.take_role(0, 1).window_id.tolist()))
        assert permanent[0] == permanent[1] == permanent[2]


def test_raw_scaler_and_centered_feature_shape() -> None:
    dataset = DaphnetDataset.load(DATA)
    rows = worker.load_subject_rows(DATA, dataset, "P01", 0)
    scaler, count = worker.fit_scaler_unique_role4_points(dataset, rows.take_role(4))
    features = worker.raw_features(scaler, worker.raw_windows(dataset, rows.take_role(6, 7)))
    assert count > 0
    assert features.shape[1:] == (30, 128)
    assert float(np.max(np.abs(features.mean(axis=2)))) < 5e-5


def test_tcn_shape_and_parameter_count() -> None:
    model = worker.RepresentationTCNM(30)
    with torch.no_grad():
        output = model(torch.zeros(2, 30, 128))
    assert tuple(output.shape) == (2,)
    assert sum(parameter.numel() for parameter in model.parameters()) == 135969


def test_threshold_rule_prefers_balanced_accuracy() -> None:
    y_true = np.asarray([0, 0, 1, 1], dtype=np.int8)
    y_prob = np.asarray([0.1, 0.4, 0.6, 0.9])
    threshold, metrics = worker.choose_threshold(y_true, y_prob)
    assert threshold == 0.6
    assert metrics["balanced_accuracy"] == 1.0


def test_event_metric_uses_nonfog_union_coverage() -> None:
    dataset = DaphnetDataset.load(DATA)
    rows = worker.load_subject_rows(DATA, dataset, "P01", 0).take_role(0, 1)
    predicted = np.zeros(len(rows), dtype=np.int8)
    metrics = worker.event_metrics(dataset, rows, predicted)
    assert metrics["event_metric_version"] == "coverage_aware.v2"
    assert metrics["evaluated_nonfog_hours"] > 0
    assert metrics["false_alarm_events"] == 0
    assert metrics["false_alarm_events_per_hour"] == 0.0


def test_launcher_grid_is_120_train_and_120_evaluate(tmp_path: Path) -> None:
    args = Namespace(
        data_dir=DATA,
        output_root=tmp_path,
        python="python",
        num_workers=0,
        batch_size=128,
        tcn_max_epochs=5,
        tcn_patience=2,
        overwrite=False,
    )
    train = launch.jobs(args, worker.SEEDS, "train")
    evaluate = launch.jobs(args, worker.SEEDS, "evaluate")
    assert len(train) == 8 * 3 * 5 == 120
    assert len(evaluate) == 120
    identities = {job["id"] for job in train}
    assert len(identities) == 120
    assert "P01_fold0_seed0" in identities
    assert "P08_fold2_seed52161" in identities


def test_evaluate_fails_closed_without_barrier(tmp_path: Path) -> None:
    try:
        worker.load_and_validate_barrier(tmp_path, "P01", 0, 0)
    except FileNotFoundError as error:
        assert "locked" in str(error)
    else:
        raise AssertionError("evaluation must fail closed without a barrier")


def test_metric_contract_contains_six_requested_metrics() -> None:
    assert worker.METRIC_KEYS == (
        "sensitivity",
        "precision",
        "specificity",
        "pr_auc",
        "event_sensitivity",
        "false_alarms_per_hour",
    )
