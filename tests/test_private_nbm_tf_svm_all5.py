from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_private_nbm_tf_svm_all5.py"
SPEC = importlib.util.spec_from_file_location("private_nbm_tf_svm_all5", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_feature_schema_is_330_finite_features() -> None:
    time = np.arange(128, dtype=np.float64) / 64.0
    windows = np.zeros((2, 128, 30), dtype=np.float64)
    windows[0] = np.sin(2.0 * np.pi * 1.5 * time)[:, None]
    windows[1] = np.sin(2.0 * np.pi * 5.0 * time)[:, None]

    features = MODULE.extract_tf_features(windows)

    assert features.shape == (2, 330)
    assert np.isfinite(features).all()
    freeze_index_offset = MODULE.FEATURES_PER_CHANNEL.index("log_freeze_index")
    assert features[1, freeze_index_offset] > features[0, freeze_index_offset]


def test_four_feature_private_schema_is_120_finite_features() -> None:
    rng = np.random.default_rng(31)
    windows = rng.normal(size=(2, 128, 30))

    features = MODULE.extract_tf_features(
        windows, feature_schema="tf120_all5_30ch_4f_v1"
    )
    names = MODULE.feature_names(
        [f"channel_{index}" for index in range(30)],
        "tf120_all5_30ch_4f_v1",
    )
    suffixes = {name.split("__", 1)[1] for name in names}

    assert features.shape == (2, 120)
    assert len(names) == 120
    assert suffixes == {
        "std",
        "peak_to_peak",
        "log_power_3_8hz",
        "log_power_8_28hz",
    }
    assert np.isfinite(features).all()


def test_four_feature_private_configuration_and_split_audit() -> None:
    config, project_root = MODULE.load_config(
        ROOT / "configs" / "private_nbm_tf_svm_all5_4f_gamma01_c10.yaml"
    )
    result = MODULE.audit_dataset(config, project_root)

    assert result["status"] == "PASS"
    assert result["feature_schema"] == "tf120_all5_30ch_4f_v1"
    assert result["feature_count"] == 120
    assert result["job_count"] == 24


def test_window_metrics_follow_positive_fog_contract() -> None:
    metrics = MODULE.compute_window_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.8, 0.7, 0.9]),
        threshold=0.6,
    )

    assert metrics["tp"] == 2
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
    assert metrics["sensitivity"] == 1.0
    assert metrics["precision"] == 2.0 / 3.0
    assert metrics["specificity"] == 0.5
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_event_and_false_alarm_episode_contract() -> None:
    predictions = pd.DataFrame(
        {
            "subject_id": ["P01"] * 4,
            "record_id": ["P01_seg000"] * 4,
            "allocation_group_id": ["normal_a", "normal_a", "normal_b", "fog_a"],
            "window_id": ["n0", "n1", "n2", "f0"],
            "role_code": [0, 0, 0, 1],
            "y_pred": [1, 1, 0, 1],
            "prob_fog": [0.8, 0.7, 0.1, 0.9],
            "start_time_sec": [0.0, 2.5, 10.0, 20.0],
            "end_time_sec": [2.0, 4.5, 12.0, 22.0],
        }
    )
    manifest = pd.DataFrame(
        {
            "subject_id": ["P01"],
            "record_id": ["P01_seg000"],
            "event_id": [0],
            "start_time_sec": [20.0],
            "end_time_sec": [22.0],
            "nbm_status": ["eligible"],
            "nbm_allocation_group_ids": ["fog_a"],
            "nbm_connector_window_ids": [""],
        }
    )

    metrics, details, alarms = MODULE.evaluate_events(
        predictions, manifest, merge_gap_sec=1.0
    )

    assert metrics["event_sensitivity"] == 1.0
    assert metrics["n_eligible_test_events"] == 1
    assert metrics["n_false_alarm_episodes"] == 1
    assert metrics["nonfog_exposure_hours"] == 6.0 / 3600.0
    assert metrics["false_alarms_per_hour"] == 600.0
    assert bool(details.loc[0, "detected"])
    assert len(alarms) == 1


def test_private_dataset_configuration_and_split_audit() -> None:
    config, project_root = MODULE.load_config(
        ROOT / "configs" / "private_nbm_tf_svm_all5.yaml"
    )
    result = MODULE.audit_dataset(config, project_root)

    assert result["status"] == "PASS"
    assert result["subjects"] == [f"P{index:02d}" for index in range(1, 9)]
    assert result["outer_folds"] == [0, 1, 2]
    assert result["channel_count"] == 30
    assert result["feature_count"] == 330
    assert result["job_count"] == 24
