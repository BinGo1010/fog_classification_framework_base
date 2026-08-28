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


def test_six_feature_private_schema_is_180_finite_features() -> None:
    rng = np.random.default_rng(41)
    windows = rng.normal(size=(2, 128, 30))

    features = MODULE.extract_tf_features(
        windows, feature_schema="tf180_all5_30ch_6f_v1"
    )
    names = MODULE.feature_names(
        [f"channel_{index}" for index in range(30)],
        "tf180_all5_30ch_6f_v1",
    )
    suffixes = {name.split("__", 1)[1] for name in names}

    assert features.shape == (2, 180)
    assert len(names) == 180
    assert suffixes == {
        "std",
        "peak_to_peak",
        "mean_abs",
        "log_power_3_8hz",
        "log_power_8_28hz",
        "log_freeze_index",
    }
    assert np.isfinite(features).all()


def test_six_feature_record_merge_configuration_and_split_audit() -> None:
    config, project_root = MODULE.load_config(
        ROOT
        / "configs"
        / "private_nbm_tf_svm_all5_6f_gamma01_c1_maxprecision_recordmerge.yaml"
    )
    result = MODULE.audit_dataset(config, project_root)

    assert result["status"] == "PASS"
    assert result["feature_schema"] == "tf180_all5_30ch_6f_v1"
    assert result["feature_count"] == 180
    assert result["job_count"] == 24


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


def test_target_sensitivity_threshold_uses_highest_feasible_threshold() -> None:
    selection = MODULE.target_sensitivity_threshold(
        np.array([0, 0, 0, 1, 1, 1, 1]),
        np.array([0.20, 0.45, 0.80, 0.30, 0.50, 0.70, 0.90]),
        target_sensitivity=0.75,
    )

    assert selection["threshold"] == 0.50
    assert selection["sensitivity"] == 0.75
    assert selection["specificity"] == 2.0 / 3.0
    assert selection["target_sensitivity"] == 0.75


def test_maximum_precision_threshold_prefers_higher_sensitivity_on_tie() -> None:
    selection = MODULE.maximum_precision_threshold(
        np.array([0, 0, 1, 1, 1]),
        np.array([0.10, 0.20, 0.30, 0.40, 0.90]),
    )

    assert selection["threshold"] == 0.30
    assert selection["precision"] == 1.0
    assert selection["sensitivity"] == 1.0
    assert selection["specificity"] == 1.0


def test_maximum_precision_configuration_and_split_audit() -> None:
    config, project_root = MODULE.load_config(
        ROOT
        / "configs"
        / "private_nbm_tf_svm_all5_4f_gamma01_c1_maxprecision.yaml"
    )
    result = MODULE.audit_dataset(config, project_root)

    assert result["status"] == "PASS"
    assert config["evaluation"]["threshold_rule"] == "validation_max_precision"
    assert result["job_count"] == 24


def test_sensitivity_target_configuration_and_split_audit() -> None:
    config, project_root = MODULE.load_config(
        ROOT
        / "configs"
        / "private_nbm_tf_svm_all5_4f_gamma01_c10_sens090.yaml"
    )
    result = MODULE.audit_dataset(config, project_root)

    assert result["status"] == "PASS"
    assert config["evaluation"]["threshold_rule"] == "validation_target_sensitivity"
    assert config["evaluation"]["target_sensitivity"] == 0.90
    assert result["job_count"] == 24


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


def test_record_level_false_alarm_merge_crosses_allocation_groups_only_within_record() -> None:
    predictions = pd.DataFrame(
        {
            "subject_id": ["P01"] * 4,
            "record_id": ["record_a", "record_a", "record_b", "record_a"],
            "allocation_group_id": ["normal_a", "normal_b", "normal_c", "fog_a"],
            "window_id": ["n0", "n1", "n2", "f0"],
            "role_code": [0, 0, 0, 1],
            "y_pred": [1, 1, 1, 1],
            "prob_fog": [0.8, 0.7, 0.6, 0.9],
            "start_time_sec": [0.0, 2.5, 2.5, 20.0],
            "end_time_sec": [2.0, 4.5, 4.5, 22.0],
        }
    )
    manifest = pd.DataFrame(
        {
            "subject_id": ["P01"],
            "record_id": ["record_a"],
            "event_id": [0],
            "start_time_sec": [20.0],
            "end_time_sec": [22.0],
            "nbm_status": ["eligible"],
            "nbm_allocation_group_ids": ["fog_a"],
            "nbm_connector_window_ids": [""],
        }
    )

    metrics, _, alarms = MODULE.evaluate_events(
        predictions,
        manifest,
        merge_gap_sec=1.0,
        false_alarm_merge_scope="record_id",
    )

    assert metrics["n_false_alarm_episodes"] == 2
    assert metrics["nonfog_exposure_hours"] == 6.0 / 3600.0
    assert metrics["false_alarms_per_hour"] == 1200.0
    assert len(alarms) == 2
    record_a = alarms.loc[alarms["record_id"] == "record_a"].iloc[0]
    assert record_a["positive_window_count"] == 2
    assert record_a["allocation_group_ids"] == "normal_a;normal_b"


def test_record_merge_configuration_and_split_audit() -> None:
    config, project_root = MODULE.load_config(
        ROOT
        / "configs"
        / "private_nbm_tf_svm_all5_4f_gamma01_c1_maxprecision_recordmerge.yaml"
    )
    result = MODULE.audit_dataset(config, project_root)

    assert result["status"] == "PASS"
    assert config["evaluation"]["event"]["false_alarm_merge_scope"] == "record_id"
    assert result["job_count"] == 24


def test_two_consecutive_positive_windows_confirm_events_and_false_alarms() -> None:
    predictions = pd.DataFrame(
        {
            "subject_id": ["P01"] * 7,
            "record_id": ["record_a"] * 7,
            "allocation_group_id": [
                "normal_a", "normal_a", "normal_b", "normal_b",
                "fog_a", "fog_a", "fog_b",
            ],
            "window_id": ["n0", "n1", "n2", "n3", "f0", "f1", "f2"],
            "role_code": [0, 0, 0, 0, 1, 1, 1],
            "y_pred": [1, 0, 1, 1, 1, 1, 1],
            "prob_fog": [0.8, 0.1, 0.7, 0.9, 0.8, 0.9, 0.8],
            "start_index": [0, 64, 128, 192, 640, 704, 1280],
            "start_time_sec": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 20.0],
            "end_time_sec": [2.0, 3.0, 4.0, 5.0, 12.0, 13.0, 22.0],
        }
    )
    manifest = pd.DataFrame(
        {
            "subject_id": ["P01", "P01"],
            "record_id": ["record_a", "record_a"],
            "event_id": [0, 1],
            "start_time_sec": [10.0, 20.0],
            "end_time_sec": [13.0, 22.0],
            "nbm_status": ["eligible", "eligible"],
            "nbm_allocation_group_ids": ["fog_a", "fog_b"],
            "nbm_connector_window_ids": ["", ""],
        }
    )

    metrics, details, alarms = MODULE.evaluate_events(
        predictions,
        manifest,
        merge_gap_sec=1.0,
        false_alarm_merge_scope="record_id",
        minimum_consecutive_positive_windows=2,
        consecutive_stride_samples=64,
    )

    assert metrics["event_sensitivity"] == 0.5
    assert metrics["n_false_alarm_episodes"] == 1
    assert metrics["nonfog_exposure_hours"] == 5.0 / 3600.0
    assert metrics["false_alarms_per_hour"] == 720.0
    assert details["detected"].tolist() == [True, False]
    assert alarms.iloc[0]["positive_window_count"] == 2


def test_two_window_configuration_and_split_audit() -> None:
    config, project_root = MODULE.load_config(
        ROOT
        / "configs"
        / "private_nbm_tf_svm_all5_11f_grid_maxf1_recordmerge_2win.yaml"
    )
    result = MODULE.audit_dataset(config, project_root)

    assert result["status"] == "PASS"
    assert config["evaluation"]["event"]["minimum_consecutive_positive_windows"] == 2
    assert result["job_count"] == 24


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
