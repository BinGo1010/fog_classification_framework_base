from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import check_processed_pipeline
from scripts.collect_fog_results import collect_one
from scripts.audit_fog_suite_results import audit_suite_results
from scripts.preflight_fog_suite import check_suite
from scripts.run_fog_experiment import build_windowing_command
from scripts.run_fog_suite import (
    concrete_experiment_stage_commands,
    existing_windows_match,
    filter_experiments,
    filter_status_report,
    suite_training_complete,
    suite_training_status_by_config,
    training_status_lines,
    training_status_payload,
)
from scripts.run_sleepyco_fog_two_stage import compute_metrics as sleepyco_metrics
from scripts.run_tcn_loso_npz import compute_metrics as tcn_metrics
from scripts.run_tcn_loso_npz import parse_folds as parse_tcn_folds


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def write_minimal_window_dataset(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.savez(
        path / "windows.npz",
        X=np.zeros((2, 1, 3), dtype=np.float32),
        class_names=np.array(["NORMAL", "FOG"]),
    )
    np.savez(path / "loso_folds.npz", fold_test_subjects=np.array(["S01"]))
    write_json(
        path / "config.json",
        {
            "class_names": ["NORMAL", "FOG"],
            "feature_names": ["AccV", "AccML", "AccAP"],
            "window_seconds": 0.01,
            "stride_seconds": 0.01,
            "target_hz": 100.0,
            "target_len": 1,
            "label_mode": "binary",
        },
    )


def test_binary_metrics_compute_auc_for_sleepyco_and_tcn() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_prob = np.array(
        [
            [0.90, 0.10],
            [0.70, 0.30],
            [0.25, 0.75],
            [0.10, 0.90],
        ],
        dtype=np.float32,
    )
    class_names = np.array(["NORMAL", "FOG"])

    sleepyco = sleepyco_metrics(y_true, y_prob, class_names)
    tcn = tcn_metrics(y_true, y_prob, class_names)

    for metrics in (sleepyco, tcn):
        assert metrics["roc_auc_ovr_macro"] == 1.0
        assert metrics["pr_auc_macro"] == 1.0
        assert "fog_f1" in metrics
        assert "pre_fog_f1" not in metrics


def test_tcn_parse_folds_accepts_all() -> None:
    assert parse_tcn_folds("all", 3) == [0, 1, 2]
    assert parse_tcn_folds("ALL", 2) == [0, 1]


def test_collect_results_supports_sleepyco_and_tcn_layouts(tmp_path: Path) -> None:
    data_dir = tmp_path / "windows"
    write_json(
        data_dir / "config.json",
        {
            "class_names": ["NORMAL", "FOG"],
            "feature_names": [f"ch{i}" for i in range(24)],
            "window_seconds": 1.0,
            "stride_seconds": 1.0,
            "target_hz": 100.0,
            "target_len": 100,
            "label_mode": "binary",
        },
    )

    sleepy_root = tmp_path / "outputs" / "sleepy"
    sleepy_variant = sleepy_root / "seq2one_gru"
    write_json(sleepy_root / "config.json", {"data_dir": str(data_dir)})
    write_json(
        sleepy_variant / "aggregate.json",
        {"f1_macro": {"mean": 0.25, "std": 0.0, "min": 0.25, "max": 0.25}},
    )
    (sleepy_variant / "summary.csv").write_text("fold,test_f1_macro\n0,0.25\n", encoding="utf-8")

    tcn_root = tmp_path / "outputs" / "tcn"
    write_json(tcn_root / "config.json", {"data_dir": str(data_dir), "input_channels": 24})
    write_json(
        tcn_root / "aggregate.json",
        {
            "folds": [{"fold": 0}],
            "aggregate": {"f1_macro": {"mean": 0.50, "std": 0.0, "min": 0.50, "max": 0.50}},
            "elapsed_sec": 1.25,
        },
    )
    (tcn_root / "summary.csv").write_text("fold,test_f1_macro\n0,0.50\n", encoding="utf-8")

    sleepy_row = collect_one(sleepy_variant / "aggregate.json")
    tcn_row = collect_one(tcn_root / "aggregate.json")

    assert sleepy_row["trainer"] == "sleepyco"
    assert sleepy_row["variant"] == "seq2one_gru"
    assert sleepy_row["input_channels"] == 24
    assert sleepy_row["f1_macro_mean"] == 0.25

    assert tcn_row["trainer"] == "tcn"
    assert tcn_row["fold_count"] == 1
    assert tcn_row["elapsed_sec"] == 1.25
    assert tcn_row["f1_macro_mean"] == 0.50


def test_existing_window_reuse_requires_matching_config(tmp_path: Path) -> None:
    output_dir = tmp_path / "windows"
    output_dir.mkdir()
    (output_dir / "windows.npz").write_bytes(b"placeholder")
    (output_dir / "loso_folds.npz").write_bytes(b"placeholder")
    write_json(
        output_dir / "config.json",
        {
            "processed_dir": str(tmp_path / "processed"),
            "label_mode": "binary",
            "label_rule": "priority",
            "window_seconds": 1.0,
            "stride_seconds": 1.0,
            "target_hz": 100.0,
            "nan_policy": "error",
            "max_records": 12,
            "num_folds": 0,
            "fold_seed": 42,
        },
    )

    config = {
        "windowing": {
            "processed_dir": str(tmp_path / "processed"),
            "output_dir": str(output_dir),
            "label_mode": "binary",
            "label_rule": "priority",
            "window_seconds": 1,
            "stride_seconds": 1,
            "target_hz": 100,
            "nan_policy": "error",
            "max_records": 12,
            "num_folds": 0,
            "fold_seed": 42,
        }
    }

    reusable, reason = existing_windows_match(config)
    assert reusable, reason

    config["windowing"]["label_mode"] = "three-class"
    reusable, reason = existing_windows_match(config)
    assert not reusable
    assert "label_mode mismatch" in reason

    config["windowing"]["label_mode"] = "binary"
    config["windowing"]["nan_policy"] = "zero"
    reusable, reason = existing_windows_match(config)
    assert not reusable
    assert "nan_policy mismatch" in reason


def test_windowing_command_passes_nan_policy(tmp_path: Path) -> None:
    command = build_windowing_command(
        sys.executable,
        {
            "processed_dir": str(tmp_path / "processed"),
            "output_dir": str(tmp_path / "windows"),
            "window_seconds": 1,
            "label_mode": "binary",
            "nan_policy": "zero",
        },
    )

    assert "--nan-policy" in command
    assert command[command.index("--nan-policy") + 1] == "zero"


def test_windowing_command_passes_require_success(tmp_path: Path) -> None:
    command = build_windowing_command(
        sys.executable,
        {
            "processed_dir": str(tmp_path / "processed"),
            "output_dir": str(tmp_path / "windows"),
            "window_seconds": 1,
            "label_mode": "binary",
            "require_success": True,
        },
    )

    assert "--require-success" in command


def test_prepare_windows_require_success_fails_before_output(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    output_dir = tmp_path / "windows"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "prepare_processed_record_windows.py"),
            "--processed-dir",
            str(processed_dir),
            "--output-dir",
            str(output_dir),
            "--window-seconds",
            "1",
            "--require-success",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "_SUCCESS.json" in result.stderr
    assert not output_dir.exists()


def test_check_processed_pipeline_passes_require_success_to_window_dry_run(tmp_path: Path, monkeypatch) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    captured_window_commands: list[list[str]] = []

    def fake_run_step(name: str, cmd: list[str], cwd: Path) -> None:
        if name == "Window dry-run":
            captured_window_commands.append(cmd)
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "file_summary.csv").write_text("record_id,windows\n", encoding="utf-8")
            (output_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_processed_pipeline.py",
            "--processed-dir",
            str(processed_dir),
            "--repo-root",
            str(REPO_ROOT),
            "--require-success",
        ],
    )
    monkeypatch.setattr(check_processed_pipeline, "run_step", fake_run_step)

    check_processed_pipeline.main()

    assert len(captured_window_commands) == 1
    assert "--require-success" in captured_window_commands[0]


def suite_args(**overrides) -> argparse.Namespace:
    defaults = {
        "only": "all",
        "skip_windowing": False,
        "skip_validation": False,
        "skip_training": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_concrete_experiment_stage_commands_expand_all_stages(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.json"
    config = {
        "windowing": {
            "processed_dir": str(tmp_path / "processed"),
            "output_dir": str(tmp_path / "windows"),
            "window_seconds": 1,
            "label_mode": "binary",
            "nan_policy": "zero",
        },
        "validation": {
            "expected_channels": 3,
            "expected_classes": 2,
        },
        "training": {
            "script": "scripts/run_tcn_loso_npz.py",
            "args": {
                "output_dir": str(tmp_path / "outputs"),
                "folds": "0",
                "epochs": 1,
            },
        },
    }

    commands = concrete_experiment_stage_commands(sys.executable, config_path, config, suite_args())
    stages = [stage for stage, _ in commands]

    assert stages == ["windowing", "validation", "training"]
    assert "--nan-policy" in commands[0][1]
    assert str(tmp_path / "windows") in commands[1][1]
    assert "--data-dir" in commands[2][1]


def test_concrete_experiment_stage_commands_catches_bad_window_config(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_experiment.json"
    config = {
        "windowing": {
            "processed_dir": str(tmp_path / "processed"),
            "output_dir": str(tmp_path / "windows"),
        }
    }

    try:
        concrete_experiment_stage_commands(sys.executable, config_path, config, suite_args(only="windowing"))
    except ValueError as exc:
        assert "window_seconds" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected missing window_seconds to fail config validation")


def test_preflight_accepts_processed_config_json_without_schema_json(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    (processed_dir / "records").mkdir(parents=True)
    (processed_dir / "manifest.csv").write_text(
        "record_id,record_path,subject_id,sampling_rate,n_samples\n",
        encoding="utf-8",
    )
    write_json(processed_dir / "config.json", {"channels": ["AccV", "AccML", "AccAP"]})

    experiment_path = tmp_path / "experiment.json"
    write_json(
        experiment_path,
        {
            "name": "config_only_processed",
            "windowing": {
                "enabled": True,
                "processed_dir": str(processed_dir),
                "output_dir": str(tmp_path / "windows"),
                "window_seconds": 1,
                "label_mode": "binary",
            },
            "validation": {
                "enabled": False,
            },
            "training": {
                "enabled": False,
            },
        },
    )
    suite_path = tmp_path / "suite.json"
    write_json(suite_path, {"name": "config_only_suite", "experiments": [{"config": str(experiment_path)}]})

    report = check_suite(argparse.Namespace(config=suite_path, require_windows=False))

    assert report["errors"] == []
    assert report["unique_windows"][0]["exists"] is False


def test_preflight_can_allow_missing_processed_before_preprocessing(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    experiment_path = tmp_path / "experiment.json"
    write_json(
        experiment_path,
        {
            "name": "missing_processed_allowed",
            "windowing": {
                "enabled": True,
                "processed_dir": str(processed_dir),
                "output_dir": str(tmp_path / "windows"),
                "window_seconds": 1,
                "label_mode": "binary",
                "require_success": True,
            },
            "validation": {
                "enabled": False,
            },
            "training": {
                "enabled": False,
            },
        },
    )
    suite_path = tmp_path / "suite.json"
    write_json(suite_path, {"name": "missing_processed_suite", "experiments": [{"config": str(experiment_path)}]})

    strict_report = check_suite(
        argparse.Namespace(config=suite_path, require_windows=False, allow_missing_processed=False)
    )
    assert any(error["message"] == "Processed directory does not exist" for error in strict_report["errors"])

    allowed_report = check_suite(
        argparse.Namespace(config=suite_path, require_windows=False, allow_missing_processed=True)
    )
    assert allowed_report["errors"] == []
    assert any(
        warning["message"] == "Processed directory does not exist"
        for warning in allowed_report["warnings"]
    )


def test_preflight_can_require_processed_success_marker(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    (processed_dir / "records").mkdir(parents=True)
    (processed_dir / "manifest.csv").write_text(
        "record_id,record_path,subject_id,sampling_rate,n_samples\n",
        encoding="utf-8",
    )
    write_json(processed_dir / "config.json", {"channels": ["AccV", "AccML", "AccAP"]})

    experiment_path = tmp_path / "experiment.json"
    write_json(
        experiment_path,
        {
            "name": "requires_success",
            "windowing": {
                "enabled": True,
                "processed_dir": str(processed_dir),
                "output_dir": str(tmp_path / "windows"),
                "window_seconds": 1,
                "label_mode": "binary",
                "require_success": True,
            },
            "validation": {
                "enabled": False,
            },
            "training": {
                "enabled": False,
            },
        },
    )
    suite_path = tmp_path / "suite.json"
    write_json(suite_path, {"name": "requires_success_suite", "experiments": [{"config": str(experiment_path)}]})

    missing_report = check_suite(argparse.Namespace(config=suite_path, require_windows=False))
    assert any("_SUCCESS.json" in error["path"] for error in missing_report["errors"])

    write_json(processed_dir / "_SUCCESS.json", {"status": "complete"})
    complete_report = check_suite(argparse.Namespace(config=suite_path, require_windows=False))
    assert complete_report["errors"] == []


def test_audit_smoke_suite_results_are_complete() -> None:
    report = audit_suite_results(REPO_ROOT / "configs" / "multimodal_smoke_suite.json")

    assert report["ok"] is True
    assert report["expected_aggregates"] == 4
    assert report["found_aggregates"] == 4
    assert report["errors"] == []
    assert all(experiment["ok"] is True for experiment in report["experiments"])
    assert all(experiment["expected_aggregates"] == 1 for experiment in report["experiments"])
    assert all(experiment["found_aggregates"] == 1 for experiment in report["experiments"])


def test_suite_training_complete_uses_audit() -> None:
    complete, reason = suite_training_complete(REPO_ROOT / "configs" / "multimodal_smoke_suite.json")

    assert complete is True
    assert reason == "4/4 aggregate outputs complete"


def test_suite_training_status_by_config_reports_each_experiment() -> None:
    _, status = suite_training_status_by_config(REPO_ROOT / "configs" / "multimodal_smoke_suite.json")

    assert len(status) == 4
    for complete, reason in status.values():
        assert complete is True
        assert reason == "1/1 aggregate outputs complete"


def test_suite_training_status_distinguishes_partial_completion(tmp_path: Path) -> None:
    data_dir = tmp_path / "windows"
    write_minimal_window_dataset(data_dir)

    complete_output = tmp_path / "outputs" / "complete"
    write_json(complete_output / "config.json", {"data_dir": str(data_dir), "input_channels": 3})
    write_json(
        complete_output / "aggregate.json",
        {
            "folds": [{"fold": 0}],
            "aggregate": {"f1_macro": {"mean": 0.75, "std": 0.0, "min": 0.75, "max": 0.75}},
        },
    )
    (complete_output / "summary.csv").write_text("fold,test_f1_macro\n0,0.75\n", encoding="utf-8")

    missing_output = tmp_path / "outputs" / "missing"
    complete_config = tmp_path / "complete.json"
    missing_config = tmp_path / "missing.json"
    for config_path, output_dir in ((complete_config, complete_output), (missing_config, missing_output)):
        write_json(
            config_path,
            {
                "name": config_path.stem,
                "windowing": {
                    "enabled": True,
                    "processed_dir": str(tmp_path / "processed"),
                    "output_dir": str(data_dir),
                    "window_seconds": 0.01,
                    "label_mode": "binary",
                },
                "training": {
                    "enabled": True,
                    "script": "scripts/run_tcn_loso_npz.py",
                    "args": {
                        "output_dir": str(output_dir),
                        "folds": "0",
                    },
                },
            },
        )

    suite_path = tmp_path / "suite.json"
    write_json(
        suite_path,
        {
            "name": "partial_suite",
            "experiments": [{"config": str(complete_config)}, {"config": str(missing_config)}],
        },
    )

    report, status = suite_training_status_by_config(suite_path)

    assert report["ok"] is False
    assert report["expected_aggregates"] == 2
    assert report["found_aggregates"] == 1
    assert status[complete_config.resolve()] == (True, "1/1 aggregate outputs complete")
    assert status[missing_config.resolve()] == (False, "0/1 aggregate outputs complete")

    lines = training_status_lines(report)
    assert lines[0] == "[STATUS] suite=partial_suite state=incomplete aggregates=1/2"
    assert any("complete complete aggregates=1/1 folds=1" in line for line in lines)
    assert any("incomplete missing aggregates=0/1 folds=1 missing=missing" in line for line in lines)

    payload = training_status_payload(report)
    assert payload["state"] == "incomplete"
    assert payload["found_aggregates"] == 1
    assert payload["expected_aggregates"] == 2
    payload_by_name = {experiment["name"]: experiment for experiment in payload["experiments"]}
    assert payload_by_name["complete"]["ok"] is True
    assert payload_by_name["missing"]["missing_variants"] == ["missing"]


def test_experiment_filter_selects_tcn_subset() -> None:
    suite = json.loads((REPO_ROOT / "configs" / "multimodal_full_suite.json").read_text(encoding="utf-8"))
    paths = [(REPO_ROOT / entry["config"]).resolve() for entry in suite["experiments"]]
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]

    selected_paths, selected_configs = filter_experiments(paths, configs, include="tcn")

    assert len(selected_paths) == 2
    assert all("tcn" in config["name"] for config in selected_configs)


def test_filtered_status_report_counts_selected_experiments() -> None:
    suite = json.loads((REPO_ROOT / "configs" / "multimodal_full_suite.json").read_text(encoding="utf-8"))
    paths = [(REPO_ROOT / entry["config"]).resolve() for entry in suite["experiments"]]
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    selected_paths, _ = filter_experiments(paths, configs, include="tcn")
    report = {
        "suite": "synthetic",
        "experiments": [
            {
                "name": "multimodal_sleepyco_binary_win1_hz100",
                "config": str(paths[0]),
                "expected_aggregates": 2,
                "found_aggregates": 0,
                "ok": False,
            },
            {
                "name": "multimodal_tcn_binary_win1_hz100",
                "config": str(paths[2]),
                "expected_aggregates": 1,
                "found_aggregates": 1,
                "ok": True,
            },
            {
                "name": "multimodal_tcn_3class_win1_hz100_prefog3",
                "config": str(paths[3]),
                "expected_aggregates": 1,
                "found_aggregates": 0,
                "ok": False,
            },
        ],
        "errors": [
            {
                "message": "Missing aggregate.json",
                "experiment": "multimodal_sleepyco_binary_win1_hz100",
            },
            {
                "message": "Missing aggregate.json",
                "experiment": "multimodal_tcn_3class_win1_hz100_prefog3",
            },
        ],
    }

    filtered = filter_status_report(report, selected_paths)

    assert filtered["ok"] is False
    assert filtered["expected_aggregates"] == 2
    assert filtered["found_aggregates"] == 1
    assert len(filtered["errors"]) == 1
    assert [experiment["name"] for experiment in filtered["experiments"]] == [
        "multimodal_tcn_binary_win1_hz100",
        "multimodal_tcn_3class_win1_hz100_prefog3",
    ]
