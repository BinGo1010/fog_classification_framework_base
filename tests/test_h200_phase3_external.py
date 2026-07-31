from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cnbr_fog.h200_phase3 import PHASE3_ARMS
from cnbr_fog.h200_phase3_external import (
    EXTERNAL_SUBJECTS,
    ExternalCellResult,
    _aggregate_results,
    _build_external_protocol,
    _ensure_support_artifact,
    moment_match_external_forecasts,
    negative_only_metrics,
)
from cnbr_fog.resume import dataset_fingerprint


MAIN_SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
CHANNELS = tuple(f"imu_{index}" for index in range(9))


def _write_synthetic_processed(root: Path, samples: int = 800) -> None:
    records = root / "records"
    records.mkdir(parents=True)
    manifest = root / "manifest.csv"
    fieldnames = [
        "record_id",
        "record_path",
        "subject_id",
        "run_id",
        "sampling_rate_hz",
        "n_samples",
        "usable",
    ]
    rows = []
    time = np.arange(samples, dtype=np.float32)
    for position, subject in enumerate((*MAIN_SUBJECTS, *EXTERNAL_SUBJECTS)):
        # A non-flat, nonzero tri-axial signal keeps the label-independent
        # validity mask true across all raw6/H200 support.
        x = np.stack(
            [
                0.2 * np.sin(0.01 * (channel + 1) * time)
                + 1.0
                + 0.01 * position
                + 0.001 * channel
                for channel in range(9)
            ],
            axis=1,
        ).astype(np.float32)
        y = np.zeros(samples, dtype=np.int8)
        record_id = f"{subject}_seg000"
        relative = f"records/{record_id}.npz"
        np.savez_compressed(root / relative, x=x, y_binary=y)
        rows.append(
            {
                "record_id": record_id,
                "record_path": relative,
                "subject_id": subject,
                "run_id": "R01",
                "sampling_rate_hz": 64,
                "n_samples": samples,
                "usable": "True",
            }
        )
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (root / "schema.json").write_text(
        json.dumps(
            {
                "sampling_rate_hz": 64,
                "channels": [{"name": name} for name in CHANNELS],
            }
        ),
        encoding="utf-8",
    )


def _external_protocol(tmp_path: Path):
    data = tmp_path / "processed"
    source = tmp_path / "source"
    _write_synthetic_processed(data)
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "sampling_rate_hz": 64,
                "context_samples": 128,
                "support_horizon_samples": 128,
                "fixed_label_samples": 32,
                "stride_samples": 16,
                "flatline_seconds": 1.0,
                "zero_tolerance": 1e-8,
                "fog_fraction_threshold": 0.5,
                "normal_guard_samples": 32,
            }
        ),
        encoding="utf-8",
    )
    protocol = SimpleNamespace(
        config={
            "data_sha256": dataset_fingerprint(data),
            "subjects": list(MAIN_SUBJECTS),
            "channel_names": list(CHANNELS),
            "protocol_fingerprint": "main-protocol",
        }
    )
    args = SimpleNamespace(data_dir=data, source_suite_dir=source)
    return _build_external_protocol(args, protocol)


def test_external_protocol_rebuilds_terminal_labels_and_two_h200_blocks(
    tmp_path: Path,
) -> None:
    external = _external_protocol(tmp_path)
    assert set(external.dataset.subjects) == set((*MAIN_SUBJECTS, *EXTERNAL_SUBJECTS))
    for subject in EXTERNAL_SUBJECTS:
        support = external.support[subject]
        assert len(support.forecast_window_index) > len(support.anchor_window_index) > 0
        assert support.history_window_index.shape == (
            len(support.anchor_window_index),
            2,
        )
        assert np.all(support.y == 0)
        assert np.array_equal(
            external.classification_windows.label[support.anchor_window_index],
            support.y,
        )
        assert np.all(
            external.classification_windows.target_end[support.anchor_window_index]
            >= 384
        )
    assert external.provenance["label_role"].startswith("evaluation_only")


def test_external_support_cache_has_hash_validated_resume(tmp_path: Path) -> None:
    external = _external_protocol(tmp_path)
    args = SimpleNamespace(
        output_dir=tmp_path / "output",
        cache_compressed=True,
        finalize_only=False,
    )
    first = _ensure_support_artifact(args, external)
    second = _ensure_support_artifact(args, external)
    assert first == second
    root = args.output_dir / "phase3b" / "external_negative_only" / "dataset"
    done = json.loads((root / "DONE.json").read_text(encoding="utf-8"))
    assert done["stage"] == "h200_phase3b_external_dataset"
    assert done["protocol_fingerprint"] == external.fingerprint
    assert set(done["artifacts"]) == {"support", "provenance"}


def test_physical_gaussian_moment_matching_includes_between_model_variance() -> None:
    target = np.ones((2, 9, 128), dtype=np.float32)
    indices = np.array([5, 9], dtype=np.int64)
    labels = np.zeros(2, dtype=np.int8)
    first = {
        "target": target,
        "mu": np.zeros_like(target),
        "sigma": np.ones_like(target),
        "y": labels,
        "window_index": indices,
    }
    second = {
        "target": target,
        "mu": np.full_like(target, 2.0),
        "sigma": np.full_like(target, 3.0),
        "y": labels,
        "window_index": indices,
    }
    result = moment_match_external_forecasts((first, second))
    np.testing.assert_allclose(result["mu"], 1.0)
    # E[sigma^2 + mu^2] - E[mu]^2 = (1 + 0 + 9 + 4)/2 - 1 = 6
    np.testing.assert_allclose(result["sigma"], np.sqrt(6.0), rtol=1e-6)
    assert np.array_equal(result["window_index"], indices)


def test_negative_only_metrics_report_specificity_and_false_alarm_rate_only(
    tmp_path: Path,
) -> None:
    external = _external_protocol(tmp_path)
    indices = external.support["S04"].anchor_window_index[:10]
    probability = np.array([0.9, 0.8, *([0.1] * 8)], dtype=np.float64)
    prediction = (probability >= 0.5).astype(np.int8)
    metrics = negative_only_metrics(
        external.dataset,
        external.classification_windows,
        indices,
        probability,
        prediction,
    )
    assert metrics["metric_scope"] == "negative_only"
    assert metrics["specificity"] == 0.8
    assert metrics["false_positive_windows"] == 2
    assert metrics["false_alarm_events"] == metrics["predicted_events"] == 1
    assert metrics["false_alarm_events_per_hour"] > 0
    for undefined in ("sensitivity", "precision", "f1", "auroc", "auprc", "mcc"):
        assert undefined not in metrics


def test_repetitions_are_averaged_within_two_external_subjects(
    tmp_path: Path,
) -> None:
    external = _external_protocol(tmp_path)
    results = []
    for subject in EXTERNAL_SUBJECTS:
        indices = external.support[subject].anchor_window_index[:12]
        for arm in PHASE3_ARMS:
            for repetition in range(2):
                probability = np.full(12, 0.1, dtype=np.float64)
                prediction = np.zeros(12, dtype=np.int8)
                if repetition == 1:
                    probability[:2] = 0.9
                    prediction[:2] = 1
                metrics = negative_only_metrics(
                    external.dataset,
                    external.classification_windows,
                    indices,
                    probability,
                    prediction,
                )
                metrics.update({"external_subject": subject, "arm": arm})
                results.append(
                    ExternalCellResult(
                        metrics=metrics,
                        window_index=indices,
                        y_prob=probability,
                        y_pred=prediction,
                        done_sha256=f"{subject}-{arm}-{repetition}",
                    )
                )
    aggregate, subject_rows, timeline_rows = _aggregate_results(
        SimpleNamespace(), external, results, expected_repetitions=2
    )
    assert aggregate["n_independent_subjects"] == 2
    assert len(subject_rows) == len(EXTERNAL_SUBJECTS) * len(PHASE3_ARMS)
    assert all(row["repetitions"] == 2 for row in subject_rows)
    assert all(row["repetitions_are_independent_subjects"] is False for row in subject_rows)
    assert all(
        payload["n_independent_subjects"] == 2
        for payload in aggregate["subject_macro"].values()
    )
    assert len(timeline_rows) == 12 * len(EXTERNAL_SUBJECTS) * len(PHASE3_ARMS)
    assert aggregate["external_labels_used_for_training_or_threshold"] is False
