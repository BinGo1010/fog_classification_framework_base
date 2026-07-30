from __future__ import annotations

import csv
import json
import sys

import pytest

from scripts.aggregate_fog_baseline_multiseed import (
    bootstrap_mean_ci,
    main as aggregate_main,
)


METHODS = ("freeze_index", "tf_svm", "tf_rf", "cnn_gru")
METRICS = (
    "pr_auc",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "fog_recall",
    "specificity",
    "precision",
    "fog_f1",
    "event_sensitivity",
    "false_alarm_events_per_hour",
    "median_detection_delay_sec",
)


def _write_seed(root, seed: int) -> None:
    seed_root = root / f"seed_{seed}"
    seed_root.mkdir(parents=True)
    config = {
        "suite_version": "fog_reference_baselines.v2",
        "dataset_adapter": "daphnet",
        "seed": seed,
        "methods_resolved": list(METHODS),
        "folds_resolved": ["S01", "S02"],
        "subjects": ["S01", "S02"],
    }
    (seed_root / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    (seed_root / "status.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )
    rows = []
    for method_index, method in enumerate(METHODS):
        for subject_index, subject in enumerate(("S01", "S02")):
            seed_effect = (
                0.0
                if method == "freeze_index"
                else 0.01 * (seed - 1)
            )
            value = 0.2 + 0.05 * method_index + 0.02 * subject_index
            value += seed_effect
            row = {
                "method": method,
                "test_subject": subject,
                **{metric: value for metric in METRICS},
            }
            row["false_alarm_events_per_hour"] = 10.0 + method_index
            row["median_detection_delay_sec"] = 0.5 + subject_index
            rows.append(row)
    with (seed_root / "fold_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "test_subject", *METRICS),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_bootstrap_mean_ci_is_subject_level_and_deterministic() -> None:
    first = bootstrap_mean_ci([0.1, 0.2, 0.3], samples=1000, seed=7)
    second = bootstrap_mean_ci([0.1, 0.2, 0.3], samples=1000, seed=7)
    assert first == second
    assert first[0] == pytest.approx(0.2)
    assert first[1] <= first[0] <= first[2]


def test_multiseed_aggregation_averages_seed_within_subject_first(
    tmp_path,
    monkeypatch,
) -> None:
    _write_seed(tmp_path, 1)
    _write_seed(tmp_path, 2)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_fog_baseline_multiseed.py",
            "--output-dir",
            str(tmp_path),
            "--seeds",
            "1,2",
            "--bootstrap-samples",
            "1000",
        ],
    )
    aggregate_main()

    with (tmp_path / "aggregate_multiseed_metrics.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        aggregate = json.load(handle)
    assert aggregate["status"] == "complete"
    assert aggregate["aggregation_unit"] == (
        "held_out_subject_after_within_subject_seed_mean"
    )
    # TF-RF: base=0.30, S02 adds .02, and the two-seed average adds .005.
    assert aggregate["aggregate"]["tf_rf"][
        "subject_macro_after_seed_mean"
    ]["pr_auc"]["mean"] == pytest.approx(0.315)
    assert aggregate["freeze_index_max_subject_seed_pr_std"] == 0.0

    with (tmp_path / "publication_table.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert all(
        row["Delta PR-AUC [95% CI]"] == "NA (reference required)"
        for row in rows
    )
    with (tmp_path / "pairwise_pr_auc_deltas.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        pairwise = list(csv.DictReader(handle))
    assert len(pairwise) == 6

    reference_path = tmp_path / "proposed_pr.csv"
    with reference_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("seed", "test_subject", "pr_auc"),
        )
        writer.writeheader()
        for seed in (1, 2):
            for subject in ("S01", "S02"):
                writer.writerow(
                    {
                        "seed": seed,
                        "test_subject": subject,
                        "pr_auc": 0.8,
                    }
                )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_fog_baseline_multiseed.py",
            "--output-dir",
            str(tmp_path),
            "--seeds",
            "1,2",
            "--bootstrap-samples",
            "1000",
            "--reference-pr-csv",
            str(reference_path),
        ],
    )
    aggregate_main()
    with (tmp_path / "publication_table.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        referenced_rows = list(csv.DictReader(handle))
    assert all(
        "reference required" not in row["Delta PR-AUC [95% CI]"]
        for row in referenced_rows
    )
