from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "report_daphnet_gru_residual_feasibility.py"
SPEC = importlib.util.spec_from_file_location("h200_feasibility_report", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _phase0_metrics(subject: str, offset: float) -> dict:
    def model_quartiles(base: float) -> list[dict]:
        return [
            {
                "quartile": quartile,
                "rmse": base + 0.02 * quartile,
                "nll": 0.4 + base + 0.03 * quartile,
                "coverage_1sigma": 0.70 - 0.01 * quartile,
                "coverage_2sigma": 0.96 - 0.005 * quartile,
            }
            for quartile in range(1, 5)
        ]

    return {
        "subject": subject,
        "gru": {"lead_quartiles": model_quartiles(0.20 + offset)},
        "persistence": {"lead_quartiles": model_quartiles(0.28 + offset)},
    }


def _make_synthetic_result(result_dir: Path) -> None:
    for index, subject in enumerate(("S01", "S05")):
        _json(
            result_dir / "phase0" / f"loso_{subject}" / "metrics.json",
            _phase0_metrics(subject, 0.01 * index),
        )
    _json(
        result_dir / "phase0" / "aggregate.json",
        {"decision": "subset_only", "gru_better_rmse_subjects": 2},
    )

    arms = REPORT.PHASE2_ARMS
    for subject_index, subject in enumerate(("S01", "S05")):
        for arm_index, arm in enumerate(arms):
            root = result_dir / "phase2" / f"loso_{subject}" / arm
            metric = {
                "phase": "2",
                "test_subject": subject,
                "arm": arm,
                "n": 12,
                "pr_auc": 0.30 + 0.04 * arm_index + 0.01 * subject_index,
                "event_sensitivity": 0.55 + 0.03 * arm_index,
                "false_alarm_events_per_hour": 4.0 - 0.25 * arm_index,
                "fog_recall": 0.58 + 0.02 * arm_index,
                "macro_f1": 0.50 + 0.02 * arm_index,
            }
            _json(root / "metrics.json", metric)
            y_true = np.asarray([0, 1] * 6, dtype=np.int8)
            y_prob = np.linspace(0.05, 0.95, len(y_true))
            y_prob = np.clip(y_prob + 0.01 * arm_index, 0.0, 1.0)
            root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(root / "predictions.npz", y_true=y_true, y_prob=y_prob)

    phase3a = []
    for subject_index, subject in enumerate(REPORT.PHASE3A_SUBJECTS):
        for arm_index, arm in enumerate(REPORT.PHASE3_ARMS):
            phase3a.append(
                {
                    "phase": "3a",
                    "test_subject": subject,
                    "arm": arm,
                    "repetitions": 2,
                    "pr_auc": 0.40 + 0.05 * arm_index + 0.01 * subject_index,
                    "event_sensitivity": 0.60 + 0.02 * arm_index,
                    "false_alarm_events_per_hour": 3.5 - 0.2 * arm_index,
                    "fog_recall": 0.61,
                    "macro_f1": 0.56,
                }
            )
    _csv(result_dir / "phase3a" / "subject_seed_averaged_metrics.csv", phase3a)

    phase3b = []
    for subject_index, subject in enumerate(REPORT.MAIN_SUBJECTS):
        for arm_index, arm in enumerate(REPORT.PHASE3_ARMS):
            phase3b.append(
                {
                    "phase": "3b",
                    "test_subject": subject,
                    "arm": arm,
                    "repetitions": 3,
                    "pr_auc": 0.42 + 0.045 * arm_index + 0.005 * subject_index,
                    "event_sensitivity": 0.62 + 0.015 * arm_index,
                    "false_alarm_events_per_hour": 3.2 - 0.15 * arm_index,
                    "fog_recall": 0.63,
                    "macro_f1": 0.58,
                }
            )
    _csv(result_dir / "phase3b" / "subject_seed_averaged_metrics.csv", phase3b)

    external_root = result_dir / "phase3b" / "external_negative_only"
    subject_metrics = []
    timeline = []
    for subject in ("S04", "S10"):
        for arm_index, arm in enumerate(REPORT.PHASE3_ARMS):
            subject_metrics.append(
                {
                    "external_subject": subject,
                    "arm": arm,
                    "repetitions": 3,
                    "specificity_mean": 0.98 - 0.005 * arm_index,
                    "positive_window_rate_mean": 0.02 + 0.005 * arm_index,
                    "false_alarm_events_per_hour_mean": 1.0 + 0.2 * arm_index,
                }
            )
            for run_id in ("R01", "R02"):
                for window in range(4):
                    timeline.append(
                        {
                            "external_subject": subject,
                            "arm": arm,
                            "record_id": f"{subject}_{run_id}",
                            "run_id": run_id,
                            "window_index": window,
                            "target_start_sec": window * 0.25,
                            "target_end_exclusive_sec": window * 0.25 + 0.5,
                            "mean_y_prob": 0.05 + 0.03 * arm_index + 0.02 * window,
                            "positive_vote_rate": 0.0,
                            "consensus_y_pred": 0,
                        }
                    )
    _csv(external_root / "subject_metrics.csv", subject_metrics)
    _csv(external_root / "subject_averaged_timeline.csv", timeline)
    _json(external_root / "aggregate.json", {"status": "complete"})


def test_report_generates_tables_figures_and_source_hash_manifest(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    output_dir = tmp_path / "report"
    _make_synthetic_result(result_dir)

    manifest = REPORT.build_report(result_dir, output_dir, dpi=72)

    assert (output_dir / "REPORT.md").is_file()
    assert (output_dir / "publication_tables.csv").is_file()
    assert (output_dir / "report_manifest.json").is_file()
    expected_figures = {
        "phase0_lead_quartile_diagnostics.png",
        "phase2_subject_waterfalls.png",
        "phase2_subject_pr_auc.png",
        "phase2_false_alarm_vs_event_sensitivity.png",
        "phase2_pr_curves_by_fold.png",
        "phase2_pr_curve_pooled_auxiliary.png",
        "phase3_seed_averaged_pr_auc_comparison.png",
        "external_S04_timeline.png",
        "external_S10_timeline.png",
    }
    assert expected_figures <= {path.name for path in (output_dir / "figures").iterdir()}

    with (output_dir / "publication_tables.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    sections = {row["section"] for row in rows}
    assert "forecast_lead_quartile" in sections
    assert "classifier_subject_delta" in sections
    assert "auxiliary_pooled_window_metric" in sections
    assert "crossfit_seed_averaged_subject_metric" in sections
    assert "external_negative_only_subject_metric" in sections

    source = result_dir / "phase2" / "loso_S01" / "raw6" / "metrics.json"
    source_entry = next(
        item for item in manifest["source_files"]
        if item["path"] == source.relative_to(result_dir).as_posix()
    )
    assert source_entry["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["primitive_caches_loaded"] is False
    assert all("primitive" not in item["path"].lower() for item in manifest["source_files"])


def test_partial_report_lists_missing_and_ignores_corrupt_primitive_cache(
    tmp_path: Path,
) -> None:
    result_dir = tmp_path / "partial"
    primitive = result_dir / "loso_S01" / "h200_primitives" / "train_primitives.npz"
    primitive.parent.mkdir(parents=True)
    primitive.write_bytes(b"not a valid npz and must never be opened")
    output_dir = tmp_path / "partial_report"

    assert REPORT.main(
        ["--result-dir", str(result_dir), "--output-dir", str(output_dir), "--dpi", "72"]
    ) == 0

    manifest = json.loads((output_dir / "report_manifest.json").read_text(encoding="utf-8"))
    report = (output_dir / "REPORT.md").read_text(encoding="utf-8")
    assert manifest["status"] == "partial"
    assert manifest["missing"]
    assert manifest["source_files"] == []
    assert "MISSING: Phase 0" in report
    assert "MISSING: S04/S10" in report
    assert primitive.exists()


def test_complete_status_requires_current_independent_audit(tmp_path: Path) -> None:
    result = tmp_path / "certified"
    result.mkdir()
    fingerprint = "a" * 64
    _json(result / "config.json", {"protocol_fingerprint": fingerprint})
    _json(
        result / "audit_report.json",
        {
            "ok": True,
            "complete": True,
            "protocol_fingerprint": fingerprint,
            "summary": {"errors": 0, "incomplete": 0},
        },
    )
    state = REPORT.ReportState(result, tmp_path / "out", 72)
    state.load_certification_inputs()
    state.finalize_certification()
    assert state.certified_complete is True

    (result / "config.json").write_text(
        json.dumps({"protocol_fingerprint": fingerprint, "changed": True}),
        encoding="utf-8",
    )
    stale = REPORT.ReportState(result, tmp_path / "out2", 72)
    stale.load_certification_inputs()
    stale.finalize_certification()
    assert stale.certified_complete is False
    assert any("audit_not_older" in item for item in stale.missing)


def test_phase3_incomplete_seed_cartesian_product_is_explicit(tmp_path: Path) -> None:
    result = tmp_path / "seed_grid"
    rows = [
        {
            "phase": "3a",
            "test_subject": subject,
            "arm": arm,
            "repetitions": 1,
            "pr_auc": 0.5,
        }
        for subject in REPORT.PHASE3A_SUBJECTS
        for arm in REPORT.PHASE3_ARMS
    ]
    _csv(result / "phase3a" / "subject_seed_averaged_metrics.csv", rows)
    state = REPORT.ReportState(result, tmp_path / "out", 72)
    state.config = {
        "protocol_fingerprint": "b" * 64,
        "phase3_nbm_seeds": "42",
        "phase3_classifier_seed_policy": {"3a": [42], "3b": [42, 43, 44]},
    }
    REPORT._read_phase3_subject_rows(state, "3a", [])
    assert any("seed Cartesian product" in item for item in state.missing)


def test_external_truncated_timeline_is_not_called_full(tmp_path: Path) -> None:
    result = tmp_path / "external"
    root = result / "phase3b" / "external_negative_only"
    subject_rows = []
    timeline_rows = []
    for subject in ("S04", "S10"):
        for arm in REPORT.PHASE3_ARMS:
            subject_rows.append(
                {
                    "external_subject": subject,
                    "arm": arm,
                    "repetitions": 24,
                    "consensus_n_negative_windows": 2,
                    "specificity_mean": 1.0,
                }
            )
            timeline_rows.append(
                {
                    "external_subject": subject,
                    "arm": arm,
                    "record_id": "R",
                    "run_id": "1",
                    "window_index": 1,
                    "target_start_sec": 0.0,
                    "mean_y_prob": 0.1,
                    "positive_vote_rate": 0.0,
                    "consensus_y_pred": 0,
                }
            )
    _csv(root / "subject_metrics.csv", subject_rows)
    _csv(root / "subject_averaged_timeline.csv", timeline_rows)
    _json(
        root / "aggregate.json",
        {
            "status": "complete",
            "subjects": ["S04", "S10"],
            "arms": list(REPORT.PHASE3_ARMS),
            "repetitions_per_subject_arm": 24,
            "main_protocol_fingerprint": "c" * 64,
        },
    )
    _json(root / "DONE.json", {"format_version": 1})
    state = REPORT.ReportState(result, tmp_path / "out", 72)
    state.config = {
        "protocol_fingerprint": "c" * 64,
        "phase3_nbm_seeds": "42",
        "phase3_classifier_seed_policy": {"3b": [42, 43, 44]},
    }
    REPORT._read_external(state, [])
    assert any("truncated or has duplicate endpoints" in item for item in state.missing)
