#!/usr/bin/env python
"""Build a publication-oriented report from completed feasibility artifacts.

The report layer is deliberately read-only with respect to ``--result-dir``.
It consumes only small JSON/CSV metric files and classifier prediction NPZs;
the large H200 primitive caches are never opened.  Partial experiment trees
are valid inputs: missing panels are recorded explicitly instead of being
filled with synthetic values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


REPORT_VERSION = "daphnet_gru_h200_feasibility_report.v1"
MAIN_SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
PHASE3A_SUBJECTS = ("S01", "S05", "S08")
PHASE1_ARMS = ("raw4", "normality", "raw4_zero", "raw4_normality")
PHASE2_ARMS = (
    "raw4",
    "raw6",
    "normality",
    "raw4_zero",
    "raw4_normality",
)
PHASE3_ARMS = ("raw6", "raw4_zero", "raw4_normality")
ARM_IDS = {
    "raw4": "F0",
    "raw6": "F1/C0",
    "normality": "F2",
    "raw4_zero": "F3/C1",
    "raw4_normality": "F4/C2",
}
ARM_LABELS = {
    "raw4": "Raw4",
    "raw6": "Raw6",
    "normality": "Normality",
    "raw4_zero": "Raw4 + zero",
    "raw4_normality": "Raw4 + normality",
}
ARM_COLORS = {
    "raw4": "#7f7f7f",
    "raw6": "#1f77b4",
    "normality": "#9467bd",
    "raw4_zero": "#ff7f0e",
    "raw4_normality": "#2ca02c",
}
TABLE_COLUMNS = (
    "section",
    "phase",
    "subject",
    "arm",
    "arm_id",
    "model",
    "comparison",
    "lead_quartile",
    "metric",
    "value",
    "n",
    "repetitions",
    "notes",
    "source",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _finite_float(value)
    return int(number) if number is not None else None


def _mean(values: Iterable[Any]) -> float | None:
    finite = [value for item in values if (value := _finite_float(item)) is not None]
    return float(np.mean(finite)) if finite else None


def _format(value: Any, digits: int = 4) -> str:
    number = _finite_float(value)
    return "missing" if number is None else f"{number:.{digits}f}"


def _subject_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("loso_"):
            return part.removeprefix("loso_")
    return ""


def _arm_sort_key(arm: str) -> tuple[int, str]:
    try:
        return PHASE2_ARMS.index(arm), arm
    except ValueError:
        return len(PHASE2_ARMS), arm


class ReportState:
    """Tracks every input that actually influenced the generated report."""

    def __init__(self, result_dir: Path, output_dir: Path, dpi: int) -> None:
        self.result_dir = result_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.figures_dir = self.output_dir / "figures"
        self.dpi = int(dpi)
        self.sources: dict[Path, set[str]] = defaultdict(set)
        self.outputs: list[Path] = []
        self.missing: list[str] = []
        self.warnings: list[str] = []
        self.config: dict[str, Any] = {}
        self.audit: dict[str, Any] = {}
        self.config_path = self.result_dir / "config.json"
        self.audit_path = self.result_dir / "audit_report.json"
        self.certified_complete = False

    def relative_source(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.result_dir).as_posix()
        except ValueError:
            return str(resolved)

    def consume(self, path: Path, role: str) -> None:
        resolved = path.resolve()
        if "primitive" in resolved.name.lower() or "primitives" in {
            part.lower() for part in resolved.parts
        }:
            raise RuntimeError(f"Refusing to load primitive cache: {resolved}")
        self.sources[resolved].add(str(role))

    def read_json(self, path: Path, role: str) -> dict[str, Any]:
        self.consume(path, role)
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object: {path}")
        return value

    def read_csv(self, path: Path, role: str) -> list[dict[str, str]]:
        self.consume(path, role)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def read_predictions(self, path: Path, role: str) -> tuple[np.ndarray, np.ndarray]:
        if path.name != "predictions.npz":
            raise ValueError(f"Only classifier predictions.npz may be read: {path}")
        self.consume(path, role)
        with np.load(path, allow_pickle=False) as payload:
            if not {"y_true", "y_prob"}.issubset(payload.files):
                raise KeyError(f"Prediction file lacks y_true/y_prob: {path}")
            y_true = np.asarray(payload["y_true"], dtype=np.int8).reshape(-1)
            y_prob = np.asarray(payload["y_prob"], dtype=np.float64).reshape(-1)
        if len(y_true) != len(y_prob) or not len(y_true):
            raise ValueError(f"Empty or misaligned predictions: {path}")
        if not np.isin(y_true, (0, 1)).all() or not np.isfinite(y_prob).all():
            raise ValueError(f"Invalid binary predictions: {path}")
        return y_true, y_prob

    def add_missing(self, message: str) -> None:
        if message not in self.missing:
            self.missing.append(message)

    def add_warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def save_figure(self, fig: plt.Figure, filename: str) -> Path:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        destination = self.figures_dir / filename
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        fig.savefig(temporary, format="png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        os.replace(temporary, destination)
        self.outputs.append(destination)
        return destination

    def load_certification_inputs(self) -> None:
        if not self.config_path.is_file():
            self.add_missing(
                "Publication certification: root config.json is missing; report remains partial."
            )
        else:
            try:
                self.config = self.read_json(self.config_path, "experiment_config")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.add_missing(f"Publication certification: config.json is invalid ({exc}).")
        fingerprint = self.config.get("protocol_fingerprint")
        if not (
            isinstance(fingerprint, str)
            and len(fingerprint) == 64
            and all(character in "0123456789abcdef" for character in fingerprint)
        ):
            self.add_missing(
                "Publication certification: config protocol_fingerprint is absent or invalid."
            )
        if not self.audit_path.is_file():
            self.add_missing(
                "Publication certification: audit_report.json is missing; run the independent audit after the experiment finishes."
            )
            return
        try:
            self.audit = self.read_json(self.audit_path, "independent_audit_report")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.add_missing(
                f"Publication certification: audit_report.json is invalid ({exc})."
            )

    def finalize_certification(self) -> None:
        fingerprint = self.config.get("protocol_fingerprint")
        checks = {
            "audit_ok": self.audit.get("ok") is True,
            "audit_complete": self.audit.get("complete") is True,
            "protocol_fingerprint_match": bool(
                fingerprint
                and self.audit.get("protocol_fingerprint") == fingerprint
            ),
            "audit_has_no_errors": _integer(
                (self.audit.get("summary") or {}).get("errors")
            ) == 0,
            "audit_has_no_incomplete_items": _integer(
                (self.audit.get("summary") or {}).get("incomplete")
            ) == 0,
        }
        if self.audit_path.is_file():
            other_sources = [
                path for path in self.sources if path.resolve() != self.audit_path.resolve()
            ]
            newest_source = max(
                (path.stat().st_mtime_ns for path in other_sources), default=0
            )
            checks["audit_not_older_than_consumed_sources"] = (
                self.audit_path.stat().st_mtime_ns >= newest_source
            )
        else:
            checks["audit_not_older_than_consumed_sources"] = False
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            self.add_missing(
                "Publication certification failed: " + ", ".join(failed) + "."
            )
        self.certified_complete = not failed


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _table_row(**values: Any) -> dict[str, Any]:
    return {column: values.get(column, "") for column in TABLE_COLUMNS}


def _read_phase0(
    state: ReportState, table: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    records: list[dict[str, Any]] = []
    paths = sorted(state.result_dir.glob("phase0/loso_*/metrics.json"))
    if not paths:
        state.add_missing(
            "Phase 0 lead-quartile diagnostics: no phase0/loso_*/metrics.json files."
        )
        return records, None
    for path in paths:
        try:
            payload = state.read_json(path, "phase0_fold_metrics")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            state.add_warning(f"Skipped invalid Phase 0 metrics {path}: {exc}")
            continue
        subject = str(payload.get("subject") or _subject_from_path(path))
        for model in ("gru", "persistence"):
            quartiles = payload.get(model, {}).get("lead_quartiles", [])
            if not isinstance(quartiles, list) or not quartiles:
                state.add_missing(f"Phase 0 {subject}/{model}: lead quartiles are missing.")
                continue
            indexed_quartiles: dict[int, dict[str, Any]] = {}
            duplicate_quartiles: set[int] = set()
            for position, block in enumerate(quartiles, start=1):
                if not isinstance(block, dict):
                    continue
                quartile = int(block.get("quartile", position))
                if quartile in indexed_quartiles:
                    duplicate_quartiles.add(quartile)
                    continue
                indexed_quartiles[quartile] = block
            if set(indexed_quartiles) != {1, 2, 3, 4} or duplicate_quartiles:
                state.add_missing(
                    f"Phase 0 {subject}/{model}: expected unique quartiles Q1-Q4; "
                    f"found {sorted(indexed_quartiles)}, duplicates {sorted(duplicate_quartiles)}."
                )
            for quartile, block in sorted(indexed_quartiles.items()):
                record = {
                    "subject": subject,
                    "model": model,
                    "quartile": quartile,
                    "source": state.relative_source(path),
                    **block,
                }
                records.append(record)
                for metric in ("rmse", "nll", "coverage_1sigma", "coverage_2sigma"):
                    value = _finite_float(block.get(metric))
                    if value is not None:
                        table.append(
                            _table_row(
                                section="forecast_lead_quartile",
                                phase="0",
                                subject=subject,
                                model=model,
                                lead_quartile=quartile,
                                metric=metric,
                                value=value,
                                notes="clean non-FOG validation windows",
                                source=state.relative_source(path),
                            )
                        )
                    else:
                        state.add_missing(
                            f"Phase 0 {subject}/{model}/Q{quartile}: {metric} is missing or non-finite."
                        )
    for model in ("gru", "persistence"):
        for quartile in range(1, 5):
            subset = [
                row for row in records
                if row["model"] == model and row["quartile"] == quartile
            ]
            for metric in ("rmse", "nll", "coverage_1sigma", "coverage_2sigma"):
                value = _mean(row.get(metric) for row in subset)
                if value is not None:
                    table.append(
                        _table_row(
                            section="forecast_lead_quartile_macro",
                            phase="0",
                            subject="subject_macro",
                            model=model,
                            lead_quartile=quartile,
                            metric=metric,
                            value=value,
                            n=len(subset),
                            notes="unweighted mean across available validation subjects",
                            source="derived from Phase 0 fold metrics",
                        )
                    )
    aggregate_path = state.result_dir / "phase0" / "aggregate.json"
    aggregate = None
    if aggregate_path.exists():
        try:
            aggregate = state.read_json(aggregate_path, "phase0_aggregate")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            state.add_warning(f"Could not read Phase 0 aggregate: {exc}")
    available = {
        subject
        for subject in {str(row["subject"]) for row in records}
        if {
            int(row["quartile"])
            for row in records
            if row["model"] == "gru" and row["subject"] == subject
        }
        == {1, 2, 3, 4}
        and {
            int(row["quartile"])
            for row in records
            if row["model"] == "persistence" and row["subject"] == subject
        }
        == {1, 2, 3, 4}
    }
    missing_subjects = sorted(set(MAIN_SUBJECTS) - available)
    if missing_subjects:
        state.add_missing(
            "Phase 0 preregistered folds without lead-quartile results: "
            + ", ".join(missing_subjects)
            + "."
        )
    return records, aggregate


def _plot_phase0(state: ReportState, records: Sequence[Mapping[str, Any]]) -> Path | None:
    if not records:
        return None
    metrics = (
        ("rmse", "RMSE", None),
        ("nll", "Gaussian NLL", None),
        ("coverage_1sigma", "Coverage within 1 sigma", 0.6827),
        ("coverage_2sigma", "Coverage within 2 sigma", 0.9545),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True)
    for axis, (metric, label, ideal) in zip(axes.flat, metrics):
        for model, color, marker in (
            ("gru", "#1f77b4", "o"),
            ("persistence", "#7f7f7f", "s"),
        ):
            means = []
            for quartile in range(1, 5):
                means.append(
                    _mean(
                        row.get(metric)
                        for row in records
                        if row.get("model") == model
                        and int(row.get("quartile", -1)) == quartile
                    )
                )
            if any(value is not None for value in means):
                y = [np.nan if value is None else value for value in means]
                axis.plot(range(1, 5), y, marker=marker, color=color, label=model)
        if ideal is not None:
            axis.axhline(ideal, color="#d62728", linestyle="--", linewidth=1, label="ideal")
            axis.set_ylim(0.0, 1.03)
        axis.set_title(label)
        axis.set_xlabel("Future lead quartile (0.5 s each)")
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    fig.suptitle("Phase 0: lead-dependent normal forecast diagnostics", y=1.01)
    fig.tight_layout()
    return state.save_figure(fig, "phase0_lead_quartile_diagnostics.png")


def _read_phase1(state: ReportState) -> dict[str, Any] | None:
    """Include the engineering smoke gate in report completeness, not science."""

    aggregate_path = state.result_dir / "phase1" / "aggregate.json"
    if not aggregate_path.is_file():
        state.add_missing(
            "Phase 1 engineering smoke: phase1/aggregate.json is missing."
        )
        return None
    try:
        aggregate = state.read_json(aggregate_path, "phase1_engineering_smoke")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        state.add_missing(f"Phase 1 engineering smoke aggregate is invalid ({exc}).")
        return None
    arms = aggregate.get("arms")
    completed_arms = {
        str(arm)
        for arm, payload in (arms.items() if isinstance(arms, dict) else ())
        if isinstance(payload, dict)
        and set(payload.get("completed_subjects", ())) == {"S01"}
    }
    if completed_arms != set(PHASE1_ARMS):
        state.add_missing(
            "Phase 1 engineering smoke does not contain the exact S01 four-arm matrix."
        )
    smoke = aggregate.get("smoke_gate")
    if not isinstance(smoke, dict) or smoke.get("status") != "pass":
        state.add_missing("Phase 1 engineering smoke gate is absent or did not pass.")
    if aggregate.get("engineering_smoke") is not True:
        state.add_missing("Phase 1 aggregate is not marked engineering_smoke=true.")
    return aggregate


def _normalise_metric_row(row: Mapping[str, Any], source: str) -> dict[str, Any]:
    result = dict(row)
    result["test_subject"] = str(
        row.get("test_subject") or row.get("subject") or ""
    )
    result["arm"] = str(row.get("arm") or "")
    result["source"] = source
    return result


def _read_phase2(
    state: ReportState, table: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(state.result_dir.glob("phase2/loso_*/*/metrics.json"))
    for path in paths:
        try:
            payload = state.read_json(path, "phase2_fold_arm_metrics")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            state.add_warning(f"Skipped invalid Phase 2 metrics {path}: {exc}")
            continue
        row = _normalise_metric_row(payload, state.relative_source(path))
        if row["arm"] and row["test_subject"]:
            rows.append(row)
    if not rows:
        fallback = state.result_dir / "phase2" / "fold_metrics.csv"
        if fallback.exists():
            for raw in state.read_csv(fallback, "phase2_fold_metrics_table"):
                row = _normalise_metric_row(raw, state.relative_source(fallback))
                if row["arm"] and row["test_subject"]:
                    rows.append(row)
    if not rows:
        state.add_missing(
            "Phase 2 subject metrics: no per-cell metrics.json or fold_metrics.csv."
        )
        return rows

    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[(row["test_subject"], row["arm"])].append(row)
    duplicate_keys = sorted(key for key, values in grouped_rows.items() if len(values) > 1)
    if duplicate_keys:
        state.add_missing(
            "Phase 2 duplicate subject/arm metric rows: "
            + ", ".join(f"{subject}/{arm}" for subject, arm in duplicate_keys)
            + "."
        )
    expected_keys = {(subject, arm) for subject in MAIN_SUBJECTS for arm in PHASE2_ARMS}
    extra_keys = sorted(set(grouped_rows) - expected_keys)
    if extra_keys:
        state.add_missing(
            "Phase 2 unexpected subject/arm metric rows: "
            + ", ".join(f"{subject}/{arm}" for subject, arm in extra_keys)
            + "."
        )
    by_key = {
        key: values[0] for key, values in grouped_rows.items() if key in expected_keys
    }
    rows = [by_key[key] for key in sorted(by_key, key=lambda x: (x[0], _arm_sort_key(x[1])))]
    for row in rows:
        subject, arm = row["test_subject"], row["arm"]
        for metric in (
            "pr_auc",
            "event_sensitivity",
            "false_alarm_events_per_hour",
            "fog_recall",
            "macro_f1",
        ):
            value = _finite_float(row.get(metric))
            if value is not None:
                table.append(
                    _table_row(
                        section="classifier_subject_metric",
                        phase="2",
                        subject=subject,
                        arm=arm,
                        arm_id=ARM_IDS.get(arm, ""),
                        metric=metric,
                        value=value,
                        n=row.get("n", ""),
                        notes="exploratory outer-fold result",
                        source=row["source"],
                    )
                )
        if _finite_float(row.get("pr_auc")) is None:
            state.add_missing(
                f"Phase 2 {subject}/{arm}: subject PR-AUC is missing or non-finite."
            )
        if (
            _finite_float(row.get("event_sensitivity")) is None
            or _finite_float(row.get("false_alarm_events_per_hour")) is None
        ):
            state.add_missing(
                f"Phase 2 {subject}/{arm}: FA/h versus event-sensitivity operating point is incomplete."
            )
    for arm in PHASE2_ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        for metric in (
            "pr_auc",
            "event_sensitivity",
            "false_alarm_events_per_hour",
            "fog_recall",
            "macro_f1",
        ):
            value = _mean(row.get(metric) for row in arm_rows)
            if value is not None:
                table.append(
                    _table_row(
                        section="classifier_subject_macro",
                        phase="2",
                        subject="subject_macro",
                        arm=arm,
                        arm_id=ARM_IDS.get(arm, ""),
                        metric=metric,
                        value=value,
                        n=sum(_finite_float(row.get(metric)) is not None for row in arm_rows),
                        notes="unweighted mean across available held-out subjects",
                        source="derived from Phase 2 subject metrics",
                    )
                )
    for reference in ("raw6", "raw4_zero"):
        comparison = f"raw4_normality_minus_{reference}"
        for subject in MAIN_SUBJECTS:
            candidate = by_key.get((subject, "raw4_normality"))
            baseline = by_key.get((subject, reference))
            candidate_value = _finite_float(candidate.get("pr_auc")) if candidate else None
            baseline_value = _finite_float(baseline.get("pr_auc")) if baseline else None
            if candidate_value is None or baseline_value is None:
                continue
            table.append(
                _table_row(
                    section="classifier_subject_delta",
                    phase="2",
                    subject=subject,
                    comparison=comparison,
                    metric="pr_auc_delta",
                    value=candidate_value - baseline_value,
                    notes="paired within held-out subject",
                    source="derived from paired Phase 2 subject metrics",
                )
            )
        deltas = [
            _finite_float(row["value"])
            for row in table
            if row["section"] == "classifier_subject_delta"
            and row["phase"] == "2"
            and row["comparison"] == comparison
        ]
        deltas = [value for value in deltas if value is not None]
        if deltas:
            table.append(
                _table_row(
                    section="classifier_subject_macro_delta",
                    phase="2",
                    subject="subject_macro",
                    comparison=comparison,
                    metric="pr_auc_delta",
                    value=float(np.mean(deltas)),
                    n=len(deltas),
                    notes="unweighted paired-subject mean",
                    source="derived from paired Phase 2 subject metrics",
                )
            )
    completed = set(by_key)
    missing_cells = [
        f"{subject}/{arm}"
        for subject in MAIN_SUBJECTS
        for arm in PHASE2_ARMS
        if (subject, arm) not in completed
    ]
    if missing_cells:
        state.add_missing(
            "Phase 2 preregistered subject/arm cells without metrics: "
            + ", ".join(missing_cells)
            + "."
        )
    aggregate_path = state.result_dir / "phase2" / "aggregate.json"
    if aggregate_path.exists():
        try:
            state.read_json(aggregate_path, "phase2_aggregate")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            state.add_missing(f"Phase 2 aggregate is invalid ({exc}).")
    else:
        state.add_missing("Phase 2 aggregate.json is missing.")
    gate_path = state.result_dir / "phase2" / "gate.json"
    if gate_path.exists():
        try:
            state.read_json(gate_path, "phase2_gate")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            state.add_missing(f"Phase 2 gate is invalid ({exc}).")
    else:
        state.add_missing("Phase 2 gate.json is missing.")
    return rows


def _phase2_by_key(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["test_subject"]), str(row["arm"])): row for row in rows}


def _plot_phase2_waterfalls(
    state: ReportState, rows: Sequence[Mapping[str, Any]]
) -> Path | None:
    by_key = _phase2_by_key(rows)
    comparisons: list[tuple[str, str, list[tuple[str, float]]]] = []
    for reference, title in (
        ("raw6", "F4 - F1 (fusion - Raw6)"),
        ("raw4_zero", "F4 - F3 (normality - zero control)"),
    ):
        values = []
        for subject in MAIN_SUBJECTS:
            candidate = by_key.get((subject, "raw4_normality"))
            baseline = by_key.get((subject, reference))
            c = _finite_float(candidate.get("pr_auc")) if candidate else None
            b = _finite_float(baseline.get("pr_auc")) if baseline else None
            if c is not None and b is not None:
                values.append((subject, c - b))
        if values:
            comparisons.append((reference, title, sorted(values, key=lambda item: item[1])))
        else:
            state.add_missing(f"Phase 2 waterfall {title}: no paired subjects.")
    if not comparisons:
        return None
    fig, axes = plt.subplots(1, len(comparisons), figsize=(6.0 * len(comparisons), 4.8))
    axes_array = np.atleast_1d(axes)
    for axis, (_, title, values) in zip(axes_array, comparisons):
        subjects = [item[0] for item in values]
        deltas = np.asarray([item[1] for item in values])
        colors = np.where(deltas >= 0.0, "#2ca02c", "#d62728")
        axis.bar(subjects, deltas, color=colors)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel("Paired PR-AUC delta")
        axis.tick_params(axis="x", rotation=45)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Phase 2: held-out-subject paired waterfalls")
    fig.tight_layout()
    return state.save_figure(fig, "phase2_subject_waterfalls.png")


def _plot_phase2_pr_auc(
    state: ReportState, rows: Sequence[Mapping[str, Any]]
) -> Path | None:
    if not rows:
        return None
    fig, axis = plt.subplots(figsize=(10.5, 5.5))
    plotted = False
    positions = {subject: index for index, subject in enumerate(MAIN_SUBJECTS)}
    offsets = np.linspace(-0.28, 0.28, len(PHASE2_ARMS))
    for offset, arm in zip(offsets, PHASE2_ARMS):
        points = [
            (positions[str(row["test_subject"])], _finite_float(row.get("pr_auc")))
            for row in rows
            if row.get("arm") == arm and str(row.get("test_subject")) in positions
        ]
        points = [(x, y) for x, y in points if y is not None]
        if not points:
            continue
        plotted = True
        axis.scatter(
            [x + offset for x, _ in points],
            [y for _, y in points],
            label=f"{ARM_IDS.get(arm, '')} {ARM_LABELS.get(arm, arm)}",
            color=ARM_COLORS.get(arm),
            s=34,
        )
    if not plotted:
        plt.close(fig)
        state.add_missing("Phase 2 subject PR-AUC plot: no finite PR-AUC values.")
        return None
    axis.set_xticks(range(len(MAIN_SUBJECTS)), MAIN_SUBJECTS)
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("PR-AUC")
    axis.set_xlabel("Held-out subject")
    axis.set_title("Phase 2: PR-AUC by subject and arm")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return state.save_figure(fig, "phase2_subject_pr_auc.png")


def _plot_phase2_operating_points(
    state: ReportState, rows: Sequence[Mapping[str, Any]]
) -> Path | None:
    fig, axis = plt.subplots(figsize=(7.8, 5.8))
    plotted = False
    for arm in PHASE2_ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        x = [_finite_float(row.get("false_alarm_events_per_hour")) for row in arm_rows]
        y = [_finite_float(row.get("event_sensitivity")) for row in arm_rows]
        points = [
            (x_value, y_value, str(row.get("test_subject")))
            for row, x_value, y_value in zip(arm_rows, x, y)
            if x_value is not None and y_value is not None
        ]
        if not points:
            continue
        plotted = True
        axis.scatter(
            [item[0] for item in points],
            [item[1] for item in points],
            color=ARM_COLORS.get(arm),
            label=f"{ARM_IDS.get(arm, '')} {ARM_LABELS.get(arm, arm)}",
            alpha=0.85,
        )
        for x_value, y_value, subject in points:
            axis.annotate(subject, (x_value, y_value), fontsize=6, alpha=0.7)
    if not plotted:
        plt.close(fig)
        state.add_missing(
            "Phase 2 FA/h versus event sensitivity: no complete operating points."
        )
        return None
    axis.set_xlabel("False-alarm events per non-FOG hour")
    axis.set_ylabel("Event sensitivity")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("Phase 2 operating-point trade-off")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    return state.save_figure(fig, "phase2_false_alarm_vs_event_sensitivity.png")


def _precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        raise ValueError("PR curve is undefined without a positive label")
    order = np.argsort(-y_prob, kind="mergesort")
    truth = y_true[order]
    probability = y_prob[order]
    true_positive = np.cumsum(truth == 1)
    false_positive = np.cumsum(truth == 0)
    distinct = np.r_[np.flatnonzero(np.diff(probability)), len(probability) - 1]
    tp = true_positive[distinct].astype(np.float64)
    fp = false_positive[distinct].astype(np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / positives
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return np.r_[0.0, recall], np.r_[1.0, precision], average_precision


def _load_phase2_predictions(
    state: ReportState, rows: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    predictions: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    missing = []
    for row in rows:
        subject, arm = str(row["test_subject"]), str(row["arm"])
        path = state.result_dir / "phase2" / f"loso_{subject}" / arm / "predictions.npz"
        if not path.exists():
            missing.append(f"{subject}/{arm}")
            continue
        try:
            predictions[(subject, arm)] = state.read_predictions(
                path, "phase2_test_predictions"
            )
        except (OSError, ValueError, KeyError) as exc:
            state.add_warning(f"Skipped invalid Phase 2 predictions {path}: {exc}")
            missing.append(f"{subject}/{arm}")
    if missing:
        state.add_missing(
            "Phase 2 PR curves missing usable predictions.npz for: "
            + ", ".join(sorted(set(missing)))
            + "."
        )
    return predictions


def _plot_pr_curves(
    state: ReportState,
    predictions: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    table: list[dict[str, Any]],
) -> list[Path]:
    if not predictions:
        state.add_missing("Phase 2 PR curves: no usable prediction arrays.")
        return []
    subjects = [subject for subject in MAIN_SUBJECTS if any(key[0] == subject for key in predictions)]
    columns = min(4, max(1, len(subjects)))
    rows = int(math.ceil(len(subjects) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.5 * rows), squeeze=False)
    fold_curves = 0
    for axis, subject in zip(axes.flat, subjects):
        for arm in PHASE2_ARMS:
            pair = predictions.get((subject, arm))
            if pair is None:
                continue
            try:
                recall, precision, ap = _precision_recall_curve(*pair)
            except ValueError as exc:
                state.add_warning(f"Skipped PR curve {subject}/{arm}: {exc}")
                continue
            fold_curves += 1
            axis.step(recall, precision, where="post", color=ARM_COLORS.get(arm), label=f"{ARM_IDS.get(arm, '')} AP={ap:.3f}")
        axis.set_title(subject)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=6)
    for axis in list(axes.flat)[len(subjects):]:
        axis.set_visible(False)
    if fold_curves:
        fig.suptitle("Phase 2: PR curves for each held-out fold")
        fig.tight_layout()
        fold_path = state.save_figure(fig, "phase2_pr_curves_by_fold.png")
    else:
        plt.close(fig)
        state.add_missing("Phase 2 per-fold PR curves: every available fold lacked positive labels.")
        fold_path = None

    fig, axis = plt.subplots(figsize=(7.2, 5.8))
    pooled_curves = 0
    for arm in PHASE2_ARMS:
        pairs = [value for key, value in predictions.items() if key[1] == arm]
        if not pairs:
            continue
        y_true = np.concatenate([pair[0] for pair in pairs])
        y_prob = np.concatenate([pair[1] for pair in pairs])
        try:
            recall, precision, ap = _precision_recall_curve(y_true, y_prob)
        except ValueError as exc:
            state.add_warning(f"Skipped auxiliary pooled PR curve {arm}: {exc}")
            continue
        pooled_curves += 1
        axis.step(recall, precision, where="post", color=ARM_COLORS.get(arm), label=f"{ARM_IDS.get(arm, '')} {ARM_LABELS.get(arm, arm)} AP={ap:.3f}")
        table.append(
            _table_row(
                section="auxiliary_pooled_window_metric",
                phase="2",
                subject="pooled_windows",
                arm=arm,
                arm_id=ARM_IDS.get(arm, ""),
                metric="pr_auc",
                value=ap,
                n=len(y_true),
                notes="descriptive only; windows and folds are not independent inference units",
                source="derived from Phase 2 predictions.npz",
            )
        )
    if pooled_curves:
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_title("Auxiliary pooled-window PR curves (descriptive only)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        fig.tight_layout()
        pooled_path = state.save_figure(fig, "phase2_pr_curve_pooled_auxiliary.png")
    else:
        plt.close(fig)
        state.add_missing("Phase 2 auxiliary pooled PR curves: no arm had positive labels.")
        pooled_path = None
    return [path for path in (fold_path, pooled_path) if path is not None]


def _seed_tuple(value: Any, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return fallback
    values = value.split(",") if isinstance(value, str) else value
    try:
        result = tuple(int(item) for item in values)
    except (TypeError, ValueError):
        return ()
    return result if result and len(result) == len(set(result)) else ()


def _phase3_seed_policy(
    state: ReportState, phase: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    nbm = _seed_tuple(state.config.get("phase3_nbm_seeds"), (42,))
    policy = state.config.get("phase3_classifier_seed_policy")
    default = (42,) if phase == "3a" else (42, 43, 44)
    classifier = _seed_tuple(
        policy.get(phase) if isinstance(policy, dict) else None, default
    )
    if not nbm or not classifier:
        state.add_missing(f"Phase {phase.upper()} seed policy in config.json is invalid.")
        return nbm or (42,), classifier or default
    return nbm, classifier


def _read_phase3_subject_rows(
    state: ReportState, phase: str, table: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    root = state.result_dir / f"phase{phase}"
    expected_subjects = PHASE3A_SUBJECTS if phase == "3a" else MAIN_SUBJECTS
    nbm_seeds, classifier_seeds = _phase3_seed_policy(state, phase)
    expected_subject_keys = {
        (subject, arm) for subject in expected_subjects for arm in PHASE3_ARMS
    }
    expected_cell_keys = {
        (subject, arm, nbm_seed, classifier_seed)
        for subject in expected_subjects
        for arm in PHASE3_ARMS
        for nbm_seed in nbm_seeds
        for classifier_seed in classifier_seeds
    }

    cell_rows: list[dict[str, Any]] = []
    cell_csv = root / "classifier_cells.csv"
    if cell_csv.is_file():
        for raw in state.read_csv(cell_csv, f"phase{phase}_classifier_cells"):
            cell_rows.append(_normalise_metric_row(raw, state.relative_source(cell_csv)))
    else:
        for path in sorted(
            root.glob("loso_*/nbm_seed_*/classifier_seed_*/*/metrics.json")
        ):
            try:
                payload = state.read_json(path, f"phase{phase}_classifier_cell_metrics")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                state.add_warning(f"Skipped invalid Phase {phase} cell {path}: {exc}")
                continue
            cell_rows.append(
                _normalise_metric_row(payload, state.relative_source(path))
            )
    cell_groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    invalid_seed_rows = 0
    for row in cell_rows:
        nbm_seed, classifier_seed = _integer(row.get("nbm_seed")), _integer(
            row.get("classifier_seed")
        )
        if nbm_seed is None or classifier_seed is None:
            invalid_seed_rows += 1
            continue
        cell_groups[
            (str(row["test_subject"]), str(row["arm"]), nbm_seed, classifier_seed)
        ].append(row)
    duplicate_cells = sorted(key for key, values in cell_groups.items() if len(values) > 1)
    missing_cells = sorted(expected_cell_keys - set(cell_groups))
    extra_cells = sorted(set(cell_groups) - expected_cell_keys)
    if invalid_seed_rows or duplicate_cells or missing_cells or extra_cells:
        state.add_missing(
            f"Phase {phase.upper()} classifier seed Cartesian product is incomplete or non-unique "
            f"(invalid={invalid_seed_rows}, duplicate={len(duplicate_cells)}, "
            f"missing={len(missing_cells)}, extra={len(extra_cells)})."
        )

    subject_csv = root / "subject_seed_averaged_metrics.csv"
    raw_subject_rows: list[dict[str, Any]] = []
    if subject_csv.is_file():
        for raw in state.read_csv(
            subject_csv, f"phase{phase}_seed_averaged_subject_metrics"
        ):
            raw_subject_rows.append(
                _normalise_metric_row(raw, state.relative_source(subject_csv))
            )
    elif cell_rows:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in cell_rows:
            grouped[(str(row["test_subject"]), str(row["arm"]))].append(row)
        for (subject, arm), repetitions in sorted(grouped.items()):
            item: dict[str, Any] = {
                "test_subject": subject,
                "arm": arm,
                "repetitions": len(repetitions),
                "source": f"derived from Phase {phase.upper()} classifier cells",
            }
            for metric in (
                "pr_auc",
                "event_sensitivity",
                "false_alarm_events_per_hour",
                "fog_recall",
                "macro_f1",
            ):
                item[metric] = _mean(row.get(metric) for row in repetitions)
            raw_subject_rows.append(item)
        state.add_warning(
            f"Phase {phase.upper()} subject means were derived from available classifier cells because the aggregate CSV is absent."
        )
    else:
        state.add_missing(
            f"Phase {phase.upper()} seed-averaged comparison has no subject or cell metrics."
        )
        return []

    subject_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_subject_rows:
        subject_groups[(str(row["test_subject"]), str(row["arm"]))].append(row)
    duplicate_subject = sorted(
        key for key, values in subject_groups.items() if len(values) > 1
    )
    extra_subject = sorted(set(subject_groups) - expected_subject_keys)
    missing_subject = sorted(expected_subject_keys - set(subject_groups))
    expected_repetitions = len(nbm_seeds) * len(classifier_seeds)
    bad_repetitions = sorted(
        key
        for key, values in subject_groups.items()
        if key in expected_subject_keys
        and _integer(values[0].get("repetitions")) != expected_repetitions
    )
    if duplicate_subject or extra_subject or missing_subject or bad_repetitions:
        state.add_missing(
            f"Phase {phase.upper()} seed-averaged subject matrix is invalid "
            f"(duplicate={len(duplicate_subject)}, extra={len(extra_subject)}, "
            f"missing={len(missing_subject)}, wrong_repetitions={len(bad_repetitions)})."
        )
    rows = [
        subject_groups[key][0]
        for key in sorted(expected_subject_keys)
        if key in subject_groups
    ]

    aggregate_path = root / "aggregate.json"
    if aggregate_path.is_file():
        try:
            aggregate = state.read_json(aggregate_path, f"phase{phase}_aggregate")
            if aggregate.get("protocol_fingerprint") != state.config.get(
                "protocol_fingerprint"
            ):
                state.add_missing(
                    f"Phase {phase.upper()} aggregate protocol fingerprint mismatch."
                )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            state.add_missing(f"Phase {phase.upper()} aggregate is invalid ({exc}).")
    else:
        state.add_missing(f"Phase {phase.upper()} aggregate.json is missing.")

    for row in rows:
        subject, arm = str(row["test_subject"]), str(row["arm"])
        for metric in (
            "pr_auc",
            "event_sensitivity",
            "false_alarm_events_per_hour",
            "fog_recall",
            "macro_f1",
        ):
            value = _finite_float(row.get(metric))
            if value is not None:
                table.append(
                    _table_row(
                        section="crossfit_seed_averaged_subject_metric",
                        phase=phase,
                        subject=subject,
                        arm=arm,
                        arm_id=ARM_IDS.get(arm, ""),
                        metric=metric,
                        value=value,
                        repetitions=row.get("repetitions", ""),
                        notes="repetitions averaged within subject; inferential only when report certification is complete",
                        source=row["source"],
                    )
                )
        if _finite_float(row.get("pr_auc")) is None:
            state.add_missing(
                f"Phase {phase.upper()} {subject}/{arm}: seed-averaged PR-AUC is missing."
            )
    for arm in PHASE3_ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        for metric in (
            "pr_auc",
            "event_sensitivity",
            "false_alarm_events_per_hour",
            "fog_recall",
            "macro_f1",
        ):
            value = _mean(row.get(metric) for row in arm_rows)
            if value is not None:
                table.append(
                    _table_row(
                        section="crossfit_subject_macro",
                        phase=phase,
                        subject="subject_macro",
                        arm=arm,
                        arm_id=ARM_IDS.get(arm, ""),
                        metric=metric,
                        value=value,
                        n=len(arm_rows),
                        notes="unweighted mean of within-subject repetition means",
                        source=f"derived from Phase {phase.upper()} subject metrics",
                    )
                )
    by_key = {(str(row["test_subject"]), str(row["arm"])): row for row in rows}
    for reference in ("raw6", "raw4_zero"):
        comparison = f"raw4_normality_minus_{reference}"
        deltas = []
        for subject in expected_subjects:
            candidate, baseline = by_key.get((subject, "raw4_normality")), by_key.get(
                (subject, reference)
            )
            c = _finite_float(candidate.get("pr_auc")) if candidate else None
            b = _finite_float(baseline.get("pr_auc")) if baseline else None
            if c is None or b is None:
                continue
            deltas.append(c - b)
            table.append(
                _table_row(
                    section="crossfit_seed_averaged_subject_delta",
                    phase=phase,
                    subject=subject,
                    comparison=comparison,
                    metric="pr_auc_delta",
                    value=c - b,
                    notes="paired after within-subject repetition averaging",
                    source=f"derived from Phase {phase.upper()} subject metrics",
                )
            )
        if deltas:
            table.append(
                _table_row(
                    section="crossfit_subject_macro_delta",
                    phase=phase,
                    subject="subject_macro",
                    comparison=comparison,
                    metric="pr_auc_delta",
                    value=float(np.mean(deltas)),
                    n=len(deltas),
                    notes="unweighted paired-subject mean",
                    source=f"derived from Phase {phase.upper()} subject metrics",
                )
            )
    return rows


def _plot_phase3_comparison(
    state: ReportState,
    phase3a: Sequence[Mapping[str, Any]],
    phase3b: Sequence[Mapping[str, Any]],
) -> Path | None:
    available = [("3A", phase3a), ("3B", phase3b)]
    available = [(label, rows) for label, rows in available if rows]
    if not available:
        return None
    fig, axes = plt.subplots(1, len(available), figsize=(5.8 * len(available), 5.0), squeeze=False)
    plotted = 0
    for axis, (label, rows) in zip(axes.flat, available):
        for arm_index, arm in enumerate(PHASE3_ARMS):
            values = [
                value
                for row in rows
                if row.get("arm") == arm
                and (value := _finite_float(row.get("pr_auc"))) is not None
            ]
            if not values:
                continue
            plotted += 1
            axis.bar(
                arm_index,
                float(np.mean(values)),
                width=0.64,
                color=ARM_COLORS.get(arm),
                alpha=0.55,
            )
            jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0.0]
            axis.scatter(
                arm_index + np.asarray(jitter),
                values,
                color=ARM_COLORS.get(arm),
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
        axis.set_xticks(
            range(len(PHASE3_ARMS)),
            [f"{ARM_IDS[arm]}\n{ARM_LABELS[arm]}" for arm in PHASE3_ARMS],
        )
        axis.set_ylim(0.0, 1.02)
        axis.set_ylabel("Seed-averaged subject PR-AUC")
        axis.set_title(f"Phase {label}")
        axis.grid(axis="y", alpha=0.25)
    if not plotted:
        plt.close(fig)
        state.add_missing("Phase 3 seed-averaged comparison: no finite subject PR-AUC values.")
        return None
    fig.suptitle("Cross-fitted comparison: repetitions averaged within subject")
    fig.tight_layout()
    return state.save_figure(fig, "phase3_seed_averaged_pr_auc_comparison.png")


def _continuous_timeline(rows: Sequence[Mapping[str, str]]) -> tuple[np.ndarray, list[tuple[float, str]]]:
    x_values: list[float] = []
    boundaries: list[tuple[float, str]] = []
    offset = 0.0
    previous_key: tuple[str, str] | None = None
    previous_end = 0.0
    for row in rows:
        key = (str(row.get("record_id", "")), str(row.get("run_id", "")))
        start = _finite_float(row.get("target_start_sec"))
        end = _finite_float(row.get("target_end_exclusive_sec"))
        start = 0.0 if start is None else start
        end = start if end is None else end
        if previous_key is not None and key != previous_key:
            offset = previous_end + 1.0
            boundaries.append((offset, "/".join(value for value in key if value)))
        elif previous_key is None:
            boundaries.append((0.0, "/".join(value for value in key if value)))
        x_values.append(offset + start)
        previous_end = max(previous_end, offset + end)
        previous_key = key
    return np.asarray(x_values, dtype=np.float64), boundaries


def _read_external(
    state: ReportState, table: list[dict[str, Any]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    root = state.result_dir / "phase3b" / "external_negative_only"
    timeline_path = root / "subject_averaged_timeline.csv"
    subject_path = root / "subject_metrics.csv"
    timelines: list[dict[str, str]] = []
    subject_metrics: list[dict[str, str]] = []
    if timeline_path.exists():
        try:
            timelines = state.read_csv(timeline_path, "external_subject_averaged_timeline")
        except (OSError, csv.Error) as exc:
            state.add_missing(f"External timeline is invalid ({exc}).")
    if subject_path.exists():
        try:
            subject_metrics = state.read_csv(subject_path, "external_subject_metrics")
        except (OSError, csv.Error) as exc:
            state.add_missing(f"External subject metrics are invalid ({exc}).")
    else:
        state.add_missing("External S04/S10 subject_metrics.csv is missing.")
    aggregate_path = root / "aggregate.json"
    aggregate: dict[str, Any] = {}
    if aggregate_path.exists():
        try:
            aggregate = state.read_json(
                aggregate_path, "external_negative_only_aggregate"
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            state.add_missing(f"External aggregate is invalid ({exc}).")
    else:
        state.add_missing("External S04/S10 aggregate.json is missing.")
    done_path = root / "DONE.json"
    if done_path.is_file():
        try:
            state.read_json(done_path, "external_negative_only_done_manifest")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            state.add_missing(f"External DONE manifest is invalid ({exc}).")
    else:
        state.add_missing("External S04/S10 DONE.json is missing.")

    nbm_seeds, classifier_seeds = _phase3_seed_policy(state, "3b")
    expected_repetitions = (
        len(MAIN_SUBJECTS) * len(nbm_seeds) * len(classifier_seeds)
    )
    if (
        aggregate.get("status") != "complete"
        or set(aggregate.get("subjects", ())) != {"S04", "S10"}
        or set(aggregate.get("arms", ())) != set(PHASE3_ARMS)
        or _integer(aggregate.get("repetitions_per_subject_arm"))
        != expected_repetitions
        or aggregate.get("main_protocol_fingerprint")
        != state.config.get("protocol_fingerprint")
    ):
        state.add_missing(
            "External S04/S10 aggregate identity, status, or expected repetition count is invalid."
        )

    expected_keys = {
        (subject, arm) for subject in ("S04", "S10") for arm in PHASE3_ARMS
    }
    subject_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in subject_metrics:
        subject_groups[
            (str(row.get("external_subject", "")), str(row.get("arm", "")))
        ].append(row)
    duplicate_keys = {key for key, values in subject_groups.items() if len(values) > 1}
    missing_keys = expected_keys - set(subject_groups)
    extra_keys = set(subject_groups) - expected_keys
    wrong_repetitions = {
        key
        for key, values in subject_groups.items()
        if key in expected_keys
        and _integer(values[0].get("repetitions")) != expected_repetitions
    }
    if duplicate_keys or missing_keys or extra_keys or wrong_repetitions:
        state.add_missing(
            "External S04/S10 subject matrix is incomplete or non-unique "
            f"(duplicate={len(duplicate_keys)}, missing={len(missing_keys)}, "
            f"extra={len(extra_keys)}, wrong_repetitions={len(wrong_repetitions)})."
        )
    subject_metrics = [
        subject_groups[key][0] for key in sorted(expected_keys) if key in subject_groups
    ]
    for row in subject_metrics:
        subject, arm = str(row.get("external_subject", "")), str(row.get("arm", ""))
        for source_key, metric in (
            ("specificity_mean", "specificity"),
            ("positive_window_rate_mean", "positive_window_rate"),
            ("false_alarm_events_per_hour_mean", "false_alarm_events_per_hour"),
        ):
            value = _finite_float(row.get(source_key))
            if value is not None:
                table.append(
                    _table_row(
                        section="external_negative_only_subject_metric",
                        phase="3b_external",
                        subject=subject,
                        arm=arm,
                        arm_id=ARM_IDS.get(arm, ""),
                        metric=metric,
                        value=value,
                        repetitions=row.get("repetitions", ""),
                        notes="negative-only; model repetitions averaged within external subject",
                        source=state.relative_source(subject_path),
                    )
                )
    if not timelines:
        state.add_missing(
            "External S04/S10 full timeline: phase3b/external_negative_only/subject_averaged_timeline.csv is missing."
        )
    elif timelines:
        required = {
            "external_subject",
            "arm",
            "record_id",
            "run_id",
            "target_start_sec",
            "mean_y_prob",
            "positive_vote_rate",
            "consensus_y_pred",
            "window_index",
        }
        absent = sorted(required - set(timelines[0]))
        if absent:
            state.add_missing(
                "External S04/S10 timeline lacks required columns: "
                + ", ".join(absent)
                + "."
            )
            timelines = []
        else:
            invalid_rows = sum(
                not str(row.get("external_subject", ""))
                or not str(row.get("arm", ""))
                or _finite_float(row.get("target_start_sec")) is None
                or _finite_float(row.get("mean_y_prob")) is None
                or _finite_float(row.get("positive_vote_rate")) is None
                or _finite_float(row.get("consensus_y_pred")) is None
                for row in timelines
            )
            if invalid_rows:
                state.add_missing(
                    f"External S04/S10 timeline has {invalid_rows} row(s) with incomplete identity or non-finite predictions."
                )
                timelines = []
    if timelines:
        timeline_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in timelines:
            timeline_groups[
                (str(row.get("external_subject", "")), str(row.get("arm", "")))
            ].append(row)
        if set(timeline_groups) != expected_keys:
            state.add_missing(
                "External S04/S10 timeline does not contain the exact 2-subject x 3-arm matrix."
            )
        endpoint_sets: dict[tuple[str, str], set[int]] = {}
        for key, rows in timeline_groups.items():
            indices = [_integer(row.get("window_index")) for row in rows]
            valid_indices = {value for value in indices if value is not None}
            endpoint_sets[key] = valid_indices
            expected_windows = None
            if key in subject_groups:
                expected_windows = _integer(
                    subject_groups[key][0].get("consensus_n_negative_windows")
                )
            if (
                len(valid_indices) != len(rows)
                or expected_windows is None
                or len(rows) != expected_windows
            ):
                state.add_missing(
                    f"External timeline {key[0]}/{key[1]} is truncated or has duplicate endpoints."
                )
        for subject in ("S04", "S10"):
            arm_sets = [
                endpoint_sets.get((subject, arm), set()) for arm in PHASE3_ARMS
            ]
            if arm_sets and any(values != arm_sets[0] for values in arm_sets[1:]):
                state.add_missing(
                    f"External timeline endpoints differ across arms for {subject}."
                )
    return timelines, subject_metrics


def _plot_external_timelines(
    state: ReportState, rows: Sequence[Mapping[str, str]]
) -> list[Path]:
    outputs = []
    for subject in ("S04", "S10"):
        subject_rows = [row for row in rows if row.get("external_subject") == subject]
        if not subject_rows:
            state.add_missing(f"External timeline contains no rows for {subject}.")
            continue
        arms = [arm for arm in PHASE3_ARMS if any(row.get("arm") == arm for row in subject_rows)]
        missing_arms = [arm for arm in PHASE3_ARMS if arm not in arms]
        if missing_arms:
            state.add_missing(
                f"External timeline {subject} lacks arms: {', '.join(missing_arms)}."
            )
        if not arms:
            state.add_missing(f"External timeline contains no recognized arms for {subject}.")
            continue
        fig, axes = plt.subplots(len(arms), 1, figsize=(12.5, 2.7 * len(arms)), sharex=True, squeeze=False)
        for axis, arm in zip(axes.flat, arms):
            arm_rows = [row for row in subject_rows if row.get("arm") == arm]
            x, boundaries = _continuous_timeline(arm_rows)
            probability = np.asarray(
                [np.nan if (value := _finite_float(row.get("mean_y_prob"))) is None else value for row in arm_rows]
            )
            vote = np.asarray(
                [np.nan if (value := _finite_float(row.get("positive_vote_rate"))) is None else value for row in arm_rows]
            )
            prediction = np.asarray(
                [np.nan if (value := _finite_float(row.get("consensus_y_pred"))) is None else value for row in arm_rows]
            )
            axis.plot(x, probability, color=ARM_COLORS.get(arm), linewidth=1.0, label="mean probability")
            if np.isfinite(vote).any():
                axis.plot(x, vote, color="#ff7f0e", linewidth=0.8, alpha=0.8, label="positive vote rate")
            if np.isfinite(prediction).any():
                axis.step(x, prediction, where="post", color="black", linewidth=0.7, alpha=0.65, label="consensus prediction")
            for boundary, _ in boundaries[1:]:
                axis.axvline(boundary, color="#bbbbbb", linewidth=0.6)
            axis.set_ylim(-0.03, 1.03)
            axis.set_ylabel(f"{ARM_IDS.get(arm, arm)}")
            axis.grid(alpha=0.2)
            axis.legend(loc="upper right", fontsize=7, ncol=3)
        axes.flat[-1].set_xlabel("Cumulative time across runs (s; run boundaries marked)")
        fig.suptitle(f"{subject} negative-only full prediction timeline")
        fig.tight_layout()
        outputs.append(state.save_figure(fig, f"external_{subject}_timeline.png"))
    return outputs


def _phase2_summary(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return ["Phase 2 metrics are unavailable."]
    lines = ["| Arm | ID | Subjects | Subject-macro PR-AUC | Event sensitivity | FA/h |", "|---|---:|---:|---:|---:|---:|"]
    for arm in PHASE2_ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        if not arm_rows:
            continue
        lines.append(
            "| "
            + " | ".join(
                (
                    ARM_LABELS.get(arm, arm),
                    ARM_IDS.get(arm, ""),
                    str(len(arm_rows)),
                    _format(_mean(row.get("pr_auc") for row in arm_rows)),
                    _format(_mean(row.get("event_sensitivity") for row in arm_rows)),
                    _format(_mean(row.get("false_alarm_events_per_hour") for row in arm_rows)),
                )
            )
            + " |"
        )
    by_key = _phase2_by_key(rows)
    lines.append("")
    for reference, label in (("raw6", "F4-F1"), ("raw4_zero", "F4-F3")):
        deltas = []
        for subject in MAIN_SUBJECTS:
            candidate = by_key.get((subject, "raw4_normality"))
            baseline = by_key.get((subject, reference))
            c = _finite_float(candidate.get("pr_auc")) if candidate else None
            b = _finite_float(baseline.get("pr_auc")) if baseline else None
            if c is not None and b is not None:
                deltas.append(c - b)
        lines.append(
            f"- {label} subject-macro PR-AUC delta: {_format(_mean(deltas))} "
            f"over {len(deltas)} paired subject(s)."
        )
    return lines


def _phase3_summary(phase: str, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return [f"Phase {phase.upper()} results are unavailable."]
    lines = ["| Arm | Subjects | Seed-averaged subject-macro PR-AUC | Event sensitivity | FA/h |", "|---|---:|---:|---:|---:|"]
    for arm in PHASE3_ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        if not arm_rows:
            continue
        lines.append(
            "| "
            + " | ".join(
                (
                    ARM_LABELS.get(arm, arm),
                    str(len(arm_rows)),
                    _format(_mean(row.get("pr_auc") for row in arm_rows)),
                    _format(_mean(row.get("event_sensitivity") for row in arm_rows)),
                    _format(_mean(row.get("false_alarm_events_per_hour") for row in arm_rows)),
                )
            )
            + " |"
        )
    return lines


def _build_report(
    state: ReportState,
    *,
    phase0_records: Sequence[Mapping[str, Any]],
    phase0_aggregate: Mapping[str, Any] | None,
    phase1_aggregate: Mapping[str, Any] | None,
    phase2_rows: Sequence[Mapping[str, Any]],
    phase3a_rows: Sequence[Mapping[str, Any]],
    phase3b_rows: Sequence[Mapping[str, Any]],
    external_rows: Sequence[Mapping[str, str]],
) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    status = (
        "complete" if state.certified_complete and not state.missing else "partial"
    )
    figure_names = {
        path.name for path in state.outputs if path.parent == state.figures_dir
    }
    lines = [
        "# Daphnet GRU-H200 residual-fusion feasibility report",
        "",
        f"- Report status: **{status}**",
        f"- Generated (UTC): `{generated}`",
        f"- Result directory: `{state.result_dir}`",
        "- Inference rule: subjects are the independent units; pooled-window PR curves are descriptive only.",
        "- Memory rule: this report did not load any H200 primitive cache.",
        "",
        "## Completeness and missing artifacts",
        "",
    ]
    if state.missing:
        lines.extend(f"- MISSING: {message}" for message in state.missing)
    else:
        lines.append("- All report-level preregistered artifacts were found.")
    if state.warnings:
        lines.extend(("", "### Warnings", ""))
        lines.extend(f"- {message}" for message in state.warnings)

    lines.extend(("", "## Phase 0: normal forecast diagnostics", ""))
    subjects = sorted({str(row.get("subject")) for row in phase0_records if row.get("model") == "gru"})
    lines.append(f"Lead-quartile diagnostics are available for {len(subjects)} fold(s): {', '.join(subjects) or 'none'}.")
    if phase0_aggregate:
        lines.append(
            f"Aggregate decision: `{phase0_aggregate.get('decision', 'not recorded')}`; "
            f"GRU better than persistence on RMSE in "
            f"`{phase0_aggregate.get('gru_better_rmse_subjects', 'missing')}` fold(s)."
        )
    if "phase0_lead_quartile_diagnostics.png" in figure_names:
        lines.append("![Phase 0 lead diagnostics](figures/phase0_lead_quartile_diagnostics.png)")

    lines.extend(("", "## Phase 1: engineering smoke", ""))
    if phase1_aggregate is None:
        lines.append("Phase 1 S01 four-arm engineering smoke is unavailable.")
    else:
        smoke = phase1_aggregate.get("smoke_gate") or {}
        lines.append(
            f"S01 four-arm smoke gate: `{smoke.get('status', 'missing')}`. "
            "This phase verifies implementation behavior only and is not scientific evidence."
        )

    lines.extend(("", "## Phase 2: exploratory residual fusion", ""))
    lines.extend(_phase2_summary(phase2_rows))
    phase2_links = (
        ("phase2_subject_waterfalls.png", "Subject waterfalls"),
        ("phase2_subject_pr_auc.png", "Subject PR-AUC"),
        ("phase2_false_alarm_vs_event_sensitivity.png", "Operating points"),
        ("phase2_pr_curves_by_fold.png", "Per-fold PR curves"),
        ("phase2_pr_curve_pooled_auxiliary.png", "Auxiliary pooled PR curve"),
    )
    for filename, label in phase2_links:
        if filename in figure_names:
            lines.extend(("", f"![{label}](figures/{filename})"))
    if "phase2_pr_curve_pooled_auxiliary.png" in figure_names:
        lines.append(
            "The pooled curve is an auxiliary visualization only; it does not replace subject-macro reporting or paired subject inference."
        )

    lines.extend(("", "## Phase 3A: cross-fitted directional check", ""))
    lines.extend(_phase3_summary("3a", phase3a_rows))
    lines.extend(("", "## Phase 3B: full cross-fitted confirmation", ""))
    lines.extend(_phase3_summary("3b", phase3b_rows))
    if "phase3_seed_averaged_pr_auc_comparison.png" in figure_names:
        lines.extend(
            (
                "",
                "![Phase 3 seed-averaged comparison](figures/phase3_seed_averaged_pr_auc_comparison.png)",
                "",
                "All displayed Phase 3 comparisons first average NBM/classifier repetitions within each test subject.",
            )
        )

    lines.extend(("", "## S04/S10 negative-only external evaluation", ""))
    external_subjects = sorted({str(row.get("external_subject")) for row in external_rows})
    if external_subjects:
        qualifier = "certified full" if state.certified_complete else "partial preview"
        lines.append(
            f"Available {qualifier} subject-averaged timelines: "
            + ", ".join(external_subjects)
            + "."
        )
        for subject in ("S04", "S10"):
            if f"external_{subject}_timeline.png" in figure_names:
                lines.extend(("", f"![{subject} timeline](figures/external_{subject}_timeline.png)"))
    else:
        lines.append(
            "MISSING: S04/S10 external subject-averaged timeline was not found; no external performance claim is made."
        )

    lines.extend(
        (
            "",
            "## Machine-readable outputs",
            "",
            "- `publication_tables.csv`: long-form subject, macro, paired-delta, forecast and external metrics.",
            "- `report_manifest.json`: exact SHA-256 hashes for every consumed source and generated output except the manifest itself.",
            "",
        )
    )
    return "\n".join(lines)


def _manifest(state: ReportState, generated_at: str) -> dict[str, Any]:
    sources = []
    for path in sorted(state.sources, key=lambda item: state.relative_source(item)):
        sources.append(
            {
                "path": state.relative_source(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "roles": sorted(state.sources[path]),
            }
        )
    outputs = []
    for path in sorted(set(state.outputs), key=lambda item: item.as_posix()):
        outputs.append(
            {
                "path": path.relative_to(state.output_dir).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "report_version": REPORT_VERSION,
        "generated_at_utc": generated_at,
        "result_dir": str(state.result_dir),
        "output_dir": str(state.output_dir),
        "status": (
            "complete" if state.certified_complete and not state.missing else "partial"
        ),
        "certified_complete": bool(state.certified_complete),
        "primitive_caches_loaded": False,
        "source_files": sources,
        "generated_files": outputs,
        "missing": state.missing,
        "warnings": state.warnings,
    }


def build_report(result_dir: Path, output_dir: Path, *, dpi: int = 140) -> dict[str, Any]:
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Result directory does not exist: {result_dir}")
    if int(dpi) < 72:
        raise ValueError("dpi must be at least 72")
    output_dir.mkdir(parents=True, exist_ok=True)
    state = ReportState(result_dir, output_dir, dpi)
    state.load_certification_inputs()
    table: list[dict[str, Any]] = []

    phase0_records, phase0_aggregate = _read_phase0(state, table)
    _plot_phase0(state, phase0_records)
    phase1_aggregate = _read_phase1(state)
    phase2_rows = _read_phase2(state, table)
    _plot_phase2_waterfalls(state, phase2_rows)
    _plot_phase2_pr_auc(state, phase2_rows)
    _plot_phase2_operating_points(state, phase2_rows)
    predictions = _load_phase2_predictions(state, phase2_rows)
    _plot_pr_curves(state, predictions, table)
    phase3a_rows = _read_phase3_subject_rows(state, "3a", table)
    phase3b_rows = _read_phase3_subject_rows(state, "3b", table)
    _plot_phase3_comparison(state, phase3a_rows, phase3b_rows)
    external_rows, _ = _read_external(state, table)
    _plot_external_timelines(state, external_rows)
    state.finalize_certification()

    publication_path = state.output_dir / "publication_tables.csv"
    _atomic_csv(publication_path, table)
    state.outputs.append(publication_path)
    report_path = state.output_dir / "REPORT.md"
    _atomic_text(
        report_path,
        _build_report(
            state,
            phase0_records=phase0_records,
            phase0_aggregate=phase0_aggregate,
            phase1_aggregate=phase1_aggregate,
            phase2_rows=phase2_rows,
            phase3a_rows=phase3a_rows,
            phase3b_rows=phase3b_rows,
            external_rows=external_rows,
        ),
    )
    state.outputs.append(report_path)
    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = _manifest(state, generated_at)
    manifest_path = state.output_dir / "report_manifest.json"
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a read-only, publication-oriented report from Daphnet GRU-H200 feasibility outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <result-dir>/feasibility_report",
    )
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result_dir = args.result_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else result_dir / "feasibility_report"
    )
    manifest = build_report(result_dir, output_dir, dpi=args.dpi)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_dir": str(output_dir),
                "sources": len(manifest["source_files"]),
                "missing": len(manifest["missing"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
