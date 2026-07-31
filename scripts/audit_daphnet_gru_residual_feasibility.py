#!/usr/bin/env python
"""Independent, read-only audit of the four-phase H200 feasibility suite.

The experiment runner is intentionally not imported here.  This keeps result
verification independent from training code and lets the audit run in a plain
Python environment without Torch, NumPy, or the original Daphnet files.

Only ``audit_report.json`` is written.  Every other operation is read-only.
Missing cells are reported separately from integrity errors; ``--allow-incomplete``
makes an in-progress matrix non-fatal, but never hides malformed or tampered
artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SUITE_VERSION = "daphnet_gru_h200_residual_feasibility.v1"
EXPECTED_SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
PHASE1_SUBJECTS = ("S01",)
PHASE3A_SUBJECTS = ("S01", "S05", "S08")
PHASE1_ARMS = ("raw4", "normality", "raw4_zero", "raw4_normality")
PHASE2_ARMS = ("raw4", "raw6", "normality", "raw4_zero", "raw4_normality")
PHASE3_ARMS = ("raw6", "raw4_zero", "raw4_normality")
VISUAL_GROUPS = (
    "clean_nonfog_first",
    "fog_onset_first",
    "clean_nonfog_high_residual",
)
RUNTIME_CONFIG_FIELDS = {
    "protocol_fingerprint",
    "data_dir",
    "source_suite_dir",
    "output_dir",
    "device",
    "resume",
    "force_next_phase",
    "force_phase3_representation_gate",
    "phase3_external_batch_size",
}
METRIC_COLUMNS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "fog_recall",
    "specificity",
    "precision",
    "fog_f1",
    "mcc",
    "event_sensitivity",
    "false_alarm_events_per_hour",
    "median_detection_delay_sec",
)


def _canonical_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _is_hex_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _parse_seed_values(value: Any, fallback: Sequence[int]) -> tuple[int, ...]:
    if value is None:
        return tuple(int(item) for item in fallback)
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
    else:
        values = [value]
    try:
        result = tuple(int(item) for item in values)
    except (TypeError, ValueError):
        return ()
    return result if len(set(result)) == len(result) else ()


@dataclass
class AuditState:
    root: Path
    allow_incomplete: bool
    findings: list[dict[str, Any]] = field(default_factory=list)
    matrices: dict[str, Any] = field(default_factory=dict)
    checked_done: set[Path] = field(default_factory=set)

    def _finding(self, kind: str, code: str, message: str, path: Path | None = None) -> None:
        item: dict[str, Any] = {"kind": kind, "code": code, "message": message}
        if path is not None:
            try:
                item["path"] = path.resolve().relative_to(self.root.resolve()).as_posix()
            except ValueError:
                item["path"] = str(path.resolve())
        self.findings.append(item)

    def error(self, code: str, message: str, path: Path | None = None) -> None:
        self._finding("error", code, message, path)

    def incomplete(self, code: str, message: str, path: Path | None = None) -> None:
        self._finding("incomplete", code, message, path)

    def warning(self, code: str, message: str, path: Path | None = None) -> None:
        self._finding("warning", code, message, path)

    def require_file(self, path: Path, code: str, description: str) -> bool:
        if not path.is_file():
            self.incomplete(code, f"missing {description}", path)
            return False
        return True

    def load_json(
        self, path: Path, *, required: bool = True, code: str = "json_missing"
    ) -> dict[str, Any] | None:
        if not path.is_file():
            if required:
                self.incomplete(code, "missing JSON artifact", path)
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            self.error("json_invalid", f"cannot parse JSON: {error}", path)
            return None
        if not isinstance(payload, dict):
            self.error("json_not_object", "top-level JSON value must be an object", path)
            return None
        return payload

    def check_done(
        self,
        path: Path,
        *,
        protocol_fingerprint: str | None = None,
        stage: str | None = None,
        task_id: str | None = None,
        upstream_sha256: str | None = None,
        required_artifacts: Iterable[str] = (),
        required: bool = True,
    ) -> tuple[dict[str, Any] | None, dict[str, Path]]:
        if not path.is_file():
            if required:
                self.incomplete("done_missing", "missing DONE manifest", path)
            return None, {}
        resolved_done = path.resolve()
        self.checked_done.add(resolved_done)
        payload = self.load_json(path, code="done_missing")
        if payload is None:
            return None, {}
        if payload.get("format_version") != 1:
            self.error("done_format", "DONE format_version must equal 1", path)
        if not isinstance(payload.get("stage"), str) or not payload.get("stage"):
            self.error("done_stage", "DONE stage is missing", path)
        if stage is not None and payload.get("stage") != stage:
            self.error(
                "done_stage",
                f"DONE stage {payload.get('stage')!r} != {stage!r}",
                path,
            )
        if protocol_fingerprint is not None and payload.get("protocol_fingerprint") != protocol_fingerprint:
            self.error("done_protocol", "DONE protocol fingerprint mismatch", path)
        if task_id is not None and payload.get("task_id") != task_id:
            self.error(
                "done_task", f"DONE task {payload.get('task_id')!r} != {task_id!r}", path
            )
        if (
            upstream_sha256 is not None
            and payload.get("upstream_nbm_sha256") != upstream_sha256
        ):
            self.error("done_upstream", "DONE upstream SHA-256 mismatch", path)
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            self.error("done_artifacts", "DONE artifacts must be a non-empty object", path)
            return payload, {}
        for name in required_artifacts:
            if name not in artifacts:
                self.error("done_artifact_missing", f"DONE lacks required artifact {name!r}", path)
        resolved: dict[str, Path] = {}
        root_resolved = self.root.resolve()
        for name, metadata in artifacts.items():
            if not isinstance(metadata, dict):
                self.error("done_artifact_metadata", f"artifact {name!r} metadata is invalid", path)
                continue
            raw_path = metadata.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                self.error("done_artifact_path", f"artifact {name!r} has no path", path)
                continue
            artifact_path = Path(raw_path)
            if not artifact_path.is_absolute():
                artifact_path = path.parent / artifact_path
            artifact_path = artifact_path.resolve()
            try:
                artifact_path.relative_to(root_resolved)
            except ValueError:
                self.error(
                    "done_artifact_escape",
                    f"artifact {name!r} points outside the result directory",
                    path,
                )
                continue
            resolved[str(name)] = artifact_path
            if not artifact_path.is_file():
                self.error("done_artifact_absent", f"artifact {name!r} is missing", artifact_path)
                continue
            expected_bytes = metadata.get("bytes")
            if not isinstance(expected_bytes, int) or expected_bytes < 0:
                self.error("done_artifact_bytes", f"artifact {name!r} has invalid byte count", path)
            elif artifact_path.stat().st_size != expected_bytes:
                self.error("done_size_mismatch", f"artifact {name!r} size mismatch", artifact_path)
            expected_sha = metadata.get("sha256")
            if not _is_hex_sha256(expected_sha):
                self.error("done_artifact_sha", f"artifact {name!r} has invalid SHA-256", path)
            elif _sha256_file(artifact_path) != expected_sha:
                self.error("done_hash_mismatch", f"artifact {name!r} SHA-256 mismatch", artifact_path)
        return payload, resolved


def _audit_config(state: AuditState) -> tuple[dict[str, Any], str]:
    config_path = state.root / "config.json"
    config = state.load_json(config_path, code="config_missing") or {}
    fingerprint = config.get("protocol_fingerprint")
    if not _is_hex_sha256(fingerprint):
        state.error("config_fingerprint", "config protocol_fingerprint is not a SHA-256", config_path)
        fingerprint = ""
    scientific = {
        key: value for key, value in config.items() if key not in RUNTIME_CONFIG_FIELDS
    }
    recomputed = _canonical_fingerprint(scientific)
    if fingerprint and recomputed != fingerprint:
        state.error(
            "config_fingerprint_mismatch",
            "protocol_fingerprint does not match the canonical scientific config",
            config_path,
        )
    if config.get("suite_version") != SUITE_VERSION:
        state.error("config_suite", f"unexpected suite_version {config.get('suite_version')!r}", config_path)
    subjects = tuple(str(item) for item in config.get("subjects", ()))
    if subjects != EXPECTED_SUBJECTS:
        state.error("config_subjects", f"subjects must be {EXPECTED_SUBJECTS}", config_path)
    folds = tuple(str(item) for item in config.get("folds_resolved", ()))
    if folds != EXPECTED_SUBJECTS:
        state.incomplete("config_folds", "full protocol requires all eight LOSO folds", config_path)
    expected_arms = {
        "phase1_arms": PHASE1_ARMS,
        "phase2_arms": PHASE2_ARMS,
        "phase3_arms": PHASE3_ARMS,
    }
    for key, expected in expected_arms.items():
        if tuple(config.get(key, ())) != expected:
            state.error("config_arms", f"{key} must equal {expected}", config_path)
    return config, str(fingerprint)


def _check_metrics_identity(
    state: AuditState,
    metrics: Mapping[str, Any],
    *,
    path: Path,
    subject: str,
    arm: str,
    phase: str,
) -> None:
    expected = {"test_subject": subject, "arm": arm, "phase": phase}
    for key, value in expected.items():
        if str(metrics.get(key)) != value:
            state.error("cell_identity", f"{key}={metrics.get(key)!r}, expected {value!r}", path)
    for key in ("endpoint_sha256", "label_sha256"):
        if not _is_hex_sha256(metrics.get(key)):
            state.error("cell_digest", f"{key} is not a SHA-256", path)


def _audit_phase0(state: AuditState, config: Mapping[str, Any], fingerprint: str) -> None:
    phase_root = state.root / "phase0"
    matrix: dict[str, str] = {}
    plots_enabled = bool(config.get("phase0_plots", True))
    per_group = int(config.get("phase0_plot_windows", 5))
    for subject in EXPECTED_SUBJECTS:
        root = phase_root / f"loso_{subject}"
        metrics_path = root / "metrics.json"
        if not metrics_path.is_file():
            state.incomplete("phase0_cell", f"Phase 0 fold {subject} is incomplete", metrics_path)
            matrix[subject] = "missing"
            continue
        metrics = state.load_json(metrics_path) or {}
        _, artifacts = state.check_done(
            root / "DONE.json",
            protocol_fingerprint=fingerprint,
            stage="phase0_diagnostics",
            task_id=f"phase0/loso_{subject}",
            required_artifacts=(
                "metrics",
                "persistence_sigma",
                *(("figure_manifest",) if plots_enabled else ()),
            ),
        )
        matrix[subject] = "complete"
        if metrics.get("subject") != subject:
            state.error("phase0_subject", "Phase 0 subject identity mismatch", metrics_path)
        for field_name in ("identity_checks", "hard_checks"):
            checks = metrics.get(field_name)
            if not isinstance(checks, dict) or not checks:
                state.error("phase0_checks", f"missing {field_name}", metrics_path)
            elif not all(value is True for value in checks.values()):
                failed = [name for name, value in checks.items() if value is not True]
                state.error("phase0_hard_check", f"failed {field_name}: {failed}", metrics_path)
        visual = metrics.get("visualizations")
        if plots_enabled:
            if not isinstance(visual, dict) or visual.get("enabled") is not True:
                state.error("phase0_visuals", "configured Phase 0 figures are not enabled", metrics_path)
                continue
            manifest_relative = visual.get("manifest")
            if not isinstance(manifest_relative, str):
                state.error("phase0_visual_manifest", "visual manifest path is missing", metrics_path)
                continue
            manifest_path = (root / manifest_relative).resolve()
            manifest = state.load_json(manifest_path, code="phase0_visual_manifest_missing") or {}
            if manifest.get("schema_version") != 1:
                state.error("phase0_visual_schema", "visual schema_version must equal 1", manifest_path)
            if int(manifest.get("per_group_requested", -1)) != per_group:
                state.error("phase0_visual_count", "visual per-group request differs from config", manifest_path)
            counts = manifest.get("selected_counts")
            selections = manifest.get("selections")
            if not isinstance(counts, dict) or not isinstance(selections, dict):
                state.error("phase0_visual_manifest", "visual counts/selections are missing", manifest_path)
                continue
            if set(counts) != set(VISUAL_GROUPS) or set(selections) != set(VISUAL_GROUPS):
                state.error("phase0_visual_groups", "visual groups differ from the preregistered groups", manifest_path)
            artifact_paths = set(artifacts.values())
            for group in VISUAL_GROUPS:
                rows = selections.get(group, [])
                if counts.get(group) != per_group or not isinstance(rows, list) or len(rows) != per_group:
                    state.error("phase0_visual_count", f"group {group} does not contain {per_group} figures", manifest_path)
                    continue
                for rank, row in enumerate(rows, start=1):
                    if not isinstance(row, dict) or row.get("selection_group") != group or row.get("selection_rank") != rank:
                        state.error("phase0_visual_selection", f"invalid {group} selection rank {rank}", manifest_path)
                        continue
                    raw_figure = row.get("figure_path")
                    if not isinstance(raw_figure, str):
                        state.error("phase0_visual_path", "figure_path is missing", manifest_path)
                        continue
                    figure_path = (manifest_path.parent / raw_figure).resolve()
                    if not figure_path.is_file():
                        state.error("phase0_visual_absent", "visual figure is missing", figure_path)
                    if figure_path not in artifact_paths:
                        state.error("phase0_visual_unsealed", "visual figure is not sealed by DONE", figure_path)
        elif isinstance(visual, dict) and visual.get("enabled") is not False:
            state.error("phase0_visuals", "figures disabled in config but enabled in metrics", metrics_path)
    aggregate_path = phase_root / "aggregate.json"
    aggregate = state.load_json(aggregate_path, code="phase0_aggregate_missing")
    if aggregate is not None:
        state.check_done(
            phase_root / "DONE.json",
            protocol_fingerprint=fingerprint,
            stage="phase0_aggregate",
            task_id="phase0/aggregate",
            required_artifacts=("aggregate",),
        )
        if set(aggregate.get("completed_folds", ())) != set(EXPECTED_SUBJECTS):
            state.incomplete("phase0_aggregate_folds", "Phase 0 aggregate does not contain eight folds", aggregate_path)
        if aggregate.get("all_hard_checks_pass") is not True:
            state.error("phase0_aggregate_checks", "Phase 0 aggregate hard checks did not pass", aggregate_path)
    state.matrices["phase0"] = matrix


def _audit_classifier_phase(
    state: AuditState,
    *,
    phase: str,
    subjects: Sequence[str],
    arms: Sequence[str],
    fingerprint: str,
) -> dict[str, dict[str, Mapping[str, Any]]]:
    phase_root = state.root / f"phase{phase}"
    actual_subjects = {
        path.name.removeprefix("loso_")
        for path in phase_root.glob("loso_*")
        if path.is_dir()
    }
    unexpected_subjects = actual_subjects - set(subjects)
    if unexpected_subjects:
        state.error(
            "classifier_subject_scope",
            f"Phase {phase} contains unexpected LOSO subjects {sorted(unexpected_subjects)}",
            phase_root,
        )
    matrix: dict[str, dict[str, str]] = {}
    rows: dict[str, dict[str, Mapping[str, Any]]] = {subject: {} for subject in subjects}
    for subject in subjects:
        matrix[subject] = {}
        subject_root = phase_root / f"loso_{subject}"
        unexpected_arms = {
            path.name for path in subject_root.iterdir() if path.is_dir()
        } - set(arms) if subject_root.is_dir() else set()
        if unexpected_arms:
            state.error(
                "classifier_arm_scope",
                f"Phase {phase}/{subject} contains unexpected arms {sorted(unexpected_arms)}",
                subject_root,
            )
        for arm in arms:
            root = phase_root / f"loso_{subject}" / arm
            metrics_path = root / "metrics.json"
            if not metrics_path.is_file():
                state.incomplete("classifier_cell", f"Phase {phase} cell {subject}/{arm} is incomplete", metrics_path)
                matrix[subject][arm] = "missing"
                continue
            metrics = state.load_json(metrics_path) or {}
            state.check_done(
                root / "DONE.json",
                protocol_fingerprint=fingerprint,
                stage="h200_classifier",
                task_id=f"phase{phase}/loso_{subject}/{arm}",
                required_artifacts=("metrics", "predictions"),
            )
            _check_metrics_identity(state, metrics, path=metrics_path, subject=subject, arm=arm, phase=phase)
            rows[subject][arm] = metrics
            matrix[subject][arm] = "complete"
        complete_rows = rows[subject]
        if len(complete_rows) > 1:
            endpoints = {str(row.get("endpoint_sha256")) for row in complete_rows.values()}
            labels = {str(row.get("label_sha256")) for row in complete_rows.values()}
            if len(endpoints) != 1:
                state.error("endpoint_mismatch", f"Phase {phase}/{subject} endpoints differ by arm", phase_root / f"loso_{subject}")
            if len(labels) != 1:
                state.error("label_mismatch", f"Phase {phase}/{subject} labels differ by arm", phase_root / f"loso_{subject}")
    state.matrices[f"phase{phase}"] = matrix
    return rows


def _audit_phase1(state: AuditState, fingerprint: str) -> None:
    phase_root = state.root / "phase1"
    extra = sorted(
        path.name for path in phase_root.glob("loso_*") if path.is_dir() and path.name != "loso_S01"
    )
    if extra:
        state.error("phase1_subject_scope", f"Phase 1 must be fixed to S01, found {extra}", phase_root)
    rows = _audit_classifier_phase(
        state, phase="1", subjects=PHASE1_SUBJECTS, arms=PHASE1_ARMS, fingerprint=fingerprint
    )
    zero = rows["S01"].get("raw4_zero")
    fusion = rows["S01"].get("raw4_normality")
    if zero is not None and fusion is not None:
        if zero.get("parameter_count") != fusion.get("parameter_count"):
            state.error("capacity_mismatch", "Phase 1 zero/fusion parameter counts differ", phase_root)
        if zero.get("initial_state_sha256") != fusion.get("initial_state_sha256"):
            state.error("initialization_mismatch", "Phase 1 zero/fusion initial hashes differ", phase_root)
    aggregate_path = phase_root / "aggregate.json"
    aggregate = state.load_json(aggregate_path, code="phase1_aggregate_missing")
    if aggregate is not None:
        state.check_done(
            phase_root / "DONE.json",
            protocol_fingerprint=fingerprint,
            stage="phase1_aggregate",
            task_id="phase1/aggregate",
            required_artifacts=("aggregate", "fold_metrics"),
        )
        if aggregate.get("completed") is not True or aggregate.get("folds") != ["S01"]:
            state.incomplete("phase1_aggregate", "Phase 1 aggregate is not the complete fixed-S01 smoke test", aggregate_path)
        smoke = aggregate.get("smoke_gate")
        if not isinstance(smoke, dict) or smoke.get("status") != "pass":
            state.error("phase1_smoke_gate", "Phase 1 engineering smoke gate did not pass", aggregate_path)


def _audit_phase2(state: AuditState, fingerprint: str) -> None:
    rows = _audit_classifier_phase(
        state, phase="2", subjects=EXPECTED_SUBJECTS, arms=PHASE2_ARMS, fingerprint=fingerprint
    )
    phase_root = state.root / "phase2"
    aggregate_path = phase_root / "aggregate.json"
    aggregate = state.load_json(aggregate_path, code="phase2_aggregate_missing")
    gate_path = phase_root / "gate.json"
    gate = state.load_json(gate_path, code="phase2_gate_missing")
    if aggregate is not None:
        state.check_done(
            phase_root / "DONE.json",
            protocol_fingerprint=fingerprint,
            stage="phase2_aggregate",
            task_id="phase2/aggregate",
            required_artifacts=("aggregate", "fold_metrics", "gate"),
        )
        if aggregate.get("completed") is not True:
            state.incomplete("phase2_aggregate", "Phase 2 aggregate is incomplete", aggregate_path)
        if gate is not None and not _json_equal(aggregate.get("gate"), gate):
            state.error("phase2_gate_mismatch", "aggregate gate differs from gate.json", aggregate_path)
    if gate is not None:
        if gate.get("decision") not in {"strong_go", "conditional_go", "stop"}:
            state.error("phase2_gate_decision", "Phase 2 gate decision is invalid", gate_path)
        if set(gate.get("subject_ids", ())) != set(EXPECTED_SUBJECTS):
            state.error("phase2_gate_subjects", "Phase 2 gate is not based on eight paired subjects", gate_path)
    if all(len(rows[subject]) == len(PHASE2_ARMS) for subject in EXPECTED_SUBJECTS):
        # The cross-arm checks above are the auditable endpoint/label contract.
        pass


def _validate_plan(
    state: AuditState,
    *,
    plan: Mapping[str, Any],
    outer_train: Sequence[str],
    scheme: str,
    expected_inner: int,
    path: Path,
) -> list[dict[str, Any]]:
    if plan.get("scheme") != scheme:
        state.error("crossfit_scheme", f"cross-fit scheme must be {scheme}", path)
    if tuple(plan.get("subjects", ())) != tuple(outer_train):
        state.error("crossfit_subjects", "cross-fit subjects differ from outer train subjects", path)
    folds = plan.get("folds")
    if not isinstance(folds, list) or len(folds) != expected_inner:
        state.error("crossfit_fold_count", f"cross-fit plan requires {expected_inner} inner folds", path)
        return []
    heldout_counts = {subject: 0 for subject in outer_train}
    expected_heldout = 2 if scheme == "3fold" else 1
    for index, fold in enumerate(folds):
        if not isinstance(fold, dict):
            state.error("crossfit_fold", f"inner fold {index} is invalid", path)
            continue
        train = tuple(str(item) for item in fold.get("train_subjects", ()))
        heldout = tuple(str(item) for item in fold.get("heldout_subjects", ()))
        if fold.get("fold_index") != index:
            state.error("crossfit_fold_index", "inner fold indices are not consecutive", path)
        if len(train) != 6 - expected_heldout or len(heldout) != expected_heldout:
            state.error("crossfit_fold_size", f"inner fold {index} has the wrong subject counts", path)
        if set(train) & set(heldout) or set(train) | set(heldout) != set(outer_train):
            state.error("crossfit_leakage", f"inner fold {index} does not partition outer train", path)
        for subject in heldout:
            if subject in heldout_counts:
                heldout_counts[subject] += 1
    if any(count != 1 for count in heldout_counts.values()):
        state.error("crossfit_coverage", "each outer-train subject must be held out exactly once", path)
    return folds


def _audit_crossfit_cell(
    state: AuditState,
    *,
    phase: str,
    subject: str,
    nbm_seed: int,
    fingerprint: str,
    expected_inner: int,
    scheme: str,
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    root = state.root / f"phase{phase}" / f"loso_{subject}" / f"nbm_seed_{nbm_seed}"
    crossfit_root = root / "crossfit"
    provenance_path = crossfit_root / "provenance.json"
    provenance = state.load_json(provenance_path, code="crossfit_provenance_missing")
    state.check_done(
        crossfit_root / "DONE.json",
        protocol_fingerprint=fingerprint,
        stage="h200_phase3_crossfit",
        task_id=f"phase{phase}/loso_{subject}/nbm_seed_{nbm_seed}/crossfit",
        required_artifacts=(
            "train_primitives",
            "validation_primitives",
            "test_primitives",
            "provenance",
            "representation_gate",
        ),
    )
    if provenance is None:
        return None, ()
    if provenance.get("phase") != phase or provenance.get("outer_test_subject") != subject:
        state.error("crossfit_identity", "cross-fit phase/outer-subject identity mismatch", provenance_path)
    if provenance.get("nbm_seed") != nbm_seed:
        state.error("crossfit_seed", "cross-fit NBM seed mismatch", provenance_path)
    if provenance.get("scheme") != scheme:
        state.error("crossfit_scheme", "cross-fit provenance scheme mismatch", provenance_path)
    outer_train = tuple(str(item) for item in provenance.get("outer_train_subjects", ()))
    outer_validation = str(provenance.get("outer_validation_subject", ""))
    if len(outer_train) != 6 or len(set(outer_train)) != 6:
        state.error("outer_train_subjects", "outer train must contain six unique subjects", provenance_path)
    if subject in outer_train or outer_validation in outer_train or subject == outer_validation:
        state.error("outer_split_leakage", "outer 6/1/1 split overlaps", provenance_path)
    plan = provenance.get("crossfit_plan")
    folds = _validate_plan(
        state,
        plan=plan if isinstance(plan, dict) else {},
        outer_train=outer_train,
        scheme=scheme,
        expected_inner=expected_inner,
        path=provenance_path,
    )
    oof = provenance.get("oof_provenance_audit")
    if not isinstance(oof, dict) or oof.get("status") != "pass" or oof.get("failures") not in ([], ()):
        state.error("oof_provenance", "OOF provenance audit did not pass cleanly", provenance_path)
    elif (
        oof.get("expected_windows") is not None
        and oof.get("expected_windows") != oof.get("observed_unique_windows")
    ):
        state.error("oof_window_coverage", "OOF expected and observed window counts differ", provenance_path)
    representation = provenance.get("representation_continuity_audit")
    if not isinstance(representation, dict) or representation.get("status") != "pass":
        state.error("representation_gate", "cross-fit representation hard gate did not pass", provenance_path)
    representation_path = crossfit_root / "representation_gate.json"
    representation_file = state.load_json(representation_path, code="representation_gate_missing")
    if representation_file is not None and not _json_equal(representation, representation_file):
        state.error("representation_gate_mismatch", "cross-fit representation gate artifacts differ", representation_path)
    if provenance.get("forecast_units_before_assembly") != "physical_imu":
        state.error("crossfit_units", "inner forecasts were not assembled in physical IMU units", provenance_path)
    diagnostics = provenance.get("variance_diagnostics")
    if not isinstance(diagnostics, dict):
        state.error("variance_diagnostics", "cross-fit variance diagnostics are missing", provenance_path)
    else:
        train_size = (diagnostics.get("train_oof") or {}).get("ensemble_size_per_row")
        val_size = (diagnostics.get("validation_ensemble") or {}).get("ensemble_size")
        test_size = (diagnostics.get("test_ensemble") or {}).get("ensemble_size")
        if train_size != 1 or val_size != expected_inner or test_size != expected_inner:
            state.error("ensemble_size", "OOF/validation/test ensemble sizes are inconsistent", provenance_path)
    inner_models = provenance.get("inner_models")
    checkpoint_hashes = provenance.get("inner_checkpoint_sha256")
    if not isinstance(inner_models, list) or len(inner_models) != expected_inner:
        state.error("inner_model_count", f"provenance requires {expected_inner} inner models", provenance_path)
        inner_models = []
    if not isinstance(checkpoint_hashes, list) or len(checkpoint_hashes) != expected_inner:
        state.error("inner_checkpoint_count", "inner checkpoint hash list has the wrong length", provenance_path)
        checkpoint_hashes = []
    for index in range(expected_inner):
        inner_root = root / "inner_models" / f"inner_{index:02d}"
        inner_path = inner_root / "inner_provenance.json"
        inner = state.load_json(inner_path, code="inner_provenance_missing")
        if inner is None:
            continue
        inner_fp = inner.get("inner_protocol_fingerprint")
        if not _is_hex_sha256(inner_fp):
            state.error("inner_protocol", "inner protocol fingerprint is invalid", inner_path)
            inner_fp = None
        expected_fold = folds[index] if index < len(folds) else {}
        for field_name, plan_name in (
            ("predictor_train_subjects", "train_subjects"),
            ("heldout_subjects", "heldout_subjects"),
        ):
            if tuple(inner.get(field_name, ())) != tuple(expected_fold.get(plan_name, ())):
                state.error("inner_plan_mismatch", f"{field_name} differs from cross-fit plan", inner_path)
        if tuple(inner.get("scaler_fit_subjects", ())) != tuple(inner.get("predictor_train_subjects", ())):
            state.error("inner_scaler_leakage", "inner scaler subjects differ from predictor train subjects", inner_path)
        forbidden = {subject, outer_validation} | set(inner.get("heldout_subjects", ()))
        if set(inner.get("predictor_train_subjects", ())) & forbidden:
            state.error("inner_subject_leakage", "inner predictor saw a forbidden subject", inner_path)
        if index < len(inner_models) and not _json_equal(inner, inner_models[index]):
            state.error("inner_provenance_mismatch", "inner provenance differs from cross-fit provenance", inner_path)
        checkpoint_sha = inner.get("checkpoint_sha256")
        if not _is_hex_sha256(checkpoint_sha):
            state.error("inner_checkpoint_sha", "inner checkpoint SHA-256 is invalid", inner_path)
        if index < len(checkpoint_hashes) and checkpoint_sha != checkpoint_hashes[index]:
            state.error("inner_checkpoint_mismatch", "inner checkpoint hash list differs", inner_path)
        nbm_done, nbm_artifacts = state.check_done(
            inner_root / "gru" / "nbm" / "DONE.json",
            protocol_fingerprint=str(inner_fp) if inner_fp else None,
        )
        best_path = nbm_artifacts.get("best")
        if best_path is not None and checkpoint_sha != _sha256_file(best_path):
            state.error(
                "inner_checkpoint_file_mismatch",
                "inner provenance checkpoint SHA-256 differs from the sealed best checkpoint",
                inner_path,
            )
        if nbm_done is not None and "best" not in nbm_artifacts:
            state.error(
                "inner_checkpoint_missing",
                "inner NBM DONE does not seal a best checkpoint",
                inner_root / "gru" / "nbm" / "DONE.json",
            )
        predictor_id = inner.get("predictor_id")
        for split in ("heldout", "validation", "test"):
            forecast_root = inner_root / "forecasts" / split
            forecast_provenance = state.load_json(
                forecast_root / "provenance.json", code="forecast_provenance_missing"
            )
            if forecast_provenance is not None:
                if forecast_provenance.get("predictor_id") != predictor_id or forecast_provenance.get("forecast_split") != split:
                    state.error("forecast_identity", "inner forecast provenance identity mismatch", forecast_root / "provenance.json")
                if forecast_provenance.get("forecast_units") != "physical_imu":
                    state.error("forecast_units", "inner forecast is not in physical IMU units", forecast_root / "provenance.json")
            state.check_done(
                forecast_root / "DONE.json",
                protocol_fingerprint=str(inner_fp) if inner_fp else None,
                stage="h200_phase3_physical_forecast",
                task_id=(f"{predictor_id}/forecast/{split}" if isinstance(predictor_id, str) else None),
                upstream_sha256=(str(checkpoint_sha) if _is_hex_sha256(checkpoint_sha) else None),
                required_artifacts=("arrays", "provenance"),
            )
    inner_parent = root / "inner_models"
    actual_inner = {
        path.name for path in inner_parent.glob("inner_*") if path.is_dir()
    }
    expected_inner_names = {f"inner_{index:02d}" for index in range(expected_inner)}
    if actual_inner - expected_inner_names:
        state.error(
            "inner_model_scope",
            f"unexpected inner model directories {sorted(actual_inner - expected_inner_names)}",
            inner_parent,
        )
    if isinstance(plan, dict) and isinstance(checkpoint_hashes, list):
        expected_upstream = _canonical_fingerprint(
            {"checkpoint_sha256": checkpoint_hashes, "plan": plan}
        )
        crossfit_done = state.load_json(crossfit_root / "DONE.json", required=False)
        if (
            crossfit_done is not None
            and crossfit_done.get("upstream_nbm_sha256") != expected_upstream
        ):
            state.error(
                "crossfit_upstream",
                "cross-fit DONE does not bind the checkpoint list and split plan",
                crossfit_root / "DONE.json",
            )
    return provenance, outer_train


def _audit_phase3_classifiers(
    state: AuditState,
    *,
    phase: str,
    subjects: Sequence[str],
    nbm_seeds: Sequence[int],
    classifier_seeds: Sequence[int],
    fingerprint: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matrix: dict[str, Any] = state.matrices.setdefault(f"phase{phase}", {})
    for subject in subjects:
        subject_matrix = matrix.setdefault(subject, {})
        for nbm_seed in nbm_seeds:
            seed_matrix = subject_matrix.setdefault(f"nbm_seed_{nbm_seed}", {})
            for classifier_seed in classifier_seeds:
                classifier_matrix = seed_matrix.setdefault(f"classifier_seed_{classifier_seed}", {})
                group: dict[str, Mapping[str, Any]] = {}
                for arm in PHASE3_ARMS:
                    root = (
                        state.root
                        / f"phase{phase}"
                        / f"loso_{subject}"
                        / f"nbm_seed_{nbm_seed}"
                        / f"classifier_seed_{classifier_seed}"
                        / arm
                    )
                    metrics_path = root / "metrics.json"
                    if not metrics_path.is_file():
                        state.incomplete("phase3_classifier_cell", f"missing Phase {phase} classifier cell", metrics_path)
                        classifier_matrix[arm] = "missing"
                        continue
                    metrics = state.load_json(metrics_path) or {}
                    state.check_done(
                        root / "DONE.json",
                        protocol_fingerprint=fingerprint,
                        stage="h200_phase3_classifier",
                        task_id=(
                            f"phase{phase}/loso_{subject}/nbm_seed_{nbm_seed}/"
                            f"classifier_seed_{classifier_seed}/{arm}"
                        ),
                        upstream_sha256=(
                            _sha256_file(
                                state.root
                                / f"phase{phase}"
                                / f"loso_{subject}"
                                / f"nbm_seed_{nbm_seed}"
                                / "crossfit"
                                / "DONE.json"
                            )
                            if (
                                state.root
                                / f"phase{phase}"
                                / f"loso_{subject}"
                                / f"nbm_seed_{nbm_seed}"
                                / "crossfit"
                                / "DONE.json"
                            ).is_file()
                            else None
                        ),
                        required_artifacts=("metrics", "predictions"),
                    )
                    _check_metrics_identity(state, metrics, path=metrics_path, subject=subject, arm=arm, phase=phase)
                    if metrics.get("nbm_seed") != nbm_seed or metrics.get("classifier_seed") != classifier_seed:
                        state.error("phase3_seed_identity", "Phase 3 classifier seed identity mismatch", metrics_path)
                    group[arm] = metrics
                    rows.append(dict(metrics))
                    classifier_matrix[arm] = "complete"
                if len(group) > 1:
                    if len({row.get("endpoint_sha256") for row in group.values()}) != 1:
                        state.error("phase3_endpoint_mismatch", "Phase 3 classifier endpoints differ by arm", root.parent)
                    if len({row.get("label_sha256") for row in group.values()}) != 1:
                        state.error("phase3_label_mismatch", "Phase 3 classifier labels differ by arm", root.parent)
                zero = group.get("raw4_zero")
                fusion = group.get("raw4_normality")
                if zero is not None and fusion is not None:
                    if zero.get("parameter_count") != fusion.get("parameter_count"):
                        state.error("phase3_capacity_mismatch", "Phase 3 zero/fusion parameter counts differ", root.parent)
                    if zero.get("initial_state_sha256") != fusion.get("initial_state_sha256"):
                        state.error("phase3_initialization_mismatch", "Phase 3 zero/fusion initial hashes differ", root.parent)
        subject_rows = [
            row for row in rows if str(row.get("test_subject")) == subject
        ]
        if len(subject_rows) > 1:
            if len({row.get("endpoint_sha256") for row in subject_rows}) != 1:
                state.error(
                    "phase3_seed_endpoint_mismatch",
                    "Phase 3 endpoints differ across NBM/classifier repetitions",
                    state.root / f"phase{phase}" / f"loso_{subject}",
                )
            if len({row.get("label_sha256") for row in subject_rows}) != 1:
                state.error(
                    "phase3_seed_label_mismatch",
                    "Phase 3 labels differ across NBM/classifier repetitions",
                    state.root / f"phase{phase}" / f"loso_{subject}",
                )
    return rows


def _read_csv(state: AuditState, path: Path, code: str) -> list[dict[str, str]] | None:
    if not path.is_file():
        state.incomplete(code, "missing CSV artifact", path)
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        state.error("csv_invalid", f"cannot parse CSV: {error}", path)
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _audit_subject_seed_averaging(
    state: AuditState,
    *,
    phase: str,
    subjects: Sequence[str],
    nbm_seeds: Sequence[int],
    classifier_seeds: Sequence[int],
    classifier_rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> None:
    path = state.root / f"phase{phase}" / "subject_seed_averaged_metrics.csv"
    averaged = _read_csv(state, path, "subject_average_missing")
    if averaged is None:
        return
    expected_repetitions = len(nbm_seeds) * len(classifier_seeds)
    expected_keys = {(subject, arm) for subject in subjects for arm in PHASE3_ARMS}
    observed_keys = {(row.get("test_subject", ""), row.get("arm", "")) for row in averaged}
    if observed_keys != expected_keys or len(averaged) != len(expected_keys):
        state.error("subject_average_grid", "subject-averaged CSV does not contain one row per subject/arm", path)
    for row in averaged:
        key = (row.get("test_subject", ""), row.get("arm", ""))
        source = [
            item for item in classifier_rows
            if (str(item.get("test_subject")), str(item.get("arm"))) == key
        ]
        try:
            repetitions = int(row.get("repetitions", ""))
        except ValueError:
            repetitions = -1
        if repetitions != expected_repetitions or len(source) != expected_repetitions:
            state.error("seed_repetition_count", f"{key} does not average all seed repetitions", path)
        for metric in METRIC_COLUMNS:
            values = [value for value in (_float_or_none(item.get(metric)) for item in source) if value is not None]
            observed = _float_or_none(row.get(metric))
            if values and (observed is None or not math.isclose(observed, sum(values) / len(values), rel_tol=1e-9, abs_tol=1e-9)):
                state.error("seed_average_value", f"{key}/{metric} is not the within-subject seed mean", path)
    for arm in PHASE3_ARMS:
        block = (aggregate.get("aggregate") or {}).get(arm)
        if not isinstance(block, dict):
            state.error("phase3_macro", f"aggregate lacks arm {arm}", state.root / f"phase{phase}" / "aggregate.json")
            continue
        for metric, summary in block.items():
            if not isinstance(summary, dict):
                continue
            expected_n = sum(
                1
                for row in averaged
                if row.get("arm") == arm and _float_or_none(row.get(metric)) is not None
            )
            if summary.get("n_subjects") != expected_n:
                state.error(
                    "phase3_macro_subject_n",
                    f"{arm}/{metric} count does not match finite subject-averaged values",
                    state.root / f"phase{phase}" / "aggregate.json",
                )
    for comparison in (aggregate.get("paired_bootstrap") or {}).values():
        if not isinstance(comparison, dict):
            continue
        for bootstrap in comparison.values():
            if bootstrap is not None and isinstance(bootstrap, dict):
                if bootstrap.get("n_subjects") != len(subjects) or set(bootstrap.get("subjects", ())) != set(subjects):
                    state.error("phase3_bootstrap_subject_n", "paired bootstrap did not use subject means", state.root / f"phase{phase}" / "aggregate.json")


def _audit_external_status(state: AuditState, aggregate: Mapping[str, Any], path: Path) -> None:
    external = aggregate.get("external_negative_only_evaluation")
    if not isinstance(external, dict):
        state.error("external_status_missing", "S04/S10 external evaluation status is missing", path)
        return
    if set(external.get("subjects", ())) != {"S04", "S10"}:
        state.error("external_subjects", "external negative-only subjects must be S04/S10", path)
    if external.get("status") not in {
        "not_applicable_before_phase3b",
        "not_executed",
        "complete",
        "completed",
        "skipped",
        "unavailable",
    }:
        state.error("external_status", "external S04/S10 status is invalid", path)


def _audit_phase3(
    state: AuditState,
    *,
    phase: str,
    config: Mapping[str, Any],
    fingerprint: str,
) -> None:
    if phase == "3a":
        subjects, expected_inner, scheme = PHASE3A_SUBJECTS, 3, "3fold"
        default_classifier = (42,)
    else:
        subjects, expected_inner, scheme = EXPECTED_SUBJECTS, 6, "loto"
        default_classifier = (42, 43, 44)
    nbm_seeds = _parse_seed_values(config.get("phase3_nbm_seeds"), (42,))
    policies = config.get("phase3_classifier_seed_policy")
    policy_value = policies.get(phase) if isinstance(policies, dict) else None
    classifier_seeds = _parse_seed_values(policy_value, default_classifier)
    if not nbm_seeds:
        state.error("phase3_nbm_seeds", "Phase 3 NBM seed policy is invalid", state.root / "config.json")
        nbm_seeds = (42,)
    if not classifier_seeds:
        state.error("phase3_classifier_seeds", "Phase 3 classifier seed policy is invalid", state.root / "config.json")
        classifier_seeds = default_classifier
    phase_root = state.root / f"phase{phase}"
    actual_subjects = {
        path.name.removeprefix("loso_")
        for path in phase_root.glob("loso_*")
        if path.is_dir()
    }
    if actual_subjects - set(subjects):
        state.error(
            "phase3_subject_scope",
            f"Phase {phase} contains unexpected outer subjects {sorted(actual_subjects - set(subjects))}",
            phase_root,
        )
    for subject in subjects:
        subject_root = phase_root / f"loso_{subject}"
        actual_nbm = {
            path.name for path in subject_root.glob("nbm_seed_*") if path.is_dir()
        }
        expected_nbm = {f"nbm_seed_{seed}" for seed in nbm_seeds}
        if actual_nbm - expected_nbm:
            state.error(
                "phase3_nbm_scope",
                f"Phase {phase}/{subject} contains unexpected NBM seeds {sorted(actual_nbm - expected_nbm)}",
                subject_root,
            )
        for nbm_seed in nbm_seeds:
            nbm_root = subject_root / f"nbm_seed_{nbm_seed}"
            actual_classifier = {
                path.name
                for path in nbm_root.glob("classifier_seed_*")
                if path.is_dir()
            }
            expected_classifier = {
                f"classifier_seed_{seed}" for seed in classifier_seeds
            }
            if actual_classifier - expected_classifier:
                state.error(
                    "phase3_classifier_seed_scope",
                    f"unexpected classifier seeds {sorted(actual_classifier - expected_classifier)}",
                    nbm_root,
                )
            for classifier_seed in classifier_seeds:
                classifier_root = nbm_root / f"classifier_seed_{classifier_seed}"
                actual_arms = {
                    path.name for path in classifier_root.iterdir() if path.is_dir()
                } if classifier_root.is_dir() else set()
                if actual_arms - set(PHASE3_ARMS):
                    state.error(
                        "phase3_arm_scope",
                        f"unexpected Phase 3 arms {sorted(actual_arms - set(PHASE3_ARMS))}",
                        classifier_root,
                    )
    matrix: dict[str, Any] = state.matrices.setdefault(f"phase{phase}", {})
    for subject in subjects:
        matrix.setdefault(subject, {})
        for nbm_seed in nbm_seeds:
            provenance, _ = _audit_crossfit_cell(
                state,
                phase=phase,
                subject=subject,
                nbm_seed=nbm_seed,
                fingerprint=fingerprint,
                expected_inner=expected_inner,
                scheme=scheme,
            )
            matrix[subject][f"nbm_seed_{nbm_seed}"] = {
                "crossfit": "complete" if provenance is not None else "missing"
            }
    classifier_rows = _audit_phase3_classifiers(
        state,
        phase=phase,
        subjects=subjects,
        nbm_seeds=nbm_seeds,
        classifier_seeds=classifier_seeds,
        fingerprint=fingerprint,
    )
    representation_path = phase_root / "representation_gate.json"
    representation = state.load_json(representation_path, code="phase3_representation_missing")
    if representation is not None:
        cells = representation.get("cells")
        expected_cells = len(subjects) * len(nbm_seeds)
        if representation.get("status") != "pass" or representation.get("hard_gate") is not True:
            state.error("phase3_representation_gate", f"Phase {phase} representation hard gate did not pass", representation_path)
        if not isinstance(cells, list) or len(cells) != expected_cells:
            state.error("phase3_representation_cells", f"Phase {phase} representation gate has the wrong cell count", representation_path)
    aggregate_path = phase_root / "aggregate.json"
    aggregate = state.load_json(aggregate_path, code="phase3_aggregate_missing")
    if aggregate is not None:
        state.check_done(
            phase_root / "DONE.json",
            protocol_fingerprint=fingerprint,
            stage="h200_phase3_aggregate",
            task_id=f"phase{phase}/aggregate",
            required_artifacts=(
                "aggregate", "classifier_cells", "subject_seed_averaged_metrics", "representation_gate"
            ),
        )
        if aggregate.get("phase") != phase or aggregate.get("protocol_fingerprint") != fingerprint:
            state.error("phase3_aggregate_identity", "Phase 3 aggregate identity mismatch", aggregate_path)
        if representation is not None and not _json_equal(aggregate.get("representation_gate"), representation):
            state.error("phase3_representation_mismatch", "aggregate representation gate differs from file", aggregate_path)
        decision = aggregate.get("decision")
        if not isinstance(decision, dict) or decision.get("status") not in {"pass", "fail"}:
            state.error("phase3_decision", "Phase 3 decision is invalid", aggregate_path)
        if phase == "3a":
            science = aggregate.get("science_gate")
            if not isinstance(science, dict) or science.get("status") not in {"pass", "fail"}:
                state.error("phase3a_science_gate", "Phase 3A science gate is missing or invalid", aggregate_path)
            elif isinstance(decision, dict):
                expected_decision = "pass" if science.get("status") == "pass" and (representation or {}).get("status") == "pass" else "fail"
                if decision.get("status") != expected_decision:
                    state.error("phase3a_gate_consistency", "Phase 3A combined decision is inconsistent", aggregate_path)
        else:
            science = aggregate.get("science_gate")
            if not isinstance(science, dict) or science.get("status") != "not_applicable":
                state.error("phase3b_science_gate", "Phase 3B science gate must be not_applicable", aggregate_path)
        _audit_external_status(state, aggregate, aggregate_path)
        external = aggregate.get("external_negative_only_evaluation") or {}
        if phase == "3a" and external.get("status") != "not_applicable_before_phase3b":
            state.error(
                "phase3a_external_status",
                "S04/S10 evaluation must be deferred until Phase 3B",
                aggregate_path,
            )
        if (
            phase == "3b"
            and bool(config.get("phase3_external_negative_only", True))
            and external.get("status") not in {"complete", "completed"}
        ):
            state.error(
                "phase3b_external_incomplete",
                "Phase 3B requires completed S04/S10 negative-only evaluation",
                aggregate_path,
            )
        _audit_subject_seed_averaging(
            state,
            phase=phase,
            subjects=subjects,
            nbm_seeds=nbm_seeds,
            classifier_seeds=classifier_seeds,
            classifier_rows=classifier_rows,
            aggregate=aggregate,
        )


def _audit_unvisited_done(state: AuditState, fingerprint: str) -> None:
    for path in sorted(state.root.rglob("DONE.json")):
        if path.resolve() in state.checked_done:
            continue
        expected = fingerprint
        if "inner_models" in path.parts:
            current = path.parent
            inner_root: Path | None = None
            while current != state.root and current.parent != current:
                if current.name.startswith("inner_") and current.parent.name == "inner_models":
                    inner_root = current
                    break
                current = current.parent
            if inner_root is not None:
                provenance = state.load_json(inner_root / "inner_provenance.json", required=False)
                if provenance is not None and _is_hex_sha256(provenance.get("inner_protocol_fingerprint")):
                    expected = str(provenance["inner_protocol_fingerprint"])
        state.check_done(path, protocol_fingerprint=expected)


def audit_feasibility_results(
    result_dir: str | Path,
    *,
    allow_incomplete: bool = False,
    report_path: str | Path | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    """Audit a feasibility result directory and optionally write its report."""

    root = Path(result_dir).resolve()
    state = AuditState(root=root, allow_incomplete=bool(allow_incomplete))
    if not root.is_dir():
        state.error("result_dir_missing", "result directory does not exist", root)
        config, fingerprint = {}, ""
    else:
        config, fingerprint = _audit_config(state)
        _audit_phase0(state, config, fingerprint)
        _audit_phase1(state, fingerprint)
        _audit_phase2(state, fingerprint)
        _audit_phase3(state, phase="3a", config=config, fingerprint=fingerprint)
        _audit_phase3(state, phase="3b", config=config, fingerprint=fingerprint)
        _audit_unvisited_done(state, fingerprint)
    errors = [item for item in state.findings if item["kind"] == "error"]
    incomplete = [item for item in state.findings if item["kind"] == "incomplete"]
    warnings = [item for item in state.findings if item["kind"] == "warning"]
    ok = not errors and (bool(allow_incomplete) or not incomplete)
    report = {
        "audit_version": "daphnet_gru_h200_residual_feasibility_audit.v1",
        "result_dir": str(root),
        "protocol_fingerprint": fingerprint,
        "allow_incomplete": bool(allow_incomplete),
        "ok": bool(ok),
        "complete": not incomplete,
        "summary": {
            "errors": len(errors),
            "incomplete": len(incomplete),
            "warnings": len(warnings),
            "done_manifests_checked": len(state.checked_done),
        },
        "matrices": state.matrices,
        "findings": state.findings,
    }
    if write_report:
        destination = Path(report_path).resolve() if report_path is not None else root / "audit_report.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        temporary.replace(destination)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently audit all four Daphnet GRU-H200 feasibility phases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Report an in-progress phase matrix without failing for missing cells",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Defaults to RESULT_DIR/audit_report.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_feasibility_results(
        args.result_dir,
        allow_incomplete=args.allow_incomplete,
        report_path=args.report_path,
        write_report=True,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    print(f"audit_report={args.report_path or (args.result_dir / 'audit_report.json')}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
