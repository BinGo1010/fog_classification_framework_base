#!/usr/bin/env python
"""Independent audit for the RF125 TCN-M versus CNN replacement suite."""

from __future__ import annotations

import argparse
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import audit_daphnet_residual_classifier_suite as audit_base

from cnbr_fog.resume import canonical_fingerprint
from cnbr_fog.rf125_classifiers import (
    CANONICAL_RF125_CLASSIFIER_NAMES,
    DEFAULT_DILATIONS,
    RF125_CLASSIFIER_DISPLAY_NAMES,
    build_rf125_classifier,
    parameter_count,
)
from run_daphnet_rf125_cnn_replacement import (
    CLASSIFIER_STAGE,
    LOWER_IS_BETTER_METRICS,
    PAIR_AGGREGATE_KEY,
    PAIR_COMPARISON,
    PAIR_DELTA_LABEL,
    PAIR_REFERENCE,
    PAIR_TOLERANCE,
    SUITE_VERSION,
)


AUDIT_VERSION = "daphnet_rf125_cnn_replacement_audit.v1"
EXPECTED_FAMILIES = {
    name: "rf125_dilated_1d_cnn"
    for name in CANONICAL_RF125_CLASSIFIER_NAMES
}
ARCHITECTURE_FIELDS = {
    name: ("hidden_channels", "dilations", "kernel_size")
    for name in CANONICAL_RF125_CLASSIFIER_NAMES
}
EXPECTED_PARAMETER_COUNT = 89_329
EXPECTED_MACS = 21_348_912
EXPECTED_RESIDUAL_ADDITIONS = 73_728
EXPECTED_LOCAL_RF_SAMPLES = 125
EXPECTED_LOCAL_RF_SECONDS = 125 / 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the Daphnet Persistence residual_h4s RF125 "
            "TCN-M versus 1D-CNN replacement suite"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--source-suite-dir",
        type=Path,
        help="Fallback completed NBM-suite path.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Fallback processed Daphnet data path.",
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--tolerance", type=float, default=2e-6)
    return parser.parse_args()


def _without_axis(architecture: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "canonical_name",
        "display_name",
        "residual_skip",
        "block_equation",
        "residual_elementwise_additions_per_window",
    }
    return {
        key: value
        for key, value in architecture.items()
        if key not in allowed
    }


def validate_replacement_protocol(
    original_validate,
    config: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    folds, definitions = original_validate(config)
    audit_base.require(int(config.get("seed", -1)) == 42, "config: seed must be 42")
    audit_base.require(
        config.get("deterministic") is True,
        "config: deterministic training must be enabled",
    )
    audit_base.require(
        config.get("amp") is True,
        "config: AMP must match the reportable TCN-M protocol",
    )
    audit_base.require(
        int(config.get("max_classifier_windows", -1)) == 0,
        "config: reportable training must use every training anchor",
    )
    audit_base.require(
        config.get("debug_small_models") is False,
        "config: debug-small models are not reportable",
    )
    for key, expected in (
        ("classifier_epochs", 12),
        ("classifier_patience", 4),
        ("batch_size", 256),
    ):
        audit_base.require(
            int(config.get(key, -1)) == expected,
            f"config: {key} must be {expected}",
        )
    for key, expected in (
        ("classifier_dropout", 0.15),
        ("classifier_lr", 1e-3),
        ("weight_decay", 1e-4),
    ):
        audit_base.assert_close(
            config.get(key),
            expected,
            f"config/{key}",
            1e-12,
        )

    by_name = {item["classifier"]: item for item in definitions}
    architectures = {
        name: by_name[name]["architecture"]
        for name in CANONICAL_RF125_CLASSIFIER_NAMES
    }
    for name, architecture in architectures.items():
        audit_base.require(
            int(architecture.get("hidden_channels", -1)) == 48,
            f"config/{name}: hidden channels must be 48",
        )
        audit_base.require(
            tuple(architecture.get("dilations", ())) == DEFAULT_DILATIONS,
            f"config/{name}: dilation schedule mismatch",
        )
        audit_base.require(
            int(architecture.get("n_blocks", -1)) == 6
            and int(architecture.get("kernel_size", -1)) == 3
            and int(architecture.get("convolutions_per_block", -1)) == 2,
            f"config/{name}: temporal block structure mismatch",
        )
        audit_base.require(
            int(architecture.get("local_receptive_field_samples", -1))
            == EXPECTED_LOCAL_RF_SAMPLES,
            f"config/{name}: local receptive field is not 125",
        )
        audit_base.assert_close(
            architecture.get("local_receptive_field_seconds"),
            EXPECTED_LOCAL_RF_SECONDS,
            f"config/{name}/local_receptive_field_seconds",
            1e-12,
        )
        audit_base.require(
            architecture.get("padding") == "symmetric_same_zero"
            and architecture.get("causal") is False
            and architecture.get("global_pooling")
            == "mean_and_max_over_full_input",
            f"config/{name}: padding/pooling contract mismatch",
        )
        audit_base.require(
            int(architecture.get("parameter_count", -1))
            == EXPECTED_PARAMETER_COUNT
            and int(architecture.get("trainable_parameter_count", -1))
            == EXPECTED_PARAMETER_COUNT,
            f"config/{name}: parameter count mismatch",
        )
        audit_base.require(
            int(architecture.get("conv_linear_macs_per_window", -1))
            == EXPECTED_MACS,
            f"config/{name}: Conv/Linear MAC count mismatch",
        )

    tcn = architectures["tcn_m"]
    cnn = architectures["cnn_rf125"]
    audit_base.require(
        tcn.get("residual_skip") is True
        and tcn.get("block_equation") == "x_plus_Fx"
        and int(tcn.get("residual_elementwise_additions_per_window", -1))
        == EXPECTED_RESIDUAL_ADDITIONS,
        "config/tcn_m: residual path contract mismatch",
    )
    audit_base.require(
        cnn.get("residual_skip") is False
        and cnn.get("block_equation") == "Fx"
        and int(cnn.get("residual_elementwise_additions_per_window", -1)) == 0,
        "config/cnn_rf125: plain CNN contract mismatch",
    )
    audit_base.require(
        canonical_fingerprint(_without_axis(tcn))
        == canonical_fingerprint(_without_axis(cnn)),
        "config: architectures differ outside the residual-skip axis",
    )
    audit_base.require(
        tcn.get("parameter_schema_sha256")
        == cnn.get("parameter_schema_sha256")
        == config.get("shared_parameter_schema_sha256"),
        "config: parameter schemas are not shared",
    )
    audit_base.require(
        by_name["tcn_m"]["protocol_initial_state_sha256"]
        == by_name["cnn_rf125"]["protocol_initial_state_sha256"]
        == config.get("shared_protocol_initial_state_sha256"),
        "config: protocol initial states are not shared",
    )
    audit_base.require(
        int(config.get("shared_parameter_count", -1))
        == EXPECTED_PARAMETER_COUNT,
        "config: shared parameter count mismatch",
    )
    audit_base.require(
        int(config.get("shared_conv_linear_macs_per_window", -1))
        == EXPECTED_MACS,
        "config: shared Conv/Linear MAC count mismatch",
    )
    audit_base.require(
        int(config.get("tcn_extra_residual_additions_per_window", -1))
        == EXPECTED_RESIDUAL_ADDITIONS,
        "config: residual-add count mismatch",
    )
    compute_delta = float(config.get("estimated_compute_delta_fraction", math.inf))
    audit_base.require(
        0.0 < compute_delta < 0.005,
        "config: matched-compute delta must remain below 0.5%",
    )
    fairness = config["fairness_contract"]
    for key in (
        "same_parameter_count",
        "same_parameter_schema",
        "same_initial_state_sha256_within_fold",
        "same_conv_linear_macs",
    ):
        audit_base.require(
            fairness.get(key) is True,
            f"config/fairness: {key} is disabled",
        )
    return folds, definitions


def validate_replacement_cross_fairness(
    original_validate,
    subject: str,
    evidence: Mapping[str, dict[str, Any]],
) -> None:
    original_validate(subject, evidence)
    if set(evidence) != set(CANONICAL_RF125_CLASSIFIER_NAMES):
        return
    tcn = evidence["tcn_m"]["metrics"]["architecture"]
    cnn = evidence["cnn_rf125"]["metrics"]["architecture"]
    audit_base.require(
        canonical_fingerprint(_without_axis(tcn))
        == canonical_fingerprint(_without_axis(cnn)),
        f"{subject}: architectures differ outside residual-skip",
    )
    audit_base.require(
        tcn.get("residual_skip") is True
        and cnn.get("residual_skip") is False,
        f"{subject}: residual-skip labels are invalid",
    )


def validate_paired_fold_csv(
    root: Path,
    cells: Mapping[tuple[str, str], dict[str, Any]],
    tolerance: float,
) -> None:
    columns, rows = audit_base.read_csv(root / "paired_fold_deltas.csv")
    expected_columns = [
        "test_subject",
        *[
            f"{metric}_delta_{PAIR_DELTA_LABEL}"
            for metric in audit_base.CLASSIFICATION_METRICS
        ],
    ]
    audit_base.require(
        columns == expected_columns,
        "paired_fold_deltas.csv: column mismatch",
    )
    expected_subjects = [
        subject
        for subject in audit_base.EXPECTED_SUBJECTS
        if (subject, PAIR_REFERENCE) in cells
        and (subject, PAIR_COMPARISON) in cells
    ]
    audit_base.require(
        [row["test_subject"] for row in rows] == expected_subjects,
        "paired_fold_deltas.csv: subject rows mismatch",
    )
    for row in rows:
        subject = row["test_subject"]
        reference = cells[(subject, PAIR_REFERENCE)]["metrics"]
        comparison = cells[(subject, PAIR_COMPARISON)]["metrics"]
        for metric in audit_base.CLASSIFICATION_METRICS:
            before = reference.get(metric)
            after = comparison.get(metric)
            expected = (
                None
                if before is None or after is None
                else float(after) - float(before)
            )
            actual = audit_base.csv_number(
                row[f"{metric}_delta_{PAIR_DELTA_LABEL}"]
            )
            audit_base.assert_close(
                actual,
                expected,
                f"paired_fold_deltas/{subject}/{metric}",
                tolerance,
            )


def validate_replacement_summaries(
    original_validate,
    root: Path,
    config: dict[str, Any],
    definitions: list[dict[str, Any]],
    cells: Mapping[tuple[str, str], dict[str, Any]],
    tolerance: float,
) -> None:
    original_validate(root, config, definitions, cells, tolerance)
    validate_paired_fold_csv(root, cells, tolerance)


@contextmanager
def configured_auditor() -> Iterator[Any]:
    original_validate_protocol = audit_base.validate_protocol
    original_cross = audit_base.validate_cross_classifier_fairness
    original_summaries = audit_base.validate_summaries
    replacements = {
        "AUDIT_VERSION": AUDIT_VERSION,
        "SUITE_VERSION": SUITE_VERSION,
        "CLASSIFIER_STAGE": CLASSIFIER_STAGE,
        "CANONICAL_CLASSIFIER_NAMES": CANONICAL_RF125_CLASSIFIER_NAMES,
        "CLASSIFIER_DISPLAY_NAMES": RF125_CLASSIFIER_DISPLAY_NAMES,
        "EXPECTED_FAMILIES": EXPECTED_FAMILIES,
        "ARCHITECTURE_FIELDS": ARCHITECTURE_FIELDS,
        "build_residual_classifier": build_rf125_classifier,
        "parameter_count": parameter_count,
        "EXPECTED_ABLATION_AXIS": "residual_skip_connection",
        "EXPECT_DIFFERENT_PARAMETER_SHAPES": False,
        "EXPECT_IDENTICAL_INITIAL_STATES": True,
        "PAIRED_REFERENCE": PAIR_REFERENCE,
        "PAIRED_AGGREGATE_KEY": PAIR_AGGREGATE_KEY,
        "PAIRED_DELTA_LABEL": PAIR_DELTA_LABEL,
        "PAIRED_INCLUDE_WIN_STATS": True,
        "PAIRED_TOLERANCE": PAIR_TOLERANCE,
        "LOWER_IS_BETTER_METRICS": LOWER_IS_BETTER_METRICS,
        "parse_args": parse_args,
    }
    saved = {
        name: getattr(audit_base, name)
        for name in replacements
    }
    try:
        for name, value in replacements.items():
            setattr(audit_base, name, value)
        audit_base.validate_protocol = lambda config: validate_replacement_protocol(
            original_validate_protocol,
            config,
        )
        audit_base.validate_cross_classifier_fairness = (
            lambda subject, evidence: validate_replacement_cross_fairness(
                original_cross,
                subject,
                evidence,
            )
        )
        audit_base.validate_summaries = (
            lambda root, config, definitions, cells, tolerance: (
                validate_replacement_summaries(
                    original_summaries,
                    root,
                    config,
                    definitions,
                    cells,
                    tolerance,
                )
            )
        )
        yield audit_base
    finally:
        audit_base.validate_protocol = original_validate_protocol
        audit_base.validate_cross_classifier_fairness = original_cross
        audit_base.validate_summaries = original_summaries
        for name, value in saved.items():
            setattr(audit_base, name, value)


def main() -> None:
    with configured_auditor() as configured:
        configured.main()


if __name__ == "__main__":
    main()
