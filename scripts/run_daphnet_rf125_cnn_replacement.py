#!/usr/bin/env python
"""Strict RF125 TCN-M versus plain dilated 1D-CNN replacement experiment.

This two-arm suite reuses the completed Persistence-NBM ``residual_h4s``
representation and the server-tested resumable classifier training pipeline.
For every canonical LOSO fold it trains:

* ``tcn_m``: six RF125 blocks using ``x + F(x)``;
* ``cnn_rf125``: the same six blocks using ``F(x)``.

The arms have identical trainable parameters and Conv/Linear MACs.  Residual
addition is the only intended architecture axis.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_daphnet_residual_classifier_suite as suite_base

from cnbr_fog.resume import atomic_json_dump, canonical_fingerprint
from cnbr_fog.rf125_classifiers import (
    CANONICAL_RF125_CLASSIFIER_NAMES,
    DEFAULT_DILATIONS,
    RF125_CLASSIFIER_DISPLAY_NAMES,
    build_rf125_classifier,
    parameter_count,
    rf125_classifier_config,
)


SUITE_VERSION = "daphnet_persistence_h4_rf125_cnn_replacement.v1"
CLASSIFIER_STAGE = "rf125_replacement_classifier"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "daphnet_persistence_h4_rf125_cnn_replacement_seed42"
)
IMPLEMENTATION_FILES = (
    "scripts/run_daphnet_rf125_cnn_replacement.py",
    "scripts/run_daphnet_residual_classifier_suite.py",
    "scripts/run_daphnet_tcn_rf_ablation.py",
    "scripts/run_cnbr_fog_loso.py",
    "cnbr_fog/rf125_classifiers.py",
    "cnbr_fog/__init__.py",
    "cnbr_fog/data.py",
    "cnbr_fog/evaluation.py",
    "cnbr_fog/histories.py",
    "cnbr_fog/models.py",
    "cnbr_fog/resume.py",
)
DEBUG_SMALL_ARCHITECTURES: dict[str, dict[str, Any]] = {
    name: {"hidden_channels": 8}
    for name in CANONICAL_RF125_CLASSIFIER_NAMES
}
PAIR_REFERENCE = "tcn_m"
PAIR_COMPARISON = "cnn_rf125"
PAIR_AGGREGATE_KEY = "paired_deltas_cnn_rf125_minus_tcn_m"
PAIR_DELTA_LABEL = "cnn_rf125_minus_tcn_m"
PAIR_TOLERANCE = 1e-12
LOWER_IS_BETTER_METRICS = {
    "false_alarm_events_per_hour",
    "median_detection_delay_sec",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Daphnet Persistence residual_h4s RF125 TCN-M versus "
            "dilated 1D-CNN LOSO replacement suite"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=suite_base.DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--source-suite-dir",
        type=Path,
        default=suite_base.DEFAULT_SOURCE_SUITE_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", default="all")
    parser.add_argument(
        "--worker-fold",
        default="",
        help="Run exactly one fold; used by the multi-GPU scheduler.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Initialize/validate protocol and rebuild root summaries only.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier-dropout", type=float, default=0.15)
    parser.add_argument("--classifier-epochs", type=int, default=12)
    parser.add_argument("--classifier-patience", type=int, default=4)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-classifier-windows",
        type=int,
        default=0,
        help="Training-only deterministic cap; zero uses every common anchor.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--debug-interrupt-classifier-after-epoch",
        type=int,
        default=0,
        help="Testing hook for exact epoch-boundary recovery.",
    )
    parser.add_argument(
        "--debug-small-models",
        action="store_true",
        help="Use tiny widths for interface smoke tests only.",
    )
    return parser.parse_args()


def _runtime_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    names = {
        "data_dir",
        "source_suite_dir",
        "output_dir",
        "device",
        "num_workers",
        "resume",
    }
    return {name: config[name] for name in names if name in config}


def _architecture_without_axis(architecture: Mapping[str, Any]) -> dict[str, Any]:
    allowed_axis_fields = {
        "canonical_name",
        "display_name",
        "residual_skip",
        "block_equation",
        "residual_elementwise_additions_per_window",
    }
    return {
        key: value
        for key, value in architecture.items()
        if key not in allowed_axis_fields
    }


def build_replacement_protocol(
    original_build_protocol,
    args,
    source_manifest,
    source_config,
    dataset,
    windows,
    data_sha256,
    device,
) -> dict[str, Any]:
    """Build the generic protocol, then strengthen it for the matched pair."""

    config = original_build_protocol(
        args,
        source_manifest,
        source_config,
        dataset,
        windows,
        data_sha256,
        device,
    )
    definitions = list(config["classifiers"])
    if [item["classifier"] for item in definitions] != list(
        CANONICAL_RF125_CLASSIFIER_NAMES
    ):
        raise AssertionError("RF125 classifier order changed")

    parameter_counts = {
        int(item["parameter_count"]) for item in definitions
    }
    initial_hashes = {
        str(item["protocol_initial_state_sha256"]) for item in definitions
    }
    schema_hashes = {
        str(item["architecture"]["parameter_schema_sha256"])
        for item in definitions
    }
    mac_counts = {
        int(item["architecture"]["conv_linear_macs_per_window"])
        for item in definitions
    }
    non_axis_architectures = {
        canonical_fingerprint(
            _architecture_without_axis(item["architecture"])
        )
        for item in definitions
    }
    if len(parameter_counts) != 1:
        raise AssertionError("RF125 arms must have identical parameter counts")
    if len(initial_hashes) != 1:
        raise AssertionError("RF125 arms must have identical initial states")
    if len(schema_hashes) != 1:
        raise AssertionError("RF125 arms must have identical state schemas")
    if len(mac_counts) != 1:
        raise AssertionError("RF125 arms must have identical Conv/Linear MACs")
    if len(non_axis_architectures) != 1:
        raise AssertionError(
            "RF125 architectures differ outside the residual-skip axis"
        )

    by_name = {item["classifier"]: item for item in definitions}
    tcn_architecture = by_name["tcn_m"]["architecture"]
    cnn_architecture = by_name["cnn_rf125"]["architecture"]
    residual_additions = int(
        tcn_architecture["residual_elementwise_additions_per_window"]
    )
    if residual_additions <= 0:
        raise AssertionError("TCN-M must execute residual additions")
    if int(cnn_architecture["residual_elementwise_additions_per_window"]) != 0:
        raise AssertionError("CNN-RF125 must not execute residual additions")
    compute_delta_fraction = residual_additions / (
        2.0 * int(next(iter(mac_counts))) + residual_additions
    )

    runtime = _runtime_fields(config)
    protocol = {
        key: value
        for key, value in config.items()
        if key not in runtime and key != "protocol_fingerprint"
    }
    protocol.update(
        {
            "suite_version": SUITE_VERSION,
            "classifier_stage": CLASSIFIER_STAGE,
            "comparison_axis": "residual_skip_connection",
            "shared_parameter_count": next(iter(parameter_counts)),
            "shared_parameter_schema_sha256": next(iter(schema_hashes)),
            "shared_protocol_initial_state_sha256": next(iter(initial_hashes)),
            "shared_conv_linear_macs_per_window": next(iter(mac_counts)),
            "tcn_extra_residual_additions_per_window": residual_additions,
            "estimated_compute_delta_fraction": compute_delta_fraction,
            "fairness_contract": {
                "ablation_axis": "residual_skip_connection",
                "architecture_family": "rf125_dilated_1d_cnn",
                "allowed_architecture_difference_fields": [
                    "canonical_name",
                    "display_name",
                    "residual_skip",
                    "block_equation",
                    "residual_elementwise_additions_per_window",
                ],
                "shared_fields": [
                    "source_persistence_residual_cache",
                    "residual_h4s_window_ids_and_labels",
                    "training_validation_test_support",
                    "training_subsample",
                    "input_shape_9_by_256",
                    "six_blocks",
                    "two_convolutions_per_block",
                    "kernel_size_3",
                    "dilations_1_2_4_8_8_8",
                    "local_receptive_field_125",
                    "symmetric_same_padding",
                    "hidden_channels",
                    "normalization",
                    "activation",
                    "dropout",
                    "global_mean_max_pooling",
                    "classification_head",
                    "parameter_count",
                    "parameter_schema",
                    "initial_parameter_values",
                    "classifier_seed",
                    "epoch_shuffle_order",
                    "optimizer",
                    "learning_rate",
                    "weight_decay",
                    "batch_size",
                    "class_weight",
                    "maximum_epochs",
                    "early_stopping",
                    "validation_threshold_rule",
                ],
                "same_classifier_seed_within_fold": True,
                "same_epoch_shuffle_within_fold": True,
                "same_parameter_count": True,
                "same_parameter_schema": True,
                "same_initial_state_sha256_within_fold": True,
                "same_conv_linear_macs": True,
                "different_parameter_shapes_expected": False,
                "epoch_shuffle_seed_rule": "classifier_seed + epoch",
                "threshold_source": "validation_only_balanced_accuracy",
            },
            "interpretation": {
                "tcn_m": (
                    "RF125 dilated temporal classifier with identity residual "
                    "connections x + F(x)."
                ),
                "cnn_rf125": (
                    "RF125 dilated 1D-CNN with the identical parameterized "
                    "transform F(x), but without identity residual additions."
                ),
                "receptive_field_scope": (
                    "125 samples is the local convolutional-feature receptive "
                    "field; final mean/max pooling spans all 256 input samples."
                ),
            },
        }
    )
    fingerprint = canonical_fingerprint(protocol)
    return {
        **protocol,
        "protocol_fingerprint": fingerprint,
        **runtime,
    }


def paired_delta_summary(
    rows_by_classifier: dict[str, dict[str, dict]],
) -> dict[str, Any]:
    """Return paired per-subject deltas for CNN-RF125 minus TCN-M."""

    reference = rows_by_classifier.get(PAIR_REFERENCE, {})
    comparison = rows_by_classifier.get(PAIR_COMPARISON, {})
    common_subjects = [
        subject
        for subject in suite_base.EXPECTED_LOSO_SUBJECTS
        if subject in reference and subject in comparison
    ]
    metrics: dict[str, dict[str, Any]] = {}
    for metric in suite_base.CLASSIFICATION_METRICS:
        deltas = [
            float(comparison[subject][metric])
            - float(reference[subject][metric])
            for subject in common_subjects
            if comparison[subject].get(metric) is not None
            and reference[subject].get(metric) is not None
        ]
        values = np.asarray(deltas, dtype=np.float64)
        lower_is_better = metric in LOWER_IS_BETTER_METRICS
        oriented = -values if lower_is_better else values
        ties = int(np.count_nonzero(np.abs(oriented) <= PAIR_TOLERANCE))
        metrics[metric] = {
            f"mean_delta_{PAIR_DELTA_LABEL}": (
                float(values.mean()) if len(values) else None
            ),
            f"std_delta_{PAIR_DELTA_LABEL}": (
                float(values.std(ddof=0)) if len(values) else None
            ),
            f"median_delta_{PAIR_DELTA_LABEL}": (
                float(np.median(values)) if len(values) else None
            ),
            "optimization_direction": (
                "lower_is_better" if lower_is_better else "higher_is_better"
            ),
            "wins": int(np.count_nonzero(oriented > PAIR_TOLERANCE)),
            "ties": ties,
            "losses": int(np.count_nonzero(oriented < -PAIR_TOLERANCE)),
            "n_paired_folds": int(len(values)),
        }
    return {
        PAIR_COMPARISON: {
            "reference": PAIR_REFERENCE,
            "delta": PAIR_DELTA_LABEL,
            "common_subjects": common_subjects,
            "metrics": metrics,
        }
    }


def _write_paired_fold_deltas(
    output_dir: Path,
    config: Mapping[str, Any],
) -> None:
    fieldnames = [
        "test_subject",
        *[
            f"{metric}_delta_{PAIR_DELTA_LABEL}"
            for metric in suite_base.CLASSIFICATION_METRICS
        ],
    ]
    rows: list[dict[str, Any]] = []
    for subject in config["folds_resolved"]:
        reference_path = (
            output_dir
            / f"loso_{subject}"
            / PAIR_REFERENCE
            / "metrics.json"
        )
        comparison_path = (
            output_dir
            / f"loso_{subject}"
            / PAIR_COMPARISON
            / "metrics.json"
        )
        if not reference_path.exists() or not comparison_path.exists():
            continue
        reference = suite_base._load_json(reference_path)
        comparison = suite_base._load_json(comparison_path)
        row: dict[str, Any] = {"test_subject": subject}
        for metric in suite_base.CLASSIFICATION_METRICS:
            before = reference.get(metric)
            after = comparison.get(metric)
            row[f"{metric}_delta_{PAIR_DELTA_LABEL}"] = (
                None
                if before is None or after is None
                else float(after) - float(before)
            )
        rows.append(row)

    path = output_dir / "paired_fold_deltas.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temporary.replace(path)


def refresh_replacement_summaries(
    original_refresh,
    output_dir: Path,
    config: dict,
) -> None:
    """Reuse root summaries, then replace the generic MLP-pair key."""

    original_refresh(output_dir, config)
    aggregate_path = output_dir / "aggregate_metrics.json"
    aggregate = suite_base._load_json(aggregate_path)
    generic_key = "paired_deltas_vs_mlp"
    if generic_key not in aggregate:
        raise AssertionError("generic summary did not produce paired deltas")
    aggregate[PAIR_AGGREGATE_KEY] = aggregate.pop(generic_key)
    atomic_json_dump(aggregate, aggregate_path)
    _write_paired_fold_deltas(output_dir, config)


@contextmanager
def configured_base_suite() -> Iterator[Any]:
    """Temporarily configure the generic runner without mutating old suites."""

    original_build_protocol = suite_base.build_protocol
    original_refresh = suite_base.refresh_summaries
    replacements = {
        "SUITE_VERSION": SUITE_VERSION,
        "CLASSIFIER_STAGE": CLASSIFIER_STAGE,
        "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
        "IMPLEMENTATION_FILES": IMPLEMENTATION_FILES,
        "CANONICAL_CLASSIFIER_NAMES": CANONICAL_RF125_CLASSIFIER_NAMES,
        "CLASSIFIER_DISPLAY_NAMES": RF125_CLASSIFIER_DISPLAY_NAMES,
        "DEBUG_SMALL_ARCHITECTURES": DEBUG_SMALL_ARCHITECTURES,
        "build_residual_classifier": build_rf125_classifier,
        "classifier_config": rf125_classifier_config,
        "parameter_count": parameter_count,
        "paired_delta_summary": paired_delta_summary,
        "parse_args": parse_args,
    }
    saved = {
        name: getattr(suite_base, name)
        for name in replacements
    }
    try:
        for name, value in replacements.items():
            setattr(suite_base, name, value)
        suite_base.build_protocol = lambda *args, **kwargs: build_replacement_protocol(
            original_build_protocol,
            *args,
            **kwargs,
        )
        suite_base.refresh_summaries = lambda output_dir, config: (
            refresh_replacement_summaries(
                original_refresh,
                output_dir,
                config,
            )
        )
        yield suite_base
    finally:
        suite_base.build_protocol = original_build_protocol
        suite_base.refresh_summaries = original_refresh
        for name, value in saved.items():
            setattr(suite_base, name, value)


def main() -> None:
    with configured_base_suite() as configured:
        configured.main()


if __name__ == "__main__":
    main()
