#!/usr/bin/env python
"""Aggregate multi-seed FoG baselines at the held-out-subject level.

Seeds are repeated model fits, not independent clinical samples.  This script
therefore averages seeds *within each held-out subject first* and only then
computes subject-macro summaries and paired bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.resume import atomic_json_dump


DEFAULT_SEEDS = (3407, 3408, 3409, 3410, 3411)
PAPER_METHODS = ("freeze_index", "tf_svm", "tf_rf", "cnn_gru")
DISPLAY_NAMES = {
    "freeze_index": "Freeze Index",
    "tf_svm": "Time-frequency + SVM",
    "tf_rf": "Time-frequency + RF",
    "cnn_gru": "CNN-GRU",
}
WINDOW_METRICS = (
    "pr_auc",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "fog_recall",
    "specificity",
    "precision",
    "fog_f1",
)
EVENT_METRICS = (
    "event_sensitivity",
    "false_alarm_events_per_hour",
    "median_detection_delay_sec",
)
ALL_METRICS = (*WINDOW_METRICS, *EVENT_METRICS)
COMPATIBILITY_KEYS = (
    "suite_version",
    "dataset_adapter",
    "data_sha256",
    "sampling_rate_hz",
    "channel_names",
    "excluded_subjects",
    "subjects",
    "folds_resolved",
    "methods_resolved",
    "context_samples",
    "horizon_samples",
    "stride_samples",
    "history_samples",
    "normal_guard_samples",
    "fog_fraction_threshold",
    "event_metrics",
    "sensor_channel_indices",
    "fi_channel_indices",
    "fi_aggregation",
    "fi_squared_ratio",
    "svm_kernel",
    "svm_c_grid",
    "rf_n_estimators",
    "rf_min_samples_leaf_grid",
    "rf_max_depth",
    "rf_max_features",
    "cnn_channels",
    "gru_hidden",
    "gru_layers",
    "dropout",
    "cnn_pos_weight_policy",
    "cnn_gradient_clip_norm",
)


def parse_int_list(specification: str, label: str) -> tuple[int, ...]:
    values = tuple(
        int(value.strip())
        for value in str(specification).split(",")
        if value.strip()
    )
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must contain unique integers")
    return values


def parse_method_list(specification: str) -> tuple[str, ...]:
    values = tuple(
        value.strip().lower()
        for value in str(specification).split(",")
        if value.strip()
    )
    if not values or len(values) != len(set(values)):
        raise ValueError("--methods must contain unique names")
    unknown = sorted(set(values) - set(PAPER_METHODS))
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; expected={PAPER_METHODS}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subject-level aggregation for the five-seed FoG baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    parser.add_argument("--seed-dir-template", default="seed_{seed}")
    parser.add_argument("--methods", default=",".join(PAPER_METHODS))
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=3411)
    parser.add_argument(
        "--reference-pr-csv",
        type=Path,
        help=(
            "Optional seed-matched proposed-model table with seed, "
            "test_subject and pr_auc columns. The paper delta is reference "
            "minus each baseline."
        ),
    )
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def atomic_csv_write(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def optional_float(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "none", "nan"}:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def mean_available(values: Iterable[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return float(np.mean(available)) if available else None


def summary(values: Iterable[float | None]) -> dict[str, Any]:
    available = np.asarray(
        [float(value) for value in values if value is not None],
        dtype=np.float64,
    )
    if not len(available):
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(available.mean()),
        "std": float(available.std(ddof=0)),
        "min": float(available.min()),
        "max": float(available.max()),
        "n": int(len(available)),
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
        raise ValueError("Bootstrap values must be a non-empty finite vector")
    generator = np.random.default_rng(int(seed))
    # Chunking bounds memory when the bootstrap count is large.
    means: list[np.ndarray] = []
    remaining = int(samples)
    while remaining:
        chunk = min(remaining, 10000)
        indices = generator.integers(
            0,
            len(vector),
            size=(chunk, len(vector)),
        )
        means.append(vector[indices].mean(axis=1))
        remaining -= chunk
    distribution = np.concatenate(means)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return float(vector.mean()), float(low), float(high)


def _load_seed_rows(
    root: Path,
    seed: int,
    methods: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_path = root / "config.json"
    status_path = root / "status.json"
    summary_path = root / "fold_summary.csv"
    for path in (config_path, status_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    with status_path.open("r", encoding="utf-8") as handle:
        status = json.load(handle)
    if int(config["seed"]) != int(seed):
        raise ValueError(f"{root} declares seed={config['seed']}, expected={seed}")
    if status.get("status") != "complete":
        raise ValueError(f"Incomplete seed suite {root}: {status}")
    if not set(methods).issubset(config["methods_resolved"]):
        raise ValueError(
            f"{root} is missing methods "
            f"{sorted(set(methods) - set(config['methods_resolved']))}"
        )
    audit_path = root / "audit_report.json"
    if audit_path.exists():
        with audit_path.open("r", encoding="utf-8") as handle:
            audit = json.load(handle)
        if audit.get("status") != "pass":
            raise ValueError(f"Seed audit did not pass: {audit_path}")

    rows: list[dict[str, Any]] = []
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") not in methods:
                continue
            converted: dict[str, Any] = {
                **row,
                "seed": int(seed),
                "test_subject": str(row["test_subject"]),
                "method": str(row["method"]),
            }
            for metric in ALL_METRICS:
                converted[metric] = optional_float(row.get(metric))
            rows.append(converted)
    expected = len(config["folds_resolved"]) * len(methods)
    if len(rows) != expected:
        raise ValueError(
            f"{root} has {len(rows)} requested fold rows, expected {expected}"
        )
    return config, rows


def _compatible(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> list[str]:
    return [
        key
        for key in COMPATIBILITY_KEYS
        if reference.get(key) != candidate.get(key)
    ]


def load_reference_subject_pr(
    path: Path,
    expected_subjects: Sequence[str],
    expected_seeds: Sequence[int],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"seed", "test_subject", "pr_auc"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} must contain columns {sorted(required)}"
            )
        for row in reader:
            value = optional_float(row.get("pr_auc"))
            if value is not None:
                subject = str(row["test_subject"])
                seed = int(row["seed"])
                key = (subject, seed)
                if key in seen:
                    raise ValueError(f"Duplicate reference row {key} in {path}")
                seen.add(key)
                grouped[subject].append(value)
    missing = sorted(set(expected_subjects) - set(grouped))
    if missing:
        raise ValueError(f"Reference PR table is missing subjects {missing}")
    expected_pairs = {
        (str(subject), int(seed))
        for subject in expected_subjects
        for seed in expected_seeds
    }
    if seen != expected_pairs:
        missing_pairs = sorted(expected_pairs - seen)
        extra_pairs = sorted(seen - expected_pairs)
        raise ValueError(
            "Reference PR table is not seed-matched; "
            f"missing={missing_pairs}, extra={extra_pairs}"
        )
    return {
        subject: float(np.mean(grouped[subject]))
        for subject in expected_subjects
    }


def formatted(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "NA"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    root = args.output_dir.resolve()
    seeds = parse_int_list(args.seeds, "--seeds")
    methods = parse_method_list(args.methods)
    failures: list[str] = []
    all_rows: list[dict[str, Any]] = []
    configs: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        seed_root = root / args.seed_dir_template.format(seed=seed)
        try:
            config, rows = _load_seed_rows(seed_root, seed, methods)
            configs[seed] = config
            all_rows.extend(rows)
        except Exception as error:
            failures.append(f"seed={seed}: {error}")
            if not args.allow_partial:
                break
    if failures and not args.allow_partial:
        raise RuntimeError("; ".join(failures))
    if not configs:
        raise RuntimeError("No complete seed suites were available")

    first_seed = next(iter(configs))
    reference_config = configs[first_seed]
    compatibility_failures: list[str] = []
    for seed, config in configs.items():
        differing = _compatible(reference_config, config)
        if differing:
            compatibility_failures.append(
                f"seed={seed} differs in {differing}"
            )
    if compatibility_failures:
        raise ValueError("; ".join(compatibility_failures))

    expected_subjects = tuple(reference_config["folds_resolved"])
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["method"], row["test_subject"])].append(row)
    subject_rows: list[dict[str, Any]] = []
    for method in methods:
        for subject in expected_subjects:
            rows = grouped.get((method, subject), [])
            if len(rows) != len(configs):
                message = (
                    f"{method}/{subject} has {len(rows)} seeds, "
                    f"expected {len(configs)}"
                )
                if not args.allow_partial:
                    raise ValueError(message)
                failures.append(message)
            result: dict[str, Any] = {
                "method": method,
                "display_name": DISPLAY_NAMES[method],
                "test_subject": subject,
                "seed_count": len(rows),
                "seeds": ",".join(str(row["seed"]) for row in rows),
            }
            for metric in ALL_METRICS:
                values = [row.get(metric) for row in rows]
                result[metric] = mean_available(values)
                available = [
                    float(value) for value in values if value is not None
                ]
                result[f"{metric}_seed_std"] = (
                    float(np.std(available, ddof=0)) if available else None
                )
            subject_rows.append(result)

    # Deterministic FI should be exactly reproducible across seeds.
    fi_seed_spread = max(
        (
            float(row["pr_auc_seed_std"] or 0.0)
            for row in subject_rows
            if row["method"] == "freeze_index"
        ),
        default=0.0,
    )
    if fi_seed_spread > 1e-12:
        failures.append(
            "Freeze Index changed across seeds; deterministic reproducibility "
            f"check failed (max subject PR std={fi_seed_spread})"
        )

    method_subject_rows = {
        method: [
            row for row in subject_rows if row["method"] == method
        ]
        for method in methods
    }
    aggregate: dict[str, Any] = {}
    for method, rows in method_subject_rows.items():
        aggregate[method] = {
            "method": method,
            "display_name": DISPLAY_NAMES[method],
            "n_subjects": len(rows),
            "n_seeds": len(configs),
            "subject_macro_after_seed_mean": {
                metric: summary(row.get(metric) for row in rows)
                for metric in ALL_METRICS
            },
        }

    pairwise_rows: list[dict[str, Any]] = []
    by_method_subject = {
        (row["method"], row["test_subject"]): row
        for row in subject_rows
    }
    for first, second in itertools.combinations(methods, 2):
        deltas = [
            float(by_method_subject[(first, subject)]["pr_auc"])
            - float(by_method_subject[(second, subject)]["pr_auc"])
            for subject in expected_subjects
        ]
        mean, low, high = bootstrap_mean_ci(
            deltas,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        pairwise_rows.append(
            {
                "comparison": f"{first}_minus_{second}",
                "new": first,
                "reference": second,
                "mean_delta_pr_auc": mean,
                "ci_low": low,
                "ci_high": high,
                "n_paired_subjects": len(deltas),
                "wins": int(np.sum(np.asarray(deltas) > 0.0)),
                "ties": int(np.sum(np.asarray(deltas) == 0.0)),
                "losses": int(np.sum(np.asarray(deltas) < 0.0)),
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_seed": args.bootstrap_seed,
            }
        )

    external_reference: dict[str, float] | None = None
    if args.reference_pr_csv is not None:
        external_reference = load_reference_subject_pr(
            args.reference_pr_csv.resolve(),
            expected_subjects,
            tuple(configs),
        )

    publication_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for method in methods:
        metrics = aggregate[method]["subject_macro_after_seed_mean"]
        delta_mean: float | None = None
        delta_low: float | None = None
        delta_high: float | None = None
        if external_reference is not None:
            # Positive values mean the proposed/external reference is better.
            deltas = [
                external_reference[subject]
                - float(by_method_subject[(method, subject)]["pr_auc"])
                for subject in expected_subjects
            ]
            delta_mean, delta_low, delta_high = bootstrap_mean_ci(
                deltas,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            )
        delta_formatted = (
            f"{delta_mean:.4f} [{delta_low:.4f}, {delta_high:.4f}]"
            if delta_mean is not None
            else "NA (reference required)"
        )
        publication_rows.append(
            {
                "Method": DISPLAY_NAMES[method],
                "PR-AUC": formatted(
                    metrics["pr_auc"]["mean"],
                    metrics["pr_auc"]["std"],
                ),
                "Delta PR-AUC [95% CI]": delta_formatted,
                "Balanced Accuracy": formatted(
                    metrics["balanced_accuracy"]["mean"],
                    metrics["balanced_accuracy"]["std"],
                ),
                "Macro-F1": formatted(
                    metrics["macro_f1"]["mean"],
                    metrics["macro_f1"]["std"],
                ),
                "AUROC": formatted(
                    metrics["roc_auc"]["mean"],
                    metrics["roc_auc"]["std"],
                ),
                "FoG Sensitivity/Recall": formatted(
                    metrics["fog_recall"]["mean"],
                    metrics["fog_recall"]["std"],
                ),
                "Specificity": formatted(
                    metrics["specificity"]["mean"],
                    metrics["specificity"]["std"],
                ),
                "FoG Precision": formatted(
                    metrics["precision"]["mean"],
                    metrics["precision"]["std"],
                ),
                "FoG F1": formatted(
                    metrics["fog_f1"]["mean"],
                    metrics["fog_f1"]["std"],
                ),
                "Subjects": len(expected_subjects),
                "Seeds": len(configs),
            }
        )
        event_rows.append(
            {
                "Method": DISPLAY_NAMES[method],
                "Event Sensitivity": formatted(
                    metrics["event_sensitivity"]["mean"],
                    metrics["event_sensitivity"]["std"],
                ),
                "FA/h": formatted(
                    metrics["false_alarm_events_per_hour"]["mean"],
                    metrics["false_alarm_events_per_hour"]["std"],
                ),
                "Median Detection Delay (s)": formatted(
                    metrics["median_detection_delay_sec"]["mean"],
                    metrics["median_detection_delay_sec"]["std"],
                ),
                "Subjects": len(expected_subjects),
                "Seeds": len(configs),
            }
        )

    raw_columns = [
        "seed",
        "method",
        "test_subject",
        *ALL_METRICS,
    ]
    atomic_csv_write(root / "fold_seed_metrics.csv", all_rows, raw_columns)
    subject_columns = [
        "method",
        "display_name",
        "test_subject",
        "seed_count",
        "seeds",
        *ALL_METRICS,
        *(f"{metric}_seed_std" for metric in ALL_METRICS),
    ]
    atomic_csv_write(
        root / "subject_seed_averaged_metrics.csv",
        subject_rows,
        subject_columns,
    )
    atomic_csv_write(
        root / "pairwise_pr_auc_deltas.csv",
        pairwise_rows,
        (
            "comparison",
            "new",
            "reference",
            "mean_delta_pr_auc",
            "ci_low",
            "ci_high",
            "n_paired_subjects",
            "wins",
            "ties",
            "losses",
            "bootstrap_samples",
            "bootstrap_seed",
        ),
    )
    atomic_csv_write(
        root / "publication_table.csv",
        publication_rows,
        tuple(publication_rows[0]),
    )
    atomic_csv_write(
        root / "event_metrics_table.csv",
        event_rows,
        tuple(event_rows[0]),
    )
    payload = {
        "aggregation_version": "fog_reference_baselines_multiseed.v1",
        "aggregation_unit": "held_out_subject_after_within_subject_seed_mean",
        "seeds_requested": list(seeds),
        "seeds_completed": list(configs),
        "methods": list(methods),
        "subjects": list(expected_subjects),
        "reference_pr_csv": (
            str(args.reference_pr_csv.resolve())
            if args.reference_pr_csv is not None
            else None
        ),
        "delta_definition": (
            "external_reference_minus_baseline"
            if external_reference is not None
            else None
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "freeze_index_max_subject_seed_pr_std": fi_seed_spread,
        "aggregate": aggregate,
        "failures": failures,
        "status": "complete" if not failures else "warning",
    }
    atomic_json_dump(payload, root / "aggregate_multiseed_metrics.json")
    atomic_json_dump(
        {
            "audit_version": "fog_reference_baselines_multiseed_audit.v1",
            "expected_seed_count": len(seeds),
            "completed_seed_count": len(configs),
            "expected_cells": len(seeds) * len(methods) * len(expected_subjects),
            "loaded_cells": len(all_rows),
            "failures": failures,
            "status": "pass" if not failures else "fail",
        },
        root / "multiseed_audit_report.json",
    )
    print(
        f"[multiseed] seeds={len(configs)}/{len(seeds)} "
        f"cells={len(all_rows)} output={root}",
        flush=True,
    )
    if failures and not args.allow_partial:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
