#!/usr/bin/env python
"""Run focused FoG-STAR MLP frequency-feature sweeps.

This script keeps training inside the existing run.py framework. It only
materializes YAML configs with different long-window trend feature sets, then
launches them sequentially and updates the shared overview after each run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from fog_results_overview import enrich_overview_row, update_overview
import run_fogstar_mlp_long_short_sweep as long_short


FEATURE_SETS: dict[str, list[str]] = {
    "time_trend": ["mean", "std", "delta", "slope"],
    "fft_global": ["mean", "std", "delta", "slope", "fft_energy", "fft_entropy", "fft_centroid", "fft_peak_freq"],
    "band_power": ["mean", "std", "delta", "slope", "bandpower_low", "bandpower_high"],
    "band_ratio": ["mean", "std", "delta", "slope", "bandpower_low", "bandpower_high", "bandpower_ratio"],
    "freeze_index": ["mean", "std", "delta", "slope", "bandpower_low", "bandpower_high", "freeze_index"],
    "dominant_power": ["mean", "std", "delta", "slope", "dominant_power"],
    "band_freeze_dom": [
        "mean",
        "std",
        "delta",
        "slope",
        "bandpower_low",
        "bandpower_high",
        "freeze_index",
        "dominant_power",
    ],
    "full_freq": [
        "mean",
        "std",
        "delta",
        "slope",
        "fft_energy",
        "fft_entropy",
        "fft_centroid",
        "fft_peak_freq",
        "bandpower_low",
        "bandpower_high",
        "bandpower_ratio",
        "freeze_index",
        "dominant_power",
    ],
}

SUMMARY_COLUMNS = [
    "feature_set",
    "status",
    "fold_count",
    "test_f1_macro_mean",
    "test_f1_macro_std",
    "test_recall_macro_mean",
    "test_recall_macro_std",
    "test_pr_auc_macro_mean",
    "test_pr_auc_macro_std",
    "pre_fog_recall_mean",
    "pre_fog_recall_std",
    "pre_fog_f1_mean",
    "pre_fog_f1_std",
    "test_balanced_accuracy_mean",
    "test_accuracy_mean",
    "cm_true_normal_pred_pre_fog",
    "cm_true_fog_pred_pre_fog",
    "elapsed_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FoG-STAR MLP focused frequency feature sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--feature-set",
        action="append",
        choices=sorted(FEATURE_SETS),
        default=[],
        help="Feature set to run. Can be repeated. Defaults to all focused sets.",
    )
    parser.add_argument(
        "--combo",
        action="append",
        default=[],
        help="Short,long seconds pair. Defaults to 1,6. Can be repeated.",
    )
    parser.add_argument("--base-config", type=Path, default=long_short.BASE_CONFIG)
    parser.add_argument(
        "--generated-config-dir",
        type=Path,
        default=Path("outputs/generated_configs/fogstar_mlp_frequency_feature_sweep"),
    )
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--python", default=long_short.sys.executable)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra run.py override applied to every config, for example train.epochs=1.",
    )
    parser.add_argument("--no-collect", action="store_true")
    parser.add_argument(
        "--overview-csv",
        type=Path,
        default=Path("outputs/fog_results_overview.csv"),
        help="Shared CSV updated after each completed experiment.",
    )
    parser.add_argument("--no-overview", action="store_true")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/fogstar_mlp_frequency_feature_sweep_summary.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/fogstar_mlp_frequency_feature_sweep_summary.json"),
    )
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def selected_combos(args: argparse.Namespace) -> list[tuple[float, float]]:
    values = args.combo or ["1,6"]
    return long_short.unique_combos([long_short.parse_combo(value) for value in values])


def selected_feature_sets(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    names = args.feature_set or list(FEATURE_SETS)
    return [(name, FEATURE_SETS[name]) for name in names]


def collect_result(config_path: Path, returncode: int, elapsed_sec: float, feature_set: str) -> dict[str, Any]:
    row = long_short.collect_result(config_path, returncode, elapsed_sec)
    row["feature_set"] = feature_set
    return enrich_overview_row(row)


def print_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        return
    widths = {
        column: max(len(column), *(len("" if row.get(column) is None else str(row.get(column))) for row in rows))
        for column in columns
    }
    print(f"\n===== {title} =====")
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(("" if row.get(column) is None else str(row.get(column))).ljust(widths[column]) for column in columns))


def print_ranked(rows: list[dict[str, Any]], metric: str, top_k: int) -> None:
    ranked = [
        row
        for row in rows
        if row.get("status") == "ok" and isinstance(row.get(metric), (int, float))
    ]
    if not ranked:
        return
    ranked.sort(key=lambda row: row[metric], reverse=True)
    columns = [
        "feature_set",
        metric,
        "test_f1_macro_mean",
        "test_recall_macro_mean",
        "test_pr_auc_macro_mean",
        "pre_fog_recall_mean",
        "pre_fog_f1_mean",
        "cm_true_normal_pred_pre_fog",
        "cm_true_fog_pred_pre_fog",
    ]
    print_table(f"Top {min(len(ranked), top_k)} by {metric}", ranked[:top_k], columns)


def main() -> None:
    args = parse_args()
    base_cfg = long_short.load_yaml(long_short.resolve_repo_path(args.base_config))
    combos = selected_combos(args)
    feature_sets = selected_feature_sets(args)
    print(
        f"[INFO] combinations={len(combos)} feature_sets={len(feature_sets)} "
        f"summary={long_short.resolve_repo_path(args.summary_csv)}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for short_seconds, long_seconds in combos:
        for feature_set, trend_features in feature_sets:
            config_path = long_short.materialize_config(
                base_cfg,
                short_seconds,
                long_seconds,
                args.generated_config_dir,
                trend_features=trend_features,
                experiment_suffix=f"_freqfeat_{feature_set}",
            )
            cfg = long_short.load_yaml(config_path)
            print(f"\n===== {feature_set}: {cfg.get('project', {}).get('name', config_path.stem)} =====", flush=True)
            existing_summary = long_short.summary_path_for(config_path)
            if args.skip_existing and existing_summary.exists() and not args.dry_run:
                print(f"[SKIP] existing summary: {existing_summary}", flush=True)
                row = collect_result(config_path, 0, 0.0, feature_set)
                rows.append(row)
                long_short.write_csv(args.summary_csv, rows)
                long_short.write_json(args.summary_json, rows)
                if not args.no_overview:
                    overview_path = update_overview(args.overview_csv, row, sweep="long_short_freq_features")
                    print(f"[OVERVIEW] updated {overview_path}", flush=True)
                continue

            command = long_short.build_command(args, config_path)
            start = time.perf_counter()
            returncode = long_short.run_command(command, args.dry_run)
            elapsed = time.perf_counter() - start
            row = collect_result(config_path, returncode, elapsed, feature_set)
            rows.append(row)
            long_short.write_csv(args.summary_csv, rows)
            long_short.write_json(args.summary_json, rows)
            if not args.dry_run and not args.no_overview:
                overview_path = update_overview(args.overview_csv, row, sweep="long_short_freq_features")
                print(f"[OVERVIEW] updated {overview_path}", flush=True)
            if returncode != 0 and not args.continue_on_error:
                print_table("Frequency feature sweep summary", rows, SUMMARY_COLUMNS)
                raise SystemExit(returncode)

    if not args.no_collect:
        print_table("Frequency feature sweep summary", rows, SUMMARY_COLUMNS)
        print_ranked(rows, "test_f1_macro_mean", args.top_k)
        print_ranked(rows, "pre_fog_f1_mean", args.top_k)
        print_ranked(rows, "test_pr_auc_macro_mean", args.top_k)
        print(f"\n[SUMMARY] csv={long_short.resolve_repo_path(args.summary_csv)}")
        print(f"[SUMMARY] json={long_short.resolve_repo_path(args.summary_json)}")


if __name__ == "__main__":
    main()
