#!/usr/bin/env python
"""Summarize S01 validation RMSE by forecast lead time from frozen v4 runs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnbr_fog.resume import (  # noqa: E402
    atomic_json_dump,
    canonical_fingerprint,
    sha256_file,
)


EXPERIMENT_VERSION = "daphnet_s01_gru_lead_rmse_summary.v1"
UPSTREAM_VERSION = "daphnet_s01_gru_convergence_sequence.v4"
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
SAMPLING_RATE_HZ = 64
BAND_SAMPLES = 16
HORIZONS = (
    ("h025", 16, 0.25),
    ("h050", 32, 0.5),
    ("h100", 64, 1.0),
    ("h200", 128, 2.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive per-lead validation RMSE from frozen S01 horizon runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / "daphnet_s01_gru_convergence_sequence_v4"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(REPO_ROOT / "outputs" / "daphnet_s01_gru_lead_rmse_v1"),
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def validate_inventory(
    root: Path, *, label: str, exact: bool = True
) -> dict[str, Any]:
    done_path = root / "DONE.json"
    if not done_path.exists():
        raise FileNotFoundError(f"Missing {label} DONE manifest: {done_path}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("status") != "complete":
        raise RuntimeError(f"Incomplete {label}: {done_path}")
    declared = dict(done.get("artifacts", {}))
    actual = {
        str(path.relative_to(root)).replace("\\", "/"): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "DONE.json"
    }
    if not set(declared).issubset(actual) or (exact and set(declared) != set(actual)):
        missing = sorted(set(declared) - set(actual))
        extra = sorted(set(actual) - set(declared))
        raise RuntimeError(
            f"{label} inventory mismatch; missing={missing}, extra={extra}"
        )
    for relative, expected in declared.items():
        if sha256_file(actual[relative]) != expected:
            raise RuntimeError(f"{label} artifact hash mismatch: {relative}")
    return done


def band_rmse_by_seed(
    per_seed_point_rmse: np.ndarray,
    band_samples: int = BAND_SAMPLES,
) -> list[np.ndarray]:
    """Return correctly pooled band RMSE for each seed.

    Pointwise values are already square roots of MSE over windows and channels,
    so pooling a time band requires RMS rather than their arithmetic mean.
    """

    values = np.asarray(per_seed_point_rmse, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Expected [seed, lead_sample] RMSE matrix")
    if band_samples <= 0 or values.shape[1] % band_samples:
        raise ValueError("Horizon must be divisible by the positive band size")
    return [
        np.sqrt(np.mean(np.square(values[:, start:end]), axis=1))
        for start, end in (
            (start, start + band_samples)
            for start in range(0, values.shape[1], band_samples)
        )
    ]


def numeric_stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("numeric_stats requires a finite non-empty vector")
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def output_is_complete(root: Path, fingerprint: str) -> bool:
    if not (root / "DONE.json").exists():
        return False
    done = validate_inventory(root, label="derived lead-RMSE output")
    if done.get("protocol_fingerprint") != fingerprint:
        raise RuntimeError("Existing lead-RMSE output has a different protocol")
    return True


def main() -> None:
    args = parse_args()
    upstream = args.upstream_dir.resolve()
    stage = upstream / "04_horizon"
    # The suite-level DONE intentionally inventories only locked root/final
    # artifacts; each stage has its own exact manifest.
    upstream_done = validate_inventory(
        upstream, label="upstream v4 suite", exact=False
    )
    stage_done = validate_inventory(stage, label="upstream horizon stage")
    upstream_config = json.loads(
        (upstream / "config.json").read_text(encoding="utf-8")
    )
    stage_config = json.loads((stage / "config.json").read_text(encoding="utf-8"))
    stage_aggregate = json.loads(
        (stage / "aggregate.json").read_text(encoding="utf-8")
    )
    if upstream_done.get("experiment_version") != UPSTREAM_VERSION:
        raise RuntimeError("Unexpected upstream suite version")
    if stage_done.get("experiment_version") != UPSTREAM_VERSION:
        raise RuntimeError("Unexpected horizon-stage version")
    if upstream_config.get("protocol_fingerprint") != upstream_done.get(
        "protocol_fingerprint"
    ):
        raise RuntimeError("Upstream suite config/DONE fingerprint mismatch")
    if stage_config.get("protocol_fingerprint") != stage_done.get(
        "protocol_fingerprint"
    ):
        raise RuntimeError("Horizon-stage config/DONE fingerprint mismatch")
    if tuple(stage_config.get("seeds", ())) != EXPECTED_SEEDS:
        raise RuntimeError("Unexpected horizon-stage seeds")
    configured_horizons = tuple(
        (item["id"], int(item["samples"]), float(item["seconds"]))
        for item in stage_config.get("horizons", ())
    )
    if configured_horizons != HORIZONS:
        raise RuntimeError(f"Unexpected horizon protocol: {configured_horizons}")

    summary_paths = [
        stage / "arms" / horizon_id / "runs" / f"seed_{seed}" / "summary.json"
        for horizon_id, _, _ in HORIZONS
        for seed in EXPECTED_SEEDS
    ]
    source_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "cnbr_fog" / "resume.py",
    )
    protocol = {
        "experiment_version": EXPERIMENT_VERSION,
        "purpose": "Report validation RMSE for every lead sample and 0.25-second band",
        "upstream_suite_protocol_fingerprint": upstream_config[
            "protocol_fingerprint"
        ],
        "upstream_horizon_protocol_fingerprint": stage_config[
            "protocol_fingerprint"
        ],
        "upstream_done_sha256": sha256_file(upstream / "DONE.json"),
        "upstream_horizon_done_sha256": sha256_file(stage / "DONE.json"),
        "seeds": list(EXPECTED_SEEDS),
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "band_samples": BAND_SAMPLES,
        "band_seconds": BAND_SAMPLES / SAMPLING_RATE_HZ,
        "horizons": [
            {"id": item[0], "samples": item[1], "seconds": item[2]}
            for item in HORIZONS
        ],
        "aggregation": {
            "point": "mean/std/min/max across five seed-specific pointwise RMSE values",
            "band": "per seed sqrt(mean(pointwise_RMSE^2)), then mean/std/min/max across seeds",
            "metric_scale": "Robust-Scaler standardized signal units",
            "split": "clean-normal validation only",
        },
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in source_paths
        },
        "input_summary_sha256": {
            str(path.relative_to(upstream)).replace("\\", "/"): sha256_file(path)
            for path in summary_paths
        },
        "test_record_evaluated": False,
    }
    fingerprint = canonical_fingerprint(protocol)
    protocol["protocol_fingerprint"] = fingerprint
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if output_is_complete(root, fingerprint):
        print(f"Completed lead-RMSE summary verified: {root}")
        return
    if any(root.iterdir()):
        raise FileExistsError(f"Non-empty incomplete output directory: {root}")
    atomic_json_dump(protocol, root / "config.json")

    sample_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    structured: dict[str, Any] = {}
    for horizon_id, horizon_samples, horizon_seconds in HORIZONS:
        summaries = [
            json.loads(
                (
                    stage
                    / "arms"
                    / horizon_id
                    / "runs"
                    / f"seed_{seed}"
                    / "summary.json"
                ).read_text(encoding="utf-8")
            )
            for seed in EXPECTED_SEEDS
        ]
        if tuple(int(item["seed"]) for item in summaries) != EXPECTED_SEEDS:
            raise RuntimeError(f"Seed order mismatch for {horizon_id}")
        point_matrix = np.asarray(
            [
                item["best"]["validation"]["per_horizon_rmse_scaled"]
                for item in summaries
            ],
            dtype=np.float64,
        )
        if point_matrix.shape != (len(EXPECTED_SEEDS), horizon_samples):
            raise RuntimeError(
                f"Unexpected per-lead shape for {horizon_id}: {point_matrix.shape}"
            )
        overall_by_seed = np.asarray(
            [item["best_validation_rmse"] for item in summaries], dtype=np.float64
        )
        reconstructed = np.sqrt(np.mean(np.square(point_matrix), axis=1))
        if not np.allclose(reconstructed, overall_by_seed, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Pointwise RMSE does not reconstruct {horizon_id}")
        expected_mean = stage_aggregate["arms"][horizon_id][
            "best_validation_rmse"
        ]["mean"]
        if not math.isclose(
            float(np.mean(overall_by_seed)), float(expected_mean), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"Aggregate RMSE mismatch for {horizon_id}")

        for index in range(horizon_samples):
            stats = numeric_stats(point_matrix[:, index])
            sample_rows.append(
                {
                    "horizon_id": horizon_id,
                    "model_horizon_seconds": horizon_seconds,
                    "lead_sample": index + 1,
                    "lead_start_seconds": index / SAMPLING_RATE_HZ,
                    "lead_end_seconds": (index + 1) / SAMPLING_RATE_HZ,
                    "rmse_mean": stats["mean"],
                    "rmse_std": stats["std"],
                    "rmse_min": stats["min"],
                    "rmse_max": stats["max"],
                    "seed_count": len(EXPECTED_SEEDS),
                }
            )
        horizon_bands: list[dict[str, Any]] = []
        for band_index, values in enumerate(band_rmse_by_seed(point_matrix)):
            start = band_index * BAND_SAMPLES
            end = start + BAND_SAMPLES
            stats = numeric_stats(values)
            row = {
                "horizon_id": horizon_id,
                "model_horizon_seconds": horizon_seconds,
                "lead_band_start_seconds": start / SAMPLING_RATE_HZ,
                "lead_band_end_seconds": end / SAMPLING_RATE_HZ,
                "lead_sample_start": start + 1,
                "lead_sample_end": end,
                "rmse_mean": stats["mean"],
                "rmse_std": stats["std"],
                "rmse_min": stats["min"],
                "rmse_max": stats["max"],
                "seed_count": len(EXPECTED_SEEDS),
            }
            band_rows.append(row)
            horizon_bands.append(row)
        overall_stats = numeric_stats(overall_by_seed)
        overall_row = {
            "horizon_id": horizon_id,
            "horizon_samples": horizon_samples,
            "horizon_seconds": horizon_seconds,
            "rmse_mean": overall_stats["mean"],
            "rmse_std": overall_stats["std"],
            "rmse_min": overall_stats["min"],
            "rmse_max": overall_stats["max"],
            "seed_count": len(EXPECTED_SEEDS),
        }
        overall_rows.append(overall_row)
        structured[horizon_id] = {
            "overall": overall_row,
            "bands": horizon_bands,
            "point_count": horizon_samples,
        }

    write_csv(root / "overall_horizon_rmse.csv", overall_rows)
    write_csv(root / "lead_band_rmse.csv", band_rows)
    write_csv(root / "lead_sample_rmse.csv", sample_rows)
    aggregate = {
        "protocol_fingerprint": fingerprint,
        "seeds": list(EXPECTED_SEEDS),
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "band_samples": BAND_SAMPLES,
        "band_seconds": BAND_SAMPLES / SAMPLING_RATE_HZ,
        "horizons": structured,
        "sample_row_count": len(sample_rows),
        "band_row_count": len(band_rows),
        "test_record_evaluated": False,
    }
    atomic_json_dump(aggregate, root / "aggregate.json")

    overall_table = "\n".join(
        f"| {row['horizon_seconds']:g} | {row['horizon_samples']} | "
        f"{row['rmse_mean']:.6f} | {row['rmse_std']:.6f} |"
        for row in overall_rows
    )
    band_table = "\n".join(
        f"| {row['model_horizon_seconds']:g} | "
        f"{row['lead_band_start_seconds']:.2f}–{row['lead_band_end_seconds']:.2f} | "
        f"{row['rmse_mean']:.6f} | {row['rmse_std']:.6f} |"
        for row in band_rows
    )
    report = f"""# S01 GRU 未来 lead-time RMSE

本报告直接读取冻结的 v4 horizon 消融最优 checkpoint 验证结果，不重新训练。所有数值均为 295 个 clean-normal 验证窗口、9 通道上的 Robust-Scaler 标准化 RMSE，并先在每个 seed 内计算，再汇总 5 个 seed。R02 未读取或评估。

## 每个预测长度的整体 RMSE

| 模型预测长度（秒） | 点数 | RMSE 均值 | seed 标准差 |
|---:|---:|---:|---:|
{overall_table}

## 每 0.25 秒未来区间 RMSE

分段 RMSE 在每个 seed 内按 `sqrt(mean(pointwise_RMSE²))` 合并，不能用逐点 RMSE 的算术平均替代。不同 horizon 行来自分别训练的模型，因此可比较难度，但不能解释为同一个模型的截断输出。

| 模型预测长度（秒） | 未来区间（秒） | RMSE 均值 | seed 标准差 |
|---:|---:|---:|---:|
{band_table}

逐个采样点（1/64 秒分辨率）的全部 {len(sample_rows)} 行见 `lead_sample_rmse.csv`。
"""
    atomic_text(root / "report.md", report)
    atomic_json_dump(
        {
            "created_utc": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "protocol_fingerprint": fingerprint,
        },
        root / "runtime.json",
    )
    artifacts = {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "DONE.json"
    }
    atomic_json_dump(
        {
            "status": "complete",
            "experiment_version": EXPERIMENT_VERSION,
            "protocol_fingerprint": fingerprint,
            "completed_utc": utc_now(),
            "test_record_evaluated": False,
            "artifacts": artifacts,
        },
        root / "DONE.json",
    )
    validate_inventory(root, label="derived lead-RMSE output")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"Results: {root}")


if __name__ == "__main__":
    main()
