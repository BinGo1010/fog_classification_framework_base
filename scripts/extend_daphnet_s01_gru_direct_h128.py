#!/usr/bin/env python
"""Strict H=128 direct mean-only budget extension for the S01 v4 suite.

The canonical reference is ``04_horizon/arms/h200`` from the completed v4
sequence.  That arm uses :class:`GRUMeanForecaster` with the pure mean direct
decoder.  ``02_long_mean`` is also validated and hashed, but it uses the
legacy :class:`GRUNBM` mean path and therefore is not treated as an identical
training trajectory.

Only the maximum optimizer-step budget changes, from 500 to 2000.  The script
starts each seed from scratch, requires the first 500 steps to reproduce the
canonical v4 history row-for-row, and never opens the held-out R02 array.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import diagnose_daphnet_s01_gru_convergence as diagnostic  # noqa: E402
import run_daphnet_s01_gru_convergence_sequence as suite  # noqa: E402
from cnbr_fog.gru_convergence_models import GRUMeanForecaster  # noqa: E402
from cnbr_fog.gru_predictor_artifact import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    load_gru_predictor_artifact,
)
from cnbr_fog.resume import (  # noqa: E402
    atomic_json_dump,
    atomic_npz_save,
    atomic_torch_save,
    canonical_fingerprint,
    sha256_file,
)


EXPERIMENT_VERSION = "daphnet_s01_gru_direct_h128_extension.v1"
SOURCE_SUITE = "daphnet_s01_gru_convergence_sequence_v4"
HORIZON_SAMPLES = 128
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_MAX_STEPS = 2000
REFERENCE_MAX_STEPS = 500
EXPECTED_PATIENCE = 15
EXPECTED_MIN_STEPS = 32
EXPECTED_HIDDEN_CHANNELS = 48
EXPECTED_DROPOUT = 0.1
EXPECTED_BATCH_SIZE = 256
EXPECTED_LEARNING_RATE = 1e-3
EXPECTED_WEIGHT_DECAY = 1e-4
EXPECTED_AMP = True
EXPECTED_TRAIN_WINDOWS = 978
EXPECTED_VALIDATION_WINDOWS = 295
FLAT_SLOPE_THRESHOLD = 5e-4
MINIMUM_SKILL = 0.05


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extend the v4 H128 pure-mean direct GRU to true early stop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "dataset"
            / "1.Daphnet Freezing of Gait Dataset"
            / "processed"
        ),
    )
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / SOURCE_SUITE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "outputs" / "daphnet_s01_gru_direct_h128_extension_v1"
        ),
    )
    # These options remain visible for an auditable CLI, but validation below
    # rejects every scientific override.
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--max-steps", type=int, default=EXPECTED_MAX_STEPS)
    parser.add_argument("--patience", type=int, default=EXPECTED_PATIENCE)
    parser.add_argument("--min-steps", type=int, default=EXPECTED_MIN_STEPS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=EXPECTED_AMP
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _hash_paths(paths: Sequence[Path], root: Path) -> dict[str, str]:
    return {_relative(path, root): sha256_file(path) for path in paths}


def _assert_hashes_unchanged(
    paths: Sequence[Path], root: Path, expected: Mapping[str, str], label: str
) -> None:
    actual = _hash_paths(paths, root)
    if actual != dict(expected):
        differing = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise RuntimeError(f"{label} changed during the run: {differing}")


def validate_done(root: Path) -> dict[str, Any] | None:
    """Validate a completed extension against its exhaustive hash inventory."""

    done_path = root / "DONE.json"
    if not done_path.exists():
        return None
    done = _load_json(done_path)
    if done.get("status") != "complete":
        raise RuntimeError(f"Invalid DONE status: {done_path}")
    declared = dict(done.get("artifacts", {}))
    actual = {
        str(path.relative_to(root)).replace("\\", "/"): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "DONE.json"
    }
    if set(declared) != set(actual):
        raise RuntimeError("Extension artifact inventory mismatch")
    for relative, expected in declared.items():
        if sha256_file(actual[relative]) != expected:
            raise RuntimeError(f"Extension artifact hash mismatch: {relative}")
    return done


def _mean_training_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the v4 flat mean-stage hyperparameters for strict checks."""

    return {
        "hidden_channels": config.get("hidden_channels"),
        "dropout": config.get("dropout"),
        "batch_size": config.get("batch_size"),
        "learning_rate": config.get("learning_rate"),
        "weight_decay": config.get("weight_decay"),
        "maximum_optimizer_steps": config.get("maximum_optimizer_steps"),
        "minimum_optimizer_steps": config.get("minimum_optimizer_steps"),
        "patience": config.get("patience_evaluations"),
        "min_delta_rmse": config.get("min_delta_rmse"),
    }


def validate_locked_protocol(
    *,
    args: argparse.Namespace,
    seeds: tuple[int, ...],
    device: torch.device,
    upstream_config: Mapping[str, Any],
    upstream_done: Mapping[str, Any],
    upstream_runtime: Mapping[str, Any],
    long_mean_config: Mapping[str, Any],
    horizon_config: Mapping[str, Any],
) -> None:
    """Prove that optimizer budget is the sole scientific change."""

    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"Seeds are locked to {EXPECTED_SEEDS}, got {seeds}")
    if args.max_steps != EXPECTED_MAX_STEPS:
        raise ValueError(f"--max-steps is locked to {EXPECTED_MAX_STEPS}")
    if args.patience != EXPECTED_PATIENCE:
        raise ValueError(f"--patience is locked to {EXPECTED_PATIENCE}")
    if args.min_steps != EXPECTED_MIN_STEPS:
        raise ValueError(f"--min-steps is locked to {EXPECTED_MIN_STEPS}")
    if bool(args.amp) is not EXPECTED_AMP:
        raise ValueError("AMP is locked on")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This exact-reproduction extension is locked to CUDA")

    if upstream_config.get("experiment_version") != suite.EXPERIMENT_VERSION:
        raise RuntimeError("Unexpected upstream experiment version")
    if upstream_done.get("experiment_version") != suite.EXPERIMENT_VERSION:
        raise RuntimeError("Unexpected upstream DONE experiment version")
    if upstream_done.get("status") != "complete":
        raise RuntimeError("Upstream v4 suite is not complete")
    if upstream_done.get("protocol_fingerprint") != upstream_config.get(
        "protocol_fingerprint"
    ):
        raise RuntimeError("Upstream config/DONE protocol fingerprint mismatch")
    if tuple(upstream_config.get("seeds", ())) != EXPECTED_SEEDS:
        raise RuntimeError("Upstream seed set differs from the extension")
    if Path(str(upstream_config.get("data_dir"))).resolve() != args.data_dir.resolve():
        raise RuntimeError("Extension data directory differs from upstream v4")
    if upstream_config.get("records_loaded") != ["S01_seg000", "S01_seg001"]:
        raise RuntimeError("Upstream loaded-record boundary is unexpected")
    if upstream_config.get("device_type") != "cuda":
        raise RuntimeError("Upstream v4 was not a CUDA run")

    expected_root_training = {
        "hidden_channels": EXPECTED_HIDDEN_CHANNELS,
        "dropout": EXPECTED_DROPOUT,
        "batch_size": EXPECTED_BATCH_SIZE,
        "learning_rate": EXPECTED_LEARNING_RATE,
        "weight_decay": EXPECTED_WEIGHT_DECAY,
        "maximum_optimizer_steps": REFERENCE_MAX_STEPS,
        "patience": EXPECTED_PATIENCE,
        "minimum_optimizer_steps": EXPECTED_MIN_STEPS,
        "amp": EXPECTED_AMP,
    }
    upstream_hyperparameters = upstream_config.get("hyperparameters", {})
    mismatches = {
        key: (upstream_hyperparameters.get(key), value)
        for key, value in expected_root_training.items()
        if upstream_hyperparameters.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Upstream root hyperparameters differ: {mismatches}")

    expected_stage_training = {
        "hidden_channels": EXPECTED_HIDDEN_CHANNELS,
        "dropout": EXPECTED_DROPOUT,
        "batch_size": EXPECTED_BATCH_SIZE,
        "learning_rate": EXPECTED_LEARNING_RATE,
        "weight_decay": EXPECTED_WEIGHT_DECAY,
        "maximum_optimizer_steps": REFERENCE_MAX_STEPS,
        "minimum_optimizer_steps": EXPECTED_MIN_STEPS,
        "patience": EXPECTED_PATIENCE,
        "min_delta_rmse": suite.MIN_DELTA_RMSE,
    }
    if _mean_training_contract(long_mean_config) != expected_stage_training:
        raise RuntimeError("02_long_mean training contract is not the locked v4 one")
    horizon_training = horizon_config.get("training", {})
    if _mean_training_contract(horizon_training) != expected_stage_training:
        raise RuntimeError("04_horizon training contract is not the locked v4 one")
    if long_mean_config.get("model_name") != "current_grunbm_direct_mean_path":
        raise RuntimeError("Unexpected 02_long_mean model implementation")
    if horizon_training.get("model_name") != "pure_mean_common_gru_direct_decoder":
        raise RuntimeError("Unexpected 04_horizon model implementation")
    if int(long_mean_config.get("horizon_samples", -1)) != HORIZON_SAMPLES:
        raise RuntimeError("02_long_mean horizon is not H128")
    horizons = {
        str(item["id"]): int(item["samples"])
        for item in horizon_config.get("horizons", ())
    }
    if horizons.get("h200") != HORIZON_SAMPLES:
        raise RuntimeError("04_horizon has no canonical H128/h200 arm")

    runtime_expectations = {
        "device": "cuda",
        "cuda_device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    runtime_mismatch = {
        key: (upstream_runtime.get(key), value)
        for key, value in runtime_expectations.items()
        if upstream_runtime.get(key) != value
    }
    if runtime_mismatch:
        raise RuntimeError(
            f"Runtime differs from canonical v4; exact prefix is not assured: "
            f"{runtime_mismatch}"
        )


def audit_history_prefix(
    reference_history: Sequence[Mapping[str, Any]],
    candidate_history: Sequence[Mapping[str, Any]],
    *,
    maximum_steps: int = REFERENCE_MAX_STEPS,
) -> dict[str, Any]:
    """Compare every field of every history row through ``maximum_steps``."""

    reference = [
        dict(row)
        for row in reference_history
        if int(row["cumulative_optimizer_steps"]) <= maximum_steps
    ]
    if not reference:
        raise ValueError("Reference history is empty")
    if int(reference[-1]["cumulative_optimizer_steps"]) != maximum_steps:
        raise ValueError("Reference history does not end exactly at 500 steps")
    candidate = [dict(row) for row in candidate_history[: len(reference)]]

    first_mismatch: dict[str, Any] | None = None
    if len(candidate) != len(reference):
        first_mismatch = {
            "reason": "row_count",
            "expected": len(reference),
            "actual": len(candidate),
        }
    else:
        for row_index, (expected_row, actual_row) in enumerate(
            zip(reference, candidate, strict=True), start=1
        ):
            if set(expected_row) != set(actual_row):
                first_mismatch = {
                    "reason": "field_set",
                    "row": row_index,
                    "expected": sorted(expected_row),
                    "actual": sorted(actual_row),
                }
                break
            for field in expected_row:
                if expected_row[field] != actual_row[field]:
                    first_mismatch = {
                        "reason": "value",
                        "row": row_index,
                        "field": field,
                        "expected": expected_row[field],
                        "actual": actual_row[field],
                    }
                    break
            if first_mismatch is not None:
                break

    reference_sha256 = canonical_fingerprint(reference)
    candidate_sha256 = canonical_fingerprint(candidate)
    exact = bool(first_mismatch is None and reference_sha256 == candidate_sha256)
    return {
        "exact_row_for_row_match": exact,
        "maximum_optimizer_steps_compared": maximum_steps,
        "rows_compared": len(reference),
        "reference_history_canonical_sha256": reference_sha256,
        "candidate_history_canonical_sha256": candidate_sha256,
        "first_mismatch": first_mismatch,
    }


def _source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        SCRIPTS_DIR / "run_daphnet_s01_gru_convergence_sequence.py",
        SCRIPTS_DIR / "diagnose_daphnet_s01_gru_convergence.py",
        SCRIPTS_DIR / "run_daphnet_s01_gru_h200_tcnm.py",
        REPO_ROOT / "cnbr_fog" / "data.py",
        REPO_ROOT / "cnbr_fog" / "gru_convergence_models.py",
        REPO_ROOT / "cnbr_fog" / "gru_mode_analysis.py",
        REPO_ROOT / "cnbr_fog" / "gru_predictor_artifact.py",
        REPO_ROOT / "cnbr_fog" / "models.py",
        REPO_ROOT / "cnbr_fog" / "nbm.py",
        REPO_ROOT / "cnbr_fog" / "nbm_representations.py",
        REPO_ROOT / "cnbr_fog" / "resume.py",
    )


def _input_paths(data_dir: Path) -> tuple[Path, ...]:
    root = data_dir.resolve()
    return (
        root / "manifest.csv",
        root / "schema.json",
        root / "records" / "S01_seg000.npz",
        root / "records" / "S01_seg001.npz",
    )


def _upstream_paths(upstream: Path, seeds: Sequence[int]) -> tuple[Path, ...]:
    common = [
        upstream / "DONE.json",
        upstream / "config.json",
        upstream / "runtime.json",
        upstream / "02_long_mean" / "DONE.json",
        upstream / "02_long_mean" / "config.json",
        upstream / "02_long_mean" / "aggregate.json",
        upstream / "04_horizon" / "DONE.json",
        upstream / "04_horizon" / "config.json",
        upstream / "04_horizon" / "aggregate.json",
        upstream / "04_horizon" / "arms" / "h200" / "aggregate.json",
    ]
    for seed in seeds:
        run = upstream / "04_horizon" / "arms" / "h200" / "runs" / f"seed_{seed}"
        common.extend((run / "summary.json", run / "history.csv", run / "best.pt"))
    return tuple(common)


def _validate_upstream_source_and_inputs(
    upstream_config: Mapping[str, Any], source_paths: Sequence[Path], input_paths: Sequence[Path], data_dir: Path
) -> None:
    current_sources = _hash_paths(source_paths[1:], REPO_ROOT)
    upstream_sources = dict(upstream_config.get("source_sha256", {}))
    mismatches = {
        key: (upstream_sources.get(key), value)
        for key, value in current_sources.items()
        if upstream_sources.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Frozen v4 dependency source differs: {mismatches}")
    current_inputs = _hash_paths(input_paths, data_dir)
    if current_inputs != dict(upstream_config.get("loaded_input_sha256", {})):
        raise RuntimeError("Loaded input hashes differ from frozen v4")


def _validate_support(
    support: Mapping[str, Any], train_indices: np.ndarray, validation_indices: np.ndarray
) -> None:
    boundary = support.get("support", {})
    expected = {
        "clean_normal_train_windows": EXPECTED_TRAIN_WINDOWS,
        "clean_normal_validation_windows": EXPECTED_VALIDATION_WINDOWS,
        "records_in_diagnostic": ["S01_seg000", "S01_seg001"],
        "excluded_test_record": "S01_seg002",
        "test_array_file_opened": False,
        "test_loader_created": False,
        "test_windows_forwarded": 0,
        "test_predictions_computed": False,
    }
    mismatches = {
        key: (boundary.get(key), value)
        for key, value in expected.items()
        if boundary.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Unexpected train/validation support boundary: {mismatches}")
    if len(train_indices) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError("Expected exactly 978 clean-normal training windows")
    if len(validation_indices) != EXPECTED_VALIDATION_WINDOWS:
        raise RuntimeError("Expected exactly 295 clean-normal validation windows")


def _per_horizon_rows(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values = np.asarray(
        [item["best"]["validation"]["per_horizon_rmse_scaled"] for item in summaries],
        dtype=np.float64,
    )
    if values.shape != (len(EXPECTED_SEEDS), HORIZON_SAMPLES):
        raise RuntimeError(f"Unexpected per-horizon RMSE shape: {values.shape}")
    return [
        {
            "lead_sample": index + 1,
            "lead_seconds": (index + 1) / 64.0,
            "validation_rmse_mean": float(values[:, index].mean()),
            "validation_rmse_std": float(values[:, index].std()),
            "validation_rmse_min": float(values[:, index].min()),
            "validation_rmse_max": float(values[:, index].max()),
        }
        for index in range(HORIZON_SAMPLES)
    ]


def _write_report(root: Path, aggregate: Mapping[str, Any]) -> None:
    table = "\n".join(
        "| {seed} | {prefix} | {stop} | {steps} | {best_step} | {rmse:.6f} | {skill:.2%} | {slope:.7f} |".format(
            seed=row["seed"],
            prefix="是" if row["first_500_rows_exact"] else "否",
            stop=row["stop_reason"],
            steps=row["cumulative_optimizer_steps"],
            best_step=row["best_step"],
            rmse=row["best_validation_rmse"],
            skill=row["rmse_skill_vs_persistence"],
            slope=row["last_five_validation_rmse_slope_per_epoch"],
        )
        for row in aggregate["run_table"]
    )
    prefix = aggregate["prefix_reproduction"]
    change = aggregate["change_from_reference_500_steps"]
    report = f"""# S01 H128 direct mean-only 延长训练

## 锁定协议

权威前缀是 v4 `04_horizon/arms/h200`：`GRUMeanForecaster`、direct decoder、2 秒（128 点）未来均值。`02_long_mean` 使用 legacy `GRUNBM` 的均值路径，虽然数据、horizon 与训练超参数相同，但初始化和 decoder 实现不同，因此只作为已验证、已哈希的相关对照，不混入逐行复现。

本扩展只把最大训练预算从 500 提高到 2000 optimizer steps。其余均锁定：S01R01 的 `seg000 + seg001`、978/295 个 clean-normal 窗口、训练集 Robust Scaler、seed 42–46、hidden=48、dropout=0.1、batch=256、AdamW lr=1e-3/weight decay=1e-4、AMP/CUDA、验证 RMSE、patience=15、min-delta=1e-4。R02 数组未打开、未建 loader、未前向或评估。

## 结果

前 500 步逐字段、逐行严格复现：**{prefix['all_seeds_exact']}**（{prefix['exact_seed_count']}/5 seed，每个 125 行）。架构族严格收敛：**{aggregate['architecture_family_convergence_achieved']}**；固定 seed 42 checkpoint 收敛：**{aggregate['fixed_seed_checkpoint_convergence_achieved']}**；总体：**{aggregate['convergence_achieved']}**。

| seed | 前500步一致 | stop | 总步数 | 最佳步数 | 验证 RMSE | 相对持久性技能 | 末5次斜率 |
|---:|:---:|---|---:|---:|---:|---:|---:|
{table}

5-seed 最佳验证 RMSE：{aggregate['best_validation_rmse']['mean']:.6f} ± {aggregate['best_validation_rmse']['std']:.6f}；相对持久性技能：{aggregate['rmse_skill_vs_persistence']['mean']:.2%}。相对 v4 500 步 checkpoint，平均 RMSE 变化 {change['relative_rmse_reduction_mean']:.2%}，5 个 seed 中 {change['later_best_step_count']} 个最佳点晚于 500 步。

严格运行收敛定义为：因 validation patience 停止，且末 5 次验证 RMSE 对 epoch 的斜率绝对值小于 {FLAT_SLOPE_THRESHOLD:g}。架构族还要求至少 4/5 seed 满足该条件，且平均持久性技能不低于 5%。这是一条工程停止规则，不等价于数学参数收敛。

## 边界

窗口每 1 秒滑动且相互重叠，5 个 seed 也不是 5 份独立数据。本实验反复使用同一验证块，只检验 500 步预算是否右删失，不提供独立测试泛化结论。最终制品固定为预注册 seed 42，而不是按最优 seed 挑选；σ 是由该 checkpoint 的正常训练残差按 channel×horizon 解析校准。
"""
    suite._atomic_text(root / "report.md", report)


def main() -> None:
    args = parse_args()
    seeds = tuple(diagnostic.parse_int_list(args.seeds))
    device = diagnostic.resolve_device(args.device)
    upstream = args.upstream_dir.resolve()

    # Validate complete upstream stages before loading any scientific summary.
    suite._validate_completed_stage(upstream / "02_long_mean")
    suite._validate_completed_stage(upstream / "04_horizon")
    upstream_done = _load_json(upstream / "DONE.json")
    upstream_config = _load_json(upstream / "config.json")
    upstream_runtime = _load_json(upstream / "runtime.json")
    long_mean_config = _load_json(upstream / "02_long_mean" / "config.json")
    horizon_config = _load_json(upstream / "04_horizon" / "config.json")
    validate_locked_protocol(
        args=args,
        seeds=seeds,
        device=device,
        upstream_config=upstream_config,
        upstream_done=upstream_done,
        upstream_runtime=upstream_runtime,
        long_mean_config=long_mean_config,
        horizon_config=horizon_config,
    )

    source_paths = _source_paths()
    input_paths = _input_paths(args.data_dir)
    upstream_paths = _upstream_paths(upstream, seeds)
    _validate_upstream_source_and_inputs(
        upstream_config, source_paths, input_paths, args.data_dir
    )
    source_sha256 = _hash_paths(source_paths, REPO_ROOT)
    input_sha256 = _hash_paths(input_paths, args.data_dir)
    upstream_sha256 = _hash_paths(upstream_paths, upstream)

    (
        dataset,
        windows,
        train_indices,
        validation_indices,
        scaler,
        support_metadata,
    ) = diagnostic.prepare_support(args.data_dir)
    _validate_support(support_metadata, train_indices, validation_indices)
    if dataset.n_channels != 9:
        raise RuntimeError(f"Expected 9 channels, got {dataset.n_channels}")

    upstream_reference: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        path = (
            upstream
            / "04_horizon"
            / "arms"
            / "h200"
            / "runs"
            / f"seed_{seed}"
            / "summary.json"
        )
        summary = _load_json(path)
        model_config = summary.get("model_config", {})
        if (
            model_config.get("name") != "gru_mean"
            or model_config.get("horizon") != HORIZON_SAMPLES
            or model_config.get("decoder", {}).get("name") != "direct"
            or summary.get("cumulative_optimizer_steps") != REFERENCE_MAX_STEPS
        ):
            raise RuntimeError(f"Unexpected canonical h200 summary: {path}")
        upstream_reference[seed] = summary

    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cuda_device_name = torch.cuda.get_device_name(device)
    protocol = {
        "experiment_version": EXPERIMENT_VERSION,
        "purpose": "Test whether v4 H128 pure-mean direct was 500-step right-censored",
        "only_changed_factor": "maximum optimizer steps: 500 -> 2000",
        "canonical_reference": "04_horizon/arms/h200 GRUMeanForecaster direct",
        "related_nonidentical_reference": (
            "02_long_mean GRUNBM direct mean path; validated but not trajectory-matched"
        ),
        "data_dir": str(args.data_dir.resolve()),
        "support": support_metadata,
        "train_window_sha256": diagnostic.array_sha256(train_indices),
        "validation_window_sha256": diagnostic.array_sha256(validation_indices),
        "model": {
            "class": "GRUMeanForecaster",
            "decoder": "direct",
            "in_channels": dataset.n_channels,
            "context_samples": diagnostic.base.CONTEXT_SAMPLES,
            "horizon_samples": HORIZON_SAMPLES,
            "hidden_channels": EXPECTED_HIDDEN_CHANNELS,
            "num_layers": 1,
            "dropout": EXPECTED_DROPOUT,
        },
        "training": {
            "seeds": list(seeds),
            "batch_size": EXPECTED_BATCH_SIZE,
            "learning_rate": EXPECTED_LEARNING_RATE,
            "weight_decay": EXPECTED_WEIGHT_DECAY,
            "max_steps": EXPECTED_MAX_STEPS,
            "reference_max_steps": REFERENCE_MAX_STEPS,
            "min_steps": EXPECTED_MIN_STEPS,
            "patience": EXPECTED_PATIENCE,
            "min_delta_rmse": suite.MIN_DELTA_RMSE,
            "fixed_artifact_seed": int(seeds[0]),
            "amp": EXPECTED_AMP,
        },
        "prefix_reproduction_gate": {
            "reference": "04_horizon/arms/h200/runs/seed_<seed>/summary.json history",
            "comparison": "all fields of all 125 rows through exactly 500 steps",
            "required": "exact match for every seed before finalization",
        },
        "device": str(device),
        "device_type": device.type,
        "cuda_device_name": cuda_device_name,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "source_sha256": source_sha256,
        "loaded_input_sha256": input_sha256,
        "upstream_reference_sha256": upstream_sha256,
        "test_boundary": {
            "record": "S01_seg002",
            "array_opened": False,
            "loader_created": False,
            "evaluated": False,
        },
    }
    fingerprint = canonical_fingerprint(protocol)
    protocol["protocol_fingerprint"] = fingerprint
    config_path = root / "config.json"
    if config_path.exists():
        existing = _load_json(config_path)
        if existing.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError("Existing extension protocol differs")
    else:
        if any(root.iterdir()):
            raise FileExistsError(f"Non-empty extension output: {root}")
        atomic_json_dump(protocol, config_path)
        atomic_json_dump(scaler.as_dict(), root / "scaler.json")
        atomic_npz_save(
            root / "locked_support.npz",
            clean_normal_train_window_index=train_indices,
            clean_normal_validation_window_index=validation_indices,
        )
    if validate_done(root) is not None:
        print(f"Completed extension verified: {root}")
        return

    summaries: list[dict[str, Any]] = []
    prefix_runs: list[dict[str, Any]] = []
    for seed in seeds:
        summary = suite.train_mean_run(
            run_dir=root / "runs" / f"seed_{seed}",
            model_factory=lambda: GRUMeanForecaster(
                in_channels=dataset.n_channels,
                horizon=HORIZON_SAMPLES,
                hidden_channels=EXPECTED_HIDDEN_CHANNELS,
                num_layers=1,
                dropout=EXPECTED_DROPOUT,
                decoder="direct",
            ),
            seed=seed,
            dataset=dataset,
            windows=windows,
            train_indices=train_indices,
            validation_indices=validation_indices,
            scaler=scaler,
            horizon_samples=HORIZON_SAMPLES,
            batch_size=EXPECTED_BATCH_SIZE,
            learning_rate=EXPECTED_LEARNING_RATE,
            weight_decay=EXPECTED_WEIGHT_DECAY,
            dropout=EXPECTED_DROPOUT,
            max_steps=EXPECTED_MAX_STEPS,
            min_steps=EXPECTED_MIN_STEPS,
            patience=EXPECTED_PATIENCE,
            protocol_fingerprint=fingerprint,
            device=device,
            amp=EXPECTED_AMP,
        )
        audit = audit_history_prefix(
            upstream_reference[seed]["history"], summary["history"]
        )
        audit["seed"] = seed
        audit["upstream_summary_sha256"] = sha256_file(
            upstream
            / "04_horizon"
            / "arms"
            / "h200"
            / "runs"
            / f"seed_{seed}"
            / "summary.json"
        )
        prefix_runs.append(audit)
        atomic_json_dump(
            {
                "all_completed_seed_prefixes_exact": all(
                    item["exact_row_for_row_match"] for item in prefix_runs
                ),
                "runs": prefix_runs,
            },
            root / "prefix_reproduction.json",
        )
        if not audit["exact_row_for_row_match"]:
            raise RuntimeError(
                f"Seed {seed} failed exact first-500-step reproduction: "
                f"{audit['first_mismatch']}"
            )
        summaries.append(summary)

    required = math.ceil(0.8 * len(seeds))
    converged = [
        item["stop_reason"] == "validation_patience"
        and abs(item["last_five_validation_rmse_slope_per_epoch"])
        < FLAT_SLOPE_THRESHOLD
        for item in summaries
    ]
    skills = [item["rmse_skill_vs_persistence"] for item in summaries]
    family_converged = bool(sum(converged) >= required and np.mean(skills) >= MINIMUM_SKILL)
    fixed = summaries[0]
    fixed_converged = bool(converged[0] and fixed["rmse_skill_vs_persistence"] >= MINIMUM_SKILL)

    reference_rmse = [upstream_reference[seed]["best_validation_rmse"] for seed in seeds]
    extended_rmse = [item["best_validation_rmse"] for item in summaries]
    relative_reductions = [
        (reference - extended) / reference
        for reference, extended in zip(reference_rmse, extended_rmse, strict=True)
    ]
    run_table = [
        {
            "seed": item["seed"],
            "first_500_rows_exact": prefix_runs[index]["exact_row_for_row_match"],
            "stop_reason": item["stop_reason"],
            "cumulative_optimizer_steps": item["cumulative_optimizer_steps"],
            "best_step": item["best_step"],
            "best_validation_rmse": item["best_validation_rmse"],
            "reference_500_best_validation_rmse": reference_rmse[index],
            "relative_rmse_reduction_vs_500": relative_reductions[index],
            "rmse_skill_vs_persistence": item["rmse_skill_vs_persistence"],
            "last_five_validation_rmse_slope_per_epoch": item[
                "last_five_validation_rmse_slope_per_epoch"
            ],
        }
        for index, item in enumerate(summaries)
    ]
    prefix_reproduction = {
        "all_seeds_exact": all(item["exact_row_for_row_match"] for item in prefix_runs),
        "exact_seed_count": sum(item["exact_row_for_row_match"] for item in prefix_runs),
        "runs": prefix_runs,
    }
    aggregate = {
        "runs": len(summaries),
        "required_converged_count": required,
        "converged_count": int(sum(converged)),
        "architecture_family_convergence_achieved": family_converged,
        "fixed_seed": fixed["seed"],
        "fixed_seed_checkpoint_convergence_achieved": fixed_converged,
        "convergence_achieved": bool(family_converged and fixed_converged),
        "best_validation_rmse": suite.numeric_stats(extended_rmse),
        "rmse_skill_vs_persistence": suite.numeric_stats(skills),
        "best_step": suite.numeric_stats([item["best_step"] for item in summaries]),
        "change_from_reference_500_steps": {
            "reference_best_validation_rmse": suite.numeric_stats(reference_rmse),
            "relative_rmse_reduction": relative_reductions,
            "relative_rmse_reduction_mean": float(np.mean(relative_reductions)),
            "later_best_step_count": int(
                sum(item["best_step"] > REFERENCE_MAX_STEPS for item in summaries)
            ),
        },
        "prefix_reproduction": prefix_reproduction,
        "run_table": run_table,
        "test_record_evaluated": False,
    }
    atomic_json_dump(prefix_reproduction, root / "prefix_reproduction.json")
    atomic_json_dump(aggregate, root / "aggregate.json")
    diagnostic.write_csv(root / "run_table.csv", run_table)
    per_horizon_rows = _per_horizon_rows(summaries)
    diagnostic.write_csv(root / "per_horizon_rmse.csv", per_horizon_rows)

    checkpoint_path = root / "runs" / f"seed_{seeds[0]}" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sigma = np.asarray(fixed["fixed_sigma_calibration"]["sigma"], dtype=np.float32)
    constructor = suite._artifact_constructor_spec(checkpoint["model_config"])
    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "protocol_fingerprint": fingerprint,
        "selection_scope": "validation-only extension; R02 not evaluated",
        "model_config": checkpoint["model_config"],
        **constructor,
        "model_state": checkpoint["model_state"],
        "fixed_sigma": sigma,
        "fixed_sigma_sha256": diagnostic.array_sha256(sigma),
        "robust_scaler": scaler.as_dict(),
        "fixed_seed_converged_by_operational_rule": fixed_converged,
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "test_record_evaluated": False,
    }
    final_path = root / "final_predictor.pt"
    atomic_torch_save(artifact, final_path)
    predictor = load_gru_predictor_artifact(final_path, map_location="cpu")
    predictor.eval()
    with torch.no_grad():
        mean, loaded_sigma = predictor(torch.zeros(2, dataset.n_channels, 128))
    expected_shape = (2, dataset.n_channels, HORIZON_SAMPLES)
    if tuple(mean.shape) != expected_shape or tuple(loaded_sigma.shape) != expected_shape:
        raise RuntimeError("Final predictor inference contract failed")
    if not torch.isfinite(mean).all() or not torch.isfinite(loaded_sigma).all():
        raise RuntimeError("Final predictor produced non-finite output")
    if not torch.all(loaded_sigma > 0):
        raise RuntimeError("Final predictor sigma is not strictly positive")
    atomic_json_dump(
        {
            "loader": "load_gru_predictor_artifact",
            "input_shape": [2, dataset.n_channels, 128],
            "mean_shape": list(mean.shape),
            "sigma_shape": list(loaded_sigma.shape),
            "finite": True,
            "sigma_strictly_positive": True,
        },
        root / "artifact_validation.json",
    )

    _write_report(root, aggregate)
    atomic_json_dump(
        {
            "created_utc": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_type": device.type,
            "cuda_device_name": cuda_device_name,
            "protocol_fingerprint": fingerprint,
        },
        root / "runtime.json",
    )

    # If source, loaded data, or the canonical v4 evidence changed while CUDA
    # training was running, do not publish a DONE marker.
    _assert_hashes_unchanged(source_paths, REPO_ROOT, source_sha256, "Source")
    _assert_hashes_unchanged(input_paths, args.data_dir, input_sha256, "Loaded input")
    _assert_hashes_unchanged(upstream_paths, upstream, upstream_sha256, "Upstream evidence")

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
            "first_500_steps_exact_all_seeds": prefix_reproduction["all_seeds_exact"],
            "architecture_family_convergence_achieved": family_converged,
            "fixed_seed_checkpoint_convergence_achieved": fixed_converged,
            "convergence_achieved": bool(family_converged and fixed_converged),
            "test_record_evaluated": False,
            "artifacts": artifacts,
        },
        root / "DONE.json",
    )
    validate_done(root)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"Results: {root}")


if __name__ == "__main__":
    main()
