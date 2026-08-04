#!/usr/bin/env python
"""Targeted long-budget confirmation for the S01 H=16 shared GRU mean model."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from cnbr_fog.gru_predictor_artifact import ARTIFACT_SCHEMA_VERSION  # noqa: E402
from cnbr_fog.nbm import parameter_count  # noqa: E402
from cnbr_fog.resume import (  # noqa: E402
    atomic_json_dump,
    atomic_npz_save,
    atomic_torch_save,
    canonical_fingerprint,
    sha256_file,
)


EXPERIMENT_VERSION = "daphnet_s01_gru_shared_h016_extension.v1"
HORIZON_SAMPLES = 16
SOURCE_SUITE = "daphnet_s01_gru_convergence_sequence_v4"
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_MAX_STEPS = 2000
EXPECTED_PATIENCE = 15
EXPECTED_MIN_STEPS = 32
EXPECTED_AMP = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extend the validation-selected H016 shared GRU mean model",
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
            REPO_ROOT / "outputs" / "daphnet_s01_gru_shared_h016_extension_v1"
        ),
    )
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-steps", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_done(root: Path) -> dict[str, Any] | None:
    done_path = root / "DONE.json"
    if not done_path.exists():
        return None
    done = json.loads(done_path.read_text(encoding="utf-8"))
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


def write_report(root: Path, aggregate: dict[str, Any]) -> None:
    rows = aggregate["run_table"]
    table = "\n".join(
        "| {seed} | {stop} | {steps} | {best_step} | {rmse:.6f} | {skill:.2%} | {slope:.7f} |".format(
            seed=row["seed"],
            stop=row["stop_reason"],
            steps=row["cumulative_optimizer_steps"],
            best_step=row["best_step"],
            rmse=row["best_validation_rmse"],
            skill=row["rmse_skill_vs_persistence"],
            slope=row["last_five_validation_rmse_slope_per_epoch"],
        )
        for row in rows
    )
    report = f"""# S01 H016 shared-horizon 延长训练

只改变训练上限：由 500 optimizer steps 延长到 2000；模型、978/295 clean-normal 支持、Robust Scaler、5 个 seed、学习率、weight decay、patience=15 与验证 RMSE 规则均保持不变。R02 未载入或评估。

架构族严格收敛：**{aggregate['architecture_family_convergence_achieved']}**；固定 seed 42 checkpoint 收敛：**{aggregate['fixed_seed_checkpoint_convergence_achieved']}**；总体：**{aggregate['convergence_achieved']}**。

| seed | stop | 总步数 | 最佳步数 | 验证 RMSE | 相对持久性技能 | 末5次斜率 |
|---:|---|---:|---:|---:|---:|---:|
{table}

`validation_patience` 只表示在工程阈值下停止产生足够大的新低，不是数学参数收敛。多个 seed 不是独立数据；窗口以 1 秒步长重叠。本扩展仍反复使用同一验证块，只用于确认 500 步是否右删失，不提供测试泛化结论。
"""
    suite._atomic_text(root / "report.md", report)


def validate_locked_protocol(
    args: argparse.Namespace,
    seeds: tuple[int, ...],
    device: torch.device,
    upstream_config: dict[str, Any],
    upstream_done: dict[str, Any],
) -> None:
    """Ensure that optimizer budget is the extension's sole scientific change."""

    expected_training = {
        "hidden_channels": 48,
        "dropout": 0.1,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "maximum_optimizer_steps": 500,
        "patience": EXPECTED_PATIENCE,
        "minimum_optimizer_steps": EXPECTED_MIN_STEPS,
        "amp": EXPECTED_AMP,
    }
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"Seeds are locked to {EXPECTED_SEEDS}, got {seeds}")
    if args.max_steps != EXPECTED_MAX_STEPS:
        raise ValueError(f"--max-steps is locked to {EXPECTED_MAX_STEPS}")
    if args.patience != EXPECTED_PATIENCE:
        raise ValueError(f"--patience is locked to {EXPECTED_PATIENCE}")
    if args.min_steps != EXPECTED_MIN_STEPS:
        raise ValueError(f"--min-steps is locked to {EXPECTED_MIN_STEPS}")
    if bool(args.amp) is not EXPECTED_AMP:
        raise ValueError("AMP is locked on to match the upstream v4 suite")
    if upstream_config.get("experiment_version") != suite.EXPERIMENT_VERSION:
        raise RuntimeError("Unexpected upstream experiment version")
    if upstream_done.get("experiment_version") != suite.EXPERIMENT_VERSION:
        raise RuntimeError("Unexpected upstream DONE experiment version")
    if upstream_done.get("protocol_fingerprint") != upstream_config.get(
        "protocol_fingerprint"
    ):
        raise RuntimeError("Upstream config/DONE protocol fingerprint mismatch")
    if tuple(upstream_config.get("seeds", ())) != EXPECTED_SEEDS:
        raise RuntimeError("Upstream v4 seed set does not match the locked extension")
    upstream_hyperparameters = upstream_config.get("hyperparameters", {})
    mismatches = {
        key: (upstream_hyperparameters.get(key), expected)
        for key, expected in expected_training.items()
        if upstream_hyperparameters.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Upstream v4 hyperparameters differ: {mismatches}")
    if upstream_config.get("device_type") != device.type:
        raise RuntimeError(
            "Extension device type must match upstream v4: "
            f"{upstream_config.get('device_type')} != {device.type}"
        )
    if Path(upstream_config["data_dir"]).resolve() != args.data_dir.resolve():
        raise RuntimeError("Extension data directory must match upstream v4")


def main() -> None:
    args = parse_args()
    seeds = tuple(diagnostic.parse_int_list(args.seeds))
    device = diagnostic.resolve_device(args.device)
    upstream = args.upstream_dir.resolve()
    upstream_done = json.loads(
        (upstream / "DONE.json").read_text(encoding="utf-8")
    )
    if upstream_done.get("status") != "complete":
        raise RuntimeError("Upstream v4 suite is not complete")
    upstream_config = json.loads(
        (upstream / "config.json").read_text(encoding="utf-8")
    )
    validate_locked_protocol(
        args, seeds, device, upstream_config, upstream_done
    )
    suite._validate_completed_stage(upstream / "04_horizon")
    suite._validate_completed_stage(upstream / "05_decoder")
    horizon_summary = suite._load_stage_summary(upstream, "04_horizon")
    decoder_summary = suite._load_stage_summary(upstream, "05_decoder")
    if int(horizon_summary["selected_horizon_samples"]) != HORIZON_SAMPLES:
        raise RuntimeError("Upstream did not select H016")
    if decoder_summary["selected_decoder"] != "shared_horizon":
        raise RuntimeError("Upstream did not select shared_horizon")

    (
        dataset,
        windows,
        train_indices,
        validation_indices,
        scaler,
        support_metadata,
    ) = diagnostic.prepare_support(args.data_dir)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_paths = (
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
    input_paths = (
        args.data_dir.resolve() / "manifest.csv",
        args.data_dir.resolve() / "schema.json",
        args.data_dir.resolve() / "records" / "S01_seg000.npz",
        args.data_dir.resolve() / "records" / "S01_seg001.npz",
    )
    cuda_device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else None
    )
    protocol = {
        "experiment_version": EXPERIMENT_VERSION,
        "purpose": "Confirm whether the 500-step H016 shared run was right-censored",
        "only_changed_factor": "maximum optimizer steps: 500 -> 2000",
        "upstream_suite_done_sha256": sha256_file(upstream / "DONE.json"),
        "upstream_suite_config_sha256": sha256_file(upstream / "config.json"),
        "upstream_decoder_done_sha256": sha256_file(
            upstream / "05_decoder" / "DONE.json"
        ),
        "data_dir": str(args.data_dir.resolve()),
        "support": support_metadata,
        "train_window_sha256": diagnostic.array_sha256(train_indices),
        "validation_window_sha256": diagnostic.array_sha256(validation_indices),
        "model": {
            "class": "GRUMeanForecaster",
            "decoder": "shared_horizon",
            "in_channels": dataset.n_channels,
            "context_samples": diagnostic.base.CONTEXT_SAMPLES,
            "horizon_samples": HORIZON_SAMPLES,
            "hidden_channels": 48,
            "dropout": 0.1,
        },
        "training": {
            "seeds": list(seeds),
            "batch_size": 256,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "max_steps": args.max_steps,
            "min_steps": args.min_steps,
            "patience": args.patience,
            "min_delta_rmse": suite.MIN_DELTA_RMSE,
            "fixed_artifact_seed": int(seeds[0]),
            "amp": EXPECTED_AMP,
        },
        "device": str(device),
        "device_type": device.type,
        "cuda_device_name": cuda_device_name,
        "torch_cuda": torch.version.cuda,
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in source_paths
        },
        "loaded_input_sha256": {
            str(path.relative_to(args.data_dir.resolve())).replace("\\", "/"): sha256_file(path)
            for path in input_paths
        },
        "test_record_evaluated": False,
    }
    fingerprint = canonical_fingerprint(protocol)
    protocol["protocol_fingerprint"] = fingerprint
    config_path = root / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
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
    for seed in seeds:
        summaries.append(
            suite.train_mean_run(
                run_dir=root / "runs" / f"seed_{seed}",
                model_factory=lambda: GRUMeanForecaster(
                    in_channels=dataset.n_channels,
                    horizon=HORIZON_SAMPLES,
                    hidden_channels=48,
                    num_layers=1,
                    dropout=0.1,
                    decoder="shared_horizon",
                ),
                seed=seed,
                dataset=dataset,
                windows=windows,
                train_indices=train_indices,
                validation_indices=validation_indices,
                scaler=scaler,
                horizon_samples=HORIZON_SAMPLES,
                batch_size=256,
                learning_rate=1e-3,
                weight_decay=1e-4,
                dropout=0.1,
                max_steps=args.max_steps,
                min_steps=args.min_steps,
                patience=args.patience,
                protocol_fingerprint=fingerprint,
                device=device,
                amp=EXPECTED_AMP,
            )
        )
    required = math.ceil(0.8 * len(seeds))
    converged = [
        item["stop_reason"] == "validation_patience"
        and abs(item["last_five_validation_rmse_slope_per_epoch"]) < 5e-4
        for item in summaries
    ]
    run_table = [
        {
            "seed": item["seed"],
            "stop_reason": item["stop_reason"],
            "cumulative_optimizer_steps": item["cumulative_optimizer_steps"],
            "best_step": item["best_step"],
            "best_validation_rmse": item["best_validation_rmse"],
            "rmse_skill_vs_persistence": item["rmse_skill_vs_persistence"],
            "last_five_validation_rmse_slope_per_epoch": item[
                "last_five_validation_rmse_slope_per_epoch"
            ],
        }
        for item in summaries
    ]
    family_converged = bool(
        sum(converged) >= required
        and np.mean([item["rmse_skill_vs_persistence"] for item in summaries])
        >= 0.05
    )
    fixed = summaries[0]
    fixed_converged = bool(converged[0] and fixed["rmse_skill_vs_persistence"] >= 0.05)
    aggregate = {
        "runs": len(summaries),
        "required_converged_count": required,
        "converged_count": int(sum(converged)),
        "architecture_family_convergence_achieved": family_converged,
        "fixed_seed": fixed["seed"],
        "fixed_seed_checkpoint_convergence_achieved": fixed_converged,
        "convergence_achieved": bool(family_converged and fixed_converged),
        "best_validation_rmse": suite.numeric_stats(
            [item["best_validation_rmse"] for item in summaries]
        ),
        "rmse_skill_vs_persistence": suite.numeric_stats(
            [item["rmse_skill_vs_persistence"] for item in summaries]
        ),
        "best_step": suite.numeric_stats([item["best_step"] for item in summaries]),
        "run_table": run_table,
        "test_record_evaluated": False,
    }
    atomic_json_dump(aggregate, root / "aggregate.json")
    diagnostic.write_csv(root / "run_table.csv", run_table)

    checkpoint_path = root / "runs" / f"seed_{seeds[0]}" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sigma = np.asarray(
        fixed["fixed_sigma_calibration"]["sigma"], dtype=np.float32
    )
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
    atomic_torch_save(artifact, root / "final_predictor.pt")
    write_report(root, aggregate)
    atomic_json_dump(
        {
            "created_utc": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_type": device.type,
            "cuda_device_name": cuda_device_name,
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
            "architecture_family_convergence_achieved": family_converged,
            "fixed_seed_checkpoint_convergence_achieved": fixed_converged,
            "convergence_achieved": bool(family_converged and fixed_converged),
            "test_record_evaluated": False,
            "artifacts": artifacts,
        },
        root / "DONE.json",
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"Results: {root}")


if __name__ == "__main__":
    main()
