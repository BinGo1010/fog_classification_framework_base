#!/usr/bin/env python
"""Paired S01 global versus frozen-KMeans hard-conditioning experiment."""

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
from cnbr_fog.gru_hard_cluster_model import (  # noqa: E402
    HardClusterConditionedGRUMeanForecaster,
)
from cnbr_fog.nbm import parameter_count  # noqa: E402
from cnbr_fog.resume import (  # noqa: E402
    atomic_json_dump,
    atomic_npz_save,
    canonical_fingerprint,
    sha256_file,
)


EXPERIMENT_VERSION = "daphnet_s01_gru_hard_cluster_conditioning.v1"
UPSTREAM_DIR_NAME = "daphnet_s01_gru_convergence_sequence_v4"
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
HORIZON_SAMPLES = 16
N_CLUSTERS = 3
MAX_STEPS = 500
MIN_STEPS = 32
PATIENCE = 15
AMP = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test explicit frozen KMeans-cluster conditioning for S01",
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
        default=REPO_ROOT / "outputs" / UPSTREAM_DIR_NAME,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / "daphnet_s01_gru_hard_cluster_conditioning_v1"
        ),
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_output(root: Path, fingerprint: str) -> bool:
    done_path = root / "DONE.json"
    if not done_path.exists():
        return False
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("status") != "complete":
        raise RuntimeError("Hard-conditioning output is incomplete")
    if done.get("protocol_fingerprint") != fingerprint:
        raise RuntimeError("Hard-conditioning output protocol mismatch")
    declared = dict(done.get("artifacts", {}))
    actual = {
        str(path.relative_to(root)).replace("\\", "/"): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "DONE.json"
    }
    if set(declared) != set(actual):
        raise RuntimeError("Hard-conditioning artifact inventory mismatch")
    for relative, expected in declared.items():
        if sha256_file(actual[relative]) != expected:
            raise RuntimeError(f"Artifact hash mismatch: {relative}")
    return True


def common_state_hash(model: torch.nn.Module) -> str:
    state = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if not name.startswith("cluster_embedding.")
    }
    return suite.state_sha256(state)


def per_cluster_stats(
    summaries: list[dict[str, Any]], n_clusters: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cluster in range(n_clusters):
        values: list[float] = []
        windows: list[int] = []
        for item in summaries:
            metric = item["best"]["validation"]["per_mode"][str(cluster)]
            values.append(float(metric["rmse_scaled"]))
            windows.append(int(metric["windows"]))
        if len(set(windows)) != 1:
            raise RuntimeError("Validation cluster support changed across seeds")
        result[str(cluster)] = {
            "validation_windows": windows[0],
            "rmse": suite.numeric_stats(values),
        }
    return result


def main() -> None:
    args = parse_args()
    device = diagnostic.resolve_device(args.device)
    upstream = args.upstream_dir.resolve()
    stage = upstream / "06_modes"
    upstream_done = json.loads(
        (upstream / "DONE.json").read_text(encoding="utf-8")
    )
    upstream_config = json.loads(
        (upstream / "config.json").read_text(encoding="utf-8")
    )
    stage_done = suite._validate_completed_stage(stage)
    stage_config = json.loads((stage / "config.json").read_text(encoding="utf-8"))
    stage_aggregate = json.loads(
        (stage / "aggregate.json").read_text(encoding="utf-8")
    )
    if upstream_done.get("status") != "complete":
        raise RuntimeError("Upstream v4 suite is incomplete")
    if upstream_done.get("experiment_version") != suite.EXPERIMENT_VERSION:
        raise RuntimeError("Unexpected upstream suite version")
    if upstream_done.get("protocol_fingerprint") != upstream_config.get(
        "protocol_fingerprint"
    ):
        raise RuntimeError("Upstream config/DONE fingerprint mismatch")
    if stage_done.get("protocol_fingerprint") != stage_config.get(
        "protocol_fingerprint"
    ):
        raise RuntimeError("Mode-stage config/DONE fingerprint mismatch")
    if upstream_config.get("device_type") != device.type:
        raise RuntimeError("Device type must match the upstream v4 suite")
    if Path(upstream_config["data_dir"]).resolve() != args.data_dir.resolve():
        raise RuntimeError("Data directory must match the upstream v4 suite")
    if int(stage_aggregate["selected_k"]) != N_CLUSTERS:
        raise RuntimeError("Upstream mode stage did not select three clusters")
    if int(stage_aggregate["selected_horizon_samples"]) != HORIZON_SAMPLES:
        raise RuntimeError("Unexpected upstream mode horizon")
    expected_root_hyperparameters = {
        "hidden_channels": 48,
        "dropout": 0.1,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "maximum_optimizer_steps": MAX_STEPS,
        "patience": PATIENCE,
        "minimum_optimizer_steps": MIN_STEPS,
        "amp": AMP,
    }
    upstream_root_hyperparameters = upstream_config.get("hyperparameters", {})
    root_hyperparameter_mismatches = {
        key: (upstream_root_hyperparameters.get(key), expected)
        for key, expected in expected_root_hyperparameters.items()
        if upstream_root_hyperparameters.get(key) != expected
    }
    if root_hyperparameter_mismatches:
        raise RuntimeError(
            "Upstream suite hyperparameters differ: "
            f"{root_hyperparameter_mismatches}"
        )
    if tuple(upstream_config.get("seeds", ())) != EXPECTED_SEEDS:
        raise RuntimeError("Upstream suite seed set differs")
    for relative, expected_hash in upstream_config.get(
        "source_sha256", {}
    ).items():
        source_path = REPO_ROOT / relative
        if sha256_file(source_path) != expected_hash:
            raise RuntimeError(f"Upstream source has drifted: {relative}")
    expected_training = {
        "seeds": list(EXPECTED_SEEDS),
        "hidden_channels": 48,
        "dropout": 0.1,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "maximum_optimizer_steps": MAX_STEPS,
        "minimum_optimizer_steps": MIN_STEPS,
        "patience_evaluations": PATIENCE,
    }
    stage_training = stage_config.get("training", {})
    training_mismatches = {
        key: (stage_training.get(key), expected)
        for key, expected in expected_training.items()
        if stage_training.get(key) != expected
    }
    if training_mismatches:
        raise RuntimeError(
            f"Upstream global training protocol differs: {training_mismatches}"
        )

    (
        dataset,
        windows,
        train_indices,
        validation_indices,
        scaler,
        support_metadata,
    ) = diagnostic.prepare_support(args.data_dir)
    frozen_path = stage / "frozen_mode_model.npz"
    with np.load(frozen_path, allow_pickle=False) as payload:
        frozen = {name: np.asarray(payload[name]) for name in payload.files}
    if not np.array_equal(frozen["train_window_indices"], train_indices):
        raise RuntimeError("Frozen train-mode support differs from current support")
    if not np.array_equal(
        frozen["validation_window_indices"], validation_indices
    ):
        raise RuntimeError("Frozen validation-mode support differs from current support")
    train_assignments = frozen["train_assignments"].astype(np.int64, copy=False)
    validation_assignments = frozen["validation_assignments"].astype(
        np.int64, copy=False
    )
    if set(np.unique(train_assignments)) != set(range(N_CLUSTERS)):
        raise RuntimeError("Frozen training assignments omit a selected cluster")
    if set(np.unique(validation_assignments)) != set(range(N_CLUSTERS)):
        raise RuntimeError("Frozen validation assignments omit a selected cluster")
    mode_by_window = np.full(len(windows), -1, dtype=np.int64)
    mode_by_window[train_indices] = train_assignments
    mode_by_window[validation_indices] = validation_assignments

    baseline_paths = [
        stage
        / "arms"
        / "global_direct"
        / "runs"
        / f"seed_{seed}"
        / "summary.json"
        for seed in EXPECTED_SEEDS
    ]
    baseline = [
        json.loads(path.read_text(encoding="utf-8")) for path in baseline_paths
    ]
    if tuple(int(item["seed"]) for item in baseline) != EXPECTED_SEEDS:
        raise RuntimeError("Baseline seed order differs")
    if any(int(item["horizon_samples"]) != HORIZON_SAMPLES for item in baseline):
        raise RuntimeError("Baseline horizon differs")
    for item in baseline:
        model_config = item.get("model_config", {})
        if model_config.get("decoder", {}).get("name") != "direct":
            raise RuntimeError("Baseline is not the locked direct decoder")
        encoder_config = model_config.get("encoder", {})
        if encoder_config.get("hidden_channels") != 48:
            raise RuntimeError("Baseline hidden width differs")
        if encoder_config.get("dropout") != 0.1:
            raise RuntimeError("Baseline encoder dropout differs")

    upstream_initial_path = stage / "initial_encoder_hashes.json"
    upstream_initial_hashes = json.loads(
        upstream_initial_path.read_text(encoding="utf-8")
    )

    source_paths = (
        Path(__file__).resolve(),
        SCRIPTS_DIR / "run_daphnet_s01_gru_convergence_sequence.py",
        SCRIPTS_DIR / "diagnose_daphnet_s01_gru_convergence.py",
        SCRIPTS_DIR / "run_daphnet_s01_gru_h200_tcnm.py",
        REPO_ROOT / "cnbr_fog" / "data.py",
        REPO_ROOT / "cnbr_fog" / "gru_convergence_models.py",
        REPO_ROOT / "cnbr_fog" / "gru_hard_cluster_model.py",
        REPO_ROOT / "cnbr_fog" / "gru_mode_analysis.py",
        REPO_ROOT / "cnbr_fog" / "gru_predictor_artifact.py",
        REPO_ROOT / "cnbr_fog" / "models.py",
        REPO_ROOT / "cnbr_fog" / "nbm.py",
        REPO_ROOT / "cnbr_fog" / "nbm_representations.py",
        REPO_ROOT / "cnbr_fog" / "resume.py",
    )
    data_dir = args.data_dir.resolve()
    input_paths = (
        upstream / "DONE.json",
        upstream / "config.json",
        stage / "DONE.json",
        stage / "config.json",
        stage / "aggregate.json",
        upstream_initial_path,
        frozen_path,
        *baseline_paths,
        data_dir / "manifest.csv",
        data_dir / "schema.json",
        data_dir / "records" / "S01_seg000.npz",
        data_dir / "records" / "S01_seg001.npz",
    )
    cuda_device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else None
    )
    protocol = {
        "experiment_version": EXPERIMENT_VERSION,
        "purpose": "Directly test frozen train-only KMeans cluster labels as predictor conditioning",
        "upstream_suite_protocol_fingerprint": upstream_config[
            "protocol_fingerprint"
        ],
        "upstream_mode_protocol_fingerprint": stage_config[
            "protocol_fingerprint"
        ],
        "data_dir": str(data_dir),
        "support": support_metadata,
        "train_window_sha256": diagnostic.array_sha256(train_indices),
        "validation_window_sha256": diagnostic.array_sha256(validation_indices),
        "mode_assignment": {
            "source": "v4 train-only StandardScaler/PCA/KMeans frozen model",
            "target_label_or_residual_used": False,
            "n_clusters": N_CLUSTERS,
            "train_assignment_sha256": diagnostic.array_sha256(
                train_assignments
            ),
            "validation_assignment_sha256": diagnostic.array_sha256(
                validation_assignments
            ),
            "passed_to_predictor": True,
        },
        "arms": {
            "reference": "upstream v4 global_direct",
            "candidate": "hard_cluster_embedding",
            "common_initial_prediction": True,
            "candidate_extra_parameters": N_CLUSTERS * 48,
        },
        "model": {
            "class": "HardClusterConditionedGRUMeanForecaster",
            "in_channels": dataset.n_channels,
            "context_samples": diagnostic.base.CONTEXT_SAMPLES,
            "horizon_samples": HORIZON_SAMPLES,
            "hidden_channels": 48,
            "dropout": 0.1,
            "decoder": "direct",
        },
        "training": {
            "seeds": list(EXPECTED_SEEDS),
            "batch_size": 256,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "max_steps": MAX_STEPS,
            "min_steps": MIN_STEPS,
            "patience": PATIENCE,
            "min_delta_rmse": suite.MIN_DELTA_RMSE,
            "amp": AMP,
        },
        "benefit_rule": "paired RMSE gain >=1% on average and wins >=4/5 seeds",
        "device": str(device),
        "device_type": device.type,
        "cuda_device_name": cuda_device_name,
        "torch_cuda": torch.version.cuda,
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in source_paths
        },
        "input_sha256": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in input_paths
        },
        "test_record_evaluated": False,
    }
    fingerprint = canonical_fingerprint(protocol)
    protocol["protocol_fingerprint"] = fingerprint
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError("Existing hard-conditioning protocol differs")
    else:
        if any(root.iterdir()):
            raise FileExistsError(f"Non-empty output directory: {root}")
        atomic_json_dump(protocol, config_path)
        atomic_json_dump(scaler.as_dict(), root / "scaler.json")
        atomic_npz_save(
            root / "locked_support_and_modes.npz",
            train_window_indices=train_indices,
            validation_window_indices=validation_indices,
            train_assignments=train_assignments,
            validation_assignments=validation_assignments,
        )
    if validate_output(root, fingerprint):
        print(f"Completed hard-conditioning experiment verified: {root}")
        return

    initial_hashes: dict[str, dict[str, str]] = {}
    for seed in EXPECTED_SEEDS:
        diagnostic.set_seed(seed, True)
        global_probe = GRUMeanForecaster(
            in_channels=dataset.n_channels,
            horizon=HORIZON_SAMPLES,
            hidden_channels=48,
            num_layers=1,
            dropout=0.1,
            decoder="direct",
        )
        diagnostic.set_seed(seed, True)
        hard_probe = HardClusterConditionedGRUMeanForecaster(
            in_channels=dataset.n_channels,
            horizon=HORIZON_SAMPLES,
            n_clusters=N_CLUSTERS,
            hidden_channels=48,
            num_layers=1,
            dropout=0.1,
        )
        global_probe.eval()
        hard_probe.eval()
        global_hash = common_state_hash(global_probe)
        hard_hash = common_state_hash(hard_probe)
        if global_hash != hard_hash:
            raise AssertionError("Global/hard common initial weights differ")
        context = torch.zeros(2, dataset.n_channels, diagnostic.base.CONTEXT_SAMPLES)
        with torch.no_grad():
            global_mean = global_probe.forward_mean(context)
            hard_mean = hard_probe.forward_mean(
                context, torch.tensor([0, N_CLUSTERS - 1], dtype=torch.long)
            )
        if not torch.equal(global_mean, hard_mean):
            raise AssertionError("Zero cluster embedding changed initial prediction")
        global_encoder_hash = suite._encoder_sha256(global_probe)
        hard_encoder_hash = suite._encoder_sha256(hard_probe)
        expected_encoder_hash = upstream_initial_hashes[str(seed)][
            "global_direct"
        ]
        if global_encoder_hash != expected_encoder_hash:
            raise AssertionError("Current global encoder does not match v4 baseline")
        if hard_encoder_hash != expected_encoder_hash:
            raise AssertionError("Hard encoder does not match v4 baseline")
        initial_hashes[str(seed)] = {
            "global_common_state_sha256": global_hash,
            "hard_common_state_sha256": hard_hash,
            "upstream_global_encoder_sha256": expected_encoder_hash,
            "current_global_encoder_sha256": global_encoder_hash,
            "hard_encoder_sha256": hard_encoder_hash,
        }
    atomic_json_dump(initial_hashes, root / "initial_common_state_hashes.json")

    summaries: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        summaries.append(
            suite.train_mean_run(
                run_dir=root / "runs" / f"seed_{seed}",
                model_factory=lambda: HardClusterConditionedGRUMeanForecaster(
                    in_channels=dataset.n_channels,
                    horizon=HORIZON_SAMPLES,
                    n_clusters=N_CLUSTERS,
                    hidden_channels=48,
                    num_layers=1,
                    dropout=0.1,
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
                max_steps=MAX_STEPS,
                min_steps=MIN_STEPS,
                patience=PATIENCE,
                protocol_fingerprint=fingerprint,
                device=device,
                amp=AMP,
                mode_by_window=mode_by_window,
            )
        )
    if tuple(int(item["seed"]) for item in summaries) != EXPECTED_SEEDS:
        raise RuntimeError("Candidate summaries are not the complete locked seed set")
    for seed, item in zip(EXPECTED_SEEDS, summaries, strict=True):
        if int(item["horizon_samples"]) != HORIZON_SAMPLES:
            raise RuntimeError(f"Candidate horizon mismatch for seed {seed}")
        model_config = item.get("model_config", {})
        conditioning = model_config.get("conditioning", {})
        if conditioning.get("type") != "external_frozen_context_cluster_embedding":
            raise RuntimeError(f"Candidate model mismatch for seed {seed}")
        if int(conditioning.get("n_clusters", -1)) != N_CLUSTERS:
            raise RuntimeError(f"Candidate cluster count mismatch for seed {seed}")
        checkpoint_path = root / "runs" / f"seed_{seed}" / "best.pt"
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if int(checkpoint.get("seed", -1)) != seed:
            raise RuntimeError(f"Candidate checkpoint seed mismatch for seed {seed}")
        if checkpoint.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"Candidate checkpoint protocol mismatch for seed {seed}")
        if int(checkpoint.get("horizon_samples", -1)) != HORIZON_SAMPLES:
            raise RuntimeError(f"Candidate checkpoint horizon mismatch for seed {seed}")
        model = HardClusterConditionedGRUMeanForecaster(
            in_channels=dataset.n_channels,
            horizon=HORIZON_SAMPLES,
            n_clusters=N_CLUSTERS,
            hidden_channels=48,
            num_layers=1,
            dropout=0.1,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        recomputed = suite.evaluate_mean(
            model,
            dataset,
            windows,
            validation_indices,
            scaler,
            HORIZON_SAMPLES,
            256,
            device,
            mode_by_window,
        )
        if not math.isclose(
            float(recomputed["rmse_scaled"]),
            float(item["best_validation_rmse"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"Candidate RMSE does not reproduce for seed {seed}")
    comparison = suite._paired_comparison(
        baseline, summaries, "best_validation_rmse"
    )
    if tuple(int(seed) for seed in comparison["seeds"]) != EXPECTED_SEEDS:
        raise RuntimeError("Paired comparison omitted or reordered a seed")
    required_wins = math.ceil(0.8 * len(EXPECTED_SEEDS))
    supported = bool(
        comparison["relative_gain_stats"]["mean"] >= 0.01
        and comparison["candidate_win_count"] >= required_wins
    )
    metric_keys = (
        "best_validation_rmse",
        "best_validation_mae",
        "rmse_skill_vs_persistence",
        "best_step",
        "cumulative_optimizer_steps",
        "last_five_validation_rmse_slope_per_epoch",
        "gradient_clip_step_fraction",
    )
    candidate = suite.aggregate_run_summaries(summaries, metric_keys)
    candidate.update(
        {
            "parameter_count": int(parameter_count(
                HardClusterConditionedGRUMeanForecaster(
                    in_channels=dataset.n_channels,
                    horizon=HORIZON_SAMPLES,
                    n_clusters=N_CLUSTERS,
                    hidden_channels=48,
                    dropout=0.1,
                )
            )),
            "per_cluster_validation": per_cluster_stats(
                summaries, N_CLUSTERS
            ),
        }
    )
    aggregate = {
        "reference": stage_aggregate["arms"]["global_direct"],
        "candidate": candidate,
        "paired_hard_vs_global_rmse": comparison,
        "required_seed_wins": required_wins,
        "hard_cluster_conditioning_supported": supported,
        "selected_arm": "hard_cluster_embedding" if supported else "global_direct",
        "scope": (
            "Frozen KMeans labels are context-only inputs. This directly tests "
            "hard-cluster conditioning, but clusters remain descriptive rather "
            "than clinically labelled activities."
        ),
        "test_record_evaluated": False,
    }
    atomic_json_dump(aggregate, root / "aggregate.json")
    diagnostic.write_csv(
        root / "run_table.csv",
        [
            {
                "seed": item["seed"],
                "stop_reason": item["stop_reason"],
                **{key: item[key] for key in metric_keys},
            }
            for item in summaries
        ],
    )
    rows = "\n".join(
        f"| {item['seed']} | {item['stop_reason']} | "
        f"{item['best_step']} | {item['best_validation_rmse']:.6f} | "
        f"{item['rmse_skill_vs_persistence']:.2%} |"
        for item in summaries
    )
    report = f"""# S01 冻结 KMeans 硬簇条件化实验

训练集 context 的 StandardScaler/PCA/KMeans 及 k=3 选择全部沿用冻结 v4 模型；验证窗口仅由该冻结模型分配。cluster ID 现在明确输入 GRU 状态加性 embedding，target、FoG 标签和残差均不参与路由。相同 seed 下，global 与 hard arm 的 GRU/decoder 初值及零 embedding 初始预测完全相同；hard 只增加 {N_CLUSTERS * 48} 个参数。R02 未读取或评估。

硬簇条件化获得支持：**{supported}**；相对 global 的平均配对 RMSE 改善为 **{comparison['relative_gain_stats']['mean']:.2%}**，获胜 {comparison['candidate_win_count']}/5；选择 `{aggregate['selected_arm']}`。

| seed | stop | 最佳步数 | 验证 RMSE | 相对持久性技能 |
|---:|---|---:|---:|---:|
{rows}

这里的簇是能量、均值、变化率等 context 统计量的无监督区域，不等同于临床活动标签。若不满足预注册收益规则，应保留 global 模型，而不是机械照搬 SCADA 的设备 embedding。
"""
    suite._atomic_text(root / "report.md", report)
    atomic_json_dump(
        {
            "created_utc": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
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
            "hard_cluster_conditioning_supported": supported,
            "selected_arm": aggregate["selected_arm"],
            "test_record_evaluated": False,
            "artifacts": artifacts,
        },
        root / "DONE.json",
    )
    validate_output(root, fingerprint)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"Results: {root}")


if __name__ == "__main__":
    main()
