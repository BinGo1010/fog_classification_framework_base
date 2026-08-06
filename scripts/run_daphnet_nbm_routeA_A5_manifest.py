"""Run manifest-driven within-subject A5 FoG/Non-FoG residual diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_daphnet_nbm_routeA_A1b_generalization_repair as a1b  # noqa: E402
import run_daphnet_nbm_routeA_A5 as legacy_a5  # noqa: E402
import run_daphnet_nbm_routeA_final_residual_validation as route_a  # noqa: E402


EXPERIMENT = "daphnet_nbm_routeA_A5_manifest_v1"
DEFAULT_SUBJECTS = ("S01", "S02", "S05", "S06", "S07", "S08", "S09")
SELECTION_SUBJECTS = ("S01", "S05", "S08", "S09")
DEFAULT_SEEDS = (20260802, 20260803, 20260804)
SCORES = ("S0", "S1", "S2", "S3")
WINDOW = 128
CHANNELS = 9


@dataclass
class SubjectData:
    subject: str
    role_rows: dict[str, list[dict[str, str]]]
    raw: dict[str, np.ndarray]
    processed: dict[str, np.ndarray]
    scaler: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed_A5",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / EXPERIMENT / "routeA_A5_manifest",
    )
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument(
        "--train-scope",
        choices=("n8", "full"),
        default="n8",
        help="Use the frozen N=8 subset or every internal-training clean Non-FoG window.",
    )
    parser.add_argument(
        "--checkpoint-origin",
        default="",
        help="Optional provenance note when mathematically identical training checkpoints are reused.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_a5_artifact(data_dir: Path, standard_name: str) -> Path:
    candidates = [data_dir / standard_name]
    if standard_name.startswith("a5_"):
        candidates.append(data_dir / standard_name.replace("a5_", "a5_50_", 1))
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing A5 artifact; tried: {candidates}")


def parse_subjects(value: str) -> tuple[str, ...]:
    subjects = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(subjects) - set(DEFAULT_SUBJECTS))
    if unknown:
        raise ValueError(f"A5 formal protocol does not include: {unknown}")
    return subjects


def parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def stack_windows(rows: Sequence[dict[str, str]], records: dict[str, Any]) -> np.ndarray:
    windows: list[np.ndarray] = []
    for row in rows:
        start = int(row["start_index"])
        end = int(row["end_index_exclusive"])
        if end - start != WINDOW:
            raise ValueError(f"bad window length in {row['window_id']}")
        values = np.asarray(records[row["record_id"]].x[start:end], dtype=np.float32)
        if values.shape != (WINDOW, CHANNELS):
            raise ValueError(f"bad window shape in {row['window_id']}: {values.shape}")
        windows.append(values)
    if not windows:
        return np.empty((0, WINDOW, CHANNELS), dtype=np.float32)
    return np.ascontiguousarray(np.stack(windows).astype(np.float32))


def fit_training_scaler(train_raw: np.ndarray, train_scope: str) -> dict[str, Any]:
    if train_raw.ndim != 3 or train_raw.shape[1:] != (WINDOW, CHANNELS):
        raise ValueError(f"unexpected training data shape: {train_raw.shape}")
    if train_scope == "n8" and train_raw.shape[0] != 8:
        raise ValueError(f"expected fixed N=8 training data, got {train_raw.shape}")
    if train_raw.shape[0] < 8:
        raise ValueError(f"too few internal-training windows: {train_raw.shape[0]}")
    points = train_raw.reshape(-1, CHANNELS).astype(np.float64)
    q25, median, q75 = np.percentile(points, [25.0, 50.0, 75.0], axis=0)
    iqr = q75 - q25
    if np.any(iqr <= 1e-6):
        raise ValueError(f"degenerate scaler channels: {np.flatnonzero(iqr <= 1e-6).tolist()}")
    return {
        "median": median.astype(float).tolist(),
        "iqr": iqr.astype(float).tolist(),
        "epsilon": 1e-6,
        "fit_scope": (
            "unique samples in frozen N=8 internal-training windows only"
            if train_scope == "n8"
            else "all samples in manifest internal-training clean Non-FoG windows"
        ),
        "training_window_count": int(train_raw.shape[0]),
        "window_axis_centering": True,
    }


def transform(raw: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    median = np.asarray(scaler["median"], dtype=np.float32)
    iqr = np.asarray(scaler["iqr"], dtype=np.float32)
    values = (np.asarray(raw, dtype=np.float32) - median) / (iqr + float(scaler["epsilon"]))
    values = values - values.mean(axis=1, keepdims=True)
    return np.ascontiguousarray(values.astype(np.float32))


def load_subject_data(
    data_dir: Path,
    subjects: Sequence[str],
    train_scope: str,
) -> tuple[dict[str, SubjectData], list[dict[str, str]]]:
    quality_path = resolve_a5_artifact(data_dir, "a5_quality_report.json")
    manifest_path = resolve_a5_artifact(data_dir, "a5_window_manifest.csv")
    n8_path = resolve_a5_artifact(data_dir, "a5_n8_training_selection.csv")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if not quality.get("overall_pass"):
        raise RuntimeError("processed_A5 quality gate is not PASS")
    dataset = route_a.DaphnetDataset.load(data_dir)
    records = {record.record_id: record for record in dataset.records}
    manifest_rows = read_csv(manifest_path)
    n8_rows = read_csv(n8_path)
    by_id = {row["window_id"]: row for row in manifest_rows}
    if len(by_id) != len(manifest_rows):
        raise ValueError("duplicate window_id in a5_window_manifest.csv")
    result: dict[str, SubjectData] = {}
    roles = (
        "nbm_internal_train_nonfog",
        "nbm_internal_earlystop_nonfog",
        "external_validation_nonfog",
        "external_validation_fog",
        "external_test_nonfog",
        "external_test_fog",
    )
    for subject in subjects:
        role_rows = {
            role: sorted(
                [row for row in manifest_rows if row["subject_id"] == subject and row["a5_role"] == role],
                key=lambda row: (row["record_id"], int(row["start_index"])),
            )
            for role in roles
        }
        if any(not role_rows[role] for role in roles):
            raise ValueError(f"{subject} has an empty A5 role")
        if train_scope == "n8":
            selected = sorted(
                [row for row in n8_rows if row["subject_id"] == subject],
                key=lambda row: int(row["selection_order"]),
            )
            if len(selected) != 8:
                raise ValueError(f"{subject} N=8 selection has {len(selected)} rows")
            training_rows = [by_id[row["window_id"]] for row in selected]
            if any(row["a5_role"] != "nbm_internal_train_nonfog" for row in training_rows):
                raise ValueError(f"{subject} N=8 selection escapes the internal-training pool")
        else:
            training_rows = role_rows["nbm_internal_train_nonfog"]
        raw = {
            "train_nonfog": stack_windows(training_rows, records),
            "earlystop_nonfog": stack_windows(role_rows["nbm_internal_earlystop_nonfog"], records),
            "validation_nonfog": stack_windows(role_rows["external_validation_nonfog"], records),
            "validation_fog": stack_windows(role_rows["external_validation_fog"], records),
            "test_nonfog": stack_windows(role_rows["external_test_nonfog"], records),
            "test_fog": stack_windows(role_rows["external_test_fog"], records),
        }
        scaler = fit_training_scaler(raw["train_nonfog"], train_scope)
        processed = {name: transform(values, scaler) for name, values in raw.items()}
        if not all(np.isfinite(values).all() for values in processed.values()):
            raise FloatingPointError(f"{subject} contains non-finite processed values")
        result[subject] = SubjectData(subject, role_rows, raw, processed, scaler)
    return result, manifest_rows


@torch.no_grad()
def residual_components(model: torch.nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    predicted, _ = a1b.predict_pairs(model, values, values, device)
    residual = np.asarray(values - predicted, dtype=np.float32)
    return legacy_a5.component_scores(residual)


def metrics(
    nonfog: np.ndarray,
    fog: np.ndarray,
    validation_reference: np.ndarray,
) -> dict[str, Any]:
    result = legacy_a5.separation_metrics(nonfog, fog, validation_reference)
    result["threshold_reference"] = "external_validation_nonfog"
    result["validation_nonfog_p95_threshold"] = result["train_nonfog_p95_threshold"]
    return result


def candidate_summary(rows: Sequence[dict[str, Any]], score: str) -> dict[str, Any]:
    selected = [
        row for row in rows if row["score"] == score and row["subject_id"] in SELECTION_SUBJECTS
    ]
    return {
        "score": score,
        "selection_runs": len(selected),
        "median_validation_auroc": legacy_a5.finite_median(selected, "auroc"),
        "median_validation_average_precision": legacy_a5.finite_median(selected, "average_precision"),
        "median_validation_cliffs_delta": legacy_a5.finite_median(selected, "cliffs_delta"),
        "median_validation_fog_nonfog_ratio": legacy_a5.finite_median(
            selected, "fog_to_nonfog_median_ratio"
        ),
        "median_validation_false_alarm_per_minute": legacy_a5.finite_median(
            selected, "false_alarm_windows_per_minute"
        ),
    }


def render_report(root: Path, gate: dict[str, Any], training_rows: Sequence[dict[str, Any]]) -> None:
    subject_lines = [
        "| 被试 | AUROC | PR-AUC/随机基线 | FoG÷Non-FoG中位数 | Recall@验证Q95 | FA/min | Cliff's δ | 可用门控 | 强门控 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in gate["subject_summaries"]:
        subject_lines.append(
            f"| {row['subject_id']} | {row['median_auroc']:.3f} | "
            f"{row['median_average_precision']:.3f}/{row['median_random_pr_baseline']:.3f} | "
            f"{row['median_fog_to_nonfog_median_ratio']:.3f} | "
            f"{row['median_recall_at_train_nonfog_p95']:.1%} | "
            f"{row['median_false_alarm_windows_per_minute']:.2f} | "
            f"{row['median_cliffs_delta']:.3f} | "
            f"{'PASS' if row['usable_subject_pass'] else 'FAIL'} | "
            f"{'PASS' if row['strong_subject_pass'] else 'FAIL'} |"
        )
    candidate_lines = [
        "| 分数 | 验证AUROC | 验证PR-AUC | Cliff's δ | FoG÷Non-FoG | FA/min |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in gate["score_candidates"]:
        candidate_lines.append(
            f"| {row['score']} | {row['median_validation_auroc']:.3f} | "
            f"{row['median_validation_average_precision']:.3f} | "
            f"{row['median_validation_cliffs_delta']:.3f} | "
            f"{row['median_validation_fog_nonfog_ratio']:.3f} | "
            f"{row['median_validation_false_alarm_per_minute']:.2f} |"
        )
    best_epochs = [float(row["best_epoch"]) for row in training_rows]
    elapsed = sum(float(row["elapsed_seconds"]) for row in training_rows)
    checkpoint_note = (
        f"- Checkpoint来源：`{gate['checkpoint_origin']}`；训练/早停/验证Non-FoG/测试Non-FoG窗口与来源实验逐一一致，因此未重复训练。"
        if gate.get("checkpoint_origin")
        else "- Checkpoint来源：在本输出目录内完成训练。"
    )
    best_subject = max(gate["subject_summaries"], key=lambda row: row["median_auroc"])
    largest_drop = max(
        gate["subject_summaries"],
        key=lambda row: row["median_validation_to_test_auroc_drop"],
    )
    usable_subjects = [
        row["subject_id"] for row in gate["subject_summaries"] if row["usable_subject_pass"]
    ]
    failed_subjects = [
        row["subject_id"] for row in gate["subject_summaries"] if not row["usable_subject_pass"]
    ]
    near_random_pr = [
        row["subject_id"]
        for row in gate["subject_summaries"]
        if row["median_pr_margin_over_random"] <= 0.03
    ]
    training_scope_text = (
        "固定N=8 clean Non-FoG"
        if gate["training_scope"] == "n8"
        else "manifest中全部内部训练clean Non-FoG"
    )
    report = f"""# Daphnet NBM A5：manifest驱动的FoG / Non-FoG残差分离诊断

生成时间（UTC）：{gate['completed_utc']}

## 实验口径

- 数据入口：`{gate['data_dir_name']}/{gate['manifest_filename']}`，直接读取既有划分，未重新生成窗口。
- 正式被试：S01、S02、S05、S06、S07、S08、S09；每名被试3个种子。
- NBM：`M3_tcdae_long + L4 + W0 + D0`；每次使用{training_scope_text}训练。
- 早停：仅使用`nbm_internal_earlystop_nonfog`。
- 校准与表示：冻结协议为`C0_clipnone + R5=[R, |R|, ΔR]`；本轮A5从原始残差`R`提取S0/S1/S2，并在验证集组合S3。
- S3权重和最终分数仅使用外部验证集选择；测试集在冻结后评价一次。
- 阈值为每个被试/种子的外部验证clean Non-FoG第95百分位，随后冻结到测试集。

## 训练执行

- 完成运行：{len(training_rows)}。
- 最佳epoch中位数：{np.median(best_epochs):.0f}。
- 累计训练时间：{elapsed / 60.0:.1f}分钟。
{checkpoint_note}

## 验证集选型

{chr(10).join(candidate_lines)}

- 选中分数：`{gate['selected_score']}`。
- S3权重：S0={gate['s3_weights'][0]:.1f}、S1={gate['s3_weights'][1]:.1f}、S2={gate['s3_weights'][2]:.1f}。

## 冻结后测试结果

{chr(10).join(subject_lines)}

## 总体结论

- A5状态：**{gate['status']}**。
- 可用门控：{gate['usable_subject_count']}/7通过，要求至少5/7。
- 强门控：{gate['strong_subject_count']}/7通过，要求至少4/7。
- 是否允许进入A6：**{'是' if gate['eligible_for_A6'] else '否'}**。

## 分离诊断

- 当前表现最好的被试是 **{best_subject['subject_id']}**：测试AUROC={best_subject['median_auroc']:.3f}、Cliff's δ={best_subject['median_cliffs_delta']:.3f}、FoG÷Non-FoG中位数={best_subject['median_fog_to_nonfog_median_ratio']:.3f}；说明该被试的FoG残差整体明显右移，但验证Q95阈值下召回仍只有{best_subject['median_recall_at_train_nonfog_p95']:.1%}。
- 可用门控通过者为{', '.join(usable_subjects)}；未通过者为{', '.join(failed_subjects)}。{', '.join(near_random_pr)}的PR-AUC与各自随机基线差值不超过0.03，残差基本没有可用的类别排序增益。
- 最大的验证到测试迁移发生在 **{largest_drop['subject_id']}**：AUROC中位数下降{largest_drop['median_validation_to_test_auroc_drop']:.3f}，提示同一被试内部仍存在明显的时间段/动作状态分布漂移。
- 验证集最终选择`S0`，S3权重退化为`[1, 0, 0]`。这说明在当前数据与冻结NBM下，频谱峰值S1和通道集中度S2没有带来稳定的额外泛化收益。
- 强门控0/7通过的直接瓶颈是阈值指标。验证Non-FoG的Q95按定义会保留约5%的窗口为假阳性；在1秒步长下，其基准量级约为3个窗口/分钟，而强门控要求不高于0.5/分钟。因此强门控失败不能单独等同于残差排序完全失败，后续若继续研究应预先冻结更严格阈值或采用事件级告警聚合，再独立复核。

## 解释边界

这是单被试内、时间块隔离的残差诊断实验，不是跨被试或跨session泛化实验。AUROC和PR-AUC判断残差排序分离能力；阈值指标同时受验证clean分布与时间相关窗口影响。窗口为2秒、步长1秒，相邻窗口并非独立，因此FA/min是窗口级诊断量，不能直接视为独立FoG事件误报率。S04、S10无FoG，S03按既定协议仅作独立诊断，均未计入本次7人门控。
"""
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "A5_manifest_residual_separation_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    subjects = parse_subjects(args.subjects)
    seeds = parse_seeds(args.seeds)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.resolve()
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    prepared, manifest_rows = load_subject_data(data_dir, subjects, args.train_scope)
    manifest_path = resolve_a5_artifact(data_dir, "a5_window_manifest.csv")
    n8_path = resolve_a5_artifact(data_dir, "a5_n8_training_selection.csv")
    training_window_counts = {
        subject: int(len(prepared[subject].processed["train_nonfog"])) for subject in subjects
    }
    protocol = {
        "experiment": f"{EXPERIMENT}_{args.train_scope}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "a5_window_manifest": str(manifest_path),
        "a5_window_manifest_sha256": sha256(manifest_path),
        "a5_n8_selection_sha256": (
            sha256(n8_path)
            if args.train_scope == "n8"
            else None
        ),
        "subjects": list(subjects),
        "selection_subjects": [subject for subject in SELECTION_SUBJECTS if subject in subjects],
        "seeds": list(seeds),
        "device": str(device),
        "frozen_pipeline": "M3_tcdae_long+L4+W0+D0+C0_clipnone+R5",
        "training_scope": args.train_scope,
        "checkpoint_origin": args.checkpoint_origin or None,
        "data_dir_name": data_dir.name,
        "manifest_filename": manifest_path.name,
        "training_window_counts": training_window_counts,
        "training_windows_total_per_seed": int(sum(training_window_counts.values())),
        "maximum_epochs": args.max_epochs,
        "patience": args.patience,
        "score_selection_split": "external_validation only",
        "threshold_reference": "external_validation_nonfog p95",
        "test_fog_used_for_selection": False,
        "manifest_row_count": len(manifest_rows),
    }
    legacy_a5.write_json(root / "protocol" / "frozen_manifest_A5_protocol.json", protocol)
    legacy_a5.write_json(
        root / "protocol" / "subject_n8_scalers.json",
        {subject: prepared[subject].scaler for subject in subjects},
    )

    all_components: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    scales: dict[tuple[str, int], np.ndarray] = {}
    training_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    print(f"A5 manifest device={device} subjects={','.join(subjects)} seeds={seeds}", flush=True)
    for subject in subjects:
        item = prepared[subject]
        for seed in seeds:
            run_dir = root / "training" / subject / f"seed{seed}"
            model, _, training = a1b.train_repair_model(
                item.processed["train_nonfog"],
                item.processed["train_nonfog"],
                item.processed["earlystop_nonfog"],
                item.processed["earlystop_nonfog"],
                run_dir,
                subject=subject,
                seed=seed,
                loss_name="L4",
                context_name="W0",
                max_epochs=args.max_epochs,
                patience=args.patience,
                workers=args.workers,
                device=device,
            )
            training_rows.append({"subject_id": subject, "seed": seed, **training})
            components = {
                name: residual_components(model, values, device)
                for name, values in item.processed.items()
                if name != "earlystop_nonfog"
            }
            all_components[(subject, seed)] = components
            scales[(subject, seed)] = legacy_a5.fit_component_scale(components["train_nonfog"])
            score_dir = root / "component_scores" / subject / f"seed{seed}"
            score_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                score_dir / "component_scores.npz",
                **components,
                train_component_scale=scales[(subject, seed)],
            )
            audit_rows.append(
                {
                    "subject_id": subject,
                    "seed": seed,
                    "train_nonfog_windows": len(components["train_nonfog"]),
                    "earlystop_nonfog_windows": len(item.processed["earlystop_nonfog"]),
                    "validation_nonfog_windows": len(components["validation_nonfog"]),
                    "validation_fog_windows": len(components["validation_fog"]),
                    "test_nonfog_windows": len(components["test_nonfog"]),
                    "test_fog_windows": len(components["test_fog"]),
                    "fallback_validation_fog_windows": sum(
                        row["window_alignment"] == "event_fallback"
                        for row in item.role_rows["external_validation_fog"]
                    ),
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"A5 manifest DONE {subject} seed={seed}", flush=True)
    legacy_a5.write_csv(root / "tables" / "training_summary.csv", training_rows)
    legacy_a5.write_csv(root / "tables" / "window_and_manifest_audit.csv", audit_rows)

    selection_subjects = tuple(subject for subject in SELECTION_SUBJECTS if subject in subjects)
    if not selection_subjects:
        raise ValueError("no frozen score-selection subject is present")
    weight_rows: list[dict[str, Any]] = []
    for weights in legacy_a5.simplex_weights(args.weight_step):
        rows: list[dict[str, Any]] = []
        for subject in selection_subjects:
            for seed in seeds:
                arrays = legacy_a5.score_arrays(
                    all_components[(subject, seed)], scales[(subject, seed)], "S3", weights
                )
                rows.append(
                    metrics(
                        arrays["validation_nonfog"],
                        arrays["validation_fog"],
                        arrays["validation_nonfog"],
                    )
                )
        weight_rows.append(
            {
                "weight_s0": weights[0],
                "weight_s1": weights[1],
                "weight_s2": weights[2],
                "selection_runs": len(rows),
                "median_validation_auroc": legacy_a5.finite_median(rows, "auroc"),
                "median_validation_average_precision": legacy_a5.finite_median(
                    rows, "average_precision"
                ),
                "median_validation_cliffs_delta": legacy_a5.finite_median(rows, "cliffs_delta"),
                "median_validation_false_alarm_per_minute": legacy_a5.finite_median(
                    rows, "false_alarm_windows_per_minute"
                ),
            }
        )
    best_weight = max(
        weight_rows,
        key=lambda row: (
            row["median_validation_auroc"],
            row["median_validation_cliffs_delta"],
            row["median_validation_average_precision"],
            -row["median_validation_false_alarm_per_minute"],
            row["weight_s0"],
            row["weight_s1"],
        ),
    )
    weights = (
        float(best_weight["weight_s0"]),
        float(best_weight["weight_s1"]),
        float(best_weight["weight_s2"]),
    )
    for row in weight_rows:
        row["selected"] = bool(
            np.allclose([row["weight_s0"], row["weight_s1"], row["weight_s2"]], weights)
        )
    legacy_a5.write_csv(root / "tables" / "S3_validation_weight_search.csv", weight_rows)

    validation_rows: list[dict[str, Any]] = []
    score_cache: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    for subject in subjects:
        for seed in seeds:
            for score in SCORES:
                arrays = legacy_a5.score_arrays(
                    all_components[(subject, seed)], scales[(subject, seed)], score, weights
                )
                score_cache[(subject, seed, score)] = arrays
                result = metrics(
                    arrays["validation_nonfog"],
                    arrays["validation_fog"],
                    arrays["validation_nonfog"],
                )
                usable, strong = legacy_a5.run_gate(result)
                validation_rows.append(
                    {
                        "stage": "A5_manifest_residual_separation",
                        "selection_split": "external_validation",
                        "score": score,
                        "subject_id": subject,
                        "seed": seed,
                        "usable_run_pass": usable,
                        "strong_run_pass": strong,
                        **result,
                    }
                )
    legacy_a5.write_csv(root / "tables" / "all_validation_score_metrics.csv", validation_rows)
    candidates = [candidate_summary(validation_rows, score) for score in SCORES]
    selected_candidate = max(
        candidates,
        key=lambda row: (
            row["median_validation_auroc"],
            row["median_validation_cliffs_delta"],
            row["median_validation_average_precision"],
            -row["median_validation_false_alarm_per_minute"],
            -SCORES.index(row["score"]),
        ),
    )
    selected_score = str(selected_candidate["score"])
    selected_validation = [row for row in validation_rows if row["score"] == selected_score]
    legacy_a5.write_csv(root / "tables" / "selected_score_validation_metrics.csv", selected_validation)

    test_rows: list[dict[str, Any]] = []
    selected_arrays: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for subject in subjects:
        for seed in seeds:
            arrays = score_cache[(subject, seed, selected_score)]
            selected_arrays[(subject, seed)] = arrays
            validation = next(
                row
                for row in selected_validation
                if row["subject_id"] == subject and int(row["seed"]) == seed
            )
            result = metrics(
                arrays["test_nonfog"], arrays["test_fog"], arrays["validation_nonfog"]
            )
            usable, strong = legacy_a5.run_gate(result)
            test_rows.append(
                {
                    "stage": "A5_manifest_residual_separation",
                    "report_split": "external_test_after_freeze",
                    "score": selected_score,
                    "subject_id": subject,
                    "seed": seed,
                    "evaluable": True,
                    "test_nonfog_windows": len(arrays["test_nonfog"]),
                    "test_fog_windows": len(arrays["test_fog"]),
                    "usable_run_pass": usable,
                    "strong_run_pass": strong,
                    "validation_to_test_auroc_drop": float(validation["auroc"]) - result["auroc"],
                    "validation_to_test_pr_drop": float(validation["average_precision"])
                    - result["average_precision"],
                    **result,
                }
            )
    legacy_a5.write_csv(root / "tables" / "selected_score_test_metrics.csv", test_rows)
    summaries = [legacy_a5.subject_summary(test_rows, subject) for subject in subjects]
    usable_count = sum(bool(row["usable_subject_pass"]) for row in summaries)
    strong_count = sum(bool(row["strong_subject_pass"]) for row in summaries)
    status = "STRONG PASS" if strong_count >= 4 else "PASS" if usable_count >= 5 else "FAIL"
    gate = {
        "stage": "A5_manifest_residual_separation",
        "status": status,
        "training_scope": args.train_scope,
        "checkpoint_origin": args.checkpoint_origin or None,
        "data_dir_name": data_dir.name,
        "manifest_filename": manifest_path.name,
        "training_window_counts": training_window_counts,
        "selected_score": selected_score,
        "s3_weights": weights,
        "selection_split": "external_validation only",
        "selection_subjects": list(selection_subjects),
        "test_subjects": list(subjects),
        "test_fog_used_for_selection": False,
        "threshold_reference": "per-run external_validation_nonfog p95",
        "score_candidates": candidates,
        "subject_summaries": summaries,
        "usable_subject_count": usable_count,
        "usable_subject_required": 5,
        "strong_subject_count": strong_count,
        "strong_subject_required": 4,
        "usable_gate_pass": bool(usable_count >= 5),
        "strong_gate_pass": bool(strong_count >= 4),
        "eligible_for_A6": bool(usable_count >= 5),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    legacy_a5.write_json(root / "A5_manifest_gate.json", gate)
    legacy_a5.plot_candidate_comparison(root, candidates)
    legacy_a5.plot_subject_metrics(root, summaries)
    legacy_a5.plot_gate_matrix(root, summaries)
    if tuple(subjects) == DEFAULT_SUBJECTS and tuple(seeds) == DEFAULT_SEEDS:
        legacy_a5.plot_curves(root, selected_arrays)
    render_report(root, gate, training_rows)
    print(f"COMPLETE status={status} selected={selected_score} results={root}", flush=True)


if __name__ == "__main__":
    main()
