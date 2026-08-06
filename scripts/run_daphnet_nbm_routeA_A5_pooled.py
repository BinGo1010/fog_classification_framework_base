"""Run a pooled seven-subject A5 NBM residual-separation experiment."""

from __future__ import annotations

import argparse
import json
import sys
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
import run_daphnet_nbm_routeA_A5_manifest as manifest_a5  # noqa: E402


EXPERIMENT = "daphnet_nbm_routeA_A5_50_pooled7_full_v1"
SUBJECTS = manifest_a5.DEFAULT_SUBJECTS
SEEDS = manifest_a5.DEFAULT_SEEDS
SCORES = manifest_a5.SCORES
ROLE_NAMES = (
    "train_nonfog",
    "earlystop_nonfog",
    "validation_nonfog",
    "validation_fog",
    "test_nonfog",
    "test_fog",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            ROOT
            / "dataset"
            / "1.Daphnet Freezing of Gait Dataset"
            / "processed_A5_50"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / EXPERIMENT / "routeA_A5_50_pooled7_full",
    )
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def concatenate_subjects(
    prepared: dict[str, manifest_a5.SubjectData], role: str
) -> np.ndarray:
    arrays = [prepared[subject].processed[role] for subject in SUBJECTS]
    return np.ascontiguousarray(np.concatenate(arrays, axis=0).astype(np.float32))


def finite_median(rows: Sequence[dict[str, Any]], key: str) -> float:
    return legacy_a5.finite_median(rows, key)


def candidate_summary(rows: Sequence[dict[str, Any]], score: str) -> dict[str, Any]:
    selected = [row for row in rows if row["score"] == score]
    return {
        "score": score,
        "selection_runs": len(selected),
        "median_validation_auroc": finite_median(selected, "auroc"),
        "median_validation_average_precision": finite_median(selected, "average_precision"),
        "median_validation_cliffs_delta": finite_median(selected, "cliffs_delta"),
        "median_validation_fog_nonfog_ratio": finite_median(
            selected, "fog_to_nonfog_median_ratio"
        ),
        "median_validation_false_alarm_per_minute": finite_median(
            selected, "false_alarm_windows_per_minute"
        ),
    }


def summary_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {
        "auroc": finite_median(rows, "auroc"),
        "average_precision": finite_median(rows, "average_precision"),
        "random_pr_baseline": finite_median(rows, "random_pr_baseline"),
        "pr_margin_over_random": finite_median(rows, "pr_margin_over_random"),
        "fog_to_nonfog_median_ratio": finite_median(rows, "fog_to_nonfog_median_ratio"),
        "recall_at_validation_nonfog_p95": finite_median(
            rows, "recall_at_train_nonfog_p95"
        ),
        "false_alarm_windows_per_minute": finite_median(
            rows, "false_alarm_windows_per_minute"
        ),
        "cliffs_delta": finite_median(rows, "cliffs_delta"),
    }


def render_report(root: Path, gate: dict[str, Any], training_rows: Sequence[dict[str, Any]]) -> None:
    validation = gate["pooled_validation_summary"]
    test = gate["pooled_test_summary"]
    subject_lines = [
        "| 被试 | 测试AUROC | PR-AUC/随机基线 | FoG÷Non-FoG | Recall@全局验证Q95 | FA/min | 可用门控 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in gate["subject_summaries"]:
        subject_lines.append(
            f"| {row['subject_id']} | {row['median_auroc']:.3f} | "
            f"{row['median_average_precision']:.3f}/{row['median_random_pr_baseline']:.3f} | "
            f"{row['median_fog_to_nonfog_median_ratio']:.3f} | "
            f"{row['median_recall_at_train_nonfog_p95']:.1%} | "
            f"{row['median_false_alarm_windows_per_minute']:.2f} | "
            f"{'PASS' if row['usable_subject_pass'] else 'FAIL'} |"
        )
    candidate_lines = [
        "| 分数 | 合并验证AUROC | 合并验证PR-AUC | Cliff's δ | FoG÷Non-FoG |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in gate["score_candidates"]:
        candidate_lines.append(
            f"| {row['score']} | {row['median_validation_auroc']:.3f} | "
            f"{row['median_validation_average_precision']:.3f} | "
            f"{row['median_validation_cliffs_delta']:.3f} | "
            f"{row['median_validation_fog_nonfog_ratio']:.3f} |"
        )
    counts = gate["window_counts"]
    best_epochs = [float(row["best_epoch"]) for row in training_rows]
    elapsed = sum(float(row["elapsed_seconds"]) for row in training_rows)
    paradox_subjects = [
        row["subject_id"]
        for row in gate["subject_summaries"]
        if row["strong_subject_pass"] and row["median_recall_at_train_nonfog_p95"] < 0.20
    ]
    paradox_note = (
        f"- 门控警示：{', '.join(paradox_subjects)}被旧强门控计为通过，但召回低于20%；旧强门控未设置召回下限，这属于阈值过高造成的表面低误报，不能解释为有效FoG检测。"
        if paradox_subjects
        else "- 门控警示：未出现强门控通过但召回低于20%的被试。"
    )
    report = f"""# Daphnet NBM A5：7被试合并训练与合并评价

生成时间（UTC）：{gate['completed_utc']}

## 实验口径

- 数据：`processed_A5_50/a5_50_window_manifest.csv`，正式7被试共同建模。
- 缩放：每名被试只用自己的内部训练clean Non-FoG拟合median/IQR，变换后再合并；避免跨被试幅值尺度主导共享模型。
- 训练：{counts['train_nonfog']}个clean Non-FoG共同训练一个`M3_tcdae_long + L4 + W0 + D0`。
- 早停：{counts['earlystop_nonfog']}个clean Non-FoG共同选择最佳epoch。
- 外部验证：{counts['validation_nonfog']}个Non-FoG和{counts['validation_fog']}个FoG共同选择S0/S1/S2/S3、S3权重和一个全局Q95阈值。
- 外部测试：{counts['test_nonfog']}个Non-FoG和{counts['test_fog']}个FoG仅在冻结后评价。
- 合并损失与指标按窗口计权；因此窗口较多的被试贡献更大，同时另报逐被试结果。

## 训练执行

- 完成种子：{len(training_rows)}。
- 最佳epoch中位数：{np.median(best_epochs):.0f}。
- 累计训练时间：{elapsed / 60.0:.1f}分钟。

## 合并验证选型

{chr(10).join(candidate_lines)}

- 选中分数：`{gate['selected_score']}`。
- S3权重：S0={gate['s3_weights'][0]:.1f}、S1={gate['s3_weights'][1]:.1f}、S2={gate['s3_weights'][2]:.1f}。

## 合并验证与冻结测试

| 指标 | 合并验证 | 合并测试 |
|---|---:|---:|
| AUROC | {validation['auroc']:.3f} | {test['auroc']:.3f} |
| PR-AUC/随机基线 | {validation['average_precision']:.3f}/{validation['random_pr_baseline']:.3f} | {test['average_precision']:.3f}/{test['random_pr_baseline']:.3f} |
| FoG÷Non-FoG中位数 | {validation['fog_to_nonfog_median_ratio']:.3f} | {test['fog_to_nonfog_median_ratio']:.3f} |
| Recall@全局验证Q95 | {validation['recall_at_validation_nonfog_p95']:.1%} | {test['recall_at_validation_nonfog_p95']:.1%} |
| FA/min | {validation['false_alarm_windows_per_minute']:.2f} | {test['false_alarm_windows_per_minute']:.2f} |
| Cliff's δ | {validation['cliffs_delta']:.3f} | {test['cliffs_delta']:.3f} |

## 逐被试冻结测试

{chr(10).join(subject_lines)}

## 门控结论

- A5状态：**{gate['status']}**。
- 逐被试可用门控：{gate['usable_subject_count']}/7，要求至少5/7。
- 逐被试强门控：{gate['strong_subject_count']}/7，要求至少4/7。
- 合并测试3个种子中，可用运行{gate['pooled_usable_run_count']}/3，强门控运行{gate['pooled_strong_run_count']}/3。
{paradox_note}

## 解释边界

这是共享NBM的窗口加权合并基线，不是留一被试泛化实验。测试中的每名被试都在共享NBM的clean Non-FoG训练池中出现过，但任何FoG窗口均未用于NBM训练或早停。全局Q95也可能掩盖被试间不同的正常残差尺度，因此总体结果必须与逐被试结果同时解释。
"""
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "A5_50_pooled7_residual_separation_report.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    seeds = manifest_a5.parse_seeds(args.seeds)
    data_dir = args.data_dir.resolve()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    prepared, manifest_rows = manifest_a5.load_subject_data(data_dir, SUBJECTS, "full")
    manifest_path = manifest_a5.resolve_a5_artifact(data_dir, "a5_window_manifest.csv")
    pooled = {role: concatenate_subjects(prepared, role) for role in ROLE_NAMES}
    window_counts = {role: int(len(values)) for role, values in pooled.items()}
    expected = {
        "train_nonfog": 3482,
        "earlystop_nonfog": 904,
        "validation_nonfog": 1319,
        "validation_fog": 561,
        "test_nonfog": 1604,
        "test_fog": 701,
    }
    if window_counts != expected:
        raise ValueError(f"unexpected pooled A5_50 counts: {window_counts}")

    protocol = {
        "experiment": EXPERIMENT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_a5.sha256(manifest_path),
        "subjects": list(SUBJECTS),
        "seeds": list(seeds),
        "device": str(device),
        "model": "M3_tcdae_long+L4+W0+D0+C0_clipnone",
        "training_mode": "one pooled seven-subject NBM per seed",
        "scaling": "per-subject train-only median/IQR before pooling",
        "pool_weighting": "per-window",
        "score_selection": "all seven subjects combined external validation",
        "threshold": "combined external validation Non-FoG p95 per seed",
        "test_fog_used_for_selection": False,
        "maximum_epochs": args.max_epochs,
        "patience": args.patience,
        "window_counts": window_counts,
        "manifest_row_count": len(manifest_rows),
    }
    legacy_a5.write_json(root / "protocol" / "pooled_A5_50_protocol.json", protocol)
    legacy_a5.write_json(
        root / "protocol" / "subject_train_only_scalers.json",
        {subject: prepared[subject].scaler for subject in SUBJECTS},
    )

    audit_rows = []
    for subject in SUBJECTS:
        item = prepared[subject]
        audit_rows.append(
            {
                "subject_id": subject,
                **{f"{role}_windows": len(item.processed[role]) for role in ROLE_NAMES},
            }
        )
    legacy_a5.write_csv(root / "tables" / "window_and_manifest_audit.csv", audit_rows)

    training_rows: list[dict[str, Any]] = []
    subject_components: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    pooled_components: dict[int, dict[str, np.ndarray]] = {}
    component_scales: dict[int, np.ndarray] = {}
    print(f"A5 pooled7 device={device} seeds={seeds}", flush=True)
    for seed in seeds:
        run_dir = root / "training" / f"seed{seed}"
        model, _, training = a1b.train_repair_model(
            pooled["train_nonfog"],
            pooled["train_nonfog"],
            pooled["earlystop_nonfog"],
            pooled["earlystop_nonfog"],
            run_dir,
            subject="POOLED7",
            seed=seed,
            loss_name="L4",
            context_name="W0",
            max_epochs=args.max_epochs,
            patience=args.patience,
            workers=args.workers,
            device=device,
        )
        training_rows.append({"model_scope": "POOLED7", "seed": seed, **training})
        for subject in SUBJECTS:
            item = prepared[subject]
            components = {
                role: manifest_a5.residual_components(model, item.processed[role], device)
                for role in ROLE_NAMES
                if role != "earlystop_nonfog"
            }
            subject_components[(subject, seed)] = components
            score_dir = root / "component_scores" / subject / f"seed{seed}"
            score_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(score_dir / "component_scores.npz", **components)
        pooled_seed = {
            role: np.concatenate(
                [subject_components[(subject, seed)][role] for subject in SUBJECTS], axis=0
            )
            for role in ROLE_NAMES
            if role != "earlystop_nonfog"
        }
        pooled_components[seed] = pooled_seed
        component_scales[seed] = legacy_a5.fit_component_scale(pooled_seed["train_nonfog"])
        pooled_dir = root / "component_scores" / "POOLED7" / f"seed{seed}"
        pooled_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            pooled_dir / "component_scores.npz",
            **pooled_seed,
            train_component_scale=component_scales[seed],
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"A5 pooled7 DONE seed={seed}", flush=True)
    legacy_a5.write_csv(root / "tables" / "training_summary.csv", training_rows)

    weight_rows: list[dict[str, Any]] = []
    for weights in legacy_a5.simplex_weights(args.weight_step):
        rows = []
        for seed in seeds:
            arrays = legacy_a5.score_arrays(
                pooled_components[seed], component_scales[seed], "S3", weights
            )
            rows.append(
                manifest_a5.metrics(
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
                "median_validation_auroc": finite_median(rows, "auroc"),
                "median_validation_average_precision": finite_median(
                    rows, "average_precision"
                ),
                "median_validation_cliffs_delta": finite_median(rows, "cliffs_delta"),
                "median_validation_false_alarm_per_minute": finite_median(
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
    legacy_a5.write_csv(root / "tables" / "S3_pooled_validation_weight_search.csv", weight_rows)

    validation_rows: list[dict[str, Any]] = []
    pooled_score_cache: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for seed in seeds:
        for score in SCORES:
            arrays = legacy_a5.score_arrays(
                pooled_components[seed], component_scales[seed], score, weights
            )
            pooled_score_cache[(seed, score)] = arrays
            result = manifest_a5.metrics(
                arrays["validation_nonfog"],
                arrays["validation_fog"],
                arrays["validation_nonfog"],
            )
            usable, strong = legacy_a5.run_gate(result)
            validation_rows.append(
                {
                    "stage": "A5_50_pooled7",
                    "model_scope": "POOLED7",
                    "selection_split": "combined_external_validation",
                    "score": score,
                    "seed": seed,
                    "usable_run_pass": usable,
                    "strong_run_pass": strong,
                    **result,
                }
            )
    legacy_a5.write_csv(root / "tables" / "all_pooled_validation_score_metrics.csv", validation_rows)
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
    legacy_a5.write_csv(
        root / "tables" / "selected_score_pooled_validation_metrics.csv",
        selected_validation,
    )

    pooled_test_rows: list[dict[str, Any]] = []
    subject_validation_rows: list[dict[str, Any]] = []
    subject_test_rows: list[dict[str, Any]] = []
    selected_subject_arrays: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for seed in seeds:
        pooled_arrays = pooled_score_cache[(seed, selected_score)]
        pooled_validation = next(row for row in selected_validation if int(row["seed"]) == seed)
        pooled_test = manifest_a5.metrics(
            pooled_arrays["test_nonfog"],
            pooled_arrays["test_fog"],
            pooled_arrays["validation_nonfog"],
        )
        usable, strong = legacy_a5.run_gate(pooled_test)
        pooled_test_rows.append(
            {
                "stage": "A5_50_pooled7",
                "model_scope": "POOLED7",
                "report_split": "combined_external_test_after_freeze",
                "score": selected_score,
                "seed": seed,
                "usable_run_pass": usable,
                "strong_run_pass": strong,
                "validation_to_test_auroc_drop": float(pooled_validation["auroc"])
                - pooled_test["auroc"],
                "validation_to_test_pr_drop": float(pooled_validation["average_precision"])
                - pooled_test["average_precision"],
                **pooled_test,
            }
        )
        for subject in SUBJECTS:
            arrays = legacy_a5.score_arrays(
                subject_components[(subject, seed)],
                component_scales[seed],
                selected_score,
                weights,
            )
            selected_subject_arrays[(subject, seed)] = arrays
            validation_result = manifest_a5.metrics(
                arrays["validation_nonfog"],
                arrays["validation_fog"],
                pooled_arrays["validation_nonfog"],
            )
            val_usable, val_strong = legacy_a5.run_gate(validation_result)
            subject_validation_rows.append(
                {
                    "stage": "A5_50_pooled7",
                    "model_scope": "POOLED7",
                    "selection_split": "subject_external_validation_global_threshold",
                    "score": selected_score,
                    "subject_id": subject,
                    "seed": seed,
                    "usable_run_pass": val_usable,
                    "strong_run_pass": val_strong,
                    **validation_result,
                }
            )
            test_result = manifest_a5.metrics(
                arrays["test_nonfog"],
                arrays["test_fog"],
                pooled_arrays["validation_nonfog"],
            )
            test_usable, test_strong = legacy_a5.run_gate(test_result)
            subject_test_rows.append(
                {
                    "stage": "A5_50_pooled7",
                    "model_scope": "POOLED7",
                    "report_split": "subject_external_test_global_threshold",
                    "score": selected_score,
                    "subject_id": subject,
                    "seed": seed,
                    "evaluable": True,
                    "test_nonfog_windows": len(arrays["test_nonfog"]),
                    "test_fog_windows": len(arrays["test_fog"]),
                    "usable_run_pass": test_usable,
                    "strong_run_pass": test_strong,
                    "validation_to_test_auroc_drop": validation_result["auroc"]
                    - test_result["auroc"],
                    "validation_to_test_pr_drop": validation_result["average_precision"]
                    - test_result["average_precision"],
                    **test_result,
                }
            )
    legacy_a5.write_csv(
        root / "tables" / "selected_score_pooled_test_metrics.csv", pooled_test_rows
    )
    legacy_a5.write_csv(
        root / "tables" / "selected_score_subject_validation_metrics.csv",
        subject_validation_rows,
    )
    legacy_a5.write_csv(
        root / "tables" / "selected_score_subject_test_metrics.csv", subject_test_rows
    )

    summaries = [legacy_a5.subject_summary(subject_test_rows, subject) for subject in SUBJECTS]
    usable_count = sum(bool(row["usable_subject_pass"]) for row in summaries)
    strong_count = sum(bool(row["strong_subject_pass"]) for row in summaries)
    status = "STRONG PASS" if strong_count >= 4 else "PASS" if usable_count >= 5 else "FAIL"
    gate = {
        "stage": "A5_50_pooled7",
        "status": status,
        "training_mode": "one pooled seven-subject NBM per seed",
        "scaling": "per-subject train-only median/IQR before pooling",
        "pool_weighting": "per-window",
        "window_counts": window_counts,
        "selected_score": selected_score,
        "s3_weights": weights,
        "selection_split": "all seven subjects combined external validation",
        "threshold_reference": "combined external validation Non-FoG p95 per seed",
        "test_fog_used_for_selection": False,
        "score_candidates": candidates,
        "pooled_validation_summary": summary_metrics(selected_validation),
        "pooled_test_summary": summary_metrics(pooled_test_rows),
        "pooled_usable_run_count": sum(bool(row["usable_run_pass"]) for row in pooled_test_rows),
        "pooled_strong_run_count": sum(bool(row["strong_run_pass"]) for row in pooled_test_rows),
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
    legacy_a5.write_json(root / "A5_50_pooled7_gate.json", gate)
    legacy_a5.plot_candidate_comparison(root, candidates)
    legacy_a5.plot_subject_metrics(root, summaries)
    legacy_a5.plot_gate_matrix(root, summaries)
    if tuple(seeds) == tuple(SEEDS):
        legacy_a5.plot_curves(root, selected_subject_arrays)
    render_report(root, gate, training_rows)
    print(f"COMPLETE pooled7 status={status} selected={selected_score} results={root}", flush=True)


if __name__ == "__main__":
    main()
