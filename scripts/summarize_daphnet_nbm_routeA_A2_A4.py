from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "outputs" / "daphnet_nbm_routeA_A2_A4_v1" / "routeA_A2_A4"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def f(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def md_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return lines


def main() -> None:
    a2 = load_json(RESULT / "A2_denoising" / "A2_gate.json")
    a3 = load_json(RESULT / "A3_residual_calibration" / "A3_gate.json")
    a4 = load_json(RESULT / "A4_representation_ablation" / "A4_gate.json")
    a4_test = pd.read_csv(RESULT / "A4_representation_ablation" / "test_metrics_after_freeze.csv")
    training = pd.read_csv(RESULT / "A2_denoising" / "training_summary.csv")
    subject = a4_test.groupby("subject_id", sort=True).median(numeric_only=True)

    lines = [
        "# Daphnet NBM Route A：A2–A4 完整实验结果",
        "",
        f"生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 最终结论",
        "",
        f"1. A2 为 **{a2['status']}**，保留 `{a2['selected_scheme']}`。D1–D3 中没有方案同时满足全部去噪门控。",
        f"2. A3 为 **{a3['status']}**，选中 `{a3['selected_scheme']}`；残差校准没有稳定优于原始残差。",
        f"3. A4 为 **{a4['status']}**，选中 `{a4['selected_representation']}`（`[R, |R|, ΔR]`，27 通道）。",
        f"4. R5 冻结后测试中位 AUROC={f(a4['test_summary_after_freeze']['median_auroc'])}，"
        f"AP={f(a4['test_summary_after_freeze']['median_average_precision'])}，"
        f"Cliff's δ={f(a4['test_summary_after_freeze']['median_cliffs_delta'])}。",
        "5. 已按授权在 A4 后停止，未进入 A5；测试 FoG 从未参与 A2–A4 结构选择。",
        "",
        "## A2 去噪消融门控",
        "",
    ]
    a2_rows: list[list[object]] = []
    for row in a2["schemes"]:
        a2_rows.append([
            row["scheme"], f"{row['subject_clean_preservation_count']}/5", f"{row['corruption_win_count']}/5",
            f(row["median_clean_nrmse"]), f(row["median_clean_pearson"]),
            f(row["median_clean_nrmse_p90"]), "PASS" if row["gate_pass"] else "FAIL",
        ])
    lines += md_table(["方案", "clean 保留被试", "优于 D0 的扰动", "clean NRMSE", "clean Pearson", "clean P90", "门控"], a2_rows)
    lines += [
        "",
        "D1 是最接近门控的候选：聚合 clean 指标略优于 D0，且 4/5 个扰动条件相对 D0 有改善；"
        "但仅 3/5 被试满足逐被试 clean 保留要求。S01 与 S08 的 clean NRMSE P90 分别由 "
        "0.6401→0.6420、0.7631→0.7814，因此按“P90 不得劣于 D0”的严格规则失败。"
        "D2、D3 在 5 名被试上均未通过逐被试 clean 保留，不能部署。",
        "",
        "## A3 残差校准",
        "",
    ]
    a3_rows: list[list[object]] = []
    for name in ("C0_clipnone", "C1_clipnone", "C2_clipnone"):
        row = next(value for value in a3["schemes"] if value["scheme"] == name)
        a3_rows.append([name, f(row["median_validation_auroc"]), f(row["median_validation_fog_nonfog_ratio"]),
                        f(row["median_validation_cliffs_delta"]), f(row["median_validation_nonfog_p95"]),
                        "PASS" if row["gate_pass"] else "FAIL"])
    lines += md_table(["方案", "验证 AUROC", "FoG/NF 比", "Cliff's δ", "NF P95", "门控"], a3_rows)
    lines += [
        "",
        f"选中 C0/不裁剪。其验证中位 AUROC={f(next(x for x in a3['schemes'] if x['scheme']=='C0_clipnone')['median_validation_auroc'])}；"
        "C1/C2 没有带来有意义的排序提升，裁剪版本还出现较高饱和率，因此保持最简单、最可解释的原始有符号残差。",
        "",
        "## A4 表征消融（验证集选型）",
        "",
    ]
    rep_rows = [[row["representation"], f(row["median_validation_auroc"]),
                 f(row["median_validation_average_precision"]), f(row["median_validation_cliffs_delta"]),
                 f(row["median_validation_fog_nonfog_ratio"]), f(row["median_validation_false_alarm_per_minute_proxy"])]
                for row in a4["representations"]]
    lines += md_table(["表征", "AUROC", "AP", "Cliff's δ", "FoG/NF 比", "FA/min 代理"], rep_rows)
    lines += [
        "",
        "R5 的验证 AUROC（0.8882）和 Cliff's δ（0.7763）最高，因此冻结为最终表征；"
        "但它相对 R0 的提升很小（AUROC 0.8847→0.8882），应视为增量改进而非结构性突破。",
        "",
        "## 冻结后逐被试测试结果（3 种子中位数）",
        "",
    ]
    subject_rows: list[list[object]] = []
    for subject_id, row in subject.iterrows():
        fog_windows = int(a4_test[a4_test.subject_id == subject_id]["test_fog_windows"].iloc[0])
        nf_windows = int(a4_test[a4_test.subject_id == subject_id]["test_nonfog_windows"].iloc[0])
        subject_rows.append([subject_id, nf_windows, fog_windows, f(row.auroc), f(row.average_precision),
                             f(row.cliffs_delta), f(row.fog_to_nonfog_median_ratio)])
    lines += md_table(["被试", "NF 窗口", "FoG 窗口", "AUROC", "AP", "Cliff's δ", "FoG/NF 比"], subject_rows)
    lines += [
        "",
        "跨被试差异明显：S09 最强（AUROC 0.9424），S07 次之（0.8296）；S08 最弱（0.5794），"
        "说明最终表征尚不能视为对所有被试稳定泛化。S08 是后续修复的首要对象。",
        "",
        "## 审计与边界",
        "",
        f"- A2 新训练检查点：{len(list((RESULT / 'A2_denoising').rglob('best_model.pt')))} 个；"
        f"含 D0 复用在内的完整运行指标：{len(list((RESULT / 'A2_denoising').rglob('metrics_all_conditions.json')))} 份。",
        f"- A2 新训练累计运行时间约 {training['elapsed_seconds'].fillna(0).sum()/3600:.2f} 小时（设备切换与系统并行负载会影响墙钟时间）。",
        "- A3/A4 只用 S01、S05、S08、S09 的验证 FoG/Non-FoG 选型；S07 验证 FoG 仅 1 窗，"
        "预先作为稀疏诊断被试，不参与排名，但其冻结后测试结果完整报告。",
        "- S09 测试 FoG 仅 5 窗，S07 为 28 窗；这些被试级结果的不确定性较高。3 个训练种子共享同一测试窗口，"
        "不能当作 3 个独立生物学重复。",
        "- 高斯扰动（标准差 0.01/0.03）相对当前标准化重建误差很小，绝对“恢复改善率”会出现很大的负值；"
        "因此 A2 主要解释相对 D0 的胜出条件数和 clean 保留门控，不将该绝对百分比作临床解释。",
        "- 所有测试 FoG 指标均在 A3/A4 方案冻结后计算，`test_fog_used_for_selection=false`。",
        "",
        "## 结果文件",
        "",
        "- `A2_denoising/A2_gate.json`：A2 逐方案、逐被试和逐扰动门控。",
        "- `A3_residual_calibration/A3_gate.json`：A3 选型和冻结后测试摘要。",
        "- `A4_representation_ablation/A4_gate.json`：A4 选型和冻结后测试摘要。",
        "- `A4_representation_ablation/test_metrics_after_freeze.csv`：最终逐被试逐种子测试指标。",
        "- `FINAL_RESULTS.json`：机器可读的完整阶段结果。",
    ]
    report = RESULT / "reports" / "A2_A4_complete_results.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(report)


if __name__ == "__main__":
    main()
