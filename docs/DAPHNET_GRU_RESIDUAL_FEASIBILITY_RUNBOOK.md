# Daphnet GRU-H200 残差融合实验运行手册

本手册对应四阶段实验大纲：Phase 0 正常预测诊断、Phase 1 单折工程
smoke、Phase 2 八折最小消融，以及 Phase 3A/3B subject-level
cross-fitting 确认。

主入口为：

```text
scripts/run_daphnet_gru_residual_feasibility.py
```

实现不会改写已有的 H200 源实验目录：

```text
outputs/daphnet_gru_horizon4_h4_tcnm_loso_seed42/
```

所有新 checkpoint、cache、预测和审计清单写入单独的输出目录。

## 1. 前置条件

默认要求以下目录已经存在：

```text
dataset/1.Daphnet Freezing of Gait Dataset/processed/
outputs/daphnet_gru_horizon4_h4_tcnm_loso_seed42/
```

查看完整参数：

```powershell
python scripts/run_daphnet_gru_residual_feasibility.py --help
```

默认 `--phase 0`，避免误启动 Phase 3B 的 48 个 inner NBM。GPU 可用时
`--device auto` 会自动使用 CUDA。

## 2. 推荐执行顺序

下列命令使用同一个输出目录。恢复运行时，所有会进入 protocol fingerprint 的
科学参数必须保持一致。

### Phase 0：H200 重放与正常预测诊断

```powershell
python scripts/run_daphnet_gru_residual_feasibility.py `
  --phase 0 `
  --folds all `
  --output-dir outputs/daphnet_gru_h200_residual_feasibility_seed42
```

每折会输出：

- `raw/mu/sigma/error/z/log_sigma` primitive cache；
- clean non-FOG 的 NLL、RMSE、MAE、coverage、lead-quartile、lag 和频带误差；
- 同窗口 Persistence 对照；
- 固定选窗的腰部三轴 `raw + mu±2sigma / error / z / spectrogram` 图；
- identity、shape、终端 0.5 s 标签、裁剪率和数值爆炸硬检查。

正式八折 gate 要求所有硬检查通过，并且 GRU 在至少 5/8 个 validation
subjects 上取得更低 RMSE。子集运行只标记为 `subset_only`，不能触发正式
Go。

### Phase 1：S01 工程 smoke

```powershell
python scripts/run_daphnet_gru_residual_feasibility.py `
  --phase 1 `
  --folds all `
  --output-dir outputs/daphnet_gru_h200_residual_feasibility_seed42
```

只训练 S01 折的四个 arm。默认最多使用 10,000 个训练窗口、5,000 个
validation/test 窗口和 3 个 epoch。smoke gate 检查 loss 有限且下降、所有
endpoint/label 一致，以及 Zero/Fusion 的参数量和初始权重 hash 相同。该阶段
指标不会进入正式科学结果。

### Phase 2：单种子八折最小消融

```powershell
python scripts/run_daphnet_gru_residual_feasibility.py `
  --phase 2 `
  --folds all `
  --output-dir outputs/daphnet_gru_h200_residual_feasibility_seed42
```

运行五个 arm：

```text
raw4
raw6
normality (z4 + log_sigma4)
raw4_zero
raw4_normality
```

`phase2/gate.json` 使用八名受试者的配对结果执行预注册 Strong Go、
Conditional Go 或 Stop。Conditional Go 还要求 Recall、Event Sensitivity 或
FA/h 至少一个二级指标在多数受试者上同方向改善，不能仅凭很小的 PR-AUC
差值自动通过。

### Phase 3A：三个固定外层折的 cross-fitting

```powershell
python scripts/run_daphnet_gru_residual_feasibility.py `
  --phase 3a `
  --folds all `
  --output-dir outputs/daphnet_gru_h200_residual_feasibility_seed42
```

固定 outer test subjects 为 S01、S05、S08。每个 outer fold 的六名训练受试者
拆成 3 个 inner folds：4 人训练 NBM、2 人生成 OOF 表征。validation/test
由三个 inner predictor 在物理单位做 Gaussian moment matching，再转回共同的
outer scaler。

分类只运行 `raw6`、`raw4_zero`、`raw4_normality`。Phase 3A gate 要求两个
PR-AUC 比较均为正方向、每个比较至少 2/3 subjects 不反转，并且融合模型的
subject-macro FA/h 不超过 Raw6 的 1.2 倍。

### Phase 3B：完整 cross-fitting 与多分类器种子

```powershell
python scripts/run_daphnet_gru_residual_feasibility.py `
  --phase 3b `
  --folds all `
  --output-dir outputs/daphnet_gru_h200_residual_feasibility_seed42
```

每个 outer fold 使用 6 个 leave-one-training-subject-out inner NBM。默认
NBM seed 为 42，classifier seeds 为 42、43、44。聚合时先在每名测试受试者
内部平均重复种子，再对八名受试者进行 paired bootstrap；种子不会被当成独立
受试者。

Phase 3B 默认还会对完整数据中的 S04、S10 执行 negative-only 外部误报评估。
六个 inner GRU 的预测先在物理单位集成，随后使用冻结的 outer scaler、分类器
checkpoint 和 validation threshold；外部标签不会参与训练或阈值选择。输出仅含
在全阴性数据上有定义的 specificity、positive-window rate 和 FA events/hour，
不伪造 AUROC、AUPRC、Recall 或 F1。完整时间线位于：

```text
phase3b/external_negative_only/subject_averaged_timeline.csv
```

`--no-phase3-external-negative-only` 只用于调试；禁用后 Phase 3B 的正式 decision
会标记为失败，因为四阶段大纲尚未完整执行。

## 3. 一条命令执行完整分层协议

```powershell
python scripts/run_daphnet_gru_residual_feasibility.py `
  --phase all `
  --folds all `
  --output-dir outputs/daphnet_gru_h200_residual_feasibility_seed42
```

`all` 会依次执行 0 → 1 → 2 → 3A → 3B，并在 Phase 0、Phase 1、Phase 2、
Phase 3A 的 gate 失败时停止。`--force-next-phase` 可用于明确的工程诊断，但该
覆盖会记录在 `status.json` 中，不应作为正式预注册结果。

Phase 3 另有 OOF-single 与 validation/test ensemble 的表征连续性硬门，默认
检查：

```text
0.5 <= z_std(eval) / z_std(OOF train) <= 2.0
abs(median(log_sigma_eval) - median(log_sigma_train)) <= log(2)
abs(z_clip_rate_eval - z_clip_rate_train) <= 0.05
```

## 4. 恢复与故障注入

默认启用 `--resume`。NBM 和 classifier 都在 epoch 边界保存：

- model；
- optimizer；
- AMP GradScaler；
- Python、NumPy、CPU/CUDA RNG；
- history、best epoch 和 early-stop 状态。

调试断点恢复可使用：

```powershell
--debug-interrupt-nbm-after-epoch 1
```

所有完成任务都有 `DONE.json`，其中记录 artifact 路径、字节数和 SHA-256。
`--finalize-only` 只允许从完整 checkpoint/cache 汇总，不会补训缺失任务。

## 5. 主要输出目录

```text
config.json
status.json
loso_SXX/h200_primitives/
phase0/loso_SXX/
phase1/loso_S01/<arm>/
phase2/loso_SXX/<arm>/
phase3a/loso_SXX/nbm_seed_*/inner_models/
phase3a/loso_SXX/nbm_seed_*/crossfit/
phase3b/loso_SXX/nbm_seed_*/inner_models/
phase3b/loso_SXX/nbm_seed_*/crossfit/
phase3*/loso_SXX/nbm_seed_*/classifier_seed_*/<arm>/
phase3b/external_negative_only/
```

每个分类 cell 保存 validation-only threshold、`predictions.csv/.npz`、逐 epoch
history、模型结构、参数量、初始权重 hash 和指标。Phase 2/3 的 aggregate
始终以 subject 为统计单位。

完整运行后执行独立只读审计：

```powershell
python scripts/audit_daphnet_gru_residual_feasibility.py `
  --result-dir outputs/daphnet_gru_h200_residual_feasibility_seed42
```

审计器重新检查所有 DONE 文件的 size/SHA-256、阶段矩阵、endpoint/label 一致性、
inner checkpoint→forecast→crossfit 血缘、OOF subject 排除和种子聚合策略，并写出
`audit_report.json`。`--allow-incomplete` 只允许尚未完成的矩阵，不能放过已经存在
但被篡改的 artifact。

生成论文式汇总表和图（允许尚未跑完的目录，并明确列出缺失面板）：

```powershell
python scripts/report_daphnet_gru_residual_feasibility.py `
  --result-dir outputs/daphnet_gru_h200_residual_feasibility_seed42
```

默认输出到 `feasibility_report/`，包括 `REPORT.md`、长表
`publication_tables.csv`、图件以及记录全部输入/输出 SHA-256 的
`report_manifest.json`。报告器只读取小型 metrics、CSV 和分类预测文件，不加载
H200 primitive cache；受试者始终是推断统计的独立单位。

## 6. 结果解释边界

- Phase 1 只证明工程链可运行。
- Phase 2 的 outer-train residual 是 in-sample，只能做方向性筛查。
- Phase 3 才排除 outer-train residual 的 subject-level in-sample 偏差。
- 2 s 是正常轨迹 forecast horizon；分类结果在实际 target 到达后产生，因此
  是因果 FOG 检测，不是“提前 2 s 预测 FOG”。
- `z` 是模型创新量，不是纯 FOG 信号；最终比较保留 Raw 分支和 Zero 容量控制。
