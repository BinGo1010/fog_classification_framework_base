# Daphnet 2 s 因果 GRU 正常预测与残差融合：快速可行性实验大纲

## 1. 实验目的

本实验不追求一次完成最终论文结论，而是用最小但公平的实验回答四个问题：

1. 2 s GRU-NBM 是否能在未见受试者的 clean non-FOG 上给出可用且校准合理的
   正常预测分布？
2. H200 标准化残差 \(z=(x-\mu)/\sigma\) 是否包含稳定的 FOG 判别信息？
3. 在保留 Raw 的条件下，加入 \(z\) 和 \(\log\sigma\) 是否比“只增加模型容量”
   更有价值？
4. 方向性结果在 subject-level cross-fitting 后是否仍然存在？

快速实验遵循：

> 先复用现有 H200 检查点验证表示价值；只有结果达到预设 Go 条件，才投入
> cross-fitting、多种子和时频扩展。

完整方法定义见：

- [2 s 因果 GRU 残差融合架构](DAPHNET_GRU_2S_CAUSAL_RESIDUAL_ARCHITECTURE.md)
- [现有 GRU horizon 协议](DAPHNET_GRU_HORIZON_ABLATION.md)
- [四阶段实现与运行手册](DAPHNET_GRU_RESIDUAL_FEASIBILITY_RUNBOOK.md)

## 2. 已有证据与本次新增问题

仓库已有审计通过的 H200：

- context：2 s；
- forecast horizon：2 s；
- residual history：两个非重叠 H200 block，共 4 s；
- 分类 stride：0.25 s；
- 标签：共同终点前最后 0.5 s；
- subject-macro PR-AUC：0.5068；
- Event Sensitivity：0.8497；
- FA/h：150.462。

因此“GRU-H200 residual-only 可以完成训练和检测”已经得到工程验证。本次无需
先重复完整 horizon sweep；真正需要快速验证的是：

```text
Raw + normality 是否优于 Raw
normality 是否优于同容量 Zero control
该结论是否能承受 out-of-subject residual
```

现有 H200 产物可复用：

```text
outputs/daphnet_gru_horizon4_h4_tcnm_loso_seed42/
```

其中每折已有：

- H200 GRU-NBM `best.pt`；
- split、scaler 和共同 history support；
- residual cache；
- residual-only TCN-M 预测和指标。

已有的 held-out-subject 描述性分析还表明：

- residual RMS 的 subject-macro ROC-AUC 为 0.6707，FOG 中位数在 8/8 名
  受试者上更高；
- 3–8 Hz residual log-power 的 subject-macro ROC-AUC 为 0.6930，方向同样
  在 8/8 名受试者上一致。

详见 [H200 residual feature separation report](../outputs/daphnet_gru_h200_residual_feature_analysis/REPORT.md)。
这些结果只证明“残差含有判别信息”，不能证明“残差对 Raw 有增量价值”；
后者才是本实验的主要问题。

当前 H200 `residual_cache.npz` 只保存标准化 residual、标签和 window index；
融合实验仍需用保存的 `best.pt` 重放同一 WindowTable，物化：

```text
raw、mu、sigma、error、z、log_sigma、Gaussian NLL
```

阶段二快速实验可以复用 NBM，不需要重新训练阶段一。cross-fitting 确认阶段
必须重新训练 inner predictors。

## 3. 固定科学协议

### 3.1 主队列

```text
S01,S02,S03,S05,S06,S07,S08,S09
```

8 折 LOSO，每折：

```text
6 train + 1 validation + 1 test
```

验证受试者使用预注册的循环映射，不按结果挑选。S04、S10 不进入主 8 折，
在确认阶段作为 negative-only 外部误报测试。

### 3.2 时间定义

```text
sampling rate              64 Hz
NBM context                2 s / 128 点
NBM horizon                2 s / 128 点
classifier history         4 s / 256 点
H200 blocks per history    2
window/output stride       0.25 s / 16 点
label support              endpoint 前最后 0.5 s / 32 点
FOG fraction threshold     50%
```

所有输入 arm 必须共享完全相同的：

- endpoint ID；
- train/validation/test labels；
- record-local history support；
- batch order、初始化种子、训练预算；
- validation-only 早停和阈值规则。

### 3.3 训练与指标

快速阶段固定一个种子 `42`：

```text
classifier       TCN-M 或预注册的双分支 TCN
optimizer        AdamW
lr               1e-3
weight decay     1e-4
epochs           ≤ 12
patience         4
loss             weighted BCE
early stop       validation PR-AUC
```

主指标：

- held-out-subject macro PR-AUC。

必须同时报告：

- Balanced Accuracy、Macro-F1、AUROC；
- FOG Recall、Precision、F1、Specificity；
- Event Sensitivity、FA/h、Detection Delay；
- 每名受试者结果和 paired subject bootstrap 95% CI。

## 4. 实验分层

## Phase 0：无训练 sanity check

### 目的

确认 H200 正常预测、时间对齐和不确定性没有明显故障。该阶段不用于比较分类器。

### 数据

直接读取 8 折已保存的 H200 GRU checkpoint、WindowTable 和 clean non-FOG
validation windows。

### 检查项目

1. `context_end == target_start`；
2. 决策时间戳等于 target end；
3. `error == raw - mu`；
4. `z == error / sigma`；
5. 无 NaN/Inf；
6. raw、mu、error、z shape 完全一致；
7. 标签只来自终点前最后 32 点；
8. clean non-FOG context、target 和 guard 内均无 FOG；
9. 所有表示使用相同 window index。

### 正常预测诊断

在每个 validation subject 的 clean non-FOG 上报告：

- NLL、MAE、RMSE；
- 每通道指标；
- 按预测距离分层：
  - 0–0.5 s；
  - 0.5–1.0 s；
  - 1.0–1.5 s；
  - 1.5–2.0 s；
- \(z\) 的均值、标准差和分位数；
- \(\mu\pm\sigma\)、\(\mu\pm2\sigma\) 的经验覆盖率；
- residual clipping 比例；
- actual/prediction 的最优 cross-correlation lag；
- 0.5–3 Hz、3–8 Hz band-power error。

在完全相同的 clean non-FOG 窗口上加入 Persistence 对照：

\[
\mu^{Persistence}_{t:t+2s}=x_{t-1}.
\]

比较 GRU 与 Persistence 的 NLL/RMSE、远端预测误差和 band-power error。
如果 GRU 在多数验证受试者上不能改善正常预测，就不能仅凭模型更复杂而将其
称为更好的 normal-behaviour model。

### 可视化

每折固定选择，不依据预测结果挑选：

- 5 个 clean non-FOG 窗口；
- 5 个 FOG onset 窗口；
- 5 个高残差 non-FOG 窗口。

绘制：

```text
raw target
mu ± 2 sigma
signed error
z
raw/z spectrogram
```

### Phase 0 通过条件

- 所有 identity、shape、support 和 label 检查通过；
- 无非有限值；
- clean non-FOG 上 clipping 比例原则上低于 5%；
- 预测误差随 horizon 增大可以上升，但最后 0.5 s 不能出现系统性数值爆炸；
- 高残差 non-FOG 不能全部由固定时间错位或单个故障通道解释。
- GRU 应至少在 5/8 validation subjects 上优于 Persistence 的主预测指标；
  若不满足，应将“GRU 是否必要”保留为显式问题，而不是直接删除对照。

如果失败，停止分类实验，先修正 scaler、时间索引、sigma 校准或相位问题。

## Phase 1：单折工程 smoke

### 目的

只验证新表示和分类器代码能正确训练，不判断科学优劣。

### 固定折

```text
test = S01
validation = S02
train = 其余 6 人
seed = 42
```

### 缩小预算

```text
复用 H200 NBM
max train windows          10,000
max validation windows     5,000
classifier epochs          2–3
```

抽样必须按 subject/label 固定并保存 index，所有 arm 使用同一子集。

### Smoke arms

| ID | 输入 | 作用 |
|---|---|---|
| Q0 | Raw4 | 基本 raw 路径 |
| Q1 | Z4 + logσ4 | normality-only 路径 |
| Q2 | Raw4 + Zero normality branch | 融合容量控制 |
| Q3 | Raw4 + Z4 + logσ4 | 提出方法 |

### Smoke 通过条件

- 所有 arm 前向、反向和保存/恢复正常；
- loss 有限且至少发生下降；
- validation/test 只执行 inference；
- Q2/Q3 架构和参数量完全一致；
- 四个 arm 的 endpoint ID 和标签逐元素一致；
- 固定种子重复运行得到一致的 window IDs 和初始权重 hash。

Smoke 指标不能进入最终结果表。

## Phase 2：单种子 8-fold 最小科学消融

### 目的

快速判断 normality 表征是否值得进入严格 cross-fitting。

### 实验矩阵

| ID | 输入 | Shape/支持 | 科学问题 |
|---|---|---|---|
| F0 | Raw4 | `[9,256]` | 4 s 原始信号基线 |
| F1 | Raw6 | `[9,384]` | 完整 6 s 因果支持控制 |
| F2 | Z4 + logσ4 | `[18,256]` | normality 表征自身是否有判别力 |
| F3 | Raw4 + Zero branch | 与 F4 完全相同 | 双分支容量控制 |
| F4 | Raw4 + Z4 + logσ4 | 双分支 | 提出方法 |

现有审计通过的 residual-only TCN-M 只有 `Z4`，其 PR-AUC 0.5068 可作为历史
参考，但由于 F2 还包含 `logσ4`，F2 必须在共同 protocol 下重新训练。

快速阶段先不加入 NLL/STFT，避免结果不佳时无法判断原因。它们仅在 F4 达到
Go 条件后加入。

注意：现有 H200 NBM 为其外层 6 名训练受试者生成的是 in-sample residual，
而验证/测试 residual 是 out-of-subject。故 Phase 2 只能作为探索性筛查，
不能作为无泄漏的最终方法结论。

### 双分支约束

```text
Raw branch         TCN encoder
Normality branch   TCN encoder
Fusion             concat + MLP
```

F3 的 normality branch 输入全零，但结构、初始化、优化器和参数量与 F4
完全一致。两个共同主比较是：

\[
\Delta PR\text{-}AUC_{practical}=F4-F1,
\]

\[
\Delta PR\text{-}AUC_{mechanism}=F4-F3.
\]

其他比较：

\[
F4-F0
\]

\[
F2-F0
\]

其中 F1 是完整时间支持控制，回答“normality 是否优于直接给分类器更多 Raw”；
F3 是参数量/双分支控制，回答“收益是否真的来自 normality 数值”。如果只与
F0 比，无法排除 F4 仅因额外看到了更早 2 s 信号而获益。

### Phase 2 方向性 Go/No-Go

以下数值是进入下一阶段的工程决策门槛，不是显著性声明；必须在读取测试结果
前冻结。

#### Strong Go

同时满足：

- F2 的 macro PR-AUC 高于对应 FOG prevalence，且判别方向至少在 5/8 名
  测试受试者上一致；
- `F4 − F1` subject-macro PR-AUC ≥ `+0.02`；
- F4 至少在 5/8 名测试受试者上优于 F1；
- `F4 − F3 > 0`，即提出方法不低于同容量 Zero control；
- Event Sensitivity/FOG Recall 相对 F1 的绝对下降不超过 0.05；
- FA/h 不超过 F1 的 1.2 倍。

#### Conditional Go

- `F4 − F1` 的 PR-AUC 在 `0` 至 `+0.02` 之间；
- F4 仍优于 F3；
- Recall、Event Sensitivity 或 FA/h 出现有价值且跨多数受试者一致的改善；
- 改善不是只来自一个受试者。

此时进入 cross-fitting，但冻结新的 operating-point 目标，不能事后挑指标。

#### Stop / Redesign

出现任一项：

- F4 与 F1/F3 基本相同，表明 residual 没有可见的增量价值；
- `F4 − F1` 的 paired CI 上界仍低于 `+0.01`；
- F4 仅在不超过 2/8 名受试者上优于 F1；
- F4 低于 F3；
- FA/h 超过 F1 的 1.2 倍，或 Event Sensitivity/FOG Recall 下降超过 0.05；
- 收益完全由单一受试者驱动；
- residual magnitude 与预测 lag 的关系强于其与 FOG 标签的关系。

此时优先检查相位对齐、固定/dynamic sigma、H050+H200 双 horizon，而不是直接
增加分类器容量。

## Phase 3：小规模 cross-fitting 确认

### 目的

验证 Phase 2 的方向不是由 in-sample 训练 residual 过于理想造成。

### Phase 3A：快速 3 折确认

预先固定三个测试受试者：

```text
S01、S05、S08
```

不得根据 Phase 2 的好坏重新选择。

对每个外层折，在 6 名训练受试者中进行 3-fold subject cross-fitting：

```text
每次 4 人训练 NBM
2 人生成 OOF residual
共 3 个 inner NBM
validation/test 使用 3 个 inner NBM 的集成，不另训 final NBM
```

这样 OOF 与 validation/test 所用的每个 predictor 都只见过 4 名受试者，避免
用“4 人 NBM 的训练表征”匹配“6 人 final NBM 的测试表征”。每个 inner
scaler 仍只由其 4 名训练受试者拟合。集成前先把各模型的 \(\mu,\sigma^2\)
逆变换到原始物理单位，再按架构文档中的 Gaussian moment matching 合并，
最后转到共同的 outer-fold classifier scale；不能直接平均不同 scaler 空间
中的输出。

只运行三个关键 arm：

| ID | 输入 |
|---|---|
| C0 | Raw6 |
| C1 | Raw4 + Zero branch |
| C2 | Raw4 + Z4 + logσ4 |

工作量：

```text
9 个新 NBM（3 外层折 × 每折 3）
9 个 classifier cells（3 外层折 × 3 arms）
```

通过条件：

- OOF train \(z\) 与 validation/test \(z\) 的尺度没有明显断层；
- C2−C0、C2−C1 与 Phase 2 保持同方向；
- 三个测试受试者中至少两个不出现明显性能反转；
- 误报没有因 cross-fitting 大幅恶化。

### Phase 3B：完整确认

只有 Phase 3A 通过后执行：

- 8 个外层 LOSO folds；
- leave-one-training-subject-out residual；
- 每折 `6 inner NBM`，validation/test 使用 6 模型 moment-matched ensemble；
- 先固定 NBM seed=42，classifier 使用 seeds `42/43/44`；
- 主结果按“同一受试者内先平均 classifier seeds，再对 8 名受试者做 paired
  bootstrap”汇总，不能把 seed 当成独立受试者扩大样本量；
- 方向通过后，再将 NBM seed 重复作为稳定性分析；
- 关键 arm：Raw6、Raw+Zero、Raw+Z+logσ；
- S04、S10 negative-only 外部误报评估。

每个 NBM seed 共训练 `8 × 6 = 48` 个 inner NBM。若需要与原架构的
“6 人 final NBM”部署方式比较，可以把 final NBM 作为额外敏感性分析，但
不混入上述主比较。Phase 3B 才能成为正式方法可行性的主要证据。

## 5. 必须输出的结果

### 5.1 表格

1. NBM clean non-FOG 预测与校准表；
2. 每个 arm 的 subject-macro 指标；
3. 每名受试者指标；
4. paired PR-AUC delta 与 95% CI；
5. Event Sensitivity、FA/h、Delay；
6. residual clipping、lag 和 sigma 诊断；
7. 模型参数量和输入原始支持范围。

### 5.2 图

1. raw、mu±2sigma、z 示例；
2. non-FOG/FOG 的 \(z\) 分布；
3. 按未来预测距离分层的 NLL/RMSE/coverage；
4. F4−F1 与 F4−F3 的每受试者 paired waterfall plot；
5. pooled PR curve 作为辅助图；
6. FA/h 与 Event Sensitivity trade-off；
7. phase lag 与 residual energy/false positive 的关系；
8. S04、S10 的完整预测时间线。

### 5.3 审计产物

每折保存：

```text
split subjects
scaler fit subjects
window IDs and labels
context/target/endpoint timestamps
NBM checkpoint hash
representation cache hash
classifier initialization hash
predictions.csv / predictions.npz
validation threshold
metrics.json
DONE manifest
```

根目录保存 protocol fingerprint、完成矩阵、aggregate tables 和独立 audit
report。

## 6. 建议的执行顺序

```text
Phase 0：重放现有 H200 checkpoint，完成预测/残差诊断
        ↓
Phase 1：S01 单折 smoke，验证新融合代码
        ↓
Phase 2：复用 H200 NBM，单种子 5 arms × 8 folds
        ↓
达到 Go/Conditional Go？
        ├─ 否：检查 lag、sigma、H050+H200
        └─ 是
             ↓
Phase 3A：3 个固定外层折，3-fold subject cross-fitting
             ↓
方向保持？
        ├─ 否：停止扩大实验
        └─ 是
             ↓
Phase 3B：完整 cross-fitting × 8 folds × 多种子
```

### 快速阶段计算单元

| Phase | 新 NBM | classifier cells | 说明 |
|---|---:|---:|---|
| Phase 0 | 0 | 0 | 重放现有 checkpoint |
| Phase 1 | 0 | 4 | 单折、缩小窗口和 epoch |
| Phase 2 | 0 | 40 | 5 arms × 8 folds，全部按共同协议比较 |
| Phase 3A | 9 | 9/分类器 seed | 3 个外层折、每折 3 个 inner NBM |
| Phase 3B | 48/NBM seed | 24/分类器 seed | 8 折、每折 6 个 inner NBM、3 个关键 arms |

## 7. 快速实验可以与不能支持的结论

Phase 2 可以支持：

- H200 normality 表征是否显示方向性增量价值；
- 是否值得投入 cross-fitting 和多种子计算；
- 主要失败模式是排序、误报还是受试者异质性。

Phase 2 不能支持：

- 残差融合稳定优于 raw；
- 跨受试者泄漏已经排除；
- 2 s horizon 优于其他 horizon；
- 方法可部署于真实生活场景；
- 对 S04/S10 等无 FOG 患者具有可接受误报率。

正式方法结论必须建立在 Phase 3B，而不是 smoke、单折或 pooled-window 指标上。
