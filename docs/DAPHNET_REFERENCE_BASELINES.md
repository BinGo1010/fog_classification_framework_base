# Daphnet 四类参考基线：5-seed 论文复现实验

本套件在统一的窗口、LOSO 划分、验证集模型选择、阈值和评价程序下比较：

1. Freeze Index；
2. 时频人工特征 + RBF-SVM；
3. 相同时频人工特征 + Random Forest；
4. 原始加速度序列 + CNN-GRU。

正式种子固定为：

```text
3407, 3408, 3409, 3410, 3411
```

声明式协议位于
`configs/daphnet_reference_baselines_5seed.json`。运行产生的 `config.json`、
环境文件、实现文件哈希、预测文件和审计报告共同构成结果的可追溯证据。

## 1. 统一实验协议

### 1.1 数据与被试

- Daphnet 为踝部、大腿和躯干三个三轴加速度计位置，共 9 个加速度通道；
- 采样率 64 Hz；
- S04 和 S10 没有 FoG，正式二分类比较在窗口化前将二者完全排除；
- 外层 LOSO 被试固定为：
  `S01,S02,S03,S05,S06,S07,S08,S09`；
- 同一被试的所有 run 和连续 segment 始终属于同一个 split。

### 1.2 因果输入与标签

- 输入为决策时刻之前的 4 s 原始加速度历史，即 256 点；
- 目标标签取末端 0.5 s；
- 该 0.5 s 内 FoG 样本比例至少为 0.5 时记为 FoG；
- 窗口步长 0.25 s；
- 窗口不跨 record、无效样本或采样断点；
- 四种方法和所有种子共用相同的 test anchors 与 `y_true`。

### 1.3 训练、验证和测试

每个外层 fold：

1. 一名完整被试作为 test；
2. 从剩余被试中按固定规则选择一名完整被试作为 validation；
3. 其余六名被试训练；
4. scaler、特征标准化和类别权重只使用训练被试；
5. SVM/RF 超参数和 CNN-GRU checkpoint 只按 validation PR-AUC 选择；
6. 最终分类阈值只在 validation 上最大化 Balanced Accuracy；
7. test 不参与任何模型、超参数、阈值或后处理选择。

禁止将重叠窗口随机拆分到 train/test。

## 2. 四组实验大纲

### B1. Freeze Index

**类别：** 传统频段阈值。

**目的：** 检验提出方法是否优于经典冻结频段指标。

**输入：** 默认只用 `ankle_acc_vertical`，不使用 fold scaler。

对每个 4 s 窗去均值，不加 taper、不补零：

```text
locomotor power = Σ|FFT(x-mean(x))|², 0.5 ≤ f < 3 Hz
freeze power    = Σ|FFT(x-mean(x))|², 3 ≤ f ≤ 8 Hz
FI              = freeze power / (locomotor power + ε)
score           = FI / (1 + FI)
```

连续 `score` 用于 PR-AUC/AUROC；阈值由 validation BA 选择。FI 为确定性
方法，5 个种子的结果应逐元素相同。重复运行用于审计复现性，不能当作 5 个
独立统计样本。

### B2. 时频特征 + SVM

**类别：** 传统人工特征机器学习。

**目的：** 检验提出方法是否优于经典 FoG 人工特征路线。

输入先用训练被试的 valid non-FoG 样本拟合 median/IQR scaler。每个物理
通道和三轴模长提取：

- mean、std、RMS、min、max、peak-to-peak、median、IQR、MAD；
- absolute mean、zero-crossing rate、derivative RMS、skewness、kurtosis；
- 0.5–3、3–8、8–15 和 0.5–15 Hz 功率及相对功率；
- log-FI、dominant frequency、spectral centroid、spectral entropy；
- 同一传感器轴间相关性和不同传感器同轴相关性。

九通道配置共产生 306 个固定特征。模型为：

```text
StandardScaler
RBF-SVC
class_weight = balanced
C ∈ {0.1, 1, 10}
gamma = scale
```

`C` 仅由 validation PR-AUC 选择。

### B3. 时频特征 + Random Forest

**类别：** 非线性人工特征集成模型。

**目的：** 检验提出方法是否优于基于相同人工特征的树集成路线。

RF 必须读取与 SVM 逐元素相同的 306 维特征，只允许分类器不同：

```text
n_estimators = 500
criterion = gini
bootstrap = true
class_weight = balanced_subsample
max_depth = None
max_features = sqrt
min_samples_leaf ∈ {1, 2, 5}
```

叶节点候选仅由 validation PR-AUC 选择；`random_state` 使用当前正式 seed。
输出保存模型、搜索结果、特征 schema 和 feature importance。

### B4. CNN-GRU

**类别：** 通用深度时序模型。

**目的：** 检验提出方法是否优于卷积与循环联合建模的通用深度路线。

输入形状为 `[batch, 9, 256]`。默认网络：

```text
Conv1d(9→32, kernel=7) + BN + GELU + MaxPool
Conv1d(32→64, kernel=5) + BN + GELU + MaxPool
unidirectional GRU(hidden=64)
temporal mean/max pooling
MLP binary head
```

训练配置：

```text
AdamW
learning rate = 1e-3
weight decay = 1e-4
batch size = 256
maximum epochs = 50
patience = 10
validation metric = PR-AUC
```

损失为 train-only 类别权重的 BCE，其中
`pos_weight=min(sqrt(N_nonFoG/N_FoG),6)`；梯度范数裁剪为 5。程序支持 AMP、
确定性 seed 和 epoch-boundary 断点续训。

## 3. 指标与统计

### 3.1 窗口级论文主表

```text
PR-AUC
ΔPR-AUC [95% CI]
Balanced Accuracy
Macro-F1
AUROC
FoG Sensitivity/Recall
Specificity
FoG Precision
FoG F1
```

主指标为 held-out-subject macro PR-AUC，不以 Accuracy 或 pooled-window
PR-AUC 选择模型。

### 3.2 事件级表

```text
Event Sensitivity
FA/h
Median Detection Delay
```

事件定义版本为 `coverage_aware.v2`：

- 至少连续两个阳性窗口形成候选事件；
- 间隔不超过 0.5 s 的候选事件可合并；
- 无效或未评估的真实时间间隔不能被数组相邻关系跨越；
- FA/h 分母为实际被评估、有效的 non-FoG target coverage 小时数；
- delay 从真实 FoG 起点到首次满足报警条件的时刻计算，早于起点的覆盖记 0。

所有最终对照方法和 Proposed 必须使用同一事件定义。历史输出若使用旧定义，
必须由保存的 predictions 重新计算后才能进入同一事件表。

### 3.3 五种子统计顺序

不能把 `8 subjects × 5 seeds = 40` 当作 40 个独立受试者。对每项指标先在
每名被试内平均 5 个 seed：

\[
\bar m_s=\frac{1}{5}\sum_{r=1}^{5}m_{s,r},
\]

再对八个 \(\bar m_s\) 计算 subject-macro mean 和 SD。

若以 Proposed 为参考：

\[
\Delta_s =
\overline{\mathrm{PR}}_{\mathrm{Proposed},s}
-
\overline{\mathrm{PR}}_{\mathrm{Baseline},s}.
\]

正值表示 Proposed 更好。对 8 个配对被试差值进行 100,000 次 subject-level
bootstrap，报告均值差和 95% percentile CI。Proposed 也应使用相同五个 seed；
否则只能做非 seed-matched 的补充比较。

聚合器接受可选参考 CSV：

```csv
seed,test_subject,pr_auc
3407,S01,0.50
...
```

没有参考 CSV 时，主表的 Δ 列明确输出 `NA (reference required)`，不会擅自
把某个 baseline 当作参考。同时输出四基线之间的全部配对 PR-AUC CI。

## 4. 正式服务器运行

在仓库根目录运行：

```bash
python -u scripts/run_fog_baseline_seed_sweep.py \
  --dataset-adapter daphnet \
  --data-dir "/path/to/Daphnet/processed" \
  --output-dir "$PWD/outputs/daphnet_reference_baselines_h4s_5seed" \
  --seeds 3407,3408,3409,3410,3411 \
  --launcher multigpu \
  --gpus 0-6 \
  --audit \
  --rf-n-jobs 1 \
  --batch-size 256 \
  --num-workers 0
```

每个 seed 使用独立目录：

```text
outputs/daphnet_reference_baselines_h4s_5seed/
  seed_3407/
  seed_3408/
  seed_3409/
  seed_3410/
  seed_3411/
  fold_seed_metrics.csv
  subject_seed_averaged_metrics.csv
  pairwise_pr_auc_deltas.csv
  publication_table.csv
  event_metrics_table.csv
  aggregate_multiseed_metrics.json
  multiseed_audit_report.json
```

若 Proposed 的 PR-AUC CSV 已准备好：

```bash
python -u scripts/aggregate_fog_baseline_multiseed.py \
  --output-dir "$PWD/outputs/daphnet_reference_baselines_h4s_5seed" \
  --seeds 3407,3408,3409,3410,3411 \
  --reference-pr-csv "/path/to/proposed_subject_seed_pr.csv"
```

## 5. 单折 smoke test

Smoke test 只验证软件链路，不进入论文：

```bash
python -u scripts/run_daphnet_baseline_suite.py \
  --dataset-adapter daphnet \
  --data-dir "/path/to/Daphnet/processed" \
  --output-dir "$PWD/outputs/baseline_smoke" \
  --folds S01 \
  --seed 3407 \
  --device cpu \
  --no-amp \
  --classifier-epochs 1 \
  --classifier-patience 1 \
  --max-train-windows 256 \
  --svm-c-grid 1 \
  --rf-n-estimators 20 \
  --rf-min-samples-leaf-grid 1
```

审计：

```bash
python -u scripts/audit_daphnet_baseline_suite.py \
  --data-dir "/path/to/Daphnet/processed" \
  --output-dir "$PWD/outputs/baseline_smoke"
```

## 6. 私有数据接入

使用：

```text
--dataset-adapter manifest_npz
```

处理目录契约：

```text
private_processed/
  schema.json
  manifest.csv
  records/
    record_001.npz
```

每个 NPZ 严格包含：

```text
x          float32 [time, channel]
y_binary   int8    [time], 0=non-FoG, 1=FoG
```

`manifest.csv` 至少包含：

```text
record_path,record_id,subject_id,run_id,n_samples,sampling_rate_hz,usable
```

`subject_id` 是 LOSO 分组键。标签映射、单位换算、重采样、通道重排和无效区间
必须在数据适配器/预处理阶段完成，训练代码只读取统一的二分类记录。若私有数据
没有 `ankle_acc_vertical` 或其他带 `vertical` 的通道，应显式指定
`--fi-channels <channel_name>`。

私有数据直接运行示例：

```bash
python -u scripts/run_fog_baseline_seed_sweep.py \
  --dataset-adapter manifest_npz \
  --data-dir "/path/to/private_processed" \
  --output-dir "$PWD/outputs/private_reference_baselines_h4s_5seed" \
  --exclude-subjects "" \
  --launcher direct \
  --device cuda \
  --fi-channels "waist_acc_vertical"
```

当前通用 adapter 要求保留至少三名被试，以形成 train/validation/test
subject-level 划分。正式评价还要求每个 test subject 同时具有 FoG 和 non-FoG，
否则 PR-AUC、AUROC 等二分类指标无法定义。
