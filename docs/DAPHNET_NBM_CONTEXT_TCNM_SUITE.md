# Daphnet NBM 上下文长度 × TCN-M 严格对照实验

## 1. 实验目的

本实验用于回答一个单一问题：在下游分类器和训练协议完全固定时，正常行为模型（NBM）的类型及其可用历史上下文长度，是否会改变 FoG 残差表示的诊断能力。

固定的数据流为：

```text
历史 IMU 上下文
  → NBM 预测未来 0.5 s 的正常均值 μ 和不确定度 σ
  → 标准化残差 r = (实际 IMU - μ) / σ
  → 最近 4 s 残差序列 [N, 9, 256]
  → 固定 TCN-M
  → FoG / non-FoG
```

NBM 只使用 clean-normal 窗口训练，但训练完成后会对 train、validation 和 held-out test 的全部有效窗口生成条件正常预测。Persistence 不属于本实验矩阵，因为这里比较的是四种可学习 NBM。

## 2. 固定科学协议

| 项目 | 固定设置 |
|---|---|
| 数据 | Daphnet，ankle/thigh/trunk 三个 IMU，共 9 个加速度通道 |
| 采样率 | 64 Hz |
| 排除受试者 | S04、S10 |
| LOSO 测试受试者 | S01、S02、S03、S05、S06、S07、S08、S09 |
| 预测目标 | 紧随上下文之后的未来 0.5 s，即 32 点 |
| 窗口步长 | 0.25 s，即 16 点 |
| FoG 窗口标签 | 最终 0.5 s 目标块中 FoG 占比至少为 0.5 |
| NBM 训练样本 | 仅 clean-normal 窗口；所有上下文共享同一套最大 4 s 支持资格 |
| 残差 | `(target - μ) / σ`，截断至 `[-12, 12]` |
| 分类器输入 | `residual_h4s`，形状 `[N, 9, 256]` |
| 分类器 | 固定 TCN-M |
| 主随机种子 | 42 |
| 重复性 | deterministic 开启，AMP 开启 |

每个 4 s 分类输入由 8 个按时间排序、互不重叠的 0.5 s 残差块拼接而成。标签取最后一个残差块对应的窗口标签，因此历史信息只用于当前窗口判别，不引入未来信息。

每个残差块都由它前方对应长度的上下文独立预测。因此从原始 IMU 覆盖范围看，C1–C4 的 4 s 残差历史分别最多依赖约 5/6/7/8 s 的过去信号；这正是 context 消融所改变的信息范围，所有计算仍严格因果。

## 3. 公共 4 s 基窗与右对齐

四种上下文不能分别独立生成窗口，否则短上下文会得到更多靠近记录起点的样本，比较结果会同时混入“样本集合变化”。本实验先按最大上下文建立唯一的公共 `WindowTable`：

```text
公共支持区间（256 点 / 4 s）                 共同目标（32 点 / 0.5 s）
[ t-256 ................................ t ) | [ t ............ t+32 )
```

在完全相同的目标边界 `t` 上，从公共支持区间末端截取不同长度：

```text
C1: [t-64,  t)  → 1 s
C2: [t-128, t)  → 2 s
C3: [t-192, t)  → 3 s
C4: [t-256, t)  → 4 s
```

因此 C1–C4 具有相同的：

- 目标 IMU 块、窗口标签和记录位置；
- train/validation/test 划分；
- clean-normal NBM 训练资格和 0.5 s normal guard；
- 归一化器、4 s residual-history anchor 及分类样本数量。

上下文长度是这些单元格之间的主要科学变量，而不是窗口数量或目标位置。右对齐还保证每种 NBM 总是看到距离预测目标最近的那段历史。

因此本套件中的 C2 也必须重新训练，不能直接复用旧 2 s NBM 套件的结果。旧套件按 2 s warmup 建窗，而这里的 C2 使用统一 4 s 支持和更严格的 clean-normal 资格；二者的可用 anchor 与训练窗口并不完全相同。

## 4. 4 × 4 实验矩阵

| NBM | C1：1 s | C2：2 s | C3：3 s | C4：4 s |
|---|---:|---:|---:|---:|
| Linear-AR | C1，实际 0.5 s | C2，实际 0.5 s | C3，实际 0.5 s | C4，实际 0.5 s |
| GRU | GRU-C1 | GRU-C2 | GRU-C3 | GRU-C4 |
| TCN | TCN-C1 | TCN-C2 | TCN-C3 | TCN-C4 |
| Transformer | Transformer-C1 | Transformer-C2 | Transformer-C3 | Transformer-C4 |

每个 LOSO fold 训练 16 个 NBM/分类器单元格；8 个 fold 共得到 128 个完整测试单元格。每个单元格的预测 horizon、残差历史、TCN-M、损失、类别权重、阈值规则和测试受试者均相同。

### Linear-AR 是实际 0.5 s 的负对照

Linear-AR 的阶数固定为 32 点，即 0.5 s。虽然 C1–C4 的接口分别传入 64/128/192/256 点，Linear-AR 始终只读取末尾 32 点：

```text
Linear-AR(C1) = Linear-AR(C2) = Linear-AR(C3) = Linear-AR(C4)
实际有效上下文均为最后 0.5 s
```

四个 Linear-AR 单元格因此是“上下文标签发生变化、模型实际输入不变”的负对照。若它们出现超出数值误差或训练随机性的系统性差异，应优先排查样本支持、随机种子、缓存或恢复流程，而不能解释为长上下文收益。

## 5. 两种 TCN 感受野的正确解释

### 5.1 TCN-NBM：RF=253，C4 距完整 4 s 只差 3 点

TCN-NBM 使用 `kernel_size=3`、每个 block 两层因果卷积，dilation 为：

```text
1, 2, 4, 8, 16, 32
```

其末端预测状态的理论局部感受野为：

```text
RF = 1 + 2 × (kernel_size - 1) × Σdilation
   = 1 + 2 × 2 × (1+2+4+8+16+32)
   = 253 点
```

C4 输入是 256 点，因此 TCN-NBM 的末端状态覆盖其中最近 253 点，只缺最早 3 点，相当于 46.875 ms。代码按 `RF / fs` 报告约 3.953 s。论文中应表述为“几乎覆盖完整 4 s 上下文”，而不是严格声称“完整覆盖 4 s”。C1–C3 的实际序列短于该理论上限，因果卷积只能利用各自提供的 1/2/3 s 有效数据。

### 5.2 TCN-M：RF=125 是局部特征感受野，最终读出仍汇聚完整 4 s

固定分类器 TCN-M 使用：

```text
dilation = 1, 2, 4, 8, 8, 8
kernel_size = 3
每个 block 两层卷积
局部 RF = 125 点 = 1.953125 s
```

125 点描述的是每个时间位置上的卷积特征可以看到多宽的局部邻域。分类头随后对全部 256 个时间位置同时执行 temporal mean pooling 和 temporal max pooling，再拼接并分类。因此最终窗口级预测会聚合整个 4 s 内各处的局部 FoG 模式，不能把 TCN-M 简化解释为“只读取最后 1.95 s”。

这两个概念必须区分：

- TCN-NBM 的 RF=253 约束末端状态用于预测未来正常信号时可利用的历史范围；
- TCN-M 的 RF=125 约束单个局部残差特征，但全局 mean/max pooling 覆盖完整 256 点输入。

## 6. LOSO、随机种子、损失和阈值

### 6.1 LOSO 划分

每个 fold 以一名受试者作为 test。validation 从测试受试者之后按固定循环顺序寻找第一名同时含 FoG 和 non-FoG 窗口的受试者，其余受试者用于训练。标准化器只在训练受试者上拟合，NBM 只读取训练/验证受试者中的 clean-normal 窗口。

### 6.2 随机种子和公平性

- 全局 seed 为 42。
- NBM fold seed 为 `42 + fold_index`。
- TCN-M classifier seed 为 `42 + 10000 + fold_index`。
- 同一 fold 的 16 个 TCN-M 使用相同 seed、相同初始化参数哈希和相同训练样本顺序规则。
- classifier 每个 epoch 的 shuffle seed 为 `classifier_seed + epoch`。

这样可以把单元格差异尽量归因于 NBM 和上下文，而不是分类器初始化。

### 6.3 NBM 训练

- 输出每个通道、未来 32 点的条件均值 `μ` 和正标准差 `σ`。
- 损失为异方差 Gaussian NLL：

  ```text
  mean(log(σ) + 0.5 × ((target - μ) / σ)^2)
  ```

- AdamW，学习率 `1e-3`，weight decay `1e-4`。
- 最多 8 epochs，按 validation NLL 保存最优模型，patience=3。
- normal train 最多确定性抽样 30,000 个窗口。

### 6.4 TCN-M 训练与阈值

- 损失为 `BCEWithLogitsLoss`。
- FoG 正类权重为：

  ```text
  pos_weight = min(sqrt(N_nonFOG / N_FOG), 6)
  ```

- AdamW，学习率 `1e-3`，weight decay `1e-4`。
- 最多 12 epochs，以 validation PR-AUC 保存最优模型，patience=4。
- 默认不限制分类训练窗口数量（`max_classifier_windows=0`）。
- 分类阈值只在 validation 受试者上选择：在 0.01–0.99、步长 0.01 的候选中最大化 Balanced Accuracy；并列时依次选择 FoG F1 更高、阈值更高者。
- 选定阈值不再调整，直接用于 held-out test 受试者。

## 7. ΔPR-AUC 与统计比较

每个 NBM 都以自己的 C2（2 s）作为唯一参考，不能跨 NBM 比较差值：

```text
ΔPR-AUC(NBM, Cj, subject)
  = PR-AUC(NBM, Cj, subject)
  - PR-AUC(NBM, C2, subject)
```

先在同一 held-out subject 内计算配对差，再对受试者差值取均值。95% CI 使用以 held-out subject 为抽样单位的非参数配对 bootstrap：

- bootstrap 次数：100,000；
- bootstrap seed：42，并结合实验 ID 生成稳定的子种子；
- 每次有放回抽取完整的受试者配对；
- CI 为 bootstrap 均值分布的 2.5% 和 97.5% 分位数。

因此 `paired_pr_auc_deltas.csv` 中的 ΔPR-AUC 衡量“在同一种 NBM 下，相对 2 s 上下文的变化”。C2 对自身的差值应为 0。主排名仍使用 8 个 held-out subject 的 macro PR-AUC 均值，而不是 pooled-window PR-AUC。

## 8. 窗口级和事件级指标

窗口级报告：

- Accuracy；
- Balanced Accuracy；
- Macro-F1；
- AUROC；
- PR-AUC；
- FoG Recall/Sensitivity；
- Specificity；
- FoG Precision；
- FoG F1；
- MCC；
- TN、FP、FN、TP。

每个实验同时保存各 held-out subject 的指标、subject-macro 均值与总体标准差，并额外保存 pooled-window 结果。由于受试者窗口数差异较大，论文主比较应采用 subject-macro 结果。

事件级报告：

- Event Sensitivity：真实 FoG 事件中被至少一个预测事件重叠命中的比例；
- FA/h：未与真实 FoG 事件匹配的预测事件数除以有效记录小时数；
- Median Detection Delay：已检出事件从真实起点到确认判定时刻的延迟中位数。

预测事件要求至少连续 2 个阳性窗口，间隔不超过 0.5 s 的相邻预测段会合并。真实事件与预测事件采用时间重叠的一对一匹配；只评价与有效目标窗口覆盖区间相交的真实事件。

## 9. 7 GPU 完整运行命令

在服务器项目根目录运行：

```bash
cd /document/home_mirror/chb/fog_classification_framework_base

DATA_DIR="$PWD/dataset/1.Daphnet Freezing of Gait Dataset/processed"
OUTPUT_DIR="$PWD/outputs/daphnet_nbm4_context4_h4_tcnm_loso_seed42"

python -u scripts/start_daphnet_nbm_context_tcnm_suite_multigpu.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --gpus 0-6 \
  --work-folds all \
  --max-retries 2 \
  --launch-delay 2 \
  --audit \
  --exclude-subjects S04,S10 \
  --nbms linear_ar,gru,tcn,transformer \
  --context-seconds 1,2,3,4 \
  --support-context-seconds 4 \
  --horizon-seconds 0.5 \
  --stride-seconds 0.25 \
  --history-seconds 4 \
  --normal-guard-seconds 0.5 \
  --fog-fraction-threshold 0.5 \
  --linear-ar-seconds 0.5 \
  --nbm-hidden 48 \
  --nbm-dropout 0.1 \
  --gru-layers 1 \
  --transformer-heads 4 \
  --transformer-layers 2 \
  --transformer-ffn 128 \
  --classifier-hidden 48 \
  --classifier-dropout 0.15 \
  --normal-epochs 8 \
  --normal-patience 3 \
  --normal-lr 0.001 \
  --classifier-epochs 12 \
  --classifier-patience 4 \
  --classifier-lr 0.001 \
  --weight-decay 0.0001 \
  --batch-size 256 \
  --max-normal-windows 30000 \
  --max-classifier-windows 0 \
  --bootstrap-samples 100000 \
  --bootstrap-seed 42 \
  --seed 42 \
  --num-workers 0 \
  --amp \
  --deterministic \
  --cache-residuals
```

调度器把一个完整 fold 作为不可拆分任务。一开始 GPU 0–6 分别运行 S01、S02、S03、S05、S06、S07、S08；最先空闲的 GPU 再运行 S09。每张 GPU 在自己的 fold 内顺序完成全部 16 个单元格。

调度器控制并强制启用 `--resume`，不要额外转发 `--resume`、`--folds`、`--worker-fold` 或 `--device`。

## 10. 监控命令

查看调度器总体状态：

```bash
watch -n 10 "python -m json.tool '$OUTPUT_DIR/multigpu_status.json'"
```

跟踪某个 fold：

```bash
tail -n 80 -f "$OUTPUT_DIR/multigpu_logs/S01.log"
```

查看最终汇总和审计日志：

```bash
tail -n 100 "$OUTPUT_DIR/multigpu_logs/finalize.log"
tail -n 100 "$OUTPUT_DIR/multigpu_logs/audit.log"
```

同时观察 GPU：

```bash
watch -n 2 nvidia-smi
```

单个 fold 内依次训练多个 NBM 和 TCN-M，数据准备、缓存读取、finalize 与 audit 阶段也会使用 CPU，因此 GPU-Util 不会始终保持满载。

## 11. 恢复、汇总和审计

正常中断后，使用完全相同的命令和 `OUTPUT_DIR` 重新运行即可。调度器会：

- 跳过已有且校验通过的 `DONE.json` 单元格；
- 从 NBM 的 `nbm/last.pt` 或分类器的 `classifier_last.pt` 继续 epoch；
- 复用带哈希校验的 `residual_cache.npz`；
- 对失败 fold 最多额外重试 2 次；
- 所有 fold 完成后重新执行 finalize，再执行严格 audit。

若只是按下 `Ctrl+Z`，原调度器仍然存活且持有锁，应先运行：

```bash
jobs -l
fg %1
```

不要在旧调度器仍存活或 worker 仍运行时删除 `.multigpu_scheduler.lock`。若进程已真正结束，直接重跑同一命令；调度器会保守检查并回收同机 stale lock。相同输出目录不能更改科学参数，否则 protocol fingerprint 校验会拒绝混合结果；需要更改协议时必须使用新输出目录。

正式 runner 会在训练前强制检查完整的 4 NBM、4 context 和 8-fold 科学协议，避免子集被误标为完整实验。仅本地开发冒烟测试可显式使用 `--allow-protocol-subset`；并行 worker 和正式 auditor 始终拒绝该模式。

只重建根目录汇总时可运行：

```bash
python -u scripts/run_daphnet_nbm_context_tcnm_suite.py \
  --resume \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --folds all \
  --finalize-only \
  --device cpu
```

只重新审计时可运行：

```bash
python -u scripts/audit_daphnet_nbm_context_tcnm_suite.py \
  --result-dir "$OUTPUT_DIR"
```

审计器会复核固定 4 × 4 × 8 协议、protocol fingerprint、`DONE.json` 与产物哈希、公共 history support、16 个单元格之间的窗口 ID/标签一致性、NBM 和 TCN-M 配置、保存预测与窗口指标，以及根目录汇总表的行数和完成计数。它还会从逐 fold 预测独立重算同 NBM C2 配对的 ΔPR-AUC、bootstrap 95% CI，并同时核对 CSV 与 aggregate JSON。审计结果写入 `audit_report.json` 和 `audit_report.txt`。

完整运行应看到 16 个实验 × 8 个 fold，即 128 个 classifier cell 全部完成，并且审计返回成功。`--allow-partial` 仅用于有意运行 fold 子集的开发检查，不用于最终论文结果。

## 12. 目录结构与主要输出

根目录主要文件：

```text
OUTPUT_DIR/
├── config.json                    # 含运行路径的完整协议
├── run_manifest.json              # 不含运行时路径的科学协议与 fingerprint
├── environment.json               # 初始化环境
├── worker_environments/           # 各 fold 的 CUDA/软件环境
├── multigpu_status.json           # 调度状态、PID、GPU、重试和完成情况
├── multigpu_logs/
│   ├── initialize.log
│   ├── S01.log ... S09.log
│   ├── finalize.log
│   └── audit.log
├── experiment_manifest.csv        # 16 个实验的完成状态
├── fold_summary.csv               # 128 个 fold-cell 的窗口/事件指标
├── aggregate_summary.csv          # subject-macro 均值、标准差及排名
├── paired_pr_auc_deltas.csv       # 相对同 NBM C2 的配对 ΔPR-AUC 和 95% CI
├── publication_table.csv          # 论文表格格式
├── aggregate_metrics.json         # macro、pooled、最佳实验和统计元数据
├── status.json                    # 期望/已完成 cell 计数
├── audit_report.json              # 机器可读的严格审计结果
└── audit_report.txt               # 便于直接检查的审计摘要
```

每个 fold/单元格的核心结构：

```text
loso_S01/
├── fold_config.json
├── context_suite_fold_config.json
├── scaler.json
├── split_indices.npz
├── history_support.npz
└── context_c1_1s__loso_s01/
    └── linear_ar/
        ├── nbm/
        │   ├── best.pt
        │   ├── last.pt
        │   ├── training.json
        │   └── DONE.json
        ├── residual_cache.npz
        ├── residual_diagnostics.json
        ├── RESIDUAL_CACHE_DONE.json
        ├── nbm_summary.json
        └── residual_h4s/
            └── tcn_m/
                ├── classifier_best.pt
                ├── classifier_last.pt
                ├── metrics.json
                ├── predictions.npz
                ├── validation_predictions.npz
                ├── predictions.csv
                └── DONE.json
```

其他 context 和 NBM 按相同结构展开。最终结果优先读取 `publication_table.csv` 和 `aggregate_summary.csv`；统计解释与复核使用 `paired_pr_auc_deltas.csv`、`aggregate_metrics.json`、各单元格 `metrics.json` 和预测文件。
