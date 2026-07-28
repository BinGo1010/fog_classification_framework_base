# Daphnet Persistence 输入表示消融实验

## 1. 实验目的

本实验在样本支持、LOSO 划分、Persistence 正常行为模型、TCN-M 分类器和训练
协议完全一致的条件下，仅改变输入表示，用于回答以下三个问题：

1. 以 Persistence 均值预测 \(\mu\) 对原始 IMU 进行中心化是否有贡献；
2. 使用正常行为误差尺度 \(\sigma\) 进行标准化是否有额外贡献；
3. 将标准化误差截断到 \([-12,12]\) 是否进一步提高稳健性。

严格消融实验包含 4 种输入表示、8 个 held-out subject，共
\(4\times8=32\) 个分类器实验单元。

`Raw-TCN-M-4s` 是直接使用自然连续 4 秒原始 IMU 的监督分类基线，应放在主要
方法对比中。它不自动属于本组严格消融，因为其自然建窗得到的 anchor 和 warm-up
支持可能与残差方法不同。本组消融中的原始信号参考组必须是
`Raw-support-matched`。

## 2. 固定实验协议

| 项目 | 固定设置 |
|---|---|
| 数据集 | Daphnet |
| 传感器 | ankle、thigh、trunk 三个 IMU，共 9 个原始加速度通道 |
| 采样率 | 64 Hz |
| 排除受试者 | S04、S10 |
| LOSO 测试受试者 | S01、S02、S03、S05、S06、S07、S08、S09 |
| 折内标准化 | Robust scaler，仅由训练受试者的有效 non-FoG 样本拟合，并截断到 `[-12,12]` |
| Persistence 接口 context | 2 秒，即 128 点 |
| Persistence 有效均值输入 | context 最后 1 个采样点 |
| 预测目标块 | 紧随 context 的 0.5 秒，即 32 点 |
| NBM 预测步长 | 0.25 秒，即 16 点 |
| 分类器输出步长 | 0.25 秒，即 16 点 |
| 分类器历史 | 4 秒，即 256 点 |
| 历史构成 | 8 个按时间排列、在单个历史内互不重叠的 0.5 秒块 |
| 分类标签 | 最后一个 0.5 秒目标块的 FoG 标签 |
| 分类器 | 固定 TCN-M |
| 主随机种子 | 42 |

Persistence 的 2 秒 context 是统一协议和样本资格的一部分，但其均值预测并不会
使用完整 128 点。模型只读取最后一个 9 维采样点并将其重复 32 次。保留 2 秒
context 可以复用相同的窗口支持、clean-normal 训练资格和已有 Persistence
checkpoint，不能将其解释为均值模型实际利用了 2 秒历史。

### 2.1 0.25 秒步长与 4 秒历史的关系

每隔 0.25 秒建立一个新的预测目标块和分类 anchor。对于任意一个分类 anchor，
4 秒输入不是直接拼接最近 16 个重叠块，而是从公共 `HistoryPlan` 中选择 8 个
目标起点相隔 0.5 秒的块：

```text
块 1       块 2                         块 8
[0.5 s] + [0.5 s] + ... + [0.5 s]  =  4 s
```

因此：

- 同一个 4 秒输入内部的 8 个块互不重叠；
- 相邻两次分类决策相隔 0.25 秒，所以相邻的两个 4 秒输入会重叠；
- 0.25 秒 predictor grid 实际形成两个相位相差 0.25 秒、各自以 0.5 秒间隔
  连接的历史序列；
- 最后一个 0.5 秒目标块必须已经观测完成才能计算误差，因此本任务是基于残差的
  FoG 检测，不是提前 0.5 秒的 FoG 预测。

更完整的时序描述是：Persistence 每 0.25 秒调用一次，以该窗口 2 秒 context
的最后一个采样点生成未来 0.5 秒的常值均值；待实际 0.5 秒 target 到达后计算
误差块。每次分类从截至当前 anchor 的同相位误差流中选择 8 个相隔 0.5 秒的
非重叠块拼成 4 秒输入，分类决策每 0.25 秒更新。

## 3. Persistence 表示的数学定义

令 \(x_i\in\mathbb{R}^{9\times32}\) 表示物理单位下的第 \(i\) 个实际目标
块。训练折 Robust scaler 的中心和尺度分别记作 \(m_{\mathrm{train}}\) 和
\(s_{\mathrm{train}}\)，则模型实际接收的 raw target 为：

\[
\tilde{x}_i
=
\operatorname{clip}
\left(
\frac{x_i-m_{\mathrm{train}}}{s_{\mathrm{train}}},
-12,12
\right).
\]

因此本文中的 `raw` 是经过折内 robust scaling 和公共输入预处理截断后的原始
IMU，而不是物理单位原始信号。令 \(c_i\) 表示在相同预处理空间中的前方
context。

Persistence 的均值为：

\[
\mu_i
=
\operatorname{repeat}\!\left(c_i[:, -1], 32\right)
\in\mathbb{R}^{9\times32}.
\]

Persistence 不训练均值预测器。唯一可学习参数是
\(\log\sigma\in\mathbb{R}^{1\times9\times32}\)，共 \(9\times32=288\)
个参数。它只使用 LOSO 折内训练受试者的 clean non-FoG 窗口，通过 Gaussian
NLL 学习每个通道和每个预测步的正常误差尺度：

\[
\sigma
=
\exp\!\left(
\operatorname{clip}(\log\sigma,\ \log\sigma_{\min},\ \log\sigma_{\max})
\right).
\]

四种块级表示依次为：

\[
\begin{aligned}
q_i^{A} &= \tilde{x}_i,\\
q_i^{B} &= \tilde{x}_i-\mu_i,\\
q_i^{C} &= \frac{\tilde{x}_i-\mu_i}{\sigma},\\
q_i^{D} &= \operatorname{clip}
\left(\frac{\tilde{x}_i-\mu_i}{\sigma},-12,12\right).
\end{aligned}
\]

对每一种表示 \(q\)，均使用同一个 `HistoryPlan` 选择 8 个块并按时间顺序拼接：

\[
H_i^q
=
\operatorname{concat}
\left(q_{i,1},q_{i,2},\ldots,q_{i,8}\right)
\in\mathbb{R}^{9\times256}.
\]

需要区分两种截断：所有组都共享 Robust scaler 对
\(\tilde{x}\) 的 `[-12,12]` 输入预处理截断；只有 D 组额外对除以 \(\sigma\)
后的标准化残差执行 `[-12,12]` 截断。因此 \(D-C\) 隔离的是残差空间截断的
贡献，而不是 raw 输入预处理截断的贡献。

## 4. 四组严格消融

| ID | 输入表示 | 使用 \(\mu\) | 使用 \(\sigma\) | 使用 clip | 输出形状 | 科学问题 |
|---|---|---:|---:|---:|---:|---|
| `raw_support_matched` | \(\tilde{x}\) | 否 | 否 | 否 | `[N,9,256]` | 在完全相同支持上，原始 IMU 能达到什么水平 |
| `error_x_minus_mu` | \(\tilde{x}-\mu\) | 是 | 否 | 否 | `[N,9,256]` | Persistence 中心化本身是否有效 |
| `standardized_error` | \((\tilde{x}-\mu)/\sigma\) | 是 | 是 | 否 | `[N,9,256]` | 正常误差尺度标准化是否提供额外信息 |
| `standardized_error_clip12` | \(\operatorname{clip}((\tilde{x}-\mu)/\sigma,-12,12)\) | 是 | 是 | 是 | `[N,9,256]` | 极值截断是否提高稳健性 |

完整方法的默认参考 ID 为 `standardized_error_clip12`。

## 5. Raw-support-matched 的严格定义

`Raw-support-matched` 不是任意重新切出的连续 4 秒窗口，而是：

1. 使用与完整残差方法完全相同的 train、validation、test `HistoryPlan`；
2. 使用完全相同的 anchor window ID；
3. 对每个 anchor 使用完全相同的 8 个目标块 ID；
4. 使用相同折内 robust scaler 及其公共 `[-12,12]` 输入预处理截断；
5. 使用最后一个目标块的相同标签；
6. 仅把块内容从完整残差替换成 robust-scaled raw target。

这 8 个 raw target 块在单个历史中恰好构成连续 4 秒信号，但它们的样本资格和
anchor 集合必须继承完整残差方法。实现时应使用块历史构造接口，例如：

```python
make_block_history_input(
    extracted=features,
    plan=common_history_plan,
    source_key="raw",
    name="raw_support_matched",
    history_samples=256,
    horizon_samples=32,
    stride_samples=16,
)
```

不能使用只提取最终 0.5 秒目标块的 `make_anchor_raw_input()` 代替
`Raw-support-matched`。

如果 `Raw-TCN-M-4s` 后续也被限制到完全相同的 anchor、标签、scaler 和训练
协议，那么它在输入张量上就等价于 `Raw-support-matched`，不应作为两个独立
实验重复计数。若二者采用不同自然支持，则 `Raw-TCN-M-4s` 只作为外部监督
基线，其差异不能被归因于输入表示本身。

## 6. 严格控制变量

四个实验组必须固定以下条件：

- 同一 fold 的 train、validation 和 held-out test 受试者；
- 同一折内 robust scaler，且 scaler 只由训练受试者拟合；
- 同一 Persistence checkpoint 和同一组 \(\sigma\)；
- 同一 train、validation、test anchor ID、8 块 history ID 和标签；
- 同一 TCN-M 结构、参数量、感受野和池化方式；
- 同一 classifier seed、初始权重 SHA256 和每个 epoch 的 shuffle 顺序；
- 同一损失函数、类别权重、优化器、学习率、weight decay、batch size、
  最大 epoch 和 early-stopping patience；
- 同一训练样本数量，不对某一输入组单独下采样；
- 同一 validation early-stopping 规则和阈值选择规则；
- 测试受试者不参与 scaler、Persistence、early stopping 或阈值选择。

建议沿用现有 TCN-M 正式协议：

| 项目 | 设置 |
|---|---|
| 损失 | `BCEWithLogitsLoss` |
| FoG 类别权重 | `min(sqrt(N_nonFoG / N_FoG), 6)` |
| 优化器 | AdamW |
| 学习率 | `1e-3` |
| Weight decay | `1e-4` |
| Batch size | 256 |
| 最大 epochs | 12 |
| Patience | 4 |
| 最优 epoch | validation PR-AUC 最大 |
| 阈值 | 仅在 validation subject 上最大化 Balanced Accuracy |
| 训练窗口上限 | 0，即使用全部窗口 |
| 可复现性 | seed 42、deterministic 开启 |

四个输入组应分别使用 validation 数据选择自己的最优 epoch 和数值阈值，但选择
算法必须完全相同。强制四组共用同一个数值阈值并不公平，因为四种输入可能产生
不同尺度的 logit 和概率校准。

## 7. 实现注意事项

现有 canonical residual cache 可能只保存已经标准化并截断后的
`residual`。不能通过已截断残差反推出 \(x-\mu\) 或未截断的
\((x-\mu)/\sigma\)。

正确实现应在每个 fold 中：

1. 加载相同的窗口表、折内 scaler 和已训练 Persistence checkpoint；
2. 对相同 window ID 一次性提取并保存：
   `raw`、`mu`、`sigma`、`error`、`standardized_error`；
3. 从未截断的 `standardized_error` 单独产生
   `standardized_error_clip12`；
4. 使用同一个 `HistoryPlan` 分别物化四种 `[N,9,256]` 输入；
5. 冻结上述输入缓存后，再训练四个 TCN-M。

Persistence 的均值没有可学习参数，因此四组之间不需要重复训练正常均值模型。
\(\sigma\) 也应在每个 LOSO fold 中只训练一次并冻结，避免把重复训练随机性混入
输入表示消融。

## 8. 评价指标

### 8.1 主指标

- Subject-macro PR-AUC：先计算每个 held-out subject 的 PR-AUC，再对 8 个
  subject 等权平均。

### 8.2 次要窗口级指标

- Accuracy；
- Balanced Accuracy；
- Macro-F1；
- AUROC；
- FoG Sensitivity/Recall；
- FoG Precision；
- FoG F1；
- Specificity；
- MCC。

类别不平衡时，Accuracy 仅作为描述性指标，不用于选择最佳输入表示。

### 8.3 事件级指标

- Event Sensitivity；
- FA/h；
- Detection Delay。

事件指标依赖验证集阈值和事件后处理规则，因此四组必须使用完全相同的事件构造、
最短连续阳性要求、合并间隔和匹配规则。

## 9. 预注册统计比较

建议预先固定以下 4 个配对比较：

| 比较 | 含义 |
|---|---|
| \(B-A\) | Persistence 中心化的增量贡献 |
| \(C-B\) | 正常误差尺度标准化的增量贡献 |
| \(D-C\) | 截断的增量贡献 |
| \(D-A\) | 完整输入表示相对 matched raw 的总贡献 |

每个差值必须先在同一个 held-out subject 内计算，再对 8 个 subject 汇总：

\[
\Delta_m(s)
=
m_{\mathrm{new}}(s)-m_{\mathrm{reference}}(s).
\]

主统计量使用 PR-AUC 的配对差值，并报告：

- 平均 \(\Delta\)PR-AUC；
- 以 held-out subject 为重采样单位的 100,000 次配对 bootstrap 95% CI；
- 8 个 subject 中新方法获胜的数量；
- bootstrap seed 42。

Balanced Accuracy、Macro-F1、AUROC、FoG Recall、FoG F1、FA/h 和事件指标
可使用相同方式给出配对效应量，但应标记为次要或探索性分析。若同时报告显著性
\(p\) 值，可补充配对 Wilcoxon 检验，并对预注册的 4 个比较进行 Holm 校正。
受试者数量仅为 8，应优先解释效应量、置信区间和跨受试者一致性，而不是只依据
\(p\) 值下结论。

## 10. 结果表模板

### 10.1 主结果

| 输入表示 | PR-AUC | BA | Macro-F1 | AUROC | FoG Recall | FoG F1 | Event Sensitivity | FA/h | Delay (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw-support-matched | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| \(x-\mu\) | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| \((x-\mu)/\sigma\) | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| \(\operatorname{clip}((x-\mu)/\sigma,-12,12)\) | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

所有主表数值报告 8-fold subject-macro `mean ± std`。

### 10.2 配对增量

| 比较 | \(\Delta\)PR-AUC [95% CI] | Wins / 8 | 解释 |
|---|---:|---:|---|
| \(x-\mu\) − Raw-support-matched | 待运行 | 待运行 | 中心化贡献 |
| \((x-\mu)/\sigma\) − \((x-\mu)\) | 待运行 | 待运行 | 尺度标准化贡献 |
| clipped − unclipped standardized error | 待运行 | 待运行 | 截断贡献 |
| 完整方法 − Raw-support-matched | 待运行 | 待运行 | 总贡献 |

## 11. 输入诊断与独立审计

正式结果必须至少通过以下检查：

1. 四组 train、validation、test 的 anchor window ID 逐元素相同；
2. 四组每个 anchor 的 8 个 history block ID 逐元素相同；
3. 四组标签和样本数量逐元素相同；
4. 所有输入形状均为 `[N,9,256]`；
5. 所有输入均不含 NaN 或 Inf，且所有 \(\sigma>0\)；
6. `error_x_minus_mu` 与 `raw - mu` 在容差内一致；
7. `standardized_error` 与 `error / sigma` 在容差内一致；
8. `standardized_error_clip12` 与
   `clip(standardized_error,-12,12)` 在容差内一致；
9. 完整方法输入与 canonical Persistence `residual_h4s` 在相同支持上逐元素
   一致；
10. 同一 fold 四个 TCN-M 的初始权重 SHA256 相同；
11. validation 阈值和所有测试指标可从保存的预测独立重算；
12. 32 个分类器实验单元全部完成后才生成 suite 完成标记。

建议额外输出以下输入诊断：

- 每组输入的 mean、std、RMS、绝对值分位数；
- 各通道未截断标准化误差超过 \(\pm12\) 的比例；
- train、validation、test 的截断比例；
- \(\sigma\) 的通道 × 预测步热图或数值摘要；
- 四组支持、标签、初始化参数和源 checkpoint 的 SHA256。

## 12. 建议输出目录

建议实验目录命名为：

```text
outputs/daphnet_persistence_input_ablation_h4_tcnm_stride025_loso_seed42/
```

建议根目录至少包含：

```text
config.json
run_manifest.json
experiment_manifest.csv
status.json
fold_summary.csv
aggregate_summary.csv
paired_pr_auc_deltas.csv
publication_table.csv
input_diagnostics.csv
support_equivalence.json
aggregate_metrics.json
AUDIT_REPORT.json
SUITE_COMPLETE.json
```

建议每个 fold 使用以下结构：

```text
loso_S01/
├── fold_config.json
├── source_provenance.json
├── input_support.npz
├── representation_cache.npz
├── raw_support_matched/
├── error_x_minus_mu/
├── standardized_error/
└── standardized_error_clip12/
```

每个输入表示目录至少保存：

```text
classifier_best.pt
classifier_last.pt
validation_predictions.npz
predictions.npz
predictions.csv
metrics.json
DONE.json
```

## 13. 实验执行顺序

1. **协议初始化**：锁定 4 输入 × 8 fold、seed、模型与训练参数；
2. **支持生成**：为每个 fold 生成唯一的 train/validation/test
   `HistoryPlan`；
3. **表示缓存**：加载并冻结 Persistence，一次生成四种块级表示；
4. **等价性检查**：训练分类器前先验证支持、标签和公式关系；
5. **本地冒烟测试**：仅用一个 fold、少量窗口和 1 epoch 验证恢复及产物链路；
6. **服务器正式运行**：使用全部窗口完成 32 个分类器实验单元；
7. **汇总与审计**：独立重算阈值、窗口指标、事件指标和配对差值；
8. **论文报告**：主表报告 subject-macro 结果，配对表报告
   \(\Delta\)PR-AUC 与 95% CI。

冒烟测试输出不得与正式结果混用，也不能用于方法优劣判断。

## 14. 结果解释规则

- 若 \(B-A>0\)，说明相对于 matched raw，减去 Persistence 均值提供了有效的
  运动变化表示；
- 若 \(C-B>0\)，说明按正常行为误差尺度标准化提供了超出中心化的贡献；
- 若 \(D-C>0\)，说明截断极端标准化误差提高了稳健性；
- 若 \(D-A>0\)，说明完整 Persistence 表示整体优于相同样本支持下的 raw；
- 若均值提升但 95% CI 跨 0，应描述为趋势，而不是确定性优势；
- 若 PR-AUC 提升但 FA/h 或 Detection Delay 恶化，应明确报告检测收益与部署
  代价之间的权衡；
- 若 `Raw-TCN-M-4s` 与 `Raw-support-matched` 差异明显，应先检查样本支持、
  anchor 数量和训练协议，不能直接解释为模型能力差异。

最终结论应建立在预注册的逐步比较 \(B-A\)、\(C-B\)、\(D-C\) 和总比较
\(D-A\) 上，而不是仅根据四组中的最高均值进行事后选择。

## 15. 已实现代码

本实验已对应以下独立入口：

```text
scripts/run_daphnet_persistence_input_ablation.py
scripts/start_daphnet_persistence_input_ablation_multigpu.py
scripts/audit_daphnet_persistence_input_ablation.py
tests/test_daphnet_persistence_input_ablation.py
```

单折程序首先验证 canonical source suite、fold scaler、Persistence best
checkpoint、split indices、history support 和 residual cache 的哈希。随后每折仅
重放一次 Persistence，生成并冻结 `representation_cache.npz`。四个 TCN-M
分类器顺序消费同一个缓存和同一个 `HistoryPlan`，不会重复训练 Persistence。

受 CPU/GPU 浮点 `exp` 和除法实现影响，重新计算的 clipped standardized error
与 canonical residual 可能有约 \(10^{-6}\) 的舍入差异。因此实现会在严格容差
内验证
`clip(standardized_error,-12,12)` 与 canonical residual 一致，但 D 组最终
保存和使用 canonical residual 数组，从而保证完整方法输入与原正式实验完全
一致。

## 16. 七卡服务器正式运行

在项目根目录执行：

```bash
python -u scripts/start_daphnet_persistence_input_ablation_multigpu.py \
  --data-dir "/home/chb/Documents/FOG/fog_classification_framework_base/dataset/1.Daphnet Freezing of Gait Dataset/processed" \
  --source-suite-dir "$PWD/outputs/daphnet_3imu_nbm_5x4_loso_seed42" \
  --output-dir "$PWD/outputs/daphnet_persistence_input_ablation_h4_tcnm_stride025_loso_seed42" \
  --gpus 0-6 \
  --work-folds all \
  --max-retries 2 \
  --launch-delay 2 \
  --audit \
  --seed 42 \
  --classifier-epochs 12 \
  --classifier-patience 4 \
  --classifier-lr 0.001 \
  --weight-decay 0.0001 \
  --batch-size 256 \
  --max-classifier-windows 0 \
  --bootstrap-samples 100000 \
  --bootstrap-seed 42 \
  --num-workers 0 \
  --amp \
  --deterministic
```

一个 GPU 独立负责一个完整 LOSO fold，并在该 fold 内顺序完成四个表示。7 张
GPU 先并行运行 7 折，最先空闲的 GPU 再运行第 8 折。程序支持 epoch 级断点
续训、fold 失败重试、状态心跳、最终汇总和独立审计。

正式协议会拒绝改变 epochs、batch size、训练窗口数等预注册设置。仅测试流程时
可使用独立输出目录并增加 `--smoke`，例如：

```bash
python -u scripts/run_daphnet_persistence_input_ablation.py \
  --data-dir "/path/to/processed" \
  --source-suite-dir "$PWD/outputs/daphnet_3imu_nbm_5x4_loso_seed42" \
  --output-dir "$PWD/outputs/input_ablation_smoke" \
  --folds all \
  --worker-fold S01 \
  --device cuda \
  --smoke \
  --classifier-epochs 1 \
  --classifier-patience 1 \
  --batch-size 64 \
  --max-classifier-windows 64 \
  --bootstrap-samples 100
```

并行 worker 前仍需先用相同参数加
`--finalize-only --device cpu` 初始化协议。Smoke 结果会标记为
`reportable=false`，独立审计不会生成正式完成标记。

## 17. 监控、续训与输出

```bash
OUTPUT_DIR="$PWD/outputs/daphnet_persistence_input_ablation_h4_tcnm_stride025_loso_seed42"

watch -n 10 "python -m json.tool '$OUTPUT_DIR/multigpu_status.json'"
tail -f "$OUTPUT_DIR/multigpu_logs/S01.log"
watch -n 2 nvidia-smi
```

中断后使用完全相同的命令和输出目录重新运行。已完成的表示缓存和分类器会进行
哈希验证并跳过，未完成分类器从 `classifier_last.pt` 继续。

主要汇总文件包括：

- `fold_summary.csv`：32 个 fold-level 分类器结果；
- `aggregate_summary.csv`：8-fold subject-macro 均值和标准差；
- `paired_pr_auc_deltas.csv`：四个预注册配对比较、95% CI 和 wins；
- `publication_table.csv`：论文主表；
- `input_diagnostics.csv`、`sigma_diagnostics.csv`：输入与尺度诊断；
- `support_equivalence.json`：公共 anchor、history 和标签协议；
- `aggregate_metrics.json`、`status.json`：机器可读汇总；
- `AUDIT_REPORT.json`、`AUDIT_REPORT.txt`：独立审计；
- `SUITE_COMPLETE.json`：仅正式 32/32 单元全部通过审计后生成。
