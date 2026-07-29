# Daphnet Persistence IMU 数量与位置消融实验

## 1. 实验目的

本实验在正常行为模型、残差表示、时间支持、TCN-M 分类器和 LOSO
划分不变的条件下，仅改变提供给分类器的 IMU 位置与通道数量，用于回答：

1. 单个 ankle、thigh 或 trunk IMU 是否足以识别 FoG；
2. 两个 IMU 是否能达到接近三个 IMU 的性能；
3. 哪一个 IMU 对三个 IMU 完整系统的增益最大；
4. 减少传感器能否在诊断性能与佩戴、传输和计算成本之间取得更好的平衡。

实验包含 7 种传感器组合和 8 个 held-out subject，共
\(7\times8=56\) 个分类器实验单元。

## 2. 固定实验协议

| 项目 | 固定设置 |
|---|---|
| 数据集 | Daphnet |
| 采样率 | 64 Hz |
| 排除受试者 | S04、S10 |
| LOSO 测试受试者 | S01、S02、S03、S05、S06、S07、S08、S09 |
| 正常行为模型 | 已完成 canonical Persistence NBM |
| Persistence context | 2 秒、128 点；均值预测实际使用最后 1 个采样点 |
| 预测目标块 | 紧随 context 的 0.5 秒、32 点 |
| 预测与分类 anchor 步长 | 0.25 秒、16 点 |
| 残差表示 | `clip((x - mu) / sigma, -12, 12)` |
| 分类器历史 | 8 个按时间排列、互不重叠的 0.5 秒残差块，共 4 秒、256 点 |
| 分类标签 | 最后一个 0.5 秒目标块的 FoG 标签 |
| 分类器 | TCN-M |
| TCN-M dilation | `1,2,4,8,8,8` |
| TCN-M 感受野 | 125 点，即约 1.953 秒 |
| 主随机种子 | 42 |

本实验冻结 canonical Persistence 表示。每个 LOSO fold 直接读取原始三
传感器实验已经完成的 fold-local robust scaler、Persistence checkpoint、
逐通道 \(\sigma\)、标准化残差缓存、窗口 ID 和历史支持，不重新训练 NBM，
也不重新估计任何预处理统计量。测试受试者不参与 scaler、NBM、early
stopping 或阈值选择。

## 3. 七种传感器组合

canonical 九通道顺序固定为：

```text
0 ankle_acc_forward
1 ankle_acc_vertical
2 ankle_acc_lateral
3 thigh_acc_forward
4 thigh_acc_vertical
5 thigh_acc_lateral
6 trunk_acc_forward
7 trunk_acc_vertical
8 trunk_acc_lateral
```

| 配置 ID | 传感器 | 通道索引 | 输入形状 |
|---|---|---|---|
| `ankle` | Ankle | `0,1,2` | `[N,3,256]` |
| `thigh` | Thigh | `3,4,5` | `[N,3,256]` |
| `trunk` | Trunk | `6,7,8` | `[N,3,256]` |
| `ankle_thigh` | Ankle + Thigh | `0,1,2,3,4,5` | `[N,6,256]` |
| `ankle_trunk` | Ankle + Trunk | `0,1,2,6,7,8` | `[N,6,256]` |
| `thigh_trunk` | Thigh + Trunk | `3,4,5,6,7,8` | `[N,6,256]` |
| `all_three` | Ankle + Thigh + Trunk | `0,1,2,3,4,5,6,7,8` | `[N,9,256]` |

每个 fold 先且只先构造一次完整残差历史：

```python
x_full.shape == (n_windows, 9, 256)
```

七组输入随后由同一张量按固定通道索引切片：

```python
x_variant = np.ascontiguousarray(x_full[:, channel_indices, :])
```

因此七组必须共享完全相同的 train、validation、test anchor ID、8 块历史
ID、标签和样本数量。不得为不同传感器组合重新生成窗口或单独下采样。

## 4. 共享初始化切片策略

输入通道数不同会改变 TCN-M 首个 `1x1 Conv1d` 的形状。若分别用相同
seed 独立构造 3、6、9 通道模型，首层随机数消耗量不同，可能使后续层
也获得不同初始化。为避免将初始化差异混入传感器消融，每个 fold 应：

1. 使用该 fold 的 classifier seed 构造一个 canonical 9 通道 TCN-M；
2. 将其完整状态作为该 fold 的共享初始化模板；
3. 对每个传感器组合，仅把首层权重
   `projection.0.weight[:, channel_indices, :]` 按通道索引切片；
4. 其余卷积块、归一化层和分类头参数逐元素复制且必须完全相同。

这一策略确保不同组合的共同参数从同一初始化出发。输入通道数造成的
首层参数量差异属于模型接口本身，应在输出中明确报告：

| 输入通道数 | TCN-M 参数量 |
|---:|---:|
| 3 | 89,041 |
| 6 | 89,185 |
| 9 | 89,329 |

同一 fold 内应保存 canonical 初始化 SHA256、每组切片初始化 SHA256，
并由独立审计器重建和校验。所有组合使用相同 classifier seed 和每个
epoch 的 shuffle 规则，但各自仅根据 validation PR-AUC 选择最佳 epoch，
并仅在 validation subject 上选择 Balanced Accuracy 最优阈值。

## 5. 固定训练设置

| 项目 | 设置 |
|---|---|
| 损失 | `BCEWithLogitsLoss` |
| FoG 类别权重 | `min(sqrt(N_nonFoG / N_FoG), 6)` |
| 优化器 | AdamW |
| 学习率 | `1e-3` |
| Weight decay | `1e-4` |
| Batch size | 256 |
| 最大 epochs | 12 |
| Early-stopping patience | 4 |
| 最佳 epoch | validation PR-AUC 最大 |
| 分类阈值 | validation Balanced Accuracy 最大 |
| 训练窗口上限 | 0，即使用全部合格窗口 |
| 可复现性 | seed 42，deterministic 开启 |

## 6. 多 GPU 调度与服务器命令

LOSO fold 是不可拆分的调度单元。一张 GPU 独立准备一个 fold 的共享
九通道历史，然后依次训练该 fold 的 7 个 TCN-M。使用 7 张 GPU 时，
前 7 折并行运行；任一 GPU 完成后继续执行第 8 折。中断后使用相同命令
和输出目录即可从 fold 内未完成的模型或最近 checkpoint 续训。

```bash
python -u scripts/start_daphnet_persistence_imu_ablation_multigpu.py \
  --data-dir "/home/chb/Documents/FOG/fog_classification_framework_base/dataset/1.Daphnet Freezing of Gait Dataset/processed" \
  --source-suite-dir "$PWD/outputs/daphnet_3imu_nbm_5x4_loso_seed42" \
  --output-dir "$PWD/outputs/daphnet_persistence_h4_tcnm_imu7_loso_seed42" \
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

运行状态：

```bash
watch -n 10 python -m json.tool \
  "$PWD/outputs/daphnet_persistence_h4_tcnm_imu7_loso_seed42/multigpu_status.json"
```

单折日志位于：

```text
outputs/daphnet_persistence_h4_tcnm_imu7_loso_seed42/multigpu_logs/
```

## 7. 输出结果

根目录至少应生成：

| 文件 | 内容 |
|---|---|
| `config.json` | 冻结实验协议与实现哈希 |
| `experiment_manifest.csv` | 7 个组合及其完成状态 |
| `fold_summary.csv` | 56 个 fold-level 分类结果 |
| `aggregate_summary.csv` | 受试者宏平均和标准差，并按 PR-AUC 排名 |
| `publication_table.csv` | 论文表格 |
| `paired_pr_auc_deltas.csv` | 相对 `all_three` 的受试者配对 PR-AUC 差值与 95% CI |
| `sensor_efficiency.csv` | 通道数、参数量、输入数据量及相对完整模型比例 |
| `aggregate_metrics.json` | 完整结构化聚合结果 |
| `support_equivalence.json` | 七组窗口、历史和标签共享证明 |
| `status.json` | 预期/完成单元及最佳实验 |
| `multigpu_status.json` | 多 GPU 调度状态 |
| `AUDIT_REPORT.json` | 独立审计报告 |
| `SUITE_COMPLETE.json` | 仅正式 56 个单元全部通过审计后生成 |

论文表格应至少包含：

```text
PR-AUC
BA
Macro-F1
AUROC
FoG Sensitivity / Recall
Specificity
FoG Precision
FoG F1
Event Sensitivity
FA/h
Median Detection Delay
```

PR-AUC 差值必须以 held-out subject 为配对单位进行 bootstrap，而不能把
高度重叠的窗口当作独立样本。

## 8. 审计要求

独立审计器必须验证：

- 配置中恰好包含上述 7 种组合和 8 个 LOSO fold；
- S04、S10 始终被排除，正式实验恰好包含 56 个分类器单元；
- source suite、Persistence checkpoint、sigma、残差缓存和历史支持的
  SHA256 与 canonical 结果一致；
- 每组 `[N,C,256]` 输入严格等于 `[N,9,256]` 完整输入的指定通道切片；
- 七组的 anchor、8 块历史索引、标签、类别计数和 `pos_weight` 一致；
- 共享初始化模板、首层通道切片以及其余参数逐元素共享关系成立；
- TCN-M dilation、感受野、训练超参数和参数量正确；
- 最佳 epoch 和阈值只由 validation subject 决定；
- 保存的预测能够重算窗口指标、混淆矩阵和事件指标；
- 聚合以 held-out subject 为单位，配对置信区间可以确定性复现；
- `--allow-partial` 仅允许缺少未完成单元，不能放过已经标记完成但损坏
  或协议不兼容的结果；
- smoke run 永远不能生成正式 `SUITE_COMPLETE.json`。
