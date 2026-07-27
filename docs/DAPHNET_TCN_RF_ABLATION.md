# Persistence residual_h4s 的 TCN 感受野消融

## 1. 实验目的

本实验固定上游表示为已完成核心 suite 中的：

- NBM：`Persistence`
- 分类器输入：`residual_h4s`
- 数据：Daphnet 三个 IMU、9 通道、64 Hz
- 评估：排除 S04、S10 后的 8-fold LOSO

每个 fold 直接读取已有的 `Persistence/residual_cache.npz`，不重新训练
NBM，也不重新生成残差。三个分类器只改变 TCN residual block 的
`dilation`。

| 配置 | 输出名称 | dilation | 局部卷积特征感受野 | 时间 |
|---|---|---|---:|---:|
| Local | TCN-S | 1,1,1,1,1,2 | 29 点 | 0.453125 s |
| Medium | TCN-M | 1,2,4,8,8,8 | 125 点 | 1.953125 s |
| Long | TCN-L | 1,2,4,8,16,32 | 253 点 | 3.953125 s |

感受野公式为：

```text
R = 1 + 2 × (kernel_size - 1) × sum(dilation)
  = 1 + 4 × sum(dilation), kernel_size = 3
```

三个模型均为 6 个 residual block，每个 block 含两个卷积；在 9 输入通道、
48 隐藏通道的正式配置下，三个模型的可训练参数量均为 `89,329`。

> 注意：这里的 29/125/253 是卷积特征的局部理论感受野。现有
> `ResidualTCNClassifier` 最后使用全局 mean/max pooling，因此最终读出仍可
> 汇总整个 4 秒窗口上的局部特征。Long 为 3.953125 秒，不应写成严格的完整
> 4 秒端到端感受野。

## 2. 严格控制变量

程序和独立审计器会检查以下约束：

- 每个 fold 的三个模型读取相同的 Persistence 残差缓存；
- `residual_h4s` 固定由 8 个按时间排列、互不重叠的 0.5 秒残差块组成；
- 训练、验证、测试 window id 和标签完全一致；
- 层数、隐藏宽度、卷积核、dropout、归一化、分类头完全一致；
- 优化器、学习率、weight decay、batch size、类别权重和 early stopping
  完全一致；
- 每个 fold 的三个模型使用相同 classifier seed；
- 三个模型的训练前参数逐元素相同，并保存相同的
  `initial_state_sha256`；
- 每一 epoch 的训练 shuffle 独立使用
  `classifier_seed + epoch`，避免模型执行顺序影响样本顺序；
- 阈值只在 validation subject 上按 Balanced Accuracy 选择，测试 subject
  不参与选阈值。

## 3. 服务器正式运行

前置条件是下列源 suite 已经存在：

```text
outputs/daphnet_3imu_nbm_5x4_loso_seed42
```

启动器会逐 fold 校验 Persistence NBM 的 `DONE.json`、残差缓存
`RESIDUAL_CACHE_DONE.json` 和所有文件 SHA256。它不会依赖其他四种 NBM
是否通过审计。

在服务器仓库根目录运行：

```bash
python -u scripts/start_daphnet_tcn_rf_ablation_multigpu.py \
  --data-dir "/home/chb/Documents/FOG/fog_classification_framework_base/dataset/1.Daphnet Freezing of Gait Dataset/processed" \
  --source-suite-dir "$PWD/outputs/daphnet_3imu_nbm_5x4_loso_seed42" \
  --output-dir "$PWD/outputs/daphnet_persistence_h4_tcn_rf_ablation_seed42" \
  --gpus 0-6 \
  --work-folds all \
  --max-retries 2 \
  --launch-delay 2 \
  --audit \
  --seed 42 \
  --batch-size 256 \
  --num-workers 0
```

调度方式如下：

- GPU 0–6 首先各自执行一个 fold；
- 每张 GPU 上的单个 worker 顺序完成该 fold 的 TCN-S、TCN-M、TCN-L；
- 第一个空闲的 GPU 自动接管第 8 个 fold；
- fold 失败时从最近的 epoch checkpoint 恢复，最多重试两次；
- 8 个 fold 结束后在 CPU 上生成总表，再运行独立审计器。

中断后直接执行完全相同的命令即可恢复。不要更换输出目录，也不要改变任何
科学参数。

## 4. 监控

```bash
watch -n 10 "python -m json.tool '$PWD/outputs/daphnet_persistence_h4_tcn_rf_ablation_seed42/multigpu_status.json'"
```

查看某个 fold：

```bash
tail -f outputs/daphnet_persistence_h4_tcn_rf_ablation_seed42/multigpu_logs/S01.log
```

查看 GPU：

```bash
watch -n 2 nvidia-smi
```

初始化或读取压缩残差时 GPU 利用率可能较低；进入分类器 epoch 后利用率应明显
上升。

## 5. 输出文件

```text
outputs/daphnet_persistence_h4_tcn_rf_ablation_seed42/
├── config.json
├── run_manifest.json
├── environment.json
├── status.json
├── multigpu_status.json
├── fold_summary.csv
├── experiment_manifest.csv
├── aggregate_summary.csv
├── aggregate_metrics.json
├── AUDIT_REPORT.json
├── SUITE_COMPLETE.json
├── multigpu_logs/
└── loso_S01/
    ├── fold_config.json
    ├── input_support.npz
    ├── source_provenance.json
    ├── local/
    ├── medium/
    └── long/
```

每个分类器目录包含：

```text
classifier_best.pt
classifier_last.pt
metrics.json
predictions.npz
validation_predictions.npz
predictions.csv
DONE.json
```

`fold_summary.csv` 报告：

- Accuracy
- Balanced Accuracy
- Macro-F1
- ROC-AUC
- PR-AUC
- FoG Recall
- FoG F1
- Specificity、Precision、MCC
- 事件级指标
- 阈值、best epoch、seed、输入缓存 SHA256 和初始化 SHA256

`aggregate_summary.csv` 给出三组核心指标的 8-fold subject-macro
mean/std；`aggregate_metrics.json` 另外包含 pooled 指标，以及 Medium/Long
相对 Local 的逐 fold 配对差值。

只有 24 个实验单元全部完成且独立审计通过，才会生成
`SUITE_COMPLETE.json`。

## 6. 独立审计

若需要单独重跑审计：

```bash
python -u scripts/audit_daphnet_tcn_rf_ablation.py \
  --result-dir "$PWD/outputs/daphnet_persistence_h4_tcn_rf_ablation_seed42" \
  --source-suite-dir "$PWD/outputs/daphnet_3imu_nbm_5x4_loso_seed42" \
  --data-dir "/home/chb/Documents/FOG/fog_classification_framework_base/dataset/1.Daphnet Freezing of Gait Dataset/processed"
```

审计器会重新验证源缓存、24 个 DONE 文件和 artifact SHA256，重算
validation threshold、测试指标和根目录汇总，并检查每个 fold 的三个模型
参数量、初始权重及输入支持完全一致。

## 7. 非正式冒烟测试

只验证接口时可以先跑一个 fold：

```bash
python -u scripts/start_daphnet_tcn_rf_ablation_multigpu.py \
  --data-dir "/path/to/processed" \
  --source-suite-dir "$PWD/outputs/daphnet_3imu_nbm_5x4_loso_seed42" \
  --output-dir "$PWD/outputs/rf_ablation_smoke" \
  --gpus 0 \
  --work-folds S01 \
  --allow-partial-audit \
  --audit \
  --classifier-epochs 1 \
  --classifier-patience 1 \
  --max-classifier-windows 256 \
  --batch-size 64 \
  --num-workers 0
```

冒烟测试改变了训练协议，不能与正式结果混用；正式运行必须使用新的输出目录。
