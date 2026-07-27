# Persistence residual_h4s 的分类器架构对比

## 1. 实验目的

本实验固定上游正常行为建模和残差表示，只替换下游二分类器，用于比较不同
时序归纳偏置对 FoG 识别的影响。

- 数据：Daphnet 三个 IMU、9 通道、64 Hz；
- NBM：`Persistence`；
- 分类器输入：`residual_h4s`，形状为 `[9, 256]`；
- 历史构造：8 个按时间顺序排列、互不重叠的 0.5 秒残差块，共 4 秒；
- 评估：排除 S04、S10 后，对
  `S01,S02,S03,S05,S06,S07,S08,S09` 执行严格 8-fold LOSO；
- 实验规模：4 个分类器 × 8 个测试 subject，共 32 个实验单元。

每个 fold 直接复用已有核心 suite 中完成并校验过的
`persistence/residual_cache.npz`，不会重新训练 NBM，也不会重新定义残差。
因此本实验唯一预期改变的变量是下游分类器架构。

本 suite 只训练上述四种替代分类器，不会覆盖或重新训练已有 TCN；正式完成数
固定为 32。已有 TCN 结果可作为单独的参考基线，但不计入本 suite 的完成状态。

## 2. 四种分类器

以下参数量是在正式输入 `9 × 256`、dropout 为 `0.15` 时得到的默认值。

| 目录名 | 显示名称 | 默认结构 | 参数量 | 主要问题 |
|---|---|---|---:|---|
| `mlp` | MLP | 展平 2304 维；隐藏层 40；GELU；二分类头 | 92,241 | 不显式建模时间局部性时，残差是否仍容易被浅层全局读出 |
| `cnn1d` | Multi-scale 1D-CNN | 3/7/15 三个卷积分支，各 32 通道；128 通道融合；mean/max pooling；64 维分类头 | 85,857 | 多尺度局部 FoG 模式是否重要 |
| `gru` | GRU | 单向 2 层 GRU；隐藏维 96；最终隐状态；32 维分类头 | 90,035 | 残差状态随时间的演化是否重要 |
| `transformer` | Lightweight Transformer | 64 维；4 头；2 层；FFN 128；可学习位置编码与 CLS token；32 维分类头 | 86,355 | 长距离依赖和注意力读出是否有效 |

四个模型的参数量处在相近范围，避免把明显更大的容量误当成架构优势。所有模型
均接收同一份 `[batch, 9, 256]` 输入，并输出每个窗口的一个二分类 logit。

> **MLP 的解释边界：** 当前 MLP 含一个隐藏层和 GELU，是“浅层、无显式时序
> 归纳偏置”的非线性读出，并不是严格的线性探针。因此，MLP 表现好可以说明残差
> 易于被简单全局映射区分，但不能据此声称残差已经线性可分。若论文需要严格的
> “线性可分”结论，应另设单层 Linear/Logistic Regression probe。

## 3. 严格控制变量

程序和独立审计器会检查以下实验约束：

- 四个分类器读取同一份 Persistence 残差缓存；
- 四个分类器使用完全相同的 train、validation、test 窗口 ID 和标签；
- 每个 fold 的训练样本、验证 subject 和测试 subject 完全相同；
- 默认使用全部训练锚点；`--max-classifier-windows 0` 表示不截断；
- 同一 fold 的四个模型使用相同 `classifier_seed`；
- 每个 epoch 的训练 shuffle 固定为
  `classifier_seed + epoch`，不依赖模型执行顺序；
- 损失函数均为 `BCEWithLogitsLoss`，类别权重为
  `min(sqrt(n_non-FOG / n_FoG), 6)`；
- 优化器均为 AdamW，默认学习率 `1e-3`、weight decay `1e-4`；
- 默认 batch size 为 256、最大 12 epochs、patience 为 4；
- early stopping 只依据 validation PR-AUC；
- 最终二分类阈值只在 validation subject 上按 Balanced Accuracy 选择；
- 测试 subject 不参与训练、early stopping 或阈值选择；
- 每个模型的架构、参数量、初始权重 SHA256、输入支持 SHA256 和源残差
  SHA256 都会写入结果文件。

由于四种架构的张量形状不同，不要求不同架构拥有逐元素相同的初始权重；公平性
约束是相同的 fold seed、确定性设置、样本支持、epoch shuffle 和训练协议。

## 4. 服务器 7-GPU 正式运行

前置条件是核心 NBM suite 已经存在：

```text
outputs/daphnet_3imu_nbm_5x4_loso_seed42
```

在服务器仓库根目录
`/document/home_mirror/chb/fog_classification_framework_base` 运行：

```bash
python -u scripts/start_daphnet_residual_classifier_suite_multigpu.py \
  --data-dir "/home/chb/Documents/FOG/fog_classification_framework_base/dataset/1.Daphnet Freezing of Gait Dataset/processed" \
  --source-suite-dir "$PWD/outputs/daphnet_3imu_nbm_5x4_loso_seed42" \
  --output-dir "$PWD/outputs/daphnet_persistence_h4_classifier4_loso_seed42" \
  --gpus 0-6 \
  --work-folds all \
  --max-retries 2 \
  --launch-delay 2 \
  --audit \
  --seed 42 \
  --batch-size 256 \
  --num-workers 0
```

调度单位是完整 LOSO fold：

- GPU 0–6 首先各自运行一个 fold；
- 每张 GPU 上的 worker 依次训练 MLP、1D-CNN、GRU、Transformer；
- 第一张空闲 GPU 自动接管第 8 个 fold；
- worker 失败时从最近的 classifier epoch checkpoint 恢复，最多重试两次；
- 所有 fold 完成后，在 CPU 上重建汇总文件，再执行独立审计。

中断后直接重新执行完全相同的命令即可恢复。必须保留相同输出目录和所有科学
参数；如果更改模型、seed、epoch、学习率等协议字段，程序会拒绝把新结果混入
已有目录。调度器还会使用输出目录锁，阻止两个调度进程同时写入同一实验。

## 5. 运行监控

查看调度器总体状态：

```bash
watch -n 10 "python -m json.tool '$PWD/outputs/daphnet_persistence_h4_classifier4_loso_seed42/multigpu_status.json'"
```

查看某个 fold 的实时日志：

```bash
tail -f outputs/daphnet_persistence_h4_classifier4_loso_seed42/multigpu_logs/S01.log
```

查看 GPU：

```bash
watch -n 2 nvidia-smi
```

`multigpu_status.json` 记录初始化、每个 fold 的 GPU/PID/尝试次数、finalize 和
audit 返回码。`status.json` 记录科学实验单元完成数，正式完整结果应为
`completed_fold_cells = 32`。初始化、数据校验和汇总阶段主要占用 CPU，GPU
利用率较低是正常现象。

## 6. 输出文件

```text
outputs/daphnet_persistence_h4_classifier4_loso_seed42/
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
├── worker_environments/
├── multigpu_logs/
│   ├── initialize.log
│   ├── S01.log
│   ├── ...
│   ├── finalize.log
│   └── audit.log
└── loso_S01/
    ├── fold_config.json
    ├── input_support.npz
    ├── source_provenance.json
    ├── mlp/
    ├── cnn1d/
    ├── gru/
    └── transformer/
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

关键汇总文件：

- `fold_summary.csv`：逐 subject、逐分类器的 Accuracy、Balanced Accuracy、
  Macro-F1、ROC-AUC、PR-AUC、FoG Recall、FoG F1、Specificity、Precision、
  MCC、事件敏感度、每小时误报事件数、检测延迟、阈值和训练元数据；
- `experiment_manifest.csv`：四种架构的参数量、预期/已完成 fold 和状态；
- `aggregate_summary.csv`：四种分类器的 8-fold subject-macro 核心指标
  mean/std；
- `aggregate_metrics.json`：subject-macro、pooled 指标，以及 CNN、GRU、
  Transformer 相对 MLP 的逐 fold 配对差值；
- `predictions.csv`：窗口级概率和最终预测，可用于后续误差分析；
- `AUDIT_REPORT.json`：独立审计的检查结果；
- `SUITE_COMPLETE.json`：只有 32 个实验单元全部完成且审计通过时才生成。

## 7. 独立重跑审计

需要单独重跑审计时：

```bash
python -u scripts/audit_daphnet_residual_classifier_suite.py \
  --result-dir "$PWD/outputs/daphnet_persistence_h4_classifier4_loso_seed42" \
  --source-suite-dir "$PWD/outputs/daphnet_3imu_nbm_5x4_loso_seed42" \
  --data-dir "/home/chb/Documents/FOG/fog_classification_framework_base/dataset/1.Daphnet Freezing of Gait Dataset/processed"
```

审计器会重新验证源 Persistence 缓存、输入支持、32 个 `DONE.json` 及 artifact
SHA256，并重算 validation 阈值、窗口级/事件级测试指标和根目录汇总。正式结果
应同时满足：

```text
multigpu_status.json: status = complete
status.json: status = complete, completed_fold_cells = 32
AUDIT_REPORT.json: status = pass, full_complete = true
SUITE_COMPLETE.json: status = complete
```

## 8. 单 fold CPU 冒烟测试

以下命令只用于验证接口、checkpoint、输出和审计链路，不需要 GPU。初始化、
worker 和 finalize 必须使用完全相同的科学参数：

```bash
SMOKE_OUT="$PWD/outputs/daphnet_classifier4_cpu_smoke"
DATA_DIR="/home/chb/Documents/FOG/fog_classification_framework_base/dataset/1.Daphnet Freezing of Gait Dataset/processed"
SOURCE_DIR="$PWD/outputs/daphnet_3imu_nbm_5x4_loso_seed42"

python -u scripts/run_daphnet_residual_classifier_suite.py \
  --data-dir "$DATA_DIR" \
  --source-suite-dir "$SOURCE_DIR" \
  --output-dir "$SMOKE_OUT" \
  --finalize-only \
  --device cpu \
  --seed 42 \
  --classifier-epochs 1 \
  --classifier-patience 1 \
  --max-classifier-windows 32 \
  --batch-size 32 \
  --num-workers 0 \
  --no-amp \
  --debug-small-models

python -u scripts/run_daphnet_residual_classifier_suite.py \
  --data-dir "$DATA_DIR" \
  --source-suite-dir "$SOURCE_DIR" \
  --output-dir "$SMOKE_OUT" \
  --worker-fold S01 \
  --device cpu \
  --seed 42 \
  --classifier-epochs 1 \
  --classifier-patience 1 \
  --max-classifier-windows 32 \
  --batch-size 32 \
  --num-workers 0 \
  --no-amp \
  --debug-small-models

python -u scripts/run_daphnet_residual_classifier_suite.py \
  --data-dir "$DATA_DIR" \
  --source-suite-dir "$SOURCE_DIR" \
  --output-dir "$SMOKE_OUT" \
  --finalize-only \
  --device cpu \
  --seed 42 \
  --classifier-epochs 1 \
  --classifier-patience 1 \
  --max-classifier-windows 32 \
  --batch-size 32 \
  --num-workers 0 \
  --no-amp \
  --debug-small-models

python -u scripts/audit_daphnet_residual_classifier_suite.py \
  --result-dir "$SMOKE_OUT" \
  --source-suite-dir "$SOURCE_DIR" \
  --data-dir "$DATA_DIR" \
  --allow-partial
```

`--debug-small-models` 会显著缩小四种架构，且上面的命令还把训练集限制到 32 个
窗口并只训练 1 epoch。因此该输出**不能报告、不能与正式结果比较，也不能复用为
正式训练的输出目录**。正式 7-GPU 实验不要传入 `--debug-small-models`、
`--max-classifier-windows` 或缩短 epoch 的参数。
