# Daphnet 参考基线实验

该 suite 为 FoG/non-FoG 二分类提供三个独立参考方法：

1. `freeze_index`：领域规则方法；
2. `tf_svm`：时频人工特征 + SVM；
3. `cnn_gru`：直接读取原始 IMU 历史的通用深度时序分类器。

它不会修改 NBM/residual suite，也不会向正在运行的 NBM 输出目录写入文件。

## 公平比较协议

- 数据：Daphnet 三个 IMU、9 个加速度通道、64 Hz；
- 先排除 S04 和 S10，再做窗口化、scaler 和 LOSO；
- 外层 8-fold LOSO：1 个 test subject；
- 按现有循环规则从剩余受试者中固定 1 个 validation subject；
- 其余 6 个 subject 用于训练；
- 输入：决策时刻之前严格因果的 4 秒 IMU 历史；
- anchor：复用最大 4 秒历史的公共支持集；
- 标签：末端 0.5 秒中 FoG 占比至少 0.5；
- 步长：0.25 秒；
- 阈值：只根据 validation subject 的 Balanced Accuracy 选择；
- test subject 只在模型、超参数和阈值全部固定后评估一次。

三个方法在每个 fold 中必须具有逐元素相同的 `window_index` 和
`y_true`。`scripts/audit_daphnet_baseline_suite.py` 会检查该条件。

## 方法定义

### Freeze Index

默认只使用 `ankle_acc_vertical` 的原始加速度，不使用 fold scaler：

```text
locomotor power = sum |FFT(x - mean(x))|², f in [0.5, 3.0) Hz
freeze power    = sum |FFT(x - mean(x))|², f in [3.0, 8.0] Hz
FI              = freeze power / locomotor power
bounded score   = FI / (1 + FI)
```

FFT 不加 taper、不补零。边界固定为半开/闭区间，3 Hz 只进入 freeze
band。bounded score 只是便于共享 `[0,1]` 阈值和指标接口，不声称是校准概率。

多通道接口通过 `--fi-channels` 提供；默认 `--fi-aggregation power_pool`
先聚合频带功率再求比值。可选 `--fi-power-gate` 仅利用 train power
候选和 validation 指标选择低运动门限。

### 时频特征 + SVM

输入先用该 fold 的六个训练 subject 中有效 non-FoG 样本拟合
median/IQR scaler。每个物理通道以及每组三轴模长提取：

- 均值、标准差、RMS、极值、峰峰值、中位数、IQR、MAD；
- 过零率、差分 RMS、偏度、峰度；
- 0.5–3、3–8、8–15 Hz 频带功率；
- Freeze Index、主频、谱质心、谱熵和相对功率；
- 同一传感器轴间相关性及不同传感器同轴相关性。

9 通道配置共产生 306 个确定性特征。`StandardScaler` 和 SVM 都只在
outer-train 上拟合。默认 `RBF-SVC + class_weight=balanced`，C 候选只由
validation PR-AUC 选择；最终概率阈值仍只由 validation Balanced Accuracy
选择。`--svm-kernel linear` 可作为线性核消融。

### CNN-GRU

输入形状为 `[batch, channel, time]`。模型由两个
`Conv1d + BatchNorm + GELU + MaxPool` block、单向 GRU、temporal
mean/max pooling 和二分类 head 组成。训练使用：

- `AdamW`；
- train-only class imbalance `pos_weight`；
- AMP（CUDA）；
- gradient clipping；
- validation PR-AUC early stopping；
- 每个 epoch 原子保存 last checkpoint，另保存 best checkpoint；
- checkpoint 包含模型、optimizer、GradScaler、RNG、early-stop 和历史状态。

## 运行接口

### 7 GPU 正式 LOSO

在服务器仓库根目录运行：

```bash
python -u scripts/start_daphnet_baseline_suite_multigpu.py \
  --data-dir "/home/chb/Documents/FOG/fog_classification_framework_base/dataset/1.Daphnet Freezing of Gait Dataset/processed" \
  --output-dir "$PWD/outputs/daphnet_reference_baselines_h4s_seed42" \
  --gpus 0-6 \
  --work-folds all \
  --max-retries 2 \
  --launch-delay 2 \
  --audit \
  --seed 42 \
  --batch-size 256 \
  --num-workers 0
```

调度器先用 CPU 初始化不可变 protocol，然后让每张 GPU 独立完成一个
fold 内的 CNN-GRU、Freeze Index 和 SVM。前 7 个 fold 完成后，空闲 GPU
自动接手第 8 个 fold。SVM 是 CPU 方法，因此一个 fold 进入 SVM 阶段后，
对应 GPU 利用率下降是正常现象。调度状态位于：

```text
outputs/daphnet_reference_baselines_h4s_seed42/multigpu_status.json
```

### 单 fold smoke test

```bash
python -u scripts/run_daphnet_baseline_suite.py \
  --data-dir "/path/to/processed" \
  --output-dir "$PWD/outputs/baseline_smoke" \
  --folds S01 \
  --device cuda \
  --classifier-epochs 1 \
  --max-train-windows 2000 \
  --svm-c-grid 1 \
  --batch-size 128
```

`--max-train-windows` 只用于 smoke/debug；正式结果保持默认 `0`，即使用全部
公共 train anchors。

### 只运行指定方法

```bash
python -u scripts/run_daphnet_baseline_suite.py \
  --data-dir "/path/to/processed" \
  --output-dir "$PWD/outputs/domain_and_ml_only" \
  --folds all \
  --methods freeze_index,tf_svm \
  --device cpu
```

### 其他输入历史

主比较固定 4 秒。需要 0.5、1 或 2 秒消融时，用独立输出目录运行
`--input-seconds 0.5`、`1` 或 `2`，不要把不同 protocol 写入同一目录。

## 输出

```text
outputs/daphnet_reference_baselines_h4s_seed42/
  config.json
  environment.json
  run_manifest.json
  status.json
  fold_summary.csv
  experiment_manifest.csv
  aggregate_metrics.json
  audit_report.json
  loso_S01/
    fold_config.json
    input_support.npz
    freeze_index/
      rule.json
      fi_features.npz
      metrics.json
      predictions.npz
      validation_predictions.npz
      predictions.csv
      DONE.json
    tf_svm/
      model.joblib
      feature_schema.json
      search_results.json
      ...
    cnn_gru/
      best.pt
      last.pt
      ...
```

每种方法均报告 Accuracy、Balanced Accuracy、Macro-F1、ROC-AUC、
PR-AUC、FoG Recall、FoG F1、Specificity、Precision、MCC，以及事件敏感度、
每小时误报事件数和检测延迟。

## 审计

```bash
python -u scripts/audit_daphnet_baseline_suite.py \
  --data-dir "/path/to/processed" \
  --output-dir "$PWD/outputs/daphnet_reference_baselines_h4s_seed42"
```

审计会验证 DONE hash、公共 anchor、标签、概率范围、阈值、窗口级指标、
事件级指标和 pooled aggregate。未完成的正式 suite 默认审计失败；调试时才使用
`--allow-partial`。
