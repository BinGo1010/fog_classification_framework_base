# Daphnet GRU-NGM 鲁棒性实验：匹配 TCN 训练

本阶段只训练后端 TCN，不执行 Gaussian-noise 或 temporal-mask 测试。输入为两组已经冻结的 GRU-NGM：

- No perturbation：3 folds × 5 seeds = 15 checkpoints；
- Gaussian + Mask：3 folds × 5 seeds = 15 checkpoints。

最终输出30个一一匹配的TCN。每个TCN只训练一次，后续所有测试扰动强度都复用同一个冻结的 `GRU-NGM + role-5 calibration + TCN` 管线。

## 源checkpoint目录

两个 `ngm-root` 分别对应一个训练臂，并支持以下主要布局：

```text
<ngm-root>/
  seed_0/fold_0/checkpoints/gru_ngm_best.pt
  seed_0/fold_1/checkpoints/gru_ngm_best.pt
  ...
  seed_52161/fold_2/checkpoints/gru_ngm_best.pt
```

checkpoint 名也可以是既有工程使用的 `gru_nbm_best.pt`。每个 fold 目录还必须有 `scaler_role4.json`；如果没有该文件，则必须有包含 `scaler` 字段的 `nbm_frozen.json`。

## 训练契约

- GRU-NGM：完全冻结，不继续更新；
- role 5：clean 数据重新计算 `b` 和 `sigma`，其中 Scheme C 只使用 `sigma`；
- TCN训练：clean roles 6/7；
- TCN验证与checkpoint选择：clean roles 2/3 AP；
- TCN结构：Scheme C `[r, |r|, Δr]`，27通道；
- TCN优化：batch 128，AdamW(`lr=1e-3`, `weight_decay=1e-4`)；
- 最大5 epochs，patience 2；
- roles 0/1 在30个TCN全部冻结前不读取。

同一fold/seed下的两组TCN使用完全相同的初始权重、训练随机种子、数据角色和class weight。

## 先执行dry-run

```bash
python -u scripts/launch_daphnet_gru_ngm_robustness_tcn_8gpu.py \
  --data-dir "$PWD/dataset/1.Daphnet Freezing of Gait Dataset/processed_NBM" \
  --none-ngm-root "/path/to/no_perturbation_nbm_source" \
  --gaussian-mask-ngm-root "/path/to/gaussian_mask_nbm_source" \
  --output-root "$PWD/outputs/daphnet_gru_ngm_robustness_matched_tcn" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --dry-run
```

dry-run会逐个核验30个checkpoint、seed、模型结构、scaler、数据指纹和文件哈希，但不会训练TCN。

## 8 GPU正式训练

```bash
mkdir -p logs
nohup bash scripts/run_daphnet_gru_ngm_robustness_tcn_8gpu.sh \
  "/path/to/no_perturbation_nbm_source" \
  "/path/to/gaussian_mask_nbm_source" \
  "$PWD/outputs/daphnet_gru_ngm_robustness_matched_tcn" \
  "$PWD/dataset/1.Daphnet Freezing of Gait Dataset/processed_NBM" \
  > logs/daphnet_gru_ngm_robustness_tcn_8gpu.log 2>&1 &
```

如果服务器使用独立环境：

```bash
PYTHON_BIN=/path/to/conda/env/bin/python \
  bash scripts/run_daphnet_gru_ngm_robustness_tcn_8gpu.sh \
  "/path/to/no_perturbation_nbm_source" \
  "/path/to/gaussian_mask_nbm_source"
```

## 输出

单个TCN示例：

```text
outputs/daphnet_gru_ngm_robustness_matched_tcn/
  runs/gaussian_mask/fold_0/seed_0/
    checkpoints/tcn.pt
    logs/tcn_history.csv
    calibration_role5.json
    FROZEN_TCN.json
    DONE_TCN.json
```

全量完成后生成：

- `EXPERIMENT_PLAN.json`：冻结30个源checkpoint及全部训练参数；
- `TCN_PAIRING_AUDIT.json`：核验两臂TCN配对初始化与数据一致性；
- `TCN_TRAINING_SUMMARY.csv`：30个TCN的clean验证AP和checkpoint哈希；
- `TCN_TRAINING_BARRIER.json`：允许下一阶段读取roles 0/1并执行鲁棒性评估；
- `DONE_TCN_TRAINING.json`：30个TCN全部完成标记。

脚本可安全重跑：哈希验证通过的完整任务自动跳过，未完成任务重新执行。
