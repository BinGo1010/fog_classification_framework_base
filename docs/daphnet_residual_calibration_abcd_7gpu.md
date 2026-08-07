# Daphnet残差校准A–D实验：7卡服务器运行说明

## 1. 实验目标

比较残差位置校准、尺度校准和逐窗口中心化。四组使用完全相同的数据角色、冻结NBM、TCN-M架构、优化器、batch size、类别权重、早停规则、阈值规则和随机种子。

所有组最终输入均为：

```text
F = [r, |r|, Δr] ∈ R^(B×27×128)
Δr[0] = 0
Δr[t] = r[t] - r[t-1]
```

| 组 | 残差定义 | Clip | 窗口中心化 |
|---|---|---|---|
| A | `(e-b)/(sigma+1e-6)` | ±12 | 否 |
| B | `(e-b)/(sigma+1e-6)` | ±12 | Clip后中心化 |
| C | `e/(sigma+1e-6)` | ±12 | Clip后中心化 |
| D | `e` | 无 | 直接中心化 |

其中：

```text
e = X_scaled_centered - X_hat
b_c = median(e_c)
sigma_c = max(1.4826 * median(|e_c-b_c|), 0.05)
```

## 2. 固定NBM

本程序不重新训练NBM，直接读取上一项正式实验中的每折冻结结果：

```text
outputs/daphnet_processed_NBM_conv_tcn_residual_repr_compare_seed20260807/
  fold_0/checkpoints/conv_tcn_nbm_best.pt
  fold_0/nbm_frozen.json
  fold_1/...
  fold_2/...
```

每个外层折拥有自己的角色4/5数据，因此每折对应一个合法NBM；在同一折内，A–D和3个TCN随机种子严格共享同一个NBM checkpoint、Scaler、`b`和`sigma`。程序记录并核对文件SHA-256。

## 3. 数据角色和测试门控

- 角色4/5：只来自已有的冻结NBM和C1校准文件。
- 角色6/7：训练TCN，`pos_weight=N_role6/N_role7`。
- 角色2/3：TCN早停、最佳模型选择和分类阈值选择。
- 角色0/1：所有36个分类器及阈值全部冻结后才允许访问。

训练阶段共有：

```text
3 folds × 4 groups × 3 TCN seeds = 36 jobs
```

36个训练任务全部完成后，程序验证：

- 每折所有任务使用相同NBM checkpoint和校准文件；
- 同一折、同一随机种子的A–D具有完全相同的TCN初始权重；
- 四组使用相同`pos_weight`；
- 每个分类器checkpoint及验证阈值已经冻结；
- 训练阶段没有访问角色0/1。

验证通过后生成`TRAINING_BARRIER.json`。没有该文件，测试worker会拒绝运行。随后才用7卡并行执行36个角色0/1测试任务。

## 4. TCN训练配置

```text
输入：[B,27,128]
TCN通道：27→32→64→64→128
膨胀率：1,2,4,8
Global Average Pooling
Dropout：0.3
Linear：128→1

损失：BCEWithLogitsLoss(pos_weight=N_role6/N_role7)
优化器：AdamW(lr=1e-3, weight_decay=1e-4)
batch size：128
梯度裁剪：1.0
最大epoch：30
patience：6
模型选择：角色2/3 PR-AUC
阈值：角色2/3 Balanced Accuracy最大
平局规则：FoG F1更高，然后选择更高阈值
```

默认3个TCN随机种子：

```text
20260807, 20260808, 20260809
```

同一个种子在A–D之间使用相同初始化和相同batch顺序。

## 5. 7卡启动命令

推荐使用Python调度器：

```bash
cd /path/to/fog-merged

python scripts/launch_daphnet_residual_calibration_abcd_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --tcn-seeds 20260807,20260808,20260809 \
  --nbm-source-root outputs/daphnet_processed_NBM_conv_tcn_residual_repr_compare_seed20260807 \
  --output-root outputs/daphnet_residual_calibration_ABCD_3seed_seed20260807 \
  --tcn-max-epochs 30 \
  --tcn-patience 6
```

Linux shell包装器：

```bash
GPU_IDS_CSV=0,1,2,3,4,5,6 \
TCN_SEEDS=20260807,20260808,20260809 \
bash scripts/launch_daphnet_residual_calibration_abcd_7gpu.sh
```

调度器采用动态任务队列，最多同时运行7个独立任务，每张卡一个进程。一个任务完成后立即领取下一个任务，因此7张卡会同时参与训练，而不是只使用前3张。

服务器预检：

```bash
python scripts/launch_daphnet_residual_calibration_abcd_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --dry-run
```

## 6. 断点续跑

默认会跳过已经完成的训练和测试任务。可以按阶段恢复：

```bash
# 只训练36个分类器并建立测试门控
python scripts/launch_daphnet_residual_calibration_abcd_7gpu.py --phase train

# 门控已经存在时，只进行测试和汇总
python scripts/launch_daphnet_residual_calibration_abcd_7gpu.py --phase evaluate

# 测试任务全部完成时，只重新汇总
python scripts/launch_daphnet_residual_calibration_abcd_7gpu.py --phase aggregate
```

`--overwrite`会重新运行对应阶段，请避免在正式实验完成后无意使用。

## 7. 指标统计口径

每个组首先得到每个随机种子的三折宏平均：

```text
metric_group,seed = mean(metric_fold0, metric_fold1, metric_fold2)
```

随后对3个seed-level宏平均计算均值和总体标准差：

```text
mean ± population std, n_seeds=3
```

这是`group_summary_3seed_mean_std.csv`中的主结果。统一输出：

- Accuracy
- Balanced Accuracy
- FoG Precision
- FoG Recall
- Specificity
- FoG F1
- PR-AUC
- AUROC

逐被试结果采用同样的“先三折宏平均，再跨3个种子计算均值±标准差”口径。

## 8. Clip rate定义

A、B、C在执行`np.clip`之前计算：

```text
clip_mask = abs(unclipped_calibrated_residual) > 12
clip_rate = clipped_points / all_points
```

分别记录：

- overall clip rate
- non-FoG clip rate
- FoG clip rate
- 9个通道各自的overall/non-FoG/FoG clip rate，同时记录通道索引和传感器轴名称
- 角色6/7训练、角色2/3验证、角色0/1测试及三者合并结果

A与B在中心化之前具有完全相同的校准残差，因此程序会强制检查两组clip rate一致。D不裁剪，标记为`not applicable`。

## 9. 主要输出

```text
output_root/
  TRAINING_BARRIER.json
  experiment_config.json
  run_metrics_36.csv
  seed_macro_over_3folds.csv
  group_summary_3seed_mean_std.csv
  subject_metrics_3seed_mean_std.csv
  clip_rates_by_fold_split.csv
  clip_rates_per_channel.csv
  summary.json
  DONE.json
  logs/
    launch_plan.json
    train/*.out.log, *.err.log
    evaluate/*.out.log, *.err.log
  runs/
    fold_0/group_A/seed_20260807/
      checkpoints/tcn.pt
      logs/tcn_history.csv
      frozen_validation.json
      metrics.json
      test_predictions.csv
      test_probabilities.npz
      DONE_TRAIN.json
      DONE_TEST.json
    ...其余35个任务...
```

## 10. 代码文件

- `scripts/run_daphnet_residual_calibration_abcd.py`：A–D特征、单任务训练、测试门控、测试和汇总。
- `scripts/launch_daphnet_residual_calibration_abcd_7gpu.py`：7卡动态任务调度器。
- `scripts/launch_daphnet_residual_calibration_abcd_7gpu.sh`：Linux启动包装器。
