# Conv-TCN NBM（200 epoch）→ B/C 三种子实验

## 实验目的

本实验只检验“当前 Conv-TCN NBM 是否因 50 epoch 上限而训练不足”。因此除
NBM 最大训练轮数和早停耐心外，其余 NBM 配置保持不变：

```text
架构：当前 Conv-TCN Autoencoder NBM
输入：[B,9,128]
瓶颈：[B,16,32]
训练数据：角色4 clean Non-FoG
验证数据：角色5 clean Non-FoG，不施加Mask
训练增强：80%完整窗口，20%轻度时间Mask
损失：SmoothL1(beta=1.0)
优化器：AdamW(lr=1e-3, weight_decay=1e-4)
Scheduler：ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)
Batch size：128
最大epoch：200
Early stopping patience：20
梯度裁剪：1.0
Checkpoint：最低角色5 validation SmoothL1
校准：恢复最佳checkpoint后，使用角色5计算b和sigma
```

没有使用复合损失，也没有把学习率改成 `3e-4`。

## B/C残差

令：

```text
e = X_scaled_centered - X_hat
```

方案B：

```text
q = clip((e-b)/(sigma+1e-6), -12, 12)
r = q - mean_t(q)
```

方案C：

```text
q = clip(e/(sigma+1e-6), -12, 12)
r = q - mean_t(q)
```

两组TCN输入均为：

```text
F = [r, abs(r), delta_t(r)] in [B,27,128]
```

TCN继续使用角色6/7训练、角色2/3早停和选阈值，角色0/1只在所有18个分类器
及阈值冻结并生成全局屏障后运行。

## 七卡启动

在服务器项目根目录运行：

```bash
bash scripts/launch_daphnet_conv_tcn_nbm200_bc_7gpu.sh
```

或直接运行Python调度器：

```bash
python scripts/launch_daphnet_conv_tcn_nbm200_bc_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --tcn-seeds 20260807,20260808,20260809 \
  --nbm-max-epochs 200 \
  --nbm-patience 20 \
  --nbm-learning-rate 1e-3 \
  --tcn-max-epochs 30 \
  --tcn-patience 6 \
  --output-root outputs/daphnet_conv_tcn_nbm200_BC_3seed_seed20260807
```

运行前可以检查任务计划而不训练：

```bash
python scripts/launch_daphnet_conv_tcn_nbm200_bc_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --dry-run
```

NBM阶段只有三折，因此同时使用3张GPU；进入分类阶段后共有
`3 folds × 2 groups × 3 seeds = 18`个训练任务，由7张GPU动态领取。

## 断点续跑

默认跳过已经具有完成标记的任务，不需要添加 `--overwrite`。

```bash
# 只训练/冻结三折NBM
python scripts/launch_daphnet_conv_tcn_nbm200_bc_7gpu.py --phase nbm

# 使用已经冻结的NBM训练18个B/C分类器，并建立测试屏障
python scripts/launch_daphnet_conv_tcn_nbm200_bc_7gpu.py --phase train

# 屏障存在后执行测试并汇总
python scripts/launch_daphnet_conv_tcn_nbm200_bc_7gpu.py --phase evaluate

# 仅重新汇总已有测试结果
python scripts/launch_daphnet_conv_tcn_nbm200_bc_7gpu.py --phase aggregate
```

`--overwrite`会重新训练对应阶段，正式运行后不要无意添加。

## 输出目录

```text
outputs/daphnet_conv_tcn_nbm200_BC_3seed_seed20260807/
  launch_plan.json
  NBM_BARRIER.json
  nbm_source/
    fold_0/
      config.json
      nbm_frozen.json
      DONE_NBM.json
      checkpoints/conv_tcn_nbm_best.pt
      logs/conv_tcn_nbm_history.csv
      conv_tcn_nbm_training_validation.png/.svg/.pdf
    fold_1/
    fold_2/
  bc_results/
    TRAINING_BARRIER.json
    experiment_config.json
    run_metrics_18.csv
    seed_macro_over_3folds.csv
    group_summary_3seed_mean_std.csv
    subject_metrics_3seed_mean_std.csv
    clip_rates_by_fold_split.csv
    clip_rates_per_channel.csv
    summary.json
    DONE.json
    runs/fold_*/group_B|C/seed_*/
  logs/
    nbm/
    classifier_train/
    classifier_evaluate/
```

主汇总口径仍然是：每个TCN种子先对三折取宏平均，再对三个种子级结果报告
均值和总体标准差。

## 关键保护

- RobustScaler只能接触角色4覆盖的唯一原始采样点。
- NBM只使用角色4更新参数；角色5只验证、早停和计算 `b/sigma`。
- B/C在同一折共享同一个冻结NBM、Scaler、`b`和`sigma`。
- 相同折和TCN种子的B/C使用相同初始化和batch顺序。
- `pos_weight=N_role6/N_role7`。
- 阈值仅由角色2/3的Balanced Accuracy确定；并列时依次选择更高FoG F1、
  更高阈值。
- 18个分类器和阈值未全部冻结前，测试worker拒绝读取角色0/1。
