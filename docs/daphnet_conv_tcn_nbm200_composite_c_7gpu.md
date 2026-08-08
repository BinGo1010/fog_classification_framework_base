# Conv-TCN NBM复合损失 → 方案C：7卡运行说明

## 已冻结的实验定义

```text
NBM架构：当前Conv-TCN Autoencoder，latent=[B,16,32]
角色4：拟合RobustScaler并训练NBM
角色5：无增强验证、早停、最佳checkpoint选择及b/sigma校准

角色4每个epoch动态互斥分配：
  40% 完整窗口
  40% Gaussian(std=0.04)
  20% 全轴时间Mask，长度4–8点，位置动态随机

训练和角色5验证损失：
  0.70 SmoothL1(beta=1.0)
  + 0.15 (1-PearsonCorr_t)
  + 0.15 SmoothL1(first differences, beta=1.0)

AdamW：lr=1e-3，weight_decay=1e-4
Scheduler：ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)
NBM：max_epoch=200，early_stopping_patience=20
最佳模型：角色5无增强复合损失最低
```

只运行方案C：

```text
e = X_scaled_centered - X_hat
q = clip(e/(sigma+1e-6), -12, 12)
r = q - mean_t(q)
F = [r, abs(r), delta_t(r)] in [B,27,128]
```

TCN使用角色6/7训练，角色2/3早停和选阈值。3折×3种子共9个分类器全部
冻结并生成全局屏障后，才允许访问角色0/1。

## 正式启动

```bash
cd /document/home_mirror/chb/fog_classification_framework_base
conda activate fogbase
mkdir -p logs

nohup bash scripts/launch_daphnet_conv_tcn_nbm200_composite_c_7gpu.sh \
  > logs/daphnet_conv_tcn_nbm200_composite_c.log 2>&1 &

echo $!
```

如果服务器项目目录不同，只替换第一行路径。

## 启动前检查

```bash
python scripts/launch_daphnet_conv_tcn_nbm200_composite_c_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --dry-run
```

## 监控

```bash
tail -f logs/daphnet_conv_tcn_nbm200_composite_c.log
nvidia-smi
```

NBM阶段只有3个折任务，因此同时使用3张GPU；方案C分类阶段有9个任务，最多
同时使用7张GPU。

## 断点续跑

直接重新执行正式启动命令即可。程序默认跳过具有完成标记的任务，不要添加
`--overwrite`。

也可以按阶段恢复：

```bash
# 只训练和冻结三折NBM
bash scripts/launch_daphnet_conv_tcn_nbm200_composite_c_7gpu.sh --phase nbm

# 使用冻结NBM训练9个C分类器，并建立测试屏障
bash scripts/launch_daphnet_conv_tcn_nbm200_composite_c_7gpu.sh --phase train

# 屏障建立后执行测试和汇总
bash scripts/launch_daphnet_conv_tcn_nbm200_composite_c_7gpu.sh --phase evaluate

# 只重新汇总已有结果
bash scripts/launch_daphnet_conv_tcn_nbm200_composite_c_7gpu.sh --phase aggregate
```

## 输出

```text
outputs/daphnet_conv_tcn_nbm200_composite_C_3seed_seed20260807/
  launch_plan.json
  NBM_BARRIER.json
  nbm_source/fold_0..2/
    config.json
    nbm_frozen.json
    DONE_NBM.json
    checkpoints/conv_tcn_nbm_best.pt
    logs/conv_tcn_nbm_history.csv
    conv_tcn_nbm_training_validation.png/.svg/.pdf
  c_results/
    TRAINING_BARRIER.json
    run_metrics_9.csv
    group_summary_3seed_mean_std.csv
    subject_metrics_3seed_mean_std.csv
    summary.json
    DONE.json
```
