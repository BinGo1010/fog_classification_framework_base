# Conv-TCN NBM高斯增强对照：仅方案C

## 唯一实验变量

本实验与已经完成的
`daphnet_conv_tcn_nbm200_BC_3seed_seed20260807`中的方案C配对比较，唯一改变为
NBM角色4训练增强：

| 项目 | 原200-epoch方案C | 本对照实验C-Gaussian |
|---|---:|---:|
| 完整clean窗口 | 80% | 40% |
| Gaussian窗口 | 0% | 40%，std=0.04 |
| 时间Mask窗口 | 20% | 20%，全轴、4–8点 |

每个角色4窗口每次进入模型时只属于上述一种类别，不会同时添加Gaussian和Mask。
比例为按窗口独立随机抽样的期望比例，每个epoch的实际三类数量写入训练日志。

以下条件不变：

```text
Conv-TCN NBM架构：[B,9,128] -> [B,16,32] -> [B,9,128]
逐窗口逐轴中心化
SmoothL1(beta=1.0)
AdamW(lr=1e-3, weight_decay=1e-4)
ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)
batch size=128
max epoch=200
early stopping patience=20
gradient clipping=1.0
角色5完整clean输入，不Mask、不加噪声
恢复角色5最低验证损失权重后计算b和sigma
```

分类阶段只运行方案C：

```text
q = clip(e/(sigma+1e-6), -12, 12)
r = q - mean_t(q)
F = [r, abs(r), delta_t(r)] in [B,27,128]
```

角色6/7训练TCN，角色2/3完成早停、模型选择和阈值选择；9个C分类器及阈值
全部冻结后才运行角色0/1测试。

## 7卡服务器启动

```bash
cd /path/to/fog-merged
bash scripts/launch_daphnet_conv_tcn_nbm200_gaussian_c_7gpu.sh
```

完整命令：

```bash
python scripts/launch_daphnet_conv_tcn_nbm200_gaussian_c_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --tcn-seeds 20260807,20260808,20260809 \
  --clean-probability 0.40 \
  --gaussian-probability 0.40 \
  --gaussian-std 0.04 \
  --mask-probability 0.20 \
  --mask-min-samples 4 \
  --mask-max-samples 8 \
  --nbm-max-epochs 200 \
  --nbm-patience 20 \
  --nbm-learning-rate 1e-3 \
  --tcn-max-epochs 30 \
  --tcn-patience 6 \
  --output-root outputs/daphnet_conv_tcn_nbm200_C_gaussian40_3seed_seed20260807
```

预检但不训练：

```bash
python scripts/launch_daphnet_conv_tcn_nbm200_gaussian_c_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --dry-run
```

NBM阶段只有3个独立任务，最多同时使用3张卡；分类阶段有
`3 folds × 1 group C × 3 seeds = 9`个任务，由7张卡动态领取。

## 断点续跑

```bash
# 只训练三折NBM
python scripts/launch_daphnet_conv_tcn_nbm200_gaussian_c_7gpu.py --phase nbm

# 训练9个C分类器并建立测试屏障
python scripts/launch_daphnet_conv_tcn_nbm200_gaussian_c_7gpu.py --phase train

# 屏障建立后测试并汇总
python scripts/launch_daphnet_conv_tcn_nbm200_gaussian_c_7gpu.py --phase evaluate

# 仅重新汇总
python scripts/launch_daphnet_conv_tcn_nbm200_gaussian_c_7gpu.py --phase aggregate
```

默认自动跳过具有完成标记的任务。正常续跑不要使用`--overwrite`。

## 输出

```text
outputs/daphnet_conv_tcn_nbm200_C_gaussian40_3seed_seed20260807/
  launch_plan.json
  NBM_BARRIER.json
  nbm_source/fold_0..2/
    config.json
    nbm_frozen.json
    DONE_NBM.json
    checkpoints/conv_tcn_nbm_best.pt
    logs/conv_tcn_nbm_history.csv
  c_results/
    TRAINING_BARRIER.json
    experiment_config.json
    run_metrics_9.csv
    seed_macro_over_3folds.csv
    group_summary_3seed_mean_std.csv
    subject_metrics_3seed_mean_std.csv
    summary.json
    DONE.json
```

最终应将`c_results/group_summary_3seed_mean_std.csv`与原实验
`daphnet_conv_tcn_nbm200_BC_3seed_seed20260807/bc_results/`中的方案C按相同统计口径
配对比较。
