# GRU-NBM残差G1/G2/G3严格对照实验

## 实验定义

冻结已有GRU-v1 NBM，不重新训练NBM。每个外折和随机种子读取对应的冻结GRU：

- NBM/TCN配对种子：`0, 52, 161`
- 三折：`0, 1, 2`
- 共27个TCN训练任务，封存后再运行27个测试任务
- 7张GPU动态排队，每张卡同时一个任务

三组均构造：

\[
F=[r,|r|,\Delta r]\in\mathbb{R}^{27\times128}.
\]

其中：

```text
G1: q=clip((e-b)/(sigma+1e-6),-12,12)
    r=q-mean_t(q)

G2: r=clip((e-b)/(sigma+1e-6),-12,12)

G3: r=asinh((e-b)/(sigma+1e-6))
```

G2和G3均不执行残差的第二次逐窗口逐轴中心化。

## 数据角色

```text
角色4：已有RobustScaler与GRU-NBM训练来源
角色5：已有GRU早停以及b、sigma校准来源
角色6/7：TCN训练
角色2/3：TCN早停、模型选择和阈值选择
角色0/1：所有27个分类器及阈值冻结后才允许测试
```

## TCN配置

```text
架构：与原GRU-NBM方案C相同的RepresentationTCNM
输入：[B,27,128]
Optimizer：AdamW(lr=1e-3, weight_decay=1e-4)
Loss：BCEWithLogitsLoss(pos_weight=N_role6/N_role7)
最大epoch：10
patience：2
阈值：角色2/3最大Balanced Accuracy；并列时FoG F1，再选择较高阈值
```

## 启动

在项目根目录执行：

```bash
conda activate fogbase

python -u scripts/launch_daphnet_gru_residual_g123_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --tcn-seeds 0,52,161 \
  --tcn-max-epochs 10 \
  --tcn-patience 2 \
  --phase full
```

也可以使用包装脚本：

```bash
bash scripts/run_daphnet_gru_residual_g123_7gpu.sh
```

默认读取：

```text
outputs/daphnet_gru_nbm300_C_vs_raw_tcn_ep10pat2_seedset_0_52_161/nbm_source
```

默认输出：

```text
outputs/daphnet_gru_nbm300_residual_G1_G2_G3_tcn_ep10pat2_seedset_0_52_161
```

## 先检查任务计划

```bash
python scripts/launch_daphnet_gru_residual_g123_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --dry-run
```

应显示：

```text
training_jobs = 27
evaluation_jobs_after_barrier = 27
nbm_retrained = false
paired_nbm_tcn_seeds = [0,52,161]
```

## 分阶段恢复

```bash
# 只训练TCN并生成全局测试屏障
python scripts/launch_daphnet_gru_residual_g123_7gpu.py --phase train

# 屏障已经存在后运行测试并聚合
python scripts/launch_daphnet_gru_residual_g123_7gpu.py --phase evaluate

# 仅重新聚合已完成结果
python scripts/launch_daphnet_gru_residual_g123_7gpu.py --phase aggregate
```

不加`--overwrite`时，已完成任务会自动跳过。不要在同一个输出目录中使用`--overwrite`覆盖一部分已封存任务；如需完全重跑，推荐使用新的输出目录。

## 输出文件

```text
TRAINING_BARRIER.json
experiment_config.json
run_metrics_27.csv
seed_macro_over_3folds.csv
group_summary_3seed_mean_std.csv
subject_metrics_3seed_mean_std.csv
clip_rates_by_fold_split.csv
clip_rates_per_channel.csv
summary.json
DONE.json
```

`group_summary_3seed_mean_std.csv`为主汇总：先在每个随机种子内宏平均三折，再对三个种子报告均值与总体标准差。
