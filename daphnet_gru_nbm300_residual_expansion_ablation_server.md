# Daphnet GRU-NBM残差扩展消融实验（7卡）

## 唯一消融变量

完整组与消融组共享相同的数据角色、RobustScaler、GRU-NBM、残差校准、
TCN骨干、优化参数、随机种子、验证阈值规则和永久测试屏障。唯一差异是TCN输入：

```text
FULL_C:
e = X - X_hat
q = clip(e / (sigma + 1e-6), -12, 12)
r = q - mean_t(q)
F = [r, abs(r), delta_t(r)]             [B, 27, 128]

RESIDUAL_R:
e = X - X_hat
q = clip(e / (sigma + 1e-6), -12, 12)
r = q - mean_t(q)
F = r                                    [B, 9, 128]
```

`RESIDUAL_R`仅删除`abs(r)`和`delta_t(r)`，不删除NBM、不更改残差标准化，
也不更改逐窗口逐轴残差中心化。

## 固定实验配置

- 数据集：`dataset/1.Daphnet Freezing of Gait Dataset/processed_NBM`
- 采样率：64 Hz
- 窗口：128点（2秒）
- 步长：64点（1秒）
- 折数：3
- 随机种子：`0,52,161,5216,52161`
- GRU-NBM：直接复用正式五种子实验中已冻结的15个检查点
- TCN：最大5 epoch，patience 2
- 分类损失：`BCEWithLogitsLoss(pos_weight=N_role6/N_role7)`
- 分类器选择：角色2/3 AP最高
- 阈值选择：角色2/3 Balanced Accuracy最高；依次以FoG F1和更高阈值打破并列
- 测试：全部30个分类器和阈值冻结后，才访问角色0/1
- 配对初始化：共享形状参数完全一致；FULL_C首层前9个通道与RESIDUAL_R一致，新增18个输入通道权重初始化为0

默认启动脚本从以下目录复用NBM、角色4 Scaler以及角色5校准参数，不会重新训练NBM：

```text
outputs/daphnet_tcn_nbm300_C_vs_raw_tcn_ep5pat2_seedset_0_52_161_5216_52161/nbm_source/
```

这是严格消融所推荐的方式：两个分类分支使用完全相同的重构误差和校准参数。

## 服务器运行

```bash
cd /document/home_mirror/chb/fog_classification_framework_base
conda activate fogbase
mkdir -p logs

nohup bash scripts/run_daphnet_gru_nbm300_residual_expansion_ablation_7gpu.sh \
  > logs/daphnet_gru_nbm300_residual_expansion_ablation.log 2>&1 &
```

查看运行状态：

```bash
tail -f logs/daphnet_gru_nbm300_residual_expansion_ablation.log
```

仅检查任务计划，不启动训练：

```bash
python scripts/launch_daphnet_gru_nbm300_c_vs_raw_ep5pat2_7gpu.py \
  --output-root outputs/daphnet_gru_nbm300_FULL_C_vs_RESIDUAL_R_tcn_ep5pat2_seedset_0_52_161_5216_52161 \
  --reuse-nbm-source-root outputs/daphnet_tcn_nbm300_C_vs_raw_tcn_ep5pat2_seedset_0_52_161_5216_52161/nbm_source \
  --experiment-methods FULL_C,RESIDUAL_R \
  --gpu-ids 0,1,2,3,4,5,6 \
  --phase full \
  --dry-run
```

## 输出目录

```text
outputs/daphnet_gru_nbm300_FULL_C_vs_RESIDUAL_R_tcn_ep5pat2_seedset_0_52_161_5216_52161/
```

主要输出包括：

- `TRAINING_BARRIER.json`：30个分类器、阈值和测试数据清单的冻结屏障；
- `run_metrics_30.csv`：逐折、逐方法、逐种子指标；
- `method_summary_5seed_mean_std.csv`：两种方法的五种子总体指标；
- `paired_delta_FULL_C_minus_RESIDUAL_R_by_seed.csv`：每个种子的配对差值；
- `paired_delta_FULL_C_minus_RESIDUAL_R_summary.csv`：配对差值汇总；
- `subject_metrics_5seed_mean_std.csv`：逐被试结果；
- `summary.json`：完整机器可读汇总。

中断后可以使用同一Python启动命令，并将`--phase`改为`nbm`、`train`、
`evaluate`或`aggregate`恢复。不要添加`--overwrite`，已完成任务会自动跳过。

如果服务器上不存在上述冻结NBM目录，删除命令中的`--reuse-nbm-source-root`即可；
启动器会先在新输出目录中完成15个GRU-NBM训练，再执行分类器消融。
