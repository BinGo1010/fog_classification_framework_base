# GRU-NBM-v1.5 三臂服务器实验

## 1. 实验目的

在当前表现最好的 GRU-NBM-v1 上只改变一个结构变量：将零输入解码器的隐藏维度从 64 增加到 96。编码器与 16 维全局瓶颈保持不变，用于检验“增强正常步态生成能力、但不放宽异常信息通道”能否提高 FoG Sensitivity，同时保持 Precision、Specificity 与 PR-AUC。

三臂必须同期训练：

1. `RAW`：逐窗口逐轴中心化原始信号 `[B,9,128]`；
2. `GRU_V1_C`：原始 GRU-v1 + 方案 C 残差 `[B,27,128]`；
3. `GRU_V15_C`：GRU-v1.5-dec96 + 相同方案 C 残差 `[B,27,128]`。

## 2. NBM 架构

### GRU-v1 基线，31,513 参数

```text
X [B,128,9]
  -> 单层单向 GRU, hidden=64
  -> 最后隐藏状态 [B,64]
  -> Linear 64->16
  -> Z [B,16]
  -> Linear 16->64，作为解码器初态
  -> 128 步全零输入的单层单向 GRU, hidden=64
  -> Linear 64->9
  -> Xhat [B,128,9]
```

### GRU-v1.5-dec96，48,761 参数

```text
X [B,128,9]
  -> 单层单向 GRU, hidden=64             # 与 v1 相同
  -> 最后隐藏状态 [B,64]
  -> Linear 64->16
  -> Z [B,16]                            # 与 v1 相同
  -> Linear 16->96，作为解码器初态       # 唯一结构改动
  -> 128 步全零输入的单层单向 GRU, hidden=96
  -> Linear 96->9
  -> Xhat [B,128,9]
```

两种 NBM 均无 skip、teacher forcing、时间编码、LayerNorm、dropout 或输出激活。

## 3. 方案 C 输入

```text
e = X - Xhat
q = clip(e / (sigma + 1e-6), -12, 12)
r = q - mean_t(q)
F = [r, abs(r), delta_t(r)] in R^[27x128]
```

`sigma` 由恢复最低角色 5 验证损失的 NBM 权重后计算。`b`只参与 MAD 尺度估计，不从最终残差中直接减去。分类器结构与当前方案相同。

## 4. 冻结的数据角色

| 角色 | 唯一用途 |
|---|---|
| 4 | 拟合 RobustScaler、训练 NBM |
| 5 | NBM 早停；恢复最佳权重后计算校准量 |
| 6/7 | 训练 TCN；仅由这里计算 `pos_weight` |
| 2/3 | TCN 早停、模型选择和分类阈值 |
| 0/1 | 全部 45 个分类器与阈值封存后，一次性测试 |

RAW 仅读取单独保存的 `scaler_role4.json`。该文件在任何角色 5 校准前生成，不含 `b`、`sigma` 或 NBM checkpoint；RAW 的屏障字段中 NBM checkpoint 必须为 `null`。

全局屏障还会冻结：每个 NBM 与 TCN 的 fold/seed/架构身份、checkpoint、`nbm_frozen.json`（因此包含角色 5 的 `sigma`）、Scaler、特征公式、阈值、完成标记和验证产物哈希。数据指纹覆盖 `manifest/schema/protocol/quality`、全部 `records` 与全部 `split_indices`；代码指纹覆盖 NBM、方案 C、TCN、阈值、评估和原子写入等科学依赖。任一项在恢复或测试前变化都会直接拒绝继续运行。

## 5. 训练配置与任务量

- 数据：`processed_NBM`，64 Hz，窗口 128 点，步长 64 点；
- folds：`0,1,2`；
- 配对种子：`0,52,161,5216,52161`，不加 fold offset；
- 两种 NBM：最大 epoch 300，patience 20，SmoothL1，AdamW `lr=1e-3`，40/40/20 clean/Gaussian/mask；
- 三个 TCN：最大 epoch 5，patience 2，其他设置不变；
- NBM：`2 x 3 x 5 = 30` 个任务；
- TCN：`3 x 3 x 5 = 45` 个任务；
- 全局屏障后测试：45 个任务；
- 默认 GPU：`0,1,2,3,4,5,6`。

两个 27 通道分类器使用完全相同的 TCN 初始状态。RAW 与残差臂所有形状兼容的参数相同；残差臂第一层前 9 通道复制 RAW 初始权重，新增 18 通道置零。

## 6. 服务器运行

```bash
cd /document/home_mirror/chb/fog_classification_framework_base
conda activate fogbase
bash scripts/run_daphnet_gru_v15_three_arm_7gpu.sh
```

也可直接运行：

```bash
python scripts/launch_daphnet_gru_v15_three_arm_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --phase full
```

默认输出目录：

```text
outputs/daphnet_gru_v15_dec96_three_arm_nbm300_C_tcn_ep5pat2_seedset_0_52_161_5216_52161
```

先检查任务计划而不训练：

```bash
python scripts/launch_daphnet_gru_v15_three_arm_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --dry-run
```

分阶段恢复可使用 `--phase nbm`、`--phase train`、`--phase evaluate` 或 `--phase aggregate`。已经完整结束且身份/哈希一致的任务会跳过；未完成的单个 NBM 或 TCN 从第 1 epoch 重训，不是 epoch 级断点续训。不要在不同代码版本间复用同一输出目录。

## 7. 汇总口径

每个种子先对 3 folds 宏平均，再对 5 个种子报告 mean +/- population SD，并同时输出：

- `GRU_V1_C - RAW`；
- `GRU_V15_C - RAW`；
- `GRU_V15_C - GRU_V1_C`。

主要指标为 Sensitivity、Precision、Specificity 与 PR-AUC。预注册的 v1.5 相对 v1 成功条件为：Sensitivity 平均提升至少 0.010，5 个种子至少 4 个方向为正；PR-AUC 不低于 -0.005，Precision 与 Specificity 各不低于 -0.010。

角色 0/1 已在此前多轮架构实验中被查看，因此本次应表述为 adaptive benchmark；若用于无偏确认性结论，需要新的外部或未见测试集。
