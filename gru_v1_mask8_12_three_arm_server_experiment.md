# GRU-v1局部Mask增强三臂对照实验

## 实验目的

检验在不改变GRU-NBM架构、高斯噪声、损失函数、优化器、数据角色、分类器与阈值规则的前提下，仅把训练阶段局部连续Mask长度从4–8个采样点增加到8–12个采样点，能否提高FoG Sensitivity。

这是单变量对照实验，不复用历史NBM、TCN、角色5校准参数或阈值。

## 三个实验臂

| 方法 | TCN输入 | NBM训练Mask |
|---|---|---|
| `RAW` | 角色4 RobustScaler + 逐窗逐轴中心化原始信号，`[B,9,128]` | 不使用NBM；仅读取角色4 Scaler |
| `GRU_BASE_C` | 方案C残差 `[r,|r|,Δr]`，`[B,27,128]` | 20%窗口使用连续4–8点全轴Mask（62.5–125 ms） |
| `GRU_MASK8_12_C` | 方案C残差 `[r,|r|,Δr]`，`[B,27,128]` | 20%窗口使用连续8–12点全轴Mask（125–187.5 ms） |

两种NBM都使用相同原始GRU-v1：

```text
X [B,128,9]
  -> 单层单向GRU，hidden=64
  -> Linear 64->16
  -> 全局瓶颈Z [B,16]
  -> Linear 16->64，作为解码器初始状态
  -> 128步全零输入的单层单向GRU，hidden=64
  -> Linear 64->9
  -> Xhat [B,128,9]
```

无skip connection、无teacher forcing、无时间编码，参数量均为31,513。每个折和种子都会保存训练前模型权重哈希，屏障要求两种Mask配置的初始权重完全一致。

## 冻结训练配置

- 数据：64 Hz，2秒窗口128点，步长64点（1秒）。
- 被试：排除S04、S10；3折。
- 配对种子：`0, 52, 161, 5216, 52161`，无fold offset。
- NBM增强：40%完整窗口、40%全窗Gaussian（std=0.04）、20%连续全轴Mask。
- NBM：SmoothL1(beta=1.0)，AdamW(lr=1e-3, weight_decay=1e-4)，最大300 epoch，patience=20。
- TCN：原分类器与方案C不变，最大5 epoch，patience=2；`pos_weight`仅由角色6/7计算。
- 阈值：角色2/3最大Balanced Accuracy；并列时依次选择更高FoG F1、更高阈值。

## 数据角色

| 角色 | 用途 |
|---:|---|
| 4 | 拟合RobustScaler并训练NBM |
| 5 | 纯净窗口监控NBM验证损失；恢复最低验证损失权重后，分别计算各NBM自己的 `b` 与 `sigma` |
| 6、7 | 训练TCN分类器 |
| 2、3 | TCN早停、模型选择和阈值选择 |
| 0、1 | 45个分类器与阈值全部冻结并生成全局屏障后，才运行永久测试 |

方案C保持原定义：

```text
e = X - Xhat
q = clip(e / (sigma + 1e-6), -12, 12)
r = q - mean_t(q)
F = [r, abs(r), delta_t(r)]
```

`b`仅用于角色5的MAD尺度估计，生成方案C残差时不从 `e` 中减去 `b`。

## 服务器运行

在项目根目录运行：

```bash
cd /document/home_mirror/chb/fog_classification_framework_base
conda activate fogbase
bash scripts/run_daphnet_gru_mask8_12_three_arm_7gpu.sh
```

默认使用GPU `0,1,2,3,4,5,6`，完整任务量为：

- 30个NBM：2种Mask × 3折 × 5种子；
- 45个TCN：3方法 × 3折 × 5种子；
- 1个全局测试屏障；
- 屏障后45个测试任务。

仅检查任务计划、不训练：

```bash
python scripts/launch_daphnet_gru_mask8_12_three_arm_7gpu.py --dry-run
```

分阶段断点续跑：

```bash
python scripts/launch_daphnet_gru_mask8_12_three_arm_7gpu.py --phase nbm
python scripts/launch_daphnet_gru_mask8_12_three_arm_7gpu.py --phase train
python scripts/launch_daphnet_gru_mask8_12_three_arm_7gpu.py --phase evaluate
```

默认输出目录：

```text
outputs/daphnet_gru_mask8_12_three_arm_nbm300_C_tcn_ep5pat2_seedset_0_52_161_5216_52161
```

若旧目录属于不同代码、数据或实验合同，程序会拒绝混用；此时请使用新的 `--output-root`，不要把旧产物复制进新目录。

## 汇总口径与预注册判断

每个种子先宏平均3折，再对5个种子计算均值与总体标准差（`ddof=0`）。主比较为：

```text
GRU_MASK8_12_C - GRU_BASE_C
```

预注册晋级条件：

- Sensitivity平均提升至少0.010；
- 5个种子中至少4个Sensitivity为正向；
- PR-AUC下降不超过0.005；
- Precision下降不超过0.010；
- Specificity下降不超过0.010。

主要输出包括 `summary.json`、`method_summary_5seed_mean_std.csv`、三组配对差值CSV、逐被试指标、45组逐运行指标和完整测试预测文件。

## 结论边界

角色0/1已经在此前多轮同数据集架构实验中被查看，因此本轮虽然在程序内部仍严格执行测试屏障，但统计定位应是探索性配对对照，而不是全新的独立确认性测试。若用于论文中的最终无偏确认，应预注册方案后在新的外部数据或从未查看的holdout上复验。
