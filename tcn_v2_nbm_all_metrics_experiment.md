# TCN-NBM-v2：方案 C 与 Raw 严格配对实验

## 1. 优化目标与设计依据

旧 Conv-TCN NBM 的参数量为 47,449，但瓶颈为 `[B,16,32]`，实际包含
512 个带时间位置的潜变量。该表示仍容易把 FoG 的局部冻结、颤抖和冲击形态
传给解码器，因此角色 5 重构损失低并不代表异常检测更好；它也可能把 FoG
重构得过好，使残差的类别间隔缩小。

TCN-NBM-v2 不以“扩大时序潜变量”为方向，而把增加的容量放在正常步态的
跨时间建模中，同时把 encoder 到 decoder 的唯一信息通道压缩为全局 16 维
向量。目标是让 NBM 更像正常步态模板生成器，而不是局部信号复制器。

## 2. TCN-NBM-v2 架构

```text
X [B,9,128]
  -> Conv1d 9->24, k7, s1 + GroupNorm + GELU
  -> Conv1d 24->32, k5, s2 + GroupNorm + GELU       [B,32,64]
  -> TCN Residual Blocks, dilation 1,2,4,8
  -> Conv1d 32->48, k5, s2 + GroupNorm + GELU       [B,48,32]
  -> TCN Residual Blocks, dilation 1,2,4
  -> Flatten                                             [B,1536]
  -> Linear 1536->16 + LayerNorm + tanh
  -> Global bottleneck Z                                 [B,16]
  -> Linear 16->32，并沿 128 个时刻广播                  [B,32,128]
  -> 拼接固定 Fourier 时间编码                           [B,48,128]
  -> Conv1d 48->48, k1 + GroupNorm + GELU
  -> TCN Residual Blocks, dilation 1,2,4,8,16
  -> Conv1d 48->32, k5 + GroupNorm + GELU
  -> Conv1d 32->9, k1
  -> X_hat                                               [B,9,128]
```

每个 TCN 残差块使用两个保持时间长度的 `k=3` 膨胀卷积，并配合
GroupNorm、GELU 和 `Dropout=0.10`。固定 Fourier 时间编码包含
0.5、0.75、1、1.25、1.5、2、2.5、3 Hz 的正余弦，共 16 通道。

关键结构约束：

- 无 encoder-decoder skip connection；
- 无 teacher forcing；
- 无原始输入直通；
- 无 encoder 时序特征直通；
- decoder 只能读取全局 `Z=[B,16]` 与固定时间坐标；
- 输入、输出仍为 `[B,9,128]`；
- 精确可训练参数量为 **186,065**。

相较旧版，参数量约增至 3.92 倍，但潜变量由 512 个时间对齐值缩小为 16 个
全局值。增加的参数用于正常步态动力学建模，而不是提高异常波形的传输带宽。

## 3. 冻结的实验协议

本实验只替换 NBM 骨干，以下设置不变：

- 数据：`processed_NBM`，64 Hz，2 秒窗口 `[9,128]`，步长 1 秒；
- 三折，排除 S04、S10；
- 配对随机种子：`0, 52, 161, 5216, 52161`，NBM seed 与 TCN seed 相同；
- 角色 4：拟合 RobustScaler，并训练 NBM；
- 角色 5：无增强验证、NBM 早停；恢复最低验证 SmoothL1 权重后计算
  `b` 和 `sigma`；
- 角色 6/7：仅用于 TCN 分类器训练，`pos_weight=N_role6/N_role7`；
- 角色 2/3：分类器早停、模型选择和最大 Balanced Accuracy 阈值；
- 角色 0/1：所有分类器和阈值全局封存后才运行一次测试；
- NBM：SmoothL1、AdamW `lr=1e-3, weight_decay=1e-4`、最大 300 epoch、
  patience 20、梯度裁剪 1.0；
- NBM 增强：40% clean、40% Gaussian (`std=0.04`)、20% 轻度 Mask；
- 分类器：原 TCN、最大 5 epoch、patience 2，其余优化器、batch size、
  初始化配对及阈值规则均不变；
- FULL_C：`e=X-X_hat`，使用角色 5 的 `sigma` 标准化，截断后逐窗逐轴
  中心化，输入 `[r, |r|, delta(r)]`，即 `[B,27,128]`；不减偏移 `b`；
- RAW：同一角色 4 RobustScaler 和逐窗逐轴中心化，直接输入 `[B,9,128]`。

任务总数为 15 个 TCN-NBM-v2、30 个分类器训练和全局屏障后的 30 个测试。
训练阶段会读取角色 0/1 的窗口标识等元数据做完整性审计，但不会物化其信号
特征、概率或指标用于训练、早停、校准、阈值或模型选择。

屏障会冻结 30 个分类器的 checkpoint、NBM checkpoint、Scaler、阈值、角色
0/1 窗口清单以及所引用原始记录文件的 SHA256。断点恢复跳过旧测试前、以及最终
聚合前，程序都会重新核对这些内容；重训导致屏障变化时，旧 `DONE_TEST` 会被
拒绝，而不会静默混入新结果。

## 4. 服务器运行

在项目根目录执行：

```bash
conda activate fogbase
bash scripts/run_daphnet_tcn_v2_nbm300_c_vs_raw_ep5pat2_7gpu.sh
```

等价的直接命令：

```bash
python scripts/launch_daphnet_tcn_v2_nbm300_c_vs_raw_ep5pat2_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --phase full
```

默认输出目录：

```text
outputs/daphnet_tcn_v2_nbm300_C_vs_raw_tcn_ep5pat2_seedset_0_52_161_5216_52161
```

若服务器项目或数据不在默认位置，可显式追加：

```bash
--data-dir /your/repo/dataset/1.Daphnet\ Freezing\ of\ Gait\ Dataset/processed_NBM \
--output-root /your/repo/outputs/daphnet_tcn_v2_nbm300_C_vs_raw_tcn_ep5pat2_seedset_0_52_161_5216_52161
```

程序会自动执行 `NBM -> classifier train -> global seal -> test -> aggregate`。
中断后直接运行同一命令会跳过已有完成标记；只有确认要重训时才添加
`--overwrite`。

## 5. 结果文件与判定

主要结果位于：

- `summary.json`：两种方法的总体均值、标准差和配对差值；
- `method_summary_5seed_mean_std.csv`：每个主指标的 5 seed 均值与总体 SD；
- `paired_delta_FULL_C_minus_RAW_by_seed.csv`：每个种子的配对差；
- `paired_delta_FULL_C_minus_RAW_summary.csv`：配对差汇总；
- `subject_metrics_5seed_mean_std.csv`：各被试结果；
- 每个 run 的 `metrics.json`、混淆矩阵、阈值和训练图。

不能在测试集上调阈值来制造 Sensitivity 提升。只有在角色 2/3 使用相同阈值
搜索规则时，FULL_C_V2 的 Sensitivity、Precision、Specificity 和与阈值无关的
PR-AUC 均高于 Raw，才能表述为四项指标同时改善。

建议预先登记工程成功门槛：

- `Delta Sensitivity >= +0.010`；
- `Delta Precision >= +0.010`；
- `Delta Specificity >= +0.005`；
- `Delta PR-AUC >= +0.010`；
- 每项至少 4/5 个配对种子方向为正。

这些门槛是实验完成后的判定标准，不是当前代码能够预先保证的结果。
此外，角色 0/1 已用于此前多轮架构比较，因此本次属于 adaptive benchmark；
若用于确认性论文结论，应在新留出的受试者或外部数据上再验证一次。

首轮结果还应查看：角色 5 的 `sigma` floor 命中通道、角色 2/3 的残差截断率、
以及 FoG/Non-FoG 中位绝对残差比。若模型对不同窗口生成几乎相同的模板，应在
下一轮单独增加 latent-shuffle/`decode(0)` 诊断，再决定是否用轻量 FiLM 加强
decoder 对 `Z` 的条件化；不要在首轮同时扩大瓶颈或更换损失函数。
