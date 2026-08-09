# Daphnet 32 Hz：Raw 与 GRU-NBM 方案 C 严格配对实验

## 1. 实验目标

在 `processed_NBM_32Hz` 的相同三折、相同角色范围和相同 TCN 设置下，比较：

- `RAW`：角色 4 的 RobustScaler → 逐窗口逐轴中心化 → TCN；
- `FULL_C`：相同预处理 → GRU-NBM 重构 → 方案 C 的 27 通道残差 → 相同 TCN。

五个配对随机种子固定为 `0、52、161、5216、52161`。同一折、同一种子下，GRU-NBM 和 TCN 使用相同种子；Raw 与 FULL_C 的 TCN 除首层输入通道外共享相同初始化。程序不增加折号偏移。

## 2. 32 Hz 输入输出修改

降采样后仍是 2 秒窗口、1 秒步长：

- 采样率：32 Hz；
- 窗口长度：64 点；
- 滑动步长：32 点；
- 原始窗口：`[B,9,64]`；
- GRU 实际张量顺序：`[B,64,9]`；
- GRU 重构输出：`[B,64,9]`。

只改变序列时间长度 `T: 128 → 64`，GRU 内部架构不变：

```text
[B,64,9]
  → 单层单向 GRU，input=9，hidden=64
  → 取最后隐藏状态 [B,64]
  → Linear 64→16，latent=[B,16]
  → Linear 16→64，得到解码器初始状态
  → 64 步全零输入的单层单向 GRU，input=9，hidden=64
  → Linear 64→9
  → [B,64,9]
```

无跳跃连接，不改变 hidden、latent、GRU 层数或方向。

## 3. 两个方法的分类器输入

### RAW

角色 4 拟合的 RobustScaler 对窗口进行变换，然后在每个窗口、每个轴的时间维上减均值：

\[
X_c=X_s-\operatorname{mean}_t(X_s).
\]

TCN 输入为 `[B,9,64]`。Raw 分支不读取 NBM 输出、角色 5 的校准量或残差。

### FULL_C（GRU-NBM + 方案 C）

NBM 输入与目标均为中心化后的缩放窗口。恢复角色 5 验证损失最低的 NBM 权重后，在角色 5 上冻结每轴残差尺度 \(\sigma\)。方案 C 为：

\[
e=X-\hat X,\qquad
q=\operatorname{clip}\left(\frac{e}{\sigma+10^{-6}},-12,12\right),
\]

\[
r=q-\operatorname{mean}_t(q),\qquad
F=[r,|r|,\Delta r]\in\mathbb{R}^{27\times64}.
\]

TCN 输入为 `[B,27,64]`。本方案 C 使用 \(\sigma\)，不从残差中减去角色 5 的偏移 \(b\)。

## 4. 数据角色与防泄漏约束

- 角色 4：拟合 RobustScaler、训练 GRU-NBM；
- 角色 5：NBM 早停、恢复最佳权重后计算残差校准量；
- 角色 6/7：训练 TCN，分别为 Non-FoG/FoG；
- 角色 2/3：TCN 早停、模型选择、分类阈值选择；
- 角色 0/1：最终测试，只有全部 30 个分类器及阈值冻结后才能访问。

Raw 与 FULL_C 使用完全相同的角色 6/7、2/3、0/1 窗口。`pos_weight=N_role6/N_role7`，阈值只在角色 2/3 上按 Balanced Accuracy 搜索；并列时先选 FoG F1 更高者，再选更高阈值。

## 5. 训练设置

GRU-NBM：

- 最大 epoch：300；
- early-stopping patience：20；
- SmoothL1，`beta=1.0`；
- AdamW，`lr=1e-3`，`weight_decay=1e-4`；
- batch size：128；
- 梯度裁剪：1.0；
- 训练增强：40% clean、40% Gaussian（std=0.04）、20% 全轴时间 Mask（4–8 点）；
- 角色 5 不增强，恢复最低角色 5 验证损失权重。

TCN 分类器：

- 架构与原方案 C/Raw 对照相同；
- 最大 epoch：20；
- early-stopping patience：5；
- weighted BCE；
- AdamW，`lr=1e-3`；
- batch size、数据角色和阈值规则保持一致。

## 6. 7 卡服务器运行

需同步以下程序：

- `scripts/run_daphnet_s01_nonfog_gru_reconstruction_tcnm.py`
- `scripts/run_daphnet_gru_nbm300_fold.py`
- `scripts/run_daphnet_nbm300_c_vs_raw_ablation.py`
- `scripts/launch_daphnet_32hz_gru_nbm300_c_vs_raw_7gpu.py`
- `scripts/run_daphnet_32hz_gru_nbm300_c_vs_raw_7gpu.sh`

在仓库根目录运行：

```bash
conda activate fogbase
bash scripts/run_daphnet_32hz_gru_nbm300_c_vs_raw_7gpu.sh
```

程序将排队执行：

- 15 个 GRU-NBM：3 折 × 5 种子；
- 30 个 TCN 训练：3 折 × 2 方法 × 5 种子；
- 建立全局测试屏障；
- 30 个测试任务；
- 自动汇总。

默认结果目录：

```text
outputs/daphnet_32Hz_gru_nbm300_C_vs_raw_tcn_ep20pat5_seedset_0_52_161_5216_52161
```

可先检查任务计划而不训练：

```bash
python scripts/launch_daphnet_32hz_gru_nbm300_c_vs_raw_7gpu.py --dry-run
```

如训练中断，可使用 `--phase nbm`、`--phase train`、`--phase evaluate` 或 `--phase aggregate` 分阶段恢复。已存在 `DONE_*.json` 的任务默认跳过；只有明确需要覆盖时才加 `--overwrite`。

## 7. 汇总输出

主要汇总文件包括：

- `run_metrics_30.csv`：30 个折/方法/种子测试结果；
- `method_summary_5seed_mean_std.csv`：五种子总体均值和标准差；
- `subject_metrics_5seed_mean_std.csv`：每名被试的五种子均值和标准差；
- `paired_delta_FULL_C_minus_RAW_by_seed.csv`：每种子的 FULL_C−RAW 差值；
- `paired_delta_FULL_C_minus_RAW_summary.csv`：配对差值汇总；
- `summary.json`：完整机器可读汇总；
- 各运行目录中的训练曲线、混淆矩阵、阈值和主指标。

