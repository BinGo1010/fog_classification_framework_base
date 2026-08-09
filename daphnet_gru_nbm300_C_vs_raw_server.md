# GRU-NBM方案C与Raw-TCN严格配对实验

## 实验目的

只把完整组的NBM骨干从Conv-TCN自编码器替换为此前效果较好的GRU重构NBM；
数据角色、预处理、方案C残差、TCN架构、训练参数、种子和阈值规则均保持一致。

## 完整组 `FULL_C`

```text
角色4拟合RobustScaler
→ 每窗口、每轴沿128个时间点中心化
→ X [B,128,9]
→ 单层单向GRU Encoder，9→64
→ 最后隐藏状态[B,64]
→ Linear 64→16，Z[B,16]
→ Linear 16→64，作为Decoder初始隐藏状态
→ 128步全零输入的单层单向GRU Decoder，9→64
→ Linear 64→9
→ X_hat [B,128,9]
→ e=X-X_hat
→ q=clip(e/(sigma+1e-6),-12,12)
→ r=q-mean_t(q)
→ F=[r,abs(r),delta_t(r)] [B,27,128]
→ TCN分类器
```

方案C使用角色5估计的`σ`，不减重构偏移`b`。程序仍保存`b`，用于完整校准审计，
但不会把它代入方案C特征。

## 消融组 `RAW`

```text
角色4拟合的同一RobustScaler
→ 每窗口、每轴沿128个时间点中心化
→ Xc [B,9,128]
→ 相同TCN分类器
```

Raw组不执行GRU-NBM，不生成残差，也不使用角色5的`b`或`σ`。

## GRU-NBM训练

- 角色4：Scaler拟合和NBM训练，仅Non-FoG。
- 角色5：未增强验证、早停和校准。
- 最大epoch：300。
- Early stopping patience：20。
- Batch size：128。
- Optimizer：AdamW，lr=`1e-3`，weight decay=`1e-4`。
- Scheduler：ReduceLROnPlateau，factor=`0.5`，patience=`3`，min lr=`1e-5`。
- Loss：SmoothL1，beta=`1.0`。
- Gradient clipping：`1.0`。
- 增强：40%完整、40%高斯噪声（std=`0.04`）、20%全轴时间Mask（4–8点）。
- Checkpoint：最低角色5验证SmoothL1；计算校准统计量前恢复最佳权重。
- 架构：单层、单向、无跳连，hidden=`64`，latent=`16`，零输入GRU解码器。

## TCN训练

- 角色6/7训练，角色2/3验证、早停和选择阈值。
- 最大epoch：10。
- Early stopping patience：2。
- Optimizer：AdamW，lr=`1e-3`，weight decay=`1e-4`。
- Loss：`BCEWithLogitsLoss(pos_weight=N_role6/N_role7)`。
- 模型选择：角色2/3 PR-AUC。
- 阈值：角色2/3最大Balanced Accuracy；并列时依次比较FoG F1和更高阈值。
- 角色0/1只有在全部18个分类器和阈值冻结后才允许测试。

## 随机种子

NBM和TCN严格一一配对：

| 重复 | NBM | TCN |
|---:|---:|---:|
| 1 | 0 | 0 |
| 2 | 52 | 52 |
| 3 | 161 | 161 |

每个种子在三个折中原样使用，不执行`seed+fold`偏移。因此共有9次GRU-NBM训练、
9次FULL_C分类器训练和9次RAW分类器训练。

## 7卡服务器运行

```bash
cd /document/home_mirror/chb/fog_classification_framework_base
conda activate fogbase

bash scripts/run_daphnet_gru_nbm300_c_vs_raw_7gpu.sh
```

等价命令：

```bash
python scripts/launch_daphnet_gru_nbm300_c_vs_raw_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --phase full \
  --nbm-seeds 0,52,161 \
  --tcn-seeds 0,52,161 \
  --nbm-max-epochs 300 \
  --nbm-patience 20 \
  --tcn-max-epochs 10 \
  --tcn-patience 2
```

分阶段恢复：

```bash
# 9个GRU-NBM
python scripts/launch_daphnet_gru_nbm300_c_vs_raw_7gpu.py --phase nbm --gpu-ids 0,1,2,3,4,5,6

# 18个分类器并建立全局测试屏障
python scripts/launch_daphnet_gru_nbm300_c_vs_raw_7gpu.py --phase train --gpu-ids 0,1,2,3,4,5,6

# 屏障后测试并汇总
python scripts/launch_daphnet_gru_nbm300_c_vs_raw_7gpu.py --phase evaluate --gpu-ids 0,1,2,3,4,5,6
```

恢复任务时不要添加`--overwrite`，已完成任务会自动跳过。

## 输出目录

```text
outputs/daphnet_gru_nbm300_C_vs_raw_tcn_ep10pat2_seedset_0_52_161/
```

主要结果：

- `nbm_source/seed_{0,52,161}/fold_*/`：9个GRU-NBM权重、训练曲线及校准。
- `TRAINING_BARRIER.json`：所有分类器和阈值冻结证明。
- `run_metrics_18.csv`：逐折、逐方法、逐种子测试结果及混淆矩阵元素。
- `method_summary_3seed_mean_std.csv`：GRU-NBM方案C和Raw总体指标。
- `paired_delta_FULL_C_minus_RAW_by_seed.csv`：每个种子的配对差值。
- `paired_delta_FULL_C_minus_RAW_summary.csv`：配对差值均值±总体标准差。
- `subject_metrics_3seed_mean_std.csv`：各被试主指标。
- 每个任务的TCN训练曲线和测试混淆矩阵PNG/SVG/PDF。

总体统计先在每个种子内对3折宏平均，再对种子`0、52、161`报告均值和总体标准差。
