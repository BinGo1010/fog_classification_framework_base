# Conv-TCN NBM（方案C）与Raw-TCN严格消融实验

## 固定实验设计

两组只改变分类器输入路径，其余数据范围、TCN训练和阈值规则保持一致。

### 完整组 `FULL_C`

```text
角色4拟合RobustScaler
→ 每窗口、每轴沿128个时间点减均值
→ Conv-TCN NBM（角色4训练，角色5早停）
→ e = X - X_hat
→ q = clip(e / (sigma + 1e-6), -12, 12)
→ r = q - mean_t(q)
→ F = [r, |r|, delta_t(r)]，形状[B,27,128]
→ TCN分类器
```

方案C不减重构偏移 `b`，但使用角色5估计的 `sigma`。

### 严格消融组 `RAW`

```text
角色4拟合的同一RobustScaler
→ 每窗口、每轴沿128个时间点减均值
→ X_c，形状[B,9,128]
→ 同一TCN分类器
```

Raw组不执行NBM推理，不使用重构信号，不生成残差，也不使用角色5的 `b` 或 `sigma`。

## 固定训练配置

- 数据折：相同3折。
- 分类器训练：角色6/7。
- 分类器验证、早停和阈值：角色2/3。
- 永久测试：角色0/1；全部18个分类器和阈值封存后才允许读取。
- NBM和TCN使用三个严格配对种子：`0, 52, 161`。
- 每个种子在每一折中原样使用，不再执行`seed + fold`隐式偏移。
- TCN：最大10 epoch，patience 2，AdamW，lr `1e-3`，weight decay `1e-4`。
- 分类损失：`BCEWithLogitsLoss(pos_weight=N_role6/N_role7)`。
- NBM：最大300 epoch，patience 20，纯SmoothL1，AdamW，lr `1e-3`。
- NBM增强：40%完整、40%高斯噪声（std=0.04）、20%轻度时间Mask。
- NBM checkpoint：最低未增强角色5验证SmoothL1，并在计算校准量前恢复最佳权重。
- 阈值：角色2/3上最大Balanced Accuracy；并列时先FoG F1，再选择更高阈值。

9通道和27通道TCN的第一层形状必然不同。程序会让全部形状兼容的参数完全相同；27通道第一层的前9通道复制Raw权重，额外18通道置零。每对实验还会重置随机状态，保持相同batch顺序和后续随机序列。

## 服务器运行

```bash
cd /path/to/fog-merged
conda activate fogbase
chmod +x scripts/run_daphnet_nbm300_c_vs_raw_ablation_7gpu.sh
bash scripts/run_daphnet_nbm300_c_vs_raw_ablation_7gpu.sh
```

等价的Python命令：

```bash
python scripts/launch_daphnet_nbm300_c_vs_raw_ablation_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --phase full \
  --nbm-max-epochs 300 \
  --nbm-patience 20 \
  --tcn-max-epochs 10 \
  --tcn-patience 2 \
  --nbm-seeds 0,52,161 \
  --tcn-seeds 0,52,161
```

如服务器拥有8张卡，可将GPU列表改为`0,1,2,3,4,5,6,7`；任务数和实验定义不变，只提高并发数。

## 分阶段恢复

```bash
# 只训练3折×3种子，共9个NBM
python scripts/launch_daphnet_nbm300_c_vs_raw_ablation_7gpu.py --gpu-ids 0,1,2,3,4,5,6 --phase nbm

# 训练18个TCN并建立全局测试屏障
python scripts/launch_daphnet_nbm300_c_vs_raw_ablation_7gpu.py --gpu-ids 0,1,2,3,4,5,6 --phase train

# 屏障后测试并汇总
python scripts/launch_daphnet_nbm300_c_vs_raw_ablation_7gpu.py --gpu-ids 0,1,2,3,4,5,6 --phase evaluate
```

已完成的任务默认跳过。只有确实要覆盖同一路径下的既有结果时才加`--overwrite`。

## 输出

默认输出目录：

```text
outputs/daphnet_conv_tcn_nbm300_C_vs_raw_tcn_ep10pat2_seedset_0_52_161/
```

主要文件：

- `nbm_source/seed_{0,52,161}/fold_*/`：9次NBM的最佳权重、训练曲线和角色5校准。
- `TRAINING_BARRIER.json`：18个分类器、阈值和配对条件全部冻结的证明。
- `run_metrics_18.csv`：每折、每方法、每种子的测试指标和混淆矩阵元素。
- `method_summary_3seed_mean_std.csv`：两组总体均值±标准差。
- `paired_delta_FULL_C_minus_RAW_by_seed.csv`：逐种子的配对差值。
- `paired_delta_FULL_C_minus_RAW_summary.csv`：配对差值均值±标准差。
- `subject_metrics_3seed_mean_std.csv`：各被试指标。
- `runs/.../tcn_training_validation.{png,svg,pdf}`：TCN训练与验证曲线。
- `runs/.../test_confusion_matrix.{png,svg,pdf}`：各测试任务混淆矩阵。

总体统计定义为：先在每个TCN种子内对3折做宏平均，再对3个种子值报告均值和总体标准差（`ddof=0`）。

## 服务器预检

```bash
python -m py_compile \
  scripts/run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py \
  scripts/run_daphnet_nbm300_c_vs_raw_ablation.py \
  scripts/launch_daphnet_nbm300_c_vs_raw_ablation_7gpu.py

python -m pytest tests/test_daphnet_nbm300_c_vs_raw_ablation.py -q

python scripts/launch_daphnet_nbm300_c_vs_raw_ablation_7gpu.py --dry-run
```

Raw中心化审计使用float64重新累计均值，并采用与float32输入幅值相关的容差；
该检查不会把约`1e-5`量级的正常舍入误差误判为中心化失败。
