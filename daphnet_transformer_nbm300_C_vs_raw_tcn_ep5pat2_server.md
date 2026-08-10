# Raw vs Transformer-NBM方案C：7卡严格配对实验

## 1. 实验范围

- 数据集：`processed_NBM`
- 采样率：64 Hz
- 窗口：2秒，即128点
- 步长：1秒，即64点
- 方法：`RAW`与`FULL_C（Transformer-NBM方案C）`
- 折数：3折
- 配对种子：`0、52、161、5216、52161`，不增加fold偏移
- Transformer-NBM：最大300 epoch，patience 20
- TCN分类器：最大5 epoch，patience 2
- GPU：`0,1,2,3,4,5,6`

## 2. Transformer-NBM

```text
X [B,9,128]
  → 16个互不重叠patch，每个patch为9×8=72维
  → [B,16,72]
  → Linear 72→192 + 可学习位置编码
  → Transformer Encoder ×4
       d_model=192, heads=6, FFN=576
  → [B,16,192]
  → 相邻token两两拼接
  → [B,8,384]
  → Linear 384→128 + GELU + Linear 128→64
  → Z [B,8,64]
  → Linear 64→192
  → 每个token复制两次，8→16
  → 可学习的解码位置编码
  → Transformer自注意力解码块 ×2
       d_model=192, heads=6, FFN=576
  → Linear 192→72
  → Fold patches
  → Xhat [B,9,128]
```

固定实现细节：

- Transformer激活函数：GELU
- Transformer dropout：0.10
- LayerNorm：PyTorch标准post-norm TransformerEncoderLayer
- 编码和解码分别使用可学习位置编码，初始化为truncated normal，std=0.02
- Pairwise Token Merge：按时间顺序拼接相邻两个token
- Token Upsampling：参数无关的 `repeat_interleave`，每个瓶颈token复制两次；解码位置编码区分两个时间位置
- 解码阶段只有自注意力，不读取编码器memory
- 不存在encoder-decoder跳跃连接，所有重构信息必须经过 `[B,8,64]` 瓶颈
- 输出层没有激活函数
- 可训练参数量：2,329,736

## 3. NBM训练

- 角色4拟合逐轴RobustScaler并训练Transformer-NBM
- 输入和目标均先经过RobustScaler，再逐窗口逐轴中心化
- 训练增强：40%完整窗口、40% Gaussian噪声（std=0.04）、20%全轴连续Mask（4–8点）
- Loss：SmoothL1，beta=1.0
- Optimizer：AdamW，lr=1e-3，weight_decay=1e-4
- Scheduler：ReduceLROnPlateau，factor=0.5，patience=3，min_lr=1e-5
- Batch size：128
- Gradient clipping：1.0
- 最大epoch：300
- Early stopping patience：20
- 角色5不增强，以最低角色5验证SmoothL1恢复最佳权重
- 恢复最佳权重后才在角色5上计算逐轴残差 `b` 和 `sigma`

## 4. 分类器输入

RAW：

```text
角色4 RobustScaler → 逐窗口逐轴中心化 → [B,9,128] → TCN
```

Transformer-NBM方案C：

```text
e = X-Xhat
q = clip(e/(sigma+1e-6), -12, 12)
r = q-mean_t(q)
F = [r, abs(r), delta(r)] ∈ [B,27,128]
F → 相同TCN
```

方案C使用角色5的 `sigma`，不减去偏移 `b`。

## 5. 分类训练和测试隔离

- 角色6/7：训练TCN，`pos_weight=N_role6/N_role7`
- 角色2/3：TCN早停、模型选择、阈值选择
- 角色0/1：永久测试，只在全部30个TCN及阈值冻结后开放
- Raw和FULL_C使用相同窗口、相同TCN架构、相同训练参数和配对初始化
- 阈值按角色2/3 Balanced Accuracy选择；并列时依次比较FoG F1和更高阈值

任务总数：

- Transformer-NBM：3折 × 5种子 = 15个
- TCN：3折 × 5种子 × 2方法 = 30个
- 最终测试：30个

## 6. 服务器运行

需要同步：

- `scripts/run_daphnet_transformer_nbm300_fold.py`
- `scripts/run_daphnet_nbm300_c_vs_raw_ablation.py`
- `scripts/launch_daphnet_transformer_nbm300_c_vs_raw_ep5pat2_7gpu.py`
- `scripts/run_daphnet_transformer_nbm300_c_vs_raw_ep5pat2_7gpu.sh`

运行：

```bash
cd /document/home_mirror/chb/fog_classification_framework_base
conda activate fogbase
bash scripts/run_daphnet_transformer_nbm300_c_vs_raw_ep5pat2_7gpu.sh
```

默认输出目录：

```text
outputs/daphnet_transformer_nbm300_C_vs_raw_tcn_ep5pat2_seedset_0_52_161_5216_52161
```

只检查任务计划：

```bash
python scripts/launch_daphnet_transformer_nbm300_c_vs_raw_ep5pat2_7gpu.py --dry-run
```

中断后可分别使用 `--phase nbm`、`--phase train`、`--phase evaluate`、`--phase aggregate`。已有完成标记的任务默认跳过；仅在确需重跑时添加 `--overwrite`。

