# Conv-TCN Autoencoder NBM：残差表示配对实验

## 实验目的

在完全相同的数据范围、NBM、TCN-M 主干、训练超参数和阈值规则下比较两组分类器输入：

```text
组 1（r）
r ∈ R^(9×128)

组 2（r_abs_delta）
F = [r, |r|, Δr] ∈ R^(27×128)
```

其中残差先按下式计算并截断：

```text
r = clip((X - X_hat - b) / (sigma + 1e-6), -12, 12)
```

本版本明确删除残差截断后的再次逐窗口逐轴中心化，不再执行：

```text
r_c' = r_c - mean_t(r_c)
```

第二组全部特征均从同一个截断后的 `r` 生成：

```text
|r|[t] = abs(r[t])
Δr[0] = 0
Δr[t] = r[t] - r[t-1], t >= 1
```

`|r|` 和 `Δr` 不再额外缩放、中心化或截断。

## 数据角色协议

- 角色 4：拟合 RobustScaler，并训练 NBM。
- 角色 5：NBM 早停、恢复最佳权重，并估计残差偏移 `b` 和尺度 `sigma`。
- 角色 6/7：分别生成两组特征并训练各自的 TCN-M；`pos_weight=N_role6/N_role7`。
- 角色 2/3：分类器早停、模型选择和阈值选择。
- 角色 0/1：两组的最佳权重和阈值全部冻结后，才统一读取并执行最终测试。

NBM 输入仍保留“RobustScaler 后逐窗口逐轴中心化”。删除的只是 **NBM 输出形成标准化残差以后** 的第二次中心化，二者不要混淆。

## NBM 架构

输入和重构输出均为 `[B,9,128]`：

```text
训练阶段 20% 窗口执行连续 4–8 点全轴 Mask
  -> Conv1d 9->32, k7,s2,p3                 [B,32,64]
  -> TCN 残差块 dilation=1,2                [B,32,64]
  -> Conv1d 32->24, k5,s2,p2                [B,24,32]
  -> TCN 残差块 dilation=1,2                [B,24,32]
  -> Conv1d 24->16, k1                      [B,16,32]
  -> Conv1d 16->24, k3                      [B,24,32]
  -> 线性 Upsample x2                       [B,24,64]
  -> Conv1d 24->32, k5                      [B,32,64]
  -> TCN 残差块 dilation=1,2                [B,32,64]
  -> 线性 Upsample x2                       [B,32,128]
  -> Conv1d 32->16, k7                      [B,16,128]
  -> Conv1d 16->9, k1                       [B,9,128]
```

卷积内部使用 GroupNorm + GELU，残差块 dropout=0.10，输出层无激活。模型共 47,449 个参数。Mask 只用于角色 4 的训练输入，训练目标始终是未遮挡的中心化 clean non-FoG 信号。

## 配对公平性

每一折只训练一个 NBM，两组共用其冻结的 Scaler、NBM、`b` 和 `sigma`，并使用相同窗口、标签、批次顺序、学习率、batch size、最大 epoch、patience、`pos_weight` 和阈值搜索规则。

两组 TCN-M 主干一致，只有输入通道数必须从 9 改为 27。为使初始化尽可能严格配对：

- 所有形状相同的参数逐元素复制，初始值完全一致；
- 27 通道组首层对应 `r` 的 9 通道权重复制自组 1；
- 新增的 18 个输入通道权重初始化为 0；
- 两组训练开始前对相同 `r` 的网络输出一致；
- 配置文件记录两组初始权重的 SHA-256。

程序采用严格的两阶段门控：先完成两组的训练、验证模型选择和阈值冻结；确认两组均冻结后，才首次读取并转换角色 0/1 数据。测试结果不会参与另一组的训练或选择。

## 文件

- `scripts/run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py`：单折训练器，同时完成两个组。
- `scripts/launch_daphnet_processed_nbm_conv_tcn_autoencoder_7gpu.py`：推荐的跨平台服务器调度器。
- `scripts/launch_daphnet_processed_nbm_conv_tcn_autoencoder_7gpu.sh`：Linux shell 调度器。

## 服务器预检

```bash
cd /path/to/fog-merged
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
python scripts/launch_daphnet_processed_nbm_conv_tcn_autoencoder_7gpu.py --dry-run
```

数据目录必须包含：

```text
dataset/1.Daphnet Freezing of Gait Dataset/processed_NBM/nbm_protocol.json
dataset/1.Daphnet Freezing of Gait Dataset/processed_NBM/split_indices/
dataset/1.Daphnet Freezing of Gait Dataset/processed_NBM/records/
```

## 推荐启动命令

Python 调度器：

```bash
python scripts/launch_daphnet_processed_nbm_conv_tcn_autoencoder_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --data-dir "dataset/1.Daphnet Freezing of Gait Dataset/processed_NBM" \
  --output-root outputs/daphnet_processed_NBM_conv_tcn_residual_repr_compare_seed20260807 \
  --representations r,r_abs_delta \
  --nbm-max-epochs 50 \
  --nbm-patience 8 \
  --tcn-max-epochs 30 \
  --tcn-patience 6
```

Linux shell：

```bash
GPU_IDS_CSV=0,1,2,3,4,5,6 \
OUTPUT_ROOT="$PWD/outputs/daphnet_processed_NBM_conv_tcn_residual_repr_compare_seed20260807" \
bash scripts/launch_daphnet_processed_nbm_conv_tcn_autoencoder_7gpu.sh \
  --representations r,r_abs_delta
```

数据协议只有 3 个独立外层折，因此默认使用 GPU 0、1、2，每张卡负责一折；在该折内先训练一个共享 NBM，再顺序训练两个分类器。其余 4 张卡保持空闲，避免用 DDP 改变 batch 和随机训练语义。

## 输出结构

```text
output_root/
  fold_0/
    checkpoints/conv_tcn_nbm_best.pt
    conv_tcn_nbm_training_validation.{png,svg,pdf}
    nbm_frozen.json
    paired_classifier_initialization.json
    groups/
      r/
        checkpoints/tcn.pt
        logs/tcn_history.csv
        metrics.json
        predictions.csv
        test_probabilities.npz
        tcn_training_validation.{png,svg,pdf}
        test_confusion_matrix.{png,svg,pdf}
      r_abs_delta/
        ...同上...
  fold_1/...
  fold_2/...
  fold_metrics_r.csv
  fold_metrics_r_abs_delta.csv
  subject_metrics_mean_r.csv
  subject_metrics_mean_r_abs_delta.csv
  representation_comparison.csv
  summary.json
  DONE.json
```

## 单折调试

只执行数据、协议、网络和特征形状预检：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py \
  --fold 0 --device cuda --dry-run \
  --representations r,r_abs_delta
```

一轮短程冒烟测试：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py \
  --fold 0 --device cuda \
  --output-root outputs/smoke_residual_repr \
  --representations r,r_abs_delta \
  --nbm-max-epochs 1 --nbm-patience 1 \
  --tcn-max-epochs 1 --tcn-patience 1
```
