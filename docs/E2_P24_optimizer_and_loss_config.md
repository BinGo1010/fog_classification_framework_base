# E2-P24 NBM 优化器与损失函数配置

## 1. 适用范围

本文档记录当前 Daphnet `processed_A5_50` 实验中 E2-P24 Normal Behaviour Model（NBM）的正式训练配置。

- 模型：`E2-P24 True Bottleneck Temporal Autoencoder`
- 输入：`[B, 9, 128]`
- 重构目标：`[B, 9, 128]`
- 潜变量：`[B, 24, 32]`
- 参数量：39,457
- 训练方式：`W0` 同窗口自重构
- 去噪设置：`D0`，不注入噪声或掩码
- NBM 训练数据：仅 clean Non-FoG
- NBM 早停数据：独立 clean Non-FoG

FoG 窗口、FoG 标签、C1-MAD、S0/S1/S2/S3、分类器损失和外部测试数据均不参与 NBM 参数更新。

## 2. 优化器配置

| 配置项 | 设置 |
|---|---:|
| 优化器 | AdamW |
| 初始学习率 | `3e-4` |
| Weight decay | `1e-4` |
| Batch size | 64 |
| 最大训练轮数 | 2,000 epochs |
| Early-stopping patience | 100 epochs |
| 最小有效改善量 | `1e-8` |
| 梯度裁剪 | Global norm ≤ 1.0 |
| 学习率调度器 | 无 |
| 混合精度训练 | 无，使用 FP32 |
| 训练集 shuffle | 开启 |
| Early-stop 集 shuffle | 关闭 |
| DataLoader workers | 0（当前正式运行） |
| 随机种子 | 20260802、20260803、20260804 |

优化器定义：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4,
)
```

反向传播后执行梯度裁剪：

```python
loss.backward()
gradient_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)
optimizer.step()
```

若梯度范数为非有限值，当前运行立即失败，不继续保存该模型为有效 checkpoint。

## 3. 训练与早停规则

每个被试、每个随机种子独立训练一个 E2-P24：

```text
10 个被试 × 3 个随机种子 = 30 个 E2-P24 模型
```

训练输入与目标相同：

```text
输入：clean Non-FoG 窗口 X
目标：同一个 clean Non-FoG 窗口 X
输出：重构窗口 X_hat
```

每个 epoch 结束后，在独立的 NBM early-stop clean Non-FoG 上计算完整 L4 损失。满足下式才记为有效改善：

```text
current_earlystop_loss < best_earlystop_loss - 1e-8
```

连续 100 个 epoch 没有有效改善时停止训练。正式预测使用 early-stop loss 最低的 `best_model.pt`，而不是最后一个 epoch 的模型。

保存文件包括：

```text
best_model.pt
last_model.pt
training_log.csv
evaluation_arrays.npz
config.json
```

## 4. L4 总损失

E2-P24 使用以下组合损失：

$$
\mathcal{L}_{L4}
=
0.70\,\mathcal{L}_{\mathrm{SmoothL1}}
+
0.15\,\mathcal{L}_{\mathrm{corr}}
+
0.15\,\mathcal{L}_{\Delta}.
$$

对应实现：

```python
loss = (
    0.70 * smooth_l1_loss(predicted, target)
    + 0.15 * correlation_loss(predicted, target)
    + 0.15 * first_difference_mse(predicted, target)
)
```

### 4.1 Smooth L1 重构损失

$$
\mathcal{L}_{\mathrm{SmoothL1}}
=
\operatorname{SmoothL1}(\hat{X}, X).
$$

其中：

- $X$：标准化后的 clean Non-FoG 目标窗口；
- $\hat{X}$：E2-P24 重构窗口。

该项权重为 0.70，是主要优化目标。它约束重构值接近目标，同时降低少数极端误差相对于普通 MSE 的支配程度。

### 4.2 时间相关性损失

对每个样本、每个通道沿时间维去均值：

$$
\tilde{X}=X-\bar{X},
\qquad
\tilde{\hat{X}}=\hat{X}-\bar{\hat{X}}.
$$

相关系数为：

$$
\rho
=
\frac{
\sum_t \tilde{\hat{X}}_t\tilde{X}_t
}{
\sqrt{
\left(\sum_t \tilde{\hat{X}}_t^2\right)
\left(\sum_t \tilde{X}_t^2\right)
+\epsilon
}
}.
$$

相关性损失为：

$$
\mathcal{L}_{\mathrm{corr}}
=
1-\operatorname{mean}(\rho).
$$

该项权重为 0.15，用于约束重构波形的时间形状和变化趋势。

### 4.3 一阶差分 MSE

定义时间一阶差分：

$$
\Delta X_t=X_t-X_{t-1},
\qquad
\Delta\hat{X}_t=\hat{X}_t-\hat{X}_{t-1}.
$$

差分损失为：

$$
\mathcal{L}_{\Delta}
=
\operatorname{MSE}(\Delta\hat{X},\Delta X).
$$

该项权重为 0.15，用于保持波形斜率、局部变化速度和快速动态，降低过度平滑。

## 5. PyTorch 参考实现

```python
import torch
from torch.nn import functional as F


EPSILON = 1e-8


def correlation_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_centered = predicted - predicted.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)

    numerator = torch.sum(pred_centered * target_centered, dim=-1)
    denominator = torch.sqrt(
        torch.sum(pred_centered.square(), dim=-1)
        * torch.sum(target_centered.square(), dim=-1)
        + EPSILON
    )
    return 1.0 - torch.mean(numerator / denominator)


def l4_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference_loss = F.mse_loss(
        predicted[..., 1:] - predicted[..., :-1],
        target[..., 1:] - target[..., :-1],
    )

    return (
        0.70 * F.smooth_l1_loss(predicted, target)
        + 0.15 * correlation_loss(predicted, target)
        + 0.15 * difference_loss
    )
```

## 6. 与残差诊断的边界

NBM 完成训练并冻结后才执行：

```text
X_hat = E2_P24(X)
R = X - X_hat
R_C1 = C1_MAD(R)
S3 = 0.1 × normalized(S0) + 0.9 × normalized(S1)
```

C1-MAD 与 S3 不属于 NBM 损失函数，也不会通过反向传播改变 E2-P24 参数。

