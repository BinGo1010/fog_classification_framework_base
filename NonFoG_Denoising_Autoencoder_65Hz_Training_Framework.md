# 基于 Non-FoG 数据的去噪自编码器训练框架

## 1. 项目目标

本项目使用帕金森病患者的 **Non-FoG（非冻结步态）IMU信号**训练一个去噪自编码器（Denoising Autoencoder, DAE），使模型学习Non-FoG步态信号的主要时序结构、频率特征以及多通道协同关系。

训练阶段仅使用Non-FoG数据。模型接收经过遮挡或噪声扰动的Non-FoG窗口，并恢复对应的干净Non-FoG窗口：

\[
\tilde{x}_{N}=\operatorname{Corrupt}(x_N)
\]

\[
z=E(\tilde{x}_{N})
\]

\[
\hat{x}_{N}=D(z)
\]

其中：

- \(x_N\)：原始干净Non-FoG信号；
- \(\tilde{x}_{N}\)：经过遮挡或噪声扰动的Non-FoG信号；
- \(E(\cdot)\)：编码器；
- \(z\)：低维潜在表示；
- \(D(\cdot)\)：解码器；
- \(\hat{x}_{N}\)：模型重建的Non-FoG信号。

模型通过最小化重建信号与原始干净Non-FoG信号之间的差异，学习正常步态数据分布。

---

## 2. 数据设定

### 2.1 采样频率

原始IMU数据采样频率为：

\[
f_s=65\text{ Hz}
\]

若采用2秒滑动窗口，则每个窗口包含：

\[
T=65\times2=130
\]

个采样点。

### 2.2 输入数据形状

假设使用3个三轴IMU，总通道数为9，则单个输入窗口为：

\[
x\in\mathbb{R}^{9\times130}
\]

批量输入模型时，数据形状为：

\[
X\in\mathbb{R}^{B\times9\times130}
\]

其中：

- \(B\)：batch size；
- 9：IMU信号通道数；
- 130：2秒窗口中的采样点数。

若实际通道数不同，应将通道数设置为可配置参数：

```yaml
data:
  sampling_rate: 65
  window_seconds: 2
  sequence_length: 130
  input_channels: 9
```

模型输出尺寸与输入尺寸相同：

\[
\hat{x}\in\mathbb{R}^{B\times9\times130}
\]

---

## 3. 数据准备

### 3.1 原始数据要求

每个IMU样本至少应包含：

- `subject_id`：受试者编号；
- `trial_id`：试验或采集轮次编号；
- `timestamp`：时间戳；
- 各IMU通道数据；
- `label`：FoG或Non-FoG标签。

建议将处理后的窗口保存为：

```text
subject_id
trial_id
window_id
start_time
end_time
label
signal: [channels, 130]
```

### 3.2 Non-FoG数据筛选

训练去噪自编码器时，只保留完整的Non-FoG窗口：

\[
\mathcal{D}_{train}
=
\{x_i\mid y_i=\text{Non-FoG}\}
\]

若一个窗口同时包含FoG和Non-FoG片段，则不应作为Non-FoG训练样本。

建议使用严格的窗口标签规则：

- 窗口内全部采样点均为Non-FoG，才标记为Non-FoG；
- 包含FoG转换边界的窗口不进入训练集；
- 标签不确定或视频无法确认的窗口不进入训练集。

### 3.3 缺失数据处理

在窗口切分前处理缺失数据：

1. 短时缺口可进行插值；
2. 长时间缺口不应强行插值；
3. 包含较长缺口的窗口应删除；
4. 每个窗口应保存数据质量标记；
5. 不应将缺失值直接以NaN形式输入神经网络。

可以设置窗口有效性条件，例如：

```text
单通道有效数据比例 ≥ 95%
所有通道均无长时间连续缺口
窗口内不存在传感器饱和
窗口内不存在明显同步错误
```

---

## 4. 数据划分

### 4.1 受试者独立划分

为了避免同一患者的数据同时出现在训练集和验证集中，应按受试者划分数据。

在每个外层留一受试者评估折中：

```text
测试受试者：
    1名完全留出的患者

训练候选受试者：
    其余患者

训练集：
    训练受试者的Non-FoG窗口

验证集：
    从训练候选受试者中独立选择患者，
    使用其Non-FoG窗口进行模型选择
```

去噪自编码器训练阶段不使用测试患者的任何数据，包括测试患者的Non-FoG数据。

### 4.2 防止重叠窗口泄漏

若使用滑动窗口，相邻窗口可能存在较高重叠率。不能将同一连续步态片段中的重叠窗口随机分配到训练集和验证集。

应优先按照以下单位划分：

1. 受试者；
2. 采集轮次；
3. 连续步行任务；
4. 连续时间段。

---

## 5. 数据标准化

### 5.1 通道级标准化

仅使用当前训练集中的Non-FoG窗口计算每个通道的均值与标准差：

\[
\mu_c
=
\frac{1}{N}
\sum_{i,t}x_{i,c,t}
\]

\[
\sigma_c
=
\sqrt{
\frac{1}{N}
\sum_{i,t}(x_{i,c,t}-\mu_c)^2
}
\]

标准化公式为：

\[
x'_{i,c,t}
=
\frac{x_{i,c,t}-\mu_c}
{\sigma_c+\epsilon}
\]

其中：

\[
\epsilon=10^{-8}
\]

训练集、验证集和后续测试数据必须使用同一组训练集标准化参数。

### 5.2 不建议逐窗口标准化

第一版不建议对每个窗口单独进行z-score标准化，因为这可能移除：

- 步态幅值变化；
- 运动能量变化；
- 患者不同状态下的幅值差异；
- FoG相关的运动幅值衰减信息。

### 5.3 保存标准化参数

每个训练折应保存：

```text
scaler_mean.npy
scaler_std.npy
```

---

## 6. 去噪扰动模块

训练时，原始Non-FoG窗口作为重建目标，经过扰动后的窗口作为模型输入：

\[
x_N
\xrightarrow{\operatorname{Corrupt}}
\tilde{x}_N
\xrightarrow{DAE}
\hat{x}_N
\]

建议第一版使用以下三种简单扰动：

1. 连续时间块遮挡；
2. 单通道遮挡；
3. 轻度高斯噪声。

每个训练窗口随机选择一种处理方式，避免多种强扰动同时叠加。

### 6.1 连续时间块遮挡

随机遮挡连续的时间区域，使模型根据窗口中的其余信号恢复缺失片段。

65 Hz采样率下：

- 100 ms约为7个采样点；
- 200 ms约为13个采样点；
- 300 ms约为20个采样点；
- 400 ms约为26个采样点。

第一版建议遮挡长度为：

\[
L_{mask}\in[7,26]
\]

即约100～400 ms。

对所有通道遮挡相同的时间区间：

```python
start = random.randint(0, sequence_length - mask_length)
corrupted[:, start:start + mask_length] = 0.0
```

由于输入已经进行训练集通道级z-score标准化，0近似表示训练数据均值。

建议概率：

\[
p_{time}=0.50
\]

### 6.2 单通道遮挡

随机选择一个IMU信号通道，并将该通道在整个2秒窗口内置零：

```python
channel = random.randint(0, input_channels - 1)
corrupted[channel, :] = 0.0
```

该任务促使模型利用：

- 同一IMU不同轴之间的关系；
- 不同IMU之间的运动协同；
- 时间上下文信息；

恢复被遮挡通道。

建议概率：

\[
p_{channel}=0.20
\]

### 6.3 高斯噪声

对标准化后的Non-FoG输入添加轻度高斯噪声：

\[
\epsilon
\sim
\mathcal{N}(0,\sigma_n^2)
\]

\[
\tilde{x}=x+\epsilon
\]

建议从以下范围随机采样噪声标准差：

\[
\sigma_n\in[0.01,0.05]
\]

建议概率：

\[
p_{noise}=0.20
\]

### 6.4 干净输入

保留一部分不经过扰动的Non-FoG窗口：

\[
\tilde{x}=x
\]

这样可以保证模型不仅具备去噪能力，也能稳定重建干净的Non-FoG输入。

建议概率：

\[
p_{clean}=0.10
\]

### 6.5 推荐扰动配置

| 处理方式 | 概率 |
|---|---:|
| 连续时间块遮挡 | 0.50 |
| 单通道遮挡 | 0.20 |
| 高斯噪声 | 0.20 |
| 不进行扰动 | 0.10 |

扰动模块只在训练阶段启用。验证阶段使用干净Non-FoG输入评估模型重建能力。

---

## 7. 模型结构

采用轻量级TCN去噪自编码器，包括：

1. 输入卷积层；
2. TCN编码器；
3. 潜变量瓶颈；
4. TCN解码器；
5. 输出卷积层。

整体流程：

```text
输入：[B, 9, 130]
        ↓
输入卷积层
        ↓
TCN编码块1
        ↓
下采样：130 → 65
        ↓
TCN编码块2
        ↓
下采样：65 → 33
        ↓
TCN编码块3
        ↓
下采样：33 → 17
        ↓
展平与全连接层
        ↓
潜变量z：[B, 128]
        ↓
全连接层与形状恢复
        ↓
TCN解码块1
        ↓
上采样：17 → 33
        ↓
TCN解码块2
        ↓
上采样：33 → 65
        ↓
TCN解码块3
        ↓
上采样：65 → 130
        ↓
输出卷积层
        ↓
重建信号：[B, 9, 130]
```

由于130经过连续二倍下采样时会出现奇数长度，程序中应显式指定目标长度，避免仅依赖`scale_factor=2`造成尺寸不一致。

---

## 8. 编码器设计

### 8.1 输入卷积层

输入形状：

\[
[B,9,130]
\]

输入卷积层：

```python
nn.Conv1d(
    in_channels=9,
    out_channels=32,
    kernel_size=7,
    stride=1,
    padding=3
)
```

后接：

```text
GroupNorm
GELU
```

推荐使用GroupNorm而不是BatchNorm，以减少batch大小和患者分布差异对训练的影响。

### 8.2 TCN残差块

每个TCN残差块包括：

```text
Conv1D
GroupNorm
GELU
Dropout
Conv1D
GroupNorm
GELU
Residual connection
```

推荐编码器配置：

| 编码模块 | 输入通道 | 输出通道 | 膨胀率 |
|---|---:|---:|---:|
| Encoder Block 1 | 32 | 32 | 1 |
| Encoder Block 2 | 32 | 64 | 2 |
| Encoder Block 3 | 64 | 128 | 4 |

若输入通道数与输出通道数不同，残差支路使用\(1\times1\)卷积匹配通道数。

### 8.3 下采样层

使用步长卷积完成时间下采样：

```python
nn.Conv1d(
    in_channels=C,
    out_channels=C,
    kernel_size=4,
    stride=2,
    padding=1
)
```

对于长度130：

```text
130 → 65 → 32或33 → 16或17
```

具体长度取决于卷积参数。为了保证解码器输出最终严格恢复为130点，应在模型初始化或第一次前向传播时记录每一层实际输出长度。

推荐直接记录：

```python
encoder_lengths = [130, 65, 33, 17]
```

并在解码阶段按照这些目标长度进行插值。

### 8.4 潜变量瓶颈

编码器最终特征假设为：

\[
h\in\mathbb{R}^{128\times17}
\]

展平后：

\[
h_{flat}\in\mathbb{R}^{2176}
\]

通过全连接层获得潜变量：

```python
nn.Linear(128 * 17, 128)
```

潜变量为：

\[
z\in\mathbb{R}^{128}
\]

潜变量瓶颈的作用是：

- 压缩输入信息；
- 限制模型直接复制输入；
- 促使模型保留Non-FoG主要结构；
- 降低异常细节被完整重建的可能性。

---

## 9. 解码器设计

### 9.1 潜变量恢复

首先将潜变量映射回卷积特征：

```python
nn.Linear(128, 128 * 17)
```

随后变形为：

\[
[B,128,17]
\]

### 9.2 上采样

建议使用：

```text
线性插值
+
Conv1D
```

而不是完全依赖转置卷积，以降低重建信号中的周期性伪影。

例如：

```python
x = F.interpolate(
    x,
    size=target_length,
    mode="linear",
    align_corners=False
)
x = conv(x)
```

解码长度按照编码阶段记录的长度恢复：

```text
17 → 33 → 65 → 130
```

### 9.3 解码器通道配置

| 解码模块 | 输入通道 | 输出通道 | 目标长度 |
|---|---:|---:|---:|
| Decoder Block 1 | 128 | 64 | 33 |
| Decoder Block 2 | 64 | 32 | 65 |
| Decoder Block 3 | 32 | 32 | 130 |
| Output Layer | 32 | 9 | 130 |

### 9.4 输出层

```python
nn.Conv1d(
    in_channels=32,
    out_channels=9,
    kernel_size=7,
    stride=1,
    padding=3
)
```

输出层不使用Sigmoid或Tanh激活，因为经过z-score标准化后的IMU信号不位于固定区间。

---

## 10. 模型连接限制

模型内部的TCN残差块可以使用短残差连接，但第一版不使用编码器与解码器之间的U-Net式长跳跃连接。

允许：

```text
TCN块输入 → TCN块输出
```

不使用：

```text
原始输入 → 最终输出

编码器浅层特征 → 解码器对应层
```

这样可以降低模型直接复制输入细节的能力，使潜变量瓶颈真正发挥作用。

---

## 11. 损失函数

总训练损失由以下三部分组成：

\[
\mathcal{L}
=
\mathcal{L}_{time}
+
\lambda_d\mathcal{L}_{diff}
+
\lambda_f\mathcal{L}_{freq}
\]

### 11.1 时域Huber损失

使用Huber损失约束重建信号与原始干净Non-FoG信号的逐点误差：

\[
\mathcal{L}_{time}
=
\operatorname{Huber}(x,\hat{x})
\]

PyTorch实现：

```python
time_loss = torch.nn.SmoothL1Loss(beta=1.0)
```

Huber损失相比MSE对IMU信号中的尖峰和异常值更稳定。

### 11.2 一阶差分损失

定义时间方向的一阶差分：

\[
\Delta x_t=x_t-x_{t-1}
\]

重建信号的一阶差分为：

\[
\Delta\hat{x}_t=\hat{x}_t-\hat{x}_{t-1}
\]

差分损失为：

\[
\mathcal{L}_{diff}
=
\operatorname{Huber}
(\Delta x,\Delta\hat{x})
\]

该损失用于保持：

- 局部变化趋势；
- 信号斜率；
- 步态动态结构；
- 重建波形的变化速度。

建议：

\[
\lambda_d=0.2
\]

### 11.3 频域损失

对原始信号与重建信号计算短时傅里叶变换：

\[
X=\operatorname{STFT}(x)
\]

\[
\hat{X}=\operatorname{STFT}(\hat{x})
\]

频域损失为：

\[
\mathcal{L}_{freq}
=
\left\|
\log(1+|X|)
-
\log(1+|\hat{X}|)
\right\|_1
\]

由于采样频率为65 Hz，Nyquist频率为：

\[
f_{Nyquist}=32.5\text{ Hz}
\]

推荐STFT参数：

```yaml
n_fft: 64
win_length: 64
hop_length: 16
```

在65 Hz采样率下：

- 64点窗对应约0.985秒；
- 16点步长对应约0.246秒；
- 频率分辨率约为\(65/64\approx1.02\) Hz。

建议：

\[
\lambda_f=0.1
\]

### 11.4 总损失

第一版推荐：

\[
\boxed{
\mathcal{L}
=
\mathcal{L}_{Huber}
+
0.2\mathcal{L}_{diff}
+
0.1\mathcal{L}_{STFT}
}
\]

---

## 12. 训练参数

推荐初始配置：

| 参数 | 推荐值 |
|---|---:|
| Optimizer | AdamW |
| Initial learning rate | \(1\times10^{-3}\) |
| Weight decay | \(1\times10^{-4}\) |
| Batch size | 128或256 |
| Maximum epochs | 100 |
| Early-stopping patience | 15 |
| Dropout | 0.10 |
| Gradient clipping | 1.0 |
| Latent dimension | 128 |
| Scheduler | ReduceLROnPlateau |

优化器：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)
```

学习率调度器：

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
    min_lr=1e-5
)
```

梯度裁剪：

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

---

## 13. 训练流程

### 13.1 单个batch训练过程

对每个训练batch：

1. 读取干净Non-FoG窗口；
2. 随机生成扰动输入；
3. 将扰动输入送入编码器；
4. 得到潜变量；
5. 通过解码器恢复完整信号；
6. 计算时域、差分和频域损失；
7. 反向传播并更新参数。

伪代码：

```python
model.train()

for clean_x in train_loader:

    clean_x = clean_x.to(device)

    corrupted_x = corruption_module(clean_x)

    reconstructed_x, latent_z = model(corrupted_x)

    loss_time = huber_loss(
        reconstructed_x,
        clean_x
    )

    clean_diff = clean_x[:, :, 1:] - clean_x[:, :, :-1]
    recon_diff = (
        reconstructed_x[:, :, 1:]
        - reconstructed_x[:, :, :-1]
    )

    loss_diff = huber_loss(
        recon_diff,
        clean_diff
    )

    loss_freq = stft_loss(
        reconstructed_x,
        clean_x,
        n_fft=64,
        win_length=64,
        hop_length=16
    )

    total_loss = (
        loss_time
        + 0.2 * loss_diff
        + 0.1 * loss_freq
    )

    optimizer.zero_grad()
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()
```

### 13.2 验证过程

验证阶段关闭随机扰动，直接输入干净Non-FoG窗口：

```python
model.eval()

with torch.no_grad():

    for clean_x in val_loader:

        reconstructed_x, latent_z = model(clean_x)

        val_loss = combined_loss(
            reconstructed_x,
            clean_x
        )
```

这样验证损失反映模型对未见Non-FoG数据的正常重建能力。

也可以额外记录被人为扰动验证数据的恢复能力，但模型选择应主要依据独立验证受试者的Non-FoG重建损失。

---

## 14. Early Stopping与模型保存

每个epoch结束后，计算验证集总损失：

\[
\mathcal{L}_{val}
\]

当验证损失低于历史最佳值时保存模型：

```python
if val_loss < best_val_loss:

    best_val_loss = val_loss

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": config
        },
        "best_model.pt"
    )
```

若连续15个epoch验证损失没有改善，则停止训练。

每个训练折建议保存：

```text
best_model.pt
last_model.pt
config.yaml
training_history.csv
scaler_mean.npy
scaler_std.npy
split_subjects.json
```

---

## 15. 模型训练输出

训练完成后，应至少输出以下内容。

### 15.1 训练记录

```text
epoch
training_total_loss
training_time_loss
training_diff_loss
training_freq_loss
validation_total_loss
validation_time_loss
validation_diff_loss
validation_freq_loss
learning_rate
```

### 15.2 模型文件

```text
best_model.pt
last_model.pt
```

### 15.3 数据处理参数

```text
scaler_mean.npy
scaler_std.npy
```

### 15.4 数据划分信息

```json
{
  "train_subjects": [],
  "validation_subjects": [],
  "test_subject": ""
}
```

### 15.5 模型配置

```yaml
data:
  sampling_rate: 65
  window_seconds: 2
  sequence_length: 130
  input_channels: 9
  train_label: non_fog
  normalization: training_fold_channel_zscore

corruption:
  time_mask_probability: 0.50
  time_mask_min_length: 7
  time_mask_max_length: 26

  channel_mask_probability: 0.20
  channel_mask_count: 1

  gaussian_noise_probability: 0.20
  gaussian_noise_std_min: 0.01
  gaussian_noise_std_max: 0.05

  clean_probability: 0.10

model:
  name: tcn_denoising_autoencoder
  encoder_channels: [32, 64, 128]
  latent_dim: 128
  dropout: 0.10
  activation: gelu
  normalization: group_norm
  use_encoder_decoder_skip_connections: false

loss:
  time_loss: huber
  huber_beta: 1.0
  temporal_difference_weight: 0.20
  frequency_weight: 0.10
  n_fft: 64
  win_length: 64
  hop_length: 16

training:
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.0001
  batch_size: 256
  max_epochs: 100
  early_stopping_patience: 15
  gradient_clip_norm: 1.0
  scheduler: reduce_on_plateau
```

---

## 16. 推荐工程目录

```text
nonfog_dae/
│
├── configs/
│   └── nonfog_dae_65hz.yaml
│
├── data/
│   ├── dataset.py
│   ├── window_loader.py
│   ├── preprocessing.py
│   ├── normalization.py
│   ├── corruptions.py
│   └── subject_split.py
│
├── models/
│   ├── tcn_block.py
│   ├── encoder.py
│   ├── decoder.py
│   └── nonfog_dae.py
│
├── losses/
│   ├── reconstruction_loss.py
│   ├── temporal_difference_loss.py
│   ├── stft_loss.py
│   └── combined_loss.py
│
├── training/
│   ├── trainer.py
│   ├── validator.py
│   ├── checkpoint.py
│   └── early_stopping.py
│
├── scripts/
│   ├── train_nonfog_dae.py
│   └── run_subject_folds.py
│
├── outputs/
│   └── fold_x/
│       ├── checkpoints/
│       ├── logs/
│       ├── configs/
│       └── scalers/
│
├── requirements.txt
├── README.md
└── main.py
```

---

## 17. 完整训练逻辑

```text
原始65 Hz IMU信号
        ↓
时间同步、缺失值处理和质量控制
        ↓
切分2秒窗口，每个窗口130个采样点
        ↓
筛选标签完全为Non-FoG的窗口
        ↓
按受试者划分训练集和验证集
        ↓
使用训练集Non-FoG计算通道级标准化参数
        ↓
训练阶段随机执行：
    连续时间块遮挡
    单通道遮挡
    轻度高斯噪声
    或保留干净输入
        ↓
扰动后的Non-FoG输入
        ↓
TCN编码器
        ↓
128维潜变量瓶颈
        ↓
TCN解码器
        ↓
恢复9通道×130点Non-FoG信号
        ↓
计算：
    Huber时域损失
    一阶差分损失
    STFT频域损失
        ↓
反向传播更新模型参数
        ↓
使用独立验证受试者的Non-FoG数据评估
        ↓
Early stopping
        ↓
保存验证损失最低的Non-FoG去噪自编码器
```

---

## 18. 最终模型定义

训练完成后的模型可以表示为：

\[
\hat{x}_{N}
=
D_{\theta_D}
\left(
E_{\theta_E}(x)
\right)
\]

其中参数：

\[
\theta_E,\theta_D
\]

通过以下优化问题获得：

\[
\theta_E^*,\theta_D^*
=
\arg\min_{\theta_E,\theta_D}
\mathbb{E}_{x_N\sim\mathcal{D}_{Non-FoG}}
\left[
\mathcal{L}
\left(
D_{\theta_D}
(E_{\theta_E}(\operatorname{Corrupt}(x_N))),
x_N
\right)
\right]
\]

训练数据只包括Non-FoG窗口，模型通过从扰动输入中恢复干净Non-FoG信号，学习Non-FoG步态的低维时序表示及其重建映射。
