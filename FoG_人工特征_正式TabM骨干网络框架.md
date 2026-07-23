# 基于人工多时间尺度特征的正式 TabM 骨干网络框架  
## FoG vs non-FoG 二分类｜LOSO 跨受试者评估

> 目标：保持现有人工特征提取与标签定义不变，仅将普通 MLP 分类器替换为符合 ICLR 2025 TabM 核心思想的参数高效集成网络，从而公平判断 TabM 对跨受试者 FoG 二分类的增益。

---

## 1. 任务定义

### 1.1 输入

以同一个判定时刻 \(t\) 对齐两组人工特征：

#### 2 秒短窗口特征

- 均值 Mean
- 标准差 Standard Deviation
- 均方根 RMS
- 峰峰值 Peak-to-Peak
- Jerk RMS

记为：

\[
\mathbf{x}_{short}\in\mathbb{R}^{D_s}
\]

#### 6 秒长窗口特征

- 冻结指数 Freezing Index
- 主频 Dominant Frequency
- 频谱熵 Spectral Entropy
- \(0.5\text{--}3\,\mathrm{Hz}\) 能量
- \(3\text{--}8\,\mathrm{Hz}\) 能量

记为：

\[
\mathbf{x}_{long}\in\mathbb{R}^{D_l}
\]

#### 最终输入

\[
\mathbf{x}
=
[\mathbf{x}_{short};\mathbf{x}_{long}]
\in\mathbb{R}^{D},
\qquad D=D_s+D_l
\]

若上述 10 种特征均对 \(C\) 个通道分别计算，则：

\[
D=10C
\]

实际实现必须由特征清单动态确定 `input_dim`，不要在模型中硬编码通道数。

### 1.2 输出

二分类：

- `0`：non-FoG
- `1`：FoG

推荐使用两个输出 logit：

\[
\mathbf{z}\in\mathbb{R}^{2}
\]

而不是先将任务写成单 logit，以便直接使用交叉熵并保持类别扩展能力。

---

## 2. 总体数据流

```text
2 秒短窗口人工特征 ─┐
                    ├─> 特征拼接
6 秒长窗口人工特征 ─┘
                         │
                         ▼
             训练折内特征预处理
                         │
                 x: [B, D]
                         │
                         ▼
                EnsembleView
                         │
              X: [B, K, D]
                         │
                         ▼
       正式 TabM BatchEnsemble 骨干
      共享主权重 W + 成员适配器 R/S/B
                         │
                         ▼
             H: [B, K, d_hidden]
                         │
                         ▼
          K 个互不共享的分类预测头
                         │
                         ▼
            logits: [B, K, 2]
             │                     │
        训练阶段                推理阶段
             │                     │
  各成员分别计算损失       softmax 后平均概率
             │                     │
   平均 K 个成员损失        p_fog: [B]
```

推荐主实验：

```yaml
model_name: fog_tabm
arch_type: tabm
ensemble_size_k: 32
n_blocks: 2
d_block: 256
dropout: 0.10
d_out: 2
activation: ReLU
normalization: null
```

---

## 3. 正式 TabM 的不可替代核心

以下条件必须同时满足，否则只能称为“多头 MLP”或“普通集成”，不能称为正式 TabM。

### 3.1 在第一次特征混合之前建立成员差异

输入首先从：

\[
X\in\mathbb{R}^{B\times D}
\]

扩展为：

\[
X^{(0)}\in\mathbb{R}^{B\times K\times D}
\]

其中 \(K\) 为隐式子模型数量。

关键要求：第一组成员专属输入缩放参数 \(R_1\) 必须在第一个共享线性变换 \(W_1\) 之前生效。

错误结构：

```text
x -> 普通 Linear -> EnsembleView -> ...
```

上述结构在成员产生差异前已经混合了特征，不符合 TabM 的关键设计。

正确结构：

```text
x -> EnsembleView -> R1 成员缩放 -> 共享 W1 -> ...
```

### 3.2 每个骨干线性层使用 BatchEnsemble 参数化

第 \(l\) 层对第 \(k\) 个隐式成员的计算为：

\[
\mathbf{h}_{l,k}
=
\operatorname{Dropout}
\left(
\operatorname{ReLU}
\left(
\mathbf{s}_{l,k}\odot
\left[
W_l
\left(
\mathbf{r}_{l,k}\odot\mathbf{h}_{l-1,k}
\right)
\right]
+
\mathbf{b}_{l,k}
\right)
\right)
\]

其中：

- \(W_l\)：所有成员共享的主权重矩阵；
- \(\mathbf{r}_{l,k}\)：第 \(k\) 个成员的输入缩放适配器；
- \(\mathbf{s}_{l,k}\)：第 \(k\) 个成员的输出缩放适配器；
- \(\mathbf{b}_{l,k}\)：第 \(k\) 个成员的独立偏置；
- \(\odot\)：逐元素乘法。

等效成员权重为：

\[
W_{l,k}^{equiv}
=
W_l\odot
\left(
\mathbf{s}_{l,k}\mathbf{r}_{l,k}^{T}
\right)
\]

因此，每个成员拥有不同的等效映射，但无需保存一套完整的 \(W_{l,k}\)。

### 3.3 参数量关系

对于输入维度 \(d_{in}\)、输出维度 \(d_{out}\) 的一层：

#### \(K\) 个完整独立线性层

\[
P_{independent}
=
K(d_{out}d_{in}+d_{out})
\]

#### TabM BatchEnsemble 层

\[
P_{TabM}
=
d_{out}d_{in}
+
K(d_{in}+2d_{out})
\]

当隐藏维度较大时，TabM 共享占主导的矩阵 \(W\)，只为每个成员增加低成本适配器。

### 3.4 独立预测头

TabM 骨干输出：

\[
H\in\mathbb{R}^{B\times K\times d_h}
\]

必须使用 \(K\) 个互不共享的线性预测头：

\[
Z_k=W_k^{head}H_k+b_k^{head}
\]

最终：

\[
Z\in\mathbb{R}^{B\times K\times 2}
\]

不要将 \(K\) 个成员先池化成一个向量后再使用单一分类头。

### 3.5 成员之间禁止交互

在得到最终预测之前，不应在成员维度使用：

- attention；
- mean/max pooling；
- 跨成员归一化；
- 跨成员特征融合；
- 将成员维度展平后使用普通线性层。

每个成员必须沿自己的预测路径独立计算，只在最终推理概率层面进行集成。

---

## 4. TabM 风格初始化

正式 TabM 与直接使用普通 BatchEnsemble 的重要区别之一是初始化。

### 4.1 第一层

第一层输入适配器 \(R_1\)：

```text
随机初始化：Normal 或 Random Signs
```

推荐：

```yaml
first_input_scaling_init: normal
```

其作用是在人工特征被共享矩阵第一次混合之前，使 \(K\) 个成员看到不同的特征缩放表示。

第一层输出适配器 \(S_1\)：

```text
初始化为 1
```

### 4.2 后续层

对第 \(l>1\) 层：

```text
R_l 初始化为 1
S_l 初始化为 1
B_l 初始化为 0
```

这意味着模型初始化时：

- 成员差异主要来自第一层输入适配器；
- 后续层最初接近共享 MLP；
- 训练过程中，各层适配器再逐渐形成成员差异。

### 4.3 预测头

每个成员预测头独立初始化，不共享权重。

### 4.4 推荐官方组件配置

```python
tabm_init = True
scaling_init = "normal"

第一层:
    scaling_init = ("normal", "ones")

后续层:
    scaling_init = "ones"
```

---

## 5. 推荐骨干网络

### 5.1 主模型

```text
Input
[B, D]
   │
   ▼
EnsembleView(K=32)
[B, 32, D]
   │
   ▼
LinearBatchEnsemble(D -> 256)
第一层 R 随机，S 为 1
   │
ReLU
   │
Dropout(0.10)
   │
   ▼
LinearBatchEnsemble(256 -> 256)
后续 R、S 初始化为 1
   │
ReLU
   │
Dropout(0.10)
   │
   ▼
LinearEnsemble(256 -> 2)
32 个独立预测头
   │
   ▼
[B, 32, 2]
```

### 5.2 不默认使用归一化层

首版建议不加入：

- BatchNorm；
- LayerNorm；
- 残差块；
- attention；
- 门控模块。

原因不是这些结构一定无效，而是正式 TabM 的基础骨干本身是简单的：

```text
Linear -> ReLU -> Dropout
```

第一阶段应只替换 MLP 为 TabM，避免同时改变多个变量。

---

## 6. PyTorch 模型接口骨架

推荐优先使用官方 `tabm` 包，而不是自行近似实现。

```bash
pip install tabm
```

### 6.1 模型定义

```python
from __future__ import annotations

import torch
from torch import nn
from tabm import TabM


class FoGTabM(nn.Module):
    """人工特征输入的正式 TabM FoG 二分类器。"""

    def __init__(
        self,
        input_dim: int,
        *,
        n_classes: int = 2,
        k: int = 32,
        n_blocks: int = 2,
        d_block: int = 256,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if n_classes != 2:
            raise ValueError("This configuration is defined for binary classification.")

        self.k = k
        self.n_classes = n_classes

        self.model = TabM.make(
            n_num_features=input_dim,
            d_out=n_classes,
            arch_type="tabm",
            k=k,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape [B, D], got {tuple(x.shape)}")

        logits = self.model(x)

        expected = (x.shape[0], self.k, self.n_classes)
        if logits.shape != expected:
            raise RuntimeError(
                f"Expected TabM output shape {expected}, got {tuple(logits.shape)}"
            )
        return logits
```

输出形状必须为：

```text
[B, K, 2]
```

而不是：

```text
[B, 2]
```

---

## 7. 正确的训练损失

### 7.1 核心原则

训练时必须让每个成员独立承担分类损失：

\[
\mathcal{L}
=
\frac{1}{BK}
\sum_{b=1}^{B}
\sum_{k=1}^{K}
\operatorname{CE}
\left(
Z_{b,k},y_b
\right)
\]

禁止先平均成员预测再计算一次损失：

\[
\operatorname{CE}
\left(
\frac{1}{K}\sum_k Z_k,y
\right)
\]

后者会改变 TabM 的训练机制。

### 7.2 实现

```python
import torch
import torch.nn.functional as F


def tabm_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    logits: [B, K, 2]
    target: [B]
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, K, C].")

    batch_size, k, n_classes = logits.shape
    target_members = (
        target[:, None]
        .expand(batch_size, k)
        .reshape(batch_size * k)
    )

    return F.cross_entropy(
        logits.reshape(batch_size * k, n_classes),
        target_members,
        weight=class_weight,
    )
```

### 7.3 类别不平衡

为公平比较，主实验必须沿用现有 MLP 的损失和采样策略。

例如，现有 MLP 若使用类别权重，则每个 LOSO 训练折仅根据训练受试者计算：

\[
w_c=\frac{N}{2N_c}
\]

不得使用测试受试者的标签分布计算类别权重。

建议分别记录：

```yaml
main_comparison:
  sampling: identical_to_mlp
  class_weight: identical_to_mlp

optional_ablation:
  - unweighted_cross_entropy
  - weighted_cross_entropy
```

不要在比较 MLP 与 TabM 时，同时更换采样器、损失函数和分类阈值。

---

## 8. 正确的推理与集成

### 8.1 概率平均

对二分类输出：

\[
P_{b,k}
=
\operatorname{Softmax}(Z_{b,k})
\]

最终预测概率：

\[
\bar{P}_b
=
\frac{1}{K}
\sum_{k=1}^{K}
P_{b,k}
\]

代码：

```python
@torch.no_grad()
def predict_tabm(
    model: nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(x)                       # [B, K, 2]
    member_probs = logits.softmax(dim=-1)  # [B, K, 2]
    mean_probs = member_probs.mean(dim=1)  # [B, 2]
    fog_score = mean_probs[:, 1]            # [B]
    return mean_probs, fog_score
```

分类任务通常应平均概率，不直接平均 logits。

### 8.2 阈值

主结果至少报告：

1. 固定阈值 \(0.5\)；
2. 仅在内层验证受试者上选择的阈值。

阈值不得根据外层测试受试者调节。

可选目标：

```yaml
threshold_objective:
  - max_fog_f1
  - max_balanced_accuracy
  - target_recall
```

真实辅助系统更关注漏检风险时，可在验证集上选择达到目标 FoG Recall 的阈值，并同步报告 Precision 和误报率。

---

## 9. LOSO 数据划分规范

### 9.1 外层测试

对每位受试者 \(S_i\)：

```text
Test      = S_i
Train/Val = 其余受试者
```

### 9.2 内层验证

验证集必须按受试者划分，不能从训练窗口中随机抽取。

推荐：

```text
Outer test subject:  S_i
Inner val subject:   从剩余受试者中轮换或固定选择 S_j
Inner train subjects: 其余受试者
```

若数据规模有限，可在外层训练受试者内部做 GroupKFold，以受试者 ID 为 group。

### 9.3 防止时间窗口泄漏

禁止：

```text
同一受试者相邻或重叠窗口同时进入 train 和 validation
```

特别是 6 秒长窗口会产生更强的邻近相关性，随机窗口划分会明显高估模型性能。

### 9.4 特征预处理

每个外层折中：

1. 仅使用内层训练受试者拟合缺失值处理参数；
2. 仅使用内层训练受试者拟合特征变换；
3. 仅使用内层训练受试者拟合标准化器；
4. 将同一个变换应用到验证和测试受试者。

推荐初版：

```text
有限值检查
   ->
训练折中位数补缺
   ->
StandardScaler
```

如果 RMS、能量、峰峰值或 jerk RMS 呈强右偏，可将 `log1p + StandardScaler` 作为独立消融，而不是默认与 TabM 同时引入。

---

## 10. 特征表结构

建议每行对应一个判定窗口：

```text
subject_id
session_id
task_id
window_start
window_end
label
short_<sensor>_<axis>_mean
short_<sensor>_<axis>_std
short_<sensor>_<axis>_rms
short_<sensor>_<axis>_p2p
short_<sensor>_<axis>_jerk_rms
long_<sensor>_<axis>_freezing_index
long_<sensor>_<axis>_dominant_frequency
long_<sensor>_<axis>_spectral_entropy
long_<sensor>_<axis>_energy_0p5_3hz
long_<sensor>_<axis>_energy_3_8hz
```

特征列顺序必须由单一配置文件控制并保存至模型检查点：

```yaml
feature_schema_version: v1
short_window_seconds: 2.0
long_window_seconds: 6.0
feature_names:
  - ...
```

模型保存时同时保存：

- 特征列名及顺序；
- scaler；
- 类别映射；
- 训练受试者列表；
- 验证受试者列表；
- 阈值；
- 模型超参数；
- 随机种子；
- 数据版本。

---

## 11. 配置文件建议

```yaml
experiment:
  name: fog_nonfog_handcrafted_tabm
  seed: 2026
  task: binary_classification
  positive_class: FoG
  evaluation: LOSO

features:
  short_window_seconds: 2.0
  long_window_seconds: 6.0
  alignment: same_decision_time
  short:
    - mean
    - std
    - rms
    - peak_to_peak
    - jerk_rms
  long:
    - freezing_index
    - dominant_frequency
    - spectral_entropy
    - energy_0p5_3hz
    - energy_3_8hz
  preprocessing:
    imputer: train_median
    scaler: standard
    log_transform: false

model:
  name: TabM
  package: tabm
  arch_type: tabm
  input_dim: auto
  d_out: 2
  k: 32
  n_blocks: 2
  d_block: 256
  dropout: 0.10
  activation: relu
  normalization: null
  tabm_init: true
  first_scaling_init: normal
  independent_heads: true

training:
  optimizer: AdamW
  learning_rate: 0.002
  weight_decay: 0.0003
  batch_size: 256
  max_epochs: 300
  early_stopping_patience: 30
  early_stopping_metric: val_roc_auc
  loss: cross_entropy
  class_weight: same_as_mlp_baseline
  mixed_precision: true
  gradient_clip_norm: null

inference:
  ensemble_aggregation: mean_probability
  default_threshold: 0.5
  threshold_tuning:
    enabled: true
    data: inner_validation_subjects
    objective: balanced_accuracy

report:
  metrics:
    - accuracy
    - balanced_accuracy
    - macro_f1
    - fog_precision
    - fog_recall
    - fog_f1
    - specificity
    - roc_auc
    - pr_auc
    - brier_score
  save_member_predictions: true
  save_pooled_confusion_matrix: true
  save_fold_metrics: true
```

---

## 12. 训练循环骨架

```python
def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    class_weight=None,
):
    model.train()
    running_loss = 0.0
    n_samples = 0

    for batch in loader:
        x = batch["features"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)  # [B, K, 2]
        loss = tabm_cross_entropy(
            logits=logits,
            target=y,
            class_weight=class_weight,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {loss.item()}")

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.shape[0]
        n_samples += x.shape[0]

    return running_loss / max(n_samples, 1)
```

验证阶段必须使用成员概率平均后的集成结果计算指标和早停：

```python
member_probs = logits.softmax(dim=-1)  # [B, K, 2]
ensemble_probs = member_probs.mean(1)  # [B, 2]
```

不要根据某一个成员的验证指标早停。

---

## 13. 实验对照设计

### 13.1 主对照

为确保结果可归因于骨干替换：

| 实验 | 特征 | 预处理 | 损失/采样 | 分类器 |
|---|---|---|---|---|
| E0 | 相同 | 相同 | 相同 | 当前 MLP |
| E1 | 相同 | 相同 | 相同 | 正式 TabM，\(K=32\) |

主结论首先回答：

> 在完全相同的人工特征、LOSO 划分、损失函数和阈值规则下，正式 TabM 是否优于普通 MLP？

### 13.2 关键消融

| 编号 | 设置 | 目的 |
|---|---|---|
| A1 | TabM \(K=16\) | 检验较低成员数的效率与性能 |
| A2 | TabM \(K=32\) | 正式主配置 |
| A3 | TabM-mini \(K=32\) | 检验逐层适配器的贡献 |
| A4 | 仅 2 秒短窗特征 | 检验局部运动统计信息 |
| A5 | 仅 6 秒长窗特征 | 检验冻结频谱上下文 |
| A6 | 2 秒 + 6 秒 | 完整多时间尺度输入 |
| A7 | 移除 Freezing Index | 检验其与两段频带能量的冗余 |
| A8 | TabM + 数值特征嵌入 | 后续性能增强，不属于首轮公平比较 |

不建议第一轮同时搜索过多结构。优先完成：

```text
MLP
vs
TabM K=32
vs
TabM-mini K=32
```

---

## 14. 评估结果输出

### 14.1 每个 LOSO 折

必须保存：

```text
subject_id
n_normal
n_fog
accuracy
balanced_accuracy
macro_f1
fog_precision
fog_recall
fog_f1
specificity
roc_auc
pr_auc
brier_score
threshold
confusion_matrix
```

### 14.2 总体结果

同时报告：

1. 各折指标的均值、标准差和中位数；
2. 全部外层测试预测汇总后的 pooled confusion matrix；
3. pooled ROC-AUC 与 PR-AUC；
4. 每位受试者 FoG Recall 和误报率；
5. 固定 \(0.5\) 阈值与验证集阈值的结果；
6. 至少 3 个随机种子的结果。

### 14.3 成员分歧

可保存：

\[
u(x)
=
\operatorname{Std}_{k}
\left(
P_k(y=\mathrm{FoG}\mid x)
\right)
\]

它可以作为 TabM 成员分歧指标，用于分析：

- 哪些窗口预测不稳定；
- 哪些受试者存在明显域偏移；
- 是否能识别潜在错误高风险样本。

注意：成员标准差不能直接等同于经过校准的概率不确定性，需要额外验证。

---

## 15. 工程目录

```text
fog_tabm/
├── configs/
│   └── tabm_fog_binary.yaml
├── data/
│   ├── feature_tables/
│   └── split_manifests/
├── src/
│   ├── datasets/
│   │   ├── feature_dataset.py
│   │   └── loso_split.py
│   ├── preprocessing/
│   │   ├── feature_schema.py
│   │   └── fold_preprocessor.py
│   ├── models/
│   │   └── fog_tabm.py
│   ├── training/
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   └── early_stopping.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── threshold.py
│   │   └── uncertainty.py
│   └── utils/
│       ├── seed.py
│       └── checkpoint.py
├── scripts/
│   ├── train_loso.py
│   ├── evaluate_loso.py
│   └── run_ablation.py
├── outputs/
│   ├── checkpoints/
│   ├── fold_predictions/
│   ├── metrics/
│   └── figures/
└── tests/
    ├── test_model_shape.py
    ├── test_member_loss.py
    ├── test_probability_aggregation.py
    └── test_no_subject_leakage.py
```

---

## 16. 正确性单元测试

### 16.1 输出维度

```python
x = torch.randn(8, input_dim)
logits = model(x)
assert logits.shape == (8, 32, 2)
```

### 16.2 成员必须具有差异

模型初始化后：

```python
assert not torch.allclose(logits[:, 0], logits[:, 1])
```

若所有成员输出完全一致，应检查：

- 第一层 \(R_1\) 是否随机；
- 成员差异是否在第一次共享线性变换前产生；
- 是否误用了普通 `nn.Linear`；
- 是否错误共享了预测头。

### 16.3 损失顺序

以下是正确目标展开顺序：

```python
target_members = y[:, None].expand(B, K).reshape(B * K)
```

### 16.4 推理聚合

```python
expected = logits.softmax(-1).mean(1)
actual = inference_function(logits)
assert torch.allclose(expected, actual)
```

### 16.5 防泄漏

每一折必须满足：

```python
assert set(train_subjects).isdisjoint(val_subjects)
assert set(train_subjects).isdisjoint(test_subjects)
assert set(val_subjects).isdisjoint(test_subjects)
```

---

## 17. 首轮实验执行顺序

### 阶段 1：锁定现有基线

冻结以下内容：

- 人工特征定义；
- 窗口位置；
- 标签定义；
- LOSO 划分；
- 训练/验证受试者划分；
- 标准化方法；
- 类别权重；
- batch size；
- 评价指标；
- 阈值选择规则。

### 阶段 2：仅替换分类骨干

```text
当前 MLP
->
正式 TabM
```

主配置：

```yaml
k: 32
n_blocks: 2
d_block: 256
dropout: 0.1
lr: 0.002
weight_decay: 0.0003
```

### 阶段 3：检查 TabM 是否真的产生集成增益

比较：

- 最佳单个 TabM 成员；
- 成员性能均值；
- 32 个成员概率平均；
- 当前 MLP。

预期应重点观察：

```text
集成预测是否明显优于平均单成员预测
```

### 阶段 4：再做小范围超参数搜索

建议范围：

```yaml
k:
  - 16
  - 32

n_blocks:
  - 2
  - 3

d_block:
  - 128
  - 256
  - 512

dropout:
  - 0.0
  - 0.1
  - 0.2

learning_rate:
  - 0.0003
  - 0.001
  - 0.002

weight_decay:
  - 0.0
  - 0.0003
  - 0.001
```

若比较不同 \(K\)，应分别搜索相匹配的网络宽度和深度，不建议将 \(K\) 与所有超参数放在同一次无约束搜索中。

---

## 18. 最终验收清单

- [ ] 输入是人工特征表，而不是直接展平原始 IMU 波形；
- [ ] 2 秒与 6 秒特征在同一判定时刻对齐；
- [ ] `EnsembleView` 位于第一次特征线性混合之前；
- [ ] 每个骨干线性层使用共享 \(W\) 和成员专属 \(R/S/B\)；
- [ ] 第一层输入适配器随机初始化；
- [ ] 后续乘性适配器按 TabM 风格初始化为 1；
- [ ] 使用 \(K\) 个互不共享的预测头；
- [ ] 模型输出为 `[B, K, 2]`；
- [ ] 训练优化“各成员损失的平均”；
- [ ] 训练时不先平均预测；
- [ ] 推理时平均 softmax 概率；
- [ ] 早停依据集成验证性能；
- [ ] 不在成员维度使用 attention 或 pooling；
- [ ] LOSO 与内层验证均按受试者划分；
- [ ] scaler、类别权重和阈值不接触测试受试者；
- [ ] MLP 与 TabM 主对照只改变分类骨干；
- [ ] 保存每个成员的预测，支持分歧与错误分析。

---

## 19. 推荐模型命名

论文和代码中建议统一使用：

```text
Handcrafted Multi-Scale Feature TabM
```

缩写：

```text
MSF-TabM
```

完整描述：

> A formal TabM classifier with BatchEnsemble-style shared MLP weights and member-specific adapters was applied to multi-scale handcrafted IMU features extracted from 2-s and 6-s windows.

避免使用：

```text
TabM-like MLP
Multi-head MLP
TabM-inspired ensemble
```

除非实现确实缺少正式 TabM 的某些关键组件。

---

## 20. 参考

1. Y. Gorishniy, A. Kotelnikov, and A. Babenko, “TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling,” *International Conference on Learning Representations (ICLR)*, 2025.
2. Yandex Research, `yandex-research/tabm`, official PyTorch implementation and `tabm` package.
