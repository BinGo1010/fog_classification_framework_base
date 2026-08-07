# Daphnet残差校准A–D实验：优化器与损失函数配置

## 1. 适用实验

本文档对应以下四组残差校准实验：

- A：Location–Scale Calibration
- B：Location–Scale Calibration + Window Centering
- C：Scale Calibration + Window Centering
- D：Window Centering Only

四组实验最终均构造：

\[
F=[r,|r|,\Delta r]\in\mathbb{R}^{B\times27\times128}
\]

每组均重新训练TCN-M。RobustScaler、NBM checkpoint、残差偏移参数 \(b\) 和尺度参数 \(\sigma\) 均来自此前已经完成的NBM训练，在A–D实验中保持冻结。

---

## 2. NBM原始训练配置

### 2.1 参数更新范围

NBM原始训练阶段仅更新Conv-TCN Autoencoder参数。

- 训练数据：角色4的clean non-FoG窗口
- 验证数据：角色5的clean non-FoG窗口
- RobustScaler：仅使用角色4原始采样点拟合
- 角色5不参与梯度更新，只用于NBM早停和最佳checkpoint选择
- A–D实验不重新训练NBM

### 2.2 NBM优化器

```python
torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)
```

| 参数 | 设置 |
|---|---:|
| 优化器 | AdamW |
| 初始学习率 | \(1\times10^{-3}\) |
| Weight decay | \(1\times10^{-4}\) |
| Batch size | 128 |
| 最大epoch | 50 |
| Early-stopping patience | 8 |
| 梯度范数上限 | 1.0 |
| TCN残差块dropout | 0.10 |

### 2.3 NBM学习率调度

```python
torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=1e-5,
)
```

- 监控指标：角色5验证SmoothL1损失
- 验证损失连续3个epoch未改善时，学习率乘以0.5
- 最低学习率：\(1\times10^{-5}\)

### 2.4 NBM损失函数

```python
criterion = torch.nn.SmoothL1Loss(beta=1.0)
```

训练输入为执行轻度Mask后的信号 \(\widetilde X\)，监督目标始终为未Mask的clean信号 \(X\)：

\[
\widehat X_N=f_\theta(\widetilde X)
\]

\[
\mathcal L_{\mathrm{NBM}}
=\frac{1}{BCT}
\sum_{b,c,t}
\operatorname{SmoothL1}
\left(\widehat X_{N,b,c,t}-X_{b,c,t}\right)
\]

当 \(\beta=1\) 时，令 \(d=\widehat X_N-X\)，单点损失为：

\[
\ell(d)=
\begin{cases}
\frac{1}{2}d^2,& |d|<1\\
|d|-\frac{1}{2},& |d|\geq1
\end{cases}
\]

SmoothL1在小误差区域接近MSE，在大误差区域接近MAE，可以降低少量异常采样点对重构模型的影响。

### 2.5 NBM轻度Mask

| 参数 | 设置 |
|---|---:|
| 被Mask窗口概率 | 0.20 |
| 连续Mask长度 | 4–8个采样点 |
| Mask通道 | 同一时间位置的全部9个通道 |
| Mask值 | 0 |
| 验证阶段Mask | 不使用 |

Mask只改变训练输入，不改变重构目标。

### 2.6 NBM模型选择

- 每个epoch计算角色5完整、未Mask信号的SmoothL1损失。
- 验证损失严格下降时保存checkpoint。
- 连续8个epoch没有改善时提前停止。
- 训练结束后恢复角色5验证SmoothL1最低的checkpoint。

恢复最佳NBM后，才在角色5上计算 \(b\) 和 \(\sigma\)。这两个参数不属于NBM损失函数，也不通过反向传播学习。

---

## 3. A–D实验的TCN-M训练配置

### 3.1 参数更新范围

A–D实验中只更新对应组的TCN-M参数：

- Scaler：冻结
- NBM：冻结并使用`eval()`模式
- \(b\)：冻结
- \(\sigma\)：冻结
- TCN-M：重新初始化并训练

每一折、每一个TCN随机种子下，A–D使用完全相同的TCN初始权重和训练batch顺序。

### 3.2 TCN优化器

```python
torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)
```

| 参数 | 设置 |
|---|---:|
| 优化器 | AdamW |
| 学习率 | \(1\times10^{-3}\) |
| Weight decay | \(1\times10^{-4}\) |
| Batch size | 128 |
| 最大epoch | 30 |
| Early-stopping patience | 6 |
| 梯度范数上限 | 1.0 |
| 分类头dropout | 0.30 |
| 学习率调度器 | 无 |

### 3.3 类别权重

TCN训练数据仅来自：

- 角色6：non-FoG，标签0
- 角色7：FoG，标签1

每一折的正类权重只根据角色6/7计算：

\[
w_{\mathrm{pos}}
=\frac{N_{\mathrm{role6}}}{N_{\mathrm{role7}}}
=\frac{N_{\mathrm{nonfog}}}{N_{\mathrm{fog}}}
\]

角色2/3和角色0/1不参与`pos_weight`计算。

### 3.4 TCN损失函数

```python
criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(N_role6 / N_role7)
)
```

模型输出未经Sigmoid的logit \(z_i\)。定义：

\[
p_i=\sigma(z_i)=\frac{1}{1+e^{-z_i}}
\]

加权二元交叉熵为：

\[
\mathcal L_{\mathrm{TCN}}
=-\frac{1}{N}\sum_{i=1}^{N}
\left[
w_{\mathrm{pos}}y_i\log(p_i)
+(1-y_i)\log(1-p_i)
\right]
\]

作用是提高FoG正类样本在训练损失中的权重，缓解non-FoG窗口数量多于FoG窗口造成的类别不平衡。

### 3.5 梯度更新

每个训练batch执行：

```python
optimizer.zero_grad(set_to_none=True)
logits = model(batch_x)
loss = criterion(logits, batch_y)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

如果梯度范数出现NaN或Inf，程序立即终止该任务，不继续产生测试结果。

### 3.6 TCN模型选择与早停

- 训练损失：角色6/7的加权BCE。
- 验证数据：角色2/3。
- 每个epoch计算角色2/3的PR-AUC。
- 角色2/3 PR-AUC严格提高时保存TCN checkpoint。
- 连续6个epoch没有提高时提前停止。
- 训练结束后恢复角色2/3 PR-AUC最高的checkpoint。

PR-AUC用于选择模型checkpoint，但不直接参与梯度反向传播。

---

## 4. 分类阈值选择

恢复最佳TCN权重后，在角色2/3上搜索分类阈值：

```text
候选阈值：0.05, 0.06, ..., 0.95
主要目标：Balanced Accuracy最大
第一平局规则：FoG F1更高
第二平局规则：选择更高阈值
```

阈值选择不属于损失函数，也不更新网络参数。

所有折、A–D组和3个随机种子的TCN checkpoint与阈值全部冻结后，程序才建立`TRAINING_BARRIER.json`并允许访问角色0/1最终测试集。

---

## 5. 配置汇总

| 模块 | 优化器 | 损失 | 学习率 | Weight decay | 最大epoch | Patience | 模型选择指标 |
|---|---|---|---:|---:|---:|---:|---|
| NBM原训练 | AdamW | SmoothL1，\(\beta=1\) | 1e-3 | 1e-4 | 50 | 8 | 角色5验证SmoothL1最低 |
| A组TCN | AdamW | 加权BCEWithLogits | 1e-3 | 1e-4 | 30 | 6 | 角色2/3 PR-AUC最高 |
| B组TCN | AdamW | 加权BCEWithLogits | 1e-3 | 1e-4 | 30 | 6 | 角色2/3 PR-AUC最高 |
| C组TCN | AdamW | 加权BCEWithLogits | 1e-3 | 1e-4 | 30 | 6 | 角色2/3 PR-AUC最高 |
| D组TCN | AdamW | 加权BCEWithLogits | 1e-3 | 1e-4 | 30 | 6 | 角色2/3 PR-AUC最高 |

## 6. 对应代码

- NBM原始训练配置：`scripts/run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py`
- A–D的TCN优化器与损失：`scripts/run_daphnet_residual_calibration_abcd.py`
- 7卡任务调度：`scripts/launch_daphnet_residual_calibration_abcd_7gpu.py`
