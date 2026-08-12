# Persistence vs Linear-AR(8) NBM + TCN实验

## 实验目的

只比较两种正常行为模型，不包含Raw和GRU对照：

1. `PERSISTENCE_C`：无训练参数的持续性基线；
2. `LINEAR_AR_C`：多变量、跨9轴联合的因果Linear-AR(8)。

两种模型均生成方案C残差，随后输入完全相同的27通道TCN分类器。

## 数据和角色隔离

- 数据：`processed_NBM`，64 Hz，9通道；
- 窗口：128点（2秒），步长64点（1秒）；
- 三折，种子：`0, 52, 161, 5216, 52161`；
- 角色4：拟合RobustScaler；Linear-AR也只在角色4训练；
- 角色5：clean验证、Linear-AR早停、两种NBM分别计算残差校准量；
- 角色6/7：训练TCN；
- 角色2/3：TCN早停、模型选择和分类阈值选择；
- 角色0/1：全部30个TCN及阈值封存后才运行测试。

RobustScaler只用角色4窗口覆盖的去重原始采样点拟合。缩放后，对每个2秒窗口的每个轴沿时间维中心化。

## Persistence-NBM

对中心化后的输入窗口：

\[
\hat X_0=X_0,\qquad \hat X_t=X_{t-1},\quad t=1,\ldots,127.
\]

- 参数量：0；
- 无优化器、无训练epoch；
- 不向测试输入人为加入Mask或噪声；
- 角色5只用于计算该方法自身的残差校准量。

## Linear-AR(8)-NBM

每个时间点联合预测9个轴：

\[
\hat{\boldsymbol{x}}_t=\boldsymbol b+
\sum_{k=1}^{\min(8,t)} W_k\boldsymbol{x}_{t-k},
\quad W_k\in\mathbb R^{9\times9}.
\]

- 严格因果：不使用当前点或未来点；
- 参数量：`8×9×9+9=657`；
- 第0点复制输入；窗口开头只使用实际存在的历史点，不补未来信息；
- 训练输入为轻度扰动窗口，监督目标始终为clean窗口；
- 角色4增强：40% clean、40% Gaussian（std=0.04）、20% 连续全轴Mask 4–8点（闭区间）；
- 损失：SmoothL1，beta=1.0；
- 优化器：AdamW，lr=1e-3，weight_decay=1e-4；
- ReduceLROnPlateau：factor=0.5，patience=3，min_lr=1e-5；
- batch=128，最大epoch=300，早停patience=20，梯度裁剪1.0；
- 角色5不增强，以最低clean SmoothL1恢复最佳权重。

## 校准和方案C残差

恢复各自冻结模型后，在角色5 clean窗口计算：

\[
b_c=\operatorname{median}(e_c),\qquad
\sigma_c=\max\left(1.4826\operatorname{median}|e_c-b_c|,0.05\right).
\]

分类器实际使用：

\[
e=X-\hat X,\quad q=\operatorname{clip}\left(\frac{e}{\sigma+10^{-6}},-12,12\right),
\quad r=q-\operatorname{mean}_t(q),
\]

\[
F=[r,|r|,\Delta r]\in\mathbb R^{27\times128}.
\]

方案C不减去`b`；`b`仅用于MAD尺度估计。

## TCN分类器

- 两个方法输入均为`[B,27,128]`；
- 相同fold/seed使用完全相同的TCN初始权重；
- 4个残差块，通道`27→32→64→64→128`，dilation为`1,2,4,8`；
- 损失：带`pos_weight=N_role6/N_role7`的BCEWithLogitsLoss；
- AdamW，lr=1e-3，weight_decay=1e-4；
- batch=128，最大epoch=5，早停patience=2，梯度裁剪1.0；
- 角色2/3验证PR-AUC最高的checkpoint被恢复；
- 分类阈值在角色2/3上从0.05到0.95、步长0.01搜索：先最大Balanced Accuracy，平局时先FoG F1，再取较高阈值。

## 服务器运行

```bash
cd /document/home_mirror/chb/fog_classification_framework_base
conda activate fogbase
bash scripts/run_daphnet_persistence_vs_linear_ar_7gpu.sh
```

直接调用启动器：

```bash
python scripts/launch_daphnet_persistence_vs_linear_ar_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --phase full
```

运行网格：30个NBM源任务（其中15个Persistence任务只做确定性重构与校准）、30个TCN训练任务、一个全局测试屏障、30个测试任务。

默认输出目录：

```text
outputs/daphnet_persistence_vs_linear_ar8_C_tcn_ep5pat2_seedset_0_52_161_5216_52161
```

可用`--phase nbm/train/evaluate/aggregate`分阶段恢复。若更改架构、数据或关键程序，使用新的输出目录，不要混用旧实验产物。

## 结果文件

- `summary.json`：两种方法的总体指标及配对差值；
- `method_summary_5seed_mean_std.csv`：5种子的均值±总体标准差；
- `paired_delta_LINEAR_AR_C_minus_PERSISTENCE_C_summary.csv`：主配对差值；
- `subject_metrics_5seed_mean_std.csv`：各被试指标；
- `run_metrics_30.csv`：全部折/方法/种子运行；
- `TRAINING_BARRIER.json`：测试前冻结的模型、阈值、校准和数据指纹。
