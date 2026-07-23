# CNBR-FoG：Daphnet 腰部三轴 LOSO 可行版本

该实验实现“条件正常行为预测 → 不确定性标准化残差 → FOG 分类”两阶段框架。

## 主协议

- 数据：`processed_trunk`，64 Hz，腰部三轴加速度；不是含陀螺仪的完整 IMU。
- 外层评估：10 折 LOSO；测试受试者从不参与归一化、训练、早停或阈值选择。
- 内层验证：从其余受试者中按固定循环顺序选择下一位同时含两类窗口的受试者；剩余 8 人训练。
- 输入 context：2.0 秒（128 点）。
- normal target：随后 0.5 秒（32 点）。
- 步长：0.25 秒（16 点）。
- 窗口标签：target 中 FOG 占比不少于 50% 时为 FOG。
- 正常预测器：仅使用 context、target 及其前后 0.5 秒缓冲均无 FOG 的训练窗口。
- 数据质量：固定排除持续至少 1 秒的全三轴零平线；稳健缩放参数只由当前训练受试者的有效 non-FOG 样本估计。
- 决策阈值：只在验证受试者上最大化 balanced accuracy；并列时依次选择较高 F1 和较高阈值。
- S04、S10 无 FOG：保留在 LOSO 中评价 specificity 和误报；其 sensitivity、F1、AUROC、AUPRC 记为未定义。

## 模型

正常预测器学习：

\[
(\mu,\log\sigma^2)=G_\theta(X_{t-2s:t}),\qquad
X_{t:t+0.5s}\mid \text{non-FOG}
\]

使用异方差高斯负对数似然，仅以 clean non-FOG 训练。分类输入为：

\[
Z=\operatorname{clip}\left(
\frac{X_{actual}-\mu}{\sigma},-12,12
\right)
\]

`residual` 与 `raw` 使用完全相同的 TCN 分类器；raw 基线仅把输入替换为稳健标准化后的实际 target。

## 运行

在项目根目录使用带 CUDA PyTorch 的环境：

```powershell
conda run -n pd_fog python scripts\run_cnbr_fog_loso.py `
  --data-dir "E:\fog-merged\dataset\1.Daphnet Freezing of Gait Dataset\processed_trunk" `
  --output-dir outputs\cnbr_fog_daphnet_trunk_loso `
  --folds all
```

快速验证单折：

```powershell
conda run -n pd_fog python scripts\run_cnbr_fog_loso.py `
  --folds S01 --normal-epochs 1 --classifier-epochs 1 `
  --max-normal-windows 512 --max-classifier-windows 1024 `
  --output-dir outputs\cnbr_fog_smoke --no-resume
```

## 输出

- `config.json`：完整数据、窗口和训练配置。
- `loso_Sxx/normal_predictor_best.pt`：每折 normal-only 预测器。
- `loso_Sxx/fold_config.json`：训练/验证/测试受试者、缩放统计及残差诊断。
- `loso_Sxx/{residual,raw}/classifier_best.pt`：分类器。
- `loso_Sxx/{residual,raw}/predictions.csv`：每个测试窗口的最终概率与类别。
- `fold_summary.csv`：逐受试者核心指标。
- `aggregate_metrics.json`：subject-macro 与 pooled 指标。

窗口级核心指标为 AUPRC、AUROC、sensitivity、specificity、precision、F1、MCC 和 balanced accuracy；事件级还输出事件检出率、误报事件/小时和检测延迟。

## 当前可行版本的边界

分类器训练残差由同一外层训练集上拟合的正常预测器生成，因此它是工程可行基线，而不是最严格的 subject-level cross-fitting 版本。若用于正式论文，下一步应在每个外层训练折内部实施 subject-level 交叉拟合，为残差分类器生成 out-of-subject 训练残差。
