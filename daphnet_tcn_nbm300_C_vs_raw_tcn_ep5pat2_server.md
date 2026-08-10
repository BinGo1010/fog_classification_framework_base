# TCN-NBM方案C与Raw-TCN对照：五种子、TCN 5/2

## 比较目标

该实验在完全相同的`processed_NBM`划分上比较：

- `FULL_C`（TCN-NBM）：TCN-NBM重建残差经过方案C后送入TCN分类器。
- `RAW`（Raw-TCN）：不使用NBM重建，预处理后的原始9通道窗口直接送入相同TCN分类器。

两组共享受试者内三折、永久测试集、训练/验证角色、TCN结构、优化器、早停规则、阈值搜索规则及配对随机种子。唯一实验变量是分类器输入表示。

## 固定配置

- 数据集：`dataset/1.Daphnet Freezing of Gait Dataset/processed_NBM`
- 采样率：64 Hz
- 窗口：2秒，128点
- 步长：1秒，64点
- 配对种子：`0,52,161,5216,52161`，不增加fold偏移
- TCN-NBM：最大300 epoch，patience 20
- TCN分类器：最大5 epoch，patience 2
- 默认GPU：`0,1,2,3,4,5,6`

TCN-NBM输入/瓶颈/输出形状为：

```text
[B,9,128] -> Conv-TCN encoder -> [B,16,32]
          -> Conv-TCN decoder -> [B,9,128]
```

方案C分类输入：

```text
r = center_t(clip((X-Xhat)/(sigma+1e-6), -12, 12))
F = [r, abs(r), delta(r)]
TCN input = [B,27,128]
```

角色5会记录残差偏置`b`和尺度`sigma`，但本实验沿用既定方案C定义：只用`sigma`归一化，不从残差中减去`b`。

Raw-TCN分类输入为角色4 RobustScaler变换并逐窗口逐轴中心化后的`[B,9,128]`。Raw分支不实例化或前向运行NBM，但使用同一角色4 scaler，避免预处理差异成为混杂因素。

## 数据角色

- 角色4：拟合RobustScaler并训练TCN-NBM
- 角色5：TCN-NBM内部早停；恢复最佳权重后估计残差偏置和尺度（方案C只使用尺度）
- 角色6/7：两个分类分支各自训练TCN
- 角色2/3：分类器早停、模型选择和分类阈值选择
- 角色0/1：所有30个分类器均冻结并生成全局屏障后，才允许最终测试

三折均执行上述流程。因此共有15个TCN-NBM任务、30个分类器训练任务和30个屏障后的最终测试任务。

## 服务器运行

```bash
cd /document/home_mirror/chb/fog_classification_framework_base
conda activate fogbase
bash scripts/run_daphnet_tcn_nbm300_c_vs_raw_tcn_ep5pat2_7gpu.sh
```

默认输出目录：

```text
outputs/daphnet_tcn_nbm300_C_vs_raw_tcn_ep5pat2_seedset_0_52_161_5216_52161
```

运行前只检查任务计划：

```bash
python scripts/launch_daphnet_tcn_nbm300_c_vs_raw_tcn_ep5pat2_7gpu.py --dry-run
```

中断后可以使用`--phase nbm`、`--phase train`、`--phase evaluate`或`--phase aggregate`分阶段恢复。已有完成标记的同构任务默认跳过，除非显式添加`--overwrite`。

启动器会拒绝在含有GRU-NBM计划或模型的同名输出目录中运行。若该目录已被旧实验占用，请先备份并改用新的`--output-root`，不要混写两种NBM结果。
