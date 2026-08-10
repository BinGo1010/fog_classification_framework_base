# GRU-NBM方案C与Raw对照：TCN 5/2服务器实验

## 固定配置

- 数据集：`dataset/1.Daphnet Freezing of Gait Dataset/processed_NBM`
- 采样率：64 Hz
- 窗口：2秒，128点
- 步长：1秒，64点
- 方法：`FULL_C`与`RAW`
- 配对种子：`0,52,161,5216,52161`，不增加fold偏移
- GRU-NBM：最大300 epoch，patience 20
- TCN：最大5 epoch，patience 2
- GPU：`0,1,2,3,4,5,6`

GRU-NBM内部架构保持不变：

```text
[B,128,9]
  → GRU(9→64)
  → Linear(64→16)
  → Linear(16→64)
  → 128步零输入GRU(9→64)
  → Linear(64→9)
  → [B,128,9]
```

方案C分类输入：

```text
r = center_t(clip((X-Xhat)/(sigma+1e-6), -12, 12))
F = [r, abs(r), delta(r)]
TCN input = [B,27,128]
```

Raw分类输入为角色4 RobustScaler变换并逐窗口逐轴中心化后的 `[B,9,128]`。两组使用相同TCN架构、角色范围、batch size、学习率、阈值搜索规则及配对初始化。

## 严格数据角色

- 角色4：拟合RobustScaler、训练GRU-NBM
- 角色5：NBM早停和恢复最佳权重后的残差校准
- 角色6/7：TCN训练
- 角色2/3：TCN早停、模型选择和阈值选择
- 角色0/1：全局训练屏障建立后最终测试

## 服务器运行

```bash
cd /document/home_mirror/chb/fog_classification_framework_base
conda activate fogbase
bash scripts/run_daphnet_gru_nbm300_c_vs_raw_ep5pat2_7gpu.sh
```

默认输出目录：

```text
outputs/daphnet_tcn_nbm300_C_vs_raw_tcn_ep5pat2_seedset_0_52_161_5216_52161
```

只检查任务计划：

```bash
python scripts/launch_daphnet_gru_nbm300_c_vs_raw_ep5pat2_7gpu.py --dry-run
```

完整运行包含15个GRU-NBM训练任务、30个TCN训练任务、全局测试屏障、30个最终测试任务及自动汇总。

中断后可用 `--phase nbm`、`--phase train`、`--phase evaluate` 或 `--phase aggregate` 分阶段恢复。已有完成标记的任务默认跳过，除非显式添加 `--overwrite`。
