# GRU-NBM残差G1/G2/G3：TCN 50/6复现实验

本实验仅重新训练TCN分类器。三个配对种子对应的GRU-NBM、角色4
RobustScaler以及角色5的偏移量和尺度参数全部从既有冻结产物读取，不重新训练。

## 冻结与训练范围

- NBM：冻结的GRU-v1，原训练配置为最大300 epoch、patience 20。
- TCN：相同`RepresentationTCNM`，输入`[B,27,128]`。
- TCN最大epoch：50。
- TCN patience：6。
- TCN种子与NBM种子配对：0、52、161。
- 角色6/7训练TCN；角色2/3早停、模型选择及阈值选择。
- 27个TCN和阈值全部冻结后，角色0/1才用于最终测试。

## 7卡启动

```bash
cd /home/chb/projects/fog-merged
conda activate fog

bash scripts/run_daphnet_gru_residual_g123_tcn50pat6_7gpu.sh
```

等价的完整命令：

```bash
python -u scripts/launch_daphnet_gru_residual_g123_7gpu.py \
  --output-root outputs/daphnet_gru_nbm300_residual_G1_G2_G3_tcn_ep50pat6_seedset_0_52_161 \
  --gpu-ids 0,1,2,3,4,5,6 \
  --tcn-seeds 0,52,161 \
  --tcn-max-epochs 50 \
  --tcn-patience 6 \
  --phase full
```

先检查任务而不训练：

```bash
python -u scripts/launch_daphnet_gru_residual_g123_7gpu.py \
  --output-root outputs/daphnet_gru_nbm300_residual_G1_G2_G3_tcn_ep50pat6_seedset_0_52_161 \
  --gpu-ids 0,1,2,3,4,5,6 \
  --tcn-seeds 0,52,161 \
  --tcn-max-epochs 50 \
  --tcn-patience 6 \
  --phase full \
  --dry-run
```

预期任务数为27个训练任务和27个屏障后测试任务。新目录与10/2实验完全
分离；不要加入`--overwrite`，也不要指向旧的`ep10pat2`输出目录。

如训练中断，可在同一输出目录重新执行同一命令，已完成任务会自动跳过。
