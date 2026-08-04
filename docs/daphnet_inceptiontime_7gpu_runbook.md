# Daphnet InceptionTime：7×2080 Ti 并行与断点续训

## 运行前提

- 保留上一轮输出中的 `splits/`；InceptionTime 复用冻结的外层折、Scaler、OOF NBM 重建和外层测试重建。
- 服务器 Python 环境需安装项目依赖，并能由 `torch.cuda.is_available()` 识别 7 张 GPU。
- 建议为服务器使用全新的输出目录，避免混入本机 CPU 烟雾实验的部分结果。

## 一条命令启动 7 卡实验

在仓库根目录运行：

```bash
python scripts/run_daphnet_full_subject_nbm_residual_inceptiontime.py \
  --launch-parallel \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6 \
  --threads 2 \
  --source-root outputs/daphnet_full_subject_nbm_residual_binary_v1/full_subject_binary_experiment \
  --output-root outputs/daphnet_full_subject_nbm_residual_inceptiontime_v1/full_subject_binary_experiment
```

父进程会启动 7 个单卡 worker。30 个外层折以互斥分片分配，每个 worker 顺序完成本分片中的 B0–B3 × 3 seeds。全部 worker 成功后，父进程自动聚合表格、统计检验、图片和最终报告。

`--threads 2` 表示每个 GPU worker 使用 2 个 CPU intra-op 线程。若两颗 CPU 的总可用核心数较少，可改为 `--threads 1`；若核心充足且 GPU 利用率长期偏低，可逐步提高到 4。

## 中断与恢复

使用 `Ctrl+C` 中断父进程。父进程会终止 7 个 worker，但保留：

- 已完成运行的 `run_metrics.json`、`test_predictions.csv` 和最佳模型；
- 未完成运行的 `inceptiontime_resume.pt` epoch 检查点；
- 每个 worker 的独立日志与 `parallel_status.json`。

恢复时重新运行完全相同的命令。脚本会：

1. 跳过同时具有指标和外层预测的完整运行；
2. 从未完成运行的最近完整 epoch 恢复模型、优化器、最佳权重、early-stopping 和 shuffle 状态；
3. 自动重试异常退出的 worker，默认最多 2 次。

不希望加载未完成 epoch 检查点时可加 `--no-resume`；该选项仍会跳过已经完整落盘的运行。

## 查看进度

```bash
python scripts/run_daphnet_full_subject_nbm_residual_inceptiontime.py \
  --status-only \
  --output-root outputs/daphnet_full_subject_nbm_residual_inceptiontime_v1/full_subject_binary_experiment
```

完整实验应显示 `completed_runs: 360`、每个方法 90 次、`ready_to_finalize: true`。

worker 日志位于：

```text
<output-root>/logs/parallel_workers/
```

## 仅重新聚合

若 360 次运行已经齐全但父进程在聚合阶段中断：

```bash
python scripts/run_daphnet_full_subject_nbm_residual_inceptiontime.py \
  --finalize-only \
  --source-root outputs/daphnet_full_subject_nbm_residual_binary_v1/full_subject_binary_experiment \
  --output-root outputs/daphnet_full_subject_nbm_residual_inceptiontime_v1/full_subject_binary_experiment
```

## 单卡烟雾验证

服务器正式运行前建议使用独立目录验证 CUDA、输入通道和写盘：

```bash
python scripts/run_daphnet_full_subject_nbm_residual_inceptiontime.py \
  --smoke --device cuda:0 --threads 2 \
  --output-root outputs/_smoke_daphnet_inceptiontime_cuda
```
