# Daphnet TCN-DAE + InceptionTime：8×2080 Ti运行手册

本实验从原始Daphnet数据重新训练TCN-DAE，不依赖上一轮输出中的`splits/`缓存。输出目录必须与旧TC-DAE或InceptionTime实验隔离。

## 启动8卡实验

```bash
python scripts/run_daphnet_full_subject_tcndae_inceptiontime.py \
  --launch-parallel \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
  --threads 2 \
  --output-root outputs/daphnet_full_subject_tcndae_inceptiontime_server_v1/full_subject_binary_experiment
```

父进程按外层训练窗口数对30个外层折做贪心负载均衡，并为每张GPU启动一个独立worker。每个外层折先训练3个OOF TCN-DAE和1个最终TCN-DAE，再训练B0–B3各3个InceptionTime种子。

如果两颗CPU的总可用核心数较少，将`--threads 2`改成`--threads 1`。2080 Ti显存为11 GB，本程序一次只在每张卡上保留一个训练模型。

## 中断和恢复

按`Ctrl+C`中断父进程后，重新执行完全相同的命令即可恢复。恢复层级包括：

- TCN-DAE每个epoch的模型、优化器、最佳模型、early-stopping、DataLoader shuffle、CPU/CUDA随机状态；
- InceptionTime每个epoch的模型、优化器、最佳模型和early-stopping状态；
- 已经同时生成`run_metrics.json`和`test_predictions.csv`的完整运行直接跳过。

worker异常退出会自动重试，默认最多2次。日志位于：

```text
<output-root>/logs/parallel_workers/
```

## 查看进度

```bash
python scripts/run_daphnet_full_subject_tcndae_inceptiontime.py \
  --status-only \
  --output-root outputs/daphnet_full_subject_tcndae_inceptiontime_server_v1/full_subject_binary_experiment
```

完整实验应显示120个TCN-DAE模型和360个分类运行，每种方法90次。

## 单卡CUDA烟雾测试

```bash
python scripts/run_daphnet_full_subject_tcndae_inceptiontime.py \
  --smoke --device cuda:0 --threads 2 \
  --output-root outputs/_smoke_tcndae_inceptiontime_cuda
```

## 仅重新聚合

```bash
python scripts/run_daphnet_full_subject_tcndae_inceptiontime.py \
  --finalize-only \
  --output-root outputs/daphnet_full_subject_tcndae_inceptiontime_server_v1/full_subject_binary_experiment
```
