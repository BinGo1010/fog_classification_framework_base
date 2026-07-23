# Daphnet 三 IMU、五种 NBM 核心实验套件

本文档对应 `scripts/run_daphnet_3imu_nbm_suite.py`，用于在本地或服务器上一条命令启动、记录日志，并在中断后用同一条命令恢复。

## 实验范围

“全部 3 个 IMU”在 Daphnet 中指三个佩戴位置的加速度计：

- ankle：forward、vertical、lateral；
- thigh：forward、vertical、lateral；
- trunk：forward、vertical、lateral。

因此模型输入是 **3 个传感器 × 3 个加速度轴 = 9 通道**，不是 3 通道，也不包含陀螺仪。

核心协议固定为：

- 在缩放、窗口构造和 LOSO 划分之前完全排除 S04、S10；
- 其余 S01、S02、S03、S05、S06、S07、S08、S09 执行 8 折 LOSO；
- 五种 NBM：Persistence、Linear-AR、GRU、TCN、Transformer；
- 所有 NBM 输出形状一致的 `(mu, sigma)`；
- residual history 为 0.5、1、2、4 秒；
- 下游分类器固定为相同结构的 TCN；
- 基础随机种子为 **42**；
- 每种 NBM 与每种 history 形成一组实验，即 `5 × 4 = 20` 组；
- 8 折全部完成后共有 `20 × 8 = 160` 个 fold cells。

同一 fold 内，每个 NBM 只训练一次，其 residual block 被四种 history 共享。四种 history 使用相同的 4 秒 maximum-support 测试锚点，避免样本集合不同造成不公平比较。

种子按 fold 确定性派生：NBM 使用 `42 + fold_index`，分类器使用
`42 + 10000 + fold_index`。同一 fold 的 20 个分类器使用同一个分类器种子，
从而保持初始化与数据顺序公平一致。

## 服务器环境准备

建议使用 Python 3.10 或更高版本。先根据服务器 CUDA 版本安装对应的 PyTorch，然后安装其余依赖：

```bash
cd /path/to/fog-merged
python3 -m venv .venv
source .venv/bin/activate

# 按 https://pytorch.org 的服务器 CUDA 版本安装 torch
pip install -r requirements.txt
```

运行前确认 PyTorch 和 GPU：

```bash
python -c "import torch; print(torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpus=', torch.cuda.device_count())"
```

处理后的 Daphnet 数据目录必须包含：

```text
processed/
├── manifest.csv
├── schema.json
└── records/
    ├── S01_seg000.npz
    └── ...
```

`schema.json` 必须声明 9 个通道，记录文件必须包含 `x` 和 `y_binary`。默认数据目录为仓库内的
`dataset/1.Daphnet Freezing of Gait Dataset/processed`；数据放在其他磁盘时，用
`--data-dir` 指定绝对路径即可。

## 一键启动

### Linux

```bash
cd /path/to/fog-merged
PYTHON_BIN="$PWD/.venv/bin/python" \
bash scripts/start_daphnet_3imu_nbm_suite.sh \
  --data-dir "/data/daphnet/processed" \
  --output-dir "/runs/daphnet_3imu_nbm_5x4_loso_seed42" \
  --device cuda \
  --num-workers 4
```

### PowerShell

```powershell
Set-Location E:\fog-merged
$env:PYTHON_BIN = "C:\path\to\python.exe"

powershell -ExecutionPolicy Bypass -File scripts\start_daphnet_3imu_nbm_suite.ps1 `
  --data-dir "E:\data\daphnet\processed" `
  --output-dir "E:\runs\daphnet_3imu_nbm_5x4_loso_seed42" `
  --device cuda `
  --num-workers 4
```

### 直接使用 Python 启动器

```bash
python scripts/start_daphnet_3imu_nbm_suite.py \
  --data-dir "/data/daphnet/processed" \
  --output-dir "/runs/daphnet_3imu_nbm_5x4_loso_seed42" \
  --device cuda
```

Python 启动器只消费下列启动层参数，其余参数全部原样传给核心 runner：

- `--python`：运行核心 runner 的 Python；
- `--log-file`：追加到指定日志文件；
- `--log-dir`：为本次启动创建独立时间戳日志的目录；
- `--show-core-help`：显示核心 runner 的完整参数；
- `--dry-run`：只显示最终命令和日志路径。

也可使用环境变量：

- `PYTHON_BIN`：包装脚本和 Python 启动器使用的解释器；
- `DAPHNET_SUITE_LOG_DIR`：默认日志目录。

启动器默认调用：

```text
python -u scripts/run_daphnet_3imu_nbm_suite.py --resume <透传参数>
```

如果明确传入 `--no-resume`，启动器不会再追加 `--resume`。

查看参数但不训练：

```bash
python scripts/start_daphnet_3imu_nbm_suite.py --help
python scripts/start_daphnet_3imu_nbm_suite.py --show-core-help
python scripts/start_daphnet_3imu_nbm_suite.py --dry-run \
  --data-dir "/data/daphnet/processed" \
  --output-dir "/runs/daphnet_suite"
```

## 日志与后台运行

默认日志保存在：

```text
outputs/logs/daphnet_3imu_nbm_suite/suite_YYYYMMDD_HHMMSS_pidNNNN.log
```

日志同时输出到终端并逐行刷新到文件。每次启动产生独立日志，不会覆盖前一次运行。需要固定文件时可使用：

```bash
bash scripts/start_daphnet_3imu_nbm_suite.sh \
  --log-file "/runs/logs/daphnet_suite.log" \
  --data-dir "/data/daphnet/processed" \
  --output-dir "/runs/daphnet_suite"
```

固定日志文件采用追加模式。服务器长期任务可交给 Slurm、systemd、tmux 或 nohup；即使外层终端输出被重定向，启动器自己的日志仍会持续保存。

## 中断后恢复

直接重新执行**完全相同的命令**即可。必须保持相同的 `--output-dir`、数据和实验协议参数：

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
bash scripts/start_daphnet_3imu_nbm_suite.sh \
  --data-dir "/data/daphnet/processed" \
  --output-dir "/runs/daphnet_3imu_nbm_5x4_loso_seed42" \
  --device cuda \
  --num-workers 4
```

恢复规则：

- `nbm/last.pt` 和每个分类器的 `classifier_last.pt` 从上一完整 epoch 继续；
- `best.pt`/`classifier_best.pt` 保存验证集最优模型；
- 已有且校验通过的 `DONE.json` 任务会跳过；
- residual cache 校验通过后直接复用；
- protocol fingerprint 不一致时拒绝恢复，防止错误复用旧模型；
- 中断发生在一个 epoch 中间时，该未完成 epoch 会在恢复时重新执行。

`device`、`num_workers`、数据挂载路径和输出目录是运行时信息，可以在恢复时调整；
恢复后 `config.json` 会刷新为当前路径。进入 protocol fingerprint 的数据内容、
核心实现源码、窗口、模型、训练轮数、batch size、seed 等必须保持不变；若要改变
这些内容，请使用新的输出目录。

## 产物结构

```text
<output-dir>/
├── config.json
├── run_manifest.json
├── environment.json
├── status.json
├── experiment_manifest.csv
├── fold_summary.csv
├── aggregate_metrics.json
└── loso_S01/
    ├── fold_config.json
    ├── scaler.json
    ├── split_indices.npz
    ├── history_support.npz
    ├── persistence/
    ├── linear_ar/
    ├── gru/
    ├── tcn/
    └── transformer/
```

每个 NBM 目录包含：

```text
<nbm>/
├── nbm/
│   ├── last.pt
│   ├── best.pt
│   ├── training.json
│   └── DONE.json
├── residual_cache.npz
├── residual_diagnostics.json
├── RESIDUAL_CACHE_DONE.json
├── nbm_summary.json
├── residual_h0p5s/
├── residual_h1s/
├── residual_h2s/
└── residual_h4s/
```

每个 history 分类器目录保存：

```text
classifier_last.pt
classifier_best.pt
metrics.json
predictions.npz
validation_predictions.npz
predictions.csv
DONE.json
```

`split_indices.npz`、`history_support.npz`、`scaler.json` 与
`residual_cache.npz` 联合保存了可精确重建的 NBM/分类器训练输入、验证输入和
测试输入；因此无需重复复制原始连续记录，也能审计和恢复每个训练任务。
核心套件因此要求保留 residual cache，不能使用 `--no-cache-residuals`。

`status.json` 给出 `expected_experiments=20`、`expected_fold_cells=160`、已完成 cell 数和总状态；`experiment_manifest.csv` 展示每组实验完成了哪些受试者；最终指标见 `fold_summary.csv` 与 `aggregate_metrics.json`。

## 完成审计

全量任务结束后执行专用审计器：

```bash
python scripts/audit_daphnet_3imu_nbm_suite.py \
  --result-dir "/runs/daphnet_3imu_nbm_5x4_loso_seed42"
```

审计应确认：

- 仅包含排除 S04、S10 后的 8 个 LOSO fold；
- 五种 NBM 和四种 history 完整形成 20 组；
- 共 160 个 fold cells；
- split、共同 history anchors、预测标签和上游 NBM checkpoint 对应关系一致；
- checkpoint、缓存、指标及 `DONE.json` 指纹有效。

调试或微型运行尚未完成全部 160 cells 时使用：

```bash
python scripts/audit_daphnet_3imu_nbm_suite.py \
  --result-dir "/runs/daphnet_suite_smoke" \
  --allow-partial
```

审计结果会原子保存为 `AUDIT_REPORT.json`。全量实验只有在审计器返回
`AUDIT_OK`，并在结果根目录生成 `SUITE_COMPLETE.json` 后才应视为完成；
partial smoke 只会返回 `AUDIT_PARTIAL_OK`，不会生成完整标记。
