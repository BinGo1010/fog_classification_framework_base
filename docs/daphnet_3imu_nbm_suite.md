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

## 7 张 GPU 并行 LOSO

核心协议共有 8 个 LOSO fold：

```text
S01 S02 S03 S05 S06 S07 S08 S09
```

`--worker-fold Sxx` 只限制当前进程实际执行的 fold，科学配置中的
`folds_resolved` 仍由 `--folds all` 固定为全部 8 折。因此所有 worker
必须使用 `--folds all`，不能把它改成 `--folds S01` 之类的单折配置。

每个 fold worker 会在一张 GPU 上依次完成：

- 5 个 NBM：Persistence、Linear-AR、GRU、TCN、Transformer；
- 每个 NBM 共享一次 residual cache；
- 每个 NBM 对应 4 个 history 分类器；
- 合计每个 fold 为 5 个 NBM 和 `5 × 4 = 20` 个分类器 cell。

一个 fold 从 5 个 NBM 到 20 个分类器会在所分配的同一张卡上完整执行，
不会把该 fold 拆到多张 GPU。

7 张 GPU 最多同时执行 7 折，不是把实验缩减为 7 折。首批将
S01、S02、S03、S05、S06、S07、S08 分配给物理 GPU 0–6；其中任意一张卡
先完成后，立即在该卡上继续执行 S09。总共仍然是 8 次单-fold worker
invocation 和 160 个 classifier cells。

多 GPU 调度器会完成 GPU 隔离、唯一 fold 分配、失败重试、日志、状态锁以及
最终审计。用户服务器上的完整命令为：

```bash
cd /document/home_mirror/chb/fog_classification_framework_base

DATA_DIR="/home/chb/Documents/FOG/fog_classification_framework_base/processed"
OUTPUT_DIR="$PWD/outputs/daphnet_3imu_nbm_5x4_loso_seed42"

mkdir -p "$OUTPUT_DIR"
test -r "$DATA_DIR/manifest.csv"
test -r "$DATA_DIR/schema.json"
test -w "$OUTPUT_DIR"

python -u scripts/start_daphnet_3imu_nbm_suite_multigpu.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --gpus 0-6 \
  --work-folds all \
  --max-retries 2 \
  --launch-delay 2 \
  --log-dir "$OUTPUT_DIR/multigpu_logs" \
  --audit \
  --seed 42 \
  --batch-size 256 \
  --num-workers 0
```

调度器按以下顺序执行：

1. 单进程 `--finalize-only` bootstrap，先固定共享
   `config.json` 和 `run_manifest.json`；
2. 在物理 GPU 0–6 上启动前 7 个 fold worker；
3. 通过 `CUDA_VISIBLE_DEVICES` 将每个 worker 隔离到一张物理 GPU，并把
   S09 交给首张完成当前 fold 的空闲卡；
4. 等待全部 8 个 fold 成功退出，再单进程 `--finalize-only` 重建权威根汇总；
5. 自动运行严格审计。全量运行不会添加 `--allow-partial`，只有审计生成
   `SUITE_COMPLETE.json` 才算完成。

调度器始终向核心 runner 传入 `--folds all --resume`，并为每个 worker 添加唯一的
`--worker-fold Sxx`。它还会在输出目录建立调度锁；通过该调度器启动时，同一
output 只会存在一个 scheduler，也不会重复分配 fold。不要绕过调度器手工启动
相同 fold。GPU worker 内部看到的 `cuda:0` 就是
`CUDA_VISIBLE_DEVICES` 指定的物理卡，不会让全部进程抢占物理 GPU 0。

正式启动前可以先确认完整调度而不创建文件：

```bash
cd /document/home_mirror/chb/fog_classification_framework_base
python -u scripts/start_daphnet_3imu_nbm_suite_multigpu.py \
  --data-dir "/home/chb/Documents/FOG/fog_classification_framework_base/processed" \
  --output-dir "$PWD/outputs/daphnet_3imu_nbm_5x4_loso_seed42" \
  --gpus 0-6 \
  --work-folds all \
  --audit \
  --seed 42 \
  --batch-size 256 \
  --num-workers 0 \
  --dry-run
```

### 中断恢复

先确认旧 worker 已经退出：

```bash
cd /document/home_mirror/chb/fog_classification_framework_base
pgrep -af "start_daphnet_3imu_nbm_suite_multigpu.py|run_daphnet_3imu_nbm_suite.py"
```

然后重新执行上面的**完全相同的多 GPU 调度命令**。同一个 `OUTPUT_DIR` 下：

- 已通过 SHA 校验的 DONE 任务直接跳过；
- 未完成的 NBM/分类器从最后一个完整 epoch 的 `last.pt` 恢复；
- 已完成 fold 会很快退出，空闲 GPU 会继续领取尚未完成的 fold；
- 失败 fold 默认最多重试 2 次，每次都复用已有 checkpoint；
- `multigpu_logs` 中的日志使用追加模式，不覆盖之前内容。

调度器被强制终止但子 worker 仍存活时，新调度器会拒绝抢锁，从而避免重复训练。
应先等待或停止旧 worker，再执行同一命令。所有 worker、bootstrap 和 finalize
使用相同的数据内容、源码和科学参数；GPU 编号和 `--num-workers` 属于运行配置。

### 日志、状态和 GPU 监控

分别另开监控终端：

```bash
cd /document/home_mirror/chb/fog_classification_framework_base
OUTPUT_DIR="$PWD/outputs/daphnet_3imu_nbm_5x4_loso_seed42"

# 物理 GPU、显存、利用率和各 worker PID。
watch -n 2 nvidia-smi
```

```bash
cd /document/home_mirror/chb/fog_classification_framework_base
OUTPUT_DIR="$PWD/outputs/daphnet_3imu_nbm_5x4_loso_seed42"

# 每次重新展开 *.log，因此稍后启动的 S09/finalize/audit 日志也会出现。
watch -n 10 "tail -n 15 '$OUTPUT_DIR'/multigpu_logs/*.log 2>/dev/null"
```

调度器状态保存在 `multigpu_status.json`。也可同时检查 classifier DONE 数量和
runner 的 `status.json`：

```bash
watch -n 15 "
  printf 'classifier DONE: '
  find '$OUTPUT_DIR' -path '*/residual_h*/DONE.json' | wc -l
  python -m json.tool '$OUTPUT_DIR/multigpu_status.json' 2>/dev/null || true
  python -m json.tool '$OUTPUT_DIR/status.json' 2>/dev/null || true
"
```

classifier DONE 应最终达到 160。为避免并发覆盖，fold worker 不写根汇总；
并行阶段的 `status.json` 会保持 bootstrap 时的 partial 快照，不会随 cell 实时推进。
实时进度以 classifier DONE 数量和 `multigpu_status.json` 为准。所有 worker 退出后的
单进程 finalize 才会刷新并生成权威 `status.json`。`multigpu_status.json` 记录
pending、running、succeeded/failed folds、对应物理 GPU、PID、重试次数和日志。
最终必须同时满足：

```text
multigpu_status.json: status = complete
status.json: status = complete
completed_fold_cells = 160
AUDIT_REPORT.json: status = verified_complete
SUITE_COMPLETE.json 存在
```

需要单独复核时，可重复执行严格审计；不要添加 `--allow-partial`：

```bash
cd /document/home_mirror/chb/fog_classification_framework_base
python -u scripts/audit_daphnet_3imu_nbm_suite.py \
  --result-dir "$PWD/outputs/daphnet_3imu_nbm_5x4_loso_seed42"
```

### 路径、权限与 OOM

命令把结果写入当前仓库下用户可写的
`"$PWD/outputs/daphnet_3imu_nbm_5x4_loso_seed42"`。不要改成根目录 `/`
或没有写权限的 `/runs/...`。启动前可检查：

```bash
cd /document/home_mirror/chb/fog_classification_framework_base
mkdir -p "$PWD/outputs"
test -w "$PWD/outputs"
df -h "$PWD" /home/chb/Documents/FOG/fog_classification_framework_base/processed
```

命令中的 `--num-workers 0` 最稳妥。确认 CPU、内存和磁盘吞吐充足后，可在
同一条调度命令中改为：

```bash
--num-workers 2
```

7 个 GPU worker 此时最多还会创建 14 个 DataLoader 子进程，不宜盲目设得过大。
`num-workers` 是运行时参数，可以在同一输出目录恢复时调整。

如果发生 CUDA OOM，不能在原输出目录直接把 batch size 从 256 改为 128，
因为 batch size 属于 protocol fingerprint。请使用一个全新的输出目录重新运行：

```bash
cd /document/home_mirror/chb/fog_classification_framework_base
python -u scripts/start_daphnet_3imu_nbm_suite_multigpu.py \
  --data-dir "/home/chb/Documents/FOG/fog_classification_framework_base/processed" \
  --output-dir "$PWD/outputs/daphnet_3imu_nbm_5x4_loso_seed42_bs128" \
  --gpus 0-6 \
  --work-folds all \
  --max-retries 2 \
  --audit \
  --seed 42 \
  --batch-size 128 \
  --num-workers 0
```

不要把旧输出目录的 checkpoint 或 DONE 复制到新的 batch-size-128 目录。

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
