# Daphnet 单被试 CA-SupCon 七卡运行说明

本程序实现实验协议 `CA-SUPCON-SUBJECT-V1`，直接读取 `processed_CA_pure/ca_window_manifest.csv` 中已经冻结的 `ca_split`。它不会重新随机划分窗口，也不会在验证集或测试集拟合 RobustScaler。

## 1. 七卡并行结构

七个正式被试分别固定到七张卡：

| GPU | 被试 |
|---:|---|
| 0 | S01 |
| 1 | S02 |
| 2 | S05 |
| 3 | S06 |
| 4 | S07 |
| 5 | S08 |
| 6 | S09 |

每个进程只看到一张卡，并在该卡内顺序运行 2026、2027、2028 三个种子和 S0–S3。没有 DDP，也不会在不同被试之间混合 batch 或梯度。

S03 保留为诊断被试，不进入正式七被试汇总；S04、S10 没有 FoG，不能定义本实验的二分类或监督对比损失。

## 2. 服务器准备

仓库与数据应保持如下相对位置：

```text
repo/
├── scripts/
├── dataset/
│   └── 1.Daphnet Freezing of Gait Dataset/
│       └── processed_CA_pure/
│           ├── ca_window_manifest.csv
│           ├── ca_protocol.json
│           └── records/*.npz
└── outputs/
```

创建环境后安装项目依赖。PyTorch 应按服务器 CUDA 版本单独安装：

```bash
python -m pip install -e .
```

## 3. 正式运行

```bash
chmod +x scripts/launch_daphnet_ca_supcon_7gpu.sh
nohup bash scripts/launch_daphnet_ca_supcon_7gpu.sh \
  > outputs/ca_supcon_launcher.log 2>&1 &
```

如果数据或输出位于其他路径：

```bash
DATA_DIR=/data/daphnet/processed_CA_pure \
OUTPUT_ROOT=/data/outputs/daphnet_ca_supcon_subject_v1 \
PYTHON_BIN=/opt/conda/envs/fog/bin/python \
bash scripts/launch_daphnet_ca_supcon_7gpu.sh
```

监控：

```bash
tail -f outputs/daphnet_ca_supcon_subject_v1/logs/S01.out.log
nvidia-smi
```

启动脚本可安全重跑：已经存在 `subject_complete.json` 的被试会跳过。若某个被试中途失败，该被试会从尚未完成的种子重新运行；已完成种子会由 `seed_complete.json` 跳过。

## 4. 小规模冒烟测试

正式上机前可只跑一个被试、一个种子和极少 epoch，验证数据路径、CUDA和输出权限：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_daphnet_ca_supcon_subject.py \
  --subject S08 --seeds 2026 --device cuda \
  --supervised-epochs 1 --supcon-epochs 1 --classifier-epochs 1 \
  --probe-every 1 --selection-probe-epochs 1 \
  --selection-probe-patience 1 --early-stopping-patience 1 \
  --output-root outputs/ca_supcon_smoke

python scripts/aggregate_daphnet_ca_supcon.py \
  --output-root outputs/ca_supcon_smoke \
  --subjects S08 --seeds 2026 --allow-incomplete
```

## 5. 实验约束实现

- S0：自然比例、普通未加权 BCE。
- S1：事件感知 1:1 batch、普通未加权 BCE；与 S0 使用相同种子的相同初始化。
- S2：1:1 CA-SupCon 预训练，丢弃投影头并冻结编码器，以自然比例训练线性头。
- S3：复用 S2 同一个已选择编码器，重新初始化线性头，以 1:1 事件感知 batch 训练。
- FoG 采样组使用原始事件 ID；Non-FoG 在冻结组内部确定性切成 20 s 采样片段。这个操作只服务 batch 采样，不改变集合归属。
- 每个标准 batch 为 16 FoG + 16 Non-FoG，每类尽量覆盖至少4个组，每组最多4窗。
- RobustScaler 只用训练窗拟合；验证与测试保持自然比例。
- 0.07、0.10、0.20 的温度只按验证集 linear-probe PR-AUC/F1 选择。
- 每个方法的分类阈值只按验证集 Balanced Accuracy/F1 选择，随后原样用于测试集。

## 6. 结果目录

```text
outputs/daphnet_ca_supcon_subject_v1/
├── S01/seed_2026/S0..S3/
│   ├── checkpoint.pt
│   ├── metrics.json
│   ├── predictions.csv
│   ├── training_history.csv/png
│   ├── test_pr_curve.png
│   ├── test_roc_curve.png
│   ├── test_confusion_matrix.png
│   ├── test_embedding_tsne.png
│   ├── test_probability_distribution.png
│   ├── test_representative_fog_timeline.png
│   └── test_false_positive_case.png（存在假阳性时）
├── all_metrics.csv
├── paired_deltas.csv
├── representation_diagnostics.csv
├── seed_metric_scatter.png
├── aggregate_confusion_matrices.png
└── CA_SupCon_experiment_report.md
```

`S06` 的训练集只有4个可用原始 FoG 事件，刚好达到每 batch 4 事件的下限，结果解释时应将事件多样性不足列为高风险项。

