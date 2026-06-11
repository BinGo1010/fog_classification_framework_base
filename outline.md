# FOG Classification Project Outline

## 0. Project Goal

本项目面向帕金森病冻结步态（Freezing of Gait, FOG）分类识别任务，输入为可穿戴 IMU 时序信号，输出为：

- `0`: normal
- `1`: freezing

项目核心目标：

1. 实现 FOG 分类任务；
2. 支持跨被试泛化实验，重点采用 LOSO（Leave-One-Subject-Out）验证；
3. 支持少传感器 / 单 IMU 输入；
4. 支持 Gumbel-Softmax 传感器选择；
5. 支持监督对比学习，提高跨被试泛化能力；
6. 支持轻量化模型设计，例如 1D-MobileOne；
7. 支持边缘端部署相关评估，包括参数量、FLOPs、模型大小、推理速度和吞吐率。

---

## 1. Recommended Project Structure

```text
FOG-Classification/
├── README.md
├── requirements.txt
├── run.py
│
├── configs/
│   ├── fog_mobileone1d.yaml
│   ├── fog_loso.yaml
│   ├── fog_supcon.yaml
│   ├── fog_gumbel_sensor_select.yaml
│   └── fog_ablation.yaml
│
├── data_provider/
│   ├── __init__.py
│   ├── data_factory.py
│   ├── fog_dataset.py
│   ├── fog_preprocess.py
│   ├── fog_window.py
│   ├── fog_loso_split.py
│   └── fog_transform.py
│
├── exp/
│   ├── __init__.py
│   ├── exp_basic.py
│   ├── exp_fog_classification.py
│   ├── exp_loso_classification.py
│   ├── exp_supcon_pretrain.py
│   ├── exp_supcon_finetune.py
│   ├── exp_gumbel_sensor_selection.py
│   └── exp_ablation.py
│
├── models/
│   ├── __init__.py
│   ├── model_factory.py
│   ├── MobileOne1D.py
│   ├── LightTCN.py
│   ├── DLinearCls.py
│   ├── PatchTSTCls.py
│   ├── CNN1D.py
│   ├── LSTMCls.py
│   ├── GRUCls.py
│   └── SupConEncoder.py
│
├── layers/
│   ├── __init__.py
│   ├── mobileone1d_block.py
│   ├── conv_blocks.py
│   ├── temporal_blocks.py
│   ├── sensor_selection.py
│   ├── gumbel_softmax_selector.py
│   ├── classifier_head.py
│   └── projection_head.py
│
├── losses/
│   ├── __init__.py
│   ├── classification_loss.py
│   ├── focal_loss.py
│   ├── supcon_loss.py
│   └── sensor_cost_loss.py
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── efficiency_metrics.py
│   ├── model_profile.py
│   ├── logger.py
│   ├── early_stopping.py
│   ├── lr_scheduler.py
│   ├── seed.py
│   ├── visualization.py
│   ├── save_results.py
│   ├── confusion_matrix.py
│   └── export_onnx.py
│
├── scripts/
│   ├── run_mobileone1d.sh
│   ├── run_dlinear.sh
│   ├── run_patchtst.sh
│   ├── run_lighttcn.sh
│   ├── run_loso_mobileone1d.sh
│   ├── run_supcon_pretrain.sh
│   ├── run_supcon_finetune.sh
│   ├── run_gumbel_sensor_selection.sh
│   ├── run_ablation_sensor_position.sh
│   ├── run_ablation_window_size.sh
│   └── run_model_profile.sh
│
├── checkpoints/
│   ├── mobileone1d/
│   ├── dlinear/
│   ├── patchtst/
│   ├── lighttcn/
│   ├── supcon_pretrain/
│   └── gumbel_sensor_selection/
│
├── results/
│   ├── ordinary/
│   ├── loso/
│   ├── supcon/
│   ├── sensor_selection/
│   ├── ablation/
│   ├── model_profile/
│   └── figures/
│
├── logs/
│   ├── tensorboard/
│   ├── train_logs/
│   ├── test_logs/
│   └── error_logs/
│
└── docs/
    ├── dataset_format.md
    ├── model_design.md
    ├── loso_protocol.md
    ├── sensor_selection.md
    ├── contrastive_learning.md
    ├── evaluation_metrics.md
    └── deployment.md
```

---

## 2. Environment

本项目不使用 Docker，推荐使用 Conda 或 Python venv。

### 2.1 requirements.txt

```text
torch
torchvision
torchaudio
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
tqdm
einops
thop
ptflops
pyyaml
tensorboard
```

如果后续需要部署或导出模型，可额外添加：

```text
onnx
onnxruntime
```

---

## 3. Main Entrance: run.py

`run.py` 是整个项目的总入口，负责解析参数、读取配置、选择实验流程并启动训练或测试。

### 3.1 Main Responsibilities

1. 解析命令行参数；
2. 加载 YAML 配置文件；
3. 设置随机种子；
4. 设置 GPU / CPU；
5. 根据 `exp_mode` 选择实验流程；
6. 调用训练、验证和测试；
7. 保存实验结果。

### 3.2 Recommended Arguments

```text
--task_name              fog_classification
--model                  MobileOne1D / LightTCN / DLinearCls / PatchTSTCls
--data                   FOG
--root_path              ./dataset/fog/
--seq_len                256
--stride                 128
--num_class              3
--enc_in                 6 / 12 / 18 / 24
--batch_size             64
--learning_rate          0.001
--train_epochs           100
--patience               15
--loss                   ce / weighted_ce / focal
--exp_mode               ordinary / loso / supcon_pretrain / supcon_finetune / sensor_select / ablation
--imu_position           ankleL / ankleR / thigh / waist / all
--use_gumbel             0 / 1
--target_sensor_num      1 / 2 / 3 / 4
--use_amp                0 / 1
--gpu                    0
```

---

## 4. Configs

`configs/` 用于保存不同实验的配置文件，方便复现实验。

### 4.1 Example: configs/fog_mobileone1d.yaml

```yaml
task_name: fog_classification
model: MobileOne1D
data: FOG
root_path: ./dataset/fog/
seq_len: 256
stride: 128
num_class: 3
enc_in: 24
batch_size: 64
learning_rate: 0.001
train_epochs: 100
patience: 15
loss: weighted_ce
exp_mode: ordinary
imu_position: all
use_gumbel: false
use_amp: true
```

### 4.2 Example: configs/fog_loso.yaml

```yaml
task_name: fog_classification
model: MobileOne1D
data: FOG
root_path: ./dataset/fog/
seq_len: 256
stride: 128
num_class: 3
enc_in: 24
batch_size: 64
learning_rate: 0.001
train_epochs: 100
patience: 15
loss: weighted_ce
exp_mode: loso
imu_position: all
use_gumbel: false
```

### 4.3 Example: configs/fog_gumbel_sensor_select.yaml

```yaml
task_name: fog_classification
model: MobileOne1D
data: FOG
root_path: ./dataset/fog/
seq_len: 256
stride: 128
num_class: 3
enc_in: 24
batch_size: 64
learning_rate: 0.001
train_epochs: 100
patience: 15
loss: weighted_ce
exp_mode: sensor_select
imu_position: all
use_gumbel: true
target_sensor_num: 1
sensor_cost_weight: 0.01
sparsity_weight: 0.001
```

---

## 5. Data Provider

`data_provider/` 负责数据读取、预处理、滑窗切分、数据增强和跨被试划分。

---

### 5.1 data_provider/data_factory.py

统一数据入口。

主要功能：

1. 根据 `args.data` 选择对应 Dataset；
2. 根据 `flag` 返回训练集、验证集或测试集；
3. 构建 PyTorch DataLoader；
4. 支持普通划分和 LOSO 划分；
5. 支持指定 IMU 位置或通道组合。

推荐接口：

```python
def data_provider(args, flag):
    dataset = Dataset_FOG(args, flag)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True if flag == "train" else False,
        num_workers=args.num_workers,
        drop_last=True if flag == "train" else False
    )
    return dataset, dataloader
```

---

### 5.2 data_provider/fog_dataset.py

FOG 数据集类。

推荐统一输入格式：

```text
X: [N, T, C]
y: [N]
subject_id: [N]
```

其中：

```text
N: 样本数量
T: 时间窗口长度
C: 输入通道数量
```

对于 4 个 IMU，每个 IMU 6 通道：

```text
ankleL: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
ankleR: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
thigh:  acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
waist:  acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
```

总通道数：

```text
C = 4 × 6 = 24
```

标签定义：

```text
0: normal
1: pre-freezing
2: freezing
```

---

### 5.3 data_provider/fog_preprocess.py

信号预处理模块。

建议包含：

1. 缺失值处理；
2. 异常值处理；
3. 重采样；
4. 滤波；
5. 标准化；
6. 按被试归一化；
7. 标签对齐；
8. 不同 IMU 之间的时间同步。

推荐预处理流程：

```text
raw IMU signals
    ↓
time synchronization
    ↓
resampling
    ↓
filtering
    ↓
normalization
    ↓
window segmentation
    ↓
train / val / test split
```

---

### 5.4 data_provider/fog_window.py

滑动窗口切分模块。

推荐参数：

```text
window_size = 256
stride = 128
```

窗口标签生成策略：

1. 中心点标签；
2. 多数投票标签；
3. freezing 优先标签；
4. pre-freezing 时间段构造标签。

对于 FOG 预警任务，建议将 freezing 前一定时间段构造为 `pre-freezing`：

```text
normal:       非 FOG 且非 FOG 前兆区域
pre-freezing: FOG 发生前 1~3 秒
freezing:     FOG 事件区域
```

---

### 5.5 data_provider/fog_loso_split.py

LOSO 跨被试划分模块。

LOSO 策略：

```text
for each subject:
    test_subject = current subject
    train_subjects = all subjects except current subject
    val_subjects = split from train_subjects
```

输出示例：

```text
fold_S01:
    train: S02, S03, S04, ...
    val:   subset of train subjects
    test:  S01

fold_S02:
    train: S01, S03, S04, ...
    val:   subset of train subjects
    test:  S02
```

---

### 5.6 data_provider/fog_transform.py

数据增强模块。

推荐增强方法：

1. Jittering；
2. Scaling；
3. Time warping；
4. Magnitude warping；
5. Random crop；
6. Channel dropout；
7. Sensor dropout；
8. Gaussian noise。

对 FOG 分类任务优先推荐：

```text
Jittering
Scaling
Channel Dropout
Sensor Dropout
```

---

## 6. Experiments

`exp/` 负责训练、验证、测试、LOSO、对比学习、传感器选择和消融实验。

---

### 6.1 exp/exp_basic.py

基础实验类。

主要功能：

1. 设备选择；
2. 模型构建；
3. 优化器构建；
4. 学习率调度；
5. 早停；
6. checkpoint 保存；
7. 日志初始化。

---

### 6.2 exp/exp_fog_classification.py

普通 FOG 分类实验。

流程：

```text
load train / val / test data
    ↓
build model
    ↓
train model
    ↓
validate model
    ↓
test model
    ↓
save metrics and figures
```

需要保存：

```text
metrics.csv
classification_report.txt
confusion_matrix.png
prediction.csv
best_model.pth
```

---

### 6.3 exp/exp_loso_classification.py

LOSO 跨被试实验。

流程：

```text
for subject in subjects:
    select current subject as test subject
    train model on remaining subjects
    validate model
    test model on current subject
    save fold metrics

summarize all fold results
```

结果保存：

```text
results/loso/
├── fold_S01_metrics.csv
├── fold_S02_metrics.csv
├── fold_S03_metrics.csv
├── ...
├── loso_summary.csv
├── loso_confusion_matrix.png
└── loso_subject_barplot.png
```

---

### 6.4 exp/exp_supcon_pretrain.py

监督对比学习预训练。

目标：

1. 提升跨被试泛化能力；
2. 使同类别样本在特征空间更接近；
3. 使不同类别样本在特征空间更分离；
4. 减少不同被试之间的个体差异影响。

输入：

```text
IMU window
class label
subject_id
```

输出：

```text
encoder checkpoint
feature embedding
projection head checkpoint
```

---

### 6.5 exp/exp_supcon_finetune.py

对比学习微调实验。

流程：

```text
load pretrained encoder
    ↓
add classification head
    ↓
finetune model
    ↓
test normal / pre-freezing / freezing classification
```

微调策略：

1. 冻结 encoder，只训练分类头；
2. 解冻 encoder，端到端微调；
3. 使用较小学习率微调 encoder；
4. 比较 ordinary training 和 SupCon pretraining 的差异。

---

### 6.6 exp/exp_gumbel_sensor_selection.py

Gumbel-Softmax 传感器选择实验。

目标：

1. 从多个 IMU 中选择最优 IMU 组合；
2. 从多个通道中选择最优通道组合；
3. 减少输入传感器数量；
4. 降低实际外骨骼系统的佩戴复杂度；
5. 降低边缘端推理计算量。

适合当前设置：

```text
4 个 IMU 位置
每个 IMU 6 个通道
总输入通道数 C = 24
目标选择 1 个或 2 个 IMU
```

推荐损失函数：

```text
total_loss = classification_loss
           + lambda_sensor * sensor_cost_loss
           + lambda_sparse * sparsity_loss
```

---

### 6.7 exp/exp_ablation.py

消融实验模块。

建议消融内容：

1. 不同模型对比；
2. 不同 IMU 位置对比；
3. 不同传感器数量对比；
4. 有无 Gumbel-Softmax 传感器选择；
5. 有无监督对比学习；
6. 不同窗口长度；
7. 不同步长；
8. 不同损失函数；
9. 不同轻量化模型宽度；
10. 不同 pre-freezing 时间定义。

---

## 7. Models

`models/` 存放模型主干。

---

### 7.1 models/MobileOne1D.py

推荐作为主模型。

输入：

```text
[B, T, C]
```

模型流程：

```text
Input: [B, T, C]
    ↓
Permute: [B, T, C] -> [B, C, T]
    ↓
Stem Conv1D
    ↓
MobileOne1D Blocks
    ↓
Global Average Pooling
    ↓
Classifier Head
    ↓
Output: [B, num_class]
```

推荐模型规格：

```text
MobileOne1D-XS
MobileOne1D-S
MobileOne1D-M
```

论文中可重点强调：

1. 重参数化结构；
2. 训练阶段多分支；
3. 推理阶段单分支；
4. 低参数量；
5. 快速推理；
6. 适合边缘端部署。

---

### 7.2 models/LightTCN.py

轻量级 TCN baseline。

特点：

1. 一维卷积结构；
2. 可使用膨胀卷积；
3. 参数量小；
4. 推理速度快；
5. 适合边缘端。

---

### 7.3 models/DLinearCls.py

DLinear 分类版本。

用途：

1. 作为轻量线性时序 baseline；
2. 与深度模型进行对比；
3. 验证简单模型在 FOG 分类任务中的表现。

---

### 7.4 models/PatchTSTCls.py

PatchTST 分类版本。

用途：

1. 作为强时序 Transformer baseline；
2. 验证 patch-based temporal modeling 的效果；
3. 与轻量化模型进行性能和效率对比。

---

### 7.5 models/SupConEncoder.py

监督对比学习编码器。

结构：

```text
Backbone encoder
    ↓
Projection head
    ↓
Normalized embedding
```

用于：

1. SupCon 预训练；
2. 特征可视化；
3. 下游分类微调。

---

### 7.6 models/model_factory.py

模型工厂。

功能：

```python
def build_model(args):
    if args.model == "MobileOne1D":
        return MobileOne1D.Model(args)
    elif args.model == "LightTCN":
        return LightTCN.Model(args)
    elif args.model == "DLinearCls":
        return DLinearCls.Model(args)
    elif args.model == "PatchTSTCls":
        return PatchTSTCls.Model(args)
    else:
        raise ValueError(f"Unknown model: {args.model}")
```

---

## 8. Layers

`layers/` 存放通用网络模块。

---

### 8.1 layers/mobileone1d_block.py

MobileOne 1D 重参数化模块。

训练阶段：

```text
Conv1D branch
1x1 Conv branch
Identity branch
BatchNorm
Activation
```

推理阶段：

```text
Equivalent single Conv1D
```

优势：

1. 训练阶段表达能力强；
2. 推理阶段结构简单；
3. 推理速度快；
4. 适合边缘端部署。

---

### 8.2 layers/gumbel_softmax_selector.py

Gumbel-Softmax 可微传感器选择器。

输入：

```text
[B, T, C]
```

输出：

```text
[B, T, C_selected]
```

或：

```text
[B, T, C] * mask
```

支持两种选择粒度：

1. IMU-level selection；
2. Channel-level selection。

---

### 8.3 layers/sensor_selection.py

传感器选择封装模块。

IMU-level selection：

```text
ankleL
ankleR
thigh
waist
```

Channel-level selection：

```text
acc_x
acc_y
acc_z
gyro_x
gyro_y
gyro_z
```

对于外骨骼适配，建议优先做 IMU-level selection，因为它更符合实际佩戴系统中减少传感器数量的目标。

---

### 8.4 layers/classifier_head.py

分类头。

推荐结构：

```text
Global Average Pooling
Dropout
Linear
Softmax
```

输出：

```text
normal
pre-freezing
freezing
```

---

### 8.5 layers/projection_head.py

对比学习投影头。

推荐结构：

```text
Linear
ReLU
Linear
L2 Normalization
```

---

## 9. Losses

`losses/` 存放损失函数。

---

### 9.1 losses/classification_loss.py

普通分类损失。

支持：

```text
CrossEntropyLoss
WeightedCrossEntropyLoss
```

对于 FOG 数据类别不平衡，建议优先使用 `WeightedCrossEntropyLoss`。

---

### 9.2 losses/focal_loss.py

Focal Loss。

适合处理类别不平衡问题，尤其是 `pre-freezing` 和 `freezing` 样本较少的情况。

---

### 9.3 losses/supcon_loss.py

监督对比学习损失。

用于增强跨被试泛化能力。

目标：

```text
same class samples closer
different class samples farther
```

---

### 9.4 losses/sensor_cost_loss.py

传感器代价损失。

用于约束传感器数量。

推荐形式：

```text
sensor_cost_loss = selected_sensor_num / total_sensor_num
```

最终损失：

```text
total_loss = classification_loss
           + lambda_sensor * sensor_cost_loss
           + lambda_sparse * sparsity_loss
```

---

## 10. Utils

`utils/` 存放工具函数。

---

### 10.1 utils/metrics.py

分类评价指标。

建议包含：

```text
Accuracy
Precision
Recall
F1-score
Macro-F1
Weighted-F1
Sensitivity
Specificity
Balanced Accuracy
AUC
```

对于 FOG 分类论文，建议重点报告：

```text
Accuracy
Macro-F1
Weighted-F1
Sensitivity
Specificity
Freezing Recall
Pre-freezing Recall
```

---

### 10.2 utils/efficiency_metrics.py

轻量化评价指标。

建议包含：

```text
Params
FLOPs
Model Size
Inference Time
Latency
Throughput
Memory Usage
CPU Inference Time
GPU Inference Time
```

论文中建议报告：

```text
Params
FLOPs
Model Size
Single-sample Latency
Batch Throughput
```

---

### 10.3 utils/model_profile.py

模型复杂度分析。

输出：

```text
model_name
params
flops
model_size
single_sample_inference_time
batch_inference_time
throughput
```

---

### 10.4 utils/visualization.py

论文图表生成。

建议支持：

1. 混淆矩阵；
2. ROC 曲线；
3. PR 曲线；
4. t-SNE / UMAP 特征分布；
5. 不同模型性能柱状图；
6. 参数量-性能散点图；
7. 推理速度-性能散点图；
8. 传感器选择热力图；
9. LOSO 各被试结果柱状图；
10. 消融实验对比图。

---

### 10.5 utils/export_onnx.py

模型导出。

功能：

```text
PyTorch model -> ONNX
```

后续可扩展：

```text
ONNX Runtime
TensorRT
TFLite
NCNN
```

---

## 11. Scripts

`scripts/` 存放实验命令。

---

### 11.1 scripts/run_mobileone1d.sh

```bash
python run.py \
  --task_name fog_classification \
  --model MobileOne1D \
  --data FOG \
  --root_path ./dataset/fog/ \
  --seq_len 256 \
  --stride 128 \
  --num_class 3 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 15
```

---

### 11.2 scripts/run_loso_mobileone1d.sh

```bash
python run.py \
  --task_name fog_classification \
  --exp_mode loso \
  --model MobileOne1D \
  --data FOG \
  --root_path ./dataset/fog/ \
  --seq_len 256 \
  --stride 128 \
  --num_class 3 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 15
```

---

### 11.3 scripts/run_gumbel_sensor_selection.sh

```bash
python run.py \
  --task_name fog_classification \
  --exp_mode sensor_select \
  --model MobileOne1D \
  --use_gumbel 1 \
  --target_sensor_num 1 \
  --data FOG \
  --root_path ./dataset/fog/ \
  --seq_len 256 \
  --stride 128 \
  --num_class 3
```

---

### 11.4 scripts/run_supcon_pretrain.sh

```bash
python run.py \
  --task_name fog_classification \
  --exp_mode supcon_pretrain \
  --model MobileOne1D \
  --data FOG \
  --root_path ./dataset/fog/ \
  --seq_len 256 \
  --stride 128 \
  --num_class 3
```

---

### 11.5 scripts/run_supcon_finetune.sh

```bash
python run.py \
  --task_name fog_classification \
  --exp_mode supcon_finetune \
  --model MobileOne1D \
  --data FOG \
  --root_path ./dataset/fog/ \
  --seq_len 256 \
  --stride 128 \
  --num_class 3
```

---

### 11.6 scripts/run_model_profile.sh

```bash
python run.py \
  --task_name fog_classification \
  --exp_mode profile \
  --model MobileOne1D \
  --data FOG \
  --root_path ./dataset/fog/ \
  --seq_len 256 \
  --num_class 3
```

---

## 12. Checkpoints

`checkpoints/` 保存模型权重。

推荐保存：

```text
best_model.pth
last_model.pth
config.yaml
training_log.csv
```

目录示例：

```text
checkpoints/
├── mobileone1d/
│   ├── best_model.pth
│   ├── last_model.pth
│   └── config.yaml
├── supcon_pretrain/
│   ├── encoder_best.pth
│   └── projection_head_best.pth
└── gumbel_sensor_selection/
    ├── best_model.pth
    └── selected_sensors.json
```

---

## 13. Results

`results/` 保存实验结果。

推荐保存：

```text
metrics.csv
classification_report.txt
confusion_matrix.png
roc_curve.png
pr_curve.png
prediction.csv
feature_tsne.png
efficiency_report.csv
```

目录示例：

```text
results/
├── ordinary/
│   ├── metrics.csv
│   ├── classification_report.txt
│   └── confusion_matrix.png
├── loso/
│   ├── fold_S01_metrics.csv
│   ├── fold_S02_metrics.csv
│   ├── loso_summary.csv
│   └── loso_confusion_matrix.png
├── supcon/
│   ├── pretrain_log.csv
│   ├── finetune_metrics.csv
│   └── feature_tsne.png
├── sensor_selection/
│   ├── selected_sensors.json
│   ├── sensor_importance.csv
│   └── sensor_heatmap.png
├── ablation/
│   ├── ablation_model.csv
│   ├── ablation_sensor.csv
│   └── ablation_window_size.csv
└── model_profile/
    ├── efficiency_report.csv
    └── params_flops_latency.csv
```

---

## 14. Logs

`logs/` 保存训练日志。

```text
logs/
├── tensorboard/
├── train_logs/
├── test_logs/
└── error_logs/
```

建议日志内容：

```text
epoch
train_loss
val_loss
accuracy
macro_f1
weighted_f1
learning_rate
time_per_epoch
```

---

## 15. Docs

`docs/` 保存项目说明文档。

```text
docs/
├── dataset_format.md
├── model_design.md
├── loso_protocol.md
├── sensor_selection.md
├── contrastive_learning.md
├── evaluation_metrics.md
└── deployment.md
```

---

## 16. Recommended Development Stages

### Stage 1: Basic FOG Classification

目标：先跑通普通分类任务。

需要完成：

```text
run.py
data_provider/fog_dataset.py
data_provider/data_factory.py
exp/exp_fog_classification.py
models/CNN1D.py
utils/metrics.py
```

最低可运行目标：

```text
input:  [B, T, C]
output: [B, 3]
metric: accuracy, macro-f1, confusion matrix
```

---

### Stage 2: Lightweight Model

目标：加入 1D-MobileOne 主模型。

需要完成：

```text
models/MobileOne1D.py
layers/mobileone1d_block.py
utils/model_profile.py
utils/efficiency_metrics.py
```

重点输出：

```text
Accuracy
Macro-F1
Params
FLOPs
Model Size
Inference Time
Throughput
```

---

### Stage 3: LOSO Cross-subject Evaluation

目标：验证跨被试泛化能力。

需要完成：

```text
data_provider/fog_loso_split.py
exp/exp_loso_classification.py
results/loso_summary.csv
```

重点输出：

```text
mean accuracy
std accuracy
mean macro-f1
std macro-f1
per-subject performance
```

---

### Stage 4: Supervised Contrastive Learning

目标：提升跨被试泛化能力。

需要完成：

```text
losses/supcon_loss.py
models/SupConEncoder.py
layers/projection_head.py
exp/exp_supcon_pretrain.py
exp/exp_supcon_finetune.py
```

对比实验：

```text
MobileOne1D without SupCon
MobileOne1D with SupCon
```

---

### Stage 5: Gumbel-Softmax Sensor Selection

目标：减少传感器数量。

需要完成：

```text
layers/gumbel_softmax_selector.py
layers/sensor_selection.py
losses/sensor_cost_loss.py
exp/exp_gumbel_sensor_selection.py
```

重点输出：

```text
selected sensor position
selected channel
classification performance
sensor number
params
FLOPs
latency
```

---

### Stage 6: Ablation and Paper Figures

目标：生成论文实验结果和图表。

需要完成：

```text
exp/exp_ablation.py
utils/visualization.py
utils/save_results.py
scripts/run_ablation_sensor_position.sh
scripts/run_ablation_window_size.sh
scripts/run_model_profile.sh
```

建议图表：

1. 不同模型分类性能对比；
2. 不同模型轻量化指标对比；
3. Accuracy / Macro-F1 / Params / FLOPs 综合表；
4. 混淆矩阵；
5. LOSO 各被试结果；
6. 传感器选择热力图；
7. t-SNE 特征可视化；
8. 推理速度-性能散点图；
9. 参数量-性能散点图。

---

## 17. Suggested Paper Experiment Design

### 17.1 Main Comparison

比较模型：

```text
CNN1D
LSTM
GRU
DLinearCls
LightTCN
PatchTSTCls
MobileOne1D
MobileOne1D + SupCon
MobileOne1D + Gumbel Sensor Selection
MobileOne1D + SupCon + Gumbel Sensor Selection
```

### 17.2 Sensor Position Study

比较输入：

```text
ankleL
ankleR
thigh
waist
ankleL + ankleR
ankleL + thigh
ankleL + waist
all IMUs
Gumbel-selected IMU
```

### 17.3 Cross-subject Study

采用：

```text
LOSO
```

输出：

```text
Mean ± Std
Per-subject Accuracy
Per-subject Macro-F1
```

### 17.4 Efficiency Study

比较：

```text
Params
FLOPs
Model Size
Latency
Throughput
```

### 17.5 Ablation Study

建议消融：

```text
without SupCon
with SupCon
without Gumbel
with Gumbel
without MobileOne re-parameterization
with MobileOne re-parameterization
different window sizes
different sensor numbers
different loss functions
```

---

## 18. Recommended Research Narrative

论文主线可以设计为：

```text
Existing FOG classification methods often depend on multiple sensors,
large models, or subject-dependent evaluation settings.

To address these limitations, this project proposes a lightweight
cross-subject FOG classification framework based on wearable IMU signals.

The proposed framework includes:
1. a 1D-MobileOne lightweight temporal classifier,
2. a Gumbel-Softmax-based sensor selection module,
3. a supervised contrastive learning strategy for cross-subject generalization.

The final goal is to provide a fast, compact, and sensor-efficient
FOG recognition model for real-time hip exoskeleton assistance.
```

中文表达：

```text
现有 FOG 分类方法往往依赖多传感器、高复杂度模型或被试相关实验设置，
难以直接应用于面向外骨骼辅助的实时边缘端系统。

因此，本项目提出一种面向跨被试冻结步态分类任务的轻量化识别框架。
该框架以 IMU 时序信号为输入，结合 1D-MobileOne 轻量模型、
Gumbel-Softmax 传感器选择机制和监督对比学习策略，
在保证分类性能的同时减少传感器数量和模型复杂度，
从而为髋关节外骨骼辅助系统提供实时、紧凑、可部署的 FOG 状态识别方法。
```

---

## 19. Minimum Viable Version

如果先实现最小可运行版本，建议只保留：

```text
FOG-Classification/
├── requirements.txt
├── run.py
├── data_provider/
│   ├── data_factory.py
│   └── fog_dataset.py
├── exp/
│   ├── exp_basic.py
│   └── exp_fog_classification.py
├── models/
│   ├── model_factory.py
│   └── CNN1D.py
├── utils/
│   ├── metrics.py
│   ├── seed.py
│   └── early_stopping.py
├── scripts/
│   └── run_cnn1d.sh
├── checkpoints/
└── results/
```

先跑通后，再逐步增加：

```text
MobileOne1D
LOSO
SupCon
Gumbel Sensor Selection
Model Profile
Visualization
```

---

## 20. Final Summary

本项目的工程核心可以概括为：

```text
data_provider:
    解决 FOG IMU 数据读取、预处理、滑窗和跨被试划分

exp:
    解决训练、验证、测试、LOSO、对比学习和传感器选择实验流程

models:
    解决 FOG 分类模型，包括轻量化 1D-MobileOne 和多种 baseline

layers:
    解决 MobileOne1D block、Gumbel-Softmax 选择器和分类头等基础模块

losses:
    解决类别不平衡、监督对比学习和传感器数量约束

utils:
    解决评价指标、轻量化指标、可视化、日志和模型导出

scripts:
    解决实验命令复现

results:
    保存论文实验结果和图表
```

最终目标是构建一个可复现、可扩展、适合论文实验，并能够面向髋关节外骨骼实时辅助应用的 FOG 分类识别框架。
