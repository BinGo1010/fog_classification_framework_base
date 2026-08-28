# GRU-NGM 四扰动臂五种子训练（8 GPU）

## 实验固定项

- 数据：`processed_NBM_Exp`，P01–P08，3 折。
- 随机种子：`0, 52, 161, 5216, 52161`。
- 模型：30 通道 GRU-NGM，参数量 40,942；各臂结构完全相同。
- 训练：role 4；clean 重建目标；SmoothL1(`beta=1.0`)。
- 验证：无扰动 role 5；每 50 updates 验证；patience 20；最多 5000 updates。
- 优化器：AdamW，学习率 `3e-4`，weight decay `1e-4`；batch size 16；梯度裁剪 1.0。
- role 0/1 永久测试集在本阶段不会读取。

四个训练臂仅改变 role-4 网络输入：

| 训练臂 | clean | Gaussian | 连续时间 mask |
|---|---:|---:|---:|
| No perturbation | 100% | 0% | 0% |
| Gaussian only | 60% | 40% | 0% |
| Mask only | 80% | 0% | 20% |
| Gaussian + Mask | 40% | 40% | 20% |

Gaussian 为标准化空间 `std=0.04`；mask 为随机连续 4–8 点、30 通道同时置零。每次窗口被读取时动态且互斥地抽取一种模式，监督目标始终是该窗口的 clean 版本。

## 服务器启动

先确认服务器环境能识别 8 张 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

执行任务清单检查，不启动训练：

```bash
python -u scripts/launch_private_gru_ngm_perturbation_4arm_8gpu.py \
  --data-dir "$PWD/dataset/0.Private/processed_NBM_Exp" \
  --output-root "$PWD/outputs/private_gru_ngm_perturbation_4arm_5seed" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --dry-run
```

正式后台训练：

```bash
mkdir -p logs
nohup bash scripts/run_private_gru_ngm_perturbation_4arm_8gpu.sh \
  "$PWD/dataset/0.Private/processed_NBM_Exp" \
  "$PWD/outputs/private_gru_ngm_perturbation_4arm_5seed" \
  > logs/private_gru_ngm_perturbation_4arm_8gpu.log 2>&1 &
```

如果服务器 Python 不在默认路径：

```bash
PYTHON_BIN=/path/to/conda/env/bin/python \
  bash scripts/run_private_gru_ngm_perturbation_4arm_8gpu.sh \
  "$PWD/dataset/0.Private/processed_NBM_Exp" \
  "$PWD/outputs/private_gru_ngm_perturbation_4arm_5seed"
```

脚本可直接用同一命令安全重跑；哈希校验通过的已完成任务会跳过，未完成任务会重新执行。不要对同一个 `output-root` 改动代码、数据或参数；如需改变实验定义，应使用新的输出目录。

## 输出

全量训练共 `8 × 3 × 5 × 4 = 480` 个 GRU-NGM checkpoint。单个模型示例：

```text
outputs/private_gru_ngm_perturbation_4arm_5seed/
  runs/gaussian_mask/P01/fold_0/seed_0/
    checkpoints/gru_ngm_best.pt
    scaler_role4.json
    ngm_history.csv
    FROZEN_TRAIN.json
    DONE_TRAIN.json
```

全部完成后，根目录还会生成：

- `EXPERIMENT_PLAN.json`：冻结的数据、代码、参数和四臂定义；
- `PAIRING_AUDIT.json`：核验四臂初始权重、scaler 和角色划分一致；
- `TRAINING_SUMMARY.csv`：480 个模型的最佳 step、clean 验证损失、实际扰动比例和 checkpoint 哈希；
- `DONE.json`：全量训练完成标记。

训练日志位于 `logs/train/`，每个任务分别保存标准输出与错误日志。
