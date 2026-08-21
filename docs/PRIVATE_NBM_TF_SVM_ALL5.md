# Private NBM：五传感器 TF+SVM 实验

## 实验定义

- 数据集：`dataset/0.Private/processed_NBM_Exp`
- 实验单位：每名被试独立建模，不把其他被试的数据加入训练
- 被试：P01–P08；每名被试执行 outer 0/1/2 三折并对三折结果取平均
- 输入：lumbar、ankle_l、ankle_r、foot_l、foot_r 五个 IMU
- 通道：每个 IMU 的三轴加速度和三轴角速度，共 30 通道
- 窗口：2 s（128 点），采样率 64 Hz
- 固定角色：6/7 训练、2/3 验证、0/1 测试
- 随机性：该流程没有随机采样、随机初始化或概率校准，因此不设置随机种子

## 特征与模型

每通道提取 11 个时频特征，共 330 维：

- 时域：标准差、峰峰值、平均绝对值、差分 RMS、线长
- 频域：0.5–3 Hz、3–8 Hz、8–28 Hz 对数频带功率，对数冻结指数，0.5–28 Hz 频谱熵和主频

模型为 `StandardScaler + class_weight=balanced` 的 RBF-SVM。使用验证集 PR-AUC 从固定网格选择 C 和 gamma，再在同一验证集按最大 FoG-F1 确定唯一报警阈值。测试集不参与模型选择或阈值选择。

## 主指标

E1/E3 使用四项窗口级指标：

- Sensitivity = TP / (TP + FN)
- Precision = TP / (TP + FP)
- Specificity = TN / (TN + FP)
- PR-AUC = 测试窗的连续 SVM 排序分数所对应的 average precision

E4 在上述四项基础上加入：

- Event Sensitivity：一个原始标注 FoG 事件只要其固定测试 FoG 窗中至少一个被检出，即记为检出
- False Alarms/hour：按记录和 NBM allocation group 合并间隔不超过 1 s 的阳性 Non-FoG 窗，除以固定测试 Non-FoG 窗区间并集的小时数

最终主表先在每名被试内平均三折，再计算 8 名被试的均值和样本标准差，避免窗口较多的被试支配结果。

## 运行

```powershell
python scripts/run_private_nbm_tf_svm_all5.py audit --config configs/private_nbm_tf_svm_all5.yaml
python scripts/run_private_nbm_tf_svm_all5.py run --config configs/private_nbm_tf_svm_all5.yaml
```

单被试试跑可增加 `--subject P01 --fold 0`。完整结果写入：

- `outputs/private_nbm_tf_svm_all5/reports/private_tf_svm_all5_30ch_v1/e1_e3_four_metric_main_table.md`
- `outputs/private_nbm_tf_svm_all5/reports/private_tf_svm_all5_30ch_v1/e4_six_metric_main_table.md`
- `outputs/private_nbm_tf_svm_all5/reports/private_tf_svm_all5_30ch_v1/per_subject_averaged_metrics.csv`

每个训练任务还保留模型、阈值、超参数搜索、窗口预测、事件明细和误报警片段，便于复核。
