# Daphnet RF125 1D-CNN replacement experiment

## Question

This suite tests whether the current TCN-M downstream classifier should be
replaced by a 1D-CNN when both classifiers have the same local
convolutional-feature receptive field.

The experiment is a paired `2 classifiers x 8 LOSO folds = 16 cells` design.
Both classifiers are retrained in the same suite; old TCN-M results are not
used as the formal paired reference.

## Locked representation and LOSO protocol

- Normal-behaviour model: `Persistence`
- Classifier input: `residual_h4s`
- Input shape: `[batch, 9, 256]`
- Sampling rate: 64 Hz
- History: 4 seconds, constructed from eight chronological 32-sample residual
  blocks
- LOSO folds: `S01,S02,S03,S05,S06,S07,S08,S09`
- Excluded subjects: `S04,S10`
- Seed: 42; fold seed is `42 + 10000 + fold_index`
- Epoch shuffle seed: `fold_seed + epoch`
- Optimizer: AdamW, learning rate `1e-3`, weight decay `1e-4`
- Loss: `BCEWithLogitsLoss`
- Positive weight:
  `min(sqrt(N_nonFOG_train / N_FOG_train), 6)`
- Batch size: 256
- AMP: enabled
- Training anchor cap: disabled; every common training anchor is used
- Maximum epochs: 12
- Early-stopping patience: 4
- Early-stopping score: validation PR-AUC
- Classification threshold: selected only on validation data by the existing
  balanced-accuracy rule

## Matched classifiers

Both arms use:

- 9-to-48 pointwise input projection, BatchNorm, and GELU
- six temporal blocks
- two `Conv1d-BatchNorm-GELU-Dropout` layers per block
- kernel size 3
- dilation schedule `1,2,4,8,8,8`
- symmetric same zero padding
- global mean and max pooling over all 256 input samples
- the same `96 -> 48 -> 1` classification head

The local receptive field is:

```text
RF = 1 + 2 * (3 - 1) * (1 + 2 + 4 + 8 + 8 + 8)
   = 125 samples
   = 1.953125 seconds at 64 Hz
```

| Arm | Block equation | Parameters | Conv/Linear MACs per window | Extra residual additions |
|---|---|---:|---:|---:|
| `tcn_m` | `x + F(x)` | 89,329 | 21,348,912 | 73,728 |
| `cnn_rf125` | `F(x)` | 89,329 | 21,348,912 | 0 |

The 125-sample value describes each local convolutional feature. The final
mean/max readout still aggregates the complete 256-sample, four-second window.
Consequently, the intended comparison axis is the residual identity path, not
the final window coverage.

## Seven-GPU server command

Run this command from the repository root:

```bash
python -u scripts/start_daphnet_rf125_cnn_replacement_multigpu.py \
  --data-dir "/home/chb/Documents/FOG/fog_classification_framework_base/dataset/1.Daphnet Freezing of Gait Dataset/processed" \
  --source-suite-dir "$PWD/outputs/daphnet_3imu_nbm_5x4_loso_seed42" \
  --output-dir "$PWD/outputs/daphnet_persistence_h4_rf125_cnn_replacement_seed42" \
  --gpus 0-6 \
  --work-folds all \
  --max-retries 2 \
  --launch-delay 2 \
  --audit \
  --seed 42 \
  --classifier-epochs 12 \
  --classifier-patience 4 \
  --classifier-lr 0.001 \
  --weight-decay 0.0001 \
  --batch-size 256 \
  --num-workers 0
```

One GPU completes both classifiers for one fold. With seven GPUs, seven folds
start concurrently and the remaining fold is assigned to the first free GPU.
The launcher initializes and finalizes the shared protocol on CPU.

Do not add `--debug-small-models` to a reportable run.

## Resume

The launcher and fold workers use resume mode automatically. Re-running the
same command with the same output directory:

- skips cells with a validated `DONE.json`;
- resumes interrupted cells from `classifier_last.pt`;
- rejects a changed scientific protocol.

Only one scheduler may own an output directory. If the shell job was stopped
with `Ctrl+Z`, resume that job with `fg` rather than starting a second
scheduler.

## Monitoring

```bash
watch -n 10 \
  "python -m json.tool '$PWD/outputs/daphnet_persistence_h4_rf125_cnn_replacement_seed42/multigpu_status.json'"
```

```bash
tail -f \
  "$PWD/outputs/daphnet_persistence_h4_rf125_cnn_replacement_seed42/multigpu_logs/S01.log"
```

## Single-process interface

```bash
python -u scripts/run_daphnet_rf125_cnn_replacement.py \
  --data-dir "/path/to/processed" \
  --source-suite-dir "/path/to/daphnet_3imu_nbm_5x4_loso_seed42" \
  --output-dir "/path/to/daphnet_persistence_h4_rf125_cnn_replacement_seed42" \
  --folds all \
  --device cuda \
  --seed 42 \
  --batch-size 256 \
  --num-workers 0
```

## Output files

The output root contains:

- `config.json` and `run_manifest.json`: immutable scientific protocol
- `loso_Sxx/tcn_m/` and `loso_Sxx/cnn_rf125/`: checkpoints, predictions,
  metrics, and completion marker
- `fold_summary.csv`: all fold-level classification and event metrics
- `aggregate_summary.csv`: subject-macro mean and standard deviation
- `paired_fold_deltas.csv`: per-subject `cnn_rf125 - tcn_m` deltas
- `aggregate_metrics.json`: subject-macro, pooled, and paired delta summaries
- `experiment_manifest.csv`: expected and completed folds
- `status.json`: expected 16 and completed cell counts
- `AUDIT_REPORT.json`: independent audit evidence
- `SUITE_COMPLETE.json`: written only after all 16 cells pass the full audit

Paired `wins/losses` treat false alarms per hour and detection delay as
lower-is-better; all other reported classification metrics are
higher-is-better. The stored numeric delta always remains `CNN - TCN`.

Primary replacement metrics are Balanced Accuracy, Macro-F1, PR-AUC, FoG
Recall, and FoG F1. Event sensitivity, false alarms per hour, detection delay,
and catastrophic subject folds should also be checked. With only seed 42 and
eight folds, a small difference should be treated as exploratory and confirmed
with multiple seeds before replacing TCN-M.
