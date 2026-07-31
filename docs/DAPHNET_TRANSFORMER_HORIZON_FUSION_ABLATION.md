# Daphnet Transformer horizon-by-fusion ablation

## Research question

This suite tests whether the diagnostic value of a Transformer normal-behaviour
error changes when the NBM prediction horizon is increased from 0.5 seconds to
1 or 2 seconds. It also tests whether any apparent gain is explained by extra
raw context or by the larger 18-channel classifier input.

The six requested horizon-specific arms are:

| Classifier input | H050 (0.5 s) | H100 (1.0 s) | H200 (2.0 s) |
|---|---:|---:|---:|
| Error, `x - mu_H` | yes | yes | yes |
| Raw4 + Error | yes | yes | yes |

Three shared controls are trained once per fold:

| Control | Shape | Purpose |
|---|---:|---|
| Raw4 | `[9,256]` | matched four-second raw baseline |
| Raw6 | `[9,384]` | raw union of the two-second NBM context and four-second diagnostic history |
| Raw4 + Zero | `[18,256]` | channel-count and capacity control for Raw4 + Error |

This produces `9 configurations x 8 LOSO folds = 72` reportable classifier
cells per seed.

## Fixed protocol

- Daphnet at 64 Hz with ankle, thigh, and trunk acceleration: nine channels
- LOSO subjects `S01,S02,S03,S05,S06,S07,S08,S09`; S04 and S10 excluded
- Transformer-NBM context: 2 seconds
- horizons: H050=32, H100=64, and H200=128 samples
- four-second diagnostic history
- TCN-M classifier with dilations `1,2,4,8,8,8`, receptive field 125 samples
- 0.25-second anchor and classifier-output stride
- seed 42 unless overridden explicitly
- validation-only early stopping and threshold selection
- test subject never contributes to scaling, fitting, early stopping, model
  selection, or threshold selection

The Transformer-NBM is fitted only with clean-normal training windows. A master
window table uses the complete two-second context, maximum two-second forecast,
and normal guard to define validity and clean-normal eligibility for every
horizon. Shorter horizons therefore cannot obtain additional NBM or classifier
samples.

## Common temporal support

All arms share endpoints, terminal 0.5-second labels, and train/validation/test
anchors. The horizon-specific histories contain non-overlapping forecast
blocks:

| Horizon | Blocks in four seconds | Error input |
|---|---:|---:|
| H050 | 8 | `[9,256]` |
| H100 | 4 | `[9,256]` |
| H200 | 2 | `[9,256]` |

The binary label always comes from the final 0.5 seconds at the common endpoint.
It does not expand to one or two seconds when the NBM horizon increases.

`Raw4` is the same terminal four-second observed signal for every horizon.
`Raw6` covers the full causal raw support used to construct the four-second
error history: the two seconds preceding the earliest target plus the terminal
four seconds. It is a context-information control, not a parameter-matched
compute control.

## Primary comparisons

For each horizon H:

1. `Raw4+Error_H - Raw4+Zero`: error information beyond 18-channel capacity.
2. `Raw4+Error_H - Raw4`: practical fusion gain over the four-second raw input.
3. `Raw4+Error_H - Raw6`: gain beyond exposing the classifier to the complete
   six-second raw support.
4. `Error_H - Raw4`: whether error can replace raw input.

Across horizons:

- `Error_H100 - Error_H050`
- `Error_H200 - Error_H050`
- `Raw4+Error_H100 - Raw4+Error_H050`
- `Raw4+Error_H200 - Raw4+Error_H050`

The primary metric is held-out-subject macro PR-AUC. Every comparison reports a
paired subject-level 95% bootstrap confidence interval and wins/ties/losses.
Secondary window metrics are Accuracy, Balanced Accuracy, Macro-F1, AUROC, FoG
Recall/Sensitivity, Specificity, FoG Precision, FoG F1, and MCC. Event-level
metrics are Event Sensitivity, false-alarm events per hour, and median detection
delay.

## Seven-GPU server command

Run from the repository root:

```bash
python -u scripts/start_daphnet_transformer_horizon_fusion_ablation_multigpu.py \
  --data-dir "/home/chb/Documents/FOG/fog_classification_framework_base/dataset/1.Daphnet Freezing of Gait Dataset/processed" \
  --output-dir "$PWD/outputs/daphnet_transformer_horizon_fusion_h4_tcnm_loso_seed42" \
  --gpus 0-6 \
  --work-folds all \
  --max-retries 2 \
  --launch-delay 2 \
  --audit \
  --seed 42 \
  --batch-size 256 \
  --num-workers 0 \
  --amp \
  --deterministic
```

The launcher initializes the immutable protocol on CPU. Seven folds start on
GPUs 0-6, and the first free GPU runs the eighth fold. A fold worker owns one
GPU while it trains its three Transformer NBMs, materializes all inputs, and
trains the nine TCN-M classifiers. Finalization and the independent audit run
only after all folds complete.

## Monitoring

```bash
watch -n 10 \
  "python -m json.tool '$PWD/outputs/daphnet_transformer_horizon_fusion_h4_tcnm_loso_seed42/multigpu_status.json'"
```

```bash
tail -f \
  "$PWD/outputs/daphnet_transformer_horizon_fusion_h4_tcnm_loso_seed42/multigpu_logs/S01.log"
```

```bash
watch -n 2 nvidia-smi
```

## Resume

Re-run the exact same command with the same output directory. The launcher
always forwards `--resume`; validated completed tasks are skipped and
interrupted checkpoints are reused. Do not start a second scheduler for the
same output directory. If the shell was suspended with `Ctrl+Z`, recover it
with `jobs -l` and `fg %<job-number>`.

## Expected outputs

- `config.json`, `run_manifest.json`, and `experiment_manifest.csv`
- fold-local master support, Transformer checkpoints, error caches, classifier
  checkpoints, predictions, metrics, and DONE manifests
- `fold_summary.csv`
- `aggregate_summary.csv`
- `paired_pr_auc_deltas.csv`
- `publication_table.csv`
- `aggregate_metrics.json`
- `status.json` and `RESULTS_DONE.json`
- `multigpu_status.json` and `multigpu_logs/`
- `AUDIT_REPORT.json`, `AUDIT_REPORT.txt`, and `SUITE_COMPLETE.json` after a
  successful complete audit

Do not rank models solely by their mean PR-AUC. A longer horizon is useful only
if the paired gain is stable and is not purchased with unacceptable FoG recall,
false-alarm, or delay degradation.
