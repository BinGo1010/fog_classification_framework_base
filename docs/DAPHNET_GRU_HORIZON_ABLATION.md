# Daphnet GRU-NBM prediction-horizon ablation

## Experiment contract

This suite isolates the effect of the GRU normal-behaviour model's prediction
horizon. It contains `4 horizons x 8 LOSO folds = 32` reportable classifier
cells.

| Horizon | GRU-NBM horizon | Horizon samples | 4 s history blocks | Classifier input |
|---|---:|---:|---:|---:|
| H025 | 0.25 s | 16 | 16 | `[9,256]` |
| H050 | 0.50 s | 32 | 8 | `[9,256]` |
| H100 | 1.00 s | 64 | 4 | `[9,256]` |
| H200 | 2.00 s | 128 | 2 | `[9,256]` |

The following conditions remain fixed:

- Daphnet three-IMU input with nine acceleration channels at 64 Hz
- LOSO subjects `S01,S02,S03,S05,S06,S07,S08,S09`; S04 and S10 excluded
- GRU-NBM context of 2 seconds
- four-second uncertainty-standardized residual history
- 0.25-second base anchor and classifier-output stride
- TCN-M downstream classifier
- common train, validation, and test anchors across all horizon variants
- window label from the terminal 0.5-second target, irrespective of the NBM
  prediction horizon
- seed 42 and the same classifier initialization/training policy within a fold

The common-anchor rule is essential: a longer prediction horizon must not gain
or lose classifier samples merely because its raw forecast window is longer.
Likewise, H100 and H200 remain binary decisions about the same terminal
0.5-second FoG label; they are not one- or two-second labels.

One master WindowTable uses the maximum two-second horizon to define validity
and clean-normal eligibility for every arm. Concretely, the same two-second
context, maximum two-second forecast support, and 0.5-second normal guard are
applied before the per-horizon windows are derived and right-aligned to their
shared endpoints. Thus no shorter-horizon arm receives extra clean-normal NBM
training rows.

The comparison keeps the GRU encoder and summary architecture and their initial
values shared. The direct Gaussian decoder is the unavoidable
horizon-dependent component: its output width, decoder parameter count, and
total GRU parameter count increase as the horizon grows. These counts are
recorded per horizon in `config.json` (`gru_architectures`), and total GRU
parameter counts are also exposed in `aggregate_summary.csv`; they must be
reported with the ablation rather than described as parameter-matched models.

## Reported metrics

The root summaries report:

- PR-AUC
- ΔPR-AUC versus H050 with a paired subject-level 95% bootstrap confidence
  interval
- Balanced Accuracy (BA)
- Macro-F1
- AUROC
- FoG Recall/Sensitivity
- Specificity
- FoG Precision
- FoG F1
- Event Sensitivity
- false-alarm events per hour (FA/h)
- median detection delay

`aggregate_summary.csv` and the fold-level artifacts additionally retain
Accuracy and MCC for completeness.

## Seven-GPU server command

Run from the repository root:

```bash
python -u scripts/start_daphnet_gru_horizon_ablation_multigpu.py \
  --data-dir "/home/chb/Documents/FOG/fog_classification_framework_base/dataset/1.Daphnet Freezing of Gait Dataset/processed" \
  --output-dir "$PWD/outputs/daphnet_gru_horizon4_h4_tcnm_loso_seed42" \
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

The launcher initializes the shared protocol on CPU and then treats one fold
as the indivisible GPU job. Each worker completes H025, H050, H100, and H200
sequentially for its fold. Seven folds start on GPUs 0-6; the first free GPU
then runs the eighth fold. Root summaries and the independent audit run only
after all selected fold workers finish.

Unknown scientific arguments are forwarded unchanged to initialization, every
fold worker, and finalization. If `--seed` is omitted, this launcher explicitly
adds `--seed 42`.

## Monitoring

Scheduler state:

```bash
watch -n 10 \
  "python -m json.tool '$PWD/outputs/daphnet_gru_horizon4_h4_tcnm_loso_seed42/multigpu_status.json'"
```

One fold's complete log:

```bash
tail -f \
  "$PWD/outputs/daphnet_gru_horizon4_h4_tcnm_loso_seed42/multigpu_logs/S01.log"
```

GPU utilization:

```bash
watch -n 2 nvidia-smi
```

## Resume and recovery

Re-run the exact same command with the same output directory. Resume is always
enabled by the launcher, completed tasks are validated and skipped, and
interrupted checkpoints are reused. A failed fold is retried up to
`--max-retries`; after the scheduler exits, the same command can resume folds
that exhausted their retries.

Only one scheduler may own an output directory. If the shell was suspended
with `Ctrl+Z`, use `jobs -l` and `fg %<job-number>` instead of launching a
second scheduler. The lock implementation reclaims only a demonstrably stale
same-host lock and refuses reclamation while recorded child processes remain
alive.

## Expected outputs

- `multigpu_status.json`: scheduler stage, fold, GPU, PID, retry, and audit state
- `multigpu_logs/initialize.log`, `Sxx.log`, `finalize.log`, and `audit.log`
- `config.json` and `run_manifest.json`: immutable protocol and fingerprint
- fold-local NBM checkpoints, residual caches, classifier checkpoints,
  predictions, metrics, and completion manifests for H025-H200
- `fold_summary.csv`: all 32 fold-level results
- `experiment_manifest.csv`, `aggregate_summary.csv`,
  `paired_pr_auc_deltas.csv`, `publication_table.csv`, and
  `aggregate_metrics.json`
- `status.json`: expected/completed experiment counts
- `RESULTS_DONE.json`: runner completion manifest after all 32 cells finish
- `AUDIT_REPORT.json` and `SUITE_COMPLETE.json` after a successful full audit

The primary ranking is subject-macro PR-AUC. All metrics above remain in the
publication and aggregate tables so a higher PR-AUC cannot conceal a FoG
recall, specificity, false-alarm, or delay trade-off.
