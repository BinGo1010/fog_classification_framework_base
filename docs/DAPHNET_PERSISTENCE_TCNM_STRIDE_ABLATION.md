# Daphnet Persistence + TCN-M stride ablation

This suite compares three deployment schedules while fixing the frozen
Persistence NBM, residual definition, four-second representation, TCN-M
classifier, LOSO splits, and training protocol. It contains
`3 variants x 8 folds = 24` classifier cells.

| Variant | NBM prediction stride | Classifier output stride | Purpose |
|---|---:|---:|---|
| S1 | 0.25 s / 16 samples | 0.5 s / 32 samples | Reduce only decision frequency |
| S2 | 0.25 s / 16 samples | 1.0 s / 64 samples | Test very-low-frequency decisions |
| S3 | 0.5 s / 32 samples | 0.5 s / 32 samples | Remove unused overlapping predictor calls |

All arms use:

- frozen canonical Persistence checkpoint, uncertainty, and residual cache;
- `residual_h4s` input with shape `[batch, 9, 256]`;
- eight chronological, non-overlapping 32-sample residual blocks;
- TCN-M dilations `1,2,4,8,8,8` and receptive field 125 samples;
- seed 42 and folds `S01,S02,S03,S05,S06,S07,S08,S09`;
- S04 and S10 excluded.

The NBM horizon and classification target stay fixed at 0.5 seconds. The
classifier stride changes only how often a decision is emitted; S2 does not
use a new one-second label. Its window-level target-time coverage is therefore
50%. For event metrics, intentional time between scheduled S2 outputs remains
monitored, so a FoG event entirely between outputs is counted and may be
missed. A genuinely missing scheduled anchor breaks event coverage.
The existing event post-processing rule still requires two positive outputs:
its minimum confirmation span is 1.0 s for S1/S3 and 1.5 s for S2. These
values are exported explicitly and must be reported when interpreting Event
Sensitivity, FA/h, or detection delay.

Because `residual_h4s` consumes horizon-spaced 0.5-second blocks, the extra
phase of S1's 0.25-second predictor grid never enters the classifier at the
0.5-second decision anchors. S1 and S3 must consequently have identical
classifier tensors and deterministic predictions. The scientific result of
S3 is whether NBM calls can be halved with exactly zero diagnostic change.

## Seven-GPU server command

Run from the repository root:

```bash
python -u scripts/start_daphnet_persistence_tcnm_stride_ablation_multigpu.py \
  --data-dir "/home/chb/Documents/FOG/fog_classification_framework_base/dataset/1.Daphnet Freezing of Gait Dataset/processed" \
  --source-suite-dir "$PWD/outputs/daphnet_3imu_nbm_5x4_loso_seed42" \
  --output-dir "$PWD/outputs/daphnet_persistence_h4_tcnm_stride3_loso_seed42" \
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
  --max-classifier-windows 0 \
  --bootstrap-samples 100000 \
  --bootstrap-seed 42 \
  --num-workers 0 \
  --amp \
  --deterministic
```

The launcher initializes the protocol on CPU, then assigns one complete fold
to each GPU. A fold worker loads the frozen source cache and trains or resumes
S1, S2, and S3 sequentially. Seven folds start first; the eighth is assigned
to the first free GPU. Finalization and the independent audit run after all
workers finish.

Scientific options are forwarded unchanged to the runner. If `--seed` is
omitted, the launcher adds `--seed 42`; the strict runner rejects another seed.

## Monitoring

```bash
OUTPUT_DIR="$PWD/outputs/daphnet_persistence_h4_tcnm_stride3_loso_seed42"

watch -n 10 "python -m json.tool '$OUTPUT_DIR/multigpu_status.json'"
tail -f "$OUTPUT_DIR/multigpu_logs/S01.log"
watch -n 2 nvidia-smi
```

## Resume

Re-run the exact same command with the same output directory. Completed
classifier and stride-metadata stages are validated and skipped; interrupted
classifiers resume from `classifier_last.pt`. Failed fold workers are retried
up to `--max-retries`.

Only one scheduler may own an output directory. If it was suspended with
`Ctrl+Z`, use `jobs -l` and `fg %<job-number>`. Do not remove the lock while
the recorded scheduler is alive.

## Main outputs

- `multigpu_status.json` and `multigpu_logs/`: scheduling and complete logs
- `loso_Sxx/{s1,s2,s3}/`: checkpoints, predictions, metrics, `DONE.json`,
  `stride_metadata.json`, and `STRIDE_METADATA_DONE.json`
- `fold_summary.csv`: all fold-level window and event metrics
- `aggregate_summary.csv`: subject-macro mean and standard deviation
- `paired_pr_auc_deltas.csv`: paired S2/S3 minus S1 PR-AUC with 95% CI
- `publication_table.csv`: concise report-ready table
- `efficiency_summary.csv`: call rates, coverage, and window counts
- `stride_equivalence.json`: exact S1/S3 equivalence evidence
- `aggregate_metrics.json` and `status.json`: machine-readable summaries
- `AUDIT_REPORT.json`: independent audit evidence
- `SUITE_COMPLETE.json`: written only after all 24 cells pass the audit

The `0.5` S3/S1 predictor-call ratio in `stride_equivalence.json` is the
steady-state nominal ratio. Use the actual finite-record window totals in
`efficiency_summary.csv` when reporting measured call counts.
