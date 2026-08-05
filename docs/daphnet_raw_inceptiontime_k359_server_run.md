# Daphnet Raw + InceptionTime [3, 5, 9] server run

This runner trains only B0 Raw-InceptionTime. It reuses the frozen outer splits and
scalers from the completed TCN-DAE/InceptionTime experiment, launches one process per
GPU, checkpoints every epoch, retries failed workers, and skips completed runs.

## Files to copy to the server

- `scripts/run_daphnet_full_subject_raw_inceptiontime_k359.py`
- `configs/daphnet_full_subject_raw_inceptiontime_k359.yaml`

The existing project files and the completed source experiment must remain available.

## Optional one-fold smoke test

Use a separate smoke-test output directory so that the two-epoch smoke checkpoint is
never mistaken for a production result.

```bash
python scripts/run_daphnet_full_subject_raw_inceptiontime_k359.py \
  --smoke \
  --device cuda:0 \
  --threads 2 \
  --source-root outputs/daphnet_full_subject_tcndae_inceptiontime_server_v1/full_subject_binary_experiment \
  --output-root outputs/daphnet_full_subject_raw_inceptiontime_k359_smoke/full_subject_binary_experiment
```

## Seven-GPU production run

```bash
python scripts/run_daphnet_full_subject_raw_inceptiontime_k359.py \
  --launch-parallel \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6 \
  --threads 2 \
  --source-root outputs/daphnet_full_subject_tcndae_inceptiontime_server_v1/full_subject_binary_experiment \
  --output-root outputs/daphnet_full_subject_raw_inceptiontime_k359_server_v1/full_subject_binary_experiment
```

Rerun the identical command after an interruption. A completed method/subject/fold/seed
run is skipped when both `run_metrics.json` and `test_predictions.csv` exist. An
incomplete run resumes from `inceptiontime_resume.pt`, including the model, optimizer,
best checkpoint, early-stopping counters, and data-loader generator state.

## Status

```bash
python scripts/run_daphnet_full_subject_raw_inceptiontime_k359.py \
  --status-only \
  --output-root outputs/daphnet_full_subject_raw_inceptiontime_k359_server_v1/full_subject_binary_experiment
```

The complete experiment contains 90 runs: 30 outer folds times 3 seeds.

## Finalize without retraining

Use this only if all workers finished but report aggregation was interrupted.

```bash
python scripts/run_daphnet_full_subject_raw_inceptiontime_k359.py \
  --finalize-only \
  --output-root outputs/daphnet_full_subject_raw_inceptiontime_k359_server_v1/full_subject_binary_experiment
```

Final results are written to:

- `FINAL_RESULTS.json`
- `reports/raw_inceptiontime_k359_report.md`
- `tables/all_subject_summary.csv`
- `tables/subject_level_main_results.csv`
- `tables/pooled_window_metrics.csv`
- `predictions/seed_median_pooled_predictions.csv`

