# FOG Dataset Training Framework

This file is the quick index for the standardized local/server training entry
points. The common flow is:

1. Sample-level records live in `processed/` and contain only `x` and
   `y_binary`.
2. `scripts/prepare_processed_record_windows.py` materializes LOSO window
   datasets from `processed/`.
3. `scripts/run_fog_experiment.py` runs one JSON experiment.
4. `scripts/run_fog_suite.py` runs a suite of experiments, reusing shared
   windows and collecting results.
5. `scripts/preflight_fog_suite.py` checks configs and existing inputs before a
   long run.
6. `scripts/audit_fog_suite_results.py` checks completed outputs after training.

## Status Matrix

| dataset | local smoke | server/full entry | current evidence |
|---|---|---|---|
| `4.Multimodal Dataset` | verified | verified and full trained | `outputs/multimodal_full_suite_status.json` is complete, 6/6 aggregates |
| `2.Kaggle Parkinson’s Freezing of Gait Prediction` | preflight verified | preflight verified | ZIP and configs pass; wait for `processed/_SUCCESS.json` before training |
| `1.Daphnet Freezing of Gait Dataset` | verified | config/preflight verified | smoke suite trained TCN + SleePyCo; full suite config exists |
| `5.Stanford imu-fog-detection/imus6_subjects7` | verified | config/preflight verified | smoke suite trained TCN + SleePyCo; full suite config exists |
| `5.Stanford imu-fog-detection/imus11_subjects4` | verified | config/preflight verified | smoke suite trained TCN + SleePyCo; full suite config exists |
| `6.FoG-STAR` | verified | config/preflight verified | smoke suite trained TCN + SleePyCo; full suite config exists; NaNs are zero-filled during windowing |

## Config Index

Multimodal:

```text
configs/multimodal_smoke_suite.json
configs/multimodal_full_suite.json
configs/multimodal_sleepyco_binary_win1_hz100.json
configs/multimodal_sleepyco_3class_win1_hz100_prefog3.json
configs/multimodal_tcn_binary_win1_hz100.json
configs/multimodal_tcn_3class_win1_hz100_prefog3.json
```

Kaggle:

```text
configs/kaggle_smoke_suite.json
configs/kaggle_full_suite.json
configs/kaggle_sleepyco_binary_smoke.json
configs/kaggle_tcn_binary_smoke.json
configs/kaggle_sleepyco_binary_win1_hz100.json
configs/kaggle_tcn_binary_win1_hz100.json
```

Daphnet:

```text
configs/daphnet_smoke_suite.json
configs/daphnet_full_suite.json
configs/daphnet_sleepyco_binary_smoke.json
configs/daphnet_tcn_binary_smoke.json
configs/daphnet_sleepyco_binary_win1_hz64.json
configs/daphnet_tcn_binary_win1_hz64.json
```

Stanford 6 IMUs:

```text
configs/stanford_imus6_smoke_suite.json
configs/stanford_imus6_full_suite.json
configs/stanford_imus6_sleepyco_binary_smoke.json
configs/stanford_imus6_tcn_binary_smoke.json
configs/stanford_imus6_sleepyco_binary_win1_hz128.json
configs/stanford_imus6_tcn_binary_win1_hz128.json
```

Stanford 11 IMUs:

```text
configs/stanford_imus11_smoke_suite.json
configs/stanford_imus11_full_suite.json
configs/stanford_imus11_sleepyco_binary_smoke.json
configs/stanford_imus11_tcn_binary_smoke.json
configs/stanford_imus11_sleepyco_binary_win1_hz128.json
configs/stanford_imus11_tcn_binary_win1_hz128.json
```

FoG-STAR:

```text
configs/fogstar_smoke_suite.json
configs/fogstar_full_suite.json
configs/fogstar_sleepyco_binary_smoke.json
configs/fogstar_tcn_binary_smoke.json
configs/fogstar_sleepyco_binary_win1_hz60.json
configs/fogstar_tcn_binary_win1_hz60.json
```

## Local Smoke Runs

Use smoke suites for quick local verification. They generate small or otherwise
lightweight window datasets and train only fold 0 with short CPU settings.

```powershell
python scripts/run_fog_suite.py --config configs/daphnet_smoke_suite.json
python scripts/run_fog_suite.py --config configs/fogstar_smoke_suite.json
python scripts/run_fog_suite.py --config configs/stanford_imus6_smoke_suite.json
python scripts/run_fog_suite.py --config configs/stanford_imus11_smoke_suite.json
python scripts/run_fog_suite.py --config configs/multimodal_smoke_suite.json
```

Kaggle uses a launcher because it streams from the large competition ZIP:

```powershell
python scripts/start_kaggle_smoke_pipeline.py --execute --overwrite
```

## Server Runs

Use full suites for long local GPU runs or server runs:

```powershell
python scripts/run_fog_suite.py --config configs/daphnet_full_suite.json
python scripts/run_fog_suite.py --config configs/fogstar_full_suite.json
python scripts/run_fog_suite.py --config configs/stanford_imus6_full_suite.json
python scripts/run_fog_suite.py --config configs/stanford_imus11_full_suite.json
python scripts/run_fog_suite.py --config configs/multimodal_full_suite.json
```

Kaggle full execution also goes through the streaming launcher:

```powershell
python scripts/start_kaggle_full_pipeline.py --execute --overwrite
```

For a server run where windows already exist and only training should resume:

```powershell
python scripts/run_fog_suite.py --config configs/<suite>.json --only training
```

## Checks

Before a full run:

```powershell
python scripts/preflight_fog_suite.py --config configs/<suite>.json --output-json outputs/<suite>_preflight.json
```

For suites whose window datasets already exist, add `--require-windows`.

After training:

```powershell
python scripts/audit_fog_suite_results.py --config configs/<suite>.json --output-json outputs/<suite>_audit.json
```

To inspect completion without running anything:

```powershell
python scripts/run_fog_suite.py --config configs/<suite>.json --status
```

Kaggle status:

```powershell
python scripts/kaggle_fog_status.py --dataset-root dataset --output-json outputs/kaggle_status.json
```

## Notes

- FoG-STAR preserves sensor NaNs at sample level. All FoG-STAR window configs
  use `nan_policy: zero`.
- Stanford is split into separate 6-IMU and 11-IMU suites because their channel
  counts differ.
- Daphnet, Stanford, FoG-STAR, and Kaggle configs currently target binary
  `NORMAL/FOG`. Multimodal has both binary and dynamic three-class
  `NORMAL/PRE_FOG/FOG` configs.
