# Multimodal FOG Pipeline

This document records the current interface for dataset 4, so the same data
can be regenerated on a local machine or a server without relying on notebook
state.

## Sample-level processed data

Processed root:

```text
dataset/4.Multimodal Dataset/processed/
```

Required files:

- `records/*.npz`: one labeled task record per file.
- `manifest.csv`: record ids, subjects, source files, counts, sampling rate.
- `fog_events.csv`: contiguous FOG event intervals.
- `loso_folds.csv`: record-level LOSO split table.
- `schema.json`: dataset id, channel names, sensor positions, units.
- `preprocessing_report.json`: aggregate counts.

Each record NPZ contains exactly two arrays:

- `x`: `[time, channel] float32`, 24 IMU channels.
- `y_binary`: `[time] int8`, `0=NORMAL`, `1=FOG`.

The 24 channels are four IMU positions:

- `lshank`: accelerometer xyz, gyroscope xyz.
- `rshank`: accelerometer xyz, gyroscope xyz.
- `waist`: accelerometer xyz, gyroscope xyz.
- `arm`: accelerometer xyz, gyroscope xyz.

EEG, EMG, ECG, EOG, NC, and SC columns are excluded from `x`. Some source
subjects only wore two sensors, so missing sensor groups are represented as
all-zero channels and listed in `manifest.csv/all_zero_channels`.

Validate sample-level records:

```powershell
python scripts/validate_processed_records.py `
  "dataset/4.Multimodal Dataset/processed" `
  --expected-channels 24
```

## Window-level training data

Window datasets are generated from the sample-level records. Window length,
stride, pre-FOG duration, and optional target sampling rate are experiment
parameters, not baked into `records/*.npz`.

Binary example:

```powershell
python scripts/prepare_processed_record_windows.py `
  --processed-dir "dataset/4.Multimodal Dataset/processed" `
  --output-dir "dataset/4.Multimodal Dataset/processed_windows_binary_win1_s1_hz100" `
  --window-seconds 1 `
  --stride-seconds 1 `
  --target-hz 100 `
  --label-mode binary `
  --overwrite
```

Equivalent binary config:

```powershell
python scripts/run_fog_experiment.py `
  --config configs/multimodal_sleepyco_binary_win1_hz100.json `
  --only windowing

python scripts/run_fog_experiment.py `
  --config configs/multimodal_sleepyco_binary_win1_hz100.json `
  --only validation
```

Three-class example:

```powershell
python scripts/prepare_processed_record_windows.py `
  --processed-dir "dataset/4.Multimodal Dataset/processed" `
  --output-dir "dataset/4.Multimodal Dataset/processed_windows_3class_win1_s1_hz100_prefog3" `
  --window-seconds 1 `
  --stride-seconds 1 `
  --target-hz 100 `
  --label-mode three-class `
  --pre-fog-seconds 3 `
  --label-rule priority `
  --overwrite
```

Window output files:

- `windows.npz`: `X`, `y`, subject ids/codes, file ids, window starts/ends.
- `loso_folds.npz`: compact fold metadata for training scripts.
- `loso_folds.csv`: fold-level class counts.
- `file_summary.csv`: record-level window counts.
- `config.json`: exact generation parameters.

Validate a window dataset:

```powershell
python scripts/validate_window_dataset.py `
  "dataset/4.Multimodal Dataset/processed_windows_3class_win1_s1_hz100_prefog3" `
  --expected-channels 24 `
  --expected-classes 3
```

## Training smoke test

Small 3-subject window data used for interface testing:

```powershell
python scripts/prepare_processed_record_windows.py `
  --processed-dir "dataset/4.Multimodal Dataset/processed" `
  --output-dir "dataset/4.Multimodal Dataset/processed_windows_smoke_3subjects_hz100" `
  --window-seconds 1 `
  --stride-seconds 1 `
  --target-hz 100 `
  --label-mode three-class `
  --pre-fog-seconds 3 `
  --max-records 12 `
  --overwrite
```

Minimal CPU training check:

```powershell
python scripts/run_sleepyco_fog_two_stage.py `
  --data-dir "dataset/4.Multimodal Dataset/processed_windows_smoke_3subjects_hz100" `
  --output-dir "outputs/multimodal_sleepyco_smoke_3subjects" `
  --stage finetune `
  --baselines seq2one_gru `
  --folds 0 `
  --seq-len 3 `
  --finetune-epochs 1 `
  --finetune-patience 1 `
  --finetune-batch-size 32 `
  --samples-per-epoch 64 `
  --feature-dim 32 `
  --projection-dim 32 `
  --hidden-dim 32 `
  --num-scales 2 `
  --num-layers 1 `
  --device cpu `
  --no-amp `
  --no-load-pretrained `
  --no-resume
```

The smoke test is only an interface check. Its metrics should not be treated as
model performance.

Binary smoke check:

```powershell
python scripts/run_fog_experiment.py `
  --config configs/multimodal_sleepyco_binary_smoke.json
```

This verifies the two-class path end to end: windowing, LOSO validation,
fine-tuning, and binary ROC-AUC/PR-AUC computation.

## Config runner

The same steps can be run from JSON configs:

```powershell
python scripts/run_fog_experiment.py `
  --config configs/multimodal_sleepyco_smoke.json
```

Full-subject 3-class windowing and validation:

```powershell
python scripts/run_fog_experiment.py `
  --config configs/multimodal_sleepyco_3class_win1_hz100_prefog3.json `
  --only windowing

python scripts/run_fog_experiment.py `
  --config configs/multimodal_sleepyco_3class_win1_hz100_prefog3.json `
  --only validation
```

The full config also contains a complete SleePyCo training command for all
LOSO folds. Run it on a server or a suitable local GPU machine:

```powershell
python scripts/run_fog_experiment.py `
  --config configs/multimodal_sleepyco_3class_win1_hz100_prefog3.json `
  --only training
```

The TCN baseline uses the same window dataset interface. Binary smoke:

```powershell
python scripts/run_fog_experiment.py `
  --config configs/multimodal_tcn_binary_smoke.json
```

Full-subject binary TCN training:

```powershell
python scripts/run_fog_experiment.py `
  --config configs/multimodal_tcn_binary_win1_hz100.json `
  --only training
```

Full-subject three-class TCN training:

```powershell
python scripts/run_fog_experiment.py `
  --config configs/multimodal_tcn_3class_win1_hz100_prefog3.json `
  --only training
```

`scripts/run_tcn_loso_npz.py` reads `class_names` and `X.shape[2]` from
`windows.npz`, so it can run both binary and three-class Multimodal windows
without hard-coded 3-channel or 3-class assumptions.

## Result Collection

After one or more experiments finish, collect comparable metrics with:

```powershell
python scripts/collect_fog_results.py `
  outputs/multimodal_sleepyco_binary_smoke_3subjects_config `
  outputs/multimodal_sleepyco_smoke_3subjects_config `
  outputs/multimodal_tcn_binary_smoke_3subjects_config `
  outputs/multimodal_tcn_3class_smoke_3subjects `
  --output-csv outputs/multimodal_smoke_results_summary.csv `
  --output-json outputs/multimodal_smoke_results_summary.json
```

The collector supports both output layouts currently used here:

- SleePyCo: `outputs/<experiment>/<baseline>/aggregate.json`
- TCN: `outputs/<experiment>/aggregate.json`

It writes one row per aggregate result, including trainer, variant, class names,
window settings, channel count, fold count, and aggregate test metrics. The
same command style can be used on server-side full LOSO outputs.

## Result Audit

Use the audit script when you need to verify that a suite actually finished and
produced the expected aggregate outputs. This is separate from preflight:
preflight checks whether a suite is ready to run; audit checks the results after
training.

Smoke audit:

```powershell
python scripts/audit_fog_suite_results.py `
  --config configs/multimodal_smoke_suite.json `
  --output-json outputs/multimodal_smoke_suite_audit.json
```

Full-suite audit after server training:

```powershell
python scripts/audit_fog_suite_results.py `
  --config configs/multimodal_full_suite.json `
  --output-json outputs/multimodal_full_suite_audit.json
```

The audit checks expected aggregate files, `summary.csv` row counts, fold counts,
class names, and input channel counts. For the full Multimodal suite it expects
six aggregate outputs: two SleePyCo variants for binary, two SleePyCo variants
for three-class, and one TCN output for each label mode.

For a non-failing, read-only status view before or during a server run:

```powershell
python scripts/run_fog_suite.py `
  --config configs/multimodal_full_suite.json `
  --status
```

This prints one line for the suite and one line per experiment, including
complete/incomplete state, aggregate counts, expected fold count, and missing
variants. Unlike `audit_fog_suite_results.py`, this status command is meant for
inspection and does not launch any stage.

For automation, write the same completion state as compact JSON:

```powershell
python scripts/run_fog_suite.py `
  --config configs/multimodal_full_suite.json `
  --status-json outputs/multimodal_full_suite_status.json
```

The JSON contains suite-level completion counts plus one entry per experiment
with `state`, `ok`, `expected_folds`, `missing_variants`, and output paths. This
command also exits without running training.

## Experiment Suites

Use a suite config when several experiment configs should be run together and
summarized in one table.

By default, `run_fog_suite.py` deduplicates shared window datasets. If two
experiment configs point to the same window output directory with the same
windowing parameters, windowing and validation run once, then each model's
training stage runs separately. Use `--no-dedupe-windowing` to force the older
per-experiment behavior.

It also reuses existing window outputs by default. If `windows.npz`,
`loso_folds.npz`, and `config.json` already exist and the key windowing
parameters match, windowing is skipped. Use `--no-reuse-existing-windows` to
force regeneration.

Training is also guarded by a result audit by default. If all expected
`aggregate.json` and `summary.csv` outputs already exist and pass audit,
`run_fog_suite.py --only training` skips the training stage instead of launching
the model scripts again. If only some experiments are complete, the suite skips
those completed experiments and launches the remaining ones. Use
`--no-skip-completed-training` to force training commands to run. The launcher
scripts expose the same switch as `-NoSkipCompletedTraining` and
`--no-skip-completed-training`.

Long server runs can be split by experiment name or config stem. The filter is
case-insensitive and matches substrings in the experiment name, config stem, or
description. For example, run only the two TCN full experiments:

```powershell
python scripts/run_fog_suite.py `
  --config configs/multimodal_full_suite.json `
  --only training `
  --include-experiments tcn
```

The launchers expose the same filter as `-IncludeExperiments tcn` and
`--include-experiments tcn`. Use `--exclude-experiments sleepyco` for the inverse
selection.

Preflight before a local or server run:

```powershell
python scripts/preflight_fog_suite.py `
  --config configs/multimodal_full_suite.json `
  --require-windows `
  --output-json outputs/multimodal_full_suite_preflight.json
```

The preflight checks experiment config files, shared window config consistency,
processed source directories, existing `windows.npz` / `loso_folds.npz`, class
counts, channel counts, training output paths, and result collection paths. It
exits non-zero if it finds an error.
If an experiment's `windowing` config sets `require_success: true`, preflight
also requires `processed/_SUCCESS.json` with `status=complete`. The window
materialization script enforces the same marker through `--require-success`, so
an interrupted sample-level processed directory is not used accidentally.
Use `--allow-missing-processed` when you want to audit suite configs before the
sample-level `processed/` directory has been generated; missing processed
directories are then reported as warnings while config/script consistency is
still checked.

Local smoke suite:

```powershell
python scripts/run_fog_suite.py `
  --config configs/multimodal_smoke_suite.json
```

This runs four small checks:

- SleePyCo binary
- SleePyCo three-class
- TCN binary
- TCN three-class

It writes:

```text
outputs/multimodal_smoke_suite_summary.csv
outputs/multimodal_smoke_suite_summary.json
```

Server-oriented full suite:

```powershell
python scripts/run_fog_suite.py `
  --config configs/multimodal_full_suite.json
```

The full suite currently reuses two shared window datasets:

- binary: `processed_windows_binary_win1_s1_hz100`
- three-class: `processed_windows_3class_win1_s1_hz100_prefog3`

If the full window datasets have already been generated and validated, run
only the training stage:

```powershell
python scripts/run_fog_suite.py `
  --config configs/multimodal_full_suite.json `
  --only training
```

Windows/PowerShell launcher:

```powershell
powershell -ExecutionPolicy Bypass `
  -File scripts/start_multimodal_full_suite.ps1 `
  -Only training `
  -RequireWindows
```

PowerShell TCN-only training:

```powershell
powershell -ExecutionPolicy Bypass `
  -File scripts/start_multimodal_full_suite.ps1 `
  -Only training `
  -RequireWindows `
  -IncludeExperiments tcn
```

Dry-run the launcher without starting training:

```powershell
powershell -ExecutionPolicy Bypass `
  -File scripts/start_multimodal_full_suite.ps1 `
  -Only training `
  -RequireWindows `
  -DryRun
```

Linux/Bash launcher:

```bash
bash scripts/start_multimodal_full_suite.sh \
  --only training \
  --require-windows
```

Both launchers run preflight first, write logs to
`outputs/logs/multimodal_full_suite.log`, and then call `run_fog_suite.py`.
For `all`, `training`, and `collection` runs, they also run
`audit_fog_suite_results.py` after the suite unless `-NoAudit` or `--no-audit`
is passed.

After the full suite finishes, it collects results into:

```text
outputs/multimodal_full_suite_summary.csv
outputs/multimodal_full_suite_summary.json
```

Current verified full-subject window output:

```text
dataset/4.Multimodal Dataset/processed_windows_3class_win1_s1_hz100_prefog3/
```

It contains 12,422 windows with shape `[window, 100, 24]`, 12 LOSO folds, and
class counts `NORMAL=6191`, `PRE_FOG=915`, `FOG=5316`.

Current verified binary window output:

```text
dataset/4.Multimodal Dataset/processed_windows_binary_win1_s1_hz100/
```

It contains 12,422 windows with shape `[window, 100, 24]`, 12 LOSO folds, and
class counts `NORMAL=7106`, `FOG=5316`.

Current verified full-subject suite outputs:

```text
outputs/multimodal_sleepyco_binary_win1_hz100/
outputs/multimodal_sleepyco_3class_win1_hz100_prefog3/
outputs/multimodal_tcn_binary_win1_hz100/
outputs/multimodal_tcn_3class_win1_hz100_prefog3/
outputs/multimodal_full_suite_summary.csv
outputs/multimodal_full_suite_summary.json
outputs/multimodal_full_suite_status.json
outputs/multimodal_full_suite_audit.json
outputs/multimodal_tcn_full_summary.csv
outputs/multimodal_tcn_full_summary.json
outputs/multimodal_tcn_full_status.json
```

All full-suite runs contain 12 LOSO folds and a complete `aggregate.json`.
Current aggregate means:

| experiment | variant | f1_macro | accuracy | balanced_accuracy | roc_auc_ovr_macro | pr_auc_macro |
|---|---:|---:|---:|---:|---:|---:|
| SleePyCo binary | `seq2one_gru` | 0.5061 | 0.6336 | 0.5767 | 0.6722 | 0.4660 |
| SleePyCo binary | `seq2seq_gru` | 0.5310 | 0.6454 | 0.5919 | 0.6817 | 0.4669 |
| TCN binary | `tcn` | 0.5112 | 0.6194 | 0.5952 | 0.7622 | 0.4654 |
| SleePyCo three-class | `seq2one_gru` | 0.3415 | 0.5106 | 0.4068 | 0.6027 | 0.4265 |
| SleePyCo three-class | `seq2seq_gru` | 0.3496 | 0.5318 | 0.4027 | 0.5954 | 0.4094 |
| TCN three-class | `tcn` | 0.2136 | 0.3588 | 0.3400 | 0.6538 | 0.4328 |

Current full suite status:

```text
outputs/multimodal_full_suite_status.json
```

The full suite is complete: 6 of 6 expected aggregate outputs are present, the
audit reports no errors, and each experiment has 12 LOSO folds.
