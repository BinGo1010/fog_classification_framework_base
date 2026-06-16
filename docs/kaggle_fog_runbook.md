# Kaggle FOG Runbook

This runbook is for `2.Kaggle Parkinson's Freezing of Gait Prediction`.

The competition zip is large because most bytes are `unlabeled/*.parquet`.
Do not fully extract the zip. The supervised train CSV files are read directly
from the archive. The streaming preprocessor reads each selected train CSV in
chunks and buffers only the currently open retained segment before writing a
record NPZ.

## Zip Inventory Snapshot

The archive should be inspected through the zip central directory, not by
extracting it:

```bash
python scripts/inspect_kaggle_fog_zip.py --dataset-root dataset
```

The CSV inventory includes one row per zip member with `group` and
`path_bucket`; the JSON summary includes matching `groups` and `path_buckets`
aggregates so train/test/metadata structure can be checked without extracting
any files.

The current supervised structure is:

| Group | Path | Files | Compressed | Uncompressed |
|---|---|---:|---:|---:|
| `tdcsfog` | `train/tdcsfog/*.csv` | 833 | 196,837,078 B / 0.183 GiB | 455,059,350 B / 0.424 GiB |
| `tdcsfog` | `test/tdcsfog/*.csv` | 1 | 126,297 B | 268,994 B |
| `defog` | `train/defog/*.csv` | 91 | 314,585,687 B / 0.293 GiB | 998,899,978 B / 0.930 GiB |
| `defog` | `test/defog/*.csv` | 1 | 7,451,421 B / 0.007 GiB | 17,643,907 B / 0.016 GiB |
| `metadata` | zip root `*.csv` metadata files | 6 | 79,133 B | 241,736 B |

The full supervised preprocessing path selects only `train/tdcsfog/*.csv` and
`train/defog/*.csv`, which is 924 train CSV files. It skips `test/`,
`train/notype/`, `sample_submission.csv`, and `unlabeled/*.parquet`. This is
why the 58.460 GiB zip can be processed safely without extracting the 63.651
GiB of auxiliary uncompressed content.

## Safety Levels

### Level 0: Preflight

Does not create `processed/` or `processed_smoke/`.

```bash
python scripts/check_kaggle_fog_preflight.py --dataset-root dataset
```

For server/CI logs, also write a structured report:

```bash
python scripts/check_kaggle_fog_preflight.py \
  --dataset-root dataset \
  --output-json outputs/kaggle_preflight_report.json
```

By default, preflight validates the smoke suite config. For a full-run report,
pass the full suite explicitly:

```bash
python scripts/check_kaggle_fog_preflight.py \
  --dataset-root dataset \
  --suite-config configs/kaggle_full_suite.json \
  --output-json outputs/kaggle_preflight_report.json
```

Relative `--suite-config` paths are resolved from `--repo-root`, not from the
current shell directory.

Preflight is fail-safe for disk space: if the storage estimate reports
insufficient free space, the preflight command exits non-zero and records the
failure in the JSON report. Use `--allow-insufficient-storage` only when you
intentionally want a report without a blocking exit code.

For a quick read-only status view from the latest reports:

```bash
python scripts/kaggle_fog_status.py \
  --dataset-root dataset \
  --output-json outputs/kaggle_status.json
```

The status report includes both smoke and full dry-run reports when present.
It also checks report zip fingerprints against the current archive; stale
preflight or dry-run reports will not be treated as ready-to-execute evidence.
When the preflight report includes `suite_preflight`, status also reports its
`ok` flag plus warning/error counts, and a failed suite preflight blocks
ready-to-execute recommendations.
For `processed/` and `processed_smoke/`, it reports whether `_SUCCESS.json`
exists, whether the directory is complete or partial, and how many
`records/*.npz` files are present. A partial directory is never treated as
suite-ready.
Ready-to-execute recommendations are suite-specific: a full preflight only
enables the full execute recommendation, and a smoke preflight only enables the
smoke execute recommendation.

This runs:

```text
py_compile
extracted competition-data directory report, if present
zip central-directory inventory
supervised zip structure check for train/tdcsfog, train/defog, and metadata
supervised storage budget estimate
streaming dry-run with CSV header checks
suite preflight with missing processed directories treated as warnings
synthetic Kaggle tests
processed-output creation guard
```

The PowerShell preflight also accepts `-OutputJson` and writes a structured
report with the same core sections: zip structure, storage estimate, streaming
dry-run, suite preflight, suite dry-run, steps, and processed-output guard.
`zip_structure.ok=false` blocks status ready-to-execute recommendations when
present in a preflight report.

The storage estimate is conservative and reads only zip metadata plus suite
configs:

```bash
python scripts/estimate_kaggle_fog_storage.py \
  --dataset-root dataset \
  --source both \
  --suite-config configs/kaggle_full_suite.json
```

For a local smoke budget, apply the same per-source limit used by smoke
preprocessing:

```bash
python scripts/estimate_kaggle_fog_storage.py \
  --dataset-root dataset \
  --source both \
  --smoke-limit 5 \
  --suite-config configs/kaggle_smoke_suite.json
```

### Level 1: Smoke Processed

Creates a small `processed_smoke/` sample-level dataset.

Server-friendly launchers are available and default to safe dry-run mode:

```bash
python scripts/start_kaggle_smoke_pipeline.py
```

```bash
bash scripts/start_kaggle_smoke_pipeline.sh
```

```powershell
.\scripts\start_kaggle_smoke_pipeline.ps1
```

These run preflight, streaming smoke dry-run, and suite dry-run. They do not
create `processed_smoke/` unless `--execute` / `-Execute` is explicitly passed.
The smoke launcher passes its `--smoke-limit` / `-SmokeLimit` into preflight, so
the storage estimate, preflight streaming dry-run, launcher dry-run, and status
gate all use the same per-source limit. Launcher-invoked preflight uses
`--skip-pytest`; run `check_kaggle_fog_preflight.py` directly when you want the
standalone synthetic test step.
The Python launcher streams child-process output to
`outputs/logs/kaggle_smoke_pipeline.log` as commands run.
Python and Bash launchers accept `--log-path`, `--preflight-json`,
`--dry-run-json`, and `--status-json`; the PowerShell launcher accepts
`-LogPath`, `-PreflightPath`, `-DryRunReportPath`, and `-StatusPath`.
Use these when running isolated CI/server probes so stale status files cannot
be confused with the current run.
Add `--profile-data` / `-ProfileData` to launcher dry-runs when the log and
dry-run JSON should include NaN/non-finite and label-count diagnostics.
If `--profile-data` is used on an execute path, the status gate also requires
retained rows to have no non-finite sensor values and no invalid or non-binary
labels.
The real preprocessing path also validates retained features and labels before
writing each record, so `processed*/records/*.npz` cannot contain non-finite
`x` values or labels outside `0/1`.
After execute preprocessing, launchers validate the sample-level records with
`validate_processed_records.py --require-success`. Add
`--post-check-window-dry-run` / `-PostCheckWindowDryRun` when you also want the
launcher to run `check_processed_pipeline.py`: this performs the same records
validation plus a window dry-run without writing `windows.npz`.
When `--execute` is used, the launchers require preflight unless
`--allow-execute-without-preflight` is also explicitly supplied for manual
recovery.
Before real preprocessing starts, the execute path also runs the matching
streaming dry-run and `kaggle_fog_status.py --require-ready smoke`; bypassing
that final gate requires `--allow-execute-without-status-gate`.
When `--resume` or `--overwrite` is used, the gate allows the target processed
directory to exist while still checking preflight, suite, zip fingerprint, and
dry-run status.

```bash
python scripts/preprocess_kaggle_fog_streaming.py \
  --dataset-root dataset \
  --source both \
  --valid-only \
  --task-only \
  --strict-metadata \
  --smoke-limit 5
```

With `--smoke-limit`, the default output directory is `processed_smoke/`.
The limit is applied per source, so `--source both --smoke-limit 5` processes
up to 5 `defog` CSV files and 5 `tdcsfog` CSV files.

Records are compressed by default. After the storage estimate passes, use
`--record-compression none` when faster server-side preprocessing matters more
than minimizing processed disk size.

Use `--check-headers --strict-metadata --dry-run` to validate selected CSV
headers and LOSO subject metadata without reading data rows or creating output
directories. Add `--dry-run-output-json outputs/kaggle_streaming_dry_run.json`
when the dry-run diagnostics should be preserved in server logs.
To check NaN/non-finite sensor or label values plus binary label counts before
creating records, add `--profile-data`. This streams the selected CSV data rows
but still does not create `processed*/` outputs. The report also includes
retained, NORMAL, and FOG durations in seconds, computed with each source's
native sampling rate.

```bash
python scripts/preprocess_kaggle_fog_streaming.py \
  --dataset-root dataset \
  --source both \
  --valid-only \
  --task-only \
  --strict-metadata \
  --smoke-limit 1 \
  --check-headers \
  --dry-run \
  --profile-data \
  --dry-run-output-json outputs/kaggle_smoke_profile_dry_run.json
```

Validate the smoke dataset:

```bash
KAGGLE_DIR="$(find dataset -maxdepth 1 -type d -name '2.Kaggle*' | head -n 1)"

python scripts/validate_processed_records.py \
  "$KAGGLE_DIR/processed_smoke" \
  --expected-channels 3 \
  --require-success

python scripts/check_processed_pipeline.py \
  --processed-dir "$KAGGLE_DIR/processed_smoke" \
  --expected-channels 3 \
  --require-success \
  --window-seconds 1 \
  --stride-seconds 1 \
  --label-mode binary
```

A successful run writes `_SUCCESS.json` next to `manifest.csv` and
`source_summary.csv`. The marker is removed at the start of a new run and
written again only after all CSV/JSON metadata has been atomically replaced.
During a long run, `manifest.csv`, `source_summary.csv`, `subjects.csv`,
`loso_folds.csv`, and `config.json` are atomically checkpointed after each
completed source CSV; an interrupted directory without `_SUCCESS.json` can be
continued with `--resume`.
`validate_processed_records.py` checks `source_summary.csv` against
`manifest.csv`, including zero-record source CSV files. It also rejects record
paths that are absolute or escape the processed directory, verifies
`loso_folds.csv` subject/segment metadata against `manifest.csv`, and enforces
`x_dtype` / `y_binary_dtype` when those dtypes are declared in `config.json`.
The Kaggle smoke suite configs set `windowing.require_success: true`, which
passes `--require-success` to `prepare_processed_record_windows.py` and makes
suite preflight require `_SUCCESS.json` with `status=complete`. This keeps
partial `processed_smoke/` directories from being windowed or used for
training.

After smoke processed validation, the binary training smoke suite can be
checked without launching training:

```bash
python scripts/run_fog_suite.py \
  --config configs/kaggle_smoke_suite.json \
  --dry-run \
  --skip-collection
```

Run it for real only after `processed_smoke/` exists:

```bash
python scripts/run_fog_suite.py --config configs/kaggle_smoke_suite.json
```

Equivalent one-command smoke execution, after reviewing the dry-run logs:

```bash
python scripts/start_kaggle_smoke_pipeline.py --execute --overwrite
```

```bash
bash scripts/start_kaggle_smoke_pipeline.sh --execute --overwrite
```

```powershell
.\scripts\start_kaggle_smoke_pipeline.ps1 -Execute -Overwrite
```

### Level 2: Full Supervised Processed

Creates the full supervised `processed/` dataset from `train/tdcsfog` and
`train/defog`. It still does not extract the whole zip and skips `unlabeled/`.

The full launcher is also safe by default. It runs preflight, full streaming
dry-run with header checks, and full suite dry-run without creating
`processed/`:
Launcher-invoked preflight uses `--skip-pytest`; run
`check_kaggle_fog_preflight.py` directly when you want the standalone
synthetic test step.

```bash
python scripts/start_kaggle_full_pipeline.py
```

```bash
bash scripts/start_kaggle_full_pipeline.sh
```

```powershell
.\scripts\start_kaggle_full_pipeline.ps1
```

After reviewing the dry-run log, launch the full processing and suite stages
explicitly:

```bash
python scripts/start_kaggle_full_pipeline.py --execute --overwrite
```

```bash
bash scripts/start_kaggle_full_pipeline.sh --execute --overwrite
```

```powershell
.\scripts\start_kaggle_full_pipeline.ps1 -Execute -Overwrite
```

Resume an interrupted launcher run:

```bash
python scripts/start_kaggle_full_pipeline.py --execute --resume
```

The Python launcher streams child-process output to
`outputs/logs/kaggle_full_pipeline.log`.
Python and Bash launchers accept `--log-path`, `--preflight-json`,
`--dry-run-json`, and `--status-json`; the PowerShell launcher accepts
`-LogPath`, `-PreflightPath`, `-DryRunReportPath`, and `-StatusPath`.
Use these when running isolated CI/server probes so stale status files cannot
be confused with the current run.
Add `--profile-data` / `-ProfileData` to launcher dry-runs when the log and
dry-run JSON should include NaN/non-finite and label-count diagnostics.
If `--profile-data` is used on an execute path, the status gate also requires
retained rows to have no non-finite sensor values and no invalid or non-binary
labels.
The real preprocessing path also validates retained features and labels before
writing each record, so `processed*/records/*.npz` cannot contain non-finite
`x` values or labels outside `0/1`.
After execute preprocessing, the launcher validates the sample-level records
with `validate_processed_records.py --require-success`. Add
`--post-check-window-dry-run` / `-PostCheckWindowDryRun` for a stricter
post-check that also runs a window dry-run through `check_processed_pipeline.py`
without writing `windows.npz`.
When `--execute` is used, the full launcher also requires preflight unless
`--allow-execute-without-preflight` is explicitly supplied.
It then runs the full streaming dry-run and
`kaggle_fog_status.py --require-ready full` before creating `processed/`;
bypassing that final gate requires `--allow-execute-without-status-gate`.
With `--resume` or `--overwrite`, the gate allows an existing `processed/`
directory but still checks the same preflight, suite, zip fingerprint, and
dry-run evidence.

```bash
python scripts/preprocess_kaggle_fog_streaming.py \
  --dataset-root dataset \
  --source both \
  --valid-only \
  --task-only \
  --strict-metadata \
  --overwrite
```

Resume an interrupted run:

```bash
python scripts/preprocess_kaggle_fog_streaming.py \
  --dataset-root dataset \
  --source both \
  --valid-only \
  --task-only \
  --strict-metadata \
  --resume
```

`--overwrite` and `--resume` are mutually exclusive: use `--overwrite` to start
over, or `--resume` to continue an existing compatible output directory.
Resume uses existing `source_summary.csv` checkpoints to skip source CSV files
that were already completed. This also covers CSV files that pass metadata
checks but produce zero records after `Valid`/`Task` filtering. Record NPZ files
and metadata files are written through temporary files and atomically replaced.
Use the same `--record-compression` setting when resuming a partially completed
processed directory. The script rejects `--resume` when key preprocessing
settings such as `--valid-only`, `--task-only`, `--min-samples`, source zip, or
record compression differ from the existing `config.json`. If the first run
used `--strict-metadata`, use it again when resuming.
The full Kaggle suite configs also set `windowing.require_success: true`, so a
partial `processed/` directory without `_SUCCESS.json` cannot be windowed or
used for training.

Validate the full dataset:

```bash
KAGGLE_DIR="$(find dataset -maxdepth 1 -type d -name '2.Kaggle*' | head -n 1)"

python scripts/validate_processed_records.py \
  "$KAGGLE_DIR/processed" \
  --expected-channels 3 \
  --require-success

python scripts/check_processed_pipeline.py \
  --processed-dir "$KAGGLE_DIR/processed" \
  --expected-channels 3 \
  --require-success \
  --window-seconds 1 \
  --stride-seconds 1 \
  --label-mode binary
```

The full binary suite is configured separately for server runs:

```bash
python scripts/run_fog_suite.py --config configs/kaggle_full_suite.json
```

## Output Contract

Sample-level processed records:

```text
processed/
  records/
    S001_seg000.npz
  manifest.csv
  source_summary.csv
  subjects.csv
  loso_folds.csv
  config.json
```

Each record contains only:

```text
x         [n_samples, 3] float32
y_binary  [n_samples] uint8
```

Label rule:

```text
y_binary = max(StartHesitation, Turn, Walking)
```

Windowing, Pre-FOG labeling, normalization, resampling, and fold-specific
imputation are not part of `processed/`; they belong to training/window
materialization scripts.

## Windows Notes

Use `prepare_processed_record_windows.py` after the sample-level dataset is
validated. Start with dry-run:

```bash
python scripts/prepare_processed_record_windows.py \
  --processed-dir "$KAGGLE_DIR/processed_smoke" \
  --output-dir outputs/kaggle_smoke_windows_dry_run \
  --window-seconds 1 \
  --stride-seconds 1 \
  --label-mode binary \
  --dry-run
```

If a dataset preserves NaN values, choose an explicit policy:

```bash
--nan-policy zero
```

The default `--nan-policy error` fails fast on NaN.
