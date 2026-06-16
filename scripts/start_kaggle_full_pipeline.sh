#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATASET_ROOT=""
RECORD_COMPRESSION="compressed"
ONLY="all"
EXECUTE=0
RESUME=0
OVERWRITE=0
NO_PREFLIGHT=0
ALLOW_EXECUTE_WITHOUT_PREFLIGHT=0
ALLOW_EXECUTE_WITHOUT_STATUS_GATE=0
NO_VALIDATION=0
POST_CHECK_WINDOW_DRY_RUN=0
NO_SUITE=0
NO_REUSE_EXISTING_WINDOWS=0
NO_DEDUPE_WINDOWING=0
NO_SKIP_COMPLETED_TRAINING=0
PROFILE_DATA=0
LOG_PATH=""
PREFLIGHT_PATH=""
DRY_RUN_REPORT_PATH=""
STATUS_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO="$2"
      shift 2
      ;;
    --python)
      PYTHON="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --record-compression)
      RECORD_COMPRESSION="$2"
      shift 2
      ;;
    --only)
      ONLY="$2"
      shift 2
      ;;
    --execute)
      EXECUTE=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --no-preflight)
      NO_PREFLIGHT=1
      shift
      ;;
    --allow-execute-without-preflight)
      ALLOW_EXECUTE_WITHOUT_PREFLIGHT=1
      shift
      ;;
    --allow-execute-without-status-gate)
      ALLOW_EXECUTE_WITHOUT_STATUS_GATE=1
      shift
      ;;
    --no-validation)
      NO_VALIDATION=1
      shift
      ;;
    --post-check-window-dry-run)
      POST_CHECK_WINDOW_DRY_RUN=1
      shift
      ;;
    --no-suite)
      NO_SUITE=1
      shift
      ;;
    --no-reuse-existing-windows)
      NO_REUSE_EXISTING_WINDOWS=1
      shift
      ;;
    --no-dedupe-windowing)
      NO_DEDUPE_WINDOWING=1
      shift
      ;;
    --no-skip-completed-training)
      NO_SKIP_COMPLETED_TRAINING=1
      shift
      ;;
    --profile-data)
      PROFILE_DATA=1
      shift
      ;;
    --log-path)
      LOG_PATH="$2"
      shift 2
      ;;
    --preflight-json)
      PREFLIGHT_PATH="$2"
      shift 2
      ;;
    --dry-run-json)
      DRY_RUN_REPORT_PATH="$2"
      shift 2
      ;;
    --status-json)
      STATUS_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$RESUME" -eq 1 && "$OVERWRITE" -eq 1 ]]; then
  echo "--resume and --overwrite are mutually exclusive." >&2
  exit 2
fi
if [[ "$EXECUTE" -eq 1 && "$NO_PREFLIGHT" -eq 1 && "$ALLOW_EXECUTE_WITHOUT_PREFLIGHT" -eq 0 ]]; then
  echo "--execute with --no-preflight requires --allow-execute-without-preflight." >&2
  exit 2
fi
if [[ "$RECORD_COMPRESSION" != "compressed" && "$RECORD_COMPRESSION" != "none" ]]; then
  echo "--record-compression must be 'compressed' or 'none'." >&2
  exit 2
fi

cd "$REPO"
REPO="$(pwd)"
if [[ -z "$DATASET_ROOT" ]]; then
  DATASET_ROOT="$REPO/dataset"
fi

find_kaggle_dir() {
  local root="$1"
  local matches=()
  shopt -s nullglob
  matches=("$root"/2.Kaggle*)
  shopt -u nullglob
  if [[ "${#matches[@]}" -ne 1 || ! -d "${matches[0]}" ]]; then
    echo "Expected one 2.Kaggle* directory under $root, found ${#matches[@]}." >&2
    exit 2
  fi
  printf '%s\n' "${matches[0]}"
}

KAGGLE_DIR="$(find_kaggle_dir "$DATASET_ROOT")"
PROCESSED="$KAGGLE_DIR/processed"
LOG_DIR="$REPO/outputs/logs"
if [[ -z "$LOG_PATH" ]]; then
  LOG_PATH="$LOG_DIR/kaggle_full_pipeline.log"
fi
if [[ -z "$PREFLIGHT_PATH" ]]; then
  PREFLIGHT_PATH="$REPO/outputs/kaggle_preflight_report.json"
fi
if [[ -z "$DRY_RUN_REPORT_PATH" ]]; then
  DRY_RUN_REPORT_PATH="$REPO/outputs/kaggle_full_streaming_dry_run.json"
fi
if [[ -z "$STATUS_PATH" ]]; then
  STATUS_PATH="$REPO/outputs/kaggle_status.json"
fi
SUITE_CONFIG="configs/kaggle_full_suite.json"
mkdir -p "$(dirname "$LOG_PATH")"
export PYTHONUNBUFFERED=1

log_line() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_PATH"
}

run_logged_python() {
  log_line "CMD $PYTHON $*"
  "$PYTHON" "$@" 2>&1 | tee -a "$LOG_PATH"
}

log_line "start Kaggle full pipeline execute=$EXECUTE"

if [[ "$NO_PREFLIGHT" -eq 0 ]]; then
  run_logged_python \
    scripts/check_kaggle_fog_preflight.py \
    --repo-root "$REPO" \
    --dataset-root "$DATASET_ROOT" \
    --suite-config "$SUITE_CONFIG" \
    --skip-pytest \
    --output-json "$PREFLIGHT_PATH"
fi

preprocess_args=(
  scripts/preprocess_kaggle_fog_streaming.py
  --dataset-root "$DATASET_ROOT"
  --source both
  --valid-only
  --task-only
  --strict-metadata
  --record-compression "$RECORD_COMPRESSION"
)
dry_run_args=(
  --check-headers
  --dry-run
  --dry-run-output-json "$DRY_RUN_REPORT_PATH"
)
if [[ "$PROFILE_DATA" -eq 1 ]]; then
  dry_run_args+=(--profile-data)
fi

if [[ "$EXECUTE" -eq 1 ]]; then
  run_logged_python "${preprocess_args[@]}" "${dry_run_args[@]}"
  if [[ "$ALLOW_EXECUTE_WITHOUT_STATUS_GATE" -eq 0 ]]; then
    status_args=(
      scripts/kaggle_fog_status.py
      --repo-root "$REPO"
      --dataset-root "$DATASET_ROOT"
      --preflight-json "$PREFLIGHT_PATH"
      --full-dry-run-json "$DRY_RUN_REPORT_PATH"
      --output-json "$STATUS_PATH"
      --require-ready full
    )
    if [[ "$RESUME" -eq 1 || "$OVERWRITE" -eq 1 ]]; then
      status_args+=(--allow-existing-output)
    fi
    run_logged_python "${status_args[@]}"
  fi
  if [[ "$RESUME" -eq 1 ]]; then
    preprocess_args+=(--resume)
  fi
  if [[ "$OVERWRITE" -eq 1 ]]; then
    preprocess_args+=(--overwrite)
  fi
else
  preprocess_args+=("${dry_run_args[@]}")
fi

run_logged_python "${preprocess_args[@]}"

if [[ "$EXECUTE" -eq 1 && "$NO_VALIDATION" -eq 0 ]]; then
  if [[ "$POST_CHECK_WINDOW_DRY_RUN" -eq 1 ]]; then
    run_logged_python \
      scripts/check_processed_pipeline.py \
      --processed-dir "$PROCESSED" \
      --expected-channels 3 \
      --require-success \
      --window-seconds 1 \
      --stride-seconds 1 \
      --label-mode binary \
      --nan-policy error \
      --target-hz 100
  else
    run_logged_python \
      scripts/validate_processed_records.py \
      "$PROCESSED" \
      --expected-channels 3 \
      --require-success
  fi
elif [[ "$EXECUTE" -eq 0 ]]; then
  log_line "skip processed validation because --execute was not provided"
fi

if [[ "$NO_SUITE" -eq 0 ]]; then
  suite_args=(
    scripts/run_fog_suite.py
    --config "$SUITE_CONFIG"
    --only "$ONLY"
    --validate-experiment-configs
  )
  if [[ "$EXECUTE" -eq 0 ]]; then
    suite_args+=(--dry-run --skip-collection)
  fi
  if [[ "$NO_REUSE_EXISTING_WINDOWS" -eq 1 ]]; then
    suite_args+=(--no-reuse-existing-windows)
  fi
  if [[ "$NO_DEDUPE_WINDOWING" -eq 1 ]]; then
    suite_args+=(--no-dedupe-windowing)
  fi
  if [[ "$NO_SKIP_COMPLETED_TRAINING" -eq 1 ]]; then
    suite_args+=(--no-skip-completed-training)
  fi
  run_logged_python "${suite_args[@]}"
fi

log_line "finished Kaggle full pipeline execute=$EXECUTE"
