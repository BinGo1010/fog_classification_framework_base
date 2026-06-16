#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
SUITE_CONFIG="configs/multimodal_full_suite.json"
ONLY="all"
INCLUDE_EXPERIMENTS=""
EXCLUDE_EXPERIMENTS=""
REQUIRE_WINDOWS=0
DRY_RUN=0
NO_PREFLIGHT=0
NO_AUDIT=0
NO_REUSE_EXISTING_WINDOWS=0
NO_DEDUPE_WINDOWING=0
NO_SKIP_COMPLETED_TRAINING=0

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
    --suite-config)
      SUITE_CONFIG="$2"
      shift 2
      ;;
    --only)
      ONLY="$2"
      shift 2
      ;;
    --include-experiments)
      INCLUDE_EXPERIMENTS="$2"
      shift 2
      ;;
    --exclude-experiments)
      EXCLUDE_EXPERIMENTS="$2"
      shift 2
      ;;
    --require-windows)
      REQUIRE_WINDOWS=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-preflight)
      NO_PREFLIGHT=1
      shift
      ;;
    --no-audit)
      NO_AUDIT=1
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
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "$REPO"
export PYTHONUNBUFFERED=1

LOG_DIR="$REPO/outputs/logs"
LOG_PATH="$LOG_DIR/multimodal_full_suite.log"
PREFLIGHT_PATH="$REPO/outputs/multimodal_full_suite_preflight.json"
AUDIT_PATH="$REPO/outputs/multimodal_full_suite_audit.json"
mkdir -p "$LOG_DIR"

run_logged_python() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] CMD $PYTHON $*" | tee -a "$LOG_PATH"
  "$PYTHON" "$@" 2>&1 | tee -a "$LOG_PATH"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] start multimodal full suite" | tee -a "$LOG_PATH"

if [[ "$NO_PREFLIGHT" -eq 0 ]]; then
  preflight_args=(
    scripts/preflight_fog_suite.py
    --config "$SUITE_CONFIG"
    --output-json "$PREFLIGHT_PATH"
  )
  if [[ "$REQUIRE_WINDOWS" -eq 1 ]]; then
    preflight_args+=(--require-windows)
  fi
  run_logged_python "${preflight_args[@]}"
fi

suite_args=(
  scripts/run_fog_suite.py
  --config "$SUITE_CONFIG"
  --only "$ONLY"
)
if [[ "$DRY_RUN" -eq 1 ]]; then
  suite_args+=(--dry-run)
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
if [[ -n "$INCLUDE_EXPERIMENTS" ]]; then
  suite_args+=(--include-experiments "$INCLUDE_EXPERIMENTS")
fi
if [[ -n "$EXCLUDE_EXPERIMENTS" ]]; then
  suite_args+=(--exclude-experiments "$EXCLUDE_EXPERIMENTS")
fi

run_logged_python "${suite_args[@]}"

if [[ "$NO_AUDIT" -eq 0 && "$DRY_RUN" -eq 0 ]]; then
  case "$ONLY" in
    all|training|collection)
      run_logged_python \
        scripts/audit_fog_suite_results.py \
        --config "$SUITE_CONFIG" \
        --output-json "$AUDIT_PATH"
      ;;
  esac
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished multimodal full suite" | tee -a "$LOG_PATH"
