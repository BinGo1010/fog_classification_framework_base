#!/usr/bin/env bash
set -euo pipefail

# Seven-GPU launcher for the subject-sharded A5_50 E0--E3 protocol.
# Usage: bash scripts/run_daphnet_nbm_e0_e3_7gpu.sh [phase1|phase2|all]

MODE="${1:-all}"
if [[ "$MODE" != "phase1" && "$MODE" != "phase2" && "$MODE" != "all" ]]; then
  echo "usage: $0 [phase1|phase2|all]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/dataset/1.Daphnet Freezing of Gait Dataset/processed_A5_50}"
SHARD_ROOT="${SHARD_ROOT:-$REPO_ROOT/outputs/daphnet_nbm_E0_E3_A5_50_7gpu_shards_v1}"
AGG_ROOT="${AGG_ROOT:-$REPO_ROOT/outputs/daphnet_nbm_E0_E3_A5_50_7gpu_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-}"
SEEDS="${SEEDS:-20260802,20260803,20260804}"
MAX_EPOCHS="${MAX_EPOCHS:-2000}"
PATIENCE="${PATIENCE:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
INCLUDE_E2_P16="${INCLUDE_E2_P16:-0}"
INCLUDE_E3B="${INCLUDE_E3B:-0}"
FORCE_RETRAIN_E0="${FORCE_RETRAIN_E0:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Workload-balanced assignment based on A5_50 train and E3 context-window counts.
SUBJECT_SHARDS=(
  "S04"
  "S10"
  "S06"
  "S01"
  "S07,S08"
  "S02,S05"
  "S03,S09"
)

if [[ -n "$CONDA_ENV" ]]; then
  PY=(conda run --no-capture-output -n "$CONDA_ENV" "$PYTHON_BIN")
else
  PY=("$PYTHON_BIN")
fi

mkdir -p "$SHARD_ROOT/logs"

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --seeds "$SEEDS"
  --max-epochs "$MAX_EPOCHS"
  --patience "$PATIENCE"
  --batch-size "$BATCH_SIZE"
  --workers 0
  --device cuda
)
OPTIONAL_ARGS=()
AGG_OPTIONAL_ARGS=()
if [[ "$INCLUDE_E2_P16" == "1" ]]; then
  OPTIONAL_ARGS+=(--include-e2-p16)
  AGG_OPTIONAL_ARGS+=(--include-e2-p16)
fi
if [[ "$INCLUDE_E3B" == "1" ]]; then
  OPTIONAL_ARGS+=(--include-e3b)
  AGG_OPTIONAL_ARGS+=(--include-e3b)
fi
if [[ "$FORCE_RETRAIN_E0" == "1" ]]; then
  OPTIONAL_ARGS+=(--force-retrain-e0)
fi

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_shards() {
  local phase="$1"
  local capacity="${2:-}"
  local -a pids=()
  local -a labels=()
  for gpu in "${!SUBJECT_SHARDS[@]}"; do
    local subjects="${SUBJECT_SHARDS[$gpu]}"
    local output="$SHARD_ROOT/gpu$gpu"
    local log="$SHARD_ROOT/logs/${phase}_gpu${gpu}.log"
    local -a phase_args=()
    if [[ "$phase" == "phase1" ]]; then
      phase_args=(--stop-after E2)
    else
      phase_args=(--stop-after E3 --e3-capacity "$capacity")
    fi
    local -a command=(
      "${PY[@]}" -u "$SCRIPT_DIR/run_daphnet_nbm_e0_e3_a5_50.py"
      "${COMMON_ARGS[@]}"
      --subjects "$subjects"
      --output-root "$output"
      "${OPTIONAL_ARGS[@]}"
      "${phase_args[@]}"
    )
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'CUDA_VISIBLE_DEVICES=%s ' "$gpu"
      print_command "${command[@]}"
      continue
    fi
    echo "START $phase GPU$gpu subjects=$subjects log=$log"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "${command[@]}"
    ) >"$log" 2>&1 &
    pids+=("$!")
    labels+=("GPU$gpu:$subjects")
  done
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi
  local failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "DONE $phase ${labels[$index]}"
    else
      echo "FAILED $phase ${labels[$index]} (inspect $SHARD_ROOT/logs)" >&2
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    exit 1
  fi
}

aggregate_phase() {
  local phase="$1"
  local -a command=(
    "${PY[@]}" -u "$SCRIPT_DIR/aggregate_daphnet_nbm_e0_e3_a5_50_shards.py"
    --phase "$phase"
    --data-dir "$DATA_DIR"
    --shard-parent "$SHARD_ROOT"
    --output-root "$AGG_ROOT"
    --seeds "$SEEDS"
    --batch-size "$BATCH_SIZE"
    --max-epochs "$MAX_EPOCHS"
    --patience "$PATIENCE"
    --bootstrap-samples "$BOOTSTRAP_SAMPLES"
    --device cuda:0
    "${AGG_OPTIONAL_ARGS[@]}"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=0 '
    print_command "${command[@]}"
  else
    CUDA_VISIBLE_DEVICES=0 "${command[@]}"
  fi
}

run_phase1() {
  run_shards phase1
  aggregate_phase e2
}

read_capacity() {
  local decision="$AGG_ROOT/E3_capacity_decision.json"
  if [[ ! -f "$decision" ]]; then
    echo "missing global capacity decision: $decision; run phase1 first" >&2
    exit 1
  fi
  "${PY[@]}" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["runner_argument"])' \
    "$decision"
}

run_phase2() {
  local capacity
  if [[ "$DRY_RUN" == "1" ]]; then
    capacity="<global-p24-or-m3>"
  else
    capacity="$(read_capacity)"
  fi
  echo "FROZEN E3 capacity=$capacity"
  run_shards phase2 "$capacity"
  aggregate_phase e3
}

case "$MODE" in
  phase1) run_phase1 ;;
  phase2) run_phase2 ;;
  all)
    run_phase1
    run_phase2
    ;;
esac

echo "COMPLETE mode=$MODE aggregate=$AGG_ROOT shards=$SHARD_ROOT"
