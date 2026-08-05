#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/daphnet_resnet8_avgpool8_b1_inceptiontime_pilot_v1/full_subject_b1_experiment}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

pids=()
for gpu in 0 1 2 3 4 5 6; do
  "$PYTHON_BIN" scripts/run_daphnet_resnet8_avgpool8_b1_inceptiontime_pilot.py \
    --device "cuda:$gpu" \
    --shard-index "$gpu" \
    --shard-count 7 \
    --output-root "$OUTPUT_ROOT" \
    "$@" \
    >"$LOG_DIR/shard${gpu}.out.log" \
    2>"$LOG_DIR/shard${gpu}.err.log" &
  pid="$!"
  pids+=("$pid")
  echo "launched shard=$gpu gpu=$gpu pid=$pid"
done

terminate_children() {
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "shard $index failed; inspect $LOG_DIR/shard${index}.err.log" >&2
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "At least one shard failed. Rerun this launcher to resume incomplete runs." >&2
  exit "$status"
fi

"$PYTHON_BIN" scripts/run_daphnet_resnet8_avgpool8_b1_inceptiontime_pilot.py \
  --finalize-only \
  --device cpu \
  --output-root "$OUTPUT_ROOT"

echo "Seven-GPU experiment and final aggregation completed: $OUTPUT_ROOT"
