#!/usr/bin/env bash
# Seven independent single-subject workers: one visible GPU per formal subject.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/dataset/1.Daphnet Freezing of Gait Dataset/processed_CA_pure}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/daphnet_ca_supcon_subject_v1}"
LOG_DIR="$OUTPUT_ROOT/logs"
SUBJECTS=(S01 S02 S05 S06 S07 S08 S09)

mkdir -p "$LOG_DIR"

if [[ ! -f "$DATA_DIR/ca_window_manifest.csv" ]]; then
  echo "processed_CA_pure manifest not found: $DATA_DIR/ca_window_manifest.csv" >&2
  exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  if [[ "$gpu_count" -lt 7 ]]; then
    echo "Seven GPUs are required, but nvidia-smi reports $gpu_count." >&2
    exit 2
  fi
else
  echo "nvidia-smi not found; cannot verify the seven-GPU server." >&2
  exit 2
fi

pids=()
for gpu in 0 1 2 3 4 5 6; do
  subject="${SUBJECTS[$gpu]}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" scripts/run_daphnet_ca_supcon_subject.py \
      --subject "$subject" \
      --data-dir "$DATA_DIR" \
      --output-root "$OUTPUT_ROOT" \
      --device cuda \
      "$@" \
      >"$LOG_DIR/${subject}.out.log" \
      2>"$LOG_DIR/${subject}.err.log" &
  pid="$!"
  pids+=("$pid")
  echo "launched subject=$subject physical_gpu=$gpu visible_device=cuda:0 pid=$pid"
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
    subject="${SUBJECTS[$index]}"
    echo "$subject failed; inspect $LOG_DIR/${subject}.err.log" >&2
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "At least one subject failed. Rerun the same command to skip completed subjects." >&2
  exit "$status"
fi

"$PYTHON_BIN" scripts/aggregate_daphnet_ca_supcon.py \
  --output-root "$OUTPUT_ROOT" \
  --subjects "S01,S02,S05,S06,S07,S08,S09" \
  --seeds "2026,2027,2028"

echo "Seven-GPU experiment and aggregation completed: $OUTPUT_ROOT"

