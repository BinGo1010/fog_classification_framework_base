#!/usr/bin/env bash
# Run the three authoritative processed_NBM folds concurrently on a 7-GPU server.
# Each fold trains one shared NBM, then the r and [r,|r|,delta(r)] groups.
# There are only three independent folds, so the default protocol uses GPUs 0,1,2.
# The remaining GPUs are intentionally left free instead of changing batch semantics.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/dataset/1.Daphnet Freezing of Gait Dataset/processed_NBM}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/daphnet_processed_NBM_conv_tcn_residual_repr_compare_seed20260807}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6}"
SEED="${SEED:-20260807}"
LOG_DIR="$OUTPUT_ROOT/logs/parallel_workers"

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"
if [[ "${#GPU_IDS[@]}" -lt 3 ]]; then
  echo "At least three GPU ids are required for the three simultaneous folds." >&2
  exit 2
fi
if [[ ! -f "$DATA_DIR/nbm_protocol.json" ]]; then
  echo "processed_NBM protocol not found: $DATA_DIR/nbm_protocol.json" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to validate the server GPUs." >&2
  exit 2
fi
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
for index in 0 1 2; do
  gpu="${GPU_IDS[$index]}"
  if ! [[ "$gpu" =~ ^[0-9]+$ ]] || (( gpu >= GPU_COUNT )); then
    echo "Invalid physical GPU id '$gpu'; nvidia-smi reports $GPU_COUNT GPUs." >&2
    exit 2
  fi
done

mkdir -p "$LOG_DIR"
pids=()
folds=(0 1 2)

terminate_children() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

for index in 0 1 2; do
  fold="${folds[$index]}"
  gpu="${GPU_IDS[$index]}"
  stdout_log="$LOG_DIR/fold${fold}.out.log"
  stderr_log="$LOG_DIR/fold${fold}.err.log"
  CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" scripts/run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py \
      --data-dir "$DATA_DIR" \
      --output-root "$OUTPUT_ROOT" \
      --fold "$fold" \
      --device cuda \
      --seed "$SEED" \
      "$@" \
      >"$stdout_log" 2>"$stderr_log" &
  pid="$!"
  pids+=("$pid")
  echo "launched fold=$fold physical_gpu=$gpu visible_device=cuda:0 pid=$pid"
done

status=0
for index in 0 1 2; do
  if ! wait "${pids[$index]}"; then
    fold="${folds[$index]}"
    echo "fold $fold failed; inspect $LOG_DIR/fold${fold}.err.log" >&2
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one fold failed; aggregation was not run." >&2
  exit "$status"
fi

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" scripts/run_daphnet_processed_nbm_conv_tcn_autoencoder_fold.py \
    --output-root "$OUTPUT_ROOT" \
    --aggregate-only

echo "Three-fold paired residual-representation experiment complete: $OUTPUT_ROOT"
echo "Used physical GPUs: ${GPU_IDS[0]},${GPU_IDS[1]},${GPU_IDS[2]}; other GPUs were not needed by this three-fold protocol."
