#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6}"
TCN_SEEDS="${TCN_SEEDS:-20260807,20260808,20260809}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/daphnet_conv_tcn_nbm200_BC_3seed_seed20260807}"

exec "$PYTHON_BIN" scripts/launch_daphnet_conv_tcn_nbm200_bc_7gpu.py \
  --gpu-ids "$GPU_IDS_CSV" \
  --tcn-seeds "$TCN_SEEDS" \
  --output-root "$OUTPUT_ROOT" \
  --nbm-max-epochs 200 \
  --nbm-patience 20 \
  --nbm-learning-rate 1e-3 \
  --tcn-max-epochs 30 \
  --tcn-patience 6 \
  "$@"
