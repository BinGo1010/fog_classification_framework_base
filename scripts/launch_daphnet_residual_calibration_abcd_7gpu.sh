#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6}"
TCN_SEEDS="${TCN_SEEDS:-20260807,20260808,20260809}"
NBM_SOURCE_ROOT="${NBM_SOURCE_ROOT:-$REPO_ROOT/outputs/daphnet_processed_NBM_conv_tcn_residual_repr_compare_seed20260807}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/daphnet_residual_calibration_ABCD_3seed_seed20260807}"

exec "$PYTHON_BIN" scripts/launch_daphnet_residual_calibration_abcd_7gpu.py \
  --gpu-ids "$GPU_IDS_CSV" \
  --tcn-seeds "$TCN_SEEDS" \
  --nbm-source-root "$NBM_SOURCE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  "$@"
