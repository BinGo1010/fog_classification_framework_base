#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.  Pass 0,1,2,3,4,5,6,7 to use eight GPUs.
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-$PWD/dataset/1.Daphnet Freezing of Gait Dataset/processed_NBM}"
SOURCE_ROOT="${SOURCE_ROOT:-$PWD/outputs/daphnet_64Hz_raw_tcn_lr3e-3_wd1e-3_batch128_ep5pat2_seedset_0_52_161_5216_52161}"
SCALER_SOURCE_ROOT="${SCALER_SOURCE_ROOT:-$PWD/outputs/daphnet_tcn_nbm300_C_vs_raw_tcn_ep5pat2_seedset_0_52_161_5216_52161/nbm_source}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs/daphnet_64Hz_raw_tcn_lr3e-3_wd1e-3_batch128_ep5pat2_test_noise_snr30_20_10_0_seedset_0_52_161_5216_52161}"

export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL || true

"$PYTHON_BIN" scripts/launch_daphnet_raw_tcn_test_noise_snr_7gpu.py \
  --data-dir "$DATA_DIR" \
  --source-root "$SOURCE_ROOT" \
  --scaler-source-root "$SCALER_SOURCE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --gpu-ids "$GPU_IDS" \
  --phase full
