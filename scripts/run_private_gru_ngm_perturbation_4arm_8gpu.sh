#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-${REPO_ROOT}/dataset/0.Private/processed_NBM_Exp}"
OUTPUT_ROOT="${2:-${REPO_ROOT}/outputs/private_gru_ngm_perturbation_4arm_5seed}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -u scripts/launch_private_gru_ngm_perturbation_4arm_8gpu.py \
  --data-dir "${DATA_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --subjects P01,P02,P03,P04,P05,P06,P07,P08 \
  --folds 0,1,2 \
  --seeds 0,52,161,5216,52161 \
  --nbm-batch-size 16 \
  --maximum-updates 5000 \
  --validation-frequency 50 \
  --validation-patience 20 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001
