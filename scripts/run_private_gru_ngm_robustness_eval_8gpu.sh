#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINED_ROOT="${1:-${REPO_ROOT}/outputs/private_gru_ngm_robustness_matched_tcn}"
DATA_DIR="${2:-${REPO_ROOT}/dataset/0.Private/processed_NBM_Exp}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -u scripts/launch_private_gru_ngm_robustness_eval_8gpu.py \
  --data-dir "${DATA_DIR}" \
  --trained-root "${TRAINED_ROOT}" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --batch-size 128
