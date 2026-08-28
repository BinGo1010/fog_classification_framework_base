#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <no-perturbation-ngm-root> <gaussian-mask-ngm-root> [output-root] [data-dir]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NONE_NGM_ROOT="$1"
GAUSSIAN_MASK_NGM_ROOT="$2"
OUTPUT_ROOT="${3:-${REPO_ROOT}/outputs/daphnet_gru_ngm_robustness_matched_tcn}"
DATA_DIR="${4:-${REPO_ROOT}/dataset/1.Daphnet Freezing of Gait Dataset/processed_NBM}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -u scripts/launch_daphnet_gru_ngm_robustness_tcn_8gpu.py \
  --data-dir "${DATA_DIR}" \
  --none-ngm-root "${NONE_NGM_ROOT}" \
  --gaussian-mask-ngm-root "${GAUSSIAN_MASK_NGM_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --tcn-max-epochs 5 \
  --tcn-patience 2
