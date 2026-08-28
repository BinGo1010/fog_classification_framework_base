#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <no-perturbation-arm-root> <gaussian-mask-arm-root> [output-root] [data-dir] [subjects]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NONE_NGM_ROOT="$1"
GAUSSIAN_MASK_NGM_ROOT="$2"
OUTPUT_ROOT="${3:-${REPO_ROOT}/outputs/private_gru_ngm_robustness_matched_tcn}"
DATA_DIR="${4:-${REPO_ROOT}/dataset/0.Private/processed_NBM_Exp}"
SUBJECTS="${5:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -u scripts/launch_private_gru_ngm_robustness_tcn_8gpu.py \
  --data-dir "${DATA_DIR}" \
  --none-ngm-root "${NONE_NGM_ROOT}" \
  --gaussian-mask-ngm-root "${GAUSSIAN_MASK_NGM_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --subjects "${SUBJECTS}" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --tcn-max-epochs 5 \
  --tcn-patience 2
