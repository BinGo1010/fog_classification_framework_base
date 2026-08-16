#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python -u scripts/launch_daphnet_gru_residual_g123_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --tcn-seeds 0,52,161 \
  --tcn-max-epochs 10 \
  --tcn-patience 2 \
  --phase full \
  "$@"
