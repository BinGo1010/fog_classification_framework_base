#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python -u scripts/launch_daphnet_gru_residual_g123_7gpu.py \
  --output-root outputs/daphnet_gru_nbm300_residual_G1_G2_G3_tcn_ep50pat6_seedset_0_52_161 \
  --gpu-ids 0,1,2,3,4,5,6 \
  --tcn-seeds 0,52,161 \
  --tcn-max-epochs 50 \
  --tcn-patience 6 \
  --phase full \
  "$@"
