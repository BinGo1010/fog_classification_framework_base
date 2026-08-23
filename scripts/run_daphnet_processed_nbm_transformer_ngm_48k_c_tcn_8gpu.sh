#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${1:-dataset/1.Daphnet Freezing of Gait Dataset/processed_NBM}"
if [[ $# -gt 0 ]]; then
  shift
fi

python scripts/launch_daphnet_processed_nbm_transformer_ngm_48k_c_tcn_8gpu.py \
  --data-dir "$DATA_DIR" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --phase full \
  "$@"
