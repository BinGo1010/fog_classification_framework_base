#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${1:-dataset/All_dataset/processed_NBM_Exp}"
if [[ $# -gt 0 ]]; then
  shift
fi

python scripts/launch_all_dataset_processed_nbm_exp_gru_r_delta_tcn_ep50pat10_7gpu.py \
  --data-dir "$DATA_DIR" \
  --output-root outputs/all_dataset_processed_NBM_Exp_within_subject_gru_base_mask4_8_r_delta_tcn_nbm300pat20_ep50pat10_seedset_0_52_161_5216_52161 \
  --gpu-ids 0,1,2,3,4,5,6 \
  --seeds 0,52,161,5216,52161 \
  --batch-size 128 \
  --nbm-max-epochs 300 \
  --nbm-patience 20 \
  --tcn-max-epochs 50 \
  --tcn-patience 10 \
  --phase full \
  "$@"
