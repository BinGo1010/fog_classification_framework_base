#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/launch_all_dataset_processed_nbm_exp_within_subject_transformer_ngm_43k_tcn_8gpu.py \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --phase full \
  "$@"
