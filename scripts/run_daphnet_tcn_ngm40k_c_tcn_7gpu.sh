#!/usr/bin/env bash
set -euo pipefail

python -m scripts.launch_daphnet_tcn_ngm40k_c_tcn_7gpu \
  --data-dir "dataset/1.Daphnet Freezing of Gait Dataset/processed_NBM" \
  --output-root outputs/daphnet_tcn_ngm40k_FULL_C_tcn_ep5pat2_seedset_0_52_161_5216_52161 \
  --experiment-methods FULL_C \
  --gpu-ids 0,1,2,3,4,5,6 \
  --nbm-seeds 0,52,161,5216,52161 \
  --tcn-seeds 0,52,161,5216,52161 \
  --nbm-max-epochs 300 \
  --nbm-patience 20 \
  --tcn-max-epochs 5 \
  --tcn-patience 2 \
  --phase full \
  "$@"
