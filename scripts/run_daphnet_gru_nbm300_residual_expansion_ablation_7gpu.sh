#!/usr/bin/env bash
set -euo pipefail

python scripts/launch_daphnet_gru_nbm300_c_vs_raw_ep5pat2_7gpu.py \
  --output-root outputs/daphnet_gru_nbm300_FULL_C_vs_RESIDUAL_R_tcn_ep5pat2_seedset_0_52_161_5216_52161 \
  --reuse-nbm-source-root outputs/daphnet_tcn_nbm300_C_vs_raw_tcn_ep5pat2_seedset_0_52_161_5216_52161/nbm_source \
  --experiment-methods FULL_C,RESIDUAL_R \
  --gpu-ids 0,1,2,3,4,5,6 \
  --phase full \
  --nbm-seeds 0,52,161,5216,52161 \
  --tcn-seeds 0,52,161,5216,52161 \
  --nbm-max-epochs 300 \
  --nbm-patience 20 \
  --tcn-max-epochs 5 \
  --tcn-patience 2 \
  "$@"
