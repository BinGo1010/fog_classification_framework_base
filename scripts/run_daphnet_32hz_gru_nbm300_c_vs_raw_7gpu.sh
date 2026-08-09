#!/usr/bin/env bash
set -euo pipefail

python scripts/launch_daphnet_32hz_gru_nbm300_c_vs_raw_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --phase full \
  --nbm-seeds 0,52,161,5216,52161 \
  --tcn-seeds 0,52,161,5216,52161 \
  --nbm-max-epochs 300 \
  --nbm-patience 20 \
  --tcn-max-epochs 20 \
  --tcn-patience 5 \
  "$@"
