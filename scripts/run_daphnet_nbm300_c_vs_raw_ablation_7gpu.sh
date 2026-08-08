#!/usr/bin/env bash
set -euo pipefail

python scripts/launch_daphnet_nbm300_c_vs_raw_ablation_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --phase full \
  --nbm-max-epochs 300 \
  --nbm-patience 20 \
  --tcn-max-epochs 10 \
  --tcn-patience 2 \
  --tcn-seeds 20260807,20260808,20260809 \
  "$@"
