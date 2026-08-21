#!/usr/bin/env bash
set -euo pipefail

python scripts/launch_all_dataset_processed_nbm_exp_within_subject_gru_nbm_tcn_8gpu.py \
  --data-dir dataset/All_dataset/processed_NBM_Exp \
  --output-root outputs/all_dataset_processed_NBM_Exp_within_subject_gru_base_mask4_8_C_tcn_nbm300pat20_ep5pat2_seedset_0_52_161_5216_52161 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --seeds 0,52,161,5216,52161 \
  --batch-size 128 \
  --nbm-max-epochs 300 \
  --nbm-patience 20 \
  --tcn-max-epochs 5 \
  --tcn-patience 2 \
  --phase full
