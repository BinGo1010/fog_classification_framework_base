# GRU-NBM-v2 Sensitivity Experiment

## Objective

Improve FoG sensitivity by increasing the capacity used to model normal gait
dynamics without increasing the information bandwidth through which an unseen
FoG window can be copied.

## Evidence motivating the design

| NBM | Trainable parameters | Latent representation | Mean role-5 SmoothL1 | Test FoG/Non-FoG median absolute residual ratio | Sensitivity | PR-AUC |
|---|---:|---|---:|---:|---:|---:|
| GRU-v1 | 31,513 | global `[B,16]` | 0.1999 | 2.63 | 0.8662 | 0.7223 |
| Conv-TCN | 47,449 | temporal `[B,16,32]` (512 values) | 0.0658 | 2.28 | 0.8545 | 0.6744 |
| Transformer | 2,329,736 | temporal `[B,8,64]` (512 values) | 0.0250 | 2.00 | 0.7929 | 0.5766 |

A lower clean reconstruction loss did not imply better anomaly detection.  The
two high-bandwidth temporal bottlenecks reconstructed FoG more successfully and
reduced the residual class gap.  GRU-v2 therefore retains a single global
16-dimensional latent vector.

## GRU-NBM-v2 architecture

```text
X [B,128,9]
    |
    v
1-layer bidirectional GRU
hidden=96 per direction
    |
    v
final forward/backward states [B,192]
    |
    v
LayerNorm -> Linear 192->64 -> GELU
    |
    v
Linear 64->16 -> LayerNorm -> tanh
    |
    v
global Z [B,16] + latent dropout 0.10 during fitting
    |
    +--> Linear 16->192 -> decoder initial state [2,B,96]
    |
fixed Fourier time code [B,128,16], 0.5-3.0 Hz
    |
    v
2-layer unidirectional GRU decoder
hidden=96, dropout=0.10
    |
    v
LayerNorm -> Linear 96->48 -> GELU -> Linear 48->9
    |
    v
X_hat [B,128,9]
```

Trainable parameters: **172,697**.  The Fourier code is a fixed buffer and is
not counted.  There are no encoder-token skips, raw-input skips, cross-attention,
teacher forcing, or previous-ground-truth inputs to the decoder.

## Frozen comparison protocol

- Dataset: `processed_NBM`, 64 Hz, 128 samples, stride 64.
- Seeds: `0,52,161,5216,52161`, with exact paired NBM/TCN seeds.
- Role 4: RobustScaler fit and clean Non-FoG NBM training.
- Role 5: clean NBM early stopping; restore the best checkpoint before computing
  calibration `b` and `sigma`.
- NBM: SmoothL1, AdamW `lr=1e-3`, `weight_decay=1e-4`, batch 128,
  max epoch 300, patience 20, gradient clipping 1.0, unchanged 40/40/20
  clean/Gaussian/mask augmentation.
- FULL_C: `e=X-X_hat`, `q=clip(e/sigma,-12,12)`, per-window/per-axis center,
  then `[r,abs(r),delta(r)]` with shape `[B,27,128]`.  Scheme C does not subtract
  calibration bias `b`.
- RAW: identical role-4 scaler and per-window/per-axis centering, `[B,9,128]`.
- Roles 6/7: unchanged TCN fit and `pos_weight` calculation.
- Roles 2/3: unchanged classifier early stopping and maximum-balanced-accuracy
  threshold selection.
- TCN: max epoch 5, patience 2.
- Roles 0/1: accessible only after all 30 classifier checkpoints and validation
  thresholds have been sealed by the global barrier.
- Evaluation rechecks the sealed Scaler, NBM, and classifier checkpoint hashes;
  resume also refuses a changed dataset path or experiment identity.

The run contains 15 NBM fits, 30 classifier fits, and 30 post-barrier tests,
scheduled dynamically across seven GPUs.

## Server command

From the repository root:

```bash
bash scripts/run_daphnet_gru_v2_nbm300_c_vs_raw_ep5pat2_7gpu.sh
```

The default output directory is:

```text
outputs/daphnet_gru_v2_nbm300_C_vs_raw_tcn_ep5pat2_seedset_0_52_161_5216_52161
```

Dry-run the complete job grid before training:

```bash
python scripts/launch_daphnet_gru_v2_nbm300_c_vs_raw_ep5pat2_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --dry-run
```

Resume individual phases without changing the output root:

```bash
python scripts/launch_daphnet_gru_v2_nbm300_c_vs_raw_ep5pat2_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --phase nbm

python scripts/launch_daphnet_gru_v2_nbm300_c_vs_raw_ep5pat2_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --phase train

python scripts/launch_daphnet_gru_v2_nbm300_c_vs_raw_ep5pat2_7gpu.py \
  --gpu-ids 0,1,2,3,4,5,6 \
  --phase evaluate
```

Do not use `--overwrite` for normal resume behavior.

## Pre-registered interpretation

The architecture is successful only if GRU-v2 improves sensitivity over GRU-v1
under the unchanged validation threshold rule while retaining PR-AUC and without
a material specificity collapse.  Suggested checks are:

- mean paired sensitivity delta versus GRU-v1 is positive;
- at least four of five paired seed deltas are positive;
- PR-AUC does not decrease;
- specificity decreases by no more than 0.02;
- FoG/Non-FoG median absolute residual ratio is at least the GRU-v1 reference
  value of approximately 2.63.

If role-5 SmoothL1 improves but the residual ratio, sensitivity, and PR-AUC all
decline, GRU-v2 is reconstructing FoG too well.  Do not respond by making the
model larger; reduce recurrent capacity or remove bidirectionality in a separate
ablation.

Because the permanent test pool has already been inspected during prior
architecture comparisons, this run is an adaptive benchmark.  A final unbiased
claim requires an external or newly held-out confirmation set.
