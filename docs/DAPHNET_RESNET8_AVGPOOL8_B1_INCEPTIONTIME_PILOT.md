# Daphnet ResNet8-AvgPool8 NBM + B1 InceptionTime pilot

## Frozen pilot scope

- Subjects: S01, S02, S03, S05, S06, S07, S08, S09.
- Outer evaluation: leave-one-valid-record-out; single-record subjects use five purged chronological blocks.
- NBM representation: three-fold purged OOF reconstruction for every classifier-training window.
- Classification input: B1 signed residual, `R = X - X_hat`, shape `[B, 9, 128]`.
- Seed: 20260802.
- NBM: maximum 3000 epochs, patience 100.
- InceptionTime: maximum 100 epochs, patience 15.
- Outer test data are excluded from scaling, NBM fitting, early stopping, class weighting, and threshold selection.

## NBM architecture

```text
Input [B,9,128]
  -> Conv7/s2 + ResBlock3                 [B,32,64]
  -> Down-ResBlock3 + ResBlock5           [B,48,32]
  -> AdaptiveAvgPool1d(8)                 [B,48,8]
  -> Flatten + LayerNorm                  [B,384]
  -> Linear(384,512) + GELU + Dropout     [B,512]
  -> Linear(512,1536) + GELU
  -> Reshape                              [B,48,32]
  -> interpolate x2 + Conv5 + ResBlock5   [B,32,64]
  -> interpolate x2 + Conv5 + ResBlock5   [B,24,128]
  -> Conv1                                [B,9,128]
```

The model has 1,058,865 trainable parameters and no encoder-decoder long skip.

## Single-GPU server command

Run from the repository root:

```bash
python scripts/run_daphnet_resnet8_avgpool8_b1_inceptiontime_pilot.py \
  --device cuda \
  --nbm-max-epochs 3000 \
  --nbm-patience 100 \
  --classifier-max-epochs 100 \
  --classifier-patience 15
```

Rerunning the same command resumes incomplete NBM training from the most recent ten-epoch checkpoint, resumes InceptionTime from its most recent epoch, and skips completed runs.

## Four-GPU sharded commands

Launch the following four commands with the same output root:

```bash
python scripts/run_daphnet_resnet8_avgpool8_b1_inceptiontime_pilot.py --device cuda:0 --shard-index 0 --shard-count 4
python scripts/run_daphnet_resnet8_avgpool8_b1_inceptiontime_pilot.py --device cuda:1 --shard-index 1 --shard-count 4
python scripts/run_daphnet_resnet8_avgpool8_b1_inceptiontime_pilot.py --device cuda:2 --shard-index 2 --shard-count 4
python scripts/run_daphnet_resnet8_avgpool8_b1_inceptiontime_pilot.py --device cuda:3 --shard-index 3 --shard-count 4
```

After all shards finish, aggregate the 30 outer folds:

```bash
python scripts/run_daphnet_resnet8_avgpool8_b1_inceptiontime_pilot.py --finalize-only --device cpu
```

The final report is written to `reports/b1_resnet8_avgpool8_inceptiontime_pilot_report.md` under the output root.
