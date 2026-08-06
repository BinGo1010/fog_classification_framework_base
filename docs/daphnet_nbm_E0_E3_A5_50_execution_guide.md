# Daphnet NBM E0--E3 / A5_50 execution guide

## Implemented entry points

- Runner: `scripts/run_daphnet_nbm_e0_e3_a5_50.py`
- Seven-GPU launcher: `scripts/run_daphnet_nbm_e0_e3_7gpu.sh`
- Cross-shard aggregator: `scripts/aggregate_daphnet_nbm_e0_e3_a5_50_shards.py`
- Models and metrics: `cnbr_fog/nbm_e0_e3.py`
- Tests: `tests/test_daphnet_nbm_e0_e3_a5_50.py`
- Dataset: `dataset/1.Daphnet Freezing of Gait Dataset/processed_A5_50`

The runner implements E0 full-A5 reproduction, E1 C1-MAD, E2-P24, optional
E2-P16, E3-A and optional E3-B.  It writes V1--V4 metrics, Q95/Q99.2 window
and event thresholds, common-support comparisons, paired subject statistics,
gates and Markdown reports.  Completed checkpoints are resumed automatically.

E0 has two evaluation tracks:

1. `*_a5_full` exactly reproduces the original A5_50 target windows.
2. Unsuffixed files use the E3-A-history-eligible common target IDs for paired
   E0--E3 comparisons.

This prevents E3 history availability from silently changing the test support.

## Recommended commands

Short non-conclusive end-to-end validation:

```powershell
conda run -n pd_fog python scripts\run_daphnet_nbm_e0_e3_a5_50.py --smoke --include-e2-p16 --include-e3b --device cuda
```

Main pre-registered experiment (E0, E1, E2-P24, E3-A):

```powershell
conda run -n pd_fog python scripts\run_daphnet_nbm_e0_e3_a5_50.py --device cuda
```

Full sensitivity experiment including P16 and E3-B:

```powershell
conda run -n pd_fog python scripts\run_daphnet_nbm_e0_e3_a5_50.py --include-e2-p16 --include-e3b --device cuda
```

Run only through a gate:

```powershell
conda run -n pd_fog python scripts\run_daphnet_nbm_e0_e3_a5_50.py --stop-after E0 --device cuda
conda run -n pd_fog python scripts\run_daphnet_nbm_e0_e3_a5_50.py --stop-after E1 --device cuda
conda run -n pd_fog python scripts\run_daphnet_nbm_e0_e3_a5_50.py --stop-after E2 --device cuda
```

Local runtime benchmark:

```powershell
conda run -n pd_fog python scripts\run_daphnet_nbm_e0_e3_a5_50.py --benchmark-only --subjects S01 --include-e3b --benchmark-epochs 10 --output-root outputs\daphnet_nbm_E0_E3_A5_50_runtime_benchmark_v1 --device cuda
```

## Measured local throughput

Measured on NVIDIA GeForce RTX 4070 Ti, batch size 64, S01, 512 training
windows, using 10 epochs:

| Setting | Parameters | Seconds/epoch |
|---|---:|---:|
| E0 M3 | 64,633 | 0.448 |
| E2-P24 | 39,457 | 0.276 |
| E2-P16 | 77,865 | 0.330 |
| E3-A C24 | 63,961 | 1.108 |
| E3-A C48 | 115,273 | 1.067 |
| E3-B C24 | 64,185 | 1.153 |
| E3-B C48 | 115,497 | 1.123 |

The existing A5_50 formal seven-subject E0 checkpoints are reused.  E0 must
still train S03, S04 and S10 (nine runs) for the ten-subject V1 audit.

Using 250--500 epochs as a planning interval (the existing A5 M3 runs had
median 317 and mean 341 final epochs), estimated sequential GPU training is:

| Work | Estimated time |
|---|---:|
| E0 auxiliary 3 subjects x 3 seeds | 0.4--0.8 h |
| E1 | no NBM retraining |
| E2-P24, 10 subjects x 3 seeds | 0.6--1.3 h |
| E3-A, 10 subjects x 3 seeds | 2.2--4.7 h |
| Main E0--E3 path total | about 3.3--6.7 h training; allow 4--7.5 h end-to-end |
| Optional E2-P16 | add 0.8--1.5 h |
| Optional E3-B | add 2.4--4.8 h |
| Main plus both optional branches | about 7--14 h end-to-end |

The range is dominated by unknown early-stopping epochs.  Evaluation, compressed
array writing, figures and paired bootstrap add overhead not represented by the
per-epoch benchmark.

## Local versus server

- The main E0--E3 path is reasonable as an overnight/local RTX 4070 Ti run.
- The full P16 plus E3-B sensitivity suite is better placed on a server.
- The base runner remains single-process and single-GPU.  The seven-GPU launcher
  assigns disjoint subjects to disjoint output roots and the aggregator rebuilds
  one authoritative population-level result.
- E1 must wait for E0, and E3 capacity selection must wait for the E2 gate.
- Do not parallelize stages in a way that lets E3 start before the E2 gate is
  frozen.

## Seven RTX 2080 Ti execution

No DDP or model-architecture change is required.  The models are small and each
subject/seed is independent, so subject sharding is more efficient and less
fragile than synchronizing seven GPUs for each mini-batch.  Keep batch size 64;
11 GB per 2080 Ti is sufficient for the implemented E0--E3 models.  Mixed
precision is intentionally not enabled because the registered experiment and
local reference use FP32.

The workload-balanced mapping is:

| GPU | Subjects |
|---:|---|
| 0 | S04 |
| 1 | S10 |
| 2 | S06 |
| 3 | S01 |
| 4 | S07, S08 |
| 5 | S02, S05 |
| 6 | S03, S09 |

The launcher has a mandatory barrier:

1. Seven shards run E0/E1/E2.
2. The global aggregator recomputes score selection and the E2 gate over all
   subjects, then writes `E3_capacity_decision.json`.
3. Seven shards resume their cached checkpoints and train E3 with the frozen
   global capacity.
4. The final aggregator recomputes all official E0--E3 tables and gates.

Run the main registered path from the repository root:

```bash
CONDA_ENV=pd_fog bash scripts/run_daphnet_nbm_e0_e3_7gpu.sh all
```

Run the full P16 and E3-B sensitivity suite:

```bash
CONDA_ENV=pd_fog INCLUDE_E2_P16=1 INCLUDE_E3B=1 \
  bash scripts/run_daphnet_nbm_e0_e3_7gpu.sh all
```

Preview all commands without starting training:

```bash
DRY_RUN=1 CONDA_ENV=pd_fog bash scripts/run_daphnet_nbm_e0_e3_7gpu.sh all
```

The two phases can be resumed independently:

```bash
CONDA_ENV=pd_fog bash scripts/run_daphnet_nbm_e0_e3_7gpu.sh phase1
CONDA_ENV=pd_fog bash scripts/run_daphnet_nbm_e0_e3_7gpu.sh phase2
```

Estimated wall time on seven RTX 2080 Ti cards is about 1.5--3 hours for the
main E0/E1/E2-P24/E3-A path and about 2.5--5 hours when both P16 and E3-B are
enabled.  This is a planning range extrapolated from the measured RTX 4070 Ti
benchmark; actual early-stop epochs, CPU speed and compressed-result I/O are the
largest uncertainties.  Copying the existing A5_50 reference/checkpoint output
to the same relative path avoids retraining reusable E0 runs.

## Probe limitation

No pre-registered A5_50-compatible frozen Raw-TCN checkpoint was supplied.
Therefore the runner writes the E2/E3 probe tables as `NE` rather than silently
reusing a classifier trained with incompatible splits.  E3 cannot receive a
fully complete mechanism PASS until a compatible independent probe is frozen.
