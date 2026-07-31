# FOG Classification Framework

This project is a modular framework for Freezing of Gait classification from
wearable IMU time-series windows.

## Structure

```text
configs/          YAML experiment configs
data_provider/    datasets, dataloaders, channel selection, augmentation
exp/              training, evaluation, LOSO, SupCon, SimCLR experiments
layers/           reusable neural network blocks
losses/           CE, focal, SupCon, SimCLR/NT-Xent, sensor-cost losses
models/           model registry and model definitions
utils/            config, IO, metrics, seed, profiling helpers
run.py            unified experiment entry point
tests/            smoke tests
```

Imports use these top-level modules directly.

## Install

```bash
pip install -e .
```

## Quick Run

```bash
python run.py --config configs/default.yaml
```

## Daphnet GRU-H200 Residual Feasibility

The staged causal normal-forecast/residual-fusion experiment is available at
`scripts/run_daphnet_gru_residual_feasibility.py`. It implements Phase 0/1/2
screening and leakage-safe Phase 3A/3B subject cross-fitting without modifying
the completed H200 source suite. See the
[Chinese runbook](docs/DAPHNET_GRU_RESIDUAL_FEASIBILITY_RUNBOOK.md) for commands,
gates, resume behavior, and output layout.

Run the unified experiment template:

```bash
python run.py --config configs/base_experiment.yaml --run_name cnn1d_win120
```

Switch model, split strategy, or contrastive method with `run_name`:

```bash
python run.py --config configs/base_experiment.yaml --run_name transformer_win120
python run.py --config configs/base_experiment.yaml --run_name cnn1d_random_window_win120
python run.py --config configs/base_experiment.yaml --run_name cnn1d_loso_win120
python run.py --config configs/base_experiment.yaml --run_name cnn1d_supcon_win120
python run.py --config configs/base_experiment.yaml --run_name cnn1d_simclr_win120
```

Window-length ablations are generated from the source CSV files and cached
automatically when `data.windowing.enabled` is true:

```bash
python run.py --config configs/base_experiment.yaml --run_name cnn1d_win60
python run.py --config configs/base_experiment.yaml --run_name cnn1d_win240
python run.py --config configs/base_experiment.yaml --run_name cnn1d_win120 --window_size 180 --stride 90
```

## Multi-GPU Training

Single-GPU and CPU runs keep using `python run.py ...`. On a multi-GPU server,
launch DDP with `torchrun`; `batch_size` is per process/GPU:

```bash
torchrun --standalone --nproc_per_node=8 run.py --config configs/base_experiment.yaml --run_name cnn1d_loso_win120 --model LightweightIMUTCN --exp_mode loso --output_dir outputs/lighttcn_loso_ordinary
```

For SupCon or SimCLR, use the matching contrastive model name and keep
`ddp_find_unused_parameters` enabled if your model has heads that are unused in
one training stage:

```bash
torchrun --standalone --nproc_per_node=8 run.py --config configs/base_experiment.yaml --run_name cnn1d_supcon_win120 --override train.ddp_find_unused_parameters=true
```

## Forecasting-Style Models

The following forecasting-library backbones are adapted to the FOG classifier
API and can run as ordinary, SupCon, or SimCLR models:

```text
iTransformer, TimesNet, NonstationaryTransformer, Informer, Autoformer, Vit/ViT
```

Use the plain name for ordinary training and the `SupCon*` / `SimCLR*` prefix
for contrastive training:

```bash
python run.py --config configs/base_experiment.yaml --run_name cnn1d_loso_win120 --model iTransformer --exp_mode loso
python run.py --config configs/base_experiment.yaml --run_name cnn1d_loso_win120 --model SupConITransformer --exp_mode loso
python run.py --config configs/base_experiment.yaml --run_name cnn1d_loso_win120 --model SimCLRITransformer --exp_mode loso
```

The same pattern applies to `TimesNet`, `NonstationaryTransformer`,
`Informer`, `Autoformer`, and `Vit`/`ViT`.

## Add A Model

Add a file under `models/` and register it:

```python
from models.registry import register_model

@register_model("MyModel")
class MyModel(nn.Module):
    ...
```

Then set:

```yaml
model:
  name: MyModel
```

## Add A Dataset

Add a dataset under `data_provider/` and register it:

```python
from data_provider.registry import register_dataset

@register_dataset("MyDataset")
class MyDataset(torch.utils.data.Dataset):
    ...
```

Then set:

```yaml
data:
  name: MyDataset
```
