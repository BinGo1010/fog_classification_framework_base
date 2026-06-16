from pathlib import Path

from torch.utils.data import DataLoader

from .registry import build_dataset
from . import channel_subset_npz_dataset  # noqa: F401, ensure registration


CHANNEL_SELECTION_KEYS = [
    "channel_indices",
    "channel_names",
    "imu_positions",
    "sensor_types",
    "axes",
]


def _dataset_kwargs(dcfg, file_path, mean=None, std=None):
    kwargs = {
        "file_path": file_path,
        "x_key": dcfg.get("x_key", "X"),
        "y_key": dcfg.get("y_key", "y"),
        "input_format": dcfg.get("input_format", "NCT"),
        "normalize": dcfg.get("normalize", "none"),
        "metadata_keys": dcfg.get("metadata_keys", []),
        "mean": mean,
        "std": std,
    }
    for key in CHANNEL_SELECTION_KEYS:
        if key in dcfg:
            kwargs[key] = dcfg[key]
    return kwargs


def build_channel_subset_dataloaders(cfg):
    dcfg = cfg["data"]
    root = Path(dcfg["root"])
    train_set = build_dataset(
        dcfg["name"],
        **_dataset_kwargs(dcfg, root / dcfg["train_file"]),
    )
    val_set = build_dataset(
        dcfg["name"],
        **_dataset_kwargs(
            dcfg,
            root / dcfg["val_file"],
            mean=getattr(train_set, "mean", None),
            std=getattr(train_set, "std", None),
        ),
    )
    test_set = build_dataset(
        dcfg["name"],
        **_dataset_kwargs(
            dcfg,
            root / dcfg["test_file"],
            mean=getattr(train_set, "mean", None),
            std=getattr(train_set, "std", None),
        ),
    )
    common = dict(
        batch_size=dcfg.get("batch_size", 64),
        num_workers=dcfg.get("num_workers", 0),
        pin_memory=dcfg.get("pin_memory", True),
    )
    return {
        "train": DataLoader(train_set, shuffle=True, drop_last=False, **common),
        "val": DataLoader(val_set, shuffle=False, drop_last=False, **common),
        "test": DataLoader(test_set, shuffle=False, drop_last=False, **common),
        "train_set": train_set,
        "val_set": val_set,
        "test_set": test_set,
    }
