from __future__ import annotations

import csv
import re
from itertools import combinations
from pathlib import Path

import numpy as np


def parse_channel_name(name):
    parts = str(name).split("_")
    if len(parts) != 3:
        raise ValueError(f"Expected channel name like ankleL_acc_x, got {name}")
    return parts[0], parts[1], parts[2]


def load_sensor_columns_from_config(cfg):
    dcfg = cfg["data"]
    path = Path(dcfg["root"]) / dcfg.get("train_file", "train.npz")
    data = np.load(path, allow_pickle=True)
    if "sensor_columns" not in data:
        raise KeyError(f"'sensor_columns' is required in {path}")
    return [str(name) for name in data["sensor_columns"]]


def group_imu_channels(sensor_columns):
    groups = {}
    for idx, name in enumerate(sensor_columns):
        imu, _, _ = parse_channel_name(name)
        groups.setdefault(imu, []).append(idx)
    sizes = {imu: len(indices) for imu, indices in groups.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"All IMUs must have the same channel count, got {sizes}")
    return groups


def imu_positions_from_columns(sensor_columns):
    return list(group_imu_channels(sensor_columns).keys())


def channels_for_imus(sensor_columns, imu_positions):
    imu_positions = set(imu_positions)
    selected = []
    for name in sensor_columns:
        imu, _, _ = parse_channel_name(name)
        if imu in imu_positions:
            selected.append(name)
    if not selected:
        raise ValueError(f"No channels selected for IMUs: {sorted(imu_positions)}")
    return selected


def imu_combinations(sensor_columns, min_imus=1, max_imus=None):
    positions = imu_positions_from_columns(sensor_columns)
    max_imus = max_imus or len(positions)
    for k in range(int(min_imus), int(max_imus) + 1):
        for combo in combinations(positions, k):
            name = "imu_" + "_".join(combo)
            yield name, list(combo), channels_for_imus(sensor_columns, combo)


def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def scalar_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and value is not None
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
