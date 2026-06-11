from __future__ import annotations

import argparse
import copy
from typing import Any

from data_provider.window_cache import prepare_window_dataset
from exp import build_experiment
from utils.config import load_config
from utils.distributed import barrier, cleanup_distributed, init_distributed, is_main_process


CLI_TO_CONFIG = {
    "task_name": "project.task_name",
    "is_training": "project.is_training",
    "model_id": "project.model_id",
    "output_dir": "project.output_dir",
    "seed": "project.seed",
    "device": "project.device",
    "model": "model.name",
    "data": "data.name",
    "root_path": "data.root",
    "data_path": "data.data_path",
    "features": "data.features",
    "seq_len": "model.seq_len",
    "pred_len": "model.pred_len",
    "num_class": "model.num_classes",
    "enc_in": "model.in_channels",
    "dec_in": "model.dec_in",
    "c_out": "model.c_out",
    "d_model": "model.d_model",
    "n_heads": "model.nhead",
    "e_layers": "model.num_layers",
    "d_layers": "model.d_layers",
    "d_ff": "model.dim_feedforward",
    "dropout": "model.dropout",
    "batch_size": "data.batch_size",
    "learning_rate": "train.lr",
    "train_epochs": "train.epochs",
    "patience": "train.early_stopping_patience",
    "loss": "train.loss",
    "lradj": "train.scheduler",
    "exp_mode": "experiment.mode",
    "imu_position": "data.imu_positions",
    "use_gumbel": "experiment.use_gumbel",
    "target_sensor_num": "experiment.target_sensor_num",
    "use_amp": "train.amp",
    "gpu": "project.device",
    "loso_root": "experiment.loso_root",
    "window_size": "data.windowing.window_size",
    "stride": "data.windowing.stride",
    "split_strategy": "data.windowing.split_strategy",
    "auto_window": "data.windowing.enabled",
    "distributed": "train.distributed",
    "ddp_find_unused_parameters": "train.ddp_find_unused_parameters",
}

MODEL_ALIASES = {
    "MobileOne1D": "MobileOne1DTiny",
    "LightTCN": "LightweightIMUTCN",
    "LSTMCls": "LSTMClassifier",
    "GRUCls": "GRUClassifier",
}

DATA_ALIASES = {
    "FOG": "ChannelSubsetNPZDataset",
    "NPZ": "NPZTimeSeriesDataset",
}

ALL_IMU_POSITIONS = ["ankleL", "ankleR", "back", "wrist"]


def _coerce_bool(value: Any):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return bool(value)
    text = str(value).lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return value


def _set_by_path(cfg: dict[str, Any], path: str, value: Any) -> None:
    cur = cfg
    parts = path.split(".")
    for key in parts[:-1]:
        cur = cur.setdefault(key, {})
    cur[parts[-1]] = value


def _parse_value(value: str) -> Any:
    if "," in value:
        return [_parse_value(part.strip()) for part in value.split(",") if part.strip()]
    if value.lower() in {"true", "false", "yes", "no", "y", "n"}:
        return _coerce_bool(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def apply_overrides(cfg: dict[str, Any], args) -> dict[str, Any]:
    for cli_name, path in CLI_TO_CONFIG.items():
        value = getattr(args, cli_name, None)
        if value is None:
            continue
        if cli_name == "model":
            value = MODEL_ALIASES.get(value, value)
        elif cli_name == "data":
            value = DATA_ALIASES.get(value, value)
        elif cli_name == "gpu":
            value = f"cuda:{value}"
        elif cli_name == "imu_position":
            if value == ["all"] or value == "all":
                value = ALL_IMU_POSITIONS
            else:
                value = value if isinstance(value, list) else [value]
        elif cli_name in {"use_amp", "use_gumbel", "auto_window", "distributed"}:
            value = _coerce_bool(value)
        elif cli_name == "is_training":
            value = int(value)
        _set_by_path(cfg, path, value)
    for override in args.override or []:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key, value = override.split("=", 1)
        _set_by_path(cfg, key, _parse_value(value))
    return cfg


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def iter_run_configs(cfg: dict[str, Any], args):
    runs = cfg.get("runs")
    if not runs:
        yield apply_overrides(cfg, args)
        return

    selected = set(args.run_name or [])
    defaults = cfg.get("defaults")
    if defaults is None:
        defaults = {key: value for key, value in cfg.items() if key != "runs"}
    for idx, run_cfg in enumerate(runs, start=1):
        name = run_cfg.get("name", f"run_{idx}")
        if selected and name not in selected:
            continue
        merged = deep_merge(defaults, run_cfg)
        merged.setdefault("project", {})
        merged["project"].setdefault("name", name)
        merged = apply_overrides(merged, args)
        yield merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modular FOG classification runner.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", action="append", help="Nested config override, for example train.epochs=5")
    parser.add_argument("--run_name", action="append", help="Run only named entries from a multi-run config.")

    parser.add_argument("--task_name", type=str)
    parser.add_argument("--is_training", type=int)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--root_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--features", type=str)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--pred_len", type=int)
    parser.add_argument("--num_class", type=int)
    parser.add_argument("--enc_in", type=int)
    parser.add_argument("--dec_in", type=int)
    parser.add_argument("--c_out", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--n_heads", type=int)
    parser.add_argument("--e_layers", type=int)
    parser.add_argument("--d_layers", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--train_epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--loss", type=str)
    parser.add_argument("--lradj", type=str)
    parser.add_argument("--exp_mode", type=str)
    parser.add_argument("--imu_position", nargs="+")
    parser.add_argument("--use_gumbel")
    parser.add_argument("--target_sensor_num", type=int)
    parser.add_argument("--use_amp")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--loso_root", type=str)
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--split_strategy", type=str, choices=["subject", "random_window", "loso"])
    parser.add_argument("--auto_window")
    parser.add_argument("--distributed", help="Use DDP when launched by torchrun: true/false/auto.")
    parser.add_argument("--ddp_find_unused_parameters")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    try:
        for run_cfg in iter_run_configs(cfg, args):
            run_cfg = init_distributed(run_cfg)
            if is_main_process(run_cfg):
                run_cfg = prepare_window_dataset(run_cfg)
            barrier(run_cfg)
            if not is_main_process(run_cfg):
                run_cfg = prepare_window_dataset(run_cfg)
            barrier(run_cfg)
            if is_main_process(run_cfg):
                print(f"\n===== Running {run_cfg['project'].get('name', 'experiment')} =====")
            experiment = build_experiment(run_cfg)
            experiment.run()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
