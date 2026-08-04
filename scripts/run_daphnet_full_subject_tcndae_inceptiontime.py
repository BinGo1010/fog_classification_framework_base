from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_daphnet_full_subject_nbm_residual_binary as exp
import run_daphnet_full_subject_nbm_residual_inceptiontime as inception
from cnbr_fog.data import DaphnetDataset


EXPERIMENT = "daphnet_full_subject_tcndae_inceptiontime_v1"
METHOD_NAMES = {
    "B0": "Raw-InceptionTime",
    "B1": "TCNDAE-R-InceptionTime",
    "B2": "TCNDAE-R5-InceptionTime",
    "B3": "Raw+TCNDAE-R5-InceptionTime",
}
METHOD_DIRS = {
    "B0": "B0_raw_inceptiontime",
    "B1": "B1_tcndae_residual_inceptiontime",
    "B2": "B2_tcndae_r5_inceptiontime",
    "B3": "B3_raw_tcndae_r5_inceptiontime",
}
RESUME_ALLOWED = True


def get_group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class TCNResidualBlock(nn.Module):
    """Centered, non-causal, same-length dilated residual block."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation,
                      padding=dilation, bias=False),
            nn.GroupNorm(get_group_count(channels), channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation,
                      padding=dilation, bias=False),
            nn.GroupNorm(get_group_count(channels), channels),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.network(inputs))


class TCNStack(nn.Module):
    def __init__(self, channels: int, dilations: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        self.channels = int(channels)
        self.dilations = tuple(int(value) for value in dilations)
        self.blocks = nn.Sequential(*(
            TCNResidualBlock(channels, dilation, dropout) for dilation in self.dilations
        ))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(inputs)


class TCNDAE(nn.Module):
    """Hierarchical non-causal TCN denoising autoencoder from the supplied design."""

    def __init__(self, input_channels: int = 9, latent_channels: int = 32,
                 dropout: float = 0.10) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.latent_channels = int(latent_channels)
        self.dropout = float(dropout)
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 48, kernel_size=7, stride=1, padding=3),
            nn.GroupNorm(get_group_count(48), 48), nn.GELU(),
        )
        self.encoder_stage1 = TCNStack(48, (1, 2), dropout)
        self.downsample1 = nn.Sequential(
            nn.Conv1d(48, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(get_group_count(64), 64), nn.GELU(),
        )
        self.encoder_stage2 = TCNStack(64, (1, 2, 4), dropout)
        self.downsample2 = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(get_group_count(96), 96), nn.GELU(),
        )
        self.encoder_stage3 = TCNStack(96, (1, 2, 4, 8), dropout)
        self.to_latent = nn.Sequential(
            nn.Conv1d(96, latent_channels, kernel_size=1),
            nn.GroupNorm(get_group_count(latent_channels), latent_channels), nn.GELU(),
        )
        self.from_latent = nn.Sequential(
            nn.Conv1d(latent_channels, 96, kernel_size=1),
            nn.GroupNorm(get_group_count(96), 96), nn.GELU(),
        )
        self.decoder_stage3 = TCNStack(96, (1, 2, 4), dropout)
        self.upsample2_conv = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=5, padding=2),
            nn.GroupNorm(get_group_count(64), 64), nn.GELU(),
        )
        self.decoder_stage2 = TCNStack(64, (1, 2, 4), dropout)
        self.upsample1_conv = nn.Sequential(
            nn.Conv1d(64, 48, kernel_size=5, padding=2),
            nn.GroupNorm(get_group_count(48), 48), nn.GELU(),
        )
        self.decoder_stage1 = TCNStack(48, (1, 2), dropout)
        self.output_head = nn.Sequential(
            nn.Conv1d(48, 24, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(24, input_channels, kernel_size=1),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.encoder_stage1(self.stem(inputs))
        features = self.encoder_stage2(self.downsample1(features))
        features = self.encoder_stage3(self.downsample2(features))
        return self.to_latent(features)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.decoder_stage3(self.from_latent(latent))
        features = F.interpolate(features, scale_factor=2, mode="linear", align_corners=False)
        features = self.decoder_stage2(self.upsample2_conv(features))
        features = F.interpolate(features, scale_factor=2, mode="linear", align_corners=False)
        features = self.decoder_stage1(self.upsample1_conv(features))
        return self.output_head(features)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[1] != self.input_channels or inputs.shape[2] != 128:
            raise ValueError(f"expected [B,{self.input_channels},128], got {tuple(inputs.shape)}")
        latent = self.encode(inputs)
        reconstruction = self.decode(latent)
        if reconstruction.shape != inputs.shape:
            raise RuntimeError(f"reconstruction {tuple(reconstruction.shape)} != input {tuple(inputs.shape)}")
        return reconstruction, latent

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": "hierarchical_noncausal_tcndae",
            "input_shape": ["batch", self.input_channels, 128],
            "latent_shape": ["batch", self.latent_channels, 32],
            "dropout": self.dropout,
            "encoder_dilations": [[1, 2], [1, 2, 4], [1, 2, 4, 8]],
            "decoder_dilations": [[1, 2, 4], [1, 2, 4], [1, 2]],
            "downsampling": "two stride-2 convolutions",
            "upsampling": "linear interpolation followed by convolution",
            "normalization": "GroupNorm",
            "activation": "GELU",
            "output_activation": None,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_nbm(inputs: np.ndarray, item: exp.SubjectWindows, candidate_indices: np.ndarray,
              scaler: exp.RobustScaler, run_dir: Path, seed: int, device: torch.device,
              max_epochs: int, patience: int) -> tuple[nn.Module, dict[str, Any]]:
    del inputs
    checkpoint = run_dir / "nbm_best.pt"
    log_path = run_dir / "training_log_nbm.csv"
    resume_path = run_dir / "tcndae_resume.pt"
    model = TCNDAE().to(device)
    if checkpoint.exists() and log_path.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        return model, dict(payload["training"])
    run_dir.mkdir(parents=True, exist_ok=True)
    train_indices, val_indices = exp.nbm_train_validation(item, candidate_indices)
    train_x = scaler.transform(item.raw[train_indices])
    val_x = scaler.transform(item.raw[val_indices])
    exp.seed_everything(seed)
    model = TCNDAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    batches = exp.a1b.pair_loader(train_x, train_x, shuffle=True, seed=seed, workers=0)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    bad_epochs = 0
    last_epoch = 0
    history: list[dict[str, Any]] = []
    elapsed_before = 0.0
    if RESUME_ALLOWED and resume_path.exists():
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        best_loss = float(payload["best_loss"])
        best_state = payload["best_state"]
        best_epoch = int(payload["best_epoch"])
        bad_epochs = int(payload["bad_epochs"])
        last_epoch = int(payload["last_epoch"])
        history = list(payload["history"])
        elapsed_before = float(payload.get("elapsed_seconds", 0.0))
        if batches.generator is not None and payload.get("loader_generator_state") is not None:
            batches.generator.set_state(payload["loader_generator_state"].cpu())
        if payload.get("torch_rng_state") is not None:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        if device.type == "cuda" and payload.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu(), device=device)
        print(f"RESUME NBM {run_dir} epoch={last_epoch + 1}/{max_epochs}", flush=True)
    started = time.perf_counter()
    first_epoch = max_epochs + 1 if bad_epochs >= patience else last_epoch + 1
    for epoch in range(first_epoch, max_epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for batch_x, batch_y in batches:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(batch_x)
            loss = exp.a1b.structural_loss("L4", reconstruction, batch_y)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient):
                raise FloatingPointError("non-finite TCN-DAE gradient")
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
        validation_loss = exp.a1b.evaluate_loss(model, val_x, val_x, "L4", device)
        improved = validation_loss < best_loss - 1e-8
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = exp.base.clone_state(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
        last_epoch = epoch
        history.append({"epoch": epoch, "train_loss": total_loss / count,
                        "validation_loss": validation_loss, "improved": improved,
                        "bad_epochs": bad_epochs})
        atomic_torch_save({
            "model_state": exp.base.clone_state(model),
            "optimizer_state": optimizer.state_dict(),
            "best_state": best_state,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "last_epoch": last_epoch,
            "history": history,
            "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            "loader_generator_state": (batches.generator.get_state()
                                       if batches.generator is not None else None),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (torch.cuda.get_rng_state(device) if device.type == "cuda" else None),
        }, resume_path)
        if bad_epochs >= patience:
            break
    if best_state is None:
        raise AssertionError("TCN-DAE produced no best checkpoint")
    training = {
        "seed": seed,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "best_validation_loss": best_loss,
        "elapsed_seconds": elapsed_before + time.perf_counter() - started,
        "train_windows": len(train_x),
        "validation_windows": len(val_x),
        "loss": "L4",
        "architecture": model.architecture_config(),
        "resumed": first_epoch > 1,
    }
    atomic_torch_save({
        "model_state": best_state,
        "training": training,
        "train_window_keys": item.keys[train_indices].tolist(),
        "validation_window_keys": item.keys[val_indices].tolist(),
    }, checkpoint)
    exp.write_csv(log_path, history)
    model.load_state_dict(best_state)
    resume_path.unlink(missing_ok=True)
    return model, training


def configure_pipeline(resume: bool = True) -> None:
    global RESUME_ALLOWED
    RESUME_ALLOWED = bool(resume)
    inception.RESUME_ALLOWED = bool(resume)
    exp.EXPERIMENT = EXPERIMENT
    exp.METHOD_NAMES = dict(METHOD_NAMES)
    exp.METHOD_DIRS = dict(METHOD_DIRS)
    exp.train_nbm = train_nbm
    exp.train_classifier = inception.train_classifier


def collect_folds(data_dir: Path) -> tuple[dict[str, exp.SubjectWindows], list[tuple[str, dict[str, Any]]]]:
    dataset = DaphnetDataset.load(data_dir)
    items = {subject: exp.build_subject_windows(dataset, subject) for subject in exp.SUBJECTS}
    folds: list[tuple[str, dict[str, Any]]] = []
    for subject, item in items.items():
        folds.extend((subject, fold) for fold in exp.outer_folds(item))
    return items, folds


def write_split_summary(root: Path, folds: list[tuple[str, dict[str, Any]]],
                        items: dict[str, exp.SubjectWindows]) -> None:
    rows = []
    for subject, fold in folds:
        item = items[subject]
        rows.append({
            "subject_id": subject, "fold_id": fold["fold_id"], "mode": fold["mode"],
            "train_windows": len(fold["train"]), "test_windows": len(fold["test"]),
            "test_positive_windows": int(np.sum(item.label[fold["test"]])),
            "test_negative_windows": int(len(fold["test"]) - np.sum(item.label[fold["test"]])),
        })
    exp.write_csv(root / "splits" / "outer_folds" / "outer_fold_summary.csv", rows)


def create_balanced_plan(root: Path, devices: list[str]) -> Path:
    rows = exp.read_csv(root / "splits" / "outer_folds" / "outer_fold_summary.csv")
    if len(rows) != 30:
        raise ValueError(f"expected 30 outer folds, found {len(rows)}")
    workers = [{"device": device, "folds": [], "estimated_train_windows": 0}
               for device in devices]
    for row in sorted(rows, key=lambda value: int(value["train_windows"]), reverse=True):
        target = min(workers, key=lambda value: (value["estimated_train_windows"], len(value["folds"])))
        target["folds"].append(f"{row['subject_id']}/{row['fold_id']}")
        target["estimated_train_windows"] += int(row["train_windows"])
    plan_path = root / "splits" / "tcndae_inceptiontime_8gpu_plan.json"
    exp.write_json(plan_path, {"strategy": "greedy_lpt_by_outer_train_windows",
                               "workers": {str(index): value for index, value in enumerate(workers)}})
    return plan_path


def validate_parallel_devices(devices: list[str]) -> None:
    if len(devices) != len(set(devices)):
        raise ValueError(f"duplicate devices are not allowed: {devices}")
    if not torch.cuda.is_available():
        raise RuntimeError("parallel CUDA launch requested but torch.cuda.is_available() is false")
    visible = torch.cuda.device_count()
    for value in devices:
        device = torch.device(value)
        if device.type != "cuda" or device.index is None:
            raise ValueError(f"parallel worker device must be explicit cuda:N, got {value}")
        if device.index >= visible:
            raise ValueError(f"requested {value}, but only {visible} CUDA devices are visible")


def handle_termination(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def progress_summary(root: Path) -> dict[str, Any]:
    completed_by_method = {
        method: sum(1 for path in (root / METHOD_DIRS[method]).glob("*/*/seed*/run_metrics.json")
                    if (path.parent / "test_predictions.csv").exists())
        for method in exp.METHODS
    }
    completed = sum(completed_by_method.values())
    return {
        "output_root": str(root), "completed_runs": completed, "total_runs": 360,
        "completion_percent": 100.0 * completed / 360.0,
        "completed_by_method": completed_by_method,
        "tcndae_epoch_checkpoints": len(list(root.glob("splits/**/tcndae_resume.pt"))),
        "classifier_epoch_checkpoints": len(list(root.glob("B*/**/inceptiontime_resume.pt"))),
        "completed_nbm_models": len(list(root.glob("splits/**/nbm_best.pt"))),
        "expected_nbm_models": 120,
        "ready_to_finalize": completed == 360,
    }


def write_protocol(root: Path, config: Path) -> None:
    exp.write_json(root / "splits" / "frozen_protocol_tcndae_inceptiontime.json", {
        "experiment": EXPERIMENT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config.resolve()),
        "subjects": list(exp.SUBJECTS),
        "outer": "leave-one-complete-valid-record-out",
        "inner": "3-fold record-first purged OOF",
        "nbm": "hierarchical_noncausal_tcndae",
        "nbm_seed_fixed": exp.NBM_SEED,
        "classifier": "InceptionTime-6module",
        "classifier_seeds": list(exp.SEEDS),
        "test_used_for_selection": False,
        "resume_granularity": "TCN-DAE epoch, InceptionTime epoch, completed method-seed run",
    })


def worker_command(args: argparse.Namespace, root: Path, device: str, index: int,
                   worker_count: int, plan: Path) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--data-dir", str(args.data_dir.resolve()), "--output-root", str(root),
        "--config", str(args.config.resolve()), "--device", device,
        "--threads", str(args.threads), "--nbm-max-epochs", str(args.nbm_max_epochs),
        "--nbm-patience", str(args.nbm_patience),
        "--classifier-max-epochs", str(args.classifier_max_epochs),
        "--classifier-patience", str(args.classifier_patience),
        "--bootstrap-samples", str(args.bootstrap_samples),
        "--shard-index", str(index), "--shard-count", str(worker_count),
        "--fold-plan", str(plan),
    ]
    if args.no_resume:
        command.append("--no-resume")
    return command


def launch_parallel(args: argparse.Namespace, root: Path, plan: Path) -> None:
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    log_dir = root / "logs" / "parallel_workers"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = set(range(len(devices)))
    attempts = {index: 0 for index in pending}
    statuses: dict[int, dict[str, Any]] = {}
    running: dict[int, tuple[subprocess.Popen[Any], Any, Any]] = {}
    try:
        while pending:
            running = {}
            for index in sorted(pending):
                attempts[index] += 1
                out_path = log_dir / f"worker{index}_attempt{attempts[index]}.out.log"
                err_path = log_dir / f"worker{index}_attempt{attempts[index]}.err.log"
                out_handle = out_path.open("a", encoding="utf-8")
                err_handle = err_path.open("a", encoding="utf-8")
                options: dict[str, Any] = {"cwd": str(ROOT), "stdout": out_handle, "stderr": err_handle}
                if os.name == "nt":
                    options["creationflags"] = subprocess.CREATE_NO_WINDOW
                else:
                    options["start_new_session"] = True
                process = subprocess.Popen(
                    worker_command(args, root, devices[index], index, len(devices), plan), **options
                )
                running[index] = (process, out_handle, err_handle)
                statuses[index] = {"device": devices[index], "pid": process.pid,
                                   "attempt": attempts[index], "state": "running",
                                   "stdout": str(out_path), "stderr": str(err_path)}
                print(f"LAUNCH worker={index} device={devices[index]} pid={process.pid}", flush=True)
            exp.write_json(log_dir / "parallel_status.json", statuses)
            failed: set[int] = set()
            for index, (process, out_handle, err_handle) in running.items():
                code = process.wait()
                out_handle.close()
                err_handle.close()
                statuses[index].update({"return_code": code,
                                        "state": "complete" if code == 0 else "failed"})
                if code != 0:
                    failed.add(index)
                print(f"EXIT worker={index} device={devices[index]} code={code}", flush=True)
            exp.write_json(log_dir / "parallel_status.json", statuses)
            exhausted = [index for index in failed if attempts[index] > args.max_retries]
            if exhausted:
                raise RuntimeError(f"workers exhausted retries: {exhausted}; inspect {log_dir}")
            pending = failed
    except KeyboardInterrupt:
        for process, _, _ in running.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.time() + 20
        for process, out_handle, err_handle in running.values():
            try:
                process.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                process.kill()
            out_handle.close()
            err_handle.close()
        print("INTERRUPTED: checkpoints retained; rerun the identical command to resume", flush=True)
        raise


def rewrite_report(root: Path) -> None:
    generic = root / "reports" / "full_subject_binary_classification_report.md"
    target = root / "reports" / "full_subject_tcndae_inceptiontime_report.md"
    text = generic.read_text(encoding="utf-8-sig")
    text = text.replace("Daphnet 全被试 NBM 残差二分类实验报告",
                        "Daphnet 全被试 TCN-DAE 残差 + InceptionTime 二分类实验报告")
    text = text.replace("NBM/TCN训练", "TCN-DAE/InceptionTime训练")
    text = text.replace("TCN训练用残差", "InceptionTime训练用的TCN-DAE残差")
    text = text.replace("NBM固定种子20260802", "TCN-DAE固定种子20260802")
    text = text.replace("改变TCN初始化", "改变InceptionTime初始化")
    target.write_text(text, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs" / EXPERIMENT / "full_subject_binary_experiment")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs" / "daphnet_full_subject_tcndae_inceptiontime.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--nbm-max-epochs", type=int, default=2000)
    parser.add_argument("--nbm-patience", type=int, default=100)
    parser.add_argument("--classifier-max-epochs", type=int, default=50)
    parser.add_argument("--classifier-patience", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--devices", default=",".join(f"cuda:{index}" for index in range(8)))
    parser.add_argument("--launch-parallel", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--fold-plan", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--only-fold", default="")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_pipeline(resume=not args.no_resume)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.status_only:
        print(json.dumps(progress_summary(root), ensure_ascii=False, indent=2), flush=True)
        return
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: {device}")
        torch.cuda.set_device(device)
    if args.finalize_only:
        result = exp.aggregate_results(root, args.bootstrap_samples)
        rewrite_report(root)
        print(f"COMPLETE {root} runs=360", flush=True)
        return

    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(
            f"processed Daphnet data directory is missing: {data_dir}; pass the correct --data-dir"
        )
    items, folds = collect_folds(data_dir)
    if not args.worker:
        write_split_summary(root, folds, items)
        write_protocol(root, args.config)
    if args.launch_parallel and not args.worker:
        devices = [value.strip() for value in args.devices.split(",") if value.strip()]
        if not devices:
            raise ValueError("--devices must contain at least one device")
        validate_parallel_devices(devices)
        signal.signal(signal.SIGTERM, handle_termination)
        plan = create_balanced_plan(root, devices)
        launch_parallel(args, root, plan)
        result = exp.aggregate_results(root, args.bootstrap_samples)
        rewrite_report(root)
        best = max(exp.METHODS,
                   key=lambda method: next(row for row in result["macro_results"]
                                           if row["method"] == method)["macro_pr_auc"])
        print(f"COMPLETE {root} best={best}", flush=True)
        return

    selected = folds
    if args.only_fold:
        wanted_subject, wanted_fold = args.only_fold.split("/", 1)
        selected = [(subject, fold) for subject, fold in selected
                    if subject == wanted_subject and str(fold["fold_id"]) == wanted_fold]
        if len(selected) != 1:
            raise ValueError(f"unknown --only-fold {args.only_fold}")
    elif args.fold_plan is not None:
        plan = json.loads(args.fold_plan.resolve().read_text(encoding="utf-8"))
        wanted = set(plan["workers"][str(args.shard_index)]["folds"])
        selected = [(subject, fold) for subject, fold in selected
                    if f"{subject}/{fold['fold_id']}" in wanted]
        if len(selected) != len(wanted):
            raise ValueError(f"fold plan mismatch for worker {args.shard_index}")
    elif args.shard_count > 1:
        selected = [entry for index, entry in enumerate(selected)
                    if index % args.shard_count == args.shard_index]
    if args.smoke:
        selected = selected[:1]
    for position, (subject, fold) in enumerate(selected, 1):
        print(f"OUTER {position}/{len(selected)} {subject}/{fold['fold_id']} device={device}", flush=True)
        exp.run_outer_fold(
            items[subject], fold, root, device,
            min(args.nbm_max_epochs, 2) if args.smoke else args.nbm_max_epochs,
            min(args.nbm_patience, 1) if args.smoke else args.nbm_patience,
            min(args.classifier_max_epochs, 2) if args.smoke else args.classifier_max_epochs,
            min(args.classifier_patience, 1) if args.smoke else args.classifier_patience,
        )
    if args.smoke or args.only_fold or args.shard_count > 1:
        print(f"PARTIAL COMPLETE {root}", flush=True)
        return
    result = exp.aggregate_results(root, args.bootstrap_samples)
    rewrite_report(root)
    print(f"COMPLETE {root} runs=360", flush=True)


if __name__ == "__main__":
    main()
