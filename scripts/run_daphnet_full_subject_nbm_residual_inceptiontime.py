from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_daphnet_full_subject_nbm_residual_binary as exp
from cnbr_fog.data import DaphnetDataset


EXPERIMENT = "daphnet_full_subject_nbm_residual_inceptiontime_v1"
SOURCE_EXPERIMENT = "daphnet_full_subject_nbm_residual_binary_v1"
METHOD_NAMES = {
    "B0": "Raw-InceptionTime",
    "B1": "R-InceptionTime",
    "B2": "R5-InceptionTime",
    "B3": "Raw+R5-InceptionTime",
}
METHOD_DIRS = {
    "B0": "B0_raw_inceptiontime",
    "B1": "B1_residual_inceptiontime",
    "B2": "B2_r5_inceptiontime",
    "B3": "B3_raw_r5_inceptiontime",
}
KERNEL_SIZES = (9, 19, 39)
BOTTLENECK_CHANNELS = 32
BRANCH_CHANNELS = 32
MODULE_COUNT = 6
RESUME_ALLOWED = True


class InceptionModule(nn.Module):
    """One 1-D InceptionTime module preserving the temporal length."""

    def __init__(self, in_channels: int, bottleneck_channels: int = BOTTLENECK_CHANNELS,
                 branch_channels: int = BRANCH_CHANNELS,
                 kernel_sizes: tuple[int, int, int] = KERNEL_SIZES) -> None:
        super().__init__()
        if any(kernel % 2 == 0 for kernel in kernel_sizes):
            raise ValueError("InceptionTime kernels must be odd to preserve temporal length")
        self.in_channels = int(in_channels)
        self.bottleneck_channels = int(bottleneck_channels)
        self.branch_channels = int(branch_channels)
        self.kernel_sizes = tuple(int(kernel) for kernel in kernel_sizes)
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
        self.convolution_branches = nn.ModuleList(
            nn.Conv1d(bottleneck_channels, branch_channels, kernel,
                      padding=kernel // 2, bias=False)
            for kernel in self.kernel_sizes
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, 1, bias=False),
        )
        self.normalization = nn.BatchNorm1d(4 * branch_channels)
        self.activation = nn.ReLU()

    @property
    def out_channels(self) -> int:
        return 4 * self.branch_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck(inputs)
        branches = [convolution(bottleneck) for convolution in self.convolution_branches]
        branches.append(self.pool_branch(inputs))
        return self.activation(self.normalization(torch.cat(branches, dim=1)))


class ResidualProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.normalization = nn.BatchNorm1d(out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.normalization(self.projection(inputs))


class InceptionTimeClassifier(nn.Module):
    """Six Inception modules with residual connections after modules 3 and 6."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        modules: list[InceptionModule] = []
        channels = in_channels
        for _ in range(MODULE_COUNT):
            module = InceptionModule(channels)
            modules.append(module)
            channels = module.out_channels
        self.modules_inception = nn.ModuleList(modules)
        self.residual_1 = ResidualProjection(in_channels, channels)
        self.residual_2 = ResidualProjection(channels, channels)
        self.residual_activation = nn.ReLU()
        self.classifier = nn.Linear(channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        block_input = inputs
        features = inputs
        for module in self.modules_inception[:3]:
            features = module(features)
        features = self.residual_activation(features + self.residual_1(block_input))
        block_input = features
        for module in self.modules_inception[3:]:
            features = module(features)
        features = self.residual_activation(features + self.residual_2(block_input))
        pooled = features.mean(dim=-1)
        return self.classifier(pooled).squeeze(1)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": "inceptiontime_6module",
            "in_channels": self.in_channels,
            "module_count": MODULE_COUNT,
            "residual_after_modules": [3, 6],
            "bottleneck_channels": BOTTLENECK_CHANNELS,
            "branch_channels": BRANCH_CHANNELS,
            "kernel_sizes": list(KERNEL_SIZES),
            "pool_branch": "MaxPool1d(3, stride=1)+Conv1d(1)",
            "normalization": "BatchNorm1d",
            "activation": "ReLU",
            "pooling": "global_average",
            "output": "one logit; sigmoid applied for probability",
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }


def train_classifier(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
                     run_dir: Path, seed: int, device: torch.device, max_epochs: int, patience: int,
                     pos_weight: float, resume: bool = True) -> tuple[nn.Module, dict[str, Any], np.ndarray]:
    checkpoint = run_dir / "inceptiontime_best.pt"
    log_path = run_dir / "training_log_inceptiontime.csv"
    resume_path = run_dir / "inceptiontime_resume.pt"
    if checkpoint.exists() and log_path.exists():
        model = InceptionTimeClassifier(train_x.shape[2]).to(device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        return model, dict(payload["training"]), exp.predict_classifier(model, val_x, device)
    run_dir.mkdir(parents=True, exist_ok=True)
    exp.seed_everything(seed)
    model = InceptionTimeClassifier(train_x.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    batches = exp.pair_loader(train_x, train_y, 128, True, seed)
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_epoch = 0
    if resume and RESUME_ALLOWED and resume_path.exists():
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        best_state = payload["best_state"]
        best_score = float(payload["best_score"])
        best_epoch = int(payload["best_epoch"])
        bad_epochs = int(payload["bad_epochs"])
        last_epoch = int(payload["last_epoch"])
        history = list(payload["history"])
        if payload.get("loader_generator_state") is not None and batches.generator is not None:
            batches.generator.set_state(payload["loader_generator_state"].cpu())
        print(f"RESUME {run_dir} epoch={last_epoch + 1}/{max_epochs}", flush=True)
    first_epoch = max_epochs + 1 if bad_epochs >= patience else last_epoch + 1
    for epoch in range(first_epoch, max_epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for batch_x, batch_y in batches:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient):
                raise FloatingPointError("non-finite InceptionTime gradient")
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
        probability = exp.predict_classifier(model, val_x, device)
        score = exp.safe_pr_auc(val_y, probability)
        improved = score > best_score + 1e-8
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = exp.base.clone_state(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
        last_epoch = epoch
        history.append({"epoch": epoch, "train_bce": total_loss / count,
                        "validation_pr_auc": score, "improved": improved,
                        "bad_epochs": bad_epochs})
        atomic_torch_save({
            "model_state": exp.base.clone_state(model),
            "optimizer_state": optimizer.state_dict(),
            "best_state": best_state,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "last_epoch": last_epoch,
            "history": history,
            "loader_generator_state": (batches.generator.get_state()
                                       if batches.generator is not None else None),
        }, resume_path)
        if bad_epochs >= patience:
            break
    if best_state is None:
        raise AssertionError("InceptionTime classifier produced no checkpoint")
    training = {
        "seed": seed,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "best_validation_pr_auc": best_score,
        "pos_weight": pos_weight,
        "elapsed_seconds": time.perf_counter() - started,
        "train_windows": len(train_x),
        "validation_windows": len(val_x),
        "architecture": model.architecture_config(),
    }
    torch.save({"model_state": best_state, "training": training}, checkpoint)
    torch.save({"model_state": exp.base.clone_state(model), "training": training},
               run_dir / "inceptiontime_last.pt")
    exp.write_csv(log_path, history)
    model.load_state_dict(best_state)
    resume_path.unlink(missing_ok=True)
    return model, training, exp.predict_classifier(model, val_x, device)


def atomic_torch_save(payload: Any, path: Path) -> None:
    """Keep the previous epoch checkpoint intact if a save is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def configure_base_module(resume: bool = True) -> None:
    """Make the audited base pipeline use InceptionTime without changing its protocol."""
    global RESUME_ALLOWED
    RESUME_ALLOWED = bool(resume)
    exp.EXPERIMENT = EXPERIMENT
    exp.METHOD_NAMES = dict(METHOD_NAMES)
    exp.METHOD_DIRS = dict(METHOD_DIRS)
    exp.train_classifier = train_classifier


def write_protocol(root: Path, source_root: Path, config: Path) -> None:
    protocol = {
        "experiment": EXPERIMENT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config.resolve()),
        "source_representation_root": str(source_root),
        "representation_reused_without_retraining": True,
        "subjects": list(exp.SUBJECTS),
        "outer": "leave-one-complete-valid-record-out",
        "inner": "3-fold record-first purged OOF",
        "nbm_seed_fixed": exp.NBM_SEED,
        "classifier": "InceptionTime-6module",
        "classifier_seeds": list(exp.SEEDS),
        "test_used_for_selection": False,
        "resume_granularity": "classifier epoch and completed method-seed run",
        "invalid_record_exclusion": {"S03_seg003": "valid_fraction=0"},
    }
    exp.write_json(root / "splits" / "frozen_protocol_inceptiontime.json", protocol)


def create_balanced_fold_plan(root: Path, devices: list[str]) -> Path:
    """Greedy LPT assignment using outer-train windows as the fold cost proxy."""
    summary_path = root / "splits" / "outer_folds" / "outer_fold_summary.csv"
    rows = exp.read_csv(summary_path)
    if len(rows) != 30:
        raise ValueError(f"expected 30 outer folds in {summary_path}, found {len(rows)}")
    workers = [{"device": device, "folds": [], "estimated_train_windows": 0}
               for device in devices]
    for row in sorted(rows, key=lambda item: int(item["train_windows"]), reverse=True):
        worker = min(workers, key=lambda item: (item["estimated_train_windows"], len(item["folds"])))
        worker["folds"].append(f"{row['subject_id']}/{row['fold_id']}")
        worker["estimated_train_windows"] += int(row["train_windows"])
    path = root / "splits" / "inceptiontime_parallel_plan.json"
    exp.write_json(path, {"strategy": "greedy_lpt_by_outer_train_windows",
                          "worker_count": len(devices),
                          "workers": {str(index): worker for index, worker in enumerate(workers)}})
    return path


def launch_parallel_workers(args: argparse.Namespace, root: Path, source_root: Path) -> None:
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one CUDA device")
    fold_plan = create_balanced_fold_plan(root, devices)
    log_dir = root / "logs" / "parallel_workers"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = set(range(len(devices)))
    attempts = {index: 0 for index in pending}
    statuses: dict[int, dict[str, Any]] = {}
    try:
        while pending:
            running: dict[int, tuple[subprocess.Popen[Any], Any, Any]] = {}
            for index in sorted(pending):
                attempts[index] += 1
                stdout_path = log_dir / f"worker{index}_attempt{attempts[index]}.out.log"
                stderr_path = log_dir / f"worker{index}_attempt{attempts[index]}.err.log"
                stdout_handle = stdout_path.open("a", encoding="utf-8")
                stderr_handle = stderr_path.open("a", encoding="utf-8")
                command = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--worker", "--source-root", str(source_root),
                    "--output-root", str(root), "--data-dir", str(args.data_dir.resolve()),
                    "--config", str(args.config.resolve()), "--device", devices[index],
                    "--threads", str(args.threads), "--classifier-max-epochs", str(args.classifier_max_epochs),
                    "--classifier-patience", str(args.classifier_patience),
                    "--bootstrap-samples", str(args.bootstrap_samples),
                    "--shard-index", str(index), "--shard-count", str(len(devices)),
                    "--fold-plan", str(fold_plan),
                ]
                if args.no_resume:
                    command.append("--no-resume")
                popen_options: dict[str, Any] = {"cwd": str(ROOT), "stdout": stdout_handle,
                                                 "stderr": stderr_handle}
                if os.name == "nt":
                    popen_options["creationflags"] = subprocess.CREATE_NO_WINDOW
                else:
                    popen_options["start_new_session"] = True
                process = subprocess.Popen(command, **popen_options)
                running[index] = (process, stdout_handle, stderr_handle)
                statuses[index] = {"device": devices[index], "pid": process.pid,
                                   "attempt": attempts[index], "state": "running",
                                   "stdout": str(stdout_path), "stderr": str(stderr_path)}
                print(f"LAUNCH worker={index} device={devices[index]} pid={process.pid}", flush=True)
            exp.write_json(log_dir / "parallel_status.json", statuses)
            failed: set[int] = set()
            for index, (process, stdout_handle, stderr_handle) in running.items():
                return_code = process.wait()
                stdout_handle.close()
                stderr_handle.close()
                statuses[index]["return_code"] = return_code
                statuses[index]["state"] = "complete" if return_code == 0 else "failed"
                if return_code != 0:
                    failed.add(index)
                print(f"EXIT worker={index} device={devices[index]} code={return_code}", flush=True)
            exp.write_json(log_dir / "parallel_status.json", statuses)
            exhausted = [index for index in failed if attempts[index] > args.max_retries]
            if exhausted:
                raise RuntimeError(f"workers exhausted retries: {exhausted}; inspect {log_dir}")
            pending = failed
    except KeyboardInterrupt:
        for process, stdout_handle, stderr_handle in running.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.time() + 15
        for process, stdout_handle, stderr_handle in running.values():
            try:
                process.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                process.kill()
            stdout_handle.close()
            stderr_handle.close()
        exp.write_json(log_dir / "parallel_status.json",
                       {**statuses, "interrupted": True,
                        "resume_command": "rerun the identical --launch-parallel command"})
        print("INTERRUPTED: epoch checkpoints retained; rerun the same command to resume", flush=True)
        raise


def hardlink_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        target = destination / relative
        if source_path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source_path, target)
            except OSError:
                shutil.copy2(source_path, target)


def rewrite_report(root: Path, source_root: Path) -> None:
    generic = root / "reports" / "full_subject_binary_classification_report.md"
    target = root / "reports" / "full_subject_inceptiontime_classification_report.md"
    text = generic.read_text(encoding="utf-8-sig")
    text = text.replace("Daphnet 全被试 NBM 残差二分类实验报告",
                        "Daphnet 全被试 NBM 残差 InceptionTime 二分类实验报告")
    text = text.replace("NBM/TCN训练", "NBM/InceptionTime训练")
    text = text.replace("TCN训练用残差", "InceptionTime训练用残差")
    text = text.replace("改变TCN初始化", "改变InceptionTime初始化")
    provenance = (
        "\n## 表征复用溯源\n\n"
        f"- OOF与外层测试NBM表征复用自 `{source_root}`；仅替换统一分类器。\n"
        "- InceptionTime由6个Inception Module构成，在第3、6模块后使用残差连接。\n"
        "- 三卷积分支核为9/19/39，另含MaxPool+1x1 Conv分支；模块输出为BatchNorm+ReLU。\n"
    )
    target.write_text(text.rstrip() + "\n" + provenance, encoding="utf-8-sig")


def progress_summary(root: Path) -> dict[str, Any]:
    completed_by_method = {
        method: sum(1 for path in (root / METHOD_DIRS[method]).glob("*/*/seed*/run_metrics.json")
                    if (path.parent / "test_predictions.csv").exists())
        for method in exp.METHODS
    }
    resume_files = list(root.glob("*/S*/**/inceptiontime_resume.pt"))
    best_checkpoints = list(root.glob("*/S*/**/inceptiontime_best.pt"))
    complete = sum(completed_by_method.values())
    return {
        "output_root": str(root),
        "completed_runs": complete,
        "total_runs": 360,
        "completion_percent": 100.0 * complete / 360.0,
        "completed_by_method": completed_by_method,
        "incomplete_epoch_checkpoints": len(resume_files),
        "classifier_best_checkpoints": len(best_checkpoints),
        "ready_to_finalize": complete == 360,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed")
    parser.add_argument("--source-root", type=Path,
                        default=ROOT / "outputs" / SOURCE_EXPERIMENT / "full_subject_binary_experiment")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs" / EXPERIMENT / "full_subject_binary_experiment")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs" / "daphnet_full_subject_nbm_residual_inceptiontime.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2,
                        help="CPU intra-op threads per GPU worker; 2 is conservative for seven workers")
    parser.add_argument("--classifier-max-epochs", type=int, default=50)
    parser.add_argument("--classifier-patience", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--only-fold", default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--fold-plan", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6")
    parser.add_argument("--launch-parallel", action="store_true",
                        help="launch one resumable fold shard per device in --devices")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore an incomplete epoch checkpoint (completed runs are still skipped)")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_base_module(resume=not args.no_resume)
    source_root = args.source_root.resolve()
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

    if not args.finalize_only:
        if not args.worker:
            hardlink_tree(source_root / "splits", root / "splits")
            write_protocol(root, source_root, args.config)
        if args.prepare_only:
            print(f"PREPARED {root}", flush=True)
            return
        if args.launch_parallel and not args.worker:
            launch_parallel_workers(args, root, source_root)
            result = exp.aggregate_results(root, args.bootstrap_samples)
            rewrite_report(root, source_root)
            best = max(exp.METHODS,
                       key=lambda method: next(row for row in result["macro_results"]
                                               if row["method"] == method)["macro_pr_auc"])
            print(f"COMPLETE {root} best={best}", flush=True)
            return

        dataset = DaphnetDataset.load(args.data_dir.resolve())
        items = {subject: exp.build_subject_windows(dataset, subject) for subject in exp.SUBJECTS}
        all_folds: list[tuple[str, dict[str, Any]]] = []
        for subject, item in items.items():
            all_folds.extend((subject, fold) for fold in exp.outer_folds(item))
        selected = all_folds
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
            if not 0 <= args.shard_index < args.shard_count:
                raise ValueError("shard-index must be in [0, shard-count)")
            selected = [entry for index, entry in enumerate(selected)
                        if index % args.shard_count == args.shard_index]
        if args.smoke:
            selected = selected[:1]
        for position, (subject, fold) in enumerate(selected, 1):
            print(f"OUTER {position}/{len(selected)} {subject}/{fold['fold_id']} device={device}", flush=True)
            exp.run_outer_fold(
                items[subject], fold, root, device, 1, 1,
                min(args.classifier_max_epochs, 2) if args.smoke else args.classifier_max_epochs,
                min(args.classifier_patience, 1) if args.smoke else args.classifier_patience,
            )
    if args.smoke or args.only_fold or args.shard_count > 1:
        print(f"PARTIAL COMPLETE {root}", flush=True)
        return
    result = exp.aggregate_results(root, args.bootstrap_samples)
    rewrite_report(root, source_root)
    best = max(exp.METHODS,
               key=lambda method: next(row for row in result["macro_results"]
                                       if row["method"] == method)["macro_pr_auc"])
    print(f"COMPLETE {root} best={best}", flush=True)


if __name__ == "__main__":
    main()
