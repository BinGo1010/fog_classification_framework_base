"""Run the pre-registered Daphnet NBM E0--E3 study on processed_A5_50.

The runner preserves the original A5 full-window E0 reproduction and also
uses an E3-history-eligible common support for paired E0--E3 comparisons.
E1 reuses E0 checkpoints; E2 and E3 train new models.  Test FoG never selects
models, residual scores, C1 parameters, thresholds or E3 capacity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for location in (ROOT, SCRIPTS):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from cnbr_fog.nbm_e0_e3 import (  # noqa: E402
    HistoryPredictor,
    TrueBottleneckAE,
    apply_c1,
    benjamini_hochberg,
    build_e3b_input,
    chronological_calibration_split,
    e3b_objective,
    fit_c1_mad,
    fixed_quarter_masks,
    l4_loss,
    paired_subject_statistics,
    profile_model,
    random_block_mask,
    reconstruction_rows,
    score_shift_metrics,
    threshold_metrics,
)
import run_daphnet_nbm_routeA_A1b_generalization_repair as a1b  # noqa: E402
import run_daphnet_nbm_routeA_A5 as a5  # noqa: E402
import run_daphnet_nbm_routeA_A5_manifest as manifest_a5  # noqa: E402
import run_daphnet_nbm_tcdae_three_rounds as base  # noqa: E402
import run_daphnet_nbm_routeA_final_residual_validation as route_a  # noqa: E402


EXPERIMENT = "daphnet_nbm_E0_E3_A5_50_v1"
FORMAL_SUBJECTS = ("S01", "S02", "S05", "S06", "S07", "S08", "S09")
SELECTION_SUBJECTS = ("S01", "S05", "S08", "S09")
DIAGNOSTIC_SUBJECTS = ("S03",)
CLEAN_CONTROLS = ("S04", "S10")
ALL_SUBJECTS = (
    "S01",
    "S02",
    "S03",
    "S04",
    "S05",
    "S06",
    "S07",
    "S08",
    "S09",
    "S10",
)
SEEDS = (20260802, 20260803, 20260804)
ROLES = (
    "nbm_internal_train_nonfog",
    "nbm_internal_earlystop_nonfog",
    "external_validation_nonfog",
    "external_validation_fog",
    "external_test_nonfog",
    "external_test_fog",
)
ARRAY_NAMES = {
    "nbm_internal_train_nonfog": "train_nonfog",
    "nbm_internal_earlystop_nonfog": "earlystop_nonfog",
    "external_validation_nonfog": "validation_nonfog",
    "external_validation_fog": "validation_fog",
    "external_test_nonfog": "test_nonfog",
    "external_test_fog": "test_fog",
}
WINDOW = 128
CHANNELS = 9
FS = 64


def parse_args() -> argparse.Namespace:
    dataset = ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed_A5_50"
    output = ROOT / "outputs" / EXPERIMENT
    reference = (
        ROOT
        / "outputs"
        / "daphnet_nbm_routeA_A5_50_manifest_full_v1"
        / "routeA_A5_50_manifest_full"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=dataset)
    parser.add_argument("--output-root", type=Path, default=output)
    parser.add_argument("--subjects", default=",".join(ALL_SUBJECTS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--stop-after", choices=("E0", "E1", "E2", "E3"), default="E3")
    parser.add_argument("--include-e2-p16", action="store_true")
    parser.add_argument("--include-e3b", action="store_true")
    parser.add_argument("--e3-capacity", choices=("auto", "p24", "m3"), default="auto")
    parser.add_argument("--reference-a5-root", type=Path, default=reference)
    parser.add_argument(
        "--reuse-e0-root",
        type=Path,
        default=reference / "training",
        help="Reuse matching A5_50 E0 checkpoints; unavailable subjects are trained.",
    )
    parser.add_argument("--force-retrain-e0", action="store_true")
    parser.add_argument("--allow-e0-mismatch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-windows-per-role", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-epochs", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    a5.write_json(path, payload)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    a5.write_csv(path, rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_subjects(text: str) -> tuple[str, ...]:
    subjects = tuple(item.strip() for item in text.split(",") if item.strip())
    unknown = sorted(set(subjects) - set(ALL_SUBJECTS))
    if unknown:
        raise ValueError(f"unknown subjects: {unknown}")
    if not subjects:
        raise ValueError("no subjects selected")
    return subjects


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not seeds:
        raise ValueError("no seeds selected")
    return seeds


def evenly_limit(rows: Sequence[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    indices = np.linspace(0, len(rows) - 1, num=limit, dtype=np.int64)
    return [rows[int(index)] for index in np.unique(indices)]


@dataclass
class SubjectBundle:
    subject: str
    scope: str
    role_rows: dict[str, list[dict[str, str]]]
    raw: dict[str, np.ndarray]
    processed: dict[str, np.ndarray]
    scaler: dict[str, Any]
    records: dict[str, Any]
    split_runs: dict[str, list[tuple[str, int, int]]]


@dataclass
class RunOutputs:
    stage: str
    subject: str
    seed: int
    model_name: str
    model_config: dict[str, Any]
    training: dict[str, Any]
    rows: dict[str, list[dict[str, str]]]
    actual: dict[str, np.ndarray]
    predicted: dict[str, np.ndarray]
    residual: dict[str, np.ndarray]
    run_dir: Path


def build_split_runs(rows: Sequence[dict[str, str]]) -> dict[str, list[tuple[str, int, int]]]:
    """Infer contiguous same-split envelopes without crossing manifest split changes."""
    result: dict[str, list[tuple[str, int, int]]] = {}
    by_record: dict[str, dict[tuple[int, int], str]] = {}
    for row in rows:
        by_record.setdefault(row["record_id"], {})[
            (int(row["start_index"]), int(row["end_index_exclusive"]))
        ] = row["a5_split"]
    for record, positions in by_record.items():
        ordered = sorted((start, end, split) for (start, end), split in positions.items())
        runs: list[tuple[str, int, int]] = []
        current_split = ""
        start = end = 0
        for row_start, row_end, split in ordered:
            if split != current_split:
                if current_split:
                    runs.append((current_split, start, end))
                current_split, start, end = split, row_start, row_end
            else:
                end = max(end, row_end)
        if current_split:
            runs.append((current_split, start, end))
        result[record] = runs
    return result


def containing_split_run(item: SubjectBundle, row: dict[str, str]) -> tuple[int, int] | None:
    split = row["a5_split"]
    target_start = int(row["start_index"])
    target_end = int(row["end_index_exclusive"])
    candidates = [
        (start, end)
        for run_split, start, end in item.split_runs.get(row["record_id"], [])
        if run_split == split and target_start >= start and target_end <= end
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda value: value[1] - value[0])


def load_data(
    data_dir: Path,
    subjects: Sequence[str],
    *,
    max_windows_per_role: int,
) -> tuple[dict[str, SubjectBundle], list[dict[str, str]], tuple[str, ...]]:
    quality_path = manifest_a5.resolve_a5_artifact(data_dir, "a5_quality_report.json")
    quality = read_json(quality_path)
    if not quality.get("overall_pass"):
        raise RuntimeError("processed_A5_50 quality gate is not PASS")
    manifest_path = manifest_a5.resolve_a5_artifact(data_dir, "a5_window_manifest.csv")
    manifest_rows = read_csv(manifest_path)
    dataset = route_a.DaphnetDataset.load(data_dir)
    records = {record.record_id: record for record in dataset.records}
    channel_names = tuple(dataset.channel_names)
    bundles: dict[str, SubjectBundle] = {}
    for subject in subjects:
        subject_rows = [row for row in manifest_rows if row["subject_id"] == subject]
        if not subject_rows:
            raise ValueError(f"{subject} is absent from A5_50 manifest")
        role_rows = {
            role: evenly_limit(
                sorted(
                    [row for row in subject_rows if row["a5_role"] == role],
                    key=lambda row: (row["record_id"], int(row["start_index"])),
                ),
                max_windows_per_role,
            )
            for role in ROLES
        }
        required = ROLES[:3] + ("external_test_nonfog",)
        missing = [role for role in required if not role_rows[role]]
        if missing:
            raise ValueError(f"{subject} missing required roles: {missing}")
        raw = {
            ARRAY_NAMES[role]: manifest_a5.stack_windows(rows, records)
            for role, rows in role_rows.items()
        }
        scaler = manifest_a5.fit_training_scaler(raw["train_nonfog"], "full")
        processed = {name: manifest_a5.transform(values, scaler) for name, values in raw.items()}
        if any(not np.isfinite(values).all() for values in processed.values()):
            raise FloatingPointError(f"{subject} preprocessing produced non-finite values")
        bundles[subject] = SubjectBundle(
            subject=subject,
            scope=subject_rows[0]["subject_scope"],
            role_rows=role_rows,
            raw=raw,
            processed=processed,
            scaler=scaler,
            records=records,
            split_runs=build_split_runs(subject_rows),
        )
    return bundles, manifest_rows, channel_names


def context_target_arrays(
    item: SubjectBundle,
    role: str,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]], list[dict[str, Any]]]:
    """Materialize E3-A past-4s or E3-B past-2s context and 2s target."""
    if mode not in ("E3A", "E3B"):
        raise ValueError(mode)
    history_samples = 256 if mode == "E3A" else 128
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    kept_rows: list[dict[str, str]] = []
    audit: list[dict[str, Any]] = []
    median = np.asarray(item.scaler["median"], dtype=np.float32)
    iqr = np.asarray(item.scaler["iqr"], dtype=np.float32)
    epsilon = float(item.scaler["epsilon"])
    for row in item.role_rows[role]:
        record = item.records[row["record_id"]]
        target_start = int(row["start_index"])
        target_end = int(row["end_index_exclusive"])
        context_start = target_start - history_samples
        split_run = containing_split_run(item, row)
        reason = "kept"
        if split_run is None:
            reason = "no_containing_split_run"
        elif context_start < split_run[0]:
            reason = "history_crosses_split_boundary"
        elif context_start < 0:
            reason = "history_before_record"
        elif not record.valid[context_start:target_end].all():
            reason = "invalid_signal_in_support"
        elif role in (
            "nbm_internal_train_nonfog",
            "nbm_internal_earlystop_nonfog",
            "external_validation_nonfog",
            "external_test_nonfog",
        ) and np.any(record.y[context_start:target_end]):
            reason = "nonfog_pair_contains_fog"
        if reason != "kept":
            audit.append(
                {
                    "subject_id": item.subject,
                    "role": role,
                    "mode": mode,
                    "window_id": row["window_id"],
                    "kept": False,
                    "reason": reason,
                }
            )
            continue
        raw_context = record.x[context_start:target_start].astype(np.float32)
        raw_target = record.x[target_start:target_end].astype(np.float32)
        context = (raw_context - median) / (iqr + epsilon)
        target = (raw_target - median) / (iqr + epsilon)
        context -= context.mean(axis=0, keepdims=True)
        target -= target.mean(axis=0, keepdims=True)
        inputs.append(context)
        targets.append(target)
        kept_rows.append(row)
        audit.append(
            {
                "subject_id": item.subject,
                "role": role,
                "mode": mode,
                "window_id": row["window_id"],
                "record_id": row["record_id"],
                "context_start_index": context_start,
                "context_end_index_exclusive": target_start,
                "target_start_index": target_start,
                "target_end_index_exclusive": target_end,
                "history_samples": history_samples,
                "history_target_overlap_samples": 0,
                "same_manifest_split": True,
                "kept": True,
                "reason": "kept",
            }
        )
    input_shape = (0, history_samples, CHANNELS)
    target_shape = (0, WINDOW, CHANNELS)
    return (
        np.ascontiguousarray(np.stack(inputs).astype(np.float32))
        if inputs
        else np.empty(input_shape, dtype=np.float32),
        np.ascontiguousarray(np.stack(targets).astype(np.float32))
        if targets
        else np.empty(target_shape, dtype=np.float32),
        kept_rows,
        audit,
    )


def common_support(
    bundles: dict[str, SubjectBundle]
) -> tuple[dict[tuple[str, str], set[str]], list[dict[str, Any]]]:
    support: dict[tuple[str, str], set[str]] = {}
    audit: list[dict[str, Any]] = []
    for subject, item in bundles.items():
        for role in ROLES:
            _, _, rows, rows_audit = context_target_arrays(item, role, mode="E3A")
            support[(subject, role)] = {row["window_id"] for row in rows}
            audit.extend(rows_audit)
    return support, audit


def pair_loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    x = torch.from_numpy(np.ascontiguousarray(inputs.transpose(0, 2, 1))).float()
    y = torch.from_numpy(np.ascontiguousarray(targets.transpose(0, 2, 1))).float()
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def predict_standard(model: nn.Module, inputs: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    dummy = np.zeros((len(inputs), WINDOW, CHANNELS), dtype=np.float32)
    for batch_x, _ in pair_loader(
        inputs, dummy, batch_size=256, shuffle=False, seed=0, workers=0
    ):
        predicted, _ = model(batch_x.to(device))
        outputs.append(predicted.transpose(1, 2).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs) if outputs else dummy


@torch.no_grad()
def predict_e3b(
    model: nn.Module,
    history: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for batch_h, batch_y in pair_loader(
        history, targets, batch_size=128, shuffle=False, seed=0, workers=0
    ):
        batch_h = batch_h.to(device)
        batch_y = batch_y.to(device)
        stitched = torch.zeros_like(batch_y)
        for mask in fixed_quarter_masks(len(batch_y), device):
            predicted, _ = model(build_e3b_input(batch_h, batch_y, mask))
            stitched = torch.where(mask.expand_as(stitched), predicted, stitched)
        outputs.append(stitched.transpose(1, 2).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs) if outputs else np.empty_like(targets)


def evaluate_epoch(
    model: nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
    e3b: bool,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch_x, batch_y in pair_loader(
            inputs, targets, batch_size=batch_size, shuffle=False, seed=0, workers=0
        ):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if not e3b:
                predicted, _ = model(batch_x)
                loss = l4_loss(predicted, batch_y)
            else:
                losses: list[torch.Tensor] = []
                for mask in fixed_quarter_masks(len(batch_y), device):
                    predicted, _ = model(build_e3b_input(batch_x, batch_y, mask))
                    losses.append(e3b_objective(predicted, batch_y, mask))
                loss = torch.mean(torch.stack(losses))
            total += float(loss) * len(batch_x)
            count += len(batch_x)
    return total / max(count, 1)


def train_pair_model(
    factory: Callable[[], nn.Module],
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    early_inputs: np.ndarray,
    early_targets: np.ndarray,
    run_dir: Path,
    *,
    stage: str,
    subject: str,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    overwrite: bool,
    e3b: bool = False,
) -> tuple[nn.Module, dict[str, Any], list[dict[str, Any]]]:
    required = (run_dir / "best_model.pt", run_dir / "last_model.pt", run_dir / "training_log.csv")
    model = factory().to(device)
    if not overwrite and all(path.exists() for path in required):
        checkpoint = torch.load(required[0], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        return model, dict(checkpoint["training"]), read_csv(required[2])
    if min(len(train_inputs), len(early_inputs)) == 0:
        raise ValueError(f"{stage}/{subject} has empty training or early-stop pairs")
    run_dir.mkdir(parents=True, exist_ok=True)
    base.set_seed(seed)
    model = factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loader = pair_loader(
        train_inputs,
        train_targets,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        workers=workers,
    )
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_train_loss = math.inf
    last_epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        max_gradient = 0.0
        mask_generator = torch.Generator().manual_seed(seed + 1_000_003 * epoch)
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if not e3b:
                predicted, _ = model(batch_x)
                loss = l4_loss(predicted, batch_y)
            else:
                mask = random_block_mask(
                    len(batch_y), generator=mask_generator, device=device
                )
                predicted, _ = model(build_e3b_input(batch_x, batch_y, mask))
                loss = e3b_objective(predicted, batch_y, mask)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient):
                raise FloatingPointError(f"non-finite gradient in {stage}/{subject}/{seed}")
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
            max_gradient = max(max_gradient, float(gradient))
        last_train_loss = total / max(count, 1)
        validation_loss = evaluate_epoch(
            model,
            early_inputs,
            early_targets,
            device,
            batch_size=batch_size,
            e3b=e3b,
        )
        improved = validation_loss < best_loss - 1e-8
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = base.clone_state(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
        last_epoch = epoch
        if epoch == 1 or epoch % 10 == 0 or improved or epoch == max_epochs:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": last_train_loss,
                    "earlystop_loss": validation_loss,
                    "best_earlystop_loss": best_loss,
                    "max_gradient_norm_before_clip": max_gradient,
                    "improved": improved,
                    "bad_epochs": bad_epochs,
                }
            )
        if epoch == 1 or epoch % 100 == 0:
            print(
                f"TRAIN {stage} {subject} seed={seed} epoch={epoch}/{max_epochs} "
                f"train={last_train_loss:.6g} early={validation_loss:.6g} "
                f"best={best_loss:.6g}@{best_epoch}",
                flush=True,
            )
        if bad_epochs >= patience:
            break
    if best_state is None:
        raise AssertionError("training produced no best checkpoint")
    training = {
        "stage": stage,
        "subject_id": subject,
        "seed": seed,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "best_earlystop_loss": best_loss,
        "last_train_loss": last_train_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "train_windows": len(train_inputs),
        "earlystop_windows": len(early_inputs),
    }
    base.torch_save(run_dir / "last_model.pt", {"model_state": base.clone_state(model), "training": training})
    base.torch_save(run_dir / "best_model.pt", {"model_state": best_state, "training": training})
    write_csv(run_dir / "training_log.csv", history)
    model.load_state_dict(best_state)
    return model, training, history


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def e0_model(
    item: SubjectBundle,
    seed: int,
    run_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    local_required = (
        run_dir / "best_model.pt",
        run_dir / "last_model.pt",
        run_dir / "training_log.csv",
    )
    if not args.overwrite and not args.force_retrain_e0 and all(
        path.exists() for path in local_required
    ):
        checkpoint = torch.load(
            run_dir / "best_model.pt", map_location="cpu", weights_only=False
        )
        model = a1b.ContextM3(WINDOW).to(device)
        model.load_state_dict(checkpoint["model_state"])
        training = dict(checkpoint.get("training", {}))
        training["checkpoint_reused"] = True
        training["checkpoint_source"] = str(run_dir.resolve())
        return model, training
    reuse = args.reuse_e0_root.resolve() / item.subject / f"seed{seed}"
    required = (reuse / "best_model.pt", reuse / "last_model.pt", reuse / "training_log.csv")
    if not args.force_retrain_e0 and all(path.exists() for path in required):
        for source in required:
            link_or_copy(source, run_dir / source.name)
        checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
        model = a1b.ContextM3(WINDOW).to(device)
        model.load_state_dict(checkpoint["model_state"])
        training = dict(checkpoint.get("training", {}))
        training["checkpoint_reused"] = True
        training["checkpoint_source"] = str(reuse.resolve())
        return model, training
    model, _, training = a1b.train_repair_model(
        item.processed["train_nonfog"],
        item.processed["train_nonfog"],
        item.processed["earlystop_nonfog"],
        item.processed["earlystop_nonfog"],
        run_dir,
        subject=item.subject,
        seed=seed,
        loss_name="L4",
        context_name="W0",
        max_epochs=args.max_epochs,
        patience=args.patience,
        workers=args.workers,
        device=device,
    )
    training = dict(training)
    training["checkpoint_reused"] = False
    return model, training


def save_run_outputs(run: RunOutputs) -> None:
    run.run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for role in ROLES:
        name = ARRAY_NAMES[role]
        payload[f"{name}_actual"] = run.actual[name]
        payload[f"{name}_predicted"] = run.predicted[name]
        payload[f"{name}_residual"] = run.residual[name]
        payload[f"{name}_window_id"] = np.asarray(
            [row["window_id"] for row in run.rows[role]], dtype="U80"
        )
    np.savez_compressed(run.run_dir / "evaluation_arrays.npz", **payload)
    write_json(
        run.run_dir / "config.json",
        {
            "experiment": EXPERIMENT,
            "stage": run.stage,
            "subject_id": run.subject,
            "seed": run.seed,
            "model": run.model_name,
            "architecture": run.model_config,
            "training": run.training,
            "preprocessing": "train-only median/IQR then per-window per-channel centering",
            "test_fog_used_for_selection": False,
        },
    )


def load_saved_runs(
    stage_root: Path,
    bundles: dict[str, SubjectBundle],
    seeds: Sequence[int],
    *,
    expected_stage: str,
) -> dict[tuple[str, int], RunOutputs]:
    """Rehydrate saved predictions for an authoritative cross-shard evaluation.

    GPU shards deliberately save predictions and window IDs, while score choice,
    gates and test summaries are recomputed only after all subjects are present.
    This keeps multi-GPU execution numerically equivalent to the sequential
    protocol without making model checkpoints portable between processes.
    """
    outputs: dict[tuple[str, int], RunOutputs] = {}
    for subject, item in bundles.items():
        row_lookup = {
            role: {row["window_id"]: row for row in item.role_rows[role]}
            for role in ROLES
        }
        for seed in seeds:
            run_dir = stage_root / "training" / subject / f"seed{seed}"
            arrays_path = run_dir / "evaluation_arrays.npz"
            config_path = run_dir / "config.json"
            if not arrays_path.exists() or not config_path.exists():
                raise FileNotFoundError(
                    f"missing saved {expected_stage} output for {subject}/seed{seed}: {run_dir}"
                )
            config = read_json(config_path)
            saved_stage = str(config.get("stage", ""))
            if saved_stage != expected_stage:
                raise ValueError(
                    f"stage mismatch in {config_path}: expected {expected_stage}, got {saved_stage}"
                )
            actual: dict[str, np.ndarray] = {}
            predicted: dict[str, np.ndarray] = {}
            residual: dict[str, np.ndarray] = {}
            rows: dict[str, list[dict[str, str]]] = {}
            with np.load(arrays_path, allow_pickle=False) as payload:
                for role in ROLES:
                    name = ARRAY_NAMES[role]
                    ids = [str(value) for value in payload[f"{name}_window_id"].tolist()]
                    missing = [window_id for window_id in ids if window_id not in row_lookup[role]]
                    if missing:
                        raise ValueError(
                            f"{arrays_path} contains {len(missing)} unknown {role} window IDs"
                        )
                    rows[role] = [row_lookup[role][window_id] for window_id in ids]
                    actual[name] = np.asarray(payload[f"{name}_actual"], dtype=np.float32)
                    predicted[name] = np.asarray(
                        payload[f"{name}_predicted"], dtype=np.float32
                    )
                    residual[name] = np.asarray(payload[f"{name}_residual"], dtype=np.float32)
                    expected_length = len(ids)
                    lengths = (len(actual[name]), len(predicted[name]), len(residual[name]))
                    if any(length != expected_length for length in lengths):
                        raise ValueError(
                            f"array/window ID length mismatch in {arrays_path} for {name}: "
                            f"ids={expected_length}, arrays={lengths}"
                        )
            key = (subject, int(seed))
            outputs[key] = RunOutputs(
                stage=saved_stage,
                subject=subject,
                seed=int(seed),
                model_name=str(config.get("model", "unknown")),
                model_config=dict(config.get("architecture", {})),
                training=dict(config.get("training", {})),
                rows=rows,
                actual=actual,
                predicted=predicted,
                residual=residual,
                run_dir=run_dir,
            )
    return outputs


def run_e0(
    bundles: dict[str, SubjectBundle],
    seeds: Sequence[int],
    root: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[tuple[str, int], RunOutputs]:
    stage_root = root / "E0"
    outputs: dict[tuple[str, int], RunOutputs] = {}
    training_rows: list[dict[str, Any]] = []
    for subject, item in bundles.items():
        for seed in seeds:
            run_dir = stage_root / "training" / subject / f"seed{seed}"
            model, training = e0_model(item, seed, run_dir, args, device)
            actual = dict(item.processed)
            predicted = {
                name: predict_standard(model, values, device) for name, values in actual.items()
            }
            rows = {role: list(item.role_rows[role]) for role in ROLES}
            run = RunOutputs(
                stage="E0",
                subject=subject,
                seed=seed,
                model_name="M3_tcdae_long",
                model_config=model.architecture_config(),
                training=training,
                rows=rows,
                actual=actual,
                predicted=predicted,
                residual={name: actual[name] - predicted[name] for name in actual},
                run_dir=run_dir,
            )
            save_run_outputs(run)
            outputs[(subject, seed)] = run
            training_rows.append({"subject_id": subject, "seed": seed, **training})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"DONE E0 {subject} seed={seed}", flush=True)
    write_csv(stage_root / "training_summary.csv", training_rows)
    return outputs


def run_e2_variant(
    variant: str,
    bundles: dict[str, SubjectBundle],
    seeds: Sequence[int],
    root: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[tuple[str, int], RunOutputs]:
    stage_name = "E2" if variant == "P24" else "E2_P16"
    stage_root = root / stage_name
    outputs: dict[tuple[str, int], RunOutputs] = {}
    training_rows: list[dict[str, Any]] = []
    for subject, item in bundles.items():
        for seed in seeds:
            run_dir = stage_root / "training" / subject / f"seed{seed}"
            model, training, _ = train_pair_model(
                lambda: TrueBottleneckAE(variant),
                item.processed["train_nonfog"],
                item.processed["train_nonfog"],
                item.processed["earlystop_nonfog"],
                item.processed["earlystop_nonfog"],
                run_dir,
                stage=stage_name,
                subject=subject,
                seed=seed,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                workers=args.workers,
                device=device,
                overwrite=args.overwrite,
            )
            actual = dict(item.processed)
            predicted = {
                name: predict_standard(model, values, device) for name, values in actual.items()
            }
            run = RunOutputs(
                stage=stage_name,
                subject=subject,
                seed=seed,
                model_name=f"E2_{variant}",
                model_config=model.architecture_config(),
                training=training,
                rows={role: list(item.role_rows[role]) for role in ROLES},
                actual=actual,
                predicted=predicted,
                residual={name: actual[name] - predicted[name] for name in actual},
                run_dir=run_dir,
            )
            save_run_outputs(run)
            outputs[(subject, seed)] = run
            training_rows.append(training)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"DONE {stage_name} {subject} seed={seed}", flush=True)
    write_csv(stage_root / "training_summary.csv", training_rows)
    model = TrueBottleneckAE(variant)
    write_json(stage_root / "architecture_summary.json", model.architecture_config())
    return outputs


def run_e3(
    mode: str,
    latent_channels: int,
    bundles: dict[str, SubjectBundle],
    seeds: Sequence[int],
    root: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[tuple[str, int], RunOutputs]:
    stage_name = mode
    stage_root = root / ("E3" if mode == "E3A" else "E3B")
    outputs: dict[tuple[str, int], RunOutputs] = {}
    training_rows: list[dict[str, Any]] = []
    all_context_audit: list[dict[str, Any]] = []
    for subject, item in bundles.items():
        pair_data: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, str]]]] = {}
        for role in ROLES:
            inputs, targets, rows, audit = context_target_arrays(item, role, mode=mode)
            pair_data[role] = (inputs, targets, rows)
            all_context_audit.extend(audit)
        train_x, train_y, _ = pair_data["nbm_internal_train_nonfog"]
        early_x, early_y, _ = pair_data["nbm_internal_earlystop_nonfog"]
        for seed in seeds:
            run_dir = stage_root / "training" / subject / f"seed{seed}"
            input_channels = CHANNELS if mode == "E3A" else CHANNELS + 1
            model, training, _ = train_pair_model(
                lambda: HistoryPredictor(latent_channels, input_channels),
                train_x,
                train_y,
                early_x,
                early_y,
                run_dir,
                stage=stage_name,
                subject=subject,
                seed=seed,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                workers=args.workers,
                device=device,
                overwrite=args.overwrite,
                e3b=mode == "E3B",
            )
            actual: dict[str, np.ndarray] = {}
            predicted: dict[str, np.ndarray] = {}
            rows_by_role: dict[str, list[dict[str, str]]] = {}
            for role in ROLES:
                inputs, targets, kept_rows = pair_data[role]
                name = ARRAY_NAMES[role]
                actual[name] = targets
                predicted[name] = (
                    predict_standard(model, inputs, device)
                    if mode == "E3A"
                    else predict_e3b(model, inputs, targets, device)
                )
                rows_by_role[role] = kept_rows
            run = RunOutputs(
                stage=stage_name,
                subject=subject,
                seed=seed,
                model_name=f"{mode}_history_predictor_C{latent_channels}",
                model_config=model.architecture_config(),
                training=training,
                rows=rows_by_role,
                actual=actual,
                predicted=predicted,
                residual={name: actual[name] - predicted[name] for name in actual},
                run_dir=run_dir,
            )
            save_run_outputs(run)
            outputs[(subject, seed)] = run
            training_rows.append(training)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"DONE {stage_name} {subject} seed={seed}", flush=True)
    write_csv(stage_root / "training_summary.csv", training_rows)
    write_csv(stage_root / "context_target_manifest.csv", all_context_audit)
    model = HistoryPredictor(latent_channels, CHANNELS if mode == "E3A" else CHANNELS + 1)
    write_json(stage_root / "architecture_summary.json", model.architecture_config())
    if mode == "E3B":
        write_json(
            stage_root / "mask_protocol.json",
            {
                "training_blocks": "1-2 random contiguous blocks",
                "training_block_samples": [16, 32],
                "mask_value": 0,
                "mask_channel": True,
                "loss": "0.7 masked L4 + 0.3 full-target L4",
                "test_masks": ["0-32", "32-64", "64-96", "96-128"],
                "stitch_policy": "each sample uses prediction made while that quarter was masked",
            },
        )
    return outputs


def restrict_run(
    run: RunOutputs,
    support: dict[tuple[str, str], set[str]] | None,
) -> RunOutputs:
    if support is None:
        return run
    rows: dict[str, list[dict[str, str]]] = {}
    actual: dict[str, np.ndarray] = {}
    predicted: dict[str, np.ndarray] = {}
    residual: dict[str, np.ndarray] = {}
    for role in ROLES:
        name = ARRAY_NAMES[role]
        allowed = support[(run.subject, role)]
        indices = np.asarray(
            [index for index, row in enumerate(run.rows[role]) if row["window_id"] in allowed],
            dtype=np.int64,
        )
        rows[role] = [run.rows[role][int(index)] for index in indices]
        actual[name] = run.actual[name][indices]
        predicted[name] = run.predicted[name][indices]
        residual[name] = run.residual[name][indices]
    return RunOutputs(
        stage=run.stage,
        subject=run.subject,
        seed=run.seed,
        model_name=run.model_name,
        model_config=run.model_config,
        training=run.training,
        rows=rows,
        actual=actual,
        predicted=predicted,
        residual=residual,
        run_dir=run.run_dir,
    )


def has_fog(run: RunOutputs, split: str) -> bool:
    return len(run.residual[f"{split}_fog"]) > 0


def safe_separation(normal: np.ndarray, fog: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if min(len(normal), len(fog), len(reference)) == 0:
        return {
            key: math.nan
            for key in (
                "nonfog_p50",
                "nonfog_p90",
                "nonfog_p95",
                "fog_p50",
                "fog_p90",
                "fog_to_nonfog_median_ratio",
                "auroc",
                "average_precision",
                "random_pr_baseline",
                "pr_margin_over_random",
                "recall_at_train_nonfog_p95",
                "nonfog_false_alarm_fraction",
                "false_alarm_windows_per_minute",
                "cliffs_delta",
                "hedges_g",
                "train_nonfog_p95_threshold",
            )
        }
    return a5.separation_metrics(normal, fog, reference)


def score_from_components(
    components: np.ndarray,
    score: str,
    scale: np.ndarray,
    weights: Sequence[float],
) -> np.ndarray:
    if score in ("S0", "S1", "S2"):
        return components[:, a5.SCORES.index(score)]
    return a5.combine_components(components, scale, weights)


@dataclass
class ScoredRun:
    run: RunOutputs
    selection_components: dict[str, np.ndarray]
    deployment_components: dict[str, np.ndarray]
    selection_scale: np.ndarray
    deployment_scale: np.ndarray
    validation_score_indices: np.ndarray
    c1_rows: list[dict[str, Any]]
    calibration_audit: list[dict[str, Any]]


def prepare_scored_run(run: RunOutputs, calibration: str) -> ScoredRun:
    if calibration not in ("C0", "C1"):
        raise ValueError(calibration)
    residual = run.residual
    val_rows = run.rows["external_validation_nonfog"]
    c1_rows: list[dict[str, Any]] = []
    calibration_audit: list[dict[str, Any]] = []
    if calibration == "C0":
        score_indices = np.arange(len(residual["validation_nonfog"]), dtype=np.int64)
        selection_residual = dict(residual)
        deployment_residual = dict(residual)
        selection_scale_source = a5.component_scores(residual["train_nonfog"])
        deployment_scale_source = selection_scale_source
    else:
        calibration_indices, score_indices, audit = chronological_calibration_split(val_rows)
        calibration_audit = [
            {"subject_id": run.subject, "seed": run.seed, **row} for row in audit
        ]
        selection_parameters = fit_c1_mad(residual["validation_nonfog"][calibration_indices])
        deployment_parameters = fit_c1_mad(residual["validation_nonfog"])
        selection_residual = {
            name: apply_c1(values, selection_parameters) for name, values in residual.items()
        }
        deployment_residual = {
            name: apply_c1(values, deployment_parameters) for name, values in residual.items()
        }
        selection_scale_source = a5.component_scores(
            selection_residual["validation_nonfog"][calibration_indices]
        )
        deployment_scale_source = a5.component_scores(
            deployment_residual["validation_nonfog"]
        )
        for fit_scope, parameters in (
            ("selection_calibration_half", selection_parameters),
            ("deployment_full_validation_nonfog", deployment_parameters),
        ):
            for channel in range(CHANNELS):
                c1_rows.append(
                    {
                        "subject_id": run.subject,
                        "seed": run.seed,
                        "fit_scope": fit_scope,
                        "channel_index": channel,
                        "residual_center": float(parameters.center[channel]),
                        "residual_scale_1p4826mad": float(parameters.scale[channel]),
                        "clip": "none",
                    }
                )
    selection_components = {
        name: a5.component_scores(values) for name, values in selection_residual.items()
    }
    deployment_components = {
        name: a5.component_scores(values) for name, values in deployment_residual.items()
    }
    return ScoredRun(
        run=run,
        selection_components=selection_components,
        deployment_components=deployment_components,
        selection_scale=a5.fit_component_scale(selection_scale_source),
        deployment_scale=a5.fit_component_scale(deployment_scale_source),
        validation_score_indices=score_indices,
        c1_rows=c1_rows,
        calibration_audit=calibration_audit,
    )


def candidate_rank(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["median_validation_auroc"]),
        float(row["median_validation_cliffs_delta"]),
        float(row["median_validation_average_precision"]),
        -float(row["median_validation_false_alarm_per_minute"]),
    )


def evaluate_stage(
    stage: str,
    runs: dict[tuple[str, int], RunOutputs],
    stage_root: Path,
    *,
    calibration: str,
    support: dict[tuple[str, str], set[str]] | None,
    weight_step: float,
    channel_names: Sequence[str],
    suffix: str = "",
) -> dict[str, Any]:
    restricted = {key: restrict_run(run, support) for key, run in runs.items()}
    scored = {key: prepare_scored_run(run, calibration) for key, run in restricted.items()}
    selected_subjects = [subject for subject in SELECTION_SUBJECTS if any(key[0] == subject for key in scored)]
    if not selected_subjects:
        selected_subjects = [subject for subject in FORMAL_SUBJECTS if any(key[0] == subject for key in scored)]

    weight_rows: list[dict[str, Any]] = []
    for weights in a5.simplex_weights(weight_step):
        rows: list[dict[str, Any]] = []
        for (subject, _), item in scored.items():
            if subject not in selected_subjects or not has_fog(item.run, "validation"):
                continue
            indices = item.validation_score_indices
            normal = score_from_components(
                item.selection_components["validation_nonfog"][indices],
                "S3",
                item.selection_scale,
                weights,
            )
            fog = score_from_components(
                item.selection_components["validation_fog"],
                "S3",
                item.selection_scale,
                weights,
            )
            rows.append(safe_separation(normal, fog, normal))
        weight_rows.append(
            {
                "weight_s0": weights[0],
                "weight_s1": weights[1],
                "weight_s2": weights[2],
                "selection_runs": len(rows),
                "median_validation_auroc": a5.finite_median(rows, "auroc"),
                "median_validation_average_precision": a5.finite_median(rows, "average_precision"),
                "median_validation_cliffs_delta": a5.finite_median(rows, "cliffs_delta"),
                "median_validation_false_alarm_per_minute": a5.finite_median(
                    rows, "false_alarm_windows_per_minute"
                ),
            }
        )
    best_weight = max(weight_rows, key=candidate_rank)
    weights = (
        float(best_weight["weight_s0"]),
        float(best_weight["weight_s1"]),
        float(best_weight["weight_s2"]),
    )
    for row in weight_rows:
        row["selected"] = bool(
            np.allclose([row["weight_s0"], row["weight_s1"], row["weight_s2"]], weights)
        )
    write_csv(stage_root / f"S3_validation_weight_search{suffix}.csv", weight_rows)

    validation_rows: list[dict[str, Any]] = []
    for (subject, seed), item in scored.items():
        indices = item.validation_score_indices
        for score in a5.SCORES:
            normal = score_from_components(
                item.selection_components["validation_nonfog"][indices],
                score,
                item.selection_scale,
                weights,
            )
            fog = score_from_components(
                item.selection_components["validation_fog"], score, item.selection_scale, weights
            )
            validation_rows.append(
                {
                    "stage": stage,
                    "support_scope": "full_A5" if support is None else "E3A_common_support",
                    "report_split": "external_validation_score_half" if calibration == "C1" else "external_validation",
                    "calibration": calibration,
                    "score": score,
                    "subject_id": subject,
                    "seed": seed,
                    "validation_nonfog_windows": len(normal),
                    "validation_fog_windows": len(fog),
                    **safe_separation(normal, fog, normal),
                }
            )
    candidate_rows: list[dict[str, Any]] = []
    for score in a5.SCORES:
        selected = [
            row
            for row in validation_rows
            if row["score"] == score and row["subject_id"] in selected_subjects
        ]
        candidate_rows.append(
            {
                "score": score,
                "selection_runs": len(selected),
                "median_validation_auroc": a5.finite_median(selected, "auroc"),
                "median_validation_average_precision": a5.finite_median(
                    selected, "average_precision"
                ),
                "median_validation_cliffs_delta": a5.finite_median(selected, "cliffs_delta"),
                "median_validation_false_alarm_per_minute": a5.finite_median(
                    selected, "false_alarm_windows_per_minute"
                ),
            }
        )
    selected_candidate = max(
        candidate_rows,
        key=lambda row: candidate_rank(row) + (-a5.SCORES.index(row["score"]),),
    )
    selected_score = str(selected_candidate["score"])
    write_csv(stage_root / f"residual_score_metrics_validation_all{suffix}.csv", validation_rows)

    test_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    reconstruction_summary: list[dict[str, Any]] = []
    channel_rows_all: list[dict[str, Any]] = []
    window_rows_all: list[dict[str, Any]] = []
    localization_rows: list[dict[str, Any]] = []
    c1_rows: list[dict[str, Any]] = []
    calibration_audit: list[dict[str, Any]] = []
    score_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for (subject, seed), item in scored.items():
        c1_rows.extend(item.c1_rows)
        calibration_audit.extend(item.calibration_audit)
        validation_normal = score_from_components(
            item.deployment_components["validation_nonfog"],
            selected_score,
            item.deployment_scale,
            weights,
        )
        test_normal = score_from_components(
            item.deployment_components["test_nonfog"],
            selected_score,
            item.deployment_scale,
            weights,
        )
        test_fog = score_from_components(
            item.deployment_components["test_fog"],
            selected_score,
            item.deployment_scale,
            weights,
        )
        score_cache[(subject, seed)] = {
            "validation_nonfog": validation_normal,
            "test_nonfog": test_normal,
            "test_fog": test_fog,
        }
        result = safe_separation(test_normal, test_fog, validation_normal)
        validation_result = next(
            row
            for row in validation_rows
            if row["subject_id"] == subject
            and int(row["seed"]) == seed
            and row["score"] == selected_score
        )
        test_rows.append(
            {
                "stage": stage,
                "support_scope": "full_A5" if support is None else "E3A_common_support",
                "report_split": "external_test_after_freeze",
                "calibration": calibration,
                "score": selected_score,
                "s3_weights": str(weights),
                "subject_id": subject,
                "seed": seed,
                "test_nonfog_windows": len(test_normal),
                "test_fog_windows": len(test_fog),
                "validation_to_test_auroc_drop": float(validation_result["auroc"]) - float(result["auroc"]),
                "validation_to_test_pr_drop": float(validation_result["average_precision"])
                - float(result["average_precision"]),
                **result,
            }
        )
        shift_rows.append(
            {
                "stage": stage,
                "subject_id": subject,
                "seed": seed,
                "score": selected_score,
                **score_shift_metrics(validation_normal, test_normal),
            }
        )
        combined_rows = list(item.run.rows["external_test_nonfog"]) + list(
            item.run.rows["external_test_fog"]
        )
        combined_scores = np.concatenate((test_normal, test_fog))
        for quantile in (95.0, 99.2):
            threshold_rows.append(
                {
                    "stage": stage,
                    "subject_id": subject,
                    "seed": seed,
                    "score": selected_score,
                    **threshold_metrics(
                        combined_rows, combined_scores, validation_normal, quantile=quantile
                    ),
                }
            )
        summary, channel_rows, window_rows = reconstruction_rows(
            item.run.actual["test_nonfog"],
            item.run.predicted["test_nonfog"],
            channel_names=channel_names,
        )
        reconstruction_summary.append(
            {"stage": stage, "subject_id": subject, "seed": seed, **summary}
        )
        channel_rows_all.extend(
            {"stage": stage, "subject_id": subject, "seed": seed, **row}
            for row in channel_rows
        )
        window_rows_all.extend(
            {
                "stage": stage,
                "subject_id": subject,
                "seed": seed,
                "window_id": item.run.rows["external_test_nonfog"][int(row["window_local_index"])]["window_id"],
                **row,
            }
            for row in window_rows
        )
        for split, label_name, residual_values in (
            ("external_test", "Non-FoG", item.run.residual["test_nonfog"]),
            ("external_test", "FoG", item.run.residual["test_fog"]),
        ):
            for channel in range(CHANNELS):
                for quarter in range(4):
                    values = residual_values[:, quarter * 32 : (quarter + 1) * 32, channel]
                    if values.size:
                        localization_rows.append(
                            {
                                "stage": stage,
                                "subject_id": subject,
                                "seed": seed,
                                "split": split,
                                "label": label_name,
                                "channel_index": channel,
                                "quarter": quarter + 1,
                                "mean_absolute_residual": float(np.mean(np.abs(values))),
                            }
                        )
    write_csv(stage_root / f"residual_score_metrics{suffix}.csv", test_rows)
    write_csv(stage_root / f"nonfog_shift_metrics{suffix}.csv", shift_rows)
    write_csv(stage_root / f"threshold_metrics{suffix}.csv", threshold_rows)
    write_csv(stage_root / f"nonfog_reconstruction_metrics{suffix}.csv", reconstruction_summary)
    write_csv(stage_root / f"nonfog_reconstruction_channel_metrics{suffix}.csv", channel_rows_all)
    write_csv(stage_root / f"nonfog_reconstruction_window_metrics{suffix}.csv", window_rows_all)
    write_csv(stage_root / f"residual_localization{suffix}.csv", localization_rows)
    if c1_rows:
        write_csv(stage_root / f"c1_parameters{suffix}.csv", c1_rows)
        write_csv(stage_root / f"calibration_split_audit{suffix}.csv", calibration_audit)

    formal_test = [row for row in test_rows if row["subject_id"] in FORMAL_SUBJECTS]
    subject_summaries: list[dict[str, Any]] = []
    for subject in [item for item in FORMAL_SUBJECTS if any(row["subject_id"] == item for row in formal_test)]:
        subject_rows = [row for row in formal_test if row["subject_id"] == subject]
        subject_summaries.append(
            {
                "subject_id": subject,
                "runs": len(subject_rows),
                "median_auroc": a5.finite_median(subject_rows, "auroc"),
                "median_average_precision": a5.finite_median(subject_rows, "average_precision"),
                "median_cliffs_delta": a5.finite_median(subject_rows, "cliffs_delta"),
                "median_recall_q95": a5.finite_median(
                    [
                        row
                        for row in threshold_rows
                        if row["subject_id"] == subject and float(row["quantile"]) == 95.0
                    ],
                    "window_recall",
                ),
                "median_event_fa_per_min_q99p2": a5.finite_median(
                    [
                        row
                        for row in threshold_rows
                        if row["subject_id"] == subject and float(row["quantile"]) == 99.2
                    ],
                    "event_false_alarm_per_minute",
                ),
            }
        )
    gate = {
        "stage": stage,
        "support_scope": "full_A5" if support is None else "E3A_common_support",
        "calibration": calibration,
        "selected_score": selected_score,
        "s3_weights": weights,
        "selection_subjects": selected_subjects,
        "test_fog_used_for_selection": False,
        "candidate_summaries": candidate_rows,
        "subject_summaries": subject_summaries,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(stage_root / f"gate{suffix}.json", gate)
    make_stage_figures(stage_root, reconstruction_summary, channel_rows_all, test_rows, shift_rows, suffix)
    render_stage_report(stage_root, gate, reconstruction_summary, test_rows, threshold_rows, suffix)
    return {
        "gate": gate,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "shift_rows": shift_rows,
        "threshold_rows": threshold_rows,
        "reconstruction_rows": reconstruction_summary,
        "score_cache": score_cache,
    }


def make_stage_figures(
    root: Path,
    reconstruction: Sequence[dict[str, Any]],
    channels: Sequence[dict[str, Any]],
    test: Sequence[dict[str, Any]],
    shift: Sequence[dict[str, Any]],
    suffix: str,
) -> None:
    figure_root = root / "figures"
    for directory in (
        "reconstruction_examples",
        "subject_channel_heatmaps",
        "residual_distributions",
        "validation_test_shift",
    ):
        (figure_root / directory).mkdir(parents=True, exist_ok=True)
    subjects = sorted({row["subject_id"] for row in reconstruction})
    if not subjects:
        return
    nrmse = [a5.finite_median([row for row in reconstruction if row["subject_id"] == subject], "nrmse_median") for subject in subjects]
    corr = [a5.finite_median([row for row in reconstruction if row["subject_id"] == subject], "pearson_median") for subject in subjects]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(subjects, nrmse)
    axes[0].set_ylabel("Test Non-FoG NRMSE")
    axes[1].bar(subjects, corr)
    axes[1].set_ylabel("Test Non-FoG Pearson")
    fig.tight_layout()
    fig.savefig(figure_root / "reconstruction_examples" / f"subject_reconstruction{suffix}.png", dpi=150)
    plt.close(fig)

    matrix = np.full((len(subjects), CHANNELS), np.nan)
    for row_index, subject in enumerate(subjects):
        for channel in range(CHANNELS):
            matrix[row_index, channel] = a5.finite_median(
                [row for row in channels if row["subject_id"] == subject and int(row["channel_index"]) == channel],
                "nrmse_median",
            )
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(subjects))))
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(subjects)), subjects)
    ax.set_xticks(range(CHANNELS), [str(index) for index in range(CHANNELS)])
    ax.set_xlabel("channel")
    fig.colorbar(image, ax=ax, label="NRMSE")
    fig.tight_layout()
    fig.savefig(figure_root / "subject_channel_heatmaps" / f"nrmse_heatmap{suffix}.png", dpi=150)
    plt.close(fig)

    formal = [subject for subject in FORMAL_SUBJECTS if any(row["subject_id"] == subject for row in test)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        formal,
        [a5.finite_median([row for row in test if row["subject_id"] == subject], "auroc") for subject in formal],
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Test AUROC")
    fig.tight_layout()
    fig.savefig(figure_root / "residual_distributions" / f"subject_auroc{suffix}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        subjects,
        [a5.finite_median([row for row in shift if row["subject_id"] == subject], "shift_robust") for subject in subjects],
    )
    ax.set_ylabel("Robust validation-test shift")
    fig.tight_layout()
    fig.savefig(figure_root / "validation_test_shift" / f"robust_shift{suffix}.png", dpi=150)
    plt.close(fig)


def render_stage_report(
    root: Path,
    gate: dict[str, Any],
    reconstruction: Sequence[dict[str, Any]],
    test: Sequence[dict[str, Any]],
    threshold: Sequence[dict[str, Any]],
    suffix: str,
) -> None:
    subjects = gate["subject_summaries"]
    lines = [
        f"# {gate['stage']} report",
        "",
        f"- Support: `{gate['support_scope']}`",
        f"- Calibration: `{gate['calibration']}`",
        f"- Selected residual score: `{gate['selected_score']}`; S3 weights={gate['s3_weights']}",
        "- Test FoG used for selection: no",
        "",
        "| Subject | AUROC | PR-AUC | Cliff delta | Recall Q95 | Event FA/min Q99.2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in subjects:
        lines.append(
            f"| {row['subject_id']} | {row['median_auroc']:.3f} | "
            f"{row['median_average_precision']:.3f} | {row['median_cliffs_delta']:.3f} | "
            f"{row['median_recall_q95']:.1%} | {row['median_event_fa_per_min_q99p2']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The full CSV outputs retain every subject, seed, channel, threshold and shift metric.",
            "Overlapping windows are not treated as independent statistical units.",
        ]
    )
    (root / f"{gate['stage']}_report{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def paired_metric_table(
    left: dict[str, Any], right: dict[str, Any], left_name: str, right_name: str
) -> list[dict[str, Any]]:
    left_rows = {
        (str(row["subject_id"]), int(row["seed"])): row for row in left["test_rows"]
    }
    right_rows = {
        (str(row["subject_id"]), int(row["seed"])): row for row in right["test_rows"]
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(set(left_rows) & set(right_rows)):
        left_row, right_row = left_rows[key], right_rows[key]
        row: dict[str, Any] = {
            "subject_id": key[0],
            "seed": key[1],
            "left_experiment": left_name,
            "right_experiment": right_name,
        }
        for metric in (
            "auroc",
            "average_precision",
            "cliffs_delta",
            "fog_to_nonfog_median_ratio",
            "recall_at_train_nonfog_p95",
            "false_alarm_windows_per_minute",
        ):
            row[f"{left_name}_{metric}"] = left_row[metric]
            row[f"{right_name}_{metric}"] = right_row[metric]
            row[f"delta_{metric}"] = float(right_row[metric]) - float(left_row[metric])
        rows.append(row)
    return rows


def architecture_markdown(name: str, config: dict[str, Any], profile: dict[str, Any]) -> str:
    return (
        f"# {name} architecture\n\n"
        f"- Input: `{config.get('input_shape')}`\n"
        f"- Target/output: `{config.get('target_shape', config.get('input_shape'))}`\n"
        f"- Latent: `{config.get('latent_shape')}` ({config.get('latent_elements')} scalars)\n"
        f"- Parameters: {profile['total_parameters']} "
        f"(encoder {profile['encoder_parameters']}, decoder {profile['decoder_parameters']})\n"
        f"- Approximate Conv1d MACs/window: {profile['approximate_conv1d_macs_per_window']}\n"
        f"- Measured batch-one inference: {profile['inference_ms_per_window_batch1']:.4f} ms\n"
        f"- Long skip: {config.get('long_skip', False)}\n"
    )


def write_stage_contract_files(
    root: Path,
    stage: str,
    result: dict[str, Any],
    *,
    protocol: dict[str, Any],
    split_audit: Sequence[dict[str, Any]],
) -> None:
    """Write the exact filenames requested by the E0--E3 outline."""
    write_json(
        root / "config.json",
        {
            "experiment": EXPERIMENT,
            "stage": stage,
            "support_scope": result["gate"]["support_scope"],
            "calibration": result["gate"]["calibration"],
            "selected_score": result["gate"]["selected_score"],
            "s3_weights": result["gate"]["s3_weights"],
            "training": protocol["training"],
            "test_fog_used_for_selection": False,
        },
    )
    write_csv(root / "split_audit.csv", split_audit)
    write_csv(root / "per_subject_seed_metrics.csv", result["test_rows"])
    if stage in ("E2", "E3A", "E3B"):
        write_csv(root / "v1_nonfog_preservation.csv", result["reconstruction_rows"])
    if stage in ("E3A", "E3B"):
        write_csv(root / "context_prediction_metrics.csv", result["reconstruction_rows"])


def compare_e0_reference(
    root: Path,
    result: dict[str, Any],
    reference_root: Path,
    *,
    subjects: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, Any]:
    reference_gate_path = reference_root / "A5_manifest_gate.json"
    reference_table_path = reference_root / "tables" / "selected_score_test_metrics.csv"
    if not reference_gate_path.exists() or not reference_table_path.exists():
        gate = {"status": "NE", "reason": "reference A5 outputs not found"}
        write_json(root / "E0" / "E0_reproduction_gate.json", gate)
        return gate
    reference_gate = read_json(reference_gate_path)
    reference_rows = read_csv(reference_table_path)
    current = {
        (str(row["subject_id"]), int(row["seed"])): row for row in result["test_rows"]
    }
    comparisons: list[dict[str, Any]] = []
    for row in reference_rows:
        key = (row["subject_id"], int(row["seed"]))
        if key not in current or key[0] not in subjects or key[1] not in seeds:
            continue
        comparisons.append(
            {
                "subject_id": key[0],
                "seed": key[1],
                "reference_auroc": float(row["auroc"]),
                "current_auroc": float(current[key]["auroc"]),
                "absolute_auroc_difference": abs(float(row["auroc"]) - float(current[key]["auroc"])),
                "reference_pr_auc": float(row["average_precision"]),
                "current_pr_auc": float(current[key]["average_precision"]),
                "absolute_pr_auc_difference": abs(
                    float(row["average_precision"]) - float(current[key]["average_precision"])
                ),
            }
        )
    full_protocol = set(FORMAL_SUBJECTS).issubset(subjects) and set(SEEDS).issubset(seeds)
    pass_metrics = bool(
        comparisons
        and max(row["absolute_auroc_difference"] for row in comparisons) <= 0.01
        and max(row["absolute_pr_auc_difference"] for row in comparisons) <= 0.01
    )
    score_match = result["gate"]["selected_score"] == reference_gate["selected_score"]
    gate = {
        "status": "PASS" if pass_metrics and score_match else "FAIL",
        "applicable_to_full_protocol": full_protocol,
        "compared_runs": len(comparisons),
        "metric_tolerance": 0.01,
        "max_absolute_auroc_difference": max(
            (row["absolute_auroc_difference"] for row in comparisons), default=math.nan
        ),
        "max_absolute_pr_auc_difference": max(
            (row["absolute_pr_auc_difference"] for row in comparisons), default=math.nan
        ),
        "selected_score_current": result["gate"]["selected_score"],
        "selected_score_reference": reference_gate["selected_score"],
        "selected_score_match": score_match,
        "best_epoch_direction": "identical checkpoints when reused; otherwise inspect training_summary.csv",
    }
    write_csv(root / "E0" / "e0_a5_reproduction_comparison.csv", comparisons)
    write_json(root / "E0" / "E0_reproduction_gate.json", gate)
    return gate


def subject_medians(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    return {
        subject: a5.finite_median([row for row in rows if row["subject_id"] == subject], key)
        for subject in FORMAL_SUBJECTS
        if any(row["subject_id"] == subject for row in rows)
    }


def v1_subject_passes(reconstruction: Sequence[dict[str, Any]]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for subject in FORMAL_SUBJECTS:
        rows = [row for row in reconstruction if row["subject_id"] == subject]
        if not rows:
            continue
        result[subject] = bool(
            a5.finite_median(rows, "pearson_median") >= 0.50
            and a5.finite_median(rows, "nrmse_median") <= 0.85
            and a5.finite_median(rows, "nrmse_p90") <= 1.30
        )
    return result


def e1_gate(e0: dict[str, Any], e1: dict[str, Any]) -> dict[str, Any]:
    e0_shift = subject_medians(e0["shift_rows"], "shift_robust")
    e1_shift = subject_medians(e1["shift_rows"], "shift_robust")
    subjects = sorted(set(e0_shift) & set(e1_shift))
    improved_shift = sum(e1_shift[s] < e0_shift[s] for s in subjects)
    baseline_median = float(np.median([e0_shift[s] for s in subjects]))
    current_median = float(np.median([e1_shift[s] for s in subjects]))
    reduction = (baseline_median - current_median) / max(baseline_median, 1e-12)
    e0_auroc = subject_medians(e0["test_rows"], "auroc")
    e1_auroc = subject_medians(e1["test_rows"], "auroc")
    e0_pr = subject_medians(e0["test_rows"], "average_precision")
    e1_pr = subject_medians(e1["test_rows"], "average_precision")
    auroc_drop = float(np.median(list(e0_auroc.values())) - np.median(list(e1_auroc.values())))
    pr_drop = float(np.median(list(e0_pr.values())) - np.median(list(e1_pr.values())))
    ranking_maintained = sum(
        e1_auroc[s] >= e0_auroc[s] or e1_pr[s] >= e0_pr[s]
        for s in sorted(set(e0_auroc) & set(e1_auroc))
    )
    passed = bool(
        improved_shift >= 5
        and reduction >= 0.30
        and auroc_drop <= 0.02
        and pr_drop <= 0.02
        and ranking_maintained >= 4
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "shift_improved_subjects": improved_shift,
        "required_shift_improved_subjects": 5,
        "median_shift_reduction_fraction": reduction,
        "required_median_shift_reduction_fraction": 0.30,
        "median_auroc_drop": auroc_drop,
        "median_pr_auc_drop": pr_drop,
        "ranking_maintained_or_improved_subjects": ranking_maintained,
        "required_ranking_subjects": 4,
    }


def e2_gate(e1: dict[str, Any], e2: dict[str, Any]) -> dict[str, Any]:
    v1 = v1_subject_passes(e2["reconstruction_rows"])
    e1_auroc = subject_medians(e1["test_rows"], "auroc")
    e2_auroc = subject_medians(e2["test_rows"], "auroc")
    e1_delta = subject_medians(e1["test_rows"], "cliffs_delta")
    e2_delta = subject_medians(e2["test_rows"], "cliffs_delta")
    subjects = sorted(set(e1_auroc) & set(e2_auroc))
    median_gain = float(
        np.median([e2_auroc[s] for s in subjects]) - np.median([e1_auroc[s] for s in subjects])
    )
    auroc_improved = sum(e2_auroc[s] > e1_auroc[s] for s in subjects)
    delta_improved = sum(e2_delta[s] > e1_delta[s] for s in subjects)
    e1_by_run = {
        (row["subject_id"], int(row["seed"])): row for row in e1["test_rows"]
    }
    e2_by_run = {
        (row["subject_id"], int(row["seed"])): row for row in e2["test_rows"]
    }
    differential = []
    for key in sorted(set(e1_by_run) & set(e2_by_run)):
        left, right = e1_by_run[key], e2_by_run[key]
        differential.append(
            (float(right["fog_p50"]) - float(left["fog_p50"]))
            - (float(right["nonfog_p50"]) - float(left["nonfog_p50"]))
        )
    differential_gain = float(np.median(differential)) if differential else math.nan
    e1_shift = subject_medians(e1["shift_rows"], "shift_robust")
    e2_shift = subject_medians(e2["shift_rows"], "shift_robust")
    common_shift = sorted(set(e1_shift) & set(e2_shift))
    shift_ratio = float(
        np.median([e2_shift[s] for s in common_shift])
        / max(np.median([e1_shift[s] for s in common_shift]), 1e-12)
    )
    passed = bool(
        sum(v1.values()) >= 5
        and (median_gain >= 0.03 or auroc_improved >= 4)
        and delta_improved >= 4
        and differential_gain > 0.0
        and shift_ratio <= 1.10
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "v1_pass_subjects": [subject for subject, value in v1.items() if value],
        "v1_pass_count": sum(v1.values()),
        "required_v1_pass_count": 5,
        "median_auroc_gain_vs_E1": median_gain,
        "auroc_improved_subjects": auroc_improved,
        "cliffs_delta_improved_subjects": delta_improved,
        "median_differential_fog_minus_nonfog_error_gain": differential_gain,
        "nonfog_shift_ratio_vs_E1": shift_ratio,
        "maximum_allowed_shift_ratio": 1.10,
        "mechanism_note": "FoG-vs-Non-FoG error asymmetry is also reported per run in fog_nonfog_reconstruction_gap.csv",
    }


def lowpass_runs(
    source: dict[tuple[str, int], RunOutputs], cutoff_hz: float = 8.0
) -> dict[tuple[str, int], RunOutputs]:
    frequency = np.fft.rfftfreq(WINDOW, d=1.0 / FS)
    keep = frequency <= cutoff_hz
    outputs: dict[tuple[str, int], RunOutputs] = {}
    for key, run in source.items():
        predicted: dict[str, np.ndarray] = {}
        for name, actual in run.actual.items():
            spectrum = np.fft.rfft(actual, axis=1)
            spectrum[:, ~keep, :] = 0.0
            predicted[name] = np.fft.irfft(spectrum, n=WINDOW, axis=1).astype(np.float32)
        outputs[key] = RunOutputs(
            stage="LOWPASS8",
            subject=run.subject,
            seed=run.seed,
            model_name="zero_phase_fft_lowpass_8hz",
            model_config={"cutoff_hz": cutoff_hz, "causal": False, "trainable": False},
            training={"trainable": False, "parameter_count": 0},
            rows=run.rows,
            actual=run.actual,
            predicted=predicted,
            residual={name: run.actual[name] - predicted[name] for name in predicted},
            run_dir=run.run_dir,
        )
    return outputs


def e3_gate(
    baseline: dict[str, Any],
    e3: dict[str, Any],
    lowpass: dict[str, Any],
    *,
    probe_available: bool,
) -> dict[str, Any]:
    v1 = v1_subject_passes(e3["reconstruction_rows"])
    baseline_auroc = subject_medians(baseline["test_rows"], "auroc")
    baseline_pr = subject_medians(baseline["test_rows"], "average_precision")
    e3_auroc = subject_medians(e3["test_rows"], "auroc")
    e3_pr = subject_medians(e3["test_rows"], "average_precision")
    lowpass_auroc = subject_medians(lowpass["test_rows"], "auroc")
    subjects = sorted(set(baseline_auroc) & set(e3_auroc))
    ranking_improved = sum(
        e3_auroc[s] > baseline_auroc[s] or e3_pr[s] > baseline_pr[s] for s in subjects
    )
    e3_shift = subject_medians(e3["shift_rows"], "shift_robust")
    baseline_shift = subject_medians(baseline["shift_rows"], "shift_robust")
    common_shift = sorted(set(e3_shift) & set(baseline_shift))
    shift_ratio = float(
        np.median([e3_shift[s] for s in common_shift])
        / max(np.median([baseline_shift[s] for s in common_shift]), 1e-12)
    )
    lowpass_outperformed = sum(
        e3_auroc[s] > lowpass_auroc[s] for s in sorted(set(e3_auroc) & set(lowpass_auroc))
    )
    numeric_pass = bool(
        sum(v1.values()) >= 5
        and ranking_improved >= 4
        and shift_ratio <= 1.10
        and lowpass_outperformed >= 4
    )
    status = "PASS" if numeric_pass and probe_available else "INCOMPLETE_PROBE" if numeric_pass else "FAIL"
    return {
        "status": status,
        "numeric_gate_pass": numeric_pass,
        "v1_pass_count": sum(v1.values()),
        "required_v1_pass_count": 5,
        "ranking_improved_subjects": ranking_improved,
        "required_ranking_improved_subjects": 4,
        "nonfog_shift_ratio_vs_best_E1_E2": shift_ratio,
        "maximum_allowed_shift_ratio": 1.10,
        "subjects_outperforming_lowpass_auroc": lowpass_outperformed,
        "required_lowpass_outperformance_subjects": 4,
        "compatible_frozen_raw_tcn_probe_available": probe_available,
        "test_used_for_selection": False,
    }


def reconstruction_gap(
    stage_result: dict[str, Any], baseline_result: dict[str, Any], output: Path
) -> None:
    rows: list[dict[str, Any]] = []
    current = {
        (row["subject_id"], int(row["seed"])): row for row in stage_result["test_rows"]
    }
    baseline = {
        (row["subject_id"], int(row["seed"])): row for row in baseline_result["test_rows"]
    }
    for key in sorted(set(current) & set(baseline)):
        rows.append(
            {
                "subject_id": key[0],
                "seed": key[1],
                "current_fog_score_median": current[key]["fog_p50"],
                "baseline_fog_score_median": baseline[key]["fog_p50"],
                "delta_fog_error": float(current[key]["fog_p50"]) - float(baseline[key]["fog_p50"]),
                "current_nonfog_score_median": current[key]["nonfog_p50"],
                "baseline_nonfog_score_median": baseline[key]["nonfog_p50"],
                "delta_nonfog_error": float(current[key]["nonfog_p50"]) - float(baseline[key]["nonfog_p50"]),
                "differential_gain": (
                    float(current[key]["fog_p50"]) - float(baseline[key]["fog_p50"])
                )
                - (
                    float(current[key]["nonfog_p50"]) - float(baseline[key]["nonfog_p50"])
                ),
            }
        )
    write_csv(output, rows)


def comparison_outputs(
    root: Path,
    results: dict[str, dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> None:
    matrix: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for experiment, result in results.items():
        for subject in FORMAL_SUBJECTS:
            test = [row for row in result["test_rows"] if row["subject_id"] == subject]
            recon = [row for row in result["reconstruction_rows"] if row["subject_id"] == subject]
            shift = [row for row in result["shift_rows"] if row["subject_id"] == subject]
            q95 = [
                row
                for row in result["threshold_rows"]
                if row["subject_id"] == subject and float(row["quantile"]) == 95.0
            ]
            q992 = [
                row
                for row in result["threshold_rows"]
                if row["subject_id"] == subject and float(row["quantile"]) == 99.2
            ]
            if not test:
                continue
            matrix.append(
                {
                    "subject_id": subject,
                    "experiment": experiment,
                    "nrmse": a5.finite_median(recon, "nrmse_median"),
                    "pearson": a5.finite_median(recon, "pearson_median"),
                    "delta_pearson": a5.finite_median(recon, "delta_pearson_median"),
                    "psd_distance": a5.finite_median(recon, "psd_log_distance_median"),
                    "auroc": a5.finite_median(test, "auroc"),
                    "pr_auc": a5.finite_median(test, "average_precision"),
                    "cliffs_delta": a5.finite_median(test, "cliffs_delta"),
                    "shift_robust": a5.finite_median(shift, "shift_robust"),
                    "recall_q95": a5.finite_median(q95, "window_recall"),
                    "recall_q99p2": a5.finite_median(q992, "window_recall"),
                    "event_fa_per_min_q99p2": a5.finite_median(
                        q992, "event_false_alarm_per_minute"
                    ),
                }
            )
        experiment_rows = [row for row in matrix if row["experiment"] == experiment]
        v1 = v1_subject_passes(result["reconstruction_rows"])
        summaries.append(
            {
                "experiment": experiment,
                "v1_pass_count": sum(v1.values()),
                "auroc_median": a5.finite_median(experiment_rows, "auroc"),
                "pr_auc_median": a5.finite_median(experiment_rows, "pr_auc"),
                "cliffs_delta_median": a5.finite_median(experiment_rows, "cliffs_delta"),
                "shift_robust_median": a5.finite_median(experiment_rows, "shift_robust"),
                "event_fa_per_min_q99p2_median": a5.finite_median(
                    experiment_rows, "event_fa_per_min_q99p2"
                ),
            }
        )
    write_csv(root / "comparison_matrix.csv", matrix)
    write_csv(root / "experiment_summary.csv", summaries)

    comparisons: list[tuple[str, str]] = []
    if "E0" in results and "E1" in results:
        comparisons.append(("E0", "E1"))
    if "E1" in results and "E2" in results:
        comparisons.append(("E1", "E2"))
    if "E0" in results:
        comparisons.extend(("E0", name) for name in results if name not in ("E0", "E1", "E2"))
    statistics_rows: list[dict[str, Any]] = []
    for left_name, right_name in comparisons:
        for metric in ("auroc", "pr_auc", "cliffs_delta", "shift_robust"):
            left = {
                row["subject_id"]: float(row[metric])
                for row in matrix
                if row["experiment"] == left_name
            }
            right = {
                row["subject_id"]: float(row[metric])
                for row in matrix
                if row["experiment"] == right_name
            }
            statistics_rows.append(
                {
                    "comparison": f"{right_name}-{left_name}",
                    "metric": metric,
                    **paired_subject_statistics(
                        left,
                        right,
                        bootstrap_samples=bootstrap_samples,
                        seed=20260806 + len(statistics_rows),
                    ),
                }
            )
    adjusted = benjamini_hochberg([float(row["wilcoxon_p"]) for row in statistics_rows])
    for row, value in zip(statistics_rows, adjusted):
        row["wilcoxon_p_bh"] = value
    write_csv(root / "paired_statistics.csv", statistics_rows)


def final_r5_eligibility(
    results: dict[str, dict[str, Any]], *, full_protocol: bool
) -> dict[str, Any]:
    main_names = [name for name in ("E0", "E1", "E2", "E3A") if name in results]
    validation_ranks: list[tuple[tuple[float, float, float], str]] = []
    for name in main_names:
        gate = results[name]["gate"]
        candidate = next(
            row for row in gate["candidate_summaries"] if row["score"] == gate["selected_score"]
        )
        validation_ranks.append(
            (
                (
                    float(candidate["median_validation_auroc"]),
                    float(candidate["median_validation_cliffs_delta"]),
                    float(candidate["median_validation_average_precision"]),
                ),
                name,
            )
        )
    selected = max(validation_ranks)[1]
    chosen = results[selected]
    v1 = v1_subject_passes(chosen["reconstruction_rows"])
    auroc = subject_medians(chosen["test_rows"], "auroc")
    delta = subject_medians(chosen["test_rows"], "cliffs_delta")
    shift = subject_medians(chosen["shift_rows"], "shift_robust")
    baseline_auroc = subject_medians(results["E0"]["test_rows"], "auroc")
    baseline_shift = subject_medians(results["E0"]["shift_rows"], "shift_robust")
    common = sorted(set(auroc) & set(baseline_auroc))
    better_than_e0 = sum(auroc[subject] > baseline_auroc[subject] for subject in common)
    shift_improved = float(np.median(list(shift.values()))) < float(
        np.median(list(baseline_shift.values()))
    )
    conditions = {
        "v1_at_least_5_of_7": sum(v1.values()) >= 5,
        "nonfog_shift_better_than_E0": shift_improved,
        "auroc_above_0p60_at_least_5_of_7": sum(value > 0.60 for value in auroc.values()) >= 5,
        "cliffs_delta_at_least_0p33_at_least_4_of_7": sum(
            value >= 0.33 for value in delta.values()
        )
        >= 4,
        "better_than_E0_in_majority": better_than_e0 >= 4,
        "selection_used_validation_not_test": True,
        "all_schemes_preregistered_before_test": True,
    }
    eligible = full_protocol and all(conditions.values())
    return {
        "status": "PASS" if eligible else "NOT_APPLICABLE_SMOKE" if not full_protocol else "FAIL",
        "selected_experiment_by_validation": selected,
        "validation_rank": [
            {"experiment": name, "rank_tuple": rank} for rank, name in sorted(validation_ranks, reverse=True)
        ],
        "conditions": conditions,
        "v1_pass_count": sum(v1.values()),
        "auroc_above_0p60_count": sum(value > 0.60 for value in auroc.values()),
        "cliffs_delta_above_0p33_count": sum(value >= 0.33 for value in delta.values()),
        "subjects_better_than_E0": better_than_e0,
        "eligible_for_R5_TCN": eligible,
        "test_used_for_model_or_score_selection": False,
    }


def render_final_report(root: Path, gate: dict[str, Any]) -> None:
    conditions = "\n".join(
        f"- {'PASS' if value else 'FAIL'}: {name}" for name, value in gate["conditions"].items()
    )
    text = f"""# Daphnet NBM E0--E3 final decision

- Status: **{gate['status']}**
- Validation-selected experiment: **{gate['selected_experiment_by_validation']}**
- Eligible for R5-TCN: **{gate['eligible_for_R5_TCN']}**
- Test used for selection: **no**

## Gate conditions

{conditions}

The comparison matrix and paired statistics use subjects, not overlapping windows,
as the inferential unit.  The E0 full-A5 reproduction is retained separately from
the E3A common-support paired comparison.
"""
    (root / "E0_E3_final_report.md").write_text(text, encoding="utf-8")


def protocol_payload(
    args: argparse.Namespace,
    subjects: Sequence[str],
    seeds: Sequence[int],
    manifest_path: Path,
    support_audit: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "outline": str(Path(r"C:\Users\bin\Downloads\daphnet_nbm_E0_E3_experiment_outline.md")),
        "data_dir": str(args.data_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "subjects": list(subjects),
        "formal_subjects": [subject for subject in FORMAL_SUBJECTS if subject in subjects],
        "diagnostic_subjects": [subject for subject in DIAGNOSTIC_SUBJECTS if subject in subjects],
        "clean_controls": [subject for subject in CLEAN_CONTROLS if subject in subjects],
        "seeds": list(seeds),
        "window_samples": WINDOW,
        "stride_samples": FS,
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "loss": "L4=0.70 SmoothL1 + 0.15 correlation + 0.15 first-difference MSE",
        },
        "comparison_support": {
            "policy": "E3A past-four-second eligible target window IDs",
            "full_E0_A5_reproduction_retained_separately": True,
            "kept_rows": sum(bool(row.get("kept")) for row in support_audit),
            "dropped_rows": sum(not bool(row.get("kept")) for row in support_audit),
        },
        "C1": {
            "selection": "chronological first half calibration; non-overlapping second half score/threshold",
            "deployment": "full validation clean Non-FoG fit, then frozen to test",
            "clip": None,
        },
        "thresholds": [95.0, 99.2],
        "test_fog_used_for_selection": False,
        "smoke": args.smoke,
    }


def benchmark(
    root: Path,
    bundles: dict[str, SubjectBundle],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    subject = "S01" if "S01" in bundles else next(iter(bundles))
    item = bundles[subject]
    epochs = max(1, args.benchmark_epochs)
    rows: list[dict[str, Any]] = []
    settings: list[tuple[str, Callable[[], nn.Module], np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]] = []
    settings.append(
        (
            "E0_M3",
            lambda: a1b.ContextM3(WINDOW),
            item.processed["train_nonfog"],
            item.processed["train_nonfog"],
            item.processed["earlystop_nonfog"],
            item.processed["earlystop_nonfog"],
            False,
        )
    )
    settings.append(
        (
            "E2_P24",
            lambda: TrueBottleneckAE("P24"),
            item.processed["train_nonfog"],
            item.processed["train_nonfog"],
            item.processed["earlystop_nonfog"],
            item.processed["earlystop_nonfog"],
            False,
        )
    )
    settings.append(
        (
            "E2_P16",
            lambda: TrueBottleneckAE("P16"),
            item.processed["train_nonfog"],
            item.processed["train_nonfog"],
            item.processed["earlystop_nonfog"],
            item.processed["earlystop_nonfog"],
            False,
        )
    )
    e3a_train_x, e3a_train_y, _, _ = context_target_arrays(
        item, "nbm_internal_train_nonfog", mode="E3A"
    )
    e3a_early_x, e3a_early_y, _, _ = context_target_arrays(
        item, "nbm_internal_earlystop_nonfog", mode="E3A"
    )
    settings.append(
        (
            "E3A_C24",
            lambda: HistoryPredictor(24, CHANNELS),
            e3a_train_x,
            e3a_train_y,
            e3a_early_x,
            e3a_early_y,
            False,
        )
    )
    settings.append(
        (
            "E3A_C48",
            lambda: HistoryPredictor(48, CHANNELS),
            e3a_train_x,
            e3a_train_y,
            e3a_early_x,
            e3a_early_y,
            False,
        )
    )
    if args.include_e3b:
        e3b_train_x, e3b_train_y, _, _ = context_target_arrays(
            item, "nbm_internal_train_nonfog", mode="E3B"
        )
        e3b_early_x, e3b_early_y, _, _ = context_target_arrays(
            item, "nbm_internal_earlystop_nonfog", mode="E3B"
        )
        settings.append(
            (
                "E3B_C24",
                lambda: HistoryPredictor(24, CHANNELS + 1),
                e3b_train_x,
                e3b_train_y,
                e3b_early_x,
                e3b_early_y,
                True,
            )
        )
        settings.append(
            (
                "E3B_C48",
                lambda: HistoryPredictor(48, CHANNELS + 1),
                e3b_train_x,
                e3b_train_y,
                e3b_early_x,
                e3b_early_y,
                True,
            )
        )
    benchmark_root = root / "runtime_benchmark"
    for index, (name, factory, train_x, train_y, early_x, early_y, is_e3b) in enumerate(settings):
        limit = min(len(train_x), 512)
        early_limit = min(len(early_x), 256)
        run_dir = benchmark_root / name
        started = time.perf_counter()
        model, training, _ = train_pair_model(
            factory,
            train_x[:limit],
            train_y[:limit],
            early_x[:early_limit],
            early_y[:early_limit],
            run_dir,
            stage=f"BENCH_{name}",
            subject=subject,
            seed=SEEDS[0] + index,
            max_epochs=epochs,
            patience=epochs + 1,
            batch_size=args.batch_size,
            workers=0,
            device=device,
            overwrite=True,
            e3b=is_e3b,
        )
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "setting": name,
                "subject": subject,
                "benchmark_epochs": epochs,
                "train_windows": limit,
                "earlystop_windows": early_limit,
                "elapsed_seconds": elapsed,
                "seconds_per_epoch": elapsed / epochs,
                "parameter_count": training["parameter_count"],
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(benchmark_root / "benchmark.csv", rows)
    payload = {
        "device": str(device),
        "torch_cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "rows": rows,
        "interpretation": "Short benchmarks include checkpoint and validation overhead; formal early-stop epochs remain uncertain.",
    }
    write_json(benchmark_root / "benchmark.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    if args.smoke:
        if args.subjects == ",".join(ALL_SUBJECTS):
            args.subjects = "S01"
        if args.seeds == ",".join(map(str, SEEDS)):
            args.seeds = str(SEEDS[0])
        args.max_epochs = min(args.max_epochs, 1)
        args.patience = min(args.patience, 1)
        if args.max_windows_per_role <= 0:
            args.max_windows_per_role = 64
        if args.output_root == ROOT / "outputs" / EXPERIMENT:
            args.output_root = ROOT / "outputs" / f"_{EXPERIMENT}_smoke"
        args.allow_e0_mismatch = True
    subjects = parse_subjects(args.subjects)
    seeds = parse_seeds(args.seeds)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.resolve()
    bundles, manifest_rows, channel_names = load_data(
        data_dir,
        subjects,
        max_windows_per_role=args.max_windows_per_role,
    )
    support, support_audit = common_support(bundles)
    write_csv(root / "protocol" / "E3A_common_support_audit.csv", support_audit)
    manifest_path = manifest_a5.resolve_a5_artifact(data_dir, "a5_window_manifest.csv")
    protocol = protocol_payload(args, subjects, seeds, manifest_path, support_audit)
    write_json(root / "protocol" / "frozen_protocol.json", protocol)
    write_json(
        root / "protocol" / "train_only_scalers.json",
        {subject: item.scaler for subject, item in bundles.items()},
    )
    split_audit = []
    for subject, item in bundles.items():
        for role in ROLES:
            split_audit.append(
                {
                    "subject_id": subject,
                    "subject_scope": item.scope,
                    "a5_role": role,
                    "windows_loaded": len(item.role_rows[role]),
                    "E3A_common_support_windows": len(support[(subject, role)]),
                }
            )
    write_csv(root / "protocol" / "split_audit.csv", split_audit)
    if args.benchmark_only:
        result = benchmark(root, bundles, device, args)
        print(f"BENCHMARK COMPLETE results={root / 'runtime_benchmark'}", flush=True)
        return

    results: dict[str, dict[str, Any]] = {}
    e0_runs = run_e0(bundles, seeds, root, args, device)
    e0_full = evaluate_stage(
        "E0",
        e0_runs,
        root / "E0",
        calibration="C0",
        support=None,
        weight_step=args.weight_step,
        channel_names=channel_names,
        suffix="_a5_full",
    )
    reproduction = compare_e0_reference(
        root, e0_full, args.reference_a5_root.resolve(), subjects=subjects, seeds=seeds
    )
    if (
        reproduction.get("status") == "FAIL"
        and reproduction.get("applicable_to_full_protocol")
        and not args.allow_e0_mismatch
    ):
        raise RuntimeError("E0 A5 reproduction gate failed; E1-E3 were not started")
    results["E0"] = evaluate_stage(
        "E0",
        e0_runs,
        root / "E0",
        calibration="C0",
        support=support,
        weight_step=args.weight_step,
        channel_names=channel_names,
    )
    write_stage_contract_files(
        root / "E0", "E0", results["E0"], protocol=protocol, split_audit=split_audit
    )
    if args.stop_after == "E0":
        comparison_outputs(root, results, bootstrap_samples=args.bootstrap_samples)
        print(f"COMPLETE stop_after=E0 results={root}", flush=True)
        return

    results["E1"] = evaluate_stage(
        "E1",
        e0_runs,
        root / "E1",
        calibration="C1",
        support=support,
        weight_step=args.weight_step,
        channel_names=channel_names,
    )
    write_stage_contract_files(
        root / "E1", "E1", results["E1"], protocol=protocol, split_audit=split_audit
    )
    write_csv(
        root / "E1" / "e0_e1_paired_metrics.csv",
        paired_metric_table(results["E0"], results["E1"], "E0", "E1"),
    )
    write_json(root / "E1" / "E1_gate.json", e1_gate(results["E0"], results["E1"]))
    if args.stop_after == "E1":
        comparison_outputs(root, results, bootstrap_samples=args.bootstrap_samples)
        print(f"COMPLETE stop_after=E1 results={root}", flush=True)
        return

    e2_runs = run_e2_variant("P24", bundles, seeds, root, args, device)
    results["E2"] = evaluate_stage(
        "E2",
        e2_runs,
        root / "E2",
        calibration="C1",
        support=support,
        weight_step=args.weight_step,
        channel_names=channel_names,
    )
    write_stage_contract_files(
        root / "E2", "E2", results["E2"], protocol=protocol, split_audit=split_audit
    )
    write_csv(
        root / "E2" / "e1_e2_paired_metrics.csv",
        paired_metric_table(results["E1"], results["E2"], "E1", "E2"),
    )
    profile_rows: list[dict[str, Any]] = []
    for name, model, input_shape in (
        ("E0_M3", a1b.ContextM3(WINDOW), (1, CHANNELS, WINDOW)),
        ("E2_P24", TrueBottleneckAE("P24"), (1, CHANNELS, WINDOW)),
    ):
        profile_rows.append({"model": name, **profile_model(model, input_shape, device=device)})
    if args.include_e2_p16:
        profile_rows.append(
            {
                "model": "E2_P16",
                **profile_model(
                    TrueBottleneckAE("P16"), (1, CHANNELS, WINDOW), device=device
                ),
            }
        )
    write_csv(root / "E2" / "parameter_comparison.csv", profile_rows)
    p24_config = TrueBottleneckAE("P24").architecture_config()
    (root / "E2" / "architecture_summary.md").write_text(
        architecture_markdown("E2-P24", p24_config, profile_rows[1]), encoding="utf-8"
    )
    e2_decision = e2_gate(results["E1"], results["E2"])
    write_json(root / "E2" / "E2_gate.json", e2_decision)
    reconstruction_gap(
        results["E2"], results["E1"], root / "E2" / "fog_nonfog_reconstruction_gap.csv"
    )
    write_csv(
        root / "E2" / "probe_classifier_results.csv",
        [
            {
                "status": "NE",
                "reason": "no preregistered A5_50-compatible frozen Raw-TCN checkpoint was supplied",
                "test_used_for_probe_selection": False,
            }
        ],
    )
    if args.include_e2_p16:
        e2_p16_runs = run_e2_variant("P16", bundles, seeds, root, args, device)
        results["E2_P16"] = evaluate_stage(
            "E2_P16",
            e2_p16_runs,
            root / "E2_P16",
            calibration="C1",
            support=support,
            weight_step=args.weight_step,
            channel_names=channel_names,
        )
        write_stage_contract_files(
            root / "E2_P16",
            "E2_P16",
            results["E2_P16"],
            protocol=protocol,
            split_audit=split_audit,
        )
    if args.stop_after == "E2":
        comparison_outputs(root, results, bootstrap_samples=args.bootstrap_samples)
        print(f"COMPLETE stop_after=E2 results={root}", flush=True)
        return

    if args.e3_capacity == "p24":
        latent_channels = 24
        capacity_reason = "explicit p24"
    elif args.e3_capacity == "m3":
        latent_channels = 48
        capacity_reason = "explicit m3"
    elif e2_decision["status"] == "PASS":
        latent_channels = 24
        capacity_reason = "pre-registered auto rule: E2 P24 passed before E3"
    else:
        latent_channels = 48
        capacity_reason = "pre-registered auto rule: E2 P24 did not pass V1/mechanism gate"
    write_json(
        root / "E3_capacity_decision.json",
        {
            "latent_channels": latent_channels,
            "reason": capacity_reason,
            "E3_test_used_for_choice": False,
        },
    )
    e3a_runs = run_e3("E3A", latent_channels, bundles, seeds, root, args, device)
    results["E3A"] = evaluate_stage(
        "E3A",
        e3a_runs,
        root / "E3",
        calibration="C1",
        support=support,
        weight_step=args.weight_step,
        channel_names=channel_names,
    )
    write_stage_contract_files(
        root / "E3", "E3A", results["E3A"], protocol=protocol, split_audit=split_audit
    )
    e3_profile = profile_model(
        HistoryPredictor(latent_channels, CHANNELS), (1, CHANNELS, 256), device=device
    )
    e3_config = HistoryPredictor(latent_channels, CHANNELS).architecture_config()
    (root / "E3" / "architecture_summary.md").write_text(
        architecture_markdown("E3-A", e3_config, e3_profile), encoding="utf-8"
    )
    write_csv(
        root / "E3" / "fog_normalization_probe.csv",
        [
            {
                "status": "NE",
                "reason": "no preregistered A5_50-compatible frozen Raw-TCN checkpoint was supplied",
                "test_used_for_probe_selection": False,
            }
        ],
    )
    lowpass_result = evaluate_stage(
        "LOWPASS8",
        lowpass_runs(e3a_runs),
        root / "E3" / "lowpass_baseline",
        calibration="C1",
        support=support,
        weight_step=args.weight_step,
        channel_names=channel_names,
    )
    write_csv(root / "E3" / "lowpass_baseline_metrics.csv", lowpass_result["test_rows"])
    baseline_for_e3 = results["E2"] if e2_decision["status"] == "PASS" else results["E1"]
    write_json(
        root / "E3" / "E3_gate.json",
        e3_gate(
            baseline_for_e3,
            results["E3A"],
            lowpass_result,
            probe_available=False,
        ),
    )
    if args.include_e3b:
        e3b_runs = run_e3("E3B", latent_channels, bundles, seeds, root, args, device)
        results["E3B"] = evaluate_stage(
            "E3B",
            e3b_runs,
            root / "E3B",
            calibration="C1",
            support=support,
            weight_step=args.weight_step,
            channel_names=channel_names,
        )
        write_stage_contract_files(
            root / "E3B",
            "E3B",
            results["E3B"],
            protocol=protocol,
            split_audit=split_audit,
        )
    comparison_outputs(root, results, bootstrap_samples=args.bootstrap_samples)
    final_gate = final_r5_eligibility(
        results,
        full_protocol=set(FORMAL_SUBJECTS).issubset(subjects) and set(SEEDS).issubset(seeds),
    )
    write_json(root / "R5_TCN_eligibility_gate.json", final_gate)
    render_final_report(root, final_gate)
    print(f"COMPLETE E0-E3 results={root}", flush=True)


if __name__ == "__main__":
    main()
