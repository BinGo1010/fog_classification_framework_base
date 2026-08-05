"""Full-subject B1 pilot with a ResNet8-style AvgPool8 MLP NBM and InceptionTime."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
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


EXPERIMENT = "daphnet_resnet8_avgpool8_b1_inceptiontime_pilot_v1"
METHOD = "B1"
METHOD_NAME = "ResNet8-AvgPool8-NBM residual + InceptionTime"
METHOD_DIR = "B1_resnet8_avgpool8_residual_inceptiontime"
DEFAULT_SEED = 20260802
NBM_RESUME_INTERVAL = 10


def normalization_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormGELU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1) -> None:
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=kernel_size // 2, bias=False),
            nn.GroupNorm(normalization_groups(out_channels), out_channels),
            nn.GELU(),
        )


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int,
                 stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.norm1 = nn.GroupNorm(normalization_groups(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, bias=False)
        self.norm2 = nn.GroupNorm(normalization_groups(out_channels), out_channels)
        self.activation = nn.GELU()
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(normalization_groups(out_channels), out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.activation(self.norm1(self.conv1(inputs)))
        features = self.norm2(self.conv2(features))
        return self.activation(features + self.shortcut(inputs))


class DecoderUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 5) -> None:
        super().__init__()
        self.projection = ConvNormGELU(in_channels, out_channels, kernel_size)
        self.refinement = ResidualBlock1D(out_channels, out_channels,
                                          kernel_size=kernel_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        upsampled = F.interpolate(inputs, scale_factor=2.0, mode="linear",
                                  align_corners=False)
        return self.refinement(self.projection(upsampled))


class ResNet8AvgPool8NBM(nn.Module):
    """Normal-behaviour autoencoder with a 512-D global embedding.

    Encoder path:
        [B,9,128] -> [B,32,64] -> [B,48,32] -> AvgPool(8)
        -> flatten [B,384] -> MLP embedding [B,512].
    Decoder path:
        [B,512] -> MLP [B,1536] -> [B,48,32]
        -> [B,32,64] -> [B,24,128] -> [B,9,128].
    """

    def __init__(self, input_samples: int = 128) -> None:
        super().__init__()
        if input_samples != 128:
            raise ValueError("This preregistered pilot requires 128 input samples")
        self.input_samples = int(input_samples)
        self.stem = ConvNormGELU(9, 32, kernel_size=7, stride=2)
        self.encoder_block1 = ResidualBlock1D(32, 32, kernel_size=3)
        self.encoder_block2 = ResidualBlock1D(32, 48, kernel_size=3, stride=2)
        self.encoder_block3 = ResidualBlock1D(48, 48, kernel_size=5)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.embedding = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(48 * 8),
            nn.Linear(48 * 8, 512),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.decoder_expansion = nn.Sequential(
            nn.Linear(512, 48 * 32),
            nn.GELU(),
        )
        self.decoder_up1 = DecoderUpBlock(48, 32, kernel_size=5)
        self.decoder_up2 = DecoderUpBlock(32, 24, kernel_size=5)
        self.output = nn.Conv1d(24, 9, kernel_size=1)

    def encode_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        features = self.encoder_block1(features)
        features = self.encoder_block2(features)
        return self.encoder_block3(features)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.pool(self.encode_features(inputs)))

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        features = self.decoder_expansion(embedding).reshape(-1, 48, 32)
        features = self.decoder_up1(features)
        features = self.decoder_up2(features)
        return self.output(features)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (9, self.input_samples):
            raise ValueError(f"expected [B,9,{self.input_samples}], got {tuple(inputs.shape)}")
        embedding = self.encode(inputs)
        return self.decode(embedding), embedding

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": "resnet8_style_avgpool8_mlp_nbm",
            "input_shape": ["batch", 9, 128],
            "encoder_feature_shape": ["batch", 48, 32],
            "pool": "AdaptiveAvgPool1d(8)",
            "pooled_shape": ["batch", 48, 8],
            "embedding_shape": ["batch", 512],
            "decoder_seed_shape": ["batch", 48, 32],
            "decoder_shapes": [["batch", 32, 64], ["batch", 24, 128]],
            "output_shape": ["batch", 9, 128],
            "encoder_kernels": [7, 3, 3, 5],
            "long_skip_connections": False,
            "normalization": "GroupNorm encoder/decoder; LayerNorm pooled vector",
            "activation": "GELU",
            "embedding_dropout": 0.1,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }


def train_nbm(inputs: np.ndarray, item: exp.SubjectWindows, candidate_indices: np.ndarray,
              scaler: exp.RobustScaler, run_dir: Path, seed: int, device: torch.device,
              max_epochs: int, patience: int) -> tuple[nn.Module, dict[str, Any]]:
    """Train the pilot NBM with a resumable ten-epoch checkpoint."""
    del inputs  # Kept for compatibility with the audited base-pipeline signature.
    checkpoint = run_dir / "nbm_best.pt"
    log_path = run_dir / "training_log_nbm.csv"
    resume_path = run_dir / "nbm_resume.pt"
    model = ResNet8AvgPool8NBM(exp.WINDOW).to(device)
    if checkpoint.exists() and log_path.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        return model, dict(payload["training"])

    run_dir.mkdir(parents=True, exist_ok=True)
    train_indices, val_indices = exp.nbm_train_validation(item, candidate_indices)
    train_x = scaler.transform(item.raw[train_indices])
    val_x = scaler.transform(item.raw[val_indices])
    exp.seed_everything(seed)
    model = ResNet8AvgPool8NBM(exp.WINDOW).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    batches = exp.a1b.pair_loader(train_x, train_x, shuffle=True, seed=seed, workers=0)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    bad_epochs = 0
    last_epoch = 0
    elapsed_before = 0.0
    history: list[dict[str, Any]] = []

    if resume_path.exists():
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        best_state = payload["best_state"]
        best_loss = float(payload["best_loss"])
        best_epoch = int(payload["best_epoch"])
        bad_epochs = int(payload["bad_epochs"])
        last_epoch = int(payload["last_epoch"])
        elapsed_before = float(payload.get("elapsed_seconds", 0.0))
        history = list(payload["history"])
        if payload.get("loader_generator_state") is not None and batches.generator is not None:
            batches.generator.set_state(payload["loader_generator_state"].cpu())
        print(f"NBM RESUME {run_dir} epoch={last_epoch + 1}/{max_epochs}", flush=True)

    started = time.perf_counter()
    first_epoch = max_epochs + 1 if bad_epochs >= patience else last_epoch + 1
    for epoch in range(first_epoch, max_epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for batch_x, batch_y in batches:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(batch_x)
            loss = exp.a1b.structural_loss("L4", prediction, batch_y)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient):
                raise FloatingPointError("non-finite pilot NBM gradient")
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
        train_loss = total_loss / count
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
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "improved": improved, "bad_epochs": bad_epochs})
        should_save = (epoch % NBM_RESUME_INTERVAL == 0 or bad_epochs >= patience
                       or epoch == max_epochs)
        if should_save:
            inception.atomic_torch_save({
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
            }, resume_path)
        if bad_epochs >= patience:
            break
    if best_state is None:
        raise AssertionError("pilot NBM produced no checkpoint")
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
    }
    torch.save({
        "model_state": best_state,
        "training": training,
        "train_window_keys": item.keys[train_indices].tolist(),
        "validation_window_keys": item.keys[val_indices].tolist(),
    }, checkpoint)
    exp.write_csv(log_path, history)
    model.load_state_dict(best_state)
    resume_path.unlink(missing_ok=True)
    return model, training


def configure_pipeline(seed: int) -> None:
    exp.EXPERIMENT = EXPERIMENT
    exp.METHODS = (METHOD,)
    exp.METHOD_NAMES = {METHOD: METHOD_NAME}
    exp.METHOD_DIRS = {METHOD: METHOD_DIR}
    exp.SEEDS = (int(seed),)
    exp.NBM_SEED = int(seed)
    exp.a1b.ContextM3 = ResNet8AvgPool8NBM
    exp.train_nbm = train_nbm
    exp.train_classifier = inception.train_classifier


def aggregate_results(root: Path, seed: int, bootstrap_samples: int) -> dict[str, Any]:
    for directory in ("metrics", "predictions", "tables", "reports"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    subject_rows: list[dict[str, Any]] = []
    for subject in exp.SUBJECTS:
        paths = sorted((root / METHOD_DIR / subject).glob(f"*/seed{seed}/test_predictions.csv"))
        if not paths:
            raise FileNotFoundError(f"missing predictions for {subject}")
        rows = [row for path in paths for row in exp.read_csv(path)]
        keys = [(row["record_id"], int(row["window_start"])) for row in rows]
        if len(keys) != len(set(keys)):
            raise AssertionError(f"duplicate outer predictions for {subject}")
        y_true = np.asarray([int(row["y_true"]) for row in rows])
        probability = np.asarray([float(row["y_prob"]) for row in rows])
        prediction = np.asarray([int(row["y_pred"]) for row in rows])
        metrics = exp.binary_metrics(y_true, probability, prediction)
        events = exp.event_metrics(rows)
        subject_rows.append({"subject_id": subject, "method": METHOD,
                             "method_name": METHOD_NAME, "seed": seed,
                             **metrics, **events})
        exp.write_csv(root / "predictions" / subject / f"{METHOD}_seed{seed}.csv", rows)
    exp.write_csv(root / "tables" / "subject_level_results.csv", subject_rows)

    metric_names = list(exp.CLASSIFICATION_METRICS)
    macro: dict[str, Any] = {"method": METHOD, "method_name": METHOD_NAME,
                             "subjects": len(subject_rows), "seed": seed}
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in subject_rows], dtype=float)
        low, high = exp.bootstrap_ci(values, "mean", bootstrap_samples,
                                     seed + exp.stable_int(metric) % 100000)
        macro[f"macro_{metric}"] = float(np.nanmean(values))
        macro[f"median_{metric}"] = float(np.nanmedian(values))
        macro[f"macro_{metric}_ci_low"] = low
        macro[f"macro_{metric}_ci_high"] = high
    exp.write_csv(root / "tables" / "macro_results.csv", [macro])

    metadata_paths = sorted((root / "splits" / "outer_folds").glob("*/*/representation_metadata.json"))
    if len(metadata_paths) != 30:
        raise AssertionError(f"expected 30 completed outer folds, found {len(metadata_paths)}")
    reconstruction_rows: list[dict[str, Any]] = []
    overlap_counts: list[int] = []
    nbm_trainings: list[dict[str, Any]] = []
    for path in metadata_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        reconstruction_rows.append({
            "subject_id": payload["subject_id"],
            "fold_id": payload["fold_id"],
            "test_clean_reconstruction_nrmse": payload["test_clean_reconstruction_nrmse"],
        })
        for manifest in payload["inner_oof"]:
            overlap_counts.append(int(manifest["overlap_count"]))
            nbm_trainings.append(dict(manifest["training"]))
        nbm_trainings.append(dict(payload["final_nbm_training"]))
    exp.write_csv(root / "tables" / "fold_reconstruction_quality.csv", reconstruction_rows)

    run_metrics = sorted((root / METHOD_DIR).glob("*/*/seed*/run_metrics.json"))
    if len(run_metrics) != 30:
        raise AssertionError(f"expected 30 classifier runs, found {len(run_metrics)}")
    classifier_trainings = [json.loads(path.read_text(encoding="utf-8"))["training"]
                            for path in run_metrics]
    audit = {
        "subjects": len(subject_rows),
        "outer_folds": len(metadata_paths),
        "nbm_runs": len(nbm_trainings),
        "classifier_runs": len(run_metrics),
        "oof_manifests": len(overlap_counts),
        "maximum_oof_overlap": max(overlap_counts, default=0),
        "strict_oof_pass": bool(overlap_counts) and max(overlap_counts) == 0,
        "test_data_used_for_selection": False,
        "nbm_hit_max_epochs": sum(int(row["last_epoch"]) == 3000 for row in nbm_trainings),
        "classifier_hit_max_epochs": sum(int(row["last_epoch"]) == 100
                                         for row in classifier_trainings),
    }
    result = {
        "experiment": EXPERIMENT,
        "architecture": ResNet8AvgPool8NBM().architecture_config(),
        "method": METHOD,
        "seed": seed,
        "macro_results": macro,
        "subject_results": subject_rows,
        "audit": audit,
    }
    exp.write_json(root / "FINAL_RESULTS.json", result)
    write_report(root, result)
    return result


def write_report(root: Path, result: dict[str, Any]) -> None:
    macro = result["macro_results"]
    audit = result["audit"]
    lines = [
        "# ResNet8-AvgPool8 NBM + B1 InceptionTime 全被试预实验",
        "",
        "## 架构",
        "",
        "- NBM：ResNet8-style Encoder → AdaptiveAvgPool1d(8) → 512维MLP嵌入。",
        "- Decoder：512→1536展开为[48,32]，两级插值残差UpBlock恢复至[9,128]。",
        "- 分类输入：严格3折OOF有符号残差 B1 = X - X_hat，共9通道。",
        "- 分类器：6模块InceptionTime，第3、6模块后残差连接。",
        "",
        "## 宏平均结果",
        "",
        "| PR-AUC | ROC-AUC | FoG F1 | Recall | Specificity | BAcc | MCC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {macro['macro_pr_auc']:.4f} | {macro['macro_roc_auc']:.4f} | "
        f"{macro['macro_fog_f1']:.4f} | {macro['macro_recall']:.4f} | "
        f"{macro['macro_specificity']:.4f} | {macro['macro_balanced_accuracy']:.4f} | "
        f"{macro['macro_mcc']:.4f} |",
        "",
        "## 审计",
        "",
        f"- 外层折：{audit['outer_folds']}；NBM运行：{audit['nbm_runs']}；分类器运行：{audit['classifier_runs']}。",
        f"- OOF清单：{audit['oof_manifests']}；最大训练/留出重叠：{audit['maximum_oof_overlap']}。",
        f"- 严格OOF：{'PASS' if audit['strict_oof_pass'] else 'FAIL'}。",
        "- 外层测试数据不参与Scaler、NBM训练、早停、类别权重或阈值选择。",
    ]
    path = root / "reports" / "b1_resnet8_avgpool8_inceptiontime_pilot_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs" / EXPERIMENT / "full_subject_b1_experiment")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs" / "daphnet_resnet8_avgpool8_b1_inceptiontime_pilot.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--nbm-max-epochs", type=int, default=3000)
    parser.add_argument("--nbm-patience", type=int, default=100)
    parser.add_argument("--classifier-max-epochs", type=int, default=100)
    parser.add_argument("--classifier-patience", type=int, default=15)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--only-fold", default="", help="SUBJECT/FOLD")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_pipeline(args.seed)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    protocol = {
        "experiment": EXPERIMENT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.resolve()),
        "architecture": ResNet8AvgPool8NBM().architecture_config(),
        "subjects": list(exp.SUBJECTS),
        "outer": "leave-one-complete-valid-record-out",
        "inner": "3-fold record-first purged OOF",
        "method": METHOD,
        "classifier": "InceptionTime-6module",
        "seed": args.seed,
        "test_used_for_selection": False,
        "resume_granularity": "NBM every 10 epochs; classifier every epoch; completed runs skipped",
    }
    exp.write_json(root / "splits" / "frozen_protocol.json", protocol)

    if not args.finalize_only:
        dataset = DaphnetDataset.load(args.data_dir.resolve())
        items = {subject: exp.build_subject_windows(dataset, subject) for subject in exp.SUBJECTS}
        all_folds: list[tuple[str, dict[str, Any]]] = []
        split_summary: list[dict[str, Any]] = []
        for subject, item in items.items():
            for fold in exp.outer_folds(item):
                all_folds.append((subject, fold))
                split_summary.append({
                    "subject_id": subject,
                    "fold_id": fold["fold_id"],
                    "mode": fold["mode"],
                    "train_windows": len(fold["train"]),
                    "test_windows": len(fold["test"]),
                    "test_positive_windows": int(np.sum(item.label[fold["test"]])),
                })
        exp.write_csv(root / "splits" / "outer_folds" / "outer_fold_summary.csv",
                      split_summary)
        selected = all_folds
        if args.only_fold:
            wanted_subject, wanted_fold = args.only_fold.split("/", 1)
            selected = [(subject, fold) for subject, fold in selected
                        if subject == wanted_subject and str(fold["fold_id"]) == wanted_fold]
            if len(selected) != 1:
                raise ValueError(f"unknown --only-fold {args.only_fold}")
        elif args.shard_count > 1:
            if not 0 <= args.shard_index < args.shard_count:
                raise ValueError("shard-index must be in [0, shard-count)")
            selected = [entry for index, entry in enumerate(selected)
                        if index % args.shard_count == args.shard_index]
        if args.smoke:
            selected = selected[:1]
        for position, (subject, fold) in enumerate(selected, 1):
            print(f"OUTER {position}/{len(selected)} {subject}/{fold['fold_id']} device={device}",
                  flush=True)
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
    result = aggregate_results(root, args.seed, args.bootstrap_samples)
    print(f"COMPLETE {root} macro_pr_auc={result['macro_results']['macro_pr_auc']:.6f}",
          flush=True)


if __name__ == "__main__":
    main()
