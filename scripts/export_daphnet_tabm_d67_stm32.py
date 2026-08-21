#!/usr/bin/env python
"""Train and export the final 3-IMU TabM-D67 model for STM32 deployment.

The LOSO checkpoints remain evaluation artifacts.  This script fits the feature
preprocessor on all eligible subjects, trains one final model for a fixed epoch
budget selected before deployment, and exports both Python and C artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import struct
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
import tabm
import torch
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SUBJECTS = ("S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09")
PARAMETER_SPECS = (
    ("feature_median", None),
    ("feature_mean", None),
    ("feature_scale", None),
    ("w1", "model.backbone.blocks.0.0.weight"),
    ("r1", "model.backbone.blocks.0.0.r"),
    ("s1", "model.backbone.blocks.0.0.s"),
    ("b1", "model.backbone.blocks.0.0.bias"),
    ("w2", "model.backbone.blocks.1.0.weight"),
    ("r2", "model.backbone.blocks.1.0.r"),
    ("s2", "model.backbone.blocks.1.0.s"),
    ("b2", "model.backbone.blocks.1.0.bias"),
    ("wout", "model.output.weight"),
    ("bout", "model.output.bias"),
)


class FoGTabM(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = tabm.TabM.make(
            n_num_features=input_dim,
            d_out=2,
            arch_type="tabm",
            k=32,
            n_blocks=2,
            d_block=67,
            dropout=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x_num=x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the final 3-IMU TabM-D67 model for STM32."
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path(
            "outputs/daphnet_binary_mlp_manual_short2_long6_stride1_"
            "excludeS04S10_loso/manual_features.npz"
        ),
    )
    parser.add_argument(
        "--records-dir",
        type=Path,
        default=Path(
            "dataset/1.Daphnet Freezing of Gait Dataset/processed/records"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/stm32_tabm_d67_3imu_all_subjects_seed2026_ep14"
        ),
    )
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0003)
    parser.add_argument("--golden-per-subject", type=int, default=4)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tabm_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weight: torch.Tensor,
) -> torch.Tensor:
    batch_size, k, n_classes = logits.shape
    expanded_target = target[:, None].expand(batch_size, k).reshape(-1)
    return F.cross_entropy(
        logits.reshape(-1, n_classes), expanded_target, weight=class_weight
    )


def predict(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    *,
    return_logits: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    probability_parts: list[np.ndarray] = []
    logits_parts: list[np.ndarray] = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features.astype(np.float32, copy=False))),
        batch_size=4096,
        shuffle=False,
    )
    with torch.inference_mode():
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            # Deployment and golden-vector inference are deliberately FP32.
            # AMP is used only while training; comparing an FP16 CUDA forward
            # pass with the STM32 FP32 reference creates a false export error.
            logits = model(batch)
            logits = logits.float()
            probabilities = logits.softmax(dim=-1).mean(dim=1)[:, 1]
            probability_parts.append(probabilities.cpu().numpy())
            if return_logits:
                logits_parts.append(logits.cpu().numpy())
    return (
        np.concatenate(probability_parts).astype(np.float32),
        np.concatenate(logits_parts).astype(np.float32) if return_logits else None,
    )


def train_final_model(
    features: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[FoGTabM, Pipeline, list[dict], dict]:
    seed_everything(args.seed)
    preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    standardized = preprocessor.fit_transform(features).astype(np.float32)
    counts = np.bincount(labels, minlength=2)
    class_weights_np = len(labels) / (2.0 * counts.astype(np.float64))
    class_weights = torch.tensor(
        class_weights_np, dtype=torch.float32, device=device
    )

    model = FoGTabM(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(standardized),
            torch.from_numpy(labels.astype(np.int64)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )

    history: list[dict] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                logits = model(batch_x)
                loss = tabm_loss(logits, batch_y, class_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(batch_x)
            sample_count += len(batch_x)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d}/{args.epochs} loss={row['train_loss']:.8f}",
            flush=True,
        )

    probability, _ = predict(model, standardized, device)
    prediction = (probability >= 0.5).astype(np.int64)
    train_metrics = {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "f1_macro": float(f1_score(labels, prediction, average="macro")),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "pr_auc_fog": float(average_precision_score(labels, probability)),
        "support": int(len(labels)),
        "class_counts": counts.tolist(),
        "class_weights": class_weights_np.tolist(),
        "runtime_seconds": time.perf_counter() - started,
        "note": "Training-set sanity metrics only; LOSO results remain the generalization estimate.",
    }
    return model, preprocessor, history, train_metrics


def get_parameter_arrays(
    model: nn.Module, preprocessor: Pipeline
) -> dict[str, np.ndarray]:
    state = model.state_dict()
    arrays: dict[str, np.ndarray] = {
        "feature_median": np.asarray(
            preprocessor.named_steps["imputer"].statistics_, dtype=np.float32
        ),
        "feature_mean": np.asarray(
            preprocessor.named_steps["scaler"].mean_, dtype=np.float32
        ),
        "feature_scale": np.asarray(
            preprocessor.named_steps["scaler"].scale_, dtype=np.float32
        ),
    }
    for export_name, state_name in PARAMETER_SPECS:
        if state_name is not None:
            arrays[export_name] = (
                state[state_name].detach().cpu().numpy().astype(np.float32)
            )
    return arrays


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def c_float(value: np.float32) -> str:
    value = np.float32(value)
    if not np.isfinite(value):
        raise ValueError(f"Non-finite parameter cannot be exported: {value}")
    text = format(float(value), ".9g")
    if "e" not in text and "." not in text:
        text += ".0"
    return text + "f"


def write_c_array(handle, name: str, values: np.ndarray) -> None:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    handle.write(f"TABM_ALIGN16 const float tabm_{name}[{len(flat)}] = {{\n")
    for start in range(0, len(flat), 8):
        chunk = ", ".join(c_float(value) for value in flat[start : start + 8])
        handle.write(f"    {chunk},\n")
    handle.write("};\n\n")


def export_binary_and_c(
    arrays: dict[str, np.ndarray], output_dir: Path
) -> dict:
    ordered_names = [name for name, _ in PARAMETER_SPECS]
    binary_path = output_dir / "tabm_d67_params_f32.bin"
    layout_entries = []
    offset = 0
    with binary_path.open("wb") as handle:
        for name in ordered_names:
            array = np.ascontiguousarray(arrays[name], dtype="<f4")
            payload = array.tobytes(order="C")
            handle.write(payload)
            layout_entries.append(
                {
                    "name": name,
                    "shape": list(array.shape),
                    "dtype": "float32_little_endian",
                    "element_count": int(array.size),
                    "byte_offset": offset,
                    "byte_length": len(payload),
                }
            )
            offset += len(payload)

    header_path = output_dir / "tabm_d67_params.h"
    with header_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "#ifndef TABM_D67_PARAMS_H\n#define TABM_D67_PARAMS_H\n\n"
            "#include <stdint.h>\n\n"
            "#if defined(__GNUC__)\n"
            "#define TABM_ALIGN16 __attribute__((aligned(16)))\n"
            "#else\n#define TABM_ALIGN16\n#endif\n\n"
            "#define TABM_INPUT_DIM 90\n"
            "#define TABM_K 32\n"
            "#define TABM_D_BLOCK 67\n"
            "#define TABM_OUTPUT_DIM 2\n"
            "#define TABM_THRESHOLD 0.5f\n\n"
        )
        for name in ordered_names:
            handle.write(
                f"extern const float tabm_{name}[{arrays[name].size}];\n"
            )
        handle.write("\n#endif\n")

    source_path = output_dir / "tabm_d67_params.c"
    with source_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write('#include "tabm_d67_params.h"\n\n')
        for name in ordered_names:
            write_c_array(handle, name, arrays[name])

    layout = {
        "binary_file": binary_path.name,
        "total_bytes": offset,
        "sha256": sha256_file(binary_path),
        "array_order": ordered_names,
        "arrays": layout_entries,
    }
    (output_dir / "parameter_layout.json").write_text(
        json.dumps(layout, indent=2), encoding="utf-8"
    )
    return layout


def write_inference_reference(output_dir: Path) -> None:
    (output_dir / "tabm_d67_inference.h").write_text(
        """#ifndef TABM_D67_INFERENCE_H
#define TABM_D67_INFERENCE_H

#include <stdbool.h>

void tabm_d67_preprocess(const float features[90], float standardized[90]);
float tabm_d67_predict_standardized(const float standardized[90]);
float tabm_d67_predict_features(const float features[90]);
bool tabm_d67_is_fog(const float features[90]);

#endif
""",
        encoding="utf-8",
    )
    (output_dir / "tabm_d67_inference.c").write_text(
        """#include "tabm_d67_inference.h"
#include "tabm_d67_params.h"

#include <math.h>
#include <stddef.h>

void tabm_d67_preprocess(const float features[90], float standardized[90]) {
    for (size_t i = 0; i < TABM_INPUT_DIM; ++i) {
        const float value = isfinite(features[i]) ? features[i] : tabm_feature_median[i];
        standardized[i] = (value - tabm_feature_mean[i]) / tabm_feature_scale[i];
    }
}

static float stable_sigmoid(float value) {
    if (value >= 0.0f) {
        return 1.0f / (1.0f + expf(-value));
    }
    const float exp_value = expf(value);
    return exp_value / (1.0f + exp_value);
}

float tabm_d67_predict_standardized(const float x[90]) {
    float x1[TABM_INPUT_DIM];
    float h1[TABM_D_BLOCK];
    float h2[TABM_D_BLOCK];
    float probability_sum = 0.0f;

    for (size_t k = 0; k < TABM_K; ++k) {
        for (size_t i = 0; i < TABM_INPUT_DIM; ++i) {
            x1[i] = x[i] * tabm_r1[k * TABM_INPUT_DIM + i];
        }
        for (size_t o = 0; o < TABM_D_BLOCK; ++o) {
            float value = 0.0f;
            const size_t weight_base = o * TABM_INPUT_DIM;
            for (size_t i = 0; i < TABM_INPUT_DIM; ++i) {
                value += tabm_w1[weight_base + i] * x1[i];
            }
            value = value * tabm_s1[k * TABM_D_BLOCK + o]
                    + tabm_b1[k * TABM_D_BLOCK + o];
            h1[o] = value > 0.0f ? value : 0.0f;
        }
        for (size_t o = 0; o < TABM_D_BLOCK; ++o) {
            float value = 0.0f;
            const size_t weight_base = o * TABM_D_BLOCK;
            for (size_t i = 0; i < TABM_D_BLOCK; ++i) {
                const float adapted = h1[i] * tabm_r2[k * TABM_D_BLOCK + i];
                value += tabm_w2[weight_base + i] * adapted;
            }
            value = value * tabm_s2[k * TABM_D_BLOCK + o]
                    + tabm_b2[k * TABM_D_BLOCK + o];
            h2[o] = value > 0.0f ? value : 0.0f;
        }

        float logit0 = tabm_bout[k * TABM_OUTPUT_DIM];
        float logit1 = tabm_bout[k * TABM_OUTPUT_DIM + 1];
        for (size_t i = 0; i < TABM_D_BLOCK; ++i) {
            const size_t base = (k * TABM_D_BLOCK + i) * TABM_OUTPUT_DIM;
            logit0 += h2[i] * tabm_wout[base];
            logit1 += h2[i] * tabm_wout[base + 1];
        }
        probability_sum += stable_sigmoid(logit1 - logit0);
    }
    return probability_sum / (float)TABM_K;
}

float tabm_d67_predict_features(const float features[90]) {
    float standardized[TABM_INPUT_DIM];
    tabm_d67_preprocess(features, standardized);
    return tabm_d67_predict_standardized(standardized);
}

bool tabm_d67_is_fog(const float features[90]) {
    return tabm_d67_predict_features(features) >= TABM_THRESHOLD;
}
""",
        encoding="utf-8",
    )


def select_golden_indices(
    subjects: np.ndarray, labels: np.ndarray, per_subject: int
) -> np.ndarray:
    selected: list[int] = []
    normal_count = max(1, per_subject // 2)
    fog_count = max(1, per_subject - normal_count)
    for subject in SUBJECTS:
        subject_indices = np.flatnonzero(subjects == subject)
        for label, count in ((0, normal_count), (1, fog_count)):
            candidates = subject_indices[labels[subject_indices] == label]
            if len(candidates) == 0:
                continue
            positions = np.linspace(0, len(candidates) - 1, count, dtype=int)
            selected.extend(candidates[positions].tolist())
    return np.asarray(selected, dtype=np.int64)


def edge_window(values: np.ndarray, start: int, end: int) -> np.ndarray:
    left = max(0, -start)
    body = values[max(0, start) : min(len(values), end)]
    if left:
        body = np.concatenate(
            [np.repeat(values[:1], left, axis=0), body], axis=0
        )
    if len(body) < end - start:
        body = np.concatenate(
            [body, np.repeat(values[-1:], end - start - len(body), axis=0)],
            axis=0,
        )
    return body[: end - start]


def export_golden_vectors(
    cache,
    selected_indices: np.ndarray,
    standardized: np.ndarray,
    model: nn.Module,
    device: torch.device,
    records_dir: Path,
    output_dir: Path,
) -> dict:
    selected_standardized = standardized[selected_indices]
    probability, logits = predict(
        model, selected_standardized, device, return_logits=True
    )
    raw_windows = []
    for index in selected_indices:
        record_id = str(cache["record_id"][index])
        decision_sample = int(cache["end_sample"][index])
        record = np.load(records_dir / f"{record_id}.npz", allow_pickle=False)
        raw_windows.append(
            edge_window(record["x"], decision_sample - 384, decision_sample)
        )
    raw_windows_array = np.stack(raw_windows).astype(np.float32)
    np.savez_compressed(
        output_dir / "golden_vectors.npz",
        raw_long_window_g=raw_windows_array,
        handcrafted_features=cache["X"][selected_indices].astype(np.float32),
        standardized_features=selected_standardized.astype(np.float32),
        member_logits=logits,
        fog_probability=probability,
        prediction=(probability >= 0.5).astype(np.int64),
        label=cache["y"][selected_indices].astype(np.int64),
        subject=cache["subject"][selected_indices],
        record_id=cache["record_id"][selected_indices],
        short_start_sample=cache["start_sample"][selected_indices],
        decision_sample=cache["end_sample"][selected_indices],
    )
    rows = []
    for local_index, source_index in enumerate(selected_indices):
        rows.append(
            {
                "golden_index": local_index,
                "cache_index": int(source_index),
                "subject": str(cache["subject"][source_index]),
                "record_id": str(cache["record_id"][source_index]),
                "label": int(cache["y"][source_index]),
                "fog_probability": float(probability[local_index]),
                "prediction": int(probability[local_index] >= 0.5),
            }
        )
    with (output_dir / "golden_vectors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "count": len(selected_indices),
        "raw_shape": list(raw_windows_array.shape),
        "feature_shape": [len(selected_indices), 90],
        "logits_shape": list(logits.shape),
    }


def manual_numpy_probability(
    standardized: np.ndarray, arrays: dict[str, np.ndarray]
) -> np.ndarray:
    probabilities = []
    for x in standardized:
        member_probabilities = []
        for k in range(32):
            h1 = (x * arrays["r1"][k]) @ arrays["w1"].T
            h1 = np.maximum(
                h1 * arrays["s1"][k] + arrays["b1"][k], 0.0
            )
            h2 = (h1 * arrays["r2"][k]) @ arrays["w2"].T
            h2 = np.maximum(
                h2 * arrays["s2"][k] + arrays["b2"][k], 0.0
            )
            logits = h2 @ arrays["wout"][k] + arrays["bout"][k]
            difference = float(logits[1] - logits[0])
            probability = (
                1.0 / (1.0 + math.exp(-difference))
                if difference >= 0.0
                else math.exp(difference) / (1.0 + math.exp(difference))
            )
            member_probabilities.append(probability)
        probabilities.append(np.mean(member_probabilities))
    return np.asarray(probabilities, dtype=np.float32)


def write_readme(output_dir: Path, layout: dict, golden: dict) -> None:
    (output_dir / "README_STM32.md").write_text(
        f"""# STM32 TabM-D67 3-IMU deployment package

This is the final all-subject deployment fit, not a LOSO evaluation fold.
Generalization performance must still be quoted from the completed LOSO experiment.

## Files

- `deployment_model.pt`: PyTorch checkpoint and metadata.
- `feature_preprocessor.joblib`: fitted median imputer and StandardScaler.
- `tabm_d67_params.npz`: named float32 arrays for Python inspection.
- `tabm_d67_params_f32.bin`: contiguous little-endian float32 parameter image.
- `parameter_layout.json`: shapes and byte offsets for the binary image.
- `tabm_d67_params.h/.c`: directly compilable parameter arrays.
- `tabm_d67_inference.h/.c`: scalar FP32 reference inference implementation.
- `golden_vectors.npz`: raw windows, features, standardized values, logits and outputs.
- `golden_vectors.csv`: human-readable golden-vector index.
- `feature_names.csv`: exact 90-feature order.
- `training_history.csv`: final all-data fit history.
- `deployment_manifest.json`: training and export audit metadata.

## Dimensions

- Input features: 90
- Ensemble members: 32
- Hidden width: 67
- Classes: 2
- Raw model parameters: 28,471
- Binary package size including preprocessing: {layout['total_bytes']} bytes
- Golden cases: {golden['count']}

`tabm_d67_predict_features()` accepts the unstandardized 90 handcrafted features.
It applies the exported imputation/standardization values, runs all 32 TabM members,
averages member probabilities, and applies a threshold of 0.5 in
`tabm_d67_is_fog()`.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(args.feature_cache, allow_pickle=False)
    features = cache["X"].astype(np.float32, copy=False)
    labels = cache["y"].astype(np.int64, copy=False)
    subjects = cache["subject"].astype(str)
    feature_names = cache["feature_names"].astype(str)
    if features.shape[1] != 90:
        raise RuntimeError(f"Expected 90 features, got {features.shape}")
    if tuple(sorted(np.unique(subjects))) != tuple(sorted(SUBJECTS)):
        raise RuntimeError(f"Unexpected subjects: {np.unique(subjects).tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} samples={len(features)} subjects={len(SUBJECTS)} "
        f"epochs={args.epochs} seed={args.seed}",
        flush=True,
    )
    model, preprocessor, history, train_metrics = train_final_model(
        features, labels, args, device
    )
    standardized = preprocessor.transform(features).astype(np.float32)
    arrays = get_parameter_arrays(model, preprocessor)
    parameter_count = sum(
        array.size
        for name, array in arrays.items()
        if name not in {"feature_median", "feature_mean", "feature_scale"}
    )
    if parameter_count != 28471:
        raise RuntimeError(f"Unexpected model parameter count: {parameter_count}")

    checkpoint = {
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model": {
            "name": "Handcrafted Multi-Scale Feature TabM deployment fit",
            "arch_type": "tabm",
            "input_dim": 90,
            "n_classes": 2,
            "k": 32,
            "n_blocks": 2,
            "d_block": 67,
            "dropout": 0.1,
            "parameter_count": parameter_count,
            "tabm_package_version": tabm.__version__,
        },
        "training_subjects": list(SUBJECTS),
        "excluded_subjects": ["S04", "S10"],
        "feature_names": feature_names.tolist(),
        "class_mapping": {"Normal": 0, "FoG": 1},
        "threshold": 0.5,
        "epochs": args.epochs,
        "seed": args.seed,
        "training_metrics": train_metrics,
    }
    torch.save(checkpoint, args.output_dir / "deployment_model.pt")
    joblib.dump(preprocessor, args.output_dir / "feature_preprocessor.joblib")
    np.savez(args.output_dir / "tabm_d67_params.npz", **arrays)
    with (args.output_dir / "feature_names.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_index", "feature_name"])
        writer.writerows(enumerate(feature_names.tolist()))
    with (args.output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    layout = export_binary_and_c(arrays, args.output_dir)
    write_inference_reference(args.output_dir)
    selected_indices = select_golden_indices(
        subjects, labels, args.golden_per_subject
    )
    golden = export_golden_vectors(
        cache,
        selected_indices,
        standardized,
        model,
        device,
        args.records_dir,
        args.output_dir,
    )

    pytorch_probability, _ = predict(
        model, standardized[selected_indices], device
    )
    numpy_probability = manual_numpy_probability(
        standardized[selected_indices], arrays
    )
    max_probability_error = float(
        np.max(np.abs(pytorch_probability - numpy_probability))
    )
    decisions_match = bool(
        np.array_equal(
            pytorch_probability >= 0.5, numpy_probability >= 0.5
        )
    )
    if max_probability_error > 2e-5 or not decisions_match:
        raise RuntimeError(
            "Export verification failed: "
            f"max_probability_error={max_probability_error}, "
            f"decisions_match={decisions_match}"
        )

    manifest = {
        "purpose": "STM32 deployment fit and parameter export",
        "deployment_fit": True,
        "not_a_loso_fold": True,
        "source_feature_cache": str(args.feature_cache.resolve()),
        "source_records_dir": str(args.records_dir.resolve()),
        "training_subjects": list(SUBJECTS),
        "excluded_subjects": ["S04", "S10"],
        "seed": args.seed,
        "epochs": args.epochs,
        "epoch_selection": "rounded median (13.5 -> 14) of 24 LOSO best epochs",
        "sampling_rate_hz": 64,
        "short_window_samples": 128,
        "long_window_samples": 384,
        "stride_samples": 64,
        "label_rule": "any_fog",
        "threshold": 0.5,
        "input_dim": 90,
        "parameter_count": parameter_count,
        "preprocessing_float_count": 270,
        "binary_export": layout,
        "golden_vectors": golden,
        "verification": {
            "manual_numpy_vs_pytorch_max_abs_probability_error": max_probability_error,
            "threshold_decisions_match": decisions_match,
        },
        "training_set_sanity_metrics": train_metrics,
        "software": {
            "python_torch": torch.__version__,
            "tabm": tabm.__version__,
            "sklearn": sklearn.__version__,
        },
        "device": str(device),
    }
    (args.output_dir / "deployment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_readme(args.output_dir, layout, golden)

    file_hashes = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            file_hashes.append(f"{sha256_file(path)}  {path.name}")
    (args.output_dir / "SHA256SUMS.txt").write_text(
        "\n".join(file_hashes) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
