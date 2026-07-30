#!/usr/bin/env python
"""Strict integrity audit for the Transformer-NBM F0--F5 fusion suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_daphnet_fusion_f0_f5_suite as suite
import run_daphnet_tcn_rf_ablation as rf
from cnbr_fog.evaluation import aggregate_fold_metrics
from cnbr_fog.fusion_representations import build_fusion_representations
from cnbr_fog.histories import make_common_history_plan
from cnbr_fog.resume import (
    atomic_json_dump,
    canonical_fingerprint,
    sha256_file,
    validate_done,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Daphnet Transformer-NBM F0--F5 fusion outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Audit a finalized subset with at least one completed classifier "
            "cell without requiring all 48"
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _close(
    observed: Any,
    expected: Any,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> bool:
    if observed == "":
        observed = None
    if expected == "":
        expected = None
    if observed is None or expected is None:
        return observed is expected
    try:
        left = float(observed)
        right = float(expected)
    except (TypeError, ValueError):
        return observed == expected
    if not math.isfinite(left) or not math.isfinite(right):
        return (
            math.isnan(left)
            and math.isnan(right)
            or left == right
        )
    return math.isclose(left, right, rel_tol=rtol, abs_tol=atol)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _issue(
    failures: list[str],
    message: str,
) -> None:
    failures.append(message)


def _validate_protocol(
    result_dir: Path,
    config: Mapping[str, Any],
    failures: list[str],
) -> None:
    if config.get("suite_version") != suite.SUITE_VERSION:
        _issue(failures, "unexpected suite_version")
    if config.get("folds_resolved") != list(suite.EXPECTED_LOSO_SUBJECTS):
        _issue(failures, "LOSO fold registry changed")
    if config.get("excluded_subjects") != ["S04", "S10"]:
        _issue(failures, "S04/S10 exclusion changed")
    if config.get("nbm") != suite.SOURCE_NBM:
        _issue(failures, "source NBM is not Transformer")
    if config.get("expected_classifier_cells") != 48:
        _issue(failures, "expected classifier count is not 48")
    cells = config.get("fusion_representations", [])
    if [cell.get("fusion_id") for cell in cells] != list(
        suite.FUSION_IDS
    ):
        _issue(failures, "F0--F5 cell registry changed")
    expected_channels = {
        name: definition["in_channels"]
        for name, definition in suite.FUSION_REPRESENTATIONS.items()
    }
    observed_channels = {
        cell.get("fusion_id"): cell.get("in_channels")
        for cell in cells
    }
    if observed_channels != expected_channels:
        _issue(failures, "F0--F5 channel widths changed")
    if config.get("comparisons") != list(suite.COMPARISONS):
        _issue(failures, "paired comparison registry changed")
    if config.get("primary_comparison") != "F2_minus_F0":
        _issue(failures, "primary comparison changed")
    if config.get("history_samples") != 256:
        _issue(failures, "history length is not 256 samples")
    if config.get("history_blocks") != 8:
        _issue(failures, "history block count is not eight")
    classifier = config.get("classifier", {})
    if classifier.get("dilations") != [1, 2, 4, 8, 8, 8]:
        _issue(failures, "TCN-M dilation schedule changed")
    if classifier.get("receptive_field_samples") != 125:
        _issue(failures, "TCN-M receptive field changed")
    if config.get("seed") != 42:
        _issue(failures, "formal seed is not 42")
    formal = {
        "reportable": True,
        "run_kind": "formal",
        "classifier_epochs": 12,
        "classifier_patience": 4,
        "classifier_lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "max_classifier_windows": 0,
        "bootstrap_samples": 100000,
        "bootstrap_seed": 42,
        "amp": True,
        "deterministic": True,
    }
    for key, expected in formal.items():
        if config.get(key) != expected:
            _issue(failures, f"formal protocol option changed: {key}")
    run_manifest_path = result_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        _issue(failures, "missing run_manifest.json")
    else:
        run_manifest = _load_json(run_manifest_path)
        runtime_fields = {
            "data_dir",
            "source_suite_dir",
            "output_dir",
            "device",
            "num_workers",
            "resume",
            "smoke",
            "source_runtime_config",
        }
        config_scientific = {
            key: value
            for key, value in config.items()
            if key not in runtime_fields
        }
        if run_manifest != config_scientific:
            _issue(
                failures,
                "config scientific payload differs from run_manifest",
            )
        fingerprint = run_manifest.pop("protocol_fingerprint", None)
        if fingerprint != config.get("protocol_fingerprint"):
            _issue(failures, "run manifest fingerprint differs from config")
        if canonical_fingerprint(run_manifest) != fingerprint:
            _issue(failures, "protocol fingerprint cannot be reproduced")
    try:
        current_impl = suite.implementation_manifest()
        if config.get("implementation") != current_impl:
            _issue(failures, "training implementation files changed")
    except Exception as exc:  # pragma: no cover - defensive audit reporting
        _issue(failures, f"implementation manifest check failed: {exc}")


def _source_and_dataset(
    config: Mapping[str, Any],
    failures: list[str],
) -> tuple[Any | None, Any | None]:
    source_dir = Path(str(config.get("source_suite_dir", "")))
    data_dir = Path(str(config.get("data_dir", "")))
    try:
        source_manifest, source_config = suite.build_source_manifest(
            source_dir
        )
        if source_manifest != config.get("source"):
            _issue(failures, "immutable source manifest drifted")
        reconstruction = {
            key: source_config[key]
            for key in (
                "sampling_rate_hz",
                "nbm_hidden",
                "nbm_dropout",
                "linear_ar_seconds",
                "gru_layers",
                "transformer_heads",
                "transformer_layers",
                "transformer_ffn",
            )
        }
        if reconstruction != config.get("source_model_reconstruction"):
            _issue(
                failures,
                "Transformer reconstruction hyperparameters drifted",
            )
    except Exception as exc:
        _issue(failures, f"source manifest validation failed: {exc}")
        return None, None
    try:
        dataset, windows, data_sha = rf.load_dataset_and_windows(
            data_dir,
            source_config,
        )
        if data_sha != config.get("data_sha256"):
            _issue(failures, "processed dataset fingerprint drifted")
        return dataset, windows
    except Exception as exc:
        _issue(failures, f"dataset validation failed: {exc}")
        return None, None


def _load_primitives(
    cache_path: Path,
) -> dict[str, dict[str, np.ndarray]]:
    with np.load(cache_path, allow_pickle=False) as payload:
        if set(payload.files) != suite._primitive_cache_keys():
            raise ValueError("primitive cache key set changed")
        return {
            split: {
                "raw": np.asarray(
                    payload[f"{split}_raw"],
                    dtype=np.float32,
                ),
                "error": np.asarray(
                    payload[f"{split}_error"],
                    dtype=np.float32,
                ),
                "sigma": np.asarray(
                    payload[f"{split}_sigma"],
                    dtype=np.float32,
                ),
                "y": np.asarray(
                    payload[f"{split}_y"],
                    dtype=np.int8,
                ),
                "window_index": np.asarray(
                    payload[f"{split}_window_index"],
                    dtype=np.int64,
                ),
            }
            for split in ("train", "validation", "test")
        }


def _validate_predictions(
    root: Path,
    metrics: Mapping[str, Any],
    support: Mapping[str, np.ndarray],
    dataset: Any,
    windows: Any,
) -> list[str]:
    failures: list[str] = []
    with np.load(
        root / "validation_predictions.npz",
        allow_pickle=False,
    ) as payload:
        validation = {
            key: np.asarray(payload[key])
            for key in ("window_index", "y_true", "y_prob", "y_pred")
        }
    with np.load(root / "predictions.npz", allow_pickle=False) as payload:
        test = {
            key: np.asarray(payload[key])
            for key in ("window_index", "y_true", "y_prob", "y_pred")
        }
    structurally_invalid = False
    for split_name, arrays in (
        ("validation", validation),
        ("test", test),
    ):
        if any(value.ndim != 1 for value in arrays.values()):
            failures.append(f"{split_name} predictions are not 1-D")
            structurally_invalid = True
        if len({len(value) for value in arrays.values()}) != 1:
            failures.append(f"{split_name} prediction lengths differ")
            structurally_invalid = True
        if (
            not np.isfinite(arrays["y_prob"]).all()
            or np.any(arrays["y_prob"] < 0)
            or np.any(arrays["y_prob"] > 1)
        ):
            failures.append(
                f"{split_name} probabilities are invalid"
            )
        if not np.isin(arrays["y_true"], (0, 1)).all():
            failures.append(f"{split_name} y_true is not binary")
        if not np.isin(arrays["y_pred"], (0, 1)).all():
            failures.append(f"{split_name} y_pred is not binary")
    if structurally_invalid:
        return failures
    if not np.array_equal(
        validation["window_index"],
        support["validation_anchor_window_index"],
    ):
        failures.append("validation prediction support differs")
    if not np.array_equal(validation["y_true"], support["validation_y"]):
        failures.append("validation prediction labels differ")
    if not np.array_equal(
        test["window_index"],
        support["test_anchor_window_index"],
    ):
        failures.append("test prediction support differs")
    if not np.array_equal(test["y_true"], support["test_y"]):
        failures.append("test prediction labels differ")
    threshold, validation_metrics = rf.choose_threshold(
        validation["y_true"],
        validation["y_prob"],
    )
    if not _close(metrics.get("threshold"), threshold):
        failures.append("threshold was not selected from validation")
    expected_validation_pred = (
        validation["y_prob"] >= threshold
    ).astype(np.int8)
    expected_test_pred = (test["y_prob"] >= threshold).astype(np.int8)
    if not np.array_equal(validation["y_pred"], expected_validation_pred):
        failures.append("validation y_pred differs from selected threshold")
    if not np.array_equal(test["y_pred"], expected_test_pred):
        failures.append("test y_pred differs from validation threshold")
    test_metrics = rf.binary_metrics(
        test["y_true"],
        test["y_prob"],
        threshold,
    )
    test_metrics.update(
        rf.event_metrics(
            dataset,
            windows,
            test["window_index"],
            expected_test_pred,
        )
    )
    rf.add_requested_metrics(test_metrics)
    for key in (
        "threshold",
        "n",
        "n_normal",
        "n_fog",
        "tn",
        "fp",
        "fn",
        "tp",
        *suite.CLASSIFICATION_METRICS,
    ):
        if not _close(metrics.get(key), test_metrics.get(key)):
            failures.append(f"test metric differs: {key}")
    saved_validation = metrics.get("validation", {})
    for key, value in validation_metrics.items():
        if not _close(saved_validation.get(key), value):
            failures.append(f"validation metric differs: {key}")
    if not _close(
        metrics.get("best_validation_auprc"),
        saved_validation.get("auprc"),
    ):
        failures.append("best validation AUPRC differs from final validation")
    history = metrics.get("history", [])
    classifier_seed = int(metrics.get("classifier_seed", -1))
    for row in history:
        if row.get("shuffle_seed") != classifier_seed + int(row["epoch"]):
            failures.append("epoch shuffle seed rule changed")
            break
    if history:
        selected = history[0]
        selected_score = float(selected["validation_auprc"])
        for row in history[1:]:
            score = float(row["validation_auprc"])
            if score > selected_score + 1e-5:
                selected = row
                selected_score = score
        if int(metrics.get("best_epoch", -1)) != int(selected["epoch"]):
            failures.append("best_epoch is not validation-selected")
    return failures


def _validate_fold(
    result_dir: Path,
    config: Mapping[str, Any],
    subject: str,
    dataset: Any,
    windows: Any,
    allow_partial: bool,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    failures: list[str] = []
    completed: dict[str, dict[str, Any]] = {}
    fold_root = result_dir / f"loso_{subject}"
    if not fold_root.is_dir():
        if not allow_partial:
            failures.append(f"{subject}: missing fold directory")
        return failures, completed
    try:
        fold_config = _load_json(fold_root / "fold_config.json")
        if fold_config.get("test_subject") != subject:
            failures.append(f"{subject}: fold_config subject mismatch")
        if fold_config.get("protocol_fingerprint") != config.get(
            "protocol_fingerprint"
        ):
            failures.append(f"{subject}: fold protocol mismatch")
    except Exception as exc:
        return [f"{subject}: invalid fold_config: {exc}"], completed

    source_fold = (
        Path(str(config["source_suite_dir"])) / f"loso_{subject}"
    )
    try:
        canonical_fold_config = _load_json(
            source_fold / "fold_config.json"
        )
        for key in ("test_subject", "val_subject", "train_subjects"):
            if fold_config.get(key) != canonical_fold_config.get(key):
                failures.append(
                    f"{subject}: fold partition differs from source: {key}"
                )
    except Exception as exc:
        failures.append(
            f"{subject}: canonical fold_config cannot be read: {exc}"
        )
    source_hashes = {
        "source_fold_config_sha256": source_fold / "fold_config.json",
        "source_scaler_sha256": source_fold / "scaler.json",
        "source_split_indices_sha256": source_fold / "split_indices.npz",
        "source_history_support_sha256": source_fold / "history_support.npz",
    }
    for key, path in source_hashes.items():
        try:
            if fold_config["source"].get(key) != sha256_file(path):
                failures.append(f"{subject}: source artifact drifted: {key}")
        except Exception as exc:
            failures.append(f"{subject}: cannot validate {key}: {exc}")

    cache_path = fold_root / "fusion_primitives.npz"
    cache_done_path = fold_root / "FUSION_PRIMITIVE_CACHE_DONE.json"
    source_model = config["source"]["folds"][subject]["models"][
        suite.SOURCE_NBM
    ]
    cache_upstream = canonical_fingerprint(
        {
            "source_nbm_best_sha256": source_model[
                "source_nbm_best_sha256"
            ],
            "source_residual_cache_sha256": source_model[
                "source_residual_cache_sha256"
            ],
            "source_residual_done_sha256": source_model[
                "source_residual_done_sha256"
            ],
            "source_scaler_sha256": config["source"]["folds"][subject][
                "source_scaler_sha256"
            ],
            "source_split_indices_sha256": config["source"]["folds"][
                subject
            ]["source_split_indices_sha256"],
            "source_history_support_sha256": config["source"]["folds"][
                subject
            ]["source_history_support_sha256"],
            "primitive_formula_version": "raw_error_sigma.v1",
        }
    )
    cache_done = validate_done(
        cache_done_path,
        stage="fusion_primitive_cache",
        protocol_fingerprint=str(config["protocol_fingerprint"]),
        task_id=f"{subject}/fusion_primitive_cache",
        upstream_sha256=cache_upstream,
    )
    if cache_done is None:
        if not allow_partial:
            failures.append(f"{subject}: primitive cache incomplete")
        return failures, completed
    try:
        cache_sha = sha256_file(cache_path)
        if cache_sha != fold_config.get("primitive_cache_sha256"):
            failures.append(f"{subject}: primitive cache hash mismatch")
        features = _load_primitives(cache_path)
    except Exception as exc:
        return [*failures, f"{subject}: invalid primitive cache: {exc}"], completed

    try:
        with np.load(
            fold_root / "input_support.npz",
            allow_pickle=False,
        ) as payload:
            support = {
                key: np.asarray(payload[key])
                for key in payload.files
            }
        if sha256_file(
            fold_root / "input_support.npz"
        ) != fold_config.get("input_support_sha256"):
            failures.append(f"{subject}: support hash mismatch")
    except Exception as exc:
        return [*failures, f"{subject}: invalid support: {exc}"], completed
    if bool(config.get("reportable")):
        try:
            with np.load(
                source_fold / "history_support.npz",
                allow_pickle=False,
            ) as source_support:
                for split in ("train", "validation", "test"):
                    for suffix in (
                        "anchor_window_index",
                        "history_window_index",
                    ):
                        key = f"{split}_{suffix}"
                        if not np.array_equal(
                            support[key],
                            source_support[key],
                        ):
                            failures.append(
                                f"{subject}: formal support differs from "
                                f"canonical source: {key}"
                            )
        except Exception as exc:
            failures.append(
                f"{subject}: canonical history support cannot be read: {exc}"
            )

    try:
        source_args = SimpleNamespace(
            source_suite_dir=Path(str(config["source_suite_dir"]))
        )
        canonical_features, _ = suite.nbm_rep._load_source_cache(
            source_args,
            config,
            subject,
            suite.SOURCE_NBM,
        )
    except Exception as exc:
        failures.append(
            f"{subject}: canonical Transformer cache cannot be loaded: {exc}"
        )
        canonical_features = {}
    try:
        scaler = suite.nbm_rep._load_scaler(
            source_fold / "scaler.json"
        )
    except Exception as exc:
        failures.append(f"{subject}: source scaler cannot be loaded: {exc}")
        scaler = None
    replay_model = None
    if scaler is not None:
        try:
            checkpoint = suite.torch.load(
                source_fold
                / suite.SOURCE_NBM
                / "nbm"
                / "best.pt",
                map_location="cpu",
                weights_only=False,
            )
            suite.validate_checkpoint(
                checkpoint,
                stage="nbm",
                protocol_fingerprint=config["source"][
                    "source_protocol_fingerprint"
                ],
                task_id=(
                    f"loso_{subject}/{suite.SOURCE_NBM}/nbm"
                ),
            )
            replay_model = suite.nbm_rep._build_source_model(
                suite.SOURCE_NBM,
                checkpoint,
                config["source_model_reconstruction"],
            )
            replay_model.eval()
        except Exception as exc:
            failures.append(
                f"{subject}: Transformer checkpoint replay setup failed: "
                f"{exc}"
            )

    for split in ("train", "validation", "test"):
        payload = features[split]
        shapes = {
            tuple(payload[key].shape)
            for key in ("raw", "error", "sigma")
        }
        if len(shapes) != 1 or next(iter(shapes))[1:] != (9, 32):
            failures.append(f"{subject}/{split}: primitive shapes differ")
            continue
        if not all(
            np.isfinite(payload[key]).all()
            for key in ("raw", "error", "sigma")
        ):
            failures.append(f"{subject}/{split}: non-finite primitives")
        if np.any(payload["sigma"] <= 0):
            failures.append(f"{subject}/{split}: non-positive sigma")
        if not np.array_equal(
            payload["y"],
            windows.label[payload["window_index"]],
        ):
            failures.append(f"{subject}/{split}: primitive labels differ")
        canonical = canonical_features.get(split)
        if canonical is not None:
            if not np.array_equal(
                payload["window_index"],
                canonical["window_index"],
            ):
                failures.append(
                    f"{subject}/{split}: source window order differs"
                )
            if not np.array_equal(payload["y"], canonical["y"]):
                failures.append(f"{subject}/{split}: source labels differ")
            replayed_z = np.clip(
                payload["error"] / payload["sigma"],
                -suite.Z_CLIP,
                suite.Z_CLIP,
            ).astype(np.float32)
            if not np.allclose(
                replayed_z,
                canonical["dynamic_standardized_error"],
                rtol=suite.SOURCE_REPLAY_TOLERANCE_AMP,
                atol=suite.SOURCE_REPLAY_TOLERANCE_AMP,
            ):
                failures.append(
                    f"{subject}/{split}: primitives do not reproduce "
                    "canonical Transformer z"
                )
        if scaler is not None:
            loader = suite.core.make_sequence_loader(
                dataset,
                windows,
                payload["window_index"],
                scaler,
                int(config["batch_size"]),
                False,
                0,
                False,
            )
            offset = 0
            for sequence, labels, indices in loader:
                count = int(len(indices))
                expected_raw = (
                    sequence[:, :, suite.CONTEXT_SAMPLES:]
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                if not np.array_equal(
                    indices.numpy(),
                    payload["window_index"][offset : offset + count],
                ):
                    failures.append(
                        f"{subject}/{split}: raw replay order differs"
                    )
                    break
                if not np.array_equal(
                    labels.numpy(),
                    payload["y"][offset : offset + count],
                ):
                    failures.append(
                        f"{subject}/{split}: raw replay labels differ"
                    )
                    break
                if not np.array_equal(
                    expected_raw,
                    payload["raw"][offset : offset + count],
                ):
                    failures.append(
                        f"{subject}/{split}: raw is not the scaled target"
                    )
                    break
                offset += count
            if offset != len(payload["window_index"]):
                failures.append(
                    f"{subject}/{split}: raw replay length differs"
                )
        if replay_model is not None and len(payload["window_index"]):
            sample_count = min(64, len(payload["window_index"]))
            sample_rows = np.linspace(
                0,
                len(payload["window_index"]) - 1,
                num=sample_count,
                dtype=np.int64,
            )
            sample_indices = payload["window_index"][sample_rows]
            loader = suite.core.make_sequence_loader(
                dataset,
                windows,
                sample_indices,
                scaler,
                sample_count,
                False,
                0,
                False,
            )
            sequence, labels, indices = next(iter(loader))
            with suite.torch.no_grad():
                mean, sigma = replay_model(
                    sequence[:, :, : suite.CONTEXT_SAMPLES]
                )
            expected_raw = sequence[
                :, :, suite.CONTEXT_SAMPLES :
            ].float()
            expected_error = expected_raw - mean.float()
            expected_sigma = sigma.float()
            if not np.array_equal(indices.numpy(), sample_indices):
                failures.append(
                    f"{subject}/{split}: sampled Transformer replay "
                    "order differs"
                )
            if not np.array_equal(
                labels.numpy(),
                payload["y"][sample_rows],
            ):
                failures.append(
                    f"{subject}/{split}: sampled Transformer replay "
                    "labels differ"
                )
            for name, expected in (
                ("raw", expected_raw),
                ("error", expected_error),
                ("sigma", expected_sigma),
            ):
                observed = payload[name][sample_rows]
                expected_array = expected.numpy().astype(
                    np.float32,
                    copy=False,
                )
                tolerance = 0.0 if name == "raw" else 3e-2
                if not np.allclose(
                    observed,
                    expected_array,
                    rtol=tolerance,
                    atol=tolerance,
                ):
                    failures.append(
                        f"{subject}/{split}: sampled Transformer {name} "
                        "replay differs"
                    )
        try:
            representations = build_fusion_representations(
                payload["raw"],
                payload["error"],
                payload["sigma"],
            )
            for fusion_id, values in representations.items():
                expected_channels = suite.FUSION_REPRESENTATIONS[
                    fusion_id
                ]["in_channels"]
                if values.shape[1:] != (expected_channels, 32):
                    failures.append(
                        f"{subject}/{split}/{fusion_id}: wrong shape"
                    )
            if np.count_nonzero(representations["F3"][:, 9:]) != 0:
                failures.append(f"{subject}/{split}: F3 zero map changed")
            if not np.array_equal(
                representations["F3"][:, :9],
                representations["F0"],
            ):
                failures.append(f"{subject}/{split}: F3 raw slot changed")
        except Exception as exc:
            failures.append(
                f"{subject}/{split}: fusion formula failure: {exc}"
            )
        plan = make_common_history_plan(
            windows,
            payload["window_index"],
            suite.HORIZON_SAMPLES,
            suite.STRIDE_SAMPLES,
            suite.HISTORY_SAMPLES,
        )
        cap_this_split = (
            int(config.get("max_classifier_windows", 0)) > 0
            and (split == "train" or bool(config.get("smoke")))
        )
        if cap_this_split:
            rows = np.arange(len(plan.anchor_rows), dtype=np.int64)
            selected = rf.deterministic_subsample(
                rows,
                int(config["max_classifier_windows"]),
                int(config["seed"])
                + 100
                + suite.EXPECTED_LOSO_SUBJECTS.index(subject)
                + {"train": 0, "validation": 1, "test": 2}[split],
                windows.label[plan.anchor_window_indices],
            )
            plan = plan.take(selected)
        if not np.array_equal(
            plan.anchor_window_indices,
            support[f"{split}_anchor_window_index"],
        ):
            failures.append(f"{subject}/{split}: anchor support differs")
        chain = payload["window_index"][plan.max_chain_rows]
        if not np.array_equal(
            chain,
            support[f"{split}_history_window_index"],
        ):
            failures.append(f"{subject}/{split}: history support differs")
        if not np.array_equal(
            payload["y"][plan.anchor_rows],
            support[f"{split}_y"],
        ):
            failures.append(f"{subject}/{split}: support labels differ")
        if len(plan.max_chain_rows):
            starts = windows.target_start[chain]
            if not np.all(np.diff(starts, axis=1) == 32):
                failures.append(
                    f"{subject}/{split}: history blocks are not 0.5s apart"
                )

    del replay_model

    states, counts, hashes, backbone_hash = suite._aligned_reference_states(
        int(fold_config["classifier_seed"]),
        int(config["classifier"]["hidden_channels"]),
        float(config["classifier"]["dropout"]),
        bool(config["deterministic"]),
    )
    del states
    expected_hashes = {str(key): value for key, value in hashes.items()}
    expected_counts = {str(key): value for key, value in counts.items()}
    if fold_config.get(
        "reference_initial_state_sha256_by_in_channels"
    ) != expected_hashes:
        failures.append(f"{subject}: initial state hashes differ")
    if fold_config.get("parameter_count_by_in_channels") != expected_counts:
        failures.append(f"{subject}: parameter counts differ")
    if fold_config.get(
        "shared_backbone_initial_state_sha256"
    ) != backbone_hash:
        failures.append(f"{subject}: shared backbone hash differs")

    expected_fingerprints = {
        fusion_id: canonical_fingerprint(
            {
                "primitive_cache_sha256": cache_sha,
                "input_support_sha256": fold_config[
                    "input_support_sha256"
                ],
                "fusion": definition,
                "history_samples": suite.HISTORY_SAMPLES,
                "history_blocks": suite.HISTORY_BLOCKS,
                "z_clip": suite.Z_CLIP,
            }
        )
        for fusion_id, definition in suite.FUSION_REPRESENTATIONS.items()
    }
    if fold_config.get("fusion_input_fingerprints") != expected_fingerprints:
        failures.append(f"{subject}: input fingerprints differ")
    try:
        if _load_json(
            fold_root / "source_provenance.json"
        ) != fold_config["source"]:
            failures.append(f"{subject}: source provenance sidecar differs")
        if _load_json(
            fold_root / "fusion_input_fingerprints.json"
        ) != {
            "protocol_fingerprint": config["protocol_fingerprint"],
            "fusion_inputs": expected_fingerprints,
        }:
            failures.append(
                f"{subject}: fusion input fingerprint sidecar differs"
            )
    except Exception as exc:
        failures.append(f"{subject}: invalid provenance sidecar: {exc}")

    for cell in config["fusion_representations"]:
        fusion_id = str(cell["fusion_id"])
        try:
            result = suite._load_completed_cell(
                result_dir,
                config,
                cell,
                subject,
            )
        except Exception as exc:
            failures.append(f"{subject}/{fusion_id}: {exc}")
            continue
        if result is None:
            if not allow_partial:
                failures.append(f"{subject}/{fusion_id}: incomplete")
            continue
        metrics, _ = result
        expected_identity = {
            "classifier_seed": fold_config["classifier_seed"],
            "val_subject": fold_config["val_subject"],
            "history_seconds": suite.HISTORY_SECONDS,
            "history_samples": suite.HISTORY_SAMPLES,
            "history_blocks": suite.HISTORY_BLOCKS,
        }
        for key, value in expected_identity.items():
            if metrics.get(key) != value:
                failures.append(
                    f"{subject}/{fusion_id}: classifier fairness "
                    f"identity differs: {key}"
                )
        train_counts = np.bincount(
            np.asarray(support["train_y"], dtype=np.int8),
            minlength=2,
        ).astype(int)
        expected_pos_weight = min(
            math.sqrt(float(train_counts[0]) / float(train_counts[1])),
            6.0,
        )
        if metrics.get("train_counts") != train_counts.tolist():
            failures.append(
                f"{subject}/{fusion_id}: train class counts differ"
            )
        if not _close(metrics.get("pos_weight"), expected_pos_weight):
            failures.append(
                f"{subject}/{fusion_id}: class weight differs"
            )
        cell_root = suite.task_root_for(
            result_dir,
            subject,
            fusion_id,
        )
        try:
            best_checkpoint = suite.torch.load(
                cell_root / "classifier_best.pt",
                map_location="cpu",
                weights_only=False,
            )
            rf.validate_rf_checkpoint(
                best_checkpoint,
                protocol_fingerprint=str(config["protocol_fingerprint"]),
                task_id=f"{subject}/{fusion_id}",
                source_residual_sha256=fold_config[
                    "fusion_input_fingerprints"
                ][fusion_id],
            )
            if best_checkpoint.get("classifier_config") != metrics.get(
                "classifier_config"
            ):
                failures.append(
                    f"{subject}/{fusion_id}: best checkpoint classifier "
                    "configuration differs"
                )
            if best_checkpoint.get("variant") != fusion_id:
                failures.append(
                    f"{subject}/{fusion_id}: best checkpoint variant differs"
                )
            if int(best_checkpoint.get("best_epoch", -1)) != int(
                metrics.get("best_epoch", -2)
            ):
                failures.append(
                    f"{subject}/{fusion_id}: best checkpoint epoch differs"
                )
            if not _close(
                best_checkpoint.get("best_validation_auprc"),
                metrics.get("best_validation_auprc"),
            ):
                failures.append(
                    f"{subject}/{fusion_id}: best checkpoint validation "
                    "AUPRC differs"
                )
        except Exception as exc:
            failures.append(
                f"{subject}/{fusion_id}: invalid best checkpoint: {exc}"
            )
        prediction_failures = _validate_predictions(
            root=cell_root,
            metrics=metrics,
            support=support,
            dataset=dataset,
            windows=windows,
        )
        failures.extend(
            f"{subject}/{fusion_id}: {message}"
            for message in prediction_failures
        )
        completed[fusion_id] = metrics
    return failures, completed


def _validate_aggregates(
    result_dir: Path,
    config: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    failures: list[str],
    allow_partial: bool,
) -> None:
    aggregate_path = result_dir / "aggregate_metrics.json"
    paired_path = result_dir / "paired_pr_auc_deltas.csv"
    status_path = result_dir / "status.json"
    fold_summary_path = result_dir / "fold_summary.csv"
    aggregate_summary_path = result_dir / "aggregate_summary.csv"
    experiment_manifest_path = result_dir / "experiment_manifest.csv"
    publication_path = result_dir / "publication_table.csv"
    required_paths = (
        aggregate_path,
        paired_path,
        status_path,
        fold_summary_path,
        aggregate_summary_path,
        experiment_manifest_path,
        publication_path,
    )
    if not all(path.is_file() for path in required_paths):
        _issue(failures, "root summary files are missing")
        return
    aggregate = _load_json(aggregate_path)
    expected_experiment_ids = {
        str(cell["experiment_id"])
        for cell in config["fusion_representations"]
    }
    if set(aggregate.get("experiments", {})) != expected_experiment_ids:
        _issue(failures, "aggregate_metrics experiment registry differs")
    if aggregate.get("primary_comparison") != config.get(
        "primary_comparison"
    ):
        _issue(failures, "aggregate primary comparison differs")
    expected_macro: dict[str, dict[str, Any]] = {}
    for cell in config["fusion_representations"]:
        fusion_id = str(cell["fusion_id"])
        group = [
            rows[subject][fusion_id]
            for subject in suite.EXPECTED_LOSO_SUBJECTS
            if fusion_id in rows.get(subject, {})
        ]
        expected = (
            aggregate_fold_metrics(
                group,
                list(suite.CLASSIFICATION_METRICS),
            )
            if group
            else {
                metric: {"mean": None, "std": None, "n_folds": 0}
                for metric in suite.CLASSIFICATION_METRICS
            }
        )
        expected_macro[fusion_id] = expected
        observed_experiment = aggregate.get("experiments", {}).get(
            cell["experiment_id"],
            {},
        )
        expected_completed_subjects = [
            subject
            for subject in suite.EXPECTED_LOSO_SUBJECTS
            if fusion_id in rows.get(subject, {})
        ]
        if observed_experiment.get(
            "completed_folds"
        ) != expected_completed_subjects:
            _issue(
                failures,
                f"aggregate completed folds differ: {fusion_id}",
            )
        observed = observed_experiment.get("subject_macro", {})
        for metric in suite.CLASSIFICATION_METRICS:
            for field in ("mean", "std", "n_folds"):
                if not _close(
                    observed.get(metric, {}).get(field),
                    expected.get(metric, {}).get(field),
                ):
                    _issue(
                        failures,
                        f"aggregate differs: {fusion_id}/{metric}/{field}",
                    )
        if expected_completed_subjects:
            pooled_arrays: dict[str, list[np.ndarray]] = {
                "y_true": [],
                "y_prob": [],
                "y_pred": [],
            }
            for subject in expected_completed_subjects:
                with np.load(
                    suite.task_root_for(
                        result_dir,
                        subject,
                        fusion_id,
                    )
                    / "predictions.npz",
                    allow_pickle=False,
                ) as payload:
                    for key in pooled_arrays:
                        pooled_arrays[key].append(np.asarray(payload[key]))
            expected_pooled = rf.prediction_metrics(
                np.concatenate(pooled_arrays["y_true"]),
                np.concatenate(pooled_arrays["y_prob"]),
                np.concatenate(pooled_arrays["y_pred"]),
            )
        else:
            expected_pooled = None
        observed_pooled = observed_experiment.get("pooled")
        if expected_pooled is None:
            if observed_pooled is not None:
                _issue(
                    failures,
                    f"aggregate pooled metrics differ: {fusion_id}",
                )
        else:
            if not isinstance(observed_pooled, Mapping):
                _issue(
                    failures,
                    f"aggregate pooled metrics missing: {fusion_id}",
                )
            else:
                for key, value in expected_pooled.items():
                    if not _close(observed_pooled.get(key), value):
                        _issue(
                            failures,
                            f"aggregate pooled differs: {fusion_id}/{key}",
                        )

    expected_fold_rows = {
        (subject, fusion_id): metrics
        for subject, fold in rows.items()
        for fusion_id, metrics in fold.items()
    }
    observed_fold_rows = _read_csv(fold_summary_path)
    observed_fold_map = {
        (row.get("test_subject", ""), row.get("fusion_id", "")): row
        for row in observed_fold_rows
    }
    if len(observed_fold_map) != len(observed_fold_rows):
        _issue(failures, "fold_summary contains duplicate cells")
    if set(observed_fold_map) != set(expected_fold_rows):
        _issue(failures, "fold_summary cell registry differs")
    for identity, expected in expected_fold_rows.items():
        observed = observed_fold_map.get(identity, {})
        fusion_id = identity[1]
        cell = next(
            item
            for item in config["fusion_representations"]
            if item["fusion_id"] == fusion_id
        )
        expected_identity = {
            "experiment_id": str(cell["experiment_id"]),
            "variant": fusion_id,
            "fusion_id": fusion_id,
            "display_name": str(cell["display_name"]),
            "formula": str(cell["formula"]),
            "in_channels": str(cell["in_channels"]),
            "parameter_count": str(cell["parameter_count"]),
            "test_subject": identity[0],
            "classifier_seed": str(expected["classifier_seed"]),
            "source_residual_sha256": str(
                expected["source_residual_sha256"]
            ),
            "input_support_sha256": str(
                expected["input_support_sha256"]
            ),
            "initial_state_sha256": str(
                expected["initial_state_sha256"]
            ),
        }
        for key, value in expected_identity.items():
            if observed.get(key) != value:
                _issue(
                    failures,
                    f"fold_summary identity differs: {identity}/{key}",
                )
        for key in (
            "threshold",
            "n",
            "n_normal",
            "n_fog",
            "tn",
            "fp",
            "fn",
            "tp",
            *suite.CLASSIFICATION_METRICS,
        ):
            if not _close(observed.get(key), expected.get(key)):
                _issue(
                    failures,
                    f"fold_summary differs: {identity}/{key}",
                )

    expected_rank = sorted(
        suite.FUSION_IDS,
        key=lambda fusion_id: (
            -float(expected_macro[fusion_id]["pr_auc"]["mean"])
            if expected_macro[fusion_id]["pr_auc"]["mean"] is not None
            else float("inf"),
            fusion_id,
        ),
    )
    aggregate_rows = _read_csv(aggregate_summary_path)
    aggregate_map = {
        row.get("fusion_id", ""): row for row in aggregate_rows
    }
    if len(aggregate_map) != len(aggregate_rows):
        _issue(failures, "aggregate_summary contains duplicate rows")
    if set(aggregate_map) != set(suite.FUSION_IDS):
        _issue(failures, "aggregate_summary registry differs")
    for rank, fusion_id in enumerate(expected_rank, start=1):
        observed = aggregate_map.get(fusion_id, {})
        if not _close(observed.get("rank"), rank):
            _issue(
                failures,
                f"aggregate_summary rank differs: {fusion_id}",
            )
        completed_folds = sum(
            fusion_id in fold for fold in rows.values()
        )
        if not _close(
            observed.get("completed_folds"),
            completed_folds,
        ):
            _issue(
                failures,
                f"aggregate_summary fold count differs: {fusion_id}",
            )
        for metric in suite.CLASSIFICATION_METRICS:
            for suffix in ("mean", "std"):
                if not _close(
                    observed.get(f"{metric}_{suffix}"),
                    expected_macro[fusion_id][metric][suffix],
                ):
                    _issue(
                        failures,
                        f"aggregate_summary differs: "
                        f"{fusion_id}/{metric}_{suffix}",
                    )

    manifest_rows = _read_csv(experiment_manifest_path)
    manifest_map = {
        row.get("fusion_id", ""): row for row in manifest_rows
    }
    if len(manifest_map) != len(manifest_rows):
        _issue(failures, "experiment_manifest contains duplicate rows")
    if set(manifest_map) != set(suite.FUSION_IDS):
        _issue(failures, "experiment_manifest registry differs")
    cells_by_id = {
        str(cell["fusion_id"]): cell
        for cell in config["fusion_representations"]
    }
    for fusion_id in suite.FUSION_IDS:
        observed = manifest_map.get(fusion_id, {})
        cell = cells_by_id[fusion_id]
        completed_subjects = [
            subject
            for subject in suite.EXPECTED_LOSO_SUBJECTS
            if fusion_id in rows.get(subject, {})
        ]
        expected_status = (
            "complete"
            if completed_subjects == list(suite.EXPECTED_LOSO_SUBJECTS)
            else ("partial" if completed_subjects else "pending")
        )
        expected_manifest = {
            "experiment_id": str(cell["experiment_id"]),
            "display_name": str(cell["display_name"]),
            "formula": str(cell["formula"]),
            "in_channels": str(cell["in_channels"]),
            "parameter_count": str(cell["parameter_count"]),
            "expected_folds": str(len(suite.EXPECTED_LOSO_SUBJECTS)),
            "completed_folds": str(len(completed_subjects)),
            "status": expected_status,
            "completed_subjects": ",".join(completed_subjects),
        }
        for key, value in expected_manifest.items():
            if observed.get(key) != value:
                _issue(
                    failures,
                    f"experiment_manifest differs: {fusion_id}/{key}",
                )
    publication_rows = _read_csv(publication_path)
    expected_display = {
        cell["display_name"] for cell in config["fusion_representations"]
    }
    if {
        row.get("Fusion input", "") for row in publication_rows
    } != expected_display:
        _issue(failures, "publication_table registry differs")
    publication_map = {
        row.get("Fusion input", ""): row
        for row in publication_rows
    }
    if len(publication_map) != len(publication_rows):
        _issue(failures, "publication_table contains duplicate rows")
    publication_metrics = {
        "PR-AUC": "pr_auc",
        "BA": "balanced_accuracy",
        "Macro-F1": "macro_f1",
        "AUROC": "roc_auc",
        "FoG Recall": "fog_recall",
        "Specificity": "specificity",
        "FoG Precision": "precision",
        "FoG F1": "fog_f1",
        "Event Sensitivity": "event_sensitivity",
        "FA/h": "false_alarm_events_per_hour",
        "Delay (s)": "median_detection_delay_sec",
    }
    for fusion_id, cell in cells_by_id.items():
        observed = publication_map.get(cell["display_name"], {})
        if observed.get("Channels") != str(cell["in_channels"]):
            _issue(
                failures,
                f"publication_table channels differ: {fusion_id}",
            )
        completed_folds = sum(
            fusion_id in fold for fold in rows.values()
        )
        if observed.get("Completed folds") != str(completed_folds):
            _issue(
                failures,
                f"publication_table fold count differs: {fusion_id}",
            )
        for column, metric in publication_metrics.items():
            expected = suite._format_mean_sd(
                expected_macro[fusion_id],
                metric,
            )
            if observed.get(column) != expected:
                _issue(
                    failures,
                    f"publication_table differs: {fusion_id}/{column}",
                )
    expected_comparisons: dict[str, dict[str, Any]] = {}
    for comparison in suite.COMPARISONS:
        subjects: list[str] = []
        differences: list[float] = []
        for subject in suite.EXPECTED_LOSO_SUBJECTS:
            fold = rows.get(subject, {})
            if comparison["new"] not in fold or comparison[
                "reference"
            ] not in fold:
                continue
            subjects.append(subject)
            differences.append(
                float(fold[comparison["new"]]["pr_auc"])
                - float(fold[comparison["reference"]]["pr_auc"])
            )
        effect = suite.input_ablation.paired_bootstrap_mean_ci(
            np.asarray(differences, dtype=np.float64),
            int(config["bootstrap_samples"]),
            suite.input_ablation.stable_bootstrap_seed(
                int(config["bootstrap_seed"]),
                comparison["comparison_id"],
            ),
        )
        expected_comparisons[comparison["comparison_id"]] = {
            **comparison,
            "common_subjects": ",".join(subjects),
            **effect,
            "bootstrap_seed": int(config["bootstrap_seed"]),
        }
    observed_rows = {
        row["comparison_id"]: row
        for row in _read_csv(paired_path)
    }
    if set(observed_rows) != set(expected_comparisons):
        _issue(failures, "paired comparison CSV registry differs")
    for comparison_id, expected in expected_comparisons.items():
        observed = observed_rows.get(comparison_id, {})
        for key, value in expected.items():
            if key in {
                "mean_delta",
                "ci_low",
                "ci_high",
                "n_paired_subjects",
                "wins",
                "ties",
                "losses",
                "bootstrap_samples",
                "bootstrap_seed",
            }:
                if not _close(observed.get(key), value):
                    _issue(
                        failures,
                        f"paired comparison differs: {comparison_id}/{key}",
                    )
            elif observed.get(key) != str(value):
                _issue(
                    failures,
                    f"paired comparison identity differs: "
                    f"{comparison_id}/{key}",
                )
    aggregate_comparisons = {
        row.get("comparison_id"): row
        for row in aggregate.get("paired_pr_auc_comparisons", [])
    }
    if set(aggregate_comparisons) != set(expected_comparisons):
        _issue(failures, "aggregate paired comparison registry differs")
    for comparison_id, expected in expected_comparisons.items():
        observed = aggregate_comparisons.get(comparison_id, {})
        for key, value in expected.items():
            if isinstance(value, (int, float)) or value is None:
                if not _close(observed.get(key), value):
                    _issue(
                        failures,
                        f"aggregate paired comparison differs: "
                        f"{comparison_id}/{key}",
                    )
            elif observed.get(key) != value:
                _issue(
                    failures,
                    f"aggregate paired comparison identity differs: "
                    f"{comparison_id}/{key}",
                )
    status = _load_json(status_path)
    completed_cells = sum(len(value) for value in rows.values())
    if allow_partial and completed_cells == 0:
        _issue(
            failures,
            "partial audit requires at least one completed classifier cell",
        )
    if status.get("completed_classifier_cells") != completed_cells:
        _issue(failures, "status classifier count differs")
    completed_caches = sum(
        validate_done(
            result_dir
            / f"loso_{subject}"
            / "FUSION_PRIMITIVE_CACHE_DONE.json",
            stage="fusion_primitive_cache",
            protocol_fingerprint=str(config["protocol_fingerprint"]),
            task_id=f"{subject}/fusion_primitive_cache",
        )
        is not None
        for subject in suite.EXPECTED_LOSO_SUBJECTS
    )
    expected_status_fields = {
        "suite_version": suite.SUITE_VERSION,
        "protocol_fingerprint": config["protocol_fingerprint"],
        "expected_primitive_cache_tasks": len(
            suite.EXPECTED_LOSO_SUBJECTS
        ),
        "completed_primitive_cache_tasks": completed_caches,
        "expected_classifier_cells": int(
            config["expected_classifier_cells"]
        ),
        "reportable": bool(config["reportable"]),
    }
    for key, value in expected_status_fields.items():
        if status.get(key) != value:
            _issue(failures, f"status field differs: {key}")
    formal_complete = (
        completed_cells == int(config["expected_classifier_cells"])
        and bool(config["reportable"])
    )
    integrity_complete = (
        completed_cells == int(config["expected_classifier_cells"])
        and completed_caches == len(suite.EXPECTED_LOSO_SUBJECTS)
    )
    expected_run_status = (
        "complete"
        if formal_complete
        else ("smoke_complete" if integrity_complete else "partial")
    )
    if status.get("status") != expected_run_status:
        _issue(failures, "status lifecycle value differs")
    if not allow_partial and not formal_complete:
        _issue(failures, "formal status is not complete")
    expected_best = (
        next(
            cell["experiment_id"]
            for cell in config["fusion_representations"]
            if cell["fusion_id"] == expected_rank[0]
        )
        if formal_complete
        else None
    )
    if aggregate.get("best_experiment") != expected_best:
        _issue(failures, "aggregate best_experiment differs")
    if status.get("best_experiment") != expected_best:
        _issue(failures, "status best_experiment differs")


def audit(
    result_dir: Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    result_dir = result_dir.resolve()
    failures: list[str] = []
    config_path = result_dir / "config.json"
    if not config_path.is_file():
        return {
            "suite_version": suite.SUITE_VERSION,
            "status": "fail",
            "failures": ["missing config.json"],
            "allow_partial": allow_partial,
            "expected_classifier_cells": 48,
            "completed_classifier_cells": 0,
            "audited_subjects": [],
        }
    config = _load_json(config_path)
    _validate_protocol(result_dir, config, failures)
    dataset, windows = _source_and_dataset(config, failures)
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    if dataset is not None and windows is not None:
        for subject in suite.EXPECTED_LOSO_SUBJECTS:
            fold_failures, completed = _validate_fold(
                result_dir,
                config,
                subject,
                dataset,
                windows,
                allow_partial,
            )
            failures.extend(fold_failures)
            rows[subject] = completed
        _validate_aggregates(
            result_dir,
            config,
            rows,
            failures,
            allow_partial,
        )
    completed_cells = sum(len(value) for value in rows.values())
    report = {
        "suite_version": suite.SUITE_VERSION,
        "protocol_fingerprint": config.get("protocol_fingerprint"),
        "status": "pass" if not failures else "fail",
        "allow_partial": bool(allow_partial),
        "expected_classifier_cells": int(
            config.get("expected_classifier_cells", 48)
        ),
        "completed_classifier_cells": completed_cells,
        "audited_subjects": list(suite.EXPECTED_LOSO_SUBJECTS),
        "failures": failures,
    }
    return report


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    try:
        report = audit(result_dir, allow_partial=args.allow_partial)
    except Exception as exc:  # pragma: no cover - last-resort CLI reporting
        report = {
            "suite_version": suite.SUITE_VERSION,
            "status": "fail",
            "allow_partial": bool(args.allow_partial),
            "expected_classifier_cells": 48,
            "completed_classifier_cells": 0,
            "audited_subjects": [],
            "failures": [
                f"unhandled audit exception: {type(exc).__name__}: {exc}"
            ],
        }
    result_dir.mkdir(parents=True, exist_ok=True)
    report_path = result_dir / "audit_report.json"
    atomic_json_dump(report, report_path)
    text_path = result_dir / "audit_report.txt"
    lines = [
        f"status: {report['status']}",
        (
            "classifier cells: "
            f"{report.get('completed_classifier_cells', 0)}/"
            f"{report.get('expected_classifier_cells', 48)}"
        ),
    ]
    lines.extend(f"- {item}" for item in report["failures"])
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
