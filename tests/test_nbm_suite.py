from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from cnbr_fog.data import DaphnetDataset
from cnbr_fog.nbm import NBM_NAMES, build_nbm, gaussian_nll_sigma
from cnbr_fog.resume import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_json_dump,
    atomic_torch_save,
    canonical_fingerprint,
    done_payload,
    validate_checkpoint,
    validate_done,
)


def expect_raises(
    exception_type: type[BaseException],
    function,
    *args,
    message_contains: str | None = None,
    **kwargs,
) -> BaseException:
    """Small pytest-independent exception assertion used by direct smoke runs."""

    try:
        function(*args, **kwargs)
    except exception_type as error:
        if message_contains is not None:
            assert message_contains in str(error)
        return error
    except BaseException as error:
        raise AssertionError(
            f"Expected {exception_type.__name__}, got {type(error).__name__}: {error}"
        ) from error
    raise AssertionError(f"Expected {exception_type.__name__} to be raised")


def test_all_nbm_models_share_mu_sigma_contract():
    context = torch.randn(3, 9, 128)
    target = torch.randn(3, 9, 32)
    for name in NBM_NAMES:
        model = build_nbm(
            name,
            in_channels=9,
            horizon=32,
            hidden_channels=16,
            linear_ar_order=16,
            transformer_heads=4,
            transformer_layers=1,
            transformer_ffn=32,
            max_context_samples=128,
            dropout=0.0,
        )
        mean, sigma = model(context)
        assert mean.shape == target.shape
        assert sigma.shape == target.shape
        assert torch.isfinite(mean).all()
        assert torch.isfinite(sigma).all()
        assert (sigma > 0).all()
        loss = gaussian_nll_sigma(target, mean, sigma)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )


def test_nbm_factory_rejects_unknown_model_name():
    expect_raises(
        ValueError,
        build_nbm,
        "not-a-real-nbm",
        in_channels=9,
        horizon=32,
        message_contains="Unknown NBM",
    )


def test_persistence_mean_exactly_repeats_latest_observation():
    context = torch.arange(2 * 9 * 7, dtype=torch.float32).reshape(2, 9, 7)
    model = build_nbm("persistence", in_channels=9, horizon=5)
    mean, sigma = model(context)
    expected = context[:, :, -1:].expand(-1, -1, 5)
    assert torch.equal(mean, expected)
    assert sigma.shape == expected.shape
    assert torch.isfinite(sigma).all()
    assert torch.all(sigma > 0)


def test_linear_ar_rejects_context_shorter_than_configured_order():
    model = build_nbm(
        "linear_ar",
        in_channels=9,
        horizon=4,
        linear_ar_order=8,
    )
    context = torch.zeros(2, 9, 7)
    expect_raises(
        ValueError,
        model,
        context,
        message_contains="needs 8 samples, got 7",
    )


def test_transformer_rejects_incompatible_attention_heads():
    expect_raises(
        ValueError,
        build_nbm,
        "transformer",
        in_channels=9,
        horizon=4,
        hidden_channels=10,
        transformer_heads=4,
        message_contains="d_model must be divisible by nhead",
    )


def test_nbm_checkpoint_round_trip_preserves_eval_outputs():
    torch.manual_seed(20260723)
    model_kwargs = {
        "in_channels": 3,
        "horizon": 4,
        "hidden_channels": 8,
        "gru_layers": 1,
        "dropout": 0.0,
    }
    model = build_nbm("gru", **model_kwargs).eval()
    context = torch.randn(2, 3, 11)
    with torch.no_grad():
        expected_mean, expected_sigma = model(context)

    protocol_fingerprint = canonical_fingerprint(
        {"model": "gru", "in_channels": 3, "horizon": 4}
    )
    task_id = "loso_S01/gru/nbm"
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": "nbm",
        "protocol_fingerprint": protocol_fingerprint,
        "task_id": task_id,
        "model_state": model.state_dict(),
    }

    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "best.pt"
        atomic_torch_save(payload, checkpoint_path)
        restored_payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    validate_checkpoint(
        restored_payload,
        stage="nbm",
        protocol_fingerprint=protocol_fingerprint,
        task_id=task_id,
    )
    restored = build_nbm("gru", **model_kwargs).eval()
    restored.load_state_dict(restored_payload["model_state"])
    with torch.no_grad():
        actual_mean, actual_sigma = restored(context)
    assert torch.equal(actual_mean, expected_mean)
    assert torch.equal(actual_sigma, expected_sigma)


def test_checkpoint_validation_rejects_protocol_fingerprint_mismatch():
    expected_fingerprint = canonical_fingerprint(
        {"seed": 42, "channels": ["ankle", "thigh", "trunk"]}
    )
    # Canonical JSON ordering must not make equivalent protocols incompatible.
    assert expected_fingerprint == canonical_fingerprint(
        {"channels": ["ankle", "thigh", "trunk"], "seed": 42}
    )
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": "nbm",
        "protocol_fingerprint": expected_fingerprint,
        "task_id": "loso_S01/gru/nbm",
    }
    validate_checkpoint(
        payload,
        stage="nbm",
        protocol_fingerprint=expected_fingerprint,
        task_id="loso_S01/gru/nbm",
    )
    changed_fingerprint = canonical_fingerprint(
        {"seed": 43, "channels": ["ankle", "thigh", "trunk"]}
    )
    expect_raises(
        ValueError,
        validate_checkpoint,
        payload,
        stage="nbm",
        protocol_fingerprint=changed_fingerprint,
        task_id="loso_S01/gru/nbm",
        message_contains="protocol_fingerprint",
    )


def test_done_validation_rejects_protocol_mismatch_and_corrupt_artifact():
    protocol_fingerprint = canonical_fingerprint({"seed": 42, "history": 4.0})
    changed_fingerprint = canonical_fingerprint({"seed": 43, "history": 4.0})
    task_id = "loso_S01/gru/residual_h4s"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact_path = root / "metrics.json"
        artifact_path.write_bytes(b"abc")
        done_path = root / "DONE.json"
        atomic_json_dump(
            done_payload(
                stage="classifier",
                protocol_fingerprint=protocol_fingerprint,
                task_id=task_id,
                artifacts={"metrics": artifact_path},
                upstream_sha256="nbm-sha256",
                relative_to=root,
            ),
            done_path,
        )
        saved_done = json.loads(done_path.read_text(encoding="utf-8"))
        assert saved_done["artifacts"]["metrics"]["path"] == "metrics.json"

        validated = validate_done(
            done_path,
            stage="classifier",
            protocol_fingerprint=protocol_fingerprint,
            task_id=task_id,
            upstream_sha256="nbm-sha256",
        )
        assert validated is not None
        expect_raises(
            ValueError,
            validate_done,
            done_path,
            stage="classifier",
            protocol_fingerprint=changed_fingerprint,
            task_id=task_id,
            upstream_sha256="nbm-sha256",
            message_contains="protocol mismatch",
        )

        # Preserve byte length so validation must check the hash, not just size.
        artifact_path.write_bytes(b"abd")
        expect_raises(
            ValueError,
            validate_done,
            done_path,
            stage="classifier",
            protocol_fingerprint=protocol_fingerprint,
            task_id=task_id,
            upstream_sha256="nbm-sha256",
            message_contains="hash mismatch",
        )


def test_daphnet_loader_accepts_complete_three_imu_schema():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        records_dir = root / "records"
        records_dir.mkdir()
        channel_names = [
            f"{sensor}_acc_{axis}"
            for sensor in ("ankle", "thigh", "trunk")
            for axis in ("forward", "vertical", "lateral")
        ]
        x = np.random.default_rng(4).normal(size=(320, 9)).astype(np.float32)
        y = np.zeros(320, dtype=np.int8)
        y[200:240] = 1
        np.savez_compressed(records_dir / "S01_seg000.npz", x=x, y_binary=y)
        with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "record_path",
                    "record_id",
                    "subject_id",
                    "run_id",
                    "n_samples",
                    "sampling_rate_hz",
                    "usable",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "record_path": "records/S01_seg000.npz",
                    "record_id": "S01_seg000",
                    "subject_id": "S01",
                    "run_id": "R01",
                    "n_samples": len(x),
                    "sampling_rate_hz": 64,
                    "usable": "True",
                }
            )
        (root / "schema.json").write_text(
            json.dumps(
                {"channels": [{"name": name} for name in channel_names]}
            ),
            encoding="utf-8",
        )
        dataset = DaphnetDataset.load(root)
        assert dataset.n_channels == 9
        assert dataset.channel_names == tuple(channel_names)
        assert dataset.records[0].x.shape == (320, 9)
        scaler = dataset.fit_scaler(["S01"])
        assert scaler.center.shape == (9,)
        assert scaler.scale.shape == (9,)
