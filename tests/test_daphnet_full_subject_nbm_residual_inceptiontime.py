from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_daphnet_full_subject_nbm_residual_inceptiontime as experiment


def test_inception_module_preserves_time_and_has_four_branches() -> None:
    module = experiment.InceptionModule(9)
    output = module(torch.randn(4, 9, 128))
    assert output.shape == (4, 128, 128)
    assert len(module.convolution_branches) == 3
    assert module.kernel_sizes == (9, 19, 39)


def test_classifier_accepts_all_four_method_channel_counts() -> None:
    for channels in (9, 27, 36):
        model = experiment.InceptionTimeClassifier(channels)
        output = model(torch.randn(3, channels, 128))
        assert output.shape == (3,)
        assert torch.isfinite(output).all()


def test_classifier_has_six_modules_and_two_residual_connections() -> None:
    model = experiment.InceptionTimeClassifier(9)
    config = model.architecture_config()
    assert len(model.modules_inception) == 6
    assert config["residual_after_modules"] == [3, 6]
    assert config["pooling"] == "global_average"
    assert config["parameter_count"] > 0


def test_base_pipeline_is_switched_without_changing_representations() -> None:
    experiment.configure_base_module()
    assert experiment.exp.train_classifier is experiment.train_classifier
    assert experiment.exp.METHOD_NAMES == experiment.METHOD_NAMES
    x = np.random.default_rng(4).normal(size=(2, 128, 9)).astype(np.float32)
    reconstruction = np.zeros_like(x)
    arrays = experiment.exp.representation_arrays(x, reconstruction)
    assert arrays["B0"].shape == (2, 128, 9)
    assert arrays["B1"].shape == (2, 128, 9)
    assert arrays["B2"].shape == (2, 128, 27)
    assert arrays["B3"].shape == (2, 128, 36)


def test_interrupted_epoch_checkpoint_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.set_num_threads(2)
    rng = np.random.default_rng(8)
    train_x = rng.normal(size=(16, 128, 9)).astype(np.float32)
    train_y = np.asarray([0, 1] * 8, dtype=int)
    val_x = rng.normal(size=(8, 128, 9)).astype(np.float32)
    val_y = np.asarray([0, 1] * 4, dtype=int)
    original_save = experiment.atomic_torch_save
    interrupted = False

    def save_then_interrupt(payload: object, path: Path) -> None:
        nonlocal interrupted
        original_save(payload, path)
        if not interrupted:
            interrupted = True
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(experiment, "atomic_torch_save", save_then_interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        experiment.train_classifier(train_x, train_y, val_x, val_y, tmp_path, 17,
                                    torch.device("cpu"), 2, 2, 1.0)
    assert (tmp_path / "inceptiontime_resume.pt").exists()
    assert not (tmp_path / "inceptiontime_best.pt").exists()

    monkeypatch.setattr(experiment, "atomic_torch_save", original_save)
    _, training, probability = experiment.train_classifier(
        train_x, train_y, val_x, val_y, tmp_path, 17, torch.device("cpu"), 2, 2, 1.0,
    )
    assert training["last_epoch"] == 2
    assert probability.shape == (8,)
    assert (tmp_path / "inceptiontime_best.pt").exists()
    assert not (tmp_path / "inceptiontime_resume.pt").exists()
