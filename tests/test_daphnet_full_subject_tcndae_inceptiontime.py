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

import run_daphnet_full_subject_tcndae_inceptiontime as experiment


def test_tcndae_shapes_match_supplied_architecture() -> None:
    model = experiment.TCNDAE()
    reconstruction, latent = model(torch.randn(2, 9, 128))
    assert reconstruction.shape == (2, 9, 128)
    assert latent.shape == (2, 32, 32)
    assert torch.isfinite(reconstruction).all()


def test_tcndae_dilation_hierarchy_and_linear_head() -> None:
    model = experiment.TCNDAE()
    assert model.encoder_stage1.dilations == (1, 2)
    assert model.encoder_stage2.dilations == (1, 2, 4)
    assert model.encoder_stage3.dilations == (1, 2, 4, 8)
    assert isinstance(model.output_head[-1], torch.nn.Conv1d)
    assert model.architecture_config()["output_activation"] is None


def test_pipeline_switches_both_models() -> None:
    experiment.configure_pipeline()
    assert experiment.exp.train_nbm is experiment.train_nbm
    assert experiment.exp.train_classifier is experiment.inception.train_classifier
    assert experiment.exp.METHOD_NAMES["B2"] == "TCNDAE-R5-InceptionTime"


def test_eight_gpu_plan_is_complete_and_disjoint(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    rows = [{"subject_id": f"S{index % 8:02d}", "fold_id": f"fold{index:02d}",
             "train_windows": 100 + index} for index in range(30)]
    experiment.exp.write_csv(root / "splits" / "outer_folds" / "outer_fold_summary.csv", rows)
    plan_path = experiment.create_balanced_plan(root, [f"cuda:{index}" for index in range(8)])
    plan = experiment.json.loads(plan_path.read_text(encoding="utf-8"))
    assigned = [fold for worker in plan["workers"].values() for fold in worker["folds"]]
    assert len(assigned) == 30
    assert len(set(assigned)) == 30
    assert len(plan["workers"]) == 8


def test_tcndae_epoch_resume_restores_interrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.set_num_threads(2)
    rng = np.random.default_rng(12)
    windows = 30
    item = experiment.exp.SubjectWindows(
        subject="SXX", records=[],
        raw=rng.normal(size=(windows, 128, 9)).astype(np.float32),
        label=np.zeros(windows, dtype=np.int8),
        strict_clean=np.ones(windows, dtype=bool),
        record_index=np.zeros(windows, dtype=np.int16),
        record_id=np.asarray(["SXX_seg000"] * windows),
        start=np.arange(windows, dtype=np.int64) * 64,
    )
    scaler = experiment.exp.RobustScaler(
        center=np.zeros(9, dtype=np.float32), scale=np.ones(9, dtype=np.float32),
    )
    original_save = experiment.atomic_torch_save
    interrupted = False

    def save_then_interrupt(payload: object, path: Path) -> None:
        nonlocal interrupted
        original_save(payload, path)
        if path.name == "tcndae_resume.pt" and not interrupted:
            interrupted = True
            raise RuntimeError("simulated TCN-DAE interruption")

    monkeypatch.setattr(experiment, "atomic_torch_save", save_then_interrupt)
    with pytest.raises(RuntimeError, match="simulated TCN-DAE interruption"):
        experiment.train_nbm(
            item.raw, item, np.arange(windows), scaler, tmp_path, 21,
            torch.device("cpu"), 2, 2,
        )
    assert (tmp_path / "tcndae_resume.pt").exists()
    assert not (tmp_path / "nbm_best.pt").exists()

    monkeypatch.setattr(experiment, "atomic_torch_save", original_save)
    _, training = experiment.train_nbm(
        item.raw, item, np.arange(windows), scaler, tmp_path, 21,
        torch.device("cpu"), 2, 2,
    )
    assert training["last_epoch"] == 2
    assert training["resumed"] is True
    assert (tmp_path / "nbm_best.pt").exists()
    assert not (tmp_path / "tcndae_resume.pt").exists()
