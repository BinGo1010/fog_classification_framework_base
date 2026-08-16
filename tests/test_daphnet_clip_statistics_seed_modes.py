from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts import run_daphnet_residual_calibration_abcd as core


SPLITS = (
    "roles_6_7_train",
    "roles_2_3_validation",
    "roles_0_1_test",
    "all_classifier_roles_6_7_2_3_0_1",
)


def _count(points: int, clipped: int) -> dict[str, float | int]:
    return {"points": points, "clipped": clipped, "rate": clipped / points}


def _clip_payload(clipped: int) -> dict:
    stats = {
        "applicable": True,
        "definition": "test",
        "overall": _count(100, clipped),
        "nonfog": _count(80, min(clipped, 80)),
        "fog": _count(20, max(0, clipped - 80)),
        "per_channel": [],
    }
    for channel, name in enumerate(core.CHANNEL_NAMES):
        stats["per_channel"].append(
            {
                "channel": channel,
                "channel_name": name,
                "overall": _count(100, clipped),
                "nonfog": _count(80, min(clipped, 80)),
                "fog": _count(20, max(0, clipped - 80)),
            }
        )
    return {split: stats for split in SPLITS}


def _write_fake_results(root: Path, groups: tuple[str, ...], seeds: tuple[int, ...]) -> None:
    metrics = {key: 0.5 for key in core.METRIC_KEYS}
    metrics.update({"tn": 1, "fp": 1, "fn": 1, "tp": 1})
    subject_metrics = {subject: metrics for subject in core.SUBJECTS}
    (root / "TRAINING_BARRIER.json").write_text("{}", encoding="utf-8")
    for fold in core.FOLDS:
        for group in groups:
            for seed_index, seed in enumerate(seeds):
                directory = core.job_directory(root, fold, group, seed)
                directory.mkdir(parents=True, exist_ok=True)
                clip = _clip_payload(clipped=fold + seed_index + 1)
                payload = {
                    "fold": fold,
                    "group": group,
                    "tcn_seed": seed,
                    "nbm_seed": seed,
                    "threshold": 0.5,
                    "test": metrics,
                    "test_by_subject": subject_metrics,
                    "clip_statistics": clip,
                }
                (directory / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
                (directory / "DONE_TEST.json").write_text("{}", encoding="utf-8")


def test_paired_nbm_mode_accepts_seed_dependent_clip_statistics(tmp_path, monkeypatch):
    groups = ("G1", "G2")
    seeds = (0, 52, 161)
    _write_fake_results(tmp_path, groups, seeds)
    monkeypatch.setattr(core, "GROUPS", groups)
    monkeypatch.setattr(core, "GROUP_CONFIG", {group: {} for group in groups})
    monkeypatch.setattr(core, "CLIP_STATISTICS_SEED_MODE", "paired_nbm")
    monkeypatch.setattr(core, "CLIP_STATISTICS_EQUIVALENT_GROUP_PAIRS", (("G1", "G2"),))
    args = argparse.Namespace(
        output_root=tmp_path,
        groups=",".join(groups),
        tcn_seeds=",".join(map(str, seeds)),
    )

    core.run_aggregate(args)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["clip_statistics_seed_mode"] == "paired_nbm"
    assert (tmp_path / "DONE.json").is_file()
    csv_text = (tmp_path / "clip_rates_by_fold_split.csv").read_text(encoding="utf-8")
    assert "paired_seed" in csv_text
    assert "all_paired_seeds" in csv_text


def test_invariant_mode_still_rejects_seed_dependent_clip_statistics(tmp_path, monkeypatch):
    groups = ("A",)
    seeds = (0, 52, 161)
    _write_fake_results(tmp_path, groups, seeds)
    monkeypatch.setattr(core, "GROUPS", groups)
    monkeypatch.setattr(core, "GROUP_CONFIG", {"A": {}})
    monkeypatch.setattr(core, "CLIP_STATISTICS_SEED_MODE", "invariant")
    monkeypatch.setattr(core, "CLIP_STATISTICS_EQUIVALENT_GROUP_PAIRS", ())
    args = argparse.Namespace(
        output_root=tmp_path,
        groups="A",
        tcn_seeds=",".join(map(str, seeds)),
    )

    with pytest.raises(AssertionError, match="clip statistics depend on TCN seed"):
        core.run_aggregate(args)
