from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_daphnet_ca_supcon_subject as ca


def test_class_aware_supcon_matches_manual_class_average() -> None:
    z = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 0, 0, 1, 1])
    temperature = 0.2
    actual = ca.ClassAwareSupConLoss(temperature)(z, labels)

    normalized = torch.nn.functional.normalize(z, dim=1)
    anchor_losses: list[torch.Tensor] = []
    for index in range(len(z)):
        logits = torch.mv(normalized, normalized[index]) / temperature
        denominator = torch.logsumexp(
            torch.cat((logits[:index], logits[index + 1 :])), dim=0
        )
        positives = [j for j in range(len(z)) if j != index and labels[j] == labels[index]]
        anchor_losses.append(
            torch.stack([-(logits[j] - denominator) for j in positives]).mean()
        )
    expected = 0.5 * (
        torch.stack(anchor_losses[:3]).mean() + torch.stack(anchor_losses[3:]).mean()
    )
    assert torch.allclose(actual, expected, atol=1e-10)


def test_event_aware_sampler_enforces_class_and_group_limits() -> None:
    labels: list[int] = []
    groups: list[str] = []
    for label in (0, 1):
        for group in range(6):
            labels.extend([label] * 6)
            groups.extend([f"c{label}_g{group}"] * 6)
    sampler = ca.EventAwareBalancedBatchSampler(
        labels, groups, batch_size=32, seed=2026, steps_per_epoch=3
    )
    first_pass = list(iter(sampler))
    second_pass = list(iter(sampler))
    assert first_pass == second_pass
    for batch in first_pass:
        batch_labels = [labels[index] for index in batch]
        assert Counter(batch_labels) == {0: 16, 1: 16}
        for label in (0, 1):
            selected_groups = [groups[index] for index in batch if labels[index] == label]
            counts = Counter(selected_groups)
            assert len(counts) >= 4
            assert max(counts.values()) <= 4


def test_train_only_robust_scale_uses_median_and_iqr() -> None:
    x = np.zeros((2, 2, 2), dtype=np.float32)
    x[:, 0, :] = np.asarray([[0, 1], [2, 3]])
    x[:, 1, :] = np.asarray([[10, 20], [30, 40]])
    scale = ca.fit_robust_scale(x)
    assert np.allclose(scale.center, [1.5, 25.0])
    assert np.allclose(scale.scale, [1.5, 15.0])
    transformed = scale.transform(x)
    assert transformed.shape == x.shape
    assert np.isfinite(transformed).all()


def test_sampling_groups_preserve_frozen_split_and_use_original_events() -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["R1", "R1", "R1", "R1"],
            "group_id": ["g_fog", "g_fog", "g_n", "g_n"],
            "y_binary": [1, 1, 0, 0],
            "overlapping_event_ids": ["3", "3", "", ""],
            "start_index": [0, 64, 0, 1408],
            "ca_split": ["train"] * 4,
        }
    )
    result = ca.add_sampling_groups(frame)
    assert result["ca_split"].tolist() == frame["ca_split"].tolist()
    assert result.loc[0, "sampling_group_id"] == "R1:fog_event:3"
    assert result.loc[1, "sampling_group_id"] == "R1:fog_event:3"
    assert result.loc[2, "sampling_group_id"].endswith(":000")
    assert result.loc[3, "sampling_group_id"].endswith(":001")


def test_event_metrics_counts_event_detection_and_false_alarm_episodes() -> None:
    metadata = pd.DataFrame(
        {
            "record_id": ["R1"] * 6,
            "group_id": ["e0", "e0", "n0", "n0", "n0", "n0"],
            "overlapping_event_ids": ["0", "0", "", "", "", ""],
            "start_time_sec": [0, 1, 10, 11, 20, 21],
            "end_time_sec": [2, 3, 12, 13, 22, 23],
        }
    )
    labels = np.asarray([1, 1, 0, 0, 0, 0])
    probability = np.asarray([0.2, 0.8, 0.9, 0.8, 0.1, 0.7])
    result = ca.event_metrics(metadata, labels, probability, threshold=0.5)
    assert result["total_fog_events_with_pure_windows"] == 1
    assert result["detected_fog_events"] == 1
    assert result["fog_event_sensitivity"] == 1.0
    assert result["mean_detection_latency_sec"] == 1.0
    assert result["false_positive_episodes"] == 2


def test_processed_ca_pure_manifest_loads_frozen_subject_counts() -> None:
    data_dir = ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed_CA_pure"
    if not data_dir.exists():
        return
    splits, scaler, audit = ca.load_subject_data(data_dir, "S08")
    assert (len(splits["train"].y), int(splits["train"].y.sum())) == (346, 114)
    assert (len(splits["validation"].y), int(splits["validation"].y.sum())) == (85, 26)
    assert (len(splits["test"].y), int(splits["test"].y.sum())) == (106, 35)
    assert audit["random_resplit"] is False
    assert scaler.center.shape == (9,)

