from __future__ import annotations

import pytest
import torch

from cnbr_fog.gru_convergence_models import GRUMeanForecaster
from cnbr_fog.gru_hard_cluster_model import (
    HardClusterConditionedGRUMeanForecaster,
)


def build_hard() -> HardClusterConditionedGRUMeanForecaster:
    return HardClusterConditionedGRUMeanForecaster(
        in_channels=9,
        horizon=4,
        n_clusters=3,
        hidden_channels=8,
        dropout=0.0,
    )


def test_zero_embedding_matches_common_global_initialization() -> None:
    torch.manual_seed(42)
    global_model = GRUMeanForecaster(
        in_channels=9,
        horizon=4,
        hidden_channels=8,
        dropout=0.0,
        decoder="direct",
    )
    torch.manual_seed(42)
    hard_model = build_hard()
    global_state = global_model.state_dict()
    hard_state = hard_model.state_dict()
    for name, value in global_state.items():
        assert name in hard_state
        assert torch.equal(value, hard_state[name])
    assert torch.count_nonzero(hard_model.cluster_embedding.weight) == 0
    context = torch.randn(5, 9, 16)
    cluster = torch.tensor([0, 1, 2, 0, 2])
    with torch.no_grad():
        expected = global_model.forward_mean(context)
        actual = hard_model.forward_mean(context, cluster)
    assert torch.equal(actual, expected)


def test_cluster_embedding_can_change_predictions() -> None:
    model = build_hard()
    with torch.no_grad():
        model.cluster_embedding.weight[1].fill_(1.0)
    context = torch.randn(1, 9, 16).repeat(2, 1, 1)
    output = model.forward_mean(context, torch.tensor([0, 1]))
    assert output.shape == (2, 9, 4)
    assert not torch.equal(output[0], output[1])


@pytest.mark.parametrize(
    "cluster,error",
    [
        (torch.tensor([[0], [1]]), ValueError),
        (torch.tensor([0.0, 1.0]), TypeError),
        (torch.tensor([0, 3]), ValueError),
    ],
)
def test_cluster_contract(cluster: torch.Tensor, error: type[Exception]) -> None:
    with pytest.raises(error):
        build_hard().forward_mean(torch.randn(2, 9, 16), cluster)
