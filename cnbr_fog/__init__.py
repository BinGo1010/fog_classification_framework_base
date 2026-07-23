"""Conditional normal-behaviour residual models for FoG detection."""

from .data import DaphnetDataset, DaphnetTrunkDataset, RobustChannelScaler, WindowTable
from .histories import (
    HistoryPlan,
    make_block_history_input,
    make_common_history_plan,
    materialize_nonoverlap_residual_history,
)
from .models import ConditionalNormalPredictor, ResidualTCNClassifier
from .nbm import (
    NBM_NAMES,
    GRUNBM,
    LinearARNBM,
    PersistenceNBM,
    TCNNBM,
    TransformerNBM,
    build_nbm,
    gaussian_nll_sigma,
)

__all__ = [
    "ConditionalNormalPredictor",
    "DaphnetDataset",
    "DaphnetTrunkDataset",
    "HistoryPlan",
    "GRUNBM",
    "LinearARNBM",
    "NBM_NAMES",
    "PersistenceNBM",
    "ResidualTCNClassifier",
    "RobustChannelScaler",
    "WindowTable",
    "TCNNBM",
    "TransformerNBM",
    "build_nbm",
    "gaussian_nll_sigma",
    "make_block_history_input",
    "make_common_history_plan",
    "materialize_nonoverlap_residual_history",
]
