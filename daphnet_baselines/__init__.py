"""Reference baselines for Daphnet FoG/non-FoG classification."""

from .adapters import (
    ADAPTERS,
    DAPHNET_CHANNEL_NAMES,
    DAPHNET_EXCLUDED_SUBJECTS,
    DAPHNET_LOSO_SUBJECTS,
    LoadedDataset,
    load_dataset,
    resolve_sensor_channel_indices,
)
from .data import HistoryWindowDataset, materialize_history_windows
from .features import (
    TimeFrequencyFeatureExtractor,
    freeze_index_features,
)
from .models import CNNGRUClassifier, parameter_count

__all__ = [
    "ADAPTERS",
    "CNNGRUClassifier",
    "DAPHNET_CHANNEL_NAMES",
    "DAPHNET_EXCLUDED_SUBJECTS",
    "DAPHNET_LOSO_SUBJECTS",
    "HistoryWindowDataset",
    "LoadedDataset",
    "TimeFrequencyFeatureExtractor",
    "freeze_index_features",
    "load_dataset",
    "materialize_history_windows",
    "parameter_count",
    "resolve_sensor_channel_indices",
]
