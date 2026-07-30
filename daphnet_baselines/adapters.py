"""Dataset adapters for the paper baseline suite.

The training code consumes the common :class:`cnbr_fog.data.DaphnetDataset`
record/window interface.  Dataset-specific validation and exclusion policies
live here so the canonical Daphnet experiment stays strict while a private
dataset can use the same trainer through the processed manifest contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from cnbr_fog.data import DaphnetDataset


DAPHNET_CHANNEL_NAMES = (
    "ankle_acc_forward",
    "ankle_acc_vertical",
    "ankle_acc_lateral",
    "thigh_acc_forward",
    "thigh_acc_vertical",
    "thigh_acc_lateral",
    "trunk_acc_forward",
    "trunk_acc_vertical",
    "trunk_acc_lateral",
)
DAPHNET_EXCLUDED_SUBJECTS = ("S04", "S10")
DAPHNET_LOSO_SUBJECTS = (
    "S01",
    "S02",
    "S03",
    "S05",
    "S06",
    "S07",
    "S08",
    "S09",
)


@dataclass(frozen=True)
class LoadedDataset:
    """A validated dataset plus adapter-level provenance."""

    adapter_name: str
    dataset: DaphnetDataset
    source_subjects: tuple[str, ...]
    excluded_subjects: tuple[str, ...]
    default_fi_channels: tuple[str, ...]
    canonical_daphnet: bool

    def metadata(self) -> dict:
        return {
            "adapter": self.adapter_name,
            "canonical_daphnet": self.canonical_daphnet,
            "source_subjects": list(self.source_subjects),
            "excluded_subjects": list(self.excluded_subjects),
            "default_fi_channels": list(self.default_fi_channels),
        }


class DatasetAdapter(Protocol):
    """Protocol implemented by dataset-specific loaders."""

    name: str

    def load(
        self,
        root: Path,
        *,
        excluded_subjects: Sequence[str],
        flatline_seconds: float,
        zero_tolerance: float,
    ) -> LoadedDataset:
        ...


def filter_subjects(
    dataset: DaphnetDataset,
    excluded_subjects: Sequence[str],
) -> DaphnetDataset:
    """Return a record-complete subject exclusion without mutating the source."""

    excluded = {str(subject) for subject in excluded_subjects}
    unknown = sorted(excluded - set(dataset.subjects))
    if unknown:
        raise ValueError(
            f"Excluded subjects are absent from the dataset: {unknown}; "
            f"available={tuple(dataset.subjects)}"
        )
    records = [
        record
        for record in dataset.records
        if record.subject_id not in excluded
    ]
    if not records:
        raise ValueError("Subject exclusion removed every record")
    return DaphnetDataset(
        root=dataset.root,
        records=records,
        sampling_rate_hz=dataset.sampling_rate_hz,
        channel_names=dataset.channel_names,
    )


def _default_vertical_channel(
    channel_names: Sequence[str],
) -> tuple[str, ...]:
    canonical = "ankle_acc_vertical"
    if canonical in channel_names:
        return (canonical,)
    vertical = [
        str(name)
        for name in channel_names
        if "vertical" in str(name).lower()
    ]
    return (vertical[0],) if vertical else ()


class DaphnetAdapter:
    """Strict adapter for the canonical public Daphnet experiment."""

    name = "daphnet"

    def load(
        self,
        root: Path,
        *,
        excluded_subjects: Sequence[str],
        flatline_seconds: float,
        zero_tolerance: float,
    ) -> LoadedDataset:
        exclusions = tuple(str(value) for value in excluded_subjects)
        if set(exclusions) != set(DAPHNET_EXCLUDED_SUBJECTS):
            raise ValueError(
                "The canonical Daphnet protocol requires exactly "
                "S04 and S10 to be excluded"
            )
        source = DaphnetDataset.load(
            root,
            flatline_seconds=flatline_seconds,
            zero_tolerance=zero_tolerance,
        )
        if source.sampling_rate_hz != 64:
            raise ValueError(
                f"Canonical Daphnet must be 64 Hz, got {source.sampling_rate_hz}"
            )
        if tuple(source.channel_names) != DAPHNET_CHANNEL_NAMES:
            raise ValueError(
                "Canonical Daphnet channel order differs: "
                f"{tuple(source.channel_names)}"
            )
        filtered = filter_subjects(source, exclusions)
        if tuple(filtered.subjects) != DAPHNET_LOSO_SUBJECTS:
            raise ValueError(
                "Canonical post-exclusion subjects differ: "
                f"expected={DAPHNET_LOSO_SUBJECTS}, got={tuple(filtered.subjects)}"
            )
        return LoadedDataset(
            adapter_name=self.name,
            dataset=filtered,
            source_subjects=tuple(source.subjects),
            excluded_subjects=tuple(sorted(exclusions)),
            default_fi_channels=("ankle_acc_vertical",),
            canonical_daphnet=True,
        )


class ManifestNPZAdapter:
    """Generic adapter for a private dataset using manifest/NPZ records.

    Required files and arrays are identical to the public processed export:

    ``manifest.csv``
        Contains ``record_path``, ``record_id``, ``subject_id``, ``run_id``,
        ``n_samples`` and ``sampling_rate_hz``.
    ``schema.json``
        Declares ordered channel names.
    record ``.npz``
        Contains exactly ``x`` (float ``[time, channel]``) and ``y_binary``
        (``0=non-FoG, 1=FoG``).
    """

    name = "manifest_npz"

    def load(
        self,
        root: Path,
        *,
        excluded_subjects: Sequence[str],
        flatline_seconds: float,
        zero_tolerance: float,
    ) -> LoadedDataset:
        source = DaphnetDataset.load(
            root,
            flatline_seconds=flatline_seconds,
            zero_tolerance=zero_tolerance,
        )
        exclusions = tuple(str(value) for value in excluded_subjects)
        filtered = (
            filter_subjects(source, exclusions)
            if exclusions
            else source
        )
        if len(filtered.subjects) < 3:
            raise ValueError(
                "LOSO with a subject-level validation split requires at least "
                "three retained subjects"
            )
        return LoadedDataset(
            adapter_name=self.name,
            dataset=filtered,
            source_subjects=tuple(source.subjects),
            excluded_subjects=tuple(sorted(exclusions)),
            default_fi_channels=_default_vertical_channel(
                filtered.channel_names
            ),
            canonical_daphnet=False,
        )


ADAPTERS: dict[str, DatasetAdapter] = {
    adapter.name: adapter
    for adapter in (DaphnetAdapter(), ManifestNPZAdapter())
}


def load_dataset(
    adapter_name: str,
    root: Path,
    *,
    excluded_subjects: Sequence[str],
    flatline_seconds: float,
    zero_tolerance: float,
) -> LoadedDataset:
    """Load a registered dataset adapter."""

    try:
        adapter = ADAPTERS[str(adapter_name).strip().lower()]
    except KeyError as error:
        raise ValueError(
            f"Unknown dataset adapter {adapter_name!r}; "
            f"available={tuple(ADAPTERS)}"
        ) from error
    return adapter.load(
        Path(root),
        excluded_subjects=excluded_subjects,
        flatline_seconds=float(flatline_seconds),
        zero_tolerance=float(zero_tolerance),
    )


def resolve_sensor_channel_indices(
    sensor_set: str,
    channel_names: Sequence[str],
) -> tuple[int, ...]:
    """Resolve ``all`` or a named sensor location from ordered channels."""

    name = str(sensor_set).strip().lower()
    channels = tuple(str(value) for value in channel_names)
    if name == "all":
        return tuple(range(len(channels)))
    if name not in {"ankle", "thigh", "trunk"}:
        raise ValueError(
            f"Unknown sensor set {sensor_set!r}; use all, ankle, thigh, or trunk"
        )
    indices = tuple(
        index
        for index, channel in enumerate(channels)
        if channel.lower().startswith(f"{name}_")
    )
    if not indices:
        raise ValueError(
            f"Sensor set {name!r} is unavailable in channels={channels}; "
            "use --sensor-set all or provide canonical location prefixes"
        )
    return indices

