"""Build the Stanford 7-subject, 5-IMU NGM dataset at 64 Hz.

The canonical 128 Hz ``imus6_subjects7`` records are transformed as follows:

1. keep lumbar, bilateral ankle, and bilateral foot IMUs (30 channels);
2. design a 65-tap, 28 Hz low-pass FIR with a Kaiser(beta=5) window;
3. reflect-pad both ends by 32 source samples;
4. filter every accelerometer and gyroscope channel and compensate the
   32-sample linear-phase group delay;
5. decimate the aligned signals from 128 Hz to 64 Hz;
6. sample binary labels by nearest neighbour on the same 64 Hz time grid.

Continuous walking-trial records are preserved.  No windows are created.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import signal


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "dataset" / "5.Stanford imu-fog-detection"

FS_IN = 128
FS_OUT = 64
FIR_TAPS = 65
FIR_CUTOFF_HZ = 28.0
KAISER_BETA = 5.0
MIRROR_PAD_SAMPLES = 32
GROUP_DELAY_SAMPLES = (FIR_TAPS - 1) // 2

DATASET_ID = "stanford_imu_fog_5imu_64hz_kaiser5"
SUBSET_ID = "imus5_subjects7"
SELECTED_SENSORS = ("lumbar", "ankle_l", "ankle_r", "foot_l", "foot_r")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DATASET_ROOT / "processed" / "imus6_subjects7",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_ROOT / "processed_NGM",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and validate all records without writing the output directory.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def true_runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    values = np.asarray(mask, dtype=np.int8)
    padded = np.pad(values, (1, 1), mode="constant")
    edges = np.flatnonzero(np.diff(padded))
    for start, end in edges.reshape(-1, 2):
        yield int(start), int(end)


def design_fir() -> np.ndarray:
    coefficients = signal.firwin(
        numtaps=FIR_TAPS,
        cutoff=FIR_CUTOFF_HZ,
        window=("kaiser", KAISER_BETA),
        pass_zero="lowpass",
        scale=True,
        fs=FS_IN,
    ).astype(np.float64)
    if not np.allclose(coefficients, coefficients[::-1], rtol=0.0, atol=1e-14):
        raise AssertionError("FIR coefficients are not symmetric")
    if GROUP_DELAY_SAMPLES != MIRROR_PAD_SAMPLES:
        raise AssertionError("padding and group delay must both be 32 samples")
    return coefficients


def target_source_indices(input_samples: int) -> np.ndarray:
    if input_samples <= 0:
        raise ValueError("input record must contain at least one sample")
    return np.arange(0, input_samples, FS_IN // FS_OUT, dtype=np.int64)


def filter_and_align_128hz(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    expected_channels = len(SELECTED_SENSORS) * 6
    if values.ndim != 2 or values.shape[1] != expected_channels:
        raise ValueError(
            f"expected signal [time, {expected_channels}], got {values.shape}"
        )
    if len(values) <= MIRROR_PAD_SAMPLES:
        raise ValueError("record is too short for 32-sample reflect padding")
    if not np.isfinite(values).all():
        raise ValueError("input signal contains non-finite values")

    padded = np.pad(
        values,
        ((MIRROR_PAD_SAMPLES, MIRROR_PAD_SAMPLES), (0, 0)),
        mode="reflect",
    )
    convolved = signal.convolve(
        padded,
        coefficients[:, None],
        mode="full",
        method="auto",
    )
    aligned_start = MIRROR_PAD_SAMPLES + GROUP_DELAY_SAMPLES
    aligned = convolved[aligned_start : aligned_start + len(values)]
    if aligned.shape != values.shape:
        raise AssertionError("delay-compensated signal shape mismatch")
    return aligned


def resample_signal_64hz(aligned_128hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(aligned_128hz, dtype=np.float64)
    source_indices = target_source_indices(len(values))
    return values[source_indices].astype(np.float32), source_indices


def resample_labels_nearest_64hz(
    y_binary: np.ndarray,
    source_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_binary, dtype=np.int8)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be a one-dimensional binary 0/1 array")
    if source_indices is None:
        source_indices = target_source_indices(len(labels))
    indices = np.asarray(source_indices, dtype=np.int64)
    if indices.size == 0 or indices[0] != 0 or indices[-1] >= len(labels):
        raise ValueError("invalid target source indices")
    return labels[indices].astype(np.int8, copy=False), indices


def preprocess_record(
    x: np.ndarray,
    y_binary: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if len(x) != len(y_binary):
        raise ValueError("signal and label lengths differ")
    aligned = filter_and_align_128hz(x, coefficients)
    x_64hz, source_indices = resample_signal_64hz(aligned)
    y_64hz, label_indices = resample_labels_nearest_64hz(y_binary, source_indices)
    if not np.array_equal(source_indices, label_indices):
        raise AssertionError("signal and label target grids differ")
    if len(x_64hz) != len(y_64hz) or not np.isfinite(x_64hz).all():
        raise AssertionError("invalid resampled record")

    source_last_time = (len(x) - 1) / FS_IN
    output_last_time = (len(x_64hz) - 1) / FS_OUT
    audit = {
        "source_samples": len(x),
        "output_samples": len(x_64hz),
        "expected_output_samples": (len(x) + 1) // 2,
        "source_last_sample_time_sec": source_last_time,
        "output_last_sample_time_sec": output_last_time,
        "endpoint_truncation_sec": source_last_time - output_last_time,
        "nearest_source_index_first": int(source_indices[0]),
        "nearest_source_index_last": int(source_indices[-1]),
        "label_match": bool(np.array_equal(y_64hz, y_binary[source_indices])),
    }
    return x_64hz, y_64hz, audit


def filter_response(coefficients: np.ndarray) -> dict[str, Any]:
    frequencies, response = signal.freqz(coefficients, worN=65536, fs=FS_IN)
    magnitude = np.maximum(np.abs(response), np.finfo(np.float64).tiny)
    db = 20.0 * np.log10(magnitude)

    def value_at(frequency: float) -> float:
        index = int(np.argmin(np.abs(frequencies - frequency)))
        return float(db[index])

    passband = db[frequencies <= 24.0]
    stopband = db[frequencies >= 32.0]
    return {
        "coefficient_count": len(coefficients),
        "coefficient_sum": float(np.sum(coefficients)),
        "symmetric": bool(
            np.allclose(coefficients, coefficients[::-1], rtol=0.0, atol=1e-14)
        ),
        "group_delay_samples_at_128hz": GROUP_DELAY_SAMPLES,
        "group_delay_seconds": GROUP_DELAY_SAMPLES / FS_IN,
        "gain_db_at_0hz": value_at(0.0),
        "gain_db_at_24hz": value_at(24.0),
        "gain_db_at_28hz": value_at(28.0),
        "gain_db_at_32hz": value_at(32.0),
        "passband_0_to_24hz_peak_to_peak_db": float(
            np.max(passband) - np.min(passband)
        ),
        "maximum_stopband_gain_db_at_or_above_32hz": float(np.max(stopband)),
    }


def selected_channel_metadata(
    source_schema: dict[str, Any],
) -> tuple[list[int], list[dict[str, Any]]]:
    source_channels = source_schema["channels"]
    indices: list[int] = []
    channels: list[dict[str, Any]] = []
    for sensor_name in SELECTED_SENSORS:
        sensor_indices = [
            index
            for index, channel in enumerate(source_channels)
            if channel["sensor"] == sensor_name
        ]
        if len(sensor_indices) != 6:
            raise ValueError(
                f"expected 6 channels for {sensor_name}, found {len(sensor_indices)}"
            )
        indices.extend(sensor_indices)
        channels.extend(dict(source_channels[index]) for index in sensor_indices)
    if len(indices) != 30 or len(set(indices)) != 30:
        raise AssertionError("selected channel mapping is invalid")
    return indices, channels


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    source_schema = json.loads((source / "schema.json").read_text(encoding="utf-8"))
    channel_indices, output_channels = selected_channel_metadata(source_schema)
    manifest_rows = read_csv(source / "manifest.csv")
    coefficients = design_fir()
    response = filter_response(coefficients)

    build: Path | None = None
    if not args.dry_run:
        build = output.with_name(f"{output.name}.__building_{os.getpid()}")
        build.mkdir(parents=True, exist_ok=False)
        (build / "records").mkdir()

    output_manifest: list[dict[str, Any]] = []
    output_events: list[dict[str, Any]] = []
    record_audits: list[dict[str, Any]] = []
    source_totals: Counter[str] = Counter()
    output_totals: Counter[str] = Counter()

    for row in manifest_rows:
        record_id = row["record_id"]
        with np.load(source / row["record_path"], allow_pickle=False) as record:
            if set(record.files) != {"x", "y_binary"}:
                raise ValueError(f"{record_id}: unexpected arrays {record.files}")
            source_x = np.asarray(record["x"], dtype=np.float32)
            source_y = np.asarray(record["y_binary"], dtype=np.int8)
        if source_x.shape[1] != len(source_schema["channels"]):
            raise ValueError(f"{record_id}: source signal/schema channel mismatch")

        selected_x = source_x[:, channel_indices]
        x_64hz, y_64hz, audit = preprocess_record(
            selected_x, source_y, coefficients
        )
        source_normal = int(np.count_nonzero(source_y == 0))
        source_fog = int(np.count_nonzero(source_y == 1))
        output_normal = int(np.count_nonzero(y_64hz == 0))
        output_fog = int(np.count_nonzero(y_64hz == 1))
        audit.update(
            {
                "record_id": record_id,
                "subject_id": row["subject_id"],
                "source_record_path": row["record_path"],
                "output_record_path": f"records/{record_id}.npz",
                "source_channels": source_x.shape[1],
                "selected_channels": selected_x.shape[1],
                "output_channels": x_64hz.shape[1],
                "source_normal_samples": source_normal,
                "source_fog_samples": source_fog,
                "output_normal_samples": output_normal,
                "output_fog_samples": output_fog,
            }
        )
        record_audits.append(audit)

        source_totals.update(
            samples=len(source_y), normal=source_normal, fog=source_fog
        )
        output_totals.update(
            samples=len(y_64hz), normal=output_normal, fog=output_fog
        )

        event_count = 0
        for event_id, (start, end) in enumerate(true_runs(y_64hz == 1)):
            output_events.append(
                {
                    "dataset_id": DATASET_ID,
                    "subset_id": SUBSET_ID,
                    "record_id": record_id,
                    "subject_id": row["subject_id"],
                    "segment_id": row["segment_id"],
                    "event_id": event_id,
                    "start_index": start,
                    "end_index": end - 1,
                    "start_time_sec": start / FS_OUT,
                    "end_time_sec": (end - 1) / FS_OUT,
                    "duration_sec": (end - start) / FS_OUT,
                }
            )
            event_count += 1
        output_totals.update(events=event_count)

        output_row: dict[str, Any] = dict(row)
        output_row.update(
            {
                "dataset_id": DATASET_ID,
                "subset_id": SUBSET_ID,
                "record_path": f"records/{record_id}.npz",
                "sampling_rate_hz": FS_OUT,
                "estimated_sampling_rate_hz": FS_OUT,
                "n_samples": len(y_64hz),
                "duration_sec": len(y_64hz) / FS_OUT,
                "n_normal_samples": output_normal,
                "n_fog_samples": output_fog,
                "fog_event_count": event_count,
                "has_fog": output_fog > 0,
                "source_sampling_rate_hz": FS_IN,
                "source_n_samples": len(source_y),
                "source_record_path": row["record_path"],
                "downsampling_method": (
                    "FIR65 cutoff28Hz Kaiser(beta=5), reflect32, "
                    "delay compensation32, decimate2"
                ),
                "label_resampling": (
                    "nearest source label on aligned 64 Hz grid "
                    "(source indices 0,2,4,...)"
                ),
                "notes": (
                    "derived from imus6_subjects7; chest removed; "
                    "continuous walking-trial record preserved"
                ),
            }
        )
        output_manifest.append(output_row)

        if build is not None:
            np.savez_compressed(
                build / "records" / f"{record_id}.npz",
                x=x_64hz,
                y_binary=y_64hz,
            )

    source_event_count = len(read_csv(source / "fog_events.csv"))
    endpoint_errors = [float(row["endpoint_truncation_sec"]) for row in record_audits]
    quality_checks = {
        "record_count_match": len(output_manifest) == len(manifest_rows),
        "subject_count_is_7": len({row["subject_id"] for row in output_manifest}) == 7,
        "all_records_have_30_channels": all(
            int(row["output_channels"]) == 30 for row in record_audits
        ),
        "all_record_lengths_match_formula": all(
            int(row["output_samples"]) == int(row["expected_output_samples"])
            for row in record_audits
        ),
        "all_label_arrays_match_nearest_source": all(
            bool(row["label_match"]) for row in record_audits
        ),
        "sample_accounting_match": (
            output_totals["normal"] + output_totals["fog"]
            == output_totals["samples"]
        ),
        "fog_event_count_preserved": output_totals["events"] == source_event_count,
        "fir_is_symmetric": bool(response["symmetric"]),
        "fir_unity_dc_gain": abs(float(response["coefficient_sum"]) - 1.0) < 1e-12,
        "stopband_at_32hz_is_below_minus_50db": (
            float(response["maximum_stopband_gain_db_at_or_above_32hz"]) < -50.0
        ),
        "maximum_endpoint_truncation_not_over_one_source_sample": (
            max(endpoint_errors, default=0.0) <= 1.0 / FS_IN + 1e-12
        ),
    }
    overall_pass = all(quality_checks.values())
    report = {
        "dataset_id": DATASET_ID,
        "subset_id": SUBSET_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source),
        "output_directory": str(output),
        "dry_run": bool(args.dry_run),
        "source_manifest_sha256": sha256(source / "manifest.csv"),
        "source_fog_events_sha256": sha256(source / "fog_events.csv"),
        "record_count": len(output_manifest),
        "subject_count": len({row["subject_id"] for row in output_manifest}),
        "subjects": sorted({row["subject_id"] for row in output_manifest}),
        "selected_sensors": list(SELECTED_SENSORS),
        "channel_count": len(output_channels),
        "sampling_rate_hz": {"input": FS_IN, "output": FS_OUT},
        "filter": {
            "type": "linear-phase FIR low-pass anti-aliasing filter",
            "design": "scipy.signal.firwin",
            "taps": FIR_TAPS,
            "cutoff_hz": FIR_CUTOFF_HZ,
            "window": "Kaiser",
            "kaiser_beta": KAISER_BETA,
            "padding": {
                "mode": "reflect",
                "samples_each_end_at_128hz": MIRROR_PAD_SAMPLES,
                "endpoint_repeated": False,
            },
            "convolution": "one full convolution over time with a [65,1] kernel",
            "group_delay_compensation_samples_at_128hz": GROUP_DELAY_SAMPLES,
            "response": response,
        },
        "resampling": {
            "signals": "decimation by 2 after aligned FIR filtering",
            "labels": "nearest source label at source indices 0,2,4,...",
            "time_origin_sec": 0.0,
        },
        "source_totals": dict(source_totals),
        "output_totals": dict(output_totals),
        "source_fog_event_count": source_event_count,
        "maximum_endpoint_truncation_sec": max(endpoint_errors, default=0.0),
        "quality_checks": quality_checks,
        "overall_pass": overall_pass,
    }
    if not overall_pass:
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    assert build is not None
    write_csv(build / "manifest.csv", output_manifest)
    write_csv(build / "fog_events.csv", output_events)
    write_csv(build / "record_resampling_audit.csv", record_audits)
    write_csv(
        build / "fir_kaiser65_cutoff28hz.csv",
        [
            {"tap_index": index, "coefficient": float(value)}
            for index, value in enumerate(coefficients)
        ],
    )
    shutil.copy2(source / "loso_folds.csv", build / "loso_folds.csv")

    output_schema = dict(source_schema)
    output_schema.update(
        {
            "dataset_id": DATASET_ID,
            "subset_id": SUBSET_ID,
            "sampling_rate_hz": FS_OUT,
            "selected_sensors": list(SELECTED_SENSORS),
            "channels": output_channels,
            "derived_from": {
                "dataset_id": source_schema["dataset_id"],
                "subset_id": source_schema["subset_id"],
                "sampling_rate_hz": FS_IN,
                "directory": str(source),
            },
            "preprocessing_64hz": report["filter"]
            | {
                "signal_resampling": report["resampling"]["signals"],
                "label_resampling": report["resampling"]["labels"],
                "output_dtype": {"x": "float32", "y_binary": "int8"},
            },
            "notes": [
                "Each record remains one continuous walking trial.",
                "The chest IMU was removed; five IMUs and 30 channels remain.",
                "No windows or pre-FOG labels are materialized.",
            ],
        }
    )
    write_json(build / "schema.json", output_schema)
    write_json(build / "preprocessing_report.json", report)
    (build / "README_64Hz.md").write_text(
        "# Stanford processed_NGM\n\n"
        "Seven-subject Stanford IMU FOG records with five retained IMUs at 64 Hz.\n\n"
        "- Sensors: lumbar, left/right ankle, left/right foot.\n"
        "- Channels: 30 (3-axis acceleration + 3-axis gyroscope per IMU).\n"
        "- Filtering: 65-tap 28 Hz low-pass FIR, Kaiser beta=5.\n"
        "- Boundary handling: reflect padding by 32 samples at both ends.\n"
        "- Delay handling: 32-sample linear-phase group-delay compensation.\n"
        "- Resampling: aligned signals decimated from 128 Hz to 64 Hz.\n"
        "- Labels: nearest source label on the same 64 Hz grid.\n"
        "- Arrays: `x [time,30] float32`, `y_binary [time] int8`.\n"
        "- Continuous records are preserved; no windows are created.\n\n"
        "See `schema.json`, `preprocessing_report.json`, "
        "`record_resampling_audit.csv`, and `fir_kaiser65_cutoff28hz.csv`.\n",
        encoding="utf-8",
    )

    persisted_samples = 0
    for row in output_manifest:
        with np.load(build / str(row["record_path"]), allow_pickle=False) as record:
            if set(record.files) != {"x", "y_binary"}:
                raise AssertionError(f"persisted {row['record_id']} arrays are invalid")
            expected_samples = int(row["n_samples"])
            if record["x"].shape != (expected_samples, 30):
                raise AssertionError(f"persisted {row['record_id']} signal shape mismatch")
            if record["y_binary"].shape != (expected_samples,):
                raise AssertionError(f"persisted {row['record_id']} label shape mismatch")
            if record["x"].dtype != np.float32 or record["y_binary"].dtype != np.int8:
                raise AssertionError(f"persisted {row['record_id']} dtype mismatch")
            persisted_samples += expected_samples
    if persisted_samples != output_totals["samples"]:
        raise AssertionError("persisted sample count does not reconcile")

    build.replace(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
