"""Create a labelled 32 Hz Daphnet dataset from canonical 64 Hz records.

Signal processing is deliberately explicit and reproducible:

1. design a 65-tap, 14 Hz low-pass FIR with a Kaiser(beta=5) window;
2. reflect-pad both ends by 32 samples without repeating the endpoints;
3. convolve once with the FIR independently over the nine channels;
4. compensate the 32-sample linear-phase group delay and crop back to the
   original 64 Hz time grid;
5. linearly interpolate acceleration from 64 Hz to 32 Hz;
6. resample binary labels at the same target times with nearest neighbour.

The output preserves continuous records.  It does not create or randomly
split windows.
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
DATASET_ROOT = ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset"

FS_IN = 64
FS_OUT = 32
FIR_TAPS = 65
FIR_CUTOFF_HZ = 14.0
KAISER_BETA = 5.0
MIRROR_PAD_SAMPLES = 32
GROUP_DELAY_SAMPLES = (FIR_TAPS - 1) // 2
DATASET_ID = "daphnet_32hz_fir14"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DATASET_ROOT / "processed")
    parser.add_argument("--output", type=Path, default=DATASET_ROOT / "processed_32Hz")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all processing and quality checks without writing the output directory.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
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
    if len(coefficients) != FIR_TAPS:
        raise AssertionError("unexpected FIR coefficient count")
    if not np.allclose(coefficients, coefficients[::-1], rtol=0.0, atol=1e-14):
        raise AssertionError("FIR coefficients are not symmetric")
    if GROUP_DELAY_SAMPLES != MIRROR_PAD_SAMPLES:
        raise AssertionError("this protocol requires 32-sample padding and delay")
    return coefficients


def target_sample_positions(input_samples: int) -> np.ndarray:
    if input_samples <= 0:
        raise ValueError("input record must contain at least one sample")
    output_samples = int(np.floor((input_samples - 1) * FS_OUT / FS_IN)) + 1
    target_times = np.arange(output_samples, dtype=np.float64) / FS_OUT
    positions = target_times * FS_IN
    if positions[-1] > input_samples - 1 + 1e-12:
        raise AssertionError("target grid extends beyond the input record")
    return positions


def filter_and_align_64hz(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 9:
        raise ValueError(f"expected acceleration [time, 9], got {values.shape}")
    if len(values) <= MIRROR_PAD_SAMPLES:
        raise ValueError("record is too short for 32-sample reflect padding")
    if not np.isfinite(values).all():
        raise ValueError("input acceleration contains non-finite values")

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
    aligned_end = aligned_start + len(values)
    aligned = convolved[aligned_start:aligned_end]
    if aligned.shape != values.shape:
        raise AssertionError(
            f"delay-compensated signal shape {aligned.shape} != input {values.shape}"
        )
    return aligned


def resample_acceleration_32hz(aligned_64hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(aligned_64hz, dtype=np.float64)
    positions = target_sample_positions(len(values))
    source_positions = np.arange(len(values), dtype=np.float64)
    output = np.empty((len(positions), values.shape[1]), dtype=np.float64)
    for channel in range(values.shape[1]):
        output[:, channel] = np.interp(
            positions,
            source_positions,
            values[:, channel],
        )
    return output.astype(np.float32), positions


def resample_labels_nearest_32hz(
    y_binary: np.ndarray,
    positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_binary, dtype=np.int8)
    if labels.ndim != 1:
        raise ValueError(f"expected one-dimensional labels, got {labels.shape}")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary 0/1")
    if positions is None:
        positions = target_sample_positions(len(labels))
    nearest = np.floor(np.asarray(positions, dtype=np.float64) + 0.5).astype(np.int64)
    nearest = np.clip(nearest, 0, len(labels) - 1)
    return labels[nearest].astype(np.int8, copy=False), nearest


def preprocess_record(
    x: np.ndarray,
    y_binary: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if len(x) != len(y_binary):
        raise ValueError("signal and label lengths differ")
    aligned = filter_and_align_64hz(x, coefficients)
    x_32hz, positions = resample_acceleration_32hz(aligned)
    y_32hz, nearest = resample_labels_nearest_32hz(y_binary, positions)
    if len(x_32hz) != len(y_32hz):
        raise AssertionError("resampled signal and labels differ in length")
    if not np.isfinite(x_32hz).all():
        raise AssertionError("resampled acceleration contains non-finite values")
    expected_labels = np.asarray(y_binary, dtype=np.int8)[nearest]
    if not np.array_equal(y_32hz, expected_labels):
        raise AssertionError("nearest-neighbour label verification failed")

    source_last_time = (len(x) - 1) / FS_IN
    output_last_time = (len(x_32hz) - 1) / FS_OUT
    audit = {
        "source_samples": len(x),
        "output_samples": len(x_32hz),
        "expected_output_samples": int(np.floor((len(x) - 1) * FS_OUT / FS_IN)) + 1,
        "source_last_sample_time_sec": source_last_time,
        "output_last_sample_time_sec": output_last_time,
        "endpoint_truncation_sec": source_last_time - output_last_time,
        "nearest_source_index_first": int(nearest[0]),
        "nearest_source_index_last": int(nearest[-1]),
        "label_match": True,
    }
    return x_32hz, y_32hz, audit


def filter_response(coefficients: np.ndarray) -> dict[str, Any]:
    frequencies, response = signal.freqz(coefficients, worN=32768, fs=FS_IN)
    magnitude = np.maximum(np.abs(response), np.finfo(np.float64).tiny)
    db = 20.0 * np.log10(magnitude)

    def value_at(frequency: float) -> float:
        index = int(np.argmin(np.abs(frequencies - frequency)))
        return float(db[index])

    passband = db[frequencies <= 12.0]
    stopband = db[frequencies >= 16.0]
    return {
        "coefficient_count": len(coefficients),
        "coefficient_sum": float(np.sum(coefficients)),
        "symmetric": bool(np.allclose(coefficients, coefficients[::-1], atol=1e-14)),
        "group_delay_samples_at_64hz": GROUP_DELAY_SAMPLES,
        "group_delay_seconds": GROUP_DELAY_SAMPLES / FS_IN,
        "gain_db_at_0hz": value_at(0.0),
        "gain_db_at_12hz": value_at(12.0),
        "gain_db_at_14hz": value_at(14.0),
        "gain_db_at_16hz": value_at(16.0),
        "passband_0_to_12hz_peak_to_peak_db": float(np.max(passband) - np.min(passband)),
        "maximum_stopband_gain_db_at_or_above_16hz": float(np.max(stopband)),
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

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
    source_totals = Counter()
    output_totals = Counter()

    for row in manifest_rows:
        record_id = row["record_id"]
        record_path = source / row["record_path"]
        with np.load(record_path, allow_pickle=False) as record:
            if set(record.files) != {"x", "y_binary"}:
                raise ValueError(f"{record_id}: unexpected arrays {record.files}")
            x = np.asarray(record["x"], dtype=np.float32)
            y = np.asarray(record["y_binary"], dtype=np.int8)

        x_32hz, y_32hz, audit = preprocess_record(x, y, coefficients)
        audit.update(
            {
                "record_id": record_id,
                "subject_id": row["subject_id"],
                "source_record_path": row["record_path"],
                "output_record_path": f"records/{record_id}.npz",
                "source_normal_samples": int(np.count_nonzero(y == 0)),
                "source_fog_samples": int(np.count_nonzero(y == 1)),
                "output_normal_samples": int(np.count_nonzero(y_32hz == 0)),
                "output_fog_samples": int(np.count_nonzero(y_32hz == 1)),
            }
        )
        record_audits.append(audit)

        source_totals["samples"] += len(y)
        source_totals["normal"] += int(np.count_nonzero(y == 0))
        source_totals["fog"] += int(np.count_nonzero(y == 1))
        output_totals["samples"] += len(y_32hz)
        output_totals["normal"] += int(np.count_nonzero(y_32hz == 0))
        output_totals["fog"] += int(np.count_nonzero(y_32hz == 1))

        event_count = 0
        for event_id, (start, end) in enumerate(true_runs(y_32hz == 1)):
            output_events.append(
                {
                    "dataset_id": DATASET_ID,
                    "record_id": record_id,
                    "subject_id": row["subject_id"],
                    "run_id": row["run_id"],
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
        output_totals["events"] += event_count

        output_manifest.append(
            {
                "dataset_id": DATASET_ID,
                "record_id": record_id,
                "record_path": f"records/{record_id}.npz",
                "source_file": row["source_file"],
                "subject_id": row["subject_id"],
                "run_id": row["run_id"],
                "segment_id": row["segment_id"],
                "source_start_row": row["source_start_row"],
                "source_end_row": row["source_end_row"],
                "sampling_rate_hz": FS_OUT,
                "n_samples": len(y_32hz),
                "duration_sec": len(y_32hz) / FS_OUT,
                "last_sample_time_sec": (len(y_32hz) - 1) / FS_OUT,
                "n_normal_samples": int(np.count_nonzero(y_32hz == 0)),
                "n_fog_samples": int(np.count_nonzero(y_32hz == 1)),
                "fog_event_count": event_count,
                "has_fog": bool(np.any(y_32hz == 1)),
                "usable": row["usable"],
                "source_sampling_rate_hz": FS_IN,
                "source_n_samples": len(y),
                "source_last_sample_time_sec": (len(y) - 1) / FS_IN,
                "downsampling_method": "FIR65 cutoff14Hz Kaiser(beta=5), reflect32, delay32, linear",
                "label_resampling": "nearest neighbour on the aligned 32 Hz time grid",
                "notes": row["notes"],
            }
        )

        if build is not None:
            np.savez_compressed(
                build / "records" / f"{record_id}.npz",
                x=x_32hz,
                y_binary=y_32hz,
            )

    source_event_rows = read_csv(source / "fog_events.csv")
    source_event_count = len(source_event_rows)
    endpoint_errors = [float(row["endpoint_truncation_sec"]) for row in record_audits]
    quality_checks = {
        "record_count_match": len(output_manifest) == len(manifest_rows),
        "all_record_lengths_match_formula": all(
            int(row["output_samples"]) == int(row["expected_output_samples"])
            for row in record_audits
        ),
        "all_label_arrays_match_nearest_source": all(
            bool(row["label_match"]) for row in record_audits
        ),
        "sample_accounting_match": (
            output_totals["normal"] + output_totals["fog"] == output_totals["samples"]
        ),
        "fog_event_count_preserved": output_totals["events"] == source_event_count,
        "fir_is_symmetric": bool(response["symmetric"]),
        "fir_unity_dc_gain": abs(float(response["coefficient_sum"]) - 1.0) < 1e-12,
        "maximum_endpoint_truncation_not_over_one_source_sample": (
            max(endpoint_errors, default=0.0) <= 1.0 / FS_IN + 1e-12
        ),
    }
    overall_pass = all(quality_checks.values())
    report = {
        "dataset_id": DATASET_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source),
        "output_directory": str(output),
        "dry_run": bool(args.dry_run),
        "source_manifest_sha256": sha256(source / "manifest.csv"),
        "source_fog_events_sha256": sha256(source / "fog_events.csv"),
        "record_count": len(output_manifest),
        "subject_count": len({row["subject_id"] for row in output_manifest}),
        "sampling_rate_hz": {"input": FS_IN, "output": FS_OUT},
        "filter": {
            "type": "linear-phase FIR low-pass",
            "design": "scipy.signal.firwin",
            "taps": FIR_TAPS,
            "cutoff_hz": FIR_CUTOFF_HZ,
            "window": "Kaiser",
            "kaiser_beta": KAISER_BETA,
            "padding": {
                "mode": "reflect",
                "samples_each_end_at_64hz": MIRROR_PAD_SAMPLES,
                "endpoint_repeated": False,
            },
            "convolution": "one full convolution over time with a [65,1] kernel",
            "group_delay_compensation_samples_at_64hz": GROUP_DELAY_SAMPLES,
            "response": response,
        },
        "resampling": {
            "acceleration": "linear interpolation on target times m/32 s",
            "labels": "nearest source label on target times m/32 s",
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
        build / "fir_kaiser65_cutoff14hz.csv",
        [
            {"tap_index": index, "coefficient": float(value)}
            for index, value in enumerate(coefficients)
        ],
    )
    shutil.copy2(source / "loso_folds.csv", build / "loso_folds.csv")

    source_schema = json.loads((source / "schema.json").read_text(encoding="utf-8"))
    source_schema["dataset_id"] = DATASET_ID
    source_schema["sampling_rate_hz"] = FS_OUT
    source_schema["derived_from"] = {
        "dataset_id": "daphnet",
        "sampling_rate_hz": FS_IN,
        "directory": str(source),
    }
    source_schema["preprocessing_32hz"] = report["filter"] | {
        "acceleration_resampling": report["resampling"]["acceleration"],
        "label_resampling": report["resampling"]["labels"],
        "output_dtype": {"x": "float32", "y_binary": "int8"},
    }
    write_json(build / "schema.json", source_schema)
    write_json(build / "preprocessing_report.json", report)
    (build / "README_32Hz.md").write_text(
        "# Daphnet processed_32Hz\n\n"
        "Continuous labelled Daphnet records downsampled from 64 Hz to 32 Hz.\n\n"
        "- Acceleration: 65-tap 14 Hz FIR, Kaiser beta=5.\n"
        "- Boundary handling: reflect padding by 32 source samples at both ends.\n"
        "- Filtering: one convolution, followed by 32-sample group-delay compensation.\n"
        "- Acceleration resampling: linear interpolation on the aligned 32 Hz grid.\n"
        "- Label resampling: nearest neighbour on the same grid.\n"
        "- Record arrays: `x [time,9] float32`, `y_binary [time] int8`.\n"
        "- No windows or train/validation/test split are created here.\n\n"
        "Consult `preprocessing_report.json`, `record_resampling_audit.csv`, and "
        "`fir_kaiser65_cutoff14hz.csv` for reproducibility and quality checks.\n",
        encoding="utf-8",
    )

    # Re-open every persisted record before publishing the directory.
    persisted_samples = 0
    for row in output_manifest:
        path = build / str(row["record_path"])
        with np.load(path, allow_pickle=False) as record:
            if set(record.files) != {"x", "y_binary"}:
                raise AssertionError(f"persisted {row['record_id']} arrays are invalid")
            if record["x"].shape != (int(row["n_samples"]), 9):
                raise AssertionError(f"persisted {row['record_id']} signal shape mismatch")
            if record["y_binary"].shape != (int(row["n_samples"]),):
                raise AssertionError(f"persisted {row['record_id']} label shape mismatch")
            if record["x"].dtype != np.float32 or record["y_binary"].dtype != np.int8:
                raise AssertionError(f"persisted {row['record_id']} dtype mismatch")
            persisted_samples += len(record["y_binary"])
    if persisted_samples != output_totals["samples"]:
        raise AssertionError("persisted sample count does not reconcile")

    build.replace(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
