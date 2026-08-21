"""Filter the labeled P08 five-IMU signals and resample them from 100 to 64 Hz.

Each output NPZ contains exactly two arrays:

* ``x``: ``[time, 30]`` float32 signal array;
* ``y_binary``: ``[time]`` int8 FOG label array (0=Non-FOG, 1=FOG).

Signal processing follows the project NGM convention: a 65-tap, 28 Hz
linear-phase FIR low-pass anti-aliasing filter with a Kaiser(beta=5) window,
32-sample reflect padding at both ends, group-delay compensation, and
time-aligned resampling to 64 Hz. Labels use nearest-neighbour sampling on the
same output time grid and are never filtered.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal


# =============================================================================
# Manual settings for running directly from PyCharm.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = (
    PROJECT_ROOT / "dataset" / "All_dataset" / "segments_experimental_labeled"
)
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "All_dataset" / "processed_NGM_P08"
OUTPUT_FILES = {
    "seg000": "P08_seg000.npz",
    "seg001": "P08_seg001.npz",
}
FS_IN = 100
FS_OUT = 64
FIR_TAPS = 65
FIR_CUTOFF_HZ = 28.0
KAISER_BETA = 5.0
MIRROR_PAD_SAMPLES = 32
GROUP_DELAY_SAMPLES = (FIR_TAPS - 1) // 2
OVERWRITE_OUTPUTS = False
# =============================================================================


CHANNEL_COLUMNS = tuple(
    f"imu{imu}_{component}"
    for imu in range(1, 6)
    for component in ("ax", "ay", "az", "gx", "gy", "gz")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=OVERWRITE_OUTPUTS,
        help="Replace the two exact P08 NPZ outputs if they already exist.",
    )
    return parser.parse_args()


def find_summary(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("*_experimental_y_binary_segments_summary.csv"))
    if len(paths) != 1:
        raise ValueError(
            f"Expected exactly one labeled experimental summary CSV, found {len(paths)}"
        )
    summary = pd.read_csv(paths[0])
    if set(OUTPUT_FILES).difference(summary["segment_id"].astype(str)):
        raise ValueError("The labeled summary does not contain both seg000 and seg001")
    return summary


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
        raise AssertionError("Unexpected FIR coefficient count")
    if not np.allclose(coefficients, coefficients[::-1], rtol=0.0, atol=1e-14):
        raise AssertionError("FIR coefficients are not symmetric")
    if GROUP_DELAY_SAMPLES != MIRROR_PAD_SAMPLES:
        raise AssertionError("Padding and FIR group delay must both be 32 samples")
    if abs(float(coefficients.sum()) - 1.0) > 1e-12:
        raise AssertionError("FIR filter does not have unity DC gain")
    return coefficients


def filter_response(coefficients: np.ndarray) -> dict[str, float]:
    frequencies, response = signal.freqz(coefficients, worN=65536, fs=FS_IN)
    magnitude = np.maximum(np.abs(response), np.finfo(np.float64).tiny)
    db = 20.0 * np.log10(magnitude)

    def value_at(frequency: float) -> float:
        return float(db[int(np.argmin(np.abs(frequencies - frequency)))])

    passband = db[frequencies <= 24.0]
    stopband = db[frequencies >= FS_OUT / 2]
    response_metrics = {
        "gain_db_at_24hz": value_at(24.0),
        "gain_db_at_28hz": value_at(28.0),
        "gain_db_at_32hz": value_at(32.0),
        "passband_0_to_24hz_peak_to_peak_db": float(np.ptp(passband)),
        "maximum_stopband_gain_db_at_or_above_32hz": float(np.max(stopband)),
    }
    if response_metrics["maximum_stopband_gain_db_at_or_above_32hz"] >= -50.0:
        raise AssertionError("FIR stopband attenuation at the 64 Hz Nyquist is insufficient")
    return response_metrics


def target_sample_positions(input_samples: int) -> np.ndarray:
    if input_samples <= 0:
        raise ValueError("Input record must contain at least one sample")
    output_samples = int(np.floor((input_samples - 1) * FS_OUT / FS_IN)) + 1
    target_times = np.arange(output_samples, dtype=np.float64) / FS_OUT
    positions = target_times * FS_IN
    if positions[-1] > input_samples - 1 + 1e-12:
        raise AssertionError("Output time grid extends beyond the input record")
    return positions


def filter_and_align(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CHANNEL_COLUMNS):
        raise ValueError(f"Expected signal [time, 30], got {values.shape}")
    if len(values) <= MIRROR_PAD_SAMPLES:
        raise ValueError("Input record is too short for 32-sample reflect padding")
    if not np.isfinite(values).all():
        raise ValueError("Input signal contains missing or non-finite values")

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
        raise AssertionError("Delay-compensated signal shape mismatch")
    return aligned


def resample_signal(aligned_100hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(aligned_100hz, dtype=np.float64)
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


def resample_labels(
    y_binary: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_binary, dtype=np.int8)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("y_binary must be a one-dimensional int8-compatible 0/1 array")
    nearest = np.floor(np.asarray(positions, dtype=np.float64) + 0.5).astype(np.int64)
    nearest = np.clip(nearest, 0, len(labels) - 1)
    output = labels[nearest].astype(np.int8, copy=False)
    if not np.array_equal(output, labels[nearest]):
        raise AssertionError("Nearest-neighbour label resampling mismatch")
    return output, nearest


def count_fog_events(y_binary: np.ndarray) -> int:
    labels = np.asarray(y_binary, dtype=np.int8)
    padded = np.pad(labels, (1, 1), mode="constant")
    return int(np.count_nonzero(np.diff(padded) == 1))


def preprocess_record(
    x: np.ndarray,
    y_binary: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float | bool]]:
    if len(x) != len(y_binary):
        raise ValueError("Signal and label lengths differ")
    aligned = filter_and_align(x, coefficients)
    x_64hz, positions = resample_signal(aligned)
    y_64hz, nearest = resample_labels(y_binary, positions)
    if len(x_64hz) != len(y_64hz):
        raise AssertionError("Resampled signal and label lengths differ")
    if x_64hz.dtype != np.float32 or y_64hz.dtype != np.int8:
        raise AssertionError("Unexpected output dtypes")
    if not np.isfinite(x_64hz).all():
        raise AssertionError("Resampled signal contains non-finite values")
    if count_fog_events(y_64hz) != count_fog_events(y_binary):
        raise AssertionError("FOG event count changed during label resampling")

    source_last_time = (len(x) - 1) / FS_IN
    output_last_time = (len(x_64hz) - 1) / FS_OUT
    audit: dict[str, int | float | bool] = {
        "source_samples": len(x),
        "output_samples": len(x_64hz),
        "expected_output_samples": int(np.floor((len(x) - 1) * FS_OUT / FS_IN)) + 1,
        "source_fog_samples": int(np.count_nonzero(y_binary == 1)),
        "output_fog_samples": int(np.count_nonzero(y_64hz == 1)),
        "source_fog_events": count_fog_events(y_binary),
        "output_fog_events": count_fog_events(y_64hz),
        "source_last_sample_time_sec": source_last_time,
        "output_last_sample_time_sec": output_last_time,
        "endpoint_truncation_sec": source_last_time - output_last_time,
        "nearest_source_index_first": int(nearest[0]),
        "nearest_source_index_last": int(nearest[-1]),
        "label_match": bool(np.array_equal(y_64hz, np.asarray(y_binary)[nearest])),
    }
    if audit["output_samples"] != audit["expected_output_samples"]:
        raise AssertionError("Output sample count does not match the 64 Hz time-grid formula")
    if float(audit["endpoint_truncation_sec"]) >= 1.0 / FS_OUT + 1e-12:
        raise AssertionError("Endpoint truncation is at least one 64 Hz sample")
    return x_64hz, y_64hz, audit


def load_labeled_record(path: Path) -> tuple[np.ndarray, np.ndarray]:
    required_columns = [*CHANNEL_COLUMNS, "relative_time", "y_binary"]
    frame = pd.read_csv(path, usecols=required_columns)
    if list(frame.loc[:, list(CHANNEL_COLUMNS)].columns) != list(CHANNEL_COLUMNS):
        raise AssertionError("Input channel order is invalid")
    relative_time = frame["relative_time"].to_numpy(dtype=np.float64)
    if len(relative_time) < 2 or not np.all(np.diff(relative_time) > 0):
        raise ValueError(f"{path.name} has invalid relative_time")
    estimated_fs = 1.0 / float(np.median(np.diff(relative_time)))
    if not np.isclose(estimated_fs, FS_IN, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"{path.name} sampling rate is {estimated_fs:.9f} Hz, expected {FS_IN} Hz"
        )
    x = frame.loc[:, list(CHANNEL_COLUMNS)].to_numpy(dtype=np.float64)
    y_binary = frame["y_binary"].to_numpy(dtype=np.int8)
    if not np.isin(y_binary, (0, 1)).all():
        raise ValueError(f"{path.name} contains labels other than 0 and 1")
    return x, y_binary


def save_npz_atomic(
    output_path: Path,
    x: np.ndarray,
    y_binary: np.ndarray,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; rerun with --overwrite if intended"
        )
    temporary_path = output_path.with_name(f"{output_path.name}.tmp.npz")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        np.savez_compressed(temporary_path, x=x, y_binary=y_binary)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def verify_persisted(
    path: Path,
    expected_x: np.ndarray,
    expected_y: np.ndarray,
) -> None:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"x", "y_binary"}:
            raise AssertionError(f"{path.name} contains unexpected arrays: {payload.files}")
        x = payload["x"]
        y_binary = payload["y_binary"]
    if x.shape != expected_x.shape or y_binary.shape != expected_y.shape:
        raise AssertionError(f"{path.name} persisted shapes do not match")
    if x.dtype != np.float32 or y_binary.dtype != np.int8:
        raise AssertionError(f"{path.name} persisted dtypes do not match")
    if not np.array_equal(x, expected_x) or not np.array_equal(y_binary, expected_y):
        raise AssertionError(f"{path.name} persisted values do not match")


def main() -> None:
    args = parse_args()
    summary = find_summary(args.input_dir)
    coefficients = design_fir()
    response = filter_response(coefficients)

    planned_paths = [args.output_dir / name for name in OUTPUT_FILES.values()]
    existing = [path for path in planned_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, Path, np.ndarray, np.ndarray, dict[str, int | float | bool]]] = []
    for segment_id, output_name in OUTPUT_FILES.items():
        matched = summary.loc[summary["segment_id"] == segment_id]
        if len(matched) != 1:
            raise ValueError(f"Expected one summary row for {segment_id}, found {len(matched)}")
        input_path = args.input_dir / str(matched.iloc[0]["file"])
        x, y_binary = load_labeled_record(input_path)
        x_64hz, y_64hz, audit = preprocess_record(x, y_binary, coefficients)
        output_path = args.output_dir / output_name
        results.append((segment_id, output_path, x_64hz, y_64hz, audit))

    for _, output_path, x_64hz, y_64hz, _ in results:
        save_npz_atomic(output_path, x_64hz, y_64hz, args.overwrite)
        verify_persisted(output_path, x_64hz, y_64hz)

    print("P08 64 Hz preprocessing complete.")
    print(
        f"  FIR: {FIR_TAPS} taps, cutoff {FIR_CUTOFF_HZ:g} Hz, "
        f"Kaiser beta={KAISER_BETA:g}, reflect pad {MIRROR_PAD_SAMPLES}, "
        f"delay compensation {GROUP_DELAY_SAMPLES}"
    )
    print(
        f"  response: passband ripple 0-24 Hz "
        f"{response['passband_0_to_24hz_peak_to_peak_db']:.6f} dB; "
        f"maximum gain >=32 Hz "
        f"{response['maximum_stopband_gain_db_at_or_above_32hz']:.3f} dB"
    )
    print(f"  channel order: {', '.join(CHANNEL_COLUMNS)}")
    for segment_id, output_path, x_64hz, y_64hz, audit in results:
        print(
            f"  {segment_id}: x {x_64hz.shape} {x_64hz.dtype}; "
            f"y_binary {y_64hz.shape} {y_64hz.dtype}; "
            f"FOG {int(y_64hz.sum()):,}; events {audit['output_fog_events']}; "
            f"saved {output_path}"
        )


if __name__ == "__main__":
    main()
