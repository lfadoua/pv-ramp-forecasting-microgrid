from __future__ import annotations

import hashlib
import gc
import json
import math
import os
import platform
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tables
import yaml
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state


TARGET_COLUMN = "ramp_rate_pu"
TIME_COLUMN = "timestamp"
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
CODE_SPLITS = {value: key for key, value in SPLIT_CODES.items()}
SECOND_VARIABLES = {
    "pv": "P_Solar[kW]",
    "poa": "POA Irr[kW1m2]",
    "ghi": "GHI[kW1m2]",
}
MINUTE_BASE_STATS = (
    "mean",
    "first",
    "last",
    "std",
    "min",
    "max",
    "range",
    "slope",
    "first_to_last_change",
    "max_abs_ramp",
    "mean_abs_ramp",
    "p90_abs_ramp",
    "p95_abs_ramp",
    "positive_ramp_count",
    "negative_ramp_count",
    "sign_change_count",
    "fraction_increasing",
    "fraction_decreasing",
    "coefficient_of_variation",
    "largest_upward_ramp",
    "largest_downward_ramp",
)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def resolve_project_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def prepare_output_tree(output_root: Path) -> dict[str, Path]:
    names = (
        "figures",
        "tables",
        "predictions",
        "models",
        "manifests",
        "logs",
        "article_ready",
        "kaggle_outputs",
    )
    output_root.mkdir(parents=True, exist_ok=False)
    paths = {name: output_root / name for name in names}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=False)
    return paths


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(packages: Sequence[str]) -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def system_memory_gb() -> tuple[float | None, float | None]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return float(memory.total / 2**30), float(memory.available / 2**30)
    except Exception:
        return None, None


def environment_manifest() -> dict[str, Any]:
    total_ram, available_ram = system_memory_gb()
    return {
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "total_ram_gb": total_ram,
        "available_ram_gb_at_preflight": available_ram,
        "packages": package_versions(
            [
                "numpy",
                "pandas",
                "pyarrow",
                "tables",
                "scikit-learn",
                "scipy",
                "matplotlib",
                "joblib",
                "reportlab",
                "clearml",
            ]
        ),
    }


class PeakMemoryMonitor:
    def __init__(self, interval_seconds: float = 0.2) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_bytes = 0

    def _sample(self) -> None:
        try:
            import psutil

            process = psutil.Process(os.getpid())
            while not self._stop.is_set():
                self.peak_rss_bytes = max(self.peak_rss_bytes, int(process.memory_info().rss))
                self._stop.wait(self.interval_seconds)
            self.peak_rss_bytes = max(self.peak_rss_bytes, int(process.memory_info().rss))
        except Exception:
            return

    def __enter__(self) -> "PeakMemoryMonitor":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def peak_gb(self) -> float | None:
        return float(self.peak_rss_bytes / 2**30) if self.peak_rss_bytes else None


class FixedFrameHDFReader:
    """Chunked reader for the pandas fixed-frame layout used by SOLETE."""

    def __init__(self, path: Path, key: str = "/DATA") -> None:
        self.path = Path(path)
        self.key = key.rstrip("/") or "/DATA"
        self.handle: tables.File | None = None
        self.values: Any = None
        self.index: Any = None
        self.columns: list[str] = []
        self.column_indices: dict[str, int] = {}

    def __enter__(self) -> "FixedFrameHDFReader":
        self.handle = tables.open_file(str(self.path), mode="r")
        group = self.handle.get_node(self.key)
        self.values = group.block0_values
        self.index = group.axis1
        raw_columns = group.block0_items[:]
        self.columns = [
            value.decode("utf-8").rstrip("\x00") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in raw_columns
        ]
        self.column_indices = {column: index for index, column in enumerate(self.columns)}
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.close()
        self.handle = None

    @property
    def row_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def column_count(self) -> int:
        return int(self.values.shape[1])

    def read(self, start: int, stop: int, columns: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        start = max(0, int(start))
        stop = min(self.row_count, int(stop))
        timestamps = np.asarray(self.index[start:stop], dtype=np.int64)
        block = np.asarray(self.values[start:stop, :])
        if columns is not None:
            missing = [column for column in columns if column not in self.column_indices]
            if missing:
                raise KeyError(f"Missing SOLETE columns: {missing}")
            indices = [self.column_indices[column] for column in columns]
            block = block[:, indices]
        return timestamps, block

    def timestamp_at(self, row: int) -> pd.Timestamp:
        return pd.Timestamp(int(self.index[int(row)]))

    def schema(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "key": self.key,
            "format": "pandas_fixed_frame",
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
            "value_dtype": str(self.values.dtype),
            "timestamp_dtype": str(self.index.dtype),
            "start_timestamp": self.timestamp_at(0).isoformat(),
            "end_timestamp": self.timestamp_at(self.row_count - 1).isoformat(),
        }


class ChronologicalSecondReader:
    """Reader for the timestamp-normalized disk-backed SOLETE second matrix."""

    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        self.path = Path(manifest["chronological_values_path"])
        self.columns = list(manifest["columns"])
        self.column_indices = {column: index for index, column in enumerate(self.columns)}
        self.start_ns = int(manifest["chronological_start_timestamp_ns"])
        self.values: np.ndarray | None = None
        self.index: Any = None

    def __enter__(self) -> "ChronologicalSecondReader":
        self.values = np.load(self.path, mmap_mode="r")
        self.index = _ImplicitSecondIndex(self.start_ns, int(self.values.shape[0]))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.values = None
        self.index = None

    @property
    def row_count(self) -> int:
        if self.values is not None:
            return int(self.values.shape[0])
        return int(self.manifest["chronological_row_count"])

    @property
    def column_count(self) -> int:
        return len(self.columns)

    def read(self, start: int, stop: int, columns: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if self.values is None:
            raise RuntimeError("ChronologicalSecondReader must be used as a context manager.")
        start = max(0, int(start))
        stop = min(self.row_count, int(stop))
        timestamps = self.start_ns + np.arange(start, stop, dtype=np.int64) * 1_000_000_000
        block = np.asarray(self.values[start:stop, :])
        if columns is not None:
            indices = [self.column_indices[column] for column in columns]
            block = block[:, indices]
        return timestamps, block

    def timestamp_at(self, row: int) -> pd.Timestamp:
        return pd.Timestamp(self.start_ns + int(row) * 1_000_000_000)

    def schema(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "source_path": self.manifest["source_path"],
            "format": "chronological_float32_npy",
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
            "value_dtype": "float32",
            "timestamp_dtype": "implicit_datetime64_ns_at_1Hz",
            "start_timestamp": self.timestamp_at(0).isoformat(),
            "end_timestamp": self.timestamp_at(self.row_count - 1).isoformat(),
        }


class _ImplicitSecondIndex:
    def __init__(self, start_ns: int, row_count: int) -> None:
        self.start_ns = int(start_ns)
        self.row_count = int(row_count)
        self.dtype = np.dtype(np.int64)
        self.shape = (self.row_count,)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, slice):
            start, stop, step = item.indices(self.row_count)
            return self.start_ns + np.arange(start, stop, step, dtype=np.int64) * 1_000_000_000
        index = int(item)
        if index < 0:
            index += self.row_count
        return np.int64(self.start_ns + index * 1_000_000_000)


def materialize_chronological_second_data(
    source_path: Path,
    output_dir: Path,
    key: str = "/DATA",
    chunk_rows: int = 600_000,
) -> dict[str, Any]:
    """Scatter the official file's shuffled day blocks onto its 1 Hz timeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    values_path = output_dir / "solete_one_second_chronological_float32.npy"
    seen_path = output_dir / "solete_one_second_seen.npy"
    started = time.perf_counter()
    with FixedFrameHDFReader(source_path, key) as source:
        raw_timestamp_magnitude = abs(int(source.index[0]))
        timestamp_scale_to_ns = 1 if raw_timestamp_magnitude >= 10**17 else 1_000 if raw_timestamp_magnitude >= 10**14 else 1_000_000
        minimum_ns = np.iinfo(np.int64).max
        maximum_ns = np.iinfo(np.int64).min
        local_non_one_second_steps = 0
        raw_order_backward_jumps = 0
        previous_timestamp: int | None = None
        for start in range(0, source.row_count, chunk_rows):
            stop = min(source.row_count, start + chunk_rows)
            timestamps = np.asarray(source.index[start:stop], dtype=np.int64) * timestamp_scale_to_ns
            minimum_ns = min(minimum_ns, int(timestamps.min()))
            maximum_ns = max(maximum_ns, int(timestamps.max()))
            if previous_timestamp is not None:
                boundary_difference = int(timestamps[0]) - previous_timestamp
                local_non_one_second_steps += int(boundary_difference != 1_000_000_000)
                raw_order_backward_jumps += int(boundary_difference < 0)
            differences = np.diff(timestamps)
            local_non_one_second_steps += int(np.sum(differences != 1_000_000_000))
            raw_order_backward_jumps += int(np.sum(differences < 0))
            previous_timestamp = int(timestamps[-1])
        span_ns = maximum_ns - minimum_ns
        if span_ns % 1_000_000_000:
            raise ValueError("One-second source endpoints are not aligned to whole seconds.")
        chronological_rows = span_ns // 1_000_000_000 + 1
        values = np.lib.format.open_memmap(
            values_path,
            mode="w+",
            dtype=np.float32,
            shape=(chronological_rows, source.column_count),
        )
        seen = np.lib.format.open_memmap(seen_path, mode="w+", dtype=np.uint8, shape=(chronological_rows,))
        seen[:] = 0
        duplicate_rows = 0
        invalid_timestamp_rows = 0
        missing_numeric_values = 0
        with PeakMemoryMonitor() as memory:
            for start in range(0, source.row_count, chunk_rows):
                stop = min(source.row_count, start + chunk_rows)
                timestamps, block = source.read(start, stop)
                timestamps = timestamps * timestamp_scale_to_ns
                offsets_ns = timestamps - minimum_ns
                invalid = (offsets_ns < 0) | (offsets_ns > span_ns) | (offsets_ns % 1_000_000_000 != 0)
                invalid_timestamp_rows += int(np.sum(invalid))
                if np.any(invalid):
                    continue
                indices = (offsets_ns // 1_000_000_000).astype(np.int64)
                unique_indices, counts = np.unique(indices, return_counts=True)
                duplicate_rows += int(np.sum(counts - 1))
                duplicate_rows += int(np.sum(seen[unique_indices] != 0))
                missing_numeric_values += int(np.isnan(block).sum())
                values[indices] = block.astype(np.float32)
                seen[indices] = 1
        values.flush()
        seen.flush()
        missing_chronological_rows = 0
        for start in range(0, chronological_rows, 5_000_000):
            stop = min(chronological_rows, start + 5_000_000)
            missing_chronological_rows += int(np.sum(seen[start:stop] == 0))
        manifest = {
            "source_path": str(source_path),
            "source_row_count": source.row_count,
            "source_column_count": source.column_count,
            "columns": source.columns,
            "source_first_stored_timestamp": pd.Timestamp(int(source.index[0]) * timestamp_scale_to_ns).isoformat(),
            "source_last_stored_timestamp": pd.Timestamp(int(source.index[source.row_count - 1]) * timestamp_scale_to_ns).isoformat(),
            "source_timestamp_scale_to_ns": timestamp_scale_to_ns,
            "source_order_non_one_second_transitions": local_non_one_second_steps,
            "source_order_backward_jumps": raw_order_backward_jumps,
            "chronological_start_timestamp": pd.Timestamp(minimum_ns).isoformat(),
            "chronological_end_timestamp": pd.Timestamp(maximum_ns).isoformat(),
            "chronological_start_timestamp_ns": int(minimum_ns),
            "chronological_end_timestamp_ns": int(maximum_ns),
            "chronological_row_count": int(chronological_rows),
            "chronological_values_path": str(values_path),
            "dtype": "float32",
            "duplicate_timestamp_rows": duplicate_rows,
            "missing_chronological_rows": missing_chronological_rows,
            "invalid_timestamp_rows": invalid_timestamp_rows,
            "missing_numeric_values": missing_numeric_values,
            "timestamp_reordering_required": local_non_one_second_steps > 0,
            "timestamp_reordering_policy": "direct scatter by integer second offset from global minimum timestamp",
            "generation_time_seconds": time.perf_counter() - started,
            "peak_memory_gb": memory.peak_gb,
        }
    del values, seen
    safe_remove(seen_path)
    if manifest["duplicate_timestamp_rows"] or manifest["missing_chronological_rows"] or manifest["invalid_timestamp_rows"]:
        raise ValueError(
            "Chronological normalization found duplicate, missing, or invalid timestamps: "
            f"duplicates={manifest['duplicate_timestamp_rows']}, missing={manifest['missing_chronological_rows']}, "
            f"invalid={manifest['invalid_timestamp_rows']}"
        )
    return manifest


def s1_target(power_kw: np.ndarray, rated_power_kw: float = 10.0) -> np.ndarray:
    power = np.asarray(power_kw, dtype=np.float64)
    return (power[2:] - power[1:-1]) / float(rated_power_kw)


def s2_target(power_kw: np.ndarray, rated_power_kw: float = 10.0) -> np.ndarray:
    power = np.asarray(power_kw, dtype=np.float64)
    if power.size < 61:
        return np.empty(0, dtype=np.float64)
    return (power[60:] - power[59:-1]) / float(rated_power_kw)


def backward_ramp(power_kw: np.ndarray, rated_power_kw: float = 10.0) -> np.ndarray:
    power = np.asarray(power_kw, dtype=np.float64)
    result = np.full(power.shape, np.nan, dtype=np.float64)
    result[1:] = np.diff(power) / float(rated_power_kw)
    return result


def persistence_prediction(current_ramp: np.ndarray) -> np.ndarray:
    return np.asarray(current_ramp, dtype=np.float64).copy()


def minute_ramp_target(minute_power_kw: np.ndarray, rated_power_kw: float = 10.0) -> np.ndarray:
    return backward_ramp(minute_power_kw, rated_power_kw)


def _ols_slopes(matrix: np.ndarray) -> np.ndarray:
    count = matrix.shape[1]
    if count <= 1:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    x = np.arange(count, dtype=np.float64)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    return np.dot(matrix, x_centered) / denominator


def _minute_descriptor_frame(
    timestamps_ns: np.ndarray,
    values: np.ndarray,
    column_indices: dict[str, int],
    rated_power_kw: float,
) -> pd.DataFrame:
    if len(values) == 0:
        return pd.DataFrame()
    if len(values) % 60 != 0:
        raise ValueError("Fast minute descriptor path requires complete 60-second blocks.")
    minute_count = len(values) // 60
    minute_starts = timestamps_ns[::60].astype("datetime64[ns]").astype("datetime64[m]").astype("datetime64[ns]")
    frame: dict[str, Any] = {TIME_COLUMN: minute_starts}
    for prefix, column in SECOND_VARIABLES.items():
        source = values[:, column_indices[column]].reshape(minute_count, 60).astype(np.float64, copy=False)
        changes = np.diff(source, axis=1) / (rated_power_kw if prefix == "pv" else 1.0)
        signs = np.sign(changes)
        sign_changes = np.sum(signs[:, 1:] * signs[:, :-1] < 0, axis=1)
        mean = np.mean(source, axis=1)
        std = np.std(source, axis=1, ddof=0)
        minimum = np.min(source, axis=1)
        maximum = np.max(source, axis=1)
        absolute_changes = np.abs(changes)
        frame[f"{prefix}_mean"] = mean.astype(np.float32)
        frame[f"{prefix}_first"] = source[:, 0].astype(np.float32)
        frame[f"{prefix}_last"] = source[:, -1].astype(np.float32)
        frame[f"{prefix}_std"] = std.astype(np.float32)
        frame[f"{prefix}_min"] = minimum.astype(np.float32)
        frame[f"{prefix}_max"] = maximum.astype(np.float32)
        frame[f"{prefix}_range"] = (maximum - minimum).astype(np.float32)
        frame[f"{prefix}_slope"] = _ols_slopes(source).astype(np.float32)
        frame[f"{prefix}_first_to_last_change"] = (source[:, -1] - source[:, 0]).astype(np.float32)
        frame[f"{prefix}_max_abs_ramp"] = np.max(absolute_changes, axis=1).astype(np.float32)
        frame[f"{prefix}_mean_abs_ramp"] = np.mean(absolute_changes, axis=1).astype(np.float32)
        frame[f"{prefix}_p90_abs_ramp"] = np.percentile(absolute_changes, 90, axis=1).astype(np.float32)
        frame[f"{prefix}_p95_abs_ramp"] = np.percentile(absolute_changes, 95, axis=1).astype(np.float32)
        frame[f"{prefix}_positive_ramp_count"] = np.sum(changes > 0, axis=1).astype(np.int16)
        frame[f"{prefix}_negative_ramp_count"] = np.sum(changes < 0, axis=1).astype(np.int16)
        frame[f"{prefix}_sign_change_count"] = sign_changes.astype(np.int16)
        frame[f"{prefix}_fraction_increasing"] = np.mean(changes > 0, axis=1).astype(np.float32)
        frame[f"{prefix}_fraction_decreasing"] = np.mean(changes < 0, axis=1).astype(np.float32)
        frame[f"{prefix}_coefficient_of_variation"] = (std / (np.abs(mean) + 1.0e-6)).astype(np.float32)
        frame[f"{prefix}_largest_upward_ramp"] = np.max(changes, axis=1).astype(np.float32)
        frame[f"{prefix}_largest_downward_ramp"] = np.min(changes, axis=1).astype(np.float32)
    pv_up = np.asarray(frame["pv_largest_upward_ramp"])
    pv_down = np.asarray(frame["pv_largest_downward_ramp"])
    frame["pv_cancellation_flag"] = ((pv_up > 0) & (pv_down < 0)).astype(np.int8)
    frame["second_count"] = np.full(minute_count, 60, dtype=np.int16)
    relevant_indices = [column_indices[column] for column in SECOND_VARIABLES.values()]
    valid_counts = np.sum(
        np.all(np.isfinite(values[:, relevant_indices]), axis=1).reshape(minute_count, 60),
        axis=1,
    ).astype(np.int16)
    frame["seconds_observed"] = np.full(minute_count, 60, dtype=np.int16)
    frame["valid_seconds_observed"] = valid_counts
    frame["complete_minute"] = valid_counts == 60
    frame["excluded_reason"] = np.where(
        valid_counts == 60,
        "",
        "nonfinite_one_second_input",
    )
    return pd.DataFrame(frame)


def _partial_minute_descriptor(
    timestamps_ns: np.ndarray,
    values: np.ndarray,
    column_indices: dict[str, int],
    rated_power_kw: float,
) -> pd.DataFrame:
    if len(values) == 0:
        return pd.DataFrame()
    row: dict[str, Any] = {
        TIME_COLUMN: pd.Timestamp(int(timestamps_ns[0])).floor("min"),
    }
    for prefix, column in SECOND_VARIABLES.items():
        source = values[:, column_indices[column]].astype(np.float64, copy=False)
        scale = rated_power_kw if prefix == "pv" else 1.0
        changes = np.diff(source) / scale
        absolute_changes = np.abs(changes)
        mean = float(np.mean(source))
        std = float(np.std(source))
        signs = np.sign(changes)
        descriptors = {
            "mean": mean,
            "first": float(source[0]),
            "last": float(source[-1]),
            "std": std,
            "min": float(np.min(source)),
            "max": float(np.max(source)),
            "range": float(np.ptp(source)),
            "slope": float(_ols_slopes(source.reshape(1, -1))[0]),
            "first_to_last_change": float(source[-1] - source[0]),
            "max_abs_ramp": float(np.max(absolute_changes)) if len(changes) else 0.0,
            "mean_abs_ramp": float(np.mean(absolute_changes)) if len(changes) else 0.0,
            "p90_abs_ramp": float(np.percentile(absolute_changes, 90)) if len(changes) else 0.0,
            "p95_abs_ramp": float(np.percentile(absolute_changes, 95)) if len(changes) else 0.0,
            "positive_ramp_count": int(np.sum(changes > 0)),
            "negative_ramp_count": int(np.sum(changes < 0)),
            "sign_change_count": int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0,
            "fraction_increasing": float(np.mean(changes > 0)) if len(changes) else 0.0,
            "fraction_decreasing": float(np.mean(changes < 0)) if len(changes) else 0.0,
            "coefficient_of_variation": std / (abs(mean) + 1.0e-6),
            "largest_upward_ramp": float(np.max(changes)) if len(changes) else 0.0,
            "largest_downward_ramp": float(np.min(changes)) if len(changes) else 0.0,
        }
        for name, value in descriptors.items():
            row[f"{prefix}_{name}"] = value
    row["pv_cancellation_flag"] = int(row["pv_largest_upward_ramp"] > 0 and row["pv_largest_downward_ramp"] < 0)
    row["second_count"] = int(len(values))
    relevant_indices = [column_indices[column] for column in SECOND_VARIABLES.values()]
    valid_count = int(np.sum(np.all(np.isfinite(values[:, relevant_indices]), axis=1)))
    row["seconds_observed"] = int(len(values))
    row["valid_seconds_observed"] = valid_count
    row["complete_minute"] = bool(len(values) == 60 and valid_count == 60)
    if len(values) != 60:
        row["excluded_reason"] = "fewer_than_60_one_second_observations"
    elif valid_count != 60:
        row["excluded_reason"] = "nonfinite_one_second_input"
    else:
        row["excluded_reason"] = ""
    return pd.DataFrame([row])


def ensure_minute_completeness_fields(
    descriptor_path: Path,
    descriptor_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add an explicit, reusable completeness audit to a descriptor checkpoint."""
    required_fields = {
        "seconds_observed",
        "valid_seconds_observed",
        "complete_minute",
        "excluded_reason",
    }
    schema_names = set(pq.read_schema(descriptor_path).names)
    upgraded = not required_fields.issubset(schema_names)
    if upgraded:
        temporary_path = descriptor_path.with_name(f"{descriptor_path.stem}.completeness_tmp.parquet")
        safe_remove(temporary_path)
        source = pq.ParquetFile(descriptor_path)
        writer: pq.ParquetWriter | None = None
        administrative = {TIME_COLUMN, "second_count", *required_fields}
        for batch in source.iter_batches(batch_size=50_000):
            frame = batch.to_pandas()
            seconds = frame["second_count"].to_numpy(dtype=np.int16, copy=False)
            descriptor_columns = [
                name
                for name in frame.columns
                if name not in administrative and pd.api.types.is_numeric_dtype(frame[name])
            ]
            finite = np.isfinite(frame[descriptor_columns].to_numpy(dtype=np.float64, copy=False)).all(axis=1)
            valid_seconds = np.where(finite, seconds, 0).astype(np.int16)
            complete = (seconds == 60) & (valid_seconds == 60)
            reason = np.full(len(frame), "", dtype=object)
            reason[seconds != 60] = "fewer_than_60_one_second_observations"
            reason[(seconds == 60) & ~finite] = "nonfinite_one_second_input"
            frame["seconds_observed"] = seconds
            frame["valid_seconds_observed"] = valid_seconds
            frame["complete_minute"] = complete
            frame["excluded_reason"] = reason
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary_path, table.schema, compression="zstd")
            writer.write_table(table)
        if writer is not None:
            writer.close()
        del source
        os.replace(temporary_path, descriptor_path)

    audit = pd.read_parquet(
        descriptor_path,
        columns=[
            TIME_COLUMN,
            "seconds_observed",
            "valid_seconds_observed",
            "complete_minute",
            "excluded_reason",
        ],
    )
    complete = audit["complete_minute"].astype(bool)
    reason_counts = (
        audit.loc[~complete, "excluded_reason"].replace("", "unspecified").value_counts().to_dict()
    )
    result = dict(descriptor_report or {})
    result.update(
        {
            "output_path": str(descriptor_path),
            "rows_written": int(len(audit)),
            "complete_minute_rows": int(complete.sum()),
            "incomplete_minute_rows": int((~complete).sum()),
            "excluded_reason_counts": {str(key): int(value) for key, value in reason_counts.items()},
            "first_minute_timestamp": pd.Timestamp(audit[TIME_COLUMN].min()).isoformat(),
            "last_minute_timestamp": pd.Timestamp(audit[TIME_COLUMN].max()).isoformat(),
            "completeness_requirement": "exactly 60 timestamped one-second rows and 60 finite PV/POA/GHI observations",
            "completeness_fields_added_to_checkpoint": bool(upgraded),
        }
    )
    return result


def build_minute_descriptors(
    reader: FixedFrameHDFReader,
    output_path: Path,
    rated_power_kw: float,
    chunk_rows: int = 600_000,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    required = list(dict.fromkeys(SECOND_VARIABLES.values()))
    indices = {column: reader.column_indices[column] for column in required}
    complete_rows = (reader.row_count // 60) * 60
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    duplicate_count = 0
    non_one_second_steps = 0
    missing_value_count = 0
    productive_row_count = 0
    start_time = time.perf_counter()
    with PeakMemoryMonitor() as memory:
        for start in range(0, complete_rows, chunk_rows - (chunk_rows % 60)):
            stop = min(complete_rows, start + chunk_rows - (chunk_rows % 60))
            timestamps, values = reader.read(start, stop)
            differences = np.diff(timestamps)
            duplicate_count += int(np.sum(differences == 0))
            non_one_second_steps += int(np.sum(differences != 1_000_000_000))
            missing_value_count += int(np.isnan(values).sum())
            productive_row_count += int(
                np.sum(
                    (values[:, reader.column_indices["Elevation[deg]"]] > 0.0)
                    & (values[:, reader.column_indices["P_Solar[kW]"]] > 0.05)
                )
            )
            frame = _minute_descriptor_frame(timestamps, values, reader.column_indices, rated_power_kw)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            rows_written += len(frame)
            del frame, table, values, timestamps
        if complete_rows < reader.row_count:
            timestamps, values = reader.read(complete_rows, reader.row_count)
            missing_value_count += int(np.isnan(values).sum())
            productive_row_count += int(
                np.sum(
                    (values[:, reader.column_indices["Elevation[deg]"]] > 0.0)
                    & (values[:, reader.column_indices["P_Solar[kW]"]] > 0.05)
                )
            )
            frame = _partial_minute_descriptor(timestamps, values, reader.column_indices, rated_power_kw)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            elif table.schema != writer.schema:
                table = table.cast(writer.schema, safe=False)
            writer.write_table(table)
            rows_written += len(frame)
    if writer is not None:
        writer.close()
    elapsed = time.perf_counter() - start_time
    schema = pq.read_schema(output_path)
    return {
        "output_path": str(output_path),
        "rows_written": int(rows_written),
        "complete_minute_rows": int(complete_rows // 60),
        "partial_minute_rows": int(reader.row_count - complete_rows),
        "feature_columns": [name for name in schema.names if name != TIME_COLUMN],
        "duplicate_timestamps_detected": duplicate_count,
        "non_one_second_steps_detected_within_chunks": non_one_second_steps,
        "missing_numeric_values": missing_value_count,
        "productive_rows_diagnostic": productive_row_count,
        "feature_generation_time_seconds": elapsed,
        "peak_memory_gb": memory.peak_gb,
    }


def second_feature_names(
    selected_columns: Sequence[str],
    lags: Sequence[int],
    windows: Sequence[int],
    extended_windows: Sequence[int],
) -> list[str]:
    names = [f"current__{column}" for column in selected_columns]
    for prefix in SECOND_VARIABLES:
        names.extend(f"{prefix}__lag_{lag}s" for lag in lags)
        for window in windows:
            names.extend((f"{prefix}__mean_{window}s", f"{prefix}__std_{window}s"))
        for window in extended_windows:
            names.extend(
                (
                    f"{prefix}__min_{window}s",
                    f"{prefix}__max_{window}s",
                    f"{prefix}__range_{window}s",
                    f"{prefix}__first_{window}s",
                    f"{prefix}__last_{window}s",
                    f"{prefix}__change_{window}s",
                    f"{prefix}__slope_{window}s",
                    f"{prefix}__mean_abs_diff_{window}s",
                    f"{prefix}__max_abs_diff_{window}s",
                    f"{prefix}__cv_{window}s",
                    f"{prefix}__sign_changes_{window}s",
                    f"{prefix}__positive_count_{window}s",
                    f"{prefix}__negative_count_{window}s",
                )
            )
    names.extend(
        (
            "physics__pv_power_pu",
            "physics__daylight_flag",
            "physics__pv_to_poa_ratio",
            "physics__pv_to_ghi_ratio",
            "physics__poa_to_ghi_ratio",
            "physics__pv_saturation_margin_pu",
            "physics__temperature_x_poa",
            "physics__wind_cooling_proxy",
            "physics__poa_variability_60s",
            "physics__pv_variability_60s",
        )
    )
    return names


def _window_mean_std(values: np.ndarray, anchors: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(values * values, dtype=np.float64)))
    starts = anchors - window + 1
    sums = prefix[anchors + 1] - prefix[starts]
    sums_sq = prefix_sq[anchors + 1] - prefix_sq[starts]
    means = sums / float(window)
    variance = np.maximum(0.0, sums_sq / float(window) - means * means)
    return means, np.sqrt(variance)


def _extended_window_descriptors(values: np.ndarray, anchors: np.ndarray, window: int) -> tuple[np.ndarray, ...]:
    offsets = np.arange(window - 1, -1, -1, dtype=np.int64)
    matrix = values[anchors[:, None] - offsets[None, :]].astype(np.float64, copy=False)
    changes = np.diff(matrix, axis=1)
    signs = np.sign(changes)
    minimum = np.min(matrix, axis=1)
    maximum = np.max(matrix, axis=1)
    mean = np.mean(matrix, axis=1)
    std = np.std(matrix, axis=1)
    return (
        minimum,
        maximum,
        maximum - minimum,
        matrix[:, 0],
        matrix[:, -1],
        matrix[:, -1] - matrix[:, 0],
        _ols_slopes(matrix),
        np.mean(np.abs(changes), axis=1),
        np.max(np.abs(changes), axis=1),
        std / (np.abs(mean) + 1.0e-6),
        np.sum(signs[:, 1:] * signs[:, :-1] < 0, axis=1),
        np.sum(changes > 0, axis=1),
        np.sum(changes < 0, axis=1),
    )


def compute_second_anchor_features(
    values: np.ndarray,
    anchors: np.ndarray,
    available_columns: Sequence[str],
    selected_columns: Sequence[str],
    lags: Sequence[int],
    windows: Sequence[int],
    extended_windows: Sequence[int],
    rated_power_kw: float,
) -> tuple[np.ndarray, list[str]]:
    index = {column: position for position, column in enumerate(available_columns)}
    names = second_feature_names(selected_columns, lags, windows, extended_windows)
    output = np.empty((len(anchors), len(names)), dtype=np.float32)
    cursor = 0
    for column in selected_columns:
        output[:, cursor] = values[anchors, index[column]].astype(np.float32)
        cursor += 1
    cached: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for prefix, column in SECOND_VARIABLES.items():
        source = values[:, index[column]].astype(np.float64, copy=False)
        for lag in lags:
            output[:, cursor] = source[anchors - lag].astype(np.float32)
            cursor += 1
        cached[prefix] = {}
        for window in windows:
            mean, std = _window_mean_std(source, anchors, int(window))
            cached[prefix][int(window)] = (mean, std)
            output[:, cursor] = mean.astype(np.float32)
            output[:, cursor + 1] = std.astype(np.float32)
            cursor += 2
        for window in extended_windows:
            for descriptor in _extended_window_descriptors(source, anchors, int(window)):
                output[:, cursor] = np.asarray(descriptor, dtype=np.float32)
                cursor += 1
    pv = values[anchors, index[SECOND_VARIABLES["pv"]]].astype(np.float64)
    poa = values[anchors, index[SECOND_VARIABLES["poa"]]].astype(np.float64)
    ghi = values[anchors, index[SECOND_VARIABLES["ghi"]]].astype(np.float64)
    elevation = values[anchors, index["Elevation[deg]"]].astype(np.float64)
    temperature = values[anchors, index["TEMPERATURE[degC]"]].astype(np.float64)
    wind = values[anchors, index["WIND_SPEED[m1s]"]].astype(np.float64)
    safe_poa = np.where(np.abs(poa) > 1.0e-6, poa, np.nan)
    safe_ghi = np.where(np.abs(ghi) > 1.0e-6, ghi, np.nan)
    poa_mean60, poa_std60 = cached["poa"][60]
    pv_mean60, pv_std60 = cached["pv"][60]
    physics = (
        pv / rated_power_kw,
        (elevation > 0.0).astype(np.float64),
        np.nan_to_num(pv / safe_poa, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(pv / safe_ghi, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(poa / safe_ghi, nan=0.0, posinf=0.0, neginf=0.0),
        np.maximum(0.0, 1.0 - pv / rated_power_kw),
        temperature * poa,
        wind * np.maximum(0.0, temperature - 25.0),
        poa_std60 / (np.abs(poa_mean60) + 1.0e-6),
        pv_std60 / (np.abs(pv_mean60) + 1.0e-6),
    )
    for feature in physics:
        output[:, cursor] = np.asarray(feature, dtype=np.float32)
        cursor += 1
    if cursor != len(names):
        raise AssertionError(f"Second feature cursor {cursor} does not match schema {len(names)}")
    if not np.isfinite(output).all():
        raise ValueError("Non-finite values detected in second-level feature matrix.")
    return output, names


@dataclass(frozen=True)
class SplitBoundary:
    name: str
    start: pd.Timestamp
    end_exclusive: pd.Timestamp


def boundaries_from_minute_manifest(manifest: dict[str, Any], raw_end: pd.Timestamp) -> list[SplitBoundary]:
    summaries = manifest["split_summary"]
    return [
        SplitBoundary("train", pd.Timestamp(summaries["train"]["start_timestamp"]), pd.Timestamp(summaries["validation"]["start_timestamp"])),
        SplitBoundary("validation", pd.Timestamp(summaries["validation"]["start_timestamp"]), pd.Timestamp(summaries["test"]["start_timestamp"])),
        SplitBoundary("test", pd.Timestamp(summaries["test"]["start_timestamp"]), raw_end + pd.Timedelta(seconds=1)),
    ]


def _split_for_timestamp(timestamp: pd.Timestamp, boundaries: Sequence[SplitBoundary]) -> SplitBoundary | None:
    for boundary in boundaries:
        if boundary.start <= timestamp < boundary.end_exclusive:
            return boundary
    return None


def build_second_anchor_parts(
    reader: FixedFrameHDFReader,
    parts_dir: Path,
    manifest: dict[str, Any],
    selected_columns: Sequence[str],
    rated_power_kw: float,
    anchor_interval_seconds: int,
    physical_history_seconds: int,
    lags: Sequence[int],
    windows: Sequence[int],
    extended_windows: Sequence[int],
    chunk_rows: int,
) -> dict[str, Any]:
    parts_dir.mkdir(parents=True, exist_ok=True)
    all_columns = reader.columns
    for required in list(selected_columns) + list(SECOND_VARIABLES.values()):
        if required not in all_columns:
            raise KeyError(f"Required one-second column is absent: {required}")
    max_history = max([physical_history_seconds, *lags, *windows, *extended_windows])
    future = 60
    first_ns = int(reader.index[0])
    boundaries = boundaries_from_minute_manifest(manifest, reader.timestamp_at(reader.row_count - 1))
    records: list[dict[str, Any]] = []
    names: list[str] | None = None
    started = time.perf_counter()
    with PeakMemoryMonitor() as memory:
        for part_id, core_start in enumerate(range(0, reader.row_count, chunk_rows)):
            core_stop = min(reader.row_count, core_start + chunk_rows)
            part_path = parts_dir / f"part_{part_id:04d}.parquet"
            if part_path.exists():
                metadata = pq.ParquetFile(part_path).metadata
                records.append({"part": part_id, "path": str(part_path), "rows": int(metadata.num_rows), "reused": True})
                continue
            read_start = max(0, core_start - max_history)
            read_stop = min(reader.row_count, core_stop + future + 1)
            timestamps, values = reader.read(read_start, read_stop)
            global_candidates = np.arange(core_start, core_stop, dtype=np.int64)
            candidate_ns = np.asarray(reader.index[core_start:core_stop], dtype=np.int64)
            aligned = ((candidate_ns - first_ns) // 1_000_000_000) % int(anchor_interval_seconds) == 0
            global_candidates = global_candidates[aligned]
            candidate_ns = candidate_ns[aligned]
            keep = np.zeros(len(global_candidates), dtype=bool)
            split_codes = np.full(len(global_candidates), -1, dtype=np.int8)
            s1_valid = np.zeros(len(global_candidates), dtype=bool)
            s2_valid = np.zeros(len(global_candidates), dtype=bool)
            for boundary in boundaries:
                start_ns = int(boundary.start.value)
                end_ns = int(boundary.end_exclusive.value)
                in_split = (candidate_ns >= start_ns + max_history * 1_000_000_000) & (candidate_ns < end_ns)
                keep |= in_split
                split_codes[in_split] = SPLIT_CODES[boundary.name]
                s1_valid[in_split] = candidate_ns[in_split] + 1_000_000_000 < end_ns
                s2_valid[in_split] = candidate_ns[in_split] + 60_000_000_000 < end_ns
            global_candidates = global_candidates[keep]
            candidate_ns = candidate_ns[keep]
            split_codes = split_codes[keep]
            s1_valid = s1_valid[keep]
            s2_valid = s2_valid[keep]
            if not len(global_candidates):
                continue
            local = global_candidates - read_start
            x, names = compute_second_anchor_features(
                values,
                local,
                all_columns,
                selected_columns,
                lags,
                windows,
                extended_windows,
                rated_power_kw,
            )
            pv_index = reader.column_indices[SECOND_VARIABLES["pv"]]
            power = values[:, pv_index].astype(np.float64, copy=False)
            current_ramp = (power[local] - power[local - 1]) / rated_power_kw
            y_s1 = np.full(len(local), np.nan, dtype=np.float64)
            y_s2 = np.full(len(local), np.nan, dtype=np.float64)
            y_s1[s1_valid] = (power[local[s1_valid] + 1] - power[local[s1_valid]]) / rated_power_kw
            y_s2[s2_valid] = (power[local[s2_valid] + 60] - power[local[s2_valid] + 59]) / rated_power_kw
            elevation = values[local, reader.column_indices["Elevation[deg]"]]
            poa_var_column = names.index("physics__poa_variability_60s")
            payload: dict[str, Any] = {
                TIME_COLUMN: candidate_ns.astype("datetime64[ns]"),
                "split_code": split_codes,
                "y_s1": y_s1,
                "y_s2": y_s2,
                "current_ramp": current_ramp,
                "solar_elevation_deg": elevation.astype(np.float32),
                "irradiance_variability_60s": x[:, poa_var_column].astype(np.float32),
            }
            for index, name in enumerate(names):
                payload[name] = x[:, index]
            table = pa.Table.from_pydict(payload)
            pq.write_table(table, part_path, compression="zstd", row_group_size=50_000)
            records.append({"part": part_id, "path": str(part_path), "rows": len(global_candidates), "reused": False})
            del table, payload, x, values, timestamps
    if names is None:
        for record in records:
            schema = pq.read_schema(record["path"])
            names = [name for name in schema.names if name.startswith(("current__", "pv__", "poa__", "ghi__", "physics__"))]
            break
    return {
        "parts": records,
        "part_count": len(records),
        "rows": int(sum(record["rows"] for record in records)),
        "feature_names": names or [],
        "feature_count": len(names or []),
        "anchor_interval_seconds": int(anchor_interval_seconds),
        "retained_fraction_of_one_second_rows": float(sum(record["rows"] for record in records) / reader.row_count),
        "history_seconds": int(max_history),
        "generation_time_seconds": time.perf_counter() - started,
        "peak_memory_gb": memory.peak_gb,
        "split_boundaries": [
            {"split": boundary.name, "start": boundary.start.isoformat(), "end_exclusive": boundary.end_exclusive.isoformat()}
            for boundary in boundaries
        ],
    }


def second_anchor_sampling_audit(parts_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize theoretical full-second and retained deterministic anchors."""
    targets = {"S1": ("y_s1", 1), "S2": ("y_s2", 60)}
    retained: dict[str, dict[str, dict[str, Any]]] = {
        target: {
            split: {"count": 0, "first": None, "last": None}
            for split in SPLIT_NAMES
        }
        for target in targets
    }
    for record in sorted(parts_manifest["parts"], key=lambda item: int(item["part"])):
        parquet = pq.ParquetFile(record["path"])
        for batch in parquet.iter_batches(
            batch_size=50_000,
            columns=[TIME_COLUMN, "split_code", "y_s1", "y_s2"],
        ):
            frame = batch.to_pandas()
            timestamps = frame[TIME_COLUMN].to_numpy(dtype="datetime64[ns]").astype(np.int64)
            split_codes = frame["split_code"].to_numpy(dtype=np.int8, copy=False)
            for target, (column, _) in targets.items():
                finite = np.isfinite(frame[column].to_numpy(dtype=np.float64, copy=False))
                for split, code in SPLIT_CODES.items():
                    selected = timestamps[finite & (split_codes == code)]
                    if not len(selected):
                        continue
                    state = retained[target][split]
                    state["count"] += int(len(selected))
                    current_first = int(np.min(selected))
                    current_last = int(np.max(selected))
                    state["first"] = current_first if state["first"] is None else min(state["first"], current_first)
                    state["last"] = current_last if state["last"] is None else max(state["last"], current_last)

    boundaries = {
        item["split"]: (pd.Timestamp(item["start"]), pd.Timestamp(item["end_exclusive"]))
        for item in parts_manifest["split_boundaries"]
    }
    history_seconds = int(parts_manifest["history_seconds"])
    rows: list[dict[str, Any]] = []
    classification = (
        "Resource-bounded second-level pilot experiments using deterministic "
        "30-second anchors across the full chronological period."
    )
    for target, (_, lead_seconds) in targets.items():
        theoretical_by_split: dict[str, int] = {}
        for split in SPLIT_NAMES:
            start, end_exclusive = boundaries[split]
            duration_seconds = int((end_exclusive - start).total_seconds())
            theoretical_by_split[split] = max(0, duration_seconds - history_seconds - lead_seconds)
        retained_total = int(sum(retained[target][split]["count"] for split in SPLIT_NAMES))
        theoretical_total = int(sum(theoretical_by_split.values()))
        first_values = [retained[target][split]["first"] for split in SPLIT_NAMES if retained[target][split]["first"] is not None]
        last_values = [retained[target][split]["last"] for split in SPLIT_NAMES if retained[target][split]["last"] is not None]
        rows.append(
            {
                "target_id": target,
                "classification": classification,
                "theoretically_available_anchors": theoretical_total,
                "retained_anchors": retained_total,
                "retained_percentage": 100.0 * retained_total / theoretical_total if theoretical_total else float("nan"),
                "theoretical_train_anchors": theoretical_by_split["train"],
                "theoretical_validation_anchors": theoretical_by_split["validation"],
                "theoretical_test_anchors": theoretical_by_split["test"],
                "train_anchors": int(retained[target]["train"]["count"]),
                "validation_anchors": int(retained[target]["validation"]["count"]),
                "test_anchors": int(retained[target]["test"]["count"]),
                "first_timestamp": pd.Timestamp(min(first_values)).isoformat() if first_values else None,
                "last_timestamp": pd.Timestamp(max(last_values)).isoformat() if last_values else None,
                "sampling_interval_seconds": int(parts_manifest["anchor_interval_seconds"]),
                "same_sampling_policy_train_validation_test": True,
                "full_second_test_evaluation": False,
            }
        )
    return rows


def materialize_second_memmaps(parts_manifest: dict[str, Any], temporary_dir: Path) -> dict[str, Any]:
    temporary_dir.mkdir(parents=True, exist_ok=True)
    feature_names = list(parts_manifest["feature_names"])
    total_rows = int(parts_manifest["rows"])
    x_path = temporary_dir / "second_anchor_features.npy"
    metadata_paths = {
        "timestamps": temporary_dir / "second_anchor_timestamps.npy",
        "split_codes": temporary_dir / "second_anchor_split_codes.npy",
        "y_s1": temporary_dir / "second_anchor_y_s1.npy",
        "y_s2": temporary_dir / "second_anchor_y_s2.npy",
        "current_ramp": temporary_dir / "second_anchor_current_ramp.npy",
        "solar_elevation_deg": temporary_dir / "second_anchor_solar_elevation.npy",
        "irradiance_variability_60s": temporary_dir / "second_anchor_irradiance_variability.npy",
    }
    # HGB validates inputs as float64. A float64 disk-backed matrix prevents an
    # otherwise unavoidable full in-memory conversion of this resource-bounded run.
    x = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float64, shape=(total_rows, len(feature_names)))
    metadata_arrays = {
        "timestamps": np.lib.format.open_memmap(metadata_paths["timestamps"], mode="w+", dtype=np.int64, shape=(total_rows,)),
        "split_codes": np.lib.format.open_memmap(metadata_paths["split_codes"], mode="w+", dtype=np.int8, shape=(total_rows,)),
        "y_s1": np.lib.format.open_memmap(metadata_paths["y_s1"], mode="w+", dtype=np.float64, shape=(total_rows,)),
        "y_s2": np.lib.format.open_memmap(metadata_paths["y_s2"], mode="w+", dtype=np.float64, shape=(total_rows,)),
        "current_ramp": np.lib.format.open_memmap(metadata_paths["current_ramp"], mode="w+", dtype=np.float64, shape=(total_rows,)),
        "solar_elevation_deg": np.lib.format.open_memmap(metadata_paths["solar_elevation_deg"], mode="w+", dtype=np.float32, shape=(total_rows,)),
        "irradiance_variability_60s": np.lib.format.open_memmap(metadata_paths["irradiance_variability_60s"], mode="w+", dtype=np.float32, shape=(total_rows,)),
    }
    cursor = 0
    columns = [
        TIME_COLUMN,
        "split_code",
        "y_s1",
        "y_s2",
        "current_ramp",
        "solar_elevation_deg",
        "irradiance_variability_60s",
        *feature_names,
    ]
    for record in sorted(parts_manifest["parts"], key=lambda item: int(item["part"])):
        parquet = pq.ParquetFile(record["path"])
        for batch in parquet.iter_batches(batch_size=50_000, columns=columns):
            frame = batch.to_pandas()
            count = len(frame)
            x[cursor : cursor + count] = frame[feature_names].to_numpy(dtype=np.float64, copy=False)
            metadata_arrays["timestamps"][cursor : cursor + count] = frame[TIME_COLUMN].to_numpy(dtype="datetime64[ns]").astype(np.int64)
            for name in metadata_arrays:
                if name == "timestamps":
                    continue
                source_name = "split_code" if name == "split_codes" else name
                metadata_arrays[name][cursor : cursor + count] = frame[source_name].to_numpy()
            cursor += count
    if cursor != total_rows:
        raise AssertionError(f"Materialized {cursor} rows, expected {total_rows}")
    x.flush()
    for array in metadata_arrays.values():
        array.flush()
    split_counts = {
        split: int(np.sum(np.asarray(metadata_arrays["split_codes"]) == code))
        for split, code in SPLIT_CODES.items()
    }
    return {
        "feature_path": str(x_path),
        "metadata_paths": {name: str(path) for name, path in metadata_paths.items()},
        "shape": [total_rows, len(feature_names)],
        "dtype": "float64_disk_backed_for_sklearn_hgb",
        "feature_names": feature_names,
        "split_counts": split_counts,
    }


def load_npy_memmap(path: Path | str, mode: str = "r") -> np.ndarray:
    return np.load(str(path), mmap_mode=mode)


def contiguous_experiment_slices(
    split_codes: np.ndarray,
    target: np.ndarray,
) -> dict[str, slice]:
    result: dict[str, slice] = {}
    for split_name, code in SPLIT_CODES.items():
        indices = np.flatnonzero((split_codes == code) & np.isfinite(target))
        if len(indices) == 0:
            raise ValueError(f"No valid rows for {split_name}")
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError(f"Rows for {split_name} are not contiguous")
        result[split_name] = slice(int(indices[0]), int(indices[-1]) + 1)
    return result


def make_hgb(parameters: dict[str, Any]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**parameters)


def fit_hgb_with_disk_backed_internal_validation(
    model: HistGradientBoostingRegressor,
    matrix: np.ndarray,
    target: np.ndarray,
    cache_dir: Path,
) -> dict[str, Any]:
    """Materialize HGB's seeded early-stopping split without an in-RAM copy."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    row_count, feature_count = matrix.shape
    random_state = check_random_state(model.random_state)
    internal_seed = int(random_state.randint(np.iinfo(np.uint32).max, dtype="u8"))
    all_indices = np.arange(row_count, dtype=np.int64)
    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=model.validation_fraction,
        random_state=internal_seed,
    )
    paths = {
        "train_x": cache_dir / "internal_train_x.npy",
        "validation_x": cache_dir / "internal_validation_x.npy",
        "train_y": cache_dir / "internal_train_y.npy",
        "validation_y": cache_dir / "internal_validation_y.npy",
    }
    train_x = np.lib.format.open_memmap(paths["train_x"], mode="w+", dtype=np.float64, shape=(len(train_indices), feature_count))
    validation_x = np.lib.format.open_memmap(paths["validation_x"], mode="w+", dtype=np.float64, shape=(len(validation_indices), feature_count))
    train_y = np.lib.format.open_memmap(paths["train_y"], mode="w+", dtype=np.float64, shape=(len(train_indices),))
    validation_y = np.lib.format.open_memmap(paths["validation_y"], mode="w+", dtype=np.float64, shape=(len(validation_indices),))
    for start in range(0, len(train_indices), 10_000):
        stop = min(len(train_indices), start + 10_000)
        indices = train_indices[start:stop]
        train_x[start:stop] = matrix[indices]
        train_y[start:stop] = target[indices]
    for start in range(0, len(validation_indices), 10_000):
        stop = min(len(validation_indices), start + 10_000)
        indices = validation_indices[start:stop]
        validation_x[start:stop] = matrix[indices]
        validation_y[start:stop] = target[indices]
    for array in (train_x, validation_x, train_y, validation_y):
        array.flush()
    started = time.perf_counter()
    with PeakMemoryMonitor() as memory:
        model.fit(train_x, train_y, X_val=validation_x, y_val=validation_y)
    elapsed = time.perf_counter() - started
    metadata = {
        "scikit_learn_version": sklearn.__version__,
        "fit_time_seconds": elapsed,
        "peak_memory_gb": memory.peak_gb,
        "original_training_rows": int(row_count),
        "internal_fit_rows": int(len(train_indices)),
        "internal_early_stopping_rows": int(len(validation_indices)),
        "validation_fraction": float(model.validation_fraction),
        "internal_split_seed": internal_seed,
        "configured_random_state": model.random_state,
        "split_construction": "random 90/10 split of the official training subset only",
        "official_validation_rows_used_for_early_stopping": 0,
        "test_rows_used_for_fitting_or_early_stopping": 0,
        "matches_current_sklearn_seeded_internal_split_mechanics": True,
        "exact_original_stage_c_reproduction_verified": False,
        "policy": "seeded HGB 90/10 split materialized on disk and supplied through X_val/y_val under the recorded scikit-learn version",
    }
    del train_x, validation_x, train_y, validation_y
    gc.collect()
    deferred_cleanup_paths: list[str] = []
    for path in paths.values():
        try:
            safe_remove(path)
        except PermissionError:
            deferred_cleanup_paths.append(str(path))
    metadata["deferred_cleanup_paths"] = deferred_cleanup_paths
    try:
        cache_dir.rmdir()
    except OSError:
        pass
    return metadata


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    persistence_pred: np.ndarray,
    nrmse_denominator: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    persistence_pred = np.asarray(persistence_pred, dtype=np.float64)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    persistence_rmse = float(np.sqrt(mean_squared_error(y_true, persistence_pred)))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "NRMSE": rmse / float(nrmse_denominator),
        "R2": float(r2_score(y_true, y_pred)),
        "persistence_RMSE": persistence_rmse,
        "persistence_skill": 1.0 - rmse / persistence_rmse if persistence_rmse else float("nan"),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


def binary_event_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "F1": f1, "tp": tp, "fp": fp, "fn": fn}


def event_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    strong = np.abs(y_true) >= threshold
    upward = y_true >= threshold
    downward = y_true <= -threshold
    combined = binary_event_metrics(strong, np.abs(y_pred) >= threshold)
    up = binary_event_metrics(upward, y_pred >= threshold)
    down = binary_event_metrics(downward, y_pred <= -threshold)
    return {
        "threshold": float(threshold),
        "strong_count": int(np.sum(strong)),
        "strong_fraction": float(np.mean(strong)),
        "strong_ramp_MAE": float(mean_absolute_error(y_true[strong], y_pred[strong])) if np.any(strong) else float("nan"),
        "strong_ramp_RMSE": float(np.sqrt(mean_squared_error(y_true[strong], y_pred[strong]))) if np.any(strong) else float("nan"),
        "strong_ramp_precision": combined["precision"],
        "strong_ramp_recall": combined["recall"],
        "strong_ramp_F1": combined["F1"],
        "upward_precision": up["precision"],
        "upward_recall": up["recall"],
        "upward_F1": up["F1"],
        "downward_precision": down["precision"],
        "downward_recall": down["recall"],
        "downward_F1": down["F1"],
        "sign_accuracy": directional_accuracy(y_true, y_pred),
    }


def magnitude_edges(training_target: np.ndarray) -> np.ndarray:
    absolute = np.abs(np.asarray(training_target, dtype=np.float64))
    values = np.unique(np.quantile(absolute, [0.0, 0.5, 0.9, 0.95, 0.99]))
    if len(values) < 2:
        values = np.array([0.0, float(np.max(absolute) + 1.0e-12)])
    return np.concatenate((values, [np.inf]))


def magnitude_labels(edges: np.ndarray) -> list[str]:
    labels = []
    for left, right in zip(edges[:-1], edges[1:]):
        if np.isinf(right):
            labels.append(f">={left:.6g}")
        else:
            labels.append(f"[{left:.6g}, {right:.6g})")
    return labels


def build_prediction_frame(
    timestamps_ns: np.ndarray,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    magnitude_bin_edges: np.ndarray,
    context: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    residual = np.asarray(y_true) - np.asarray(y_pred)
    absolute = np.abs(np.asarray(y_true))
    labels = magnitude_labels(magnitude_bin_edges)
    bins = np.clip(np.digitize(absolute, magnitude_bin_edges[1:-1], right=False), 0, len(labels) - 1)
    event = np.where(y_true >= threshold, "strong_up", np.where(y_true <= -threshold, "strong_down", "non_strong"))
    payload: dict[str, Any] = {
        TIME_COLUMN: np.asarray(timestamps_ns, dtype=np.int64).astype("datetime64[ns]"),
        "split": split,
        "y_true": np.asarray(y_true, dtype=np.float64),
        "y_pred": np.asarray(y_pred, dtype=np.float64),
        "residual": residual,
        "absolute_error": np.abs(residual),
        "squared_error": residual * residual,
        "target_magnitude_bin": np.asarray(labels, dtype=object)[bins],
        "event_category": event,
    }
    if context:
        payload.update(context)
    return pd.DataFrame(payload)


def save_prediction_frames(path: Path, frames: Sequence[pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(path, index=False, compression="zstd")


def run_second_experiment(
    experiment_id: str,
    memmap_manifest: dict[str, Any],
    target_key: str,
    output_paths: dict[str, Path],
    hgb_parameters: dict[str, Any],
    rated_power_kw: float,
    nrmse_denominator: float,
    event_percentiles: Sequence[int],
    primary_event_percentile: int,
    feature_generation_time_seconds: float,
    preprocessing_time_seconds: float,
    peak_preprocessing_memory_gb: float | None,
    model_metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x = load_npy_memmap(memmap_manifest["feature_path"])
    arrays = {name: load_npy_memmap(path) for name, path in memmap_manifest["metadata_paths"].items()}
    target = arrays[target_key]
    slices = contiguous_experiment_slices(arrays["split_codes"], target)
    train_slice = slices["train"]
    y_train = np.asarray(target[train_slice])
    thresholds = {str(percentile): float(np.percentile(np.abs(y_train), percentile)) for percentile in event_percentiles}
    primary_threshold = thresholds[str(primary_event_percentile)]
    mag_edges = magnitude_edges(y_train)
    model = make_hgb(hgb_parameters)
    fit_metadata = fit_hgb_with_disk_backed_internal_validation(
        model,
        x[train_slice],
        y_train,
        Path(memmap_manifest["feature_path"]).parent / f"{experiment_id.lower()}_hgb_fit_cache",
    )
    training_time = float(fit_metadata["fit_time_seconds"])
    predictions: dict[str, dict[str, np.ndarray | float]] = {}
    prediction_times: dict[str, float] = {}
    for split in ("validation", "test"):
        current_slice = slices[split]
        started = time.perf_counter()
        predicted = model.predict(x[current_slice])
        prediction_times[split] = time.perf_counter() - started
        predictions[split] = {
            "y_true": np.asarray(target[current_slice], dtype=np.float64),
            "y_pred": np.asarray(predicted, dtype=np.float64),
            "persistence": np.asarray(arrays["current_ramp"][current_slice], dtype=np.float64),
            "timestamps": np.asarray(arrays["timestamps"][current_slice], dtype=np.int64),
            "solar_elevation_deg": np.asarray(arrays["solar_elevation_deg"][current_slice]),
            "irradiance_variability_60s": np.asarray(arrays["irradiance_variability_60s"][current_slice]),
        }
    model_path = output_paths["models"] / f"{experiment_id.lower()}_hist_gradient_boosting.joblib"
    joblib.dump(model, model_path, compress=3)
    write_json(
        output_paths["models"] / f"{experiment_id.lower()}_hist_gradient_boosting_metadata.json",
        {
            **model_metadata,
            "experiment_id": experiment_id,
            "parameters": hgb_parameters,
            "model_path": str(model_path),
            "n_features_in": int(model.n_features_in_),
            "training_rows": int(train_slice.stop - train_slice.start),
            "event_thresholds_from_training": thresholds,
            "fit_policy": fit_metadata,
        },
    )
    result_rows: list[dict[str, Any]] = []
    prediction_frames_model: list[pd.DataFrame] = []
    prediction_frames_persistence: list[pd.DataFrame] = []
    split_metrics: dict[str, Any] = {}
    for split in ("validation", "test"):
        data = predictions[split]
        y_true = np.asarray(data["y_true"])
        y_pred = np.asarray(data["y_pred"])
        persistence = np.asarray(data["persistence"])
        metrics = regression_metrics(y_true, y_pred, persistence, nrmse_denominator)
        events = event_metrics(y_true, y_pred, primary_threshold)
        persistence_metrics = regression_metrics(y_true, persistence, persistence, nrmse_denominator)
        persistence_events = event_metrics(y_true, persistence, primary_threshold)
        split_metrics[split] = {
            "HistGradientBoosting": {**metrics, **events},
            "Persistence": {**persistence_metrics, **persistence_events},
        }
        context = {
            "solar_elevation_deg": np.asarray(data["solar_elevation_deg"]),
            "irradiance_variability_60s": np.asarray(data["irradiance_variability_60s"]),
        }
        prediction_frames_model.append(
            build_prediction_frame(data["timestamps"], split, y_true, y_pred, primary_threshold, mag_edges, context)
        )
        prediction_frames_persistence.append(
            build_prediction_frame(data["timestamps"], split, y_true, persistence, primary_threshold, mag_edges, context)
        )
    save_prediction_frames(
        output_paths["predictions"] / f"{experiment_id.lower()}_hist_gradient_boosting_predictions.parquet",
        prediction_frames_model,
    )
    save_prediction_frames(
        output_paths["predictions"] / f"{experiment_id.lower()}_persistence_predictions.parquet",
        prediction_frames_persistence,
    )
    test_metrics = split_metrics["test"]["HistGradientBoosting"]
    persistence_test = split_metrics["test"]["Persistence"]
    base = {
        "experiment_id": experiment_id,
        "target_name": "next_one_second_ramp" if experiment_id == "S1" else "one_second_ramp_60_seconds_ahead",
        "input_resolution": "one_second",
        "output_resolution": "one_second",
        "target_formula": "(p[s+1]-p[s])/10" if experiment_id == "S1" else "(p[s+60]-p[s+59])/10",
        "feature_variant": "compact_causal_one_second",
        "feature_count": int(memmap_manifest["shape"][1]),
        "lookback_steps": 480,
        "lookback_duration_seconds": 480,
        "horizon_steps": 1 if experiment_id == "S1" else 60,
        "forecast_lead_seconds": 1 if experiment_id == "S1" else 60,
        "target_interval_seconds": 1,
        "train_rows": int(slices["train"].stop - slices["train"].start),
        "validation_rows": int(slices["validation"].stop - slices["validation"].start),
        "test_rows": int(slices["test"].stop - slices["test"].start),
        "sampling_interval_used": 30,
    }
    result_rows.append(
        {
            **base,
            "model": "HistGradientBoosting",
            **{key: test_metrics.get(key) for key in ("MAE", "RMSE", "NRMSE", "R2", "persistence_RMSE", "persistence_skill", "directional_accuracy")},
            **{key: test_metrics.get(key) for key in ("strong_ramp_MAE", "strong_ramp_RMSE", "strong_ramp_precision", "strong_ramp_recall", "strong_ramp_F1")},
            "training_time_seconds": training_time,
            "validation_prediction_time_seconds": prediction_times["validation"],
            "test_prediction_time_seconds": prediction_times["test"],
            "inference_time_per_sample": prediction_times["test"] / base["test_rows"],
            "feature_generation_time_seconds": feature_generation_time_seconds,
            "end_to_end_preprocessing_time_seconds": preprocessing_time_seconds,
            "peak_memory_gb": max(value for value in (fit_metadata["peak_memory_gb"], peak_preprocessing_memory_gb) if value is not None),
            "execution_status": "success_resource_bounded_pilot",
            "notes": "Resource-bounded second-level pilot experiment using deterministic 30-second anchors across the full chronological period; no hyperparameter tuning.",
        }
    )
    result_rows.append(
        {
            **base,
            "model": "Persistence",
            **{key: persistence_test.get(key) for key in ("MAE", "RMSE", "NRMSE", "R2", "persistence_RMSE", "persistence_skill", "directional_accuracy")},
            **{key: persistence_test.get(key) for key in ("strong_ramp_MAE", "strong_ramp_RMSE", "strong_ramp_precision", "strong_ramp_recall", "strong_ramp_F1")},
            "training_time_seconds": 0.0,
            "validation_prediction_time_seconds": 0.0,
            "test_prediction_time_seconds": 0.0,
            "inference_time_per_sample": 0.0,
            "feature_generation_time_seconds": 0.0,
            "end_to_end_preprocessing_time_seconds": 0.0,
            "peak_memory_gb": None,
            "execution_status": "success",
            "notes": "Target-specific persistence: the ramp observed at forecast origin is carried forward.",
        }
    )
    return result_rows, {
        "experiment_id": experiment_id,
        "event_thresholds": thresholds,
        "primary_event_threshold": primary_threshold,
        "magnitude_bin_edges": mag_edges.tolist(),
        "split_metrics": split_metrics,
        "model_path": str(model_path),
        "training_time_seconds": training_time,
        "prediction_times_seconds": prediction_times,
        "peak_training_memory_gb": fit_metadata["peak_memory_gb"],
        "fit_policy": fit_metadata,
        "slices": {name: [value.start, value.stop] for name, value in slices.items()},
    }


def materialize_official_minute_data(manifest: dict[str, Any], temporary_dir: Path) -> dict[str, Any]:
    temporary_dir.mkdir(parents=True, exist_ok=True)
    row_count = int(manifest["row_count"])
    feature_names = list(manifest["feature_columns"])
    source_path = Path(manifest["source_path"])
    scaler = joblib.load(manifest["scaler_file"])
    feature_path = temporary_dir / "official_minute_scaled_features.npy"
    target_path = temporary_dir / "official_minute_target.npy"
    timestamp_path = temporary_dir / "official_minute_timestamps.npy"
    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(row_count, len(feature_names)),
    )
    target = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.float64, shape=(row_count,))
    timestamps = np.lib.format.open_memmap(timestamp_path, mode="w+", dtype=np.int64, shape=(row_count,))
    usecols = [TIME_COLUMN, *feature_names, TARGET_COLUMN]
    cursor = 0
    started = time.perf_counter()
    with PeakMemoryMonitor() as memory:
        for chunk in pd.read_csv(source_path, usecols=usecols, chunksize=40_000):
            count = len(chunk)
            parsed = pd.to_datetime(chunk[TIME_COLUMN], errors="raise")
            raw = chunk[feature_names].to_numpy(dtype=np.float64, copy=True)
            scaled = scaler.transform(raw).astype(np.float32, copy=False)
            values = pd.to_numeric(chunk[TARGET_COLUMN], errors="raise").to_numpy(dtype=np.float64)
            if not np.isfinite(scaled).all() or not np.isfinite(values).all():
                raise ValueError("Non-finite official minute data encountered during materialization.")
            features[cursor : cursor + count] = scaled
            target[cursor : cursor + count] = values
            timestamps[cursor : cursor + count] = parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64)
            cursor += count
            del chunk, raw, scaled, values
    if cursor != row_count:
        raise AssertionError(f"Official minute materialization wrote {cursor}, expected {row_count}")
    features.flush()
    target.flush()
    timestamps.flush()
    if pd.Timestamp(int(timestamps[0])).isoformat() != pd.Timestamp(manifest["split_summary"]["train"]["start_timestamp"]).isoformat():
        raise ValueError("Official minute start timestamp does not match the split manifest.")
    return {
        "feature_path": str(feature_path),
        "target_path": str(target_path),
        "timestamp_path": str(timestamp_path),
        "shape": [row_count, len(feature_names)],
        "feature_names": feature_names,
        "split_summary": manifest["split_summary"],
        "materialization_time_seconds": time.perf_counter() - started,
        "peak_memory_gb": memory.peak_gb,
        "source_path": str(source_path),
        "scaler_path": manifest["scaler_file"],
    }


def minute_descriptor_feature_names() -> list[str]:
    return [f"{prefix}_{stat}" for prefix in SECOND_VARIABLES for stat in MINUTE_BASE_STATS] + ["pv_cancellation_flag"]


def build_high_frequency_minute_features(
    descriptor_path: Path,
    official_timestamps_path: Path,
    temporary_dir: Path,
    context_lags: Sequence[int],
    context_window: int,
) -> dict[str, Any]:
    temporary_dir.mkdir(parents=True, exist_ok=True)
    base_names = minute_descriptor_feature_names()
    audit_names = [
        "seconds_observed",
        "valid_seconds_observed",
        "complete_minute",
        "excluded_reason",
    ]
    columns = [TIME_COLUMN, *base_names, *audit_names]
    started = time.perf_counter()
    with PeakMemoryMonitor() as memory:
        descriptors = pd.read_parquet(descriptor_path, columns=columns)
        descriptors[TIME_COLUMN] = pd.to_datetime(descriptors[TIME_COLUMN], errors="raise")
        descriptors = descriptors.sort_values(TIME_COLUMN).drop_duplicates(TIME_COLUMN, keep="last").set_index(TIME_COLUMN)
        context_sources = [
            "pv_std",
            "pv_range",
            "pv_max_abs_ramp",
            "pv_mean_abs_ramp",
            "pv_slope",
            "poa_std",
            "poa_range",
            "poa_max_abs_ramp",
            "ghi_std",
            "ghi_range",
            "ghi_max_abs_ramp",
            "pv_cancellation_flag",
        ]
        context_names: list[str] = []
        for source in context_sources:
            for lag in context_lags:
                name = f"{source}_lag_{lag}min"
                descriptors[name] = descriptors[source].shift(int(lag))
                context_names.append(name)
            mean_name = f"{source}_roll_mean_{context_window}min"
            std_name = f"{source}_roll_std_{context_window}min"
            descriptors[mean_name] = descriptors[source].rolling(int(context_window), min_periods=int(context_window)).mean()
            descriptors[std_name] = descriptors[source].rolling(int(context_window), min_periods=int(context_window)).std(ddof=0)
            context_names.extend((mean_name, std_name))
        official_timestamps = load_npy_memmap(official_timestamps_path)
        official_index = pd.DatetimeIndex(np.asarray(official_timestamps, dtype=np.int64).astype("datetime64[ns]"))
        selected_names = [*base_names, *context_names]
        aligned = descriptors.reindex(official_index)[selected_names]
        if aligned.isna().any().any():
            missing = aligned.columns[aligned.isna().any()].tolist()
            raise ValueError(f"High-frequency minute descriptors are not fully aligned: {missing[:10]}")
        output_path = temporary_dir / "high_frequency_minute_features.npy"
        output = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(aligned), len(selected_names)),
        )
        for start in range(0, len(aligned), 50_000):
            stop = min(len(aligned), start + 50_000)
            output[start:stop] = aligned.iloc[start:stop].to_numpy(dtype=np.float32, copy=False)
        output.flush()
        aligned_audit = descriptors.reindex(official_index)[audit_names]
        if aligned_audit.isna().any().any():
            raise ValueError("Minute completeness fields are not fully aligned to official timestamps.")
        complete_path = temporary_dir / "official_minute_complete_mask.npy"
        seconds_path = temporary_dir / "official_minute_seconds_observed.npy"
        valid_seconds_path = temporary_dir / "official_minute_valid_seconds_observed.npy"
        complete_output = np.lib.format.open_memmap(
            complete_path,
            mode="w+",
            dtype=np.bool_,
            shape=(len(aligned_audit),),
        )
        seconds_output = np.lib.format.open_memmap(
            seconds_path,
            mode="w+",
            dtype=np.int16,
            shape=(len(aligned_audit),),
        )
        valid_seconds_output = np.lib.format.open_memmap(
            valid_seconds_path,
            mode="w+",
            dtype=np.int16,
            shape=(len(aligned_audit),),
        )
        complete_output[:] = aligned_audit["complete_minute"].to_numpy(dtype=bool, copy=False)
        seconds_output[:] = aligned_audit["seconds_observed"].to_numpy(dtype=np.int16, copy=False)
        valid_seconds_output[:] = aligned_audit["valid_seconds_observed"].to_numpy(dtype=np.int16, copy=False)
        complete_output.flush()
        seconds_output.flush()
        valid_seconds_output.flush()
    return {
        "feature_path": str(output_path),
        "shape": [len(aligned), len(selected_names)],
        "feature_names": selected_names,
        "base_descriptor_count": len(base_names),
        "cross_minute_context_count": len(context_names),
        "context_lags_minutes": list(context_lags),
        "context_window_minutes": int(context_window),
        "complete_minute_mask_path": str(complete_path),
        "seconds_observed_path": str(seconds_path),
        "valid_seconds_observed_path": str(valid_seconds_path),
        "official_timestamp_rows": int(len(aligned_audit)),
        "complete_minute_rows": int(aligned_audit["complete_minute"].astype(bool).sum()),
        "incomplete_minute_rows": int((~aligned_audit["complete_minute"].astype(bool)).sum()),
        "completeness_requirement": "exactly 60 timestamped one-second rows and 60 finite PV/POA/GHI observations",
        "generation_time_seconds": time.perf_counter() - started,
        "peak_memory_gb": memory.peak_gb,
    }


def minute_sample_count(row_count: int, lookback: int, horizon: int) -> int:
    return max(0, int(row_count) - int(lookback) - int(horizon) + 1)


def build_minute_window_memmap(
    official_manifest: dict[str, Any],
    high_frequency_manifest: dict[str, Any] | None,
    temporary_dir: Path,
    lookback: int,
    horizon: int,
    label: str,
    completeness_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    temporary_dir.mkdir(parents=True, exist_ok=True)
    minute_features = load_npy_memmap(official_manifest["feature_path"])
    minute_target = load_npy_memmap(official_manifest["target_path"])
    minute_timestamps = load_npy_memmap(official_manifest["timestamp_path"])
    high_frequency = None
    high_frequency_names: list[str] = []
    if high_frequency_manifest is not None:
        high_frequency = load_npy_memmap(high_frequency_manifest["feature_path"])
        high_frequency_names = list(high_frequency_manifest["feature_names"])
    base_dimension = int(minute_features.shape[1]) * int(lookback)
    dimension = base_dimension + (int(high_frequency.shape[1]) if high_frequency is not None else 0)
    split_summary = official_manifest["split_summary"]
    raw_split_counts = {
        split: minute_sample_count(int(split_summary[split]["row_count"]), lookback, horizon)
        for split in SPLIT_NAMES
    }
    complete_mask = None
    seconds_observed = None
    valid_seconds_observed = None
    if completeness_manifest is not None:
        complete_mask = load_npy_memmap(completeness_manifest["complete_minute_mask_path"])
        seconds_observed = load_npy_memmap(completeness_manifest["seconds_observed_path"])
        valid_seconds_observed = load_npy_memmap(completeness_manifest["valid_seconds_observed_path"])
        if len(complete_mask) != len(minute_timestamps):
            raise ValueError("Completeness mask length does not match official minute rows.")
    eligible_offsets: dict[str, np.ndarray] = {}
    split_counts: dict[str, int] = {}
    excluded_samples: list[dict[str, Any]] = []
    excluded_reason_counts: dict[str, dict[str, int]] = {}
    for split in SPLIT_NAMES:
        metadata = split_summary[split]
        split_start = int(metadata["start_row_in_dataset"])
        offsets = np.arange(raw_split_counts[split], dtype=np.int64)
        if complete_mask is None:
            eligible = np.ones(len(offsets), dtype=bool)
            input_complete = eligible.copy()
            target_complete = eligible.copy()
        else:
            row_indices = split_start + offsets[:, None] + np.arange(lookback, dtype=np.int64)[None, :]
            target_indices = split_start + offsets + lookback + horizon - 1
            input_complete = np.all(complete_mask[row_indices], axis=1)
            target_complete = np.asarray(complete_mask[target_indices], dtype=bool)
            eligible = input_complete & target_complete
        eligible_offsets[split] = offsets[eligible]
        split_counts[split] = int(np.sum(eligible))
        reason_counts = {
            "incomplete_input_minute": int(np.sum(~input_complete & target_complete)),
            "incomplete_target_minute": int(np.sum(input_complete & ~target_complete)),
            "incomplete_input_and_target_minutes": int(np.sum(~input_complete & ~target_complete)),
        }
        excluded_reason_counts[split] = reason_counts
        if np.any(~eligible):
            target_indices = split_start + offsets + lookback + horizon - 1
            for position in np.flatnonzero(~eligible):
                if not input_complete[position] and not target_complete[position]:
                    reason = "incomplete_input_and_target_minutes"
                elif not input_complete[position]:
                    reason = "incomplete_input_minute"
                else:
                    reason = "incomplete_target_minute"
                target_index = int(target_indices[position])
                excluded_samples.append(
                    {
                        "split": split,
                        "target_timestamp": pd.Timestamp(int(minute_timestamps[target_index])).isoformat(),
                        "excluded_reason": reason,
                        "target_seconds_observed": (
                            int(seconds_observed[target_index]) if seconds_observed is not None else None
                        ),
                        "target_valid_seconds_observed": (
                            int(valid_seconds_observed[target_index]) if valid_seconds_observed is not None else None
                        ),
                    }
                )
    total = int(sum(split_counts.values()))
    matrix_path = temporary_dir / f"{label}_minute_window_matrix.npy"
    matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float64, shape=(total, dimension))
    target_path = temporary_dir / f"{label}_minute_sample_target.npy"
    timestamp_path = temporary_dir / f"{label}_minute_sample_timestamps.npy"
    split_path = temporary_dir / f"{label}_minute_sample_split_codes.npy"
    persistence_path = temporary_dir / f"{label}_minute_sample_persistence.npy"
    target = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.float64, shape=(total,))
    timestamps = np.lib.format.open_memmap(timestamp_path, mode="w+", dtype=np.int64, shape=(total,))
    split_codes = np.lib.format.open_memmap(split_path, mode="w+", dtype=np.int8, shape=(total,))
    persistence = np.lib.format.open_memmap(persistence_path, mode="w+", dtype=np.float64, shape=(total,))
    context_paths: dict[str, Path] = {}
    context_arrays: dict[str, np.ndarray] = {}
    for context_name in ("pv_std", "poa_std", "ghi_std", "pv_max_abs_ramp", "pv_cancellation_flag"):
        context_path = temporary_dir / f"{label}_minute_context_{context_name}.npy"
        context_paths[context_name] = context_path
        context_arrays[context_name] = np.lib.format.open_memmap(context_path, mode="w+", dtype=np.float32, shape=(total,))
    hf_context_indices = {name: high_frequency_names.index(name) for name in context_arrays} if high_frequency_names else {}
    cursor = 0
    started = time.perf_counter()
    with PeakMemoryMonitor() as memory:
        for split in SPLIT_NAMES:
            metadata = split_summary[split]
            split_start = int(metadata["start_row_in_dataset"])
            offsets_for_split = eligible_offsets[split]
            for local_start in range(0, len(offsets_for_split), 10_000):
                local_stop = min(len(offsets_for_split), local_start + 10_000)
                offsets = offsets_for_split[local_start:local_stop]
                row_indices = split_start + offsets[:, None] + np.arange(lookback, dtype=np.int64)[None, :]
                target_indices = split_start + offsets + lookback + horizon - 1
                origin_indices = split_start + offsets + lookback - 1
                batch_count = len(offsets)
                batch = minute_features[row_indices].reshape(batch_count, base_dimension)
                matrix[cursor : cursor + batch_count, :base_dimension] = batch
                if high_frequency is not None:
                    matrix[cursor : cursor + batch_count, base_dimension:] = high_frequency[origin_indices]
                target[cursor : cursor + batch_count] = minute_target[target_indices]
                persistence[cursor : cursor + batch_count] = minute_target[origin_indices]
                timestamps[cursor : cursor + batch_count] = minute_timestamps[target_indices]
                split_codes[cursor : cursor + batch_count] = SPLIT_CODES[split]
                for context_name, context_array in context_arrays.items():
                    if high_frequency is not None:
                        context_array[cursor : cursor + batch_count] = high_frequency[origin_indices, hf_context_indices[context_name]]
                    else:
                        context_array[cursor : cursor + batch_count] = np.nan
                cursor += batch_count
    if cursor != total:
        raise AssertionError(f"Minute window materialization wrote {cursor}, expected {total}")
    matrix.flush()
    target.flush()
    timestamps.flush()
    split_codes.flush()
    persistence.flush()
    for array in context_arrays.values():
        array.flush()
    flattened_names = [
        f"input_step_{step - lookback + 1}__{feature}"
        for step in range(lookback)
        for feature in official_manifest["feature_names"]
    ]
    return {
        "feature_path": str(matrix_path),
        "metadata_paths": {
            "timestamps": str(timestamp_path),
            "split_codes": str(split_path),
            "target": str(target_path),
            "current_ramp": str(persistence_path),
            **{name: str(path) for name, path in context_paths.items()},
        },
        "shape": [total, dimension],
        "feature_names": [*flattened_names, *high_frequency_names],
        "base_window_dimension": base_dimension,
        "high_frequency_dimension": len(high_frequency_names),
        "raw_split_counts": raw_split_counts,
        "split_counts": split_counts,
        "excluded_counts": {
            split: int(raw_split_counts[split] - split_counts[split]) for split in SPLIT_NAMES
        },
        "excluded_reason_counts": excluded_reason_counts,
        "excluded_samples": excluded_samples,
        "completeness_filter_applied": bool(completeness_manifest is not None),
        "completeness_requirement": (
            completeness_manifest.get("completeness_requirement") if completeness_manifest is not None else None
        ),
        "lookback": int(lookback),
        "horizon": int(horizon),
        "generation_time_seconds": time.perf_counter() - started,
        "peak_memory_gb": memory.peak_gb,
    }


def fit_minute_hgb(
    matrix_manifest: dict[str, Any],
    model_name: str,
    prediction_filename: str,
    model_filename: str,
    output_paths: dict[str, Path],
    hgb_parameters: dict[str, Any],
    nrmse_denominator: float,
    strong_threshold: float,
    save_persistence: bool,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    matrix = load_npy_memmap(matrix_manifest["feature_path"])
    arrays = {name: load_npy_memmap(path) for name, path in matrix_manifest["metadata_paths"].items()}
    slices = contiguous_experiment_slices(arrays["split_codes"], arrays["target"])
    train_slice = slices["train"]
    model = make_hgb(hgb_parameters)
    fit_metadata = fit_hgb_with_disk_backed_internal_validation(
        model,
        matrix[train_slice],
        arrays["target"][train_slice],
        Path(matrix_manifest["feature_path"]).parent / f"{Path(model_filename).stem}_fit_cache",
    )
    training_time = float(fit_metadata["fit_time_seconds"])
    model_path = output_paths["models"] / model_filename
    joblib.dump(model, model_path, compress=3)
    frames: list[pd.DataFrame] = []
    persistence_frames: list[pd.DataFrame] = []
    metrics_by_split: dict[str, Any] = {}
    prediction_times: dict[str, float] = {}
    mag_edges = magnitude_edges(np.asarray(arrays["target"][train_slice]))
    for split in ("validation", "test"):
        current_slice = slices[split]
        y_true = np.asarray(arrays["target"][current_slice], dtype=np.float64)
        persistence = np.asarray(arrays["current_ramp"][current_slice], dtype=np.float64)
        predict_started = time.perf_counter()
        y_pred = model.predict(matrix[current_slice])
        prediction_times[split] = time.perf_counter() - predict_started
        model_metrics = regression_metrics(y_true, y_pred, persistence, nrmse_denominator)
        model_events = event_metrics(y_true, y_pred, strong_threshold)
        persistence_metrics = regression_metrics(y_true, persistence, persistence, nrmse_denominator)
        persistence_events = event_metrics(y_true, persistence, strong_threshold)
        metrics_by_split[split] = {
            model_name: {**model_metrics, **model_events},
            "Persistence": {**persistence_metrics, **persistence_events},
        }
        context = {
            name: np.asarray(array[current_slice])
            for name, array in arrays.items()
            if name not in {"timestamps", "split_codes", "target", "current_ramp"}
        }
        frames.append(
            build_prediction_frame(
                arrays["timestamps"][current_slice],
                split,
                y_true,
                y_pred,
                strong_threshold,
                mag_edges,
                context,
            )
        )
        if save_persistence:
            persistence_frames.append(
                build_prediction_frame(
                    arrays["timestamps"][current_slice],
                    split,
                    y_true,
                    persistence,
                    strong_threshold,
                    mag_edges,
                    context,
                )
            )
    save_prediction_frames(output_paths["predictions"] / prediction_filename, frames)
    if save_persistence:
        save_prediction_frames(output_paths["predictions"] / "s3_persistence_predictions.parquet", persistence_frames)
    model_metadata_path = output_paths["models"] / f"{Path(model_filename).stem}_metadata.json"
    write_json(
        model_metadata_path,
        {
            **metadata,
            "parameters": hgb_parameters,
            "n_features_in": int(model.n_features_in_),
            "training_rows": int(train_slice.stop - train_slice.start),
            "model_path": str(model_path),
            "prediction_path": str(output_paths["predictions"] / prediction_filename),
            "training_time_seconds": training_time,
            "prediction_times_seconds": prediction_times,
            "peak_training_memory_gb": fit_metadata["peak_memory_gb"],
            "fit_policy": fit_metadata,
        },
    )
    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "model_metadata_path": str(model_metadata_path),
        "prediction_path": str(output_paths["predictions"] / prediction_filename),
        "persistence_prediction_path": str(output_paths["predictions"] / "s3_persistence_predictions.parquet") if save_persistence else None,
        "metrics_by_split": metrics_by_split,
        "training_time_seconds": training_time,
        "prediction_times_seconds": prediction_times,
        "peak_training_memory_gb": fit_metadata["peak_memory_gb"],
        "fit_policy": fit_metadata,
        "slices": {name: [value.start, value.stop] for name, value in slices.items()},
        "shape": matrix_manifest["shape"],
    }


def s3_result_rows(
    s3_run: dict[str, Any],
    matrix_manifest: dict[str, Any],
    high_frequency_manifest: dict[str, Any],
    total_feature_generation_time: float,
    total_preprocessing_time: float,
    preprocessing_peak_memory_gb: float | None,
) -> list[dict[str, Any]]:
    test = s3_run["metrics_by_split"]["test"]
    hgb = test["S3 HistGradientBoosting"]
    persistence = test["Persistence"]
    base = {
        "experiment_id": "S3",
        "target_name": "next_one_minute_ramp_from_one_second_information",
        "input_resolution": "one_second_descriptors_plus_one_minute_context",
        "output_resolution": "one_minute",
        "target_formula": "(Pbar[m+1]-Pbar[m])/10",
        "feature_variant": "selected_physics_plus_causal_one_second_descriptors",
        "feature_count": int(matrix_manifest["shape"][1]),
        "lookback_steps": 8,
        "lookback_duration_seconds": 480,
        "horizon_steps": 1,
        "forecast_lead_seconds": 60,
        "target_interval_seconds": 60,
        "train_rows": int(matrix_manifest["split_counts"]["train"]),
        "validation_rows": int(matrix_manifest["split_counts"]["validation"]),
        "test_rows": int(matrix_manifest["split_counts"]["test"]),
        "sampling_interval_used": 60,
    }
    return [
        {
            **base,
            "model": "HistGradientBoosting",
            **{key: hgb.get(key) for key in ("MAE", "RMSE", "NRMSE", "R2", "persistence_RMSE", "persistence_skill", "directional_accuracy")},
            **{key: hgb.get(key) for key in ("strong_ramp_MAE", "strong_ramp_RMSE", "strong_ramp_precision", "strong_ramp_recall", "strong_ramp_F1")},
            "training_time_seconds": s3_run["training_time_seconds"],
            "validation_prediction_time_seconds": s3_run["prediction_times_seconds"]["validation"],
            "test_prediction_time_seconds": s3_run["prediction_times_seconds"]["test"],
            "inference_time_per_sample": s3_run["prediction_times_seconds"]["test"] / base["test_rows"],
            "feature_generation_time_seconds": total_feature_generation_time,
            "end_to_end_preprocessing_time_seconds": total_preprocessing_time,
            "peak_memory_gb": max(
                value for value in (s3_run["peak_training_memory_gb"], preprocessing_peak_memory_gb) if value is not None
            ),
            "execution_status": "success",
            "notes": f"All aligned minute rows; {high_frequency_manifest['shape'][1]} causal high-frequency descriptors appended to the official L=8 selected-physics window.",
        },
        {
            **base,
            "model": "Persistence",
            **{key: persistence.get(key) for key in ("MAE", "RMSE", "NRMSE", "R2", "persistence_RMSE", "persistence_skill", "directional_accuracy")},
            **{key: persistence.get(key) for key in ("strong_ramp_MAE", "strong_ramp_RMSE", "strong_ramp_precision", "strong_ramp_recall", "strong_ramp_F1")},
            "training_time_seconds": 0.0,
            "validation_prediction_time_seconds": 0.0,
            "test_prediction_time_seconds": 0.0,
            "inference_time_per_sample": 0.0,
            "feature_generation_time_seconds": 0.0,
            "end_to_end_preprocessing_time_seconds": 0.0,
            "peak_memory_gb": None,
            "execution_status": "success",
            "notes": "Same-target one-minute persistence baseline: r[m] is used to predict r[m+1].",
        },
    ]


def build_one_second_ramp_memmap(
    reader: FixedFrameHDFReader,
    temporary_dir: Path,
    rated_power_kw: float,
    chunk_rows: int,
) -> dict[str, Any]:
    temporary_dir.mkdir(parents=True, exist_ok=True)
    output_path = temporary_dir / "all_one_second_ramps.npy"
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(reader.row_count - 1,))
    pv_index = reader.column_indices[SECOND_VARIABLES["pv"]]
    cursor = 0
    previous: float | None = None
    started = time.perf_counter()
    with PeakMemoryMonitor() as memory:
        for start in range(0, reader.row_count, chunk_rows):
            stop = min(reader.row_count, start + chunk_rows)
            _, values = reader.read(start, stop)
            pv = values[:, pv_index].astype(np.float64, copy=False)
            if previous is None:
                ramps = np.diff(pv) / rated_power_kw
            else:
                ramps = np.diff(np.concatenate(([previous], pv))) / rated_power_kw
            output[cursor : cursor + len(ramps)] = ramps.astype(np.float32)
            cursor += len(ramps)
            previous = float(pv[-1])
    if cursor != reader.row_count - 1:
        raise AssertionError(f"One-second ramp materialization wrote {cursor}, expected {reader.row_count - 1}")
    output.flush()
    return {
        "path": str(output_path),
        "row_count": int(cursor),
        "dtype": "float32",
        "generation_time_seconds": time.perf_counter() - started,
        "peak_memory_gb": memory.peak_gb,
    }


def distribution_summary(values: np.ndarray, label: str, threshold: float | None = None) -> list[dict[str, Any]]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    output = [
        {"series": label, "statistic": "count", "value": int(len(finite))},
        {"series": label, "statistic": "mean", "value": float(np.mean(finite))},
        {"series": label, "statistic": "standard_deviation", "value": float(np.std(finite))},
        {"series": label, "statistic": "minimum", "value": float(np.min(finite))},
        {"series": label, "statistic": "maximum", "value": float(np.max(finite))},
        {"series": label, "statistic": "mean_absolute", "value": float(np.mean(np.abs(finite)))},
    ]
    for percentile in (1, 5, 50, 90, 95, 99, 99.9):
        output.append(
            {
                "series": label,
                "statistic": f"absolute_percentile_{percentile:g}",
                "value": float(np.percentile(np.abs(finite), percentile)),
            }
        )
    if threshold is not None:
        output.extend(
            (
                {"series": label, "statistic": "strong_threshold", "value": float(threshold)},
                {"series": label, "statistic": "strong_event_count", "value": int(np.sum(np.abs(finite) >= threshold))},
                {"series": label, "statistic": "strong_event_fraction", "value": float(np.mean(np.abs(finite) >= threshold))},
            )
        )
    return output


def averaging_loss_analysis(
    descriptor_path: Path,
    official_minute_manifest: dict[str, Any],
    one_second_ramp_path: Path,
    one_second_threshold: float,
    output_detail_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    descriptors = pd.read_parquet(descriptor_path)
    descriptors[TIME_COLUMN] = pd.to_datetime(descriptors[TIME_COLUMN], errors="raise")
    official_timestamps = load_npy_memmap(official_minute_manifest["timestamp_path"])
    official_target = load_npy_memmap(official_minute_manifest["target_path"])
    target_frame = pd.DataFrame(
        {
            TIME_COLUMN: np.asarray(official_timestamps, dtype=np.int64).astype("datetime64[ns]"),
            "one_minute_ramp_pu": np.asarray(official_target, dtype=np.float64),
        }
    )
    detail = descriptors.merge(target_frame, on=TIME_COLUMN, how="inner", validate="one_to_one")
    detail["attenuation_indicator"] = 1.0 - np.abs(detail["one_minute_ramp_pu"]) / (detail["pv_max_abs_ramp"] + 1.0e-8)
    detail["both_strong_directions"] = (
        (detail["pv_largest_upward_ramp"] >= one_second_threshold)
        & (detail["pv_largest_downward_ramp"] <= -one_second_threshold)
    )
    variability_threshold = float(detail["pv_std"].quantile(0.95))
    detail["high_variability_small_minute_ramp"] = (
        (detail["pv_std"] >= variability_threshold) & (np.abs(detail["one_minute_ramp_pu"]) <= 0.01)
    )
    detail.to_parquet(output_detail_path, index=False, compression="zstd")
    one_second_ramps = load_npy_memmap(one_second_ramp_path)
    summary_rows = distribution_summary(one_second_ramps, "one_second_ramp_pu", one_second_threshold)
    summary_rows.extend(distribution_summary(detail["one_minute_ramp_pu"].to_numpy(), "one_minute_ramp_pu", 0.01))
    descriptive = {
        "minutes_compared": len(detail),
        "mean_attenuation_indicator": float(detail["attenuation_indicator"].mean()),
        "median_attenuation_indicator": float(detail["attenuation_indicator"].median()),
        "attenuation_positive_fraction": float(np.mean(detail["attenuation_indicator"] > 0)),
        "cancellation_fraction": float(detail["pv_cancellation_flag"].mean()),
        "both_strong_directions_fraction": float(detail["both_strong_directions"].mean()),
        "high_variability_small_minute_ramp_fraction": float(detail["high_variability_small_minute_ramp"].mean()),
        "pv_variability_95th_percentile_kw": variability_threshold,
        "correlation_max_abs_one_second_vs_abs_one_minute": float(
            detail[["pv_max_abs_ramp", "one_minute_ramp_pu"]]
            .assign(one_minute_abs=lambda frame: np.abs(frame["one_minute_ramp_pu"]))
            [["pv_max_abs_ramp", "one_minute_abs"]]
            .corr()
            .iloc[0, 1]
        ),
    }
    summary_rows.extend(
        {"series": "averaging_loss", "statistic": key, "value": value}
        for key, value in descriptive.items()
    )
    return pd.DataFrame(summary_rows), detail


def build_feature_classification_audit(
    one_minute_columns: Sequence[str],
    selected_feature_names: Sequence[str],
    engineered_report_path: Path,
) -> pd.DataFrame:
    engineered = pd.read_csv(engineered_report_path)
    temporal_families = {
        "irradiance_pv_dynamics",
        "irradiance_pv_rolling_context",
        "ramp_specific_past_dynamics",
    }
    rows: list[dict[str, Any]] = []
    selected = set(selected_feature_names)
    for name in one_minute_columns:
        variants = ["full", "full_physics"]
        if name in selected:
            variants.extend(("selected", "selected_physics"))
        rows.append(
            {
                "feature_name": name,
                "source_variable": name,
                "feature_variant": ";".join(variants),
                "category": "original_measurement",
                "formula_or_definition": "Measured SOLETE input at the forecast timestamp.",
                "temporal_statistical": False,
                "physics_guided": False,
                "uses_current_information": True,
                "uses_past_information": False,
                "uses_future_information": False,
                "leakage_safe": True,
                "notes": "Original measured predictor; membership in selected variants follows the validated feature-selection output.",
            }
        )
    for _, item in engineered.iterrows():
        family = str(item["family"])
        temporal = family in temporal_families
        causal_rule = str(item["causal_rule"])
        rows.append(
            {
                "feature_name": item["feature_name"],
                "source_variable": item["source_columns"],
                "feature_variant": "full_physics;selected_physics",
                "category": "temporal_statistical" if temporal else "physics_guided_domain_guided",
                "formula_or_definition": item["description"],
                "temporal_statistical": temporal,
                "physics_guided": not temporal,
                "uses_current_information": "current" in causal_rule or "timestamp" in causal_rule,
                "uses_past_information": any(token in causal_rule for token in ("past", "shift", "trailing")),
                "uses_future_information": False,
                "leakage_safe": not bool(item["uses_target"]),
                "notes": f"Validated family: {family}. Physical knowledge is introduced at feature level, not through a governing-equation loss.",
            }
        )
    return pd.DataFrame(rows)


def controlled_feature_ablation(detailed_path: Path, summary_path: Path) -> pd.DataFrame:
    detailed = pd.read_csv(detailed_path)
    summary = pd.read_csv(summary_path)
    pairs = (
        ("full", "selected", "Full versus Selected"),
        ("full", "full_physics", "Full versus Full Physics-Informed"),
        ("selected", "selected_physics", "Selected versus Selected Physics-Informed"),
        ("full_physics", "selected_physics", "Full Physics-Informed versus Selected Physics-Informed"),
    )
    scopes = [
        ("one_hour", "HistGradientBoosting"),
        ("one_minute", "Ridge"),
        ("one_minute", "HistGradientBoosting"),
    ]
    output: list[dict[str, Any]] = []
    for resolution, model in scopes:
        detail_scope = detailed[
            (detailed["resolution"] == resolution)
            & (detailed["model"] == model)
            & (detailed["lookback"] == 8)
            & (detailed["horizon"] == 1)
            & (detailed["status"].isin(["screening", "success"]))
        ]
        summary_scope = summary[
            (summary["resolution"] == resolution)
            & (summary["model"] == model)
            & (summary["lookback"] == 8)
            & (summary["horizon"] == 1)
        ]
        for reference_variant, new_variant, comparison in pairs:
            available = set(detail_scope["dataset_variant"])
            if reference_variant not in available or new_variant not in available:
                output.append(
                    {
                        "resolution": resolution,
                        "model": model,
                        "lookback": 8,
                        "horizon": 1,
                        "comparison": comparison,
                        "reference_variant": reference_variant,
                        "new_variant": new_variant,
                        "status": "unavailable_controlled_pair",
                        "notes": "At least one variant is absent for this model/resolution in the validated outputs; no values were imputed.",
                    }
                )
                continue
            row: dict[str, Any] = {
                "resolution": resolution,
                "model": model,
                "lookback": 8,
                "horizon": 1,
                "comparison": comparison,
                "reference_variant": reference_variant,
                "new_variant": new_variant,
                "status": "available_controlled_pair",
                "delta_sign_convention": "100*(new-reference)/reference; negative RMSE or time is favorable",
            }
            for variant_role, variant in (("reference", reference_variant), ("new", new_variant)):
                variant_detail = detail_scope[detail_scope["dataset_variant"] == variant]
                variant_summary = summary_scope[summary_scope["dataset_variant"] == variant]
                if len(variant_summary) != 1:
                    raise ValueError(f"Expected one summary row for {resolution}/{model}/{variant}, found {len(variant_summary)}")
                summary_row = variant_summary.iloc[0]
                row[f"{variant_role}_feature_count"] = int(summary_row["feature_count"])
                row[f"{variant_role}_training_time_seconds"] = float(summary_row["elapsed_seconds_train"])
                row[f"{variant_role}_prediction_time_total_seconds"] = float(summary_row["elapsed_seconds_predict_total"])
                for split in ("validation", "test"):
                    split_row = variant_detail[variant_detail["split"] == split]
                    if len(split_row) != 1:
                        raise ValueError(f"Expected one {split} row for {resolution}/{model}/{variant}")
                    split_row = split_row.iloc[0]
                    for metric in ("mae", "rmse", "r2", "skill_rmse_vs_persistence", "elapsed_seconds_predict"):
                        row[f"{variant_role}_{split}_{metric}"] = float(split_row[metric])
            for metric in ("validation_rmse", "test_rmse"):
                reference_value = row[f"reference_{metric}"]
                new_value = row[f"new_{metric}"]
                row[f"percent_delta_{metric}"] = 100.0 * (new_value - reference_value) / reference_value
            row["percent_delta_training_time"] = 100.0 * (
                row["new_training_time_seconds"] - row["reference_training_time_seconds"]
            ) / row["reference_training_time_seconds"]
            row["percent_delta_prediction_time_total"] = 100.0 * (
                row["new_prediction_time_total_seconds"] - row["reference_prediction_time_total_seconds"]
            ) / row["reference_prediction_time_total_seconds"]
            row["percent_delta_feature_count"] = 100.0 * (
                row["new_feature_count"] - row["reference_feature_count"]
            ) / row["reference_feature_count"]
            row["feature_construction_time_seconds"] = np.nan
            row["peak_memory_gb"] = np.nan
            row["notes"] = "Feature-construction time and peak memory were not recorded in the validated Stage A output."
            output.append(row)
    return pd.DataFrame(output)


def blocked_day_bootstrap(
    reference: pd.DataFrame,
    high_frequency: pd.DataFrame,
    strong_threshold: float,
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    left = reference[[TIME_COLUMN, "y_true", "y_pred"]].rename(columns={"y_pred": "reference_pred"})
    right = high_frequency[[TIME_COLUMN, "y_true", "y_pred"]].rename(columns={"y_pred": "high_frequency_pred"})
    merged = left.merge(right, on=[TIME_COLUMN, "y_true"], how="inner", validate="one_to_one")
    merged["day"] = pd.to_datetime(merged[TIME_COLUMN]).dt.floor("D")
    rows: list[dict[str, Any]] = []
    for day, group in merged.groupby("day", sort=True):
        true = group["y_true"].to_numpy(dtype=np.float64)
        reference_error = true - group["reference_pred"].to_numpy(dtype=np.float64)
        hf_error = true - group["high_frequency_pred"].to_numpy(dtype=np.float64)
        strong = np.abs(true) >= strong_threshold
        rows.append(
            {
                "day": day,
                "n": len(group),
                "reference_abs_sum": np.abs(reference_error).sum(),
                "hf_abs_sum": np.abs(hf_error).sum(),
                "reference_sq_sum": np.square(reference_error).sum(),
                "hf_sq_sum": np.square(hf_error).sum(),
                "strong_n": int(strong.sum()),
                "reference_strong_sq_sum": np.square(reference_error[strong]).sum(),
                "hf_strong_sq_sum": np.square(hf_error[strong]).sum(),
                "reference_direction_sum": int((np.sign(true) == np.sign(group["reference_pred"].to_numpy())) .sum()),
                "hf_direction_sum": int((np.sign(true) == np.sign(group["high_frequency_pred"].to_numpy())).sum()),
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        raise ValueError("No complete day blocks available for S3 bootstrap.")
    numeric = daily.drop(columns=["day"]).to_numpy(dtype=np.float64)
    columns = list(daily.drop(columns=["day"]).columns)
    col = {name: index for index, name in enumerate(columns)}
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, 4), dtype=np.float64)
    day_count = len(daily)
    for replicate in range(replicates):
        selected = numeric[rng.integers(0, day_count, size=day_count)].sum(axis=0)
        count = selected[col["n"]]
        strong_count = selected[col["strong_n"]]
        reference_rmse = math.sqrt(selected[col["reference_sq_sum"]] / count)
        hf_rmse = math.sqrt(selected[col["hf_sq_sum"]] / count)
        reference_mae = selected[col["reference_abs_sum"]] / count
        hf_mae = selected[col["hf_abs_sum"]] / count
        if strong_count:
            reference_strong_rmse = math.sqrt(selected[col["reference_strong_sq_sum"]] / strong_count)
            hf_strong_rmse = math.sqrt(selected[col["hf_strong_sq_sum"]] / strong_count)
        else:
            reference_strong_rmse = np.nan
            hf_strong_rmse = np.nan
        reference_direction = selected[col["reference_direction_sum"]] / count
        hf_direction = selected[col["hf_direction_sum"]] / count
        samples[replicate] = (
            hf_rmse - reference_rmse,
            hf_mae - reference_mae,
            hf_strong_rmse - reference_strong_rmse,
            hf_direction - reference_direction,
        )
    alpha = (1.0 - confidence_level) / 2.0
    metric_names = ("RMSE", "MAE", "strong_ramp_RMSE", "directional_accuracy")
    point_values = []
    true = merged["y_true"].to_numpy(dtype=np.float64)
    ref = merged["reference_pred"].to_numpy(dtype=np.float64)
    hf = merged["high_frequency_pred"].to_numpy(dtype=np.float64)
    strong = np.abs(true) >= strong_threshold
    point_values.extend(
        (
            np.sqrt(np.mean(np.square(true - hf))) - np.sqrt(np.mean(np.square(true - ref))),
            np.mean(np.abs(true - hf)) - np.mean(np.abs(true - ref)),
            np.sqrt(np.mean(np.square(true[strong] - hf[strong]))) - np.sqrt(np.mean(np.square(true[strong] - ref[strong]))),
            directional_accuracy(true, hf) - directional_accuracy(true, ref),
        )
    )
    output = []
    for index, metric in enumerate(metric_names):
        lower, upper = np.nanquantile(samples[:, index], [alpha, 1.0 - alpha])
        improvement_when = "negative" if metric != "directional_accuracy" else "positive"
        favors_hf = upper < 0 if improvement_when == "negative" else lower > 0
        favors_reference = lower > 0 if improvement_when == "negative" else upper < 0
        conclusion = "high_frequency" if favors_hf else "reference" if favors_reference else "inconclusive"
        output.append(
            {
                "metric": metric,
                "difference_definition": "high_frequency_minus_existing_reference",
                "point_difference": float(point_values[index]),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "confidence_level": confidence_level,
                "bootstrap_replicates": int(replicates),
                "block": "complete_day",
                "day_blocks": int(day_count),
                "improvement_sign": improvement_when,
                "conclusion": conclusion,
            }
        )
    return pd.DataFrame(output)


def safe_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
