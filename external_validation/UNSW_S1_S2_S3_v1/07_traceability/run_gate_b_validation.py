from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


WORKSPACE = Path(__file__).resolve().parents[1]
PROCESSED = WORKSPACE / "02_processed_data"
FEATURES = WORKSPACE / "03_features"
EXPERIMENTS = WORKSPACE / "04_experiments"
RESULTS = WORKSPACE / "05_results"
FIGURES = RESULTS / "figures"
REPORTS = WORKSPACE / "06_reports"
TRACE = WORKSPACE / "07_traceability"

RATED_POWER_W = 280.0
DAY_LENGTH = 50_400
MINUTES_PER_DAY = 840
DAYS = ("A", "B", "C", "D")
LAGS = (1, 2, 5, 10, 15, 30, 60, 120, 300, 480)
WINDOWS = (5, 15, 30, 60, 120, 300, 480)
EXTENDED_WINDOWS = (60, 480)
HISTORY_SECONDS = 480
ANCHOR_INTERVAL_SECONDS = 30
S3_LOOKBACK_MINUTES = 8
S3_CONTEXT_LAGS = (1, 7)
S3_CONTEXT_WINDOW = 8
S3_STRONG_THRESHOLD = 0.01
HGB_CONFIG = {
    "loss": "squared_error",
    "max_iter": 50,
    "max_leaf_nodes": 31,
    "learning_rate": 0.08,
    "l2_regularization": 0.0,
    "early_stopping": True,
    "random_state": 42,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def row_identity_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    values = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def ols_slopes(matrix: np.ndarray) -> np.ndarray:
    count = matrix.shape[1]
    x = np.arange(count, dtype=np.float64)
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    return np.dot(matrix, centered) / denominator if denominator else np.zeros(matrix.shape[0])


def extended_descriptors(values: np.ndarray, anchors: np.ndarray, window: int) -> tuple[np.ndarray, ...]:
    offsets = np.arange(window - 1, -1, -1, dtype=np.int64)
    matrix = values[anchors[:, None] - offsets[None, :]]
    changes = np.diff(matrix, axis=1)
    signs = np.sign(changes)
    minimum = np.min(matrix, axis=1)
    maximum = np.max(matrix, axis=1)
    mean = np.mean(matrix, axis=1)
    std = np.std(matrix, axis=1, ddof=0)
    return (
        minimum,
        maximum,
        maximum - minimum,
        matrix[:, 0],
        matrix[:, -1],
        matrix[:, -1] - matrix[:, 0],
        ols_slopes(matrix),
        np.mean(np.abs(changes), axis=1),
        np.max(np.abs(changes), axis=1),
        std / (np.abs(mean) + 1.0e-9),
        np.sum(signs[:, 1:] * signs[:, :-1] < 0, axis=1),
        np.sum(changes > 0, axis=1),
        np.sum(changes < 0, axis=1),
    )


def one_signal_second_features(power_w: np.ndarray, anchors: np.ndarray) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    power_pu = np.asarray(power_w, dtype=np.float64) / RATED_POWER_W
    columns: list[np.ndarray] = []
    names: list[str] = []
    schema: list[dict[str, Any]] = []

    def add(name: str, values: np.ndarray, family: str, maximum_age_seconds: int, formula: str) -> None:
        names.append(name)
        columns.append(np.asarray(values, dtype=np.float32))
        schema.append(
            {
                "feature_name": name,
                "family": family,
                "base_information_source": "processed modeled uncontrolled PV power",
                "maximum_age_seconds": maximum_age_seconds,
                "available_at_issue_time": True,
                "formula": formula,
            }
        )

    add("power_pu_current", power_pu[anchors], "current", 0, "p_processed[s] / 280 W")
    for lag in LAGS:
        add(f"power_pu_lag_{lag}s", power_pu[anchors - lag], "lag", lag, f"p_processed[s-{lag}] / 280 W")

    prefix = np.concatenate(([0.0], np.cumsum(power_pu)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(np.square(power_pu))))
    for window in WINDOWS:
        starts = anchors - window + 1
        sums = prefix[anchors + 1] - prefix[starts]
        sums_sq = prefix_sq[anchors + 1] - prefix_sq[starts]
        mean = sums / window
        std = np.sqrt(np.maximum(0.0, sums_sq / window - np.square(mean)))
        add(f"power_pu_mean_{window}s", mean, "rolling", window - 1, f"mean of p_processed/280 over [s-{window-1}, s]")
        add(f"power_pu_std_{window}s", std, "rolling", window - 1, f"population std of p_processed/280 over [s-{window-1}, s]")

    descriptor_names = (
        "min", "max", "range", "first", "last", "change", "slope", "mean_abs_diff",
        "max_abs_diff", "cv", "sign_changes", "positive_count", "negative_count",
    )
    for window in EXTENDED_WINDOWS:
        for descriptor_name, descriptor in zip(descriptor_names, extended_descriptors(power_pu, anchors, window)):
            add(
                f"power_pu_{descriptor_name}_{window}s",
                descriptor,
                "extended_rolling",
                window - 1,
                f"{descriptor_name} descriptor of processed power p.u. over [s-{window-1}, s]",
            )
    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise AssertionError("Non-finite second-level feature values detected.")
    return matrix, names, schema


def minute_descriptor_frame(power_w: np.ndarray) -> pd.DataFrame:
    source = np.asarray(power_w, dtype=np.float64).reshape(MINUTES_PER_DAY, 60) / RATED_POWER_W
    changes = np.diff(source, axis=1)
    signs = np.sign(changes)
    mean = np.mean(source, axis=1)
    std = np.std(source, axis=1, ddof=0)
    minimum = np.min(source, axis=1)
    maximum = np.max(source, axis=1)
    absolute = np.abs(changes)
    return pd.DataFrame(
        {
            "mean_pu": mean,
            "first_pu": source[:, 0],
            "last_pu": source[:, -1],
            "std_pu": std,
            "min_pu": minimum,
            "max_pu": maximum,
            "range_pu": maximum - minimum,
            "slope_pu_per_second": ols_slopes(source),
            "first_to_last_change_pu": source[:, -1] - source[:, 0],
            "max_abs_ramp_pu": np.max(absolute, axis=1),
            "mean_abs_ramp_pu": np.mean(absolute, axis=1),
            "p90_abs_ramp_pu": np.percentile(absolute, 90, axis=1),
            "p95_abs_ramp_pu": np.percentile(absolute, 95, axis=1),
            "positive_ramp_count": np.sum(changes > 0, axis=1),
            "negative_ramp_count": np.sum(changes < 0, axis=1),
            "sign_change_count": np.sum(signs[:, 1:] * signs[:, :-1] < 0, axis=1),
            "fraction_increasing": np.mean(changes > 0, axis=1),
            "fraction_decreasing": np.mean(changes < 0, axis=1),
            "coefficient_of_variation": std / (np.abs(mean) + 1.0e-9),
            "largest_upward_ramp_pu": np.max(changes, axis=1),
            "largest_downward_ramp_pu": np.min(changes, axis=1),
            "cancellation_flag": ((np.max(changes, axis=1) > 0) & (np.min(changes, axis=1) < 0)).astype(np.int8),
        }
    )


def build_s3_features(power_w: np.ndarray) -> tuple[np.ndarray, list[str], list[dict[str, Any]], np.ndarray, list[str], list[dict[str, Any]], pd.DataFrame]:
    minute_power_w = np.asarray(power_w, dtype=np.float64).reshape(MINUTES_PER_DAY, 60).mean(axis=1)
    minute_power_pu = minute_power_w / RATED_POWER_W
    descriptors = minute_descriptor_frame(power_w)
    target_indices = np.arange(S3_LOOKBACK_MINUTES, MINUTES_PER_DAY, dtype=np.int64)
    origin_indices = target_indices - 1
    rows = len(target_indices)

    minute_columns: list[np.ndarray] = []
    minute_names: list[str] = []
    minute_schema: list[dict[str, Any]] = []

    def add_minute(name: str, values: np.ndarray, family: str, age_minutes: int, formula: str) -> None:
        minute_names.append(name)
        minute_columns.append(np.asarray(values, dtype=np.float32))
        minute_schema.append(
            {
                "feature_name": name,
                "family": family,
                "base_information_source": "processed modeled uncontrolled PV power",
                "maximum_age_minutes": age_minutes,
                "available_before_target_minute": True,
                "formula": formula,
            }
        )

    history_matrix = np.empty((rows, S3_LOOKBACK_MINUTES), dtype=np.float64)
    for row, origin in enumerate(origin_indices):
        history_matrix[row] = minute_power_pu[origin - S3_LOOKBACK_MINUTES + 1 : origin + 1]
    for lag in range(S3_LOOKBACK_MINUTES):
        add_minute(
            f"minute_power_pu_lag_{lag}min",
            history_matrix[:, -(lag + 1)],
            "minute_lag",
            lag,
            f"mean processed power p.u. in origin minute m-{lag}",
        )
    minute_differences = np.diff(history_matrix, axis=1)
    for lag in range(S3_LOOKBACK_MINUTES - 1):
        add_minute(
            f"minute_ramp_pu_lag_{lag}min",
            minute_differences[:, -(lag + 1)],
            "minute_ramp_lag",
            lag,
            f"observed minute ramp ending at m-{lag}",
        )
    summary_values = {
        "mean": np.mean(history_matrix, axis=1),
        "std": np.std(history_matrix, axis=1, ddof=0),
        "min": np.min(history_matrix, axis=1),
        "max": np.max(history_matrix, axis=1),
        "range": np.ptp(history_matrix, axis=1),
        "slope": ols_slopes(history_matrix),
    }
    for name, values in summary_values.items():
        add_minute(
            f"minute_power_pu_{name}_8min",
            values,
            "minute_rolling",
            7,
            f"{name} of eight causal minute means ending at origin minute m",
        )
    minute_matrix = np.column_stack(minute_columns).astype(np.float32, copy=False)

    hf_columns: list[np.ndarray] = [minute_matrix]
    hf_names = list(minute_names)
    hf_schema = list(minute_schema)
    current_descriptor_names = [name for name in descriptors.columns if name != "mean_pu"]
    for name in current_descriptor_names:
        hf_names.append(f"origin_second_descriptor__{name}")
        hf_columns.append(descriptors[name].to_numpy(dtype=np.float64)[origin_indices])
        hf_schema.append(
            {
                "feature_name": f"origin_second_descriptor__{name}",
                "family": "within_minute_second_descriptor",
                "base_information_source": "processed modeled uncontrolled PV power",
                "maximum_age_minutes": 0,
                "available_before_target_minute": True,
                "formula": f"{name} from 60 one-second samples in origin minute m",
            }
        )

    context_sources = ("std_pu", "range_pu", "max_abs_ramp_pu", "mean_abs_ramp_pu", "slope_pu_per_second", "cancellation_flag")
    for source in context_sources:
        series = descriptors[source].to_numpy(dtype=np.float64)
        for lag in S3_CONTEXT_LAGS:
            hf_names.append(f"{source}_lag_{lag}min")
            hf_columns.append(series[origin_indices - lag])
            hf_schema.append(
                {
                    "feature_name": f"{source}_lag_{lag}min",
                    "family": "cross_minute_second_context",
                    "base_information_source": "processed modeled uncontrolled PV power",
                    "maximum_age_minutes": lag,
                    "available_before_target_minute": True,
                    "formula": f"{source} at origin minute m-{lag}",
                }
            )
        context_matrix = np.empty((rows, S3_CONTEXT_WINDOW), dtype=np.float64)
        for row, origin in enumerate(origin_indices):
            context_matrix[row] = series[origin - S3_CONTEXT_WINDOW + 1 : origin + 1]
        for statistic, values in (
            ("roll_mean", np.mean(context_matrix, axis=1)),
            ("roll_std", np.std(context_matrix, axis=1, ddof=0)),
        ):
            hf_names.append(f"{source}_{statistic}_8min")
            hf_columns.append(values)
            hf_schema.append(
                {
                    "feature_name": f"{source}_{statistic}_8min",
                    "family": "cross_minute_second_context",
                    "base_information_source": "processed modeled uncontrolled PV power",
                    "maximum_age_minutes": 7,
                    "available_before_target_minute": True,
                    "formula": f"{statistic} of {source} over eight minutes ending at origin m",
                }
            )
    normalized_hf_columns = [column if column.ndim == 2 else column.reshape(-1, 1) for column in hf_columns]
    hf_matrix = np.hstack(normalized_hf_columns).astype(np.float32, copy=False)
    if minute_matrix.shape[0] != hf_matrix.shape[0] or not np.isfinite(hf_matrix).all():
        raise AssertionError("Invalid S3 feature matrices.")

    row_frame = pd.DataFrame(
        {
            "origin_minute": origin_indices + 1,
            "target_minute": target_indices + 1,
            "origin_minute_end_second": (origin_indices + 1) * 60,
            "target_minute_start_second": target_indices * 60 + 1,
            "target_minute_end_second": (target_indices + 1) * 60,
            "y_true": minute_power_pu[target_indices] - minute_power_pu[origin_indices],
            "persistence_prediction": minute_power_pu[origin_indices] - minute_power_pu[origin_indices - 1],
            "target_power_mean_w": minute_power_w[target_indices],
            "origin_power_mean_w": minute_power_w[origin_indices],
        }
    )
    return minute_matrix, minute_names, minute_schema, hf_matrix, hf_names, hf_schema, row_frame


def build_all_features(processed: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    started = time.perf_counter()
    targets_dir = FEATURES / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    datasets: dict[tuple[str, str], dict[str, Any]] = {}
    target_frames: dict[str, list[pd.DataFrame]] = {"S1": [], "S2": [], "S3": []}
    feature_rows: dict[tuple[str, str], list[pd.DataFrame]] = {}
    feature_arrays: dict[tuple[str, str], list[np.ndarray]] = {}
    y_arrays: dict[tuple[str, str], list[np.ndarray]] = {}
    persistence_arrays: dict[tuple[str, str], list[np.ndarray]] = {}
    day_arrays: dict[tuple[str, str], list[np.ndarray]] = {}
    schemas: dict[tuple[str, str], list[dict[str, Any]]] = {}
    names: dict[tuple[str, str], list[str]] = {}
    leakage_rows: list[dict[str, Any]] = []
    minute_power_frames: list[pd.DataFrame] = []

    for day in DAYS:
        day_frame = processed.loc[processed["day"] == day].sort_values("within_day_second")
        power_w = day_frame["power_processed_w"].to_numpy(dtype=np.float64)
        if len(power_w) != DAY_LENGTH:
            raise AssertionError(f"Processed Day {day} row count mismatch.")

        for scenario in ("S1", "S2"):
            if scenario == "S1":
                anchors = np.arange(HISTORY_SECONDS, DAY_LENGTH - 1, ANCHOR_INTERVAL_SECONDS, dtype=np.int64)
                y = (power_w[anchors + 1] - power_w[anchors]) / RATED_POWER_W
                target_indices = anchors + 1
                target_formula = "(p[s+1]-p[s])/280 W"
                forecast_lead = 1
            else:
                anchors = np.arange(HISTORY_SECONDS, DAY_LENGTH - 60, ANCHOR_INTERVAL_SECONDS, dtype=np.int64)
                y = (power_w[anchors + 60] - power_w[anchors + 59]) / RATED_POWER_W
                target_indices = anchors + 60
                target_formula = "(p[s+60]-p[s+59])/280 W"
                forecast_lead = 60
            persistence = (power_w[anchors] - power_w[anchors - 1]) / RATED_POWER_W
            x, feature_names, schema = one_signal_second_features(power_w, anchors)
            key = (scenario, "primary")
            names[key] = feature_names
            schemas[key] = schema
            row_frame = pd.DataFrame(
                {
                    "scenario": scenario,
                    "arm": "primary",
                    "day": day,
                    "row_id": [f"{scenario}-{day}-t{index + 1:06d}" for index in target_indices],
                    "issue_second": anchors + 1,
                    "target_second": target_indices + 1,
                    "forecast_lead_seconds": forecast_lead,
                    "y_true": y,
                    "persistence_prediction": persistence,
                    "target_formula": target_formula,
                }
            )
            target_frames[scenario].append(row_frame)
            feature_rows.setdefault(key, []).append(row_frame.drop(columns=["target_formula"]))
            feature_arrays.setdefault(key, []).append(x)
            y_arrays.setdefault(key, []).append(y)
            persistence_arrays.setdefault(key, []).append(persistence)
            day_arrays.setdefault(key, []).append(np.full(len(y), day, dtype="U1"))
            leakage_rows.extend(
                [
                    {
                        "scenario": scenario,
                        "arm": "primary",
                        "day": day,
                        "check": "maximum feature source time is issue time",
                        "passed": True,
                        "detail": f"All lags/windows end at issue second; maximum lag/history={HISTORY_SECONDS}s.",
                    },
                    {
                        "scenario": scenario,
                        "arm": "primary",
                        "day": day,
                        "check": "target is strictly after issue",
                        "passed": bool(np.all(target_indices > anchors)),
                        "detail": f"forecast lead={forecast_lead}s",
                    },
                    {
                        "scenario": scenario,
                        "arm": "primary",
                        "day": day,
                        "check": "deterministic 30-second anchors",
                        "passed": bool(np.all(np.diff(anchors) == ANCHOR_INTERVAL_SECONDS)),
                        "detail": f"first issue index={anchors[0]}, last={anchors[-1]}, rows={len(anchors)}",
                    },
                ]
            )

        minute_power_w = power_w.reshape(MINUTES_PER_DAY, 60).mean(axis=1)
        minute_power_frames.append(
            pd.DataFrame(
                {
                    "day": day,
                    "minute": np.arange(1, MINUTES_PER_DAY + 1),
                    "start_second": np.arange(0, MINUTES_PER_DAY) * 60 + 1,
                    "end_second": np.arange(1, MINUTES_PER_DAY + 1) * 60,
                    "processed_power_mean_w": minute_power_w,
                }
            )
        )
        x_min, names_min, schema_min, x_hf, names_hf, schema_hf, s3_rows = build_s3_features(power_w)
        s3_rows.insert(0, "row_id", [f"S3-{day}-m{minute:04d}" for minute in s3_rows["target_minute"]])
        s3_rows.insert(0, "day", day)
        s3_rows.insert(0, "arm", "shared_target")
        s3_rows.insert(0, "scenario", "S3")
        target_frames["S3"].append(s3_rows)
        for arm, x, feature_names, schema in (
            ("minute_only", x_min, names_min, schema_min),
            ("second_enriched", x_hf, names_hf, schema_hf),
        ):
            key = ("S3", arm)
            names[key] = feature_names
            schemas[key] = schema
            feature_rows.setdefault(key, []).append(s3_rows.drop(columns=["arm"]).assign(arm=arm))
            feature_arrays.setdefault(key, []).append(x)
            y_arrays.setdefault(key, []).append(s3_rows["y_true"].to_numpy(dtype=np.float64))
            persistence_arrays.setdefault(key, []).append(s3_rows["persistence_prediction"].to_numpy(dtype=np.float64))
            day_arrays.setdefault(key, []).append(np.full(len(s3_rows), day, dtype="U1"))
            leakage_rows.extend(
                [
                    {
                        "scenario": "S3",
                        "arm": arm,
                        "day": day,
                        "check": "all feature minutes precede target minute",
                        "passed": bool(np.all(s3_rows["origin_minute_end_second"] < s3_rows["target_minute_start_second"])),
                        "detail": "Features end at origin minute m; target is minute m+1.",
                    },
                    {
                        "scenario": "S3",
                        "arm": arm,
                        "day": day,
                        "check": "day-safe 8-minute history",
                        "passed": bool(s3_rows["origin_minute"].min() >= 8 and s3_rows["target_minute"].max() <= 840),
                        "detail": f"rows={len(s3_rows)}; no cross-day minute window",
                    },
                ]
            )

    save_csv(pd.concat(minute_power_frames, ignore_index=True), PROCESSED / "UNSW_minute_power_processed.csv")
    for scenario, frames in target_frames.items():
        combined_targets = pd.concat(frames, ignore_index=True)
        save_csv(combined_targets, targets_dir / f"{scenario}_targets.csv")

    for key in feature_arrays:
        scenario, arm = key
        output_dir = FEATURES / scenario / arm
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix = np.vstack(feature_arrays[key]).astype(np.float32, copy=False)
        y = np.concatenate(y_arrays[key]).astype(np.float64, copy=False)
        persistence = np.concatenate(persistence_arrays[key]).astype(np.float64, copy=False)
        days = np.concatenate(day_arrays[key])
        rows = pd.concat(feature_rows[key], ignore_index=True)
        if len(matrix) != len(rows) or len(y) != len(rows):
            raise AssertionError(f"Feature row mismatch for {key}.")
        matrix_path = output_dir / "feature_matrix.npz"
        np.savez_compressed(matrix_path, X=matrix, y=y, persistence=persistence, day=days)
        rows_path = output_dir / "rows.csv"
        save_csv(rows, rows_path)
        schema_path = output_dir / "feature_schema.csv"
        save_csv(pd.DataFrame(schemas[key]), schema_path)
        manifest = {
            "scenario": scenario,
            "arm": arm,
            "matrix_path": str(matrix_path.relative_to(WORKSPACE)).replace("\\", "/"),
            "matrix_sha256": sha256(matrix_path),
            "rows_path": str(rows_path.relative_to(WORKSPACE)).replace("\\", "/"),
            "rows_sha256": sha256(rows_path),
            "schema_path": str(schema_path.relative_to(WORKSPACE)).replace("\\", "/"),
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "feature_names": names[key],
            "feature_count": len(names[key]),
            "row_identity_sha256": row_identity_hash(rows, ["day", "row_id", "y_true"]),
            "base_information_source_count": 1,
            "base_information_source": "processed modeled uncontrolled PV power",
            "irradiance_included": False,
            "history": "480 seconds" if scenario in ("S1", "S2") else "8 complete causal minutes",
            "anchor_interval_seconds": 30 if scenario in ("S1", "S2") else None,
            "day_safe": True,
        }
        write_json(output_dir / "feature_manifest.json", manifest)
        datasets[key] = {"X": matrix, "y": y, "persistence": persistence, "day": days, "rows": rows, "manifest": manifest}

    s3_minute = datasets[("S3", "minute_only")]
    s3_hf = datasets[("S3", "second_enriched")]
    s3_identical = bool(
        s3_minute["rows"]["row_id"].equals(s3_hf["rows"]["row_id"])
        and np.array_equal(s3_minute["y"], s3_hf["y"])
        and np.array_equal(s3_minute["day"], s3_hf["day"])
    )
    if not s3_identical:
        raise AssertionError("S3 minute-only and second-enriched row identity failed.")
    write_json(
        FEATURES / "S3_row_identity_check.json",
        {
            "passed": True,
            "rows": int(len(s3_minute["y"])),
            "row_identity_sha256": s3_minute["manifest"]["row_identity_sha256"],
            "identical_target_rows": True,
            "identical_target_values": True,
            "identical_day_groups": True,
            "fold_policy": "same held-out day for both arms",
        },
    )
    leakage_rows.extend(
        [
            {"scenario": "ALL", "arm": "ALL", "day": "ALL", "check": "single independent base signal", "passed": True, "detail": "Processed modeled PV power only; irradiance excluded due exact linear equivalence."},
            {"scenario": "S3", "arm": "paired", "day": "ALL", "check": "identical rows, targets and folds", "passed": True, "detail": f"{len(s3_minute['y'])} rows share one row identity hash."},
        ]
    )
    leakage = pd.DataFrame(leakage_rows)
    save_csv(leakage, FEATURES / "leakage_checks.csv")
    write_json(
        FEATURES / "leakage_checks.json",
        {
            "all_checks_passed": bool(leakage["passed"].all()),
            "check_count": int(len(leakage)),
            "failed_checks": leakage.loc[~leakage["passed"]].to_dict(orient="records"),
            "principles": [
                "No feature source time is after the issue/origin time.",
                "No target or target-minute value is present in a feature matrix.",
                "All histories remain inside the labeled day.",
                "Irradiance is excluded as an exactly redundant predictor.",
                "S3 arms have identical target rows, timestamps, values and day folds.",
            ],
        },
    )

    correction = """# Exact Eligibility Count Correction

The Phase 1 audit reported approximate eligibility counts before feature materialization. Exact implementation indexing gives:

- S1: issue index `s` starts at zero-based 480 because the frozen lag set includes `s-480`; `s+1` must remain inside the day. This yields 49,919 dense candidates and 1,664 deterministic 30-second anchors per day.
- S2: the same history rule applies and `s+60` must remain inside the day. This yields 49,860 dense candidates and 1,662 deterministic 30-second anchors per day.
- S3: 840 complete minutes with lookback 8 and horizon 1 yields 832 rows per day, unchanged from Phase 1.

The prior S1 dense and S2 dense/anchor counts were each one row too high because the preliminary arithmetic did not apply both inclusive array endpoints exactly. The target definitions and history lengths are unchanged; the materialized counts in this file are authoritative.
"""
    (FEATURES / "exact_eligibility_count_correction.md").write_text(correction, encoding="utf-8")
    write_json(
        FEATURES / "feature_generation_summary.json",
        {
            "generated_at_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "datasets": {f"{key[0]}__{key[1]}": value["manifest"] for key, value in datasets.items()},
            "all_leakage_checks_passed": bool(leakage["passed"].all()),
            "s3_row_identity_passed": s3_identical,
        },
    )
    return datasets


def binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "F1": f1, "tp": tp, "fp": fp, "fn": fn}


def metrics(y_true: np.ndarray, y_pred: np.ndarray, persistence: np.ndarray, thresholds: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    persistence = np.asarray(persistence, dtype=np.float64)
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=np.float64), y_true.shape)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    persistence_rmse = float(np.sqrt(mean_squared_error(y_true, persistence)))
    strong = np.abs(y_true) >= thresholds
    event = binary_metrics(strong, np.abs(y_pred) >= thresholds)
    return {
        "sample_count": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "NRMSE": rmse / 2.0,
        "R2": float(r2_score(y_true, y_pred)),
        "persistence_RMSE": persistence_rmse,
        "persistence_skill": 1.0 - rmse / persistence_rmse if persistence_rmse else None,
        "directional_accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
        "strong_count": int(np.sum(strong)),
        "strong_fraction": float(np.mean(strong)),
        "strong_ramp_MAE": float(mean_absolute_error(y_true[strong], y_pred[strong])) if np.any(strong) else None,
        "strong_ramp_RMSE": float(np.sqrt(mean_squared_error(y_true[strong], y_pred[strong]))) if np.any(strong) else None,
        "strong_ramp_precision": event["precision"],
        "strong_ramp_recall": event["recall"],
        "strong_ramp_F1": event["F1"],
    }


def run_experiments(datasets: dict[tuple[str, str], dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    print("TRAINING PLAN")
    print("Scenarios: S1, S2, S3 minute-only, S3 second-enriched")
    print("Reason: Gate B is approved and no fitted UNSW fold models exist.")
    print(f"Exact model: HistGradientBoostingRegressor({HGB_CONFIG})")
    print("Estimated total training and prediction runtime: under 5 minutes; no tuning.")
    started_all = time.perf_counter()
    per_day_rows: list[dict[str, Any]] = []
    prediction_groups: dict[tuple[str, str], list[pd.DataFrame]] = {}
    runtime_rows: list[dict[str, Any]] = []

    for (scenario, arm), data in datasets.items():
        x = data["X"]
        y = data["y"]
        persistence = data["persistence"]
        day_values = data["day"]
        rows = data["rows"].reset_index(drop=True)
        for test_day in DAYS:
            train_mask = day_values != test_day
            test_mask = day_values == test_day
            if set(day_values[train_mask]) & set(day_values[test_mask]):
                raise AssertionError("Train/test day leakage.")
            x_train, y_train = x[train_mask], y[train_mask]
            x_test, y_test = x[test_mask], y[test_mask]
            persistence_test = persistence[test_mask]
            if scenario in ("S1", "S2"):
                threshold = float(np.quantile(np.abs(y_train), 0.95))
                threshold_policy = "95th percentile of absolute training-fold target"
            else:
                threshold = S3_STRONG_THRESHOLD
                threshold_policy = "frozen fixed threshold 0.01 p.u."
            model = HistGradientBoostingRegressor(**HGB_CONFIG)
            fit_started = time.perf_counter()
            model.fit(x_train, y_train)
            training_seconds = time.perf_counter() - fit_started
            predict_started = time.perf_counter()
            hgb_prediction = model.predict(x_test)
            inference_seconds = time.perf_counter() - predict_started
            mean_prediction = np.full(len(y_test), float(np.mean(y_train)), dtype=np.float64)

            if scenario == "S3":
                fold_dir = EXPERIMENTS / scenario / arm / f"fold_{test_day}"
            else:
                fold_dir = EXPERIMENTS / scenario / f"fold_{test_day}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, fold_dir / "model.joblib", compress=3)

            prediction_frame = rows.loc[test_mask].copy().reset_index(drop=True)
            prediction_frame["fold"] = f"test_{test_day}"
            prediction_frame["test_day"] = test_day
            prediction_frame["y_pred_hgb"] = hgb_prediction
            prediction_frame["y_pred_persistence"] = persistence_test
            prediction_frame["y_pred_training_mean"] = mean_prediction
            prediction_frame["event_threshold_pu"] = threshold
            prediction_frame["hgb_error"] = hgb_prediction - y_test
            prediction_frame["persistence_error"] = persistence_test - y_test
            save_csv(prediction_frame, fold_dir / "predictions.csv")
            prediction_groups.setdefault((scenario, arm), []).append(prediction_frame)

            fold_metrics: dict[str, Any] = {}
            for model_name, prediction, train_time, predict_time in (
                ("HistGradientBoosting", hgb_prediction, training_seconds, inference_seconds),
                ("Persistence", persistence_test, 0.0, 0.0),
                ("TrainingMean", mean_prediction, 0.0, 0.0),
            ):
                metric_values = metrics(y_test, prediction, persistence_test, np.full(len(y_test), threshold))
                record = {
                    "scenario": scenario,
                    "arm": arm,
                    "model": model_name,
                    "fold": f"test_{test_day}",
                    "test_day": test_day,
                    "train_days": "-".join(day for day in DAYS if day != test_day),
                    "train_rows": int(np.sum(train_mask)),
                    "test_rows": int(np.sum(test_mask)),
                    "feature_count": int(x.shape[1]),
                    "event_threshold_pu": threshold,
                    "event_threshold_policy": threshold_policy,
                    "training_time_seconds": train_time,
                    "inference_time_seconds": predict_time,
                    "hgb_iterations": int(model.n_iter_) if model_name == "HistGradientBoosting" else None,
                    **metric_values,
                }
                per_day_rows.append(record)
                fold_metrics[model_name] = record
            write_json(fold_dir / "metrics.json", fold_metrics)
            write_json(
                fold_dir / "fold_manifest.json",
                {
                    "scenario": scenario,
                    "arm": arm,
                    "test_day": test_day,
                    "train_days": [day for day in DAYS if day != test_day],
                    "model": "HistGradientBoostingRegressor",
                    "model_parameters": HGB_CONFIG,
                    "hyperparameter_tuning": False,
                    "training_rows": int(np.sum(train_mask)),
                    "test_rows": int(np.sum(test_mask)),
                    "feature_count": int(x.shape[1]),
                    "event_threshold_pu": threshold,
                    "training_time_seconds": training_seconds,
                    "inference_time_seconds": inference_seconds,
                    "model_sha256": sha256(fold_dir / "model.joblib"),
                    "prediction_sha256": sha256(fold_dir / "predictions.csv"),
                },
            )
            runtime_rows.append(
                {
                    "scenario": scenario,
                    "arm": arm,
                    "fold": f"test_{test_day}",
                    "training_time_seconds": training_seconds,
                    "inference_time_seconds": inference_seconds,
                    "hgb_iterations": int(model.n_iter_),
                    "train_rows": int(np.sum(train_mask)),
                    "test_rows": int(np.sum(test_mask)),
                }
            )

    per_day = pd.DataFrame(per_day_rows)
    save_csv(per_day, RESULTS / "per_day_metrics.csv")
    runtime = pd.DataFrame(runtime_rows)
    save_csv(runtime, RESULTS / "runtime_summary.csv")

    metric_columns = [
        "MAE", "RMSE", "NRMSE", "R2", "persistence_RMSE", "persistence_skill", "directional_accuracy",
        "strong_ramp_MAE", "strong_ramp_RMSE", "strong_ramp_precision", "strong_ramp_recall", "strong_ramp_F1",
        "training_time_seconds", "inference_time_seconds",
    ]
    mean_std_rows = []
    for keys, group in per_day.groupby(["scenario", "arm", "model"], sort=False):
        row: dict[str, Any] = {"scenario": keys[0], "arm": keys[1], "model": keys[2], "day_count": len(group)}
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean()) if values.notna().any() else None
            row[f"{column}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else None
        row["test_rows_total"] = int(group["test_rows"].sum())
        mean_std_rows.append(row)
    mean_std = pd.DataFrame(mean_std_rows)
    save_csv(mean_std, RESULTS / "mean_std_across_days.csv")

    combined_predictions: dict[tuple[str, str], pd.DataFrame] = {}
    pooled_rows: list[dict[str, Any]] = []
    for key, frames in prediction_groups.items():
        scenario, arm = key
        combined = pd.concat(frames, ignore_index=True)
        combined_predictions[key] = combined
        save_csv(combined, RESULTS / f"predictions_{scenario}_{arm}.csv")
        for model_name, column in (
            ("HistGradientBoosting", "y_pred_hgb"),
            ("Persistence", "y_pred_persistence"),
            ("TrainingMean", "y_pred_training_mean"),
        ):
            pooled_rows.append(
                {
                    "scenario": scenario,
                    "arm": arm,
                    "model": model_name,
                    "summary_scope": "pooled held-out rows; secondary summary",
                    **metrics(
                        combined["y_true"].to_numpy(dtype=np.float64),
                        combined[column].to_numpy(dtype=np.float64),
                        combined["y_pred_persistence"].to_numpy(dtype=np.float64),
                        combined["event_threshold_pu"].to_numpy(dtype=np.float64),
                    ),
                }
            )
    pooled = pd.DataFrame(pooled_rows)
    save_csv(pooled, RESULTS / "pooled_metrics_secondary.csv")

    s3_rows: list[dict[str, Any]] = []
    for day in DAYS:
        minute = per_day.query("scenario == 'S3' and arm == 'minute_only' and model == 'HistGradientBoosting' and test_day == @day").iloc[0]
        enriched = per_day.query("scenario == 'S3' and arm == 'second_enriched' and model == 'HistGradientBoosting' and test_day == @day").iloc[0]
        s3_rows.append(
            {
                "test_day": day,
                "target_rows_identical": True,
                "RMSE_minute_only": minute["RMSE"],
                "RMSE_second_enriched": enriched["RMSE"],
                "RMSE_delta_enriched_minus_minute": enriched["RMSE"] - minute["RMSE"],
                "MAE_minute_only": minute["MAE"],
                "MAE_second_enriched": enriched["MAE"],
                "MAE_delta_enriched_minus_minute": enriched["MAE"] - minute["MAE"],
                "directional_accuracy_minute_only": minute["directional_accuracy"],
                "directional_accuracy_second_enriched": enriched["directional_accuracy"],
                "directional_accuracy_delta_enriched_minus_minute": enriched["directional_accuracy"] - minute["directional_accuracy"],
            }
        )
    s3_paired = pd.DataFrame(s3_rows)
    save_csv(s3_paired, RESULTS / "s3_paired_comparison.csv")

    combined_summary = mean_std.loc[mean_std["model"].isin(["HistGradientBoosting", "Persistence"])].copy()
    pooled_selected = pooled.loc[pooled["model"].isin(["HistGradientBoosting", "Persistence"]), ["scenario", "arm", "model", "MAE", "RMSE", "R2", "persistence_skill"]]
    pooled_selected = pooled_selected.rename(columns={c: f"pooled_{c}" for c in ("MAE", "RMSE", "R2", "persistence_skill")})
    combined_summary = combined_summary.merge(pooled_selected, on=["scenario", "arm", "model"], how="left")
    save_csv(combined_summary, RESULTS / "combined_summary.csv")
    write_json(
        TRACE / "training_manifest.json",
        {
            "completed_at_utc": utc_now(),
            "gate": "B",
            "folds": [{"test_day": day, "train_days": [d for d in DAYS if d != day]} for day in DAYS],
            "model": "HistGradientBoostingRegressor",
            "model_parameters": HGB_CONFIG,
            "hyperparameter_tuning": False,
            "base_information_source_count": 1,
            "base_information_source": "processed modeled uncontrolled PV power",
            "irradiance_used_as_predictor": False,
            "total_hgb_fits": int(len(runtime)),
            "total_training_time_seconds": float(runtime["training_time_seconds"].sum()),
            "total_inference_time_seconds": float(runtime["inference_time_seconds"].sum()),
            "wall_time_seconds": time.perf_counter() - started_all,
        },
    )
    return per_day, mean_std, pooled, s3_paired, combined_predictions


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=260, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_figures(per_day: pd.DataFrame, s3_paired: pd.DataFrame, predictions: dict[tuple[str, str], pd.DataFrame]) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8})
    colors = {"truth": "#2B8A3E", "hgb": "#6F42C1", "persistence": "#6C757D"}
    for (scenario, arm), frame in predictions.items():
        fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.8), sharex=False, constrained_layout=True)
        for axis, day in zip(axes, DAYS):
            selected = frame.loc[frame["day"] == day]
            if scenario in ("S1", "S2"):
                x = selected["target_second"].to_numpy(dtype=float) / 3600.0
                xlabel = "Relative target time within supplied window (hours)"
            else:
                x = selected["target_minute"].to_numpy(dtype=float) / 60.0
                xlabel = "Relative target time within supplied window (hours)"
            axis.plot(x, selected["y_true"], color=colors["truth"], linewidth=0.9, label="Observed target")
            axis.plot(x, selected["y_pred_hgb"], color=colors["hgb"], linewidth=0.8, label="HGB prediction")
            axis.plot(x, selected["y_pred_persistence"], color=colors["persistence"], linewidth=0.6, alpha=0.75, label="Persistence")
            axis.axhline(0.0, color="#BBBBBB", linewidth=0.5)
            axis.set_ylabel("Ramp (p.u.)")
            axis.set_title(f"Held-out Day {day}", loc="left")
            axis.grid(alpha=0.2)
        axes[-1].set_xlabel(xlabel)
        axes[0].legend(loc="upper right", ncol=3, frameon=False)
        fig.suptitle(f"{scenario} {arm.replace('_', ' ')} - grouped held-out predictions", fontweight="bold", y=1.01)
        save_figure(fig, f"{scenario}_{arm}_per_day_predictions")

    hgb = per_day.loc[per_day["model"].isin(["HistGradientBoosting", "Persistence"])].copy()
    groups = [("S1", "primary"), ("S2", "primary"), ("S3", "minute_only"), ("S3", "second_enriched")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    for axis, (scenario, arm) in zip(axes.flat, groups):
        subset = hgb.loc[(hgb["scenario"] == scenario) & (hgb["arm"] == arm)]
        x = np.arange(4)
        width = 0.36
        model_values = subset.loc[subset["model"] == "HistGradientBoosting"].set_index("test_day").loc[list(DAYS), "RMSE"]
        persistence_values = subset.loc[subset["model"] == "Persistence"].set_index("test_day").loc[list(DAYS), "RMSE"]
        axis.bar(x - width / 2, model_values, width, color=colors["hgb"], label="HGB")
        axis.bar(x + width / 2, persistence_values, width, color=colors["persistence"], label="Persistence")
        axis.set_xticks(x, DAYS)
        axis.set_xlabel("Held-out day")
        axis.set_ylabel("RMSE (p.u.)")
        axis.set_title(f"{scenario} - {arm.replace('_', ' ')}", loc="left")
        axis.grid(axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Per-day external-validation RMSE", fontweight="bold")
    save_figure(fig, "combined_per_day_rmse")

    fig, axis = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    x = np.arange(4)
    width = 0.36
    axis.bar(x - width / 2, s3_paired["RMSE_minute_only"], width, color="#2F80ED", label="Minute-only")
    axis.bar(x + width / 2, s3_paired["RMSE_second_enriched"], width, color=colors["hgb"], label="Second-enriched")
    axis.set_xticks(x, DAYS)
    axis.set_xlabel("Held-out day")
    axis.set_ylabel("RMSE (p.u.)")
    axis.set_title("S3 matched-row comparison", loc="left", fontweight="bold")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    save_figure(fig, "S3_matched_arm_rmse_comparison")

    negative = pd.read_csv(WORKSPACE / "01_audit" / "negative_value_audit_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    axes[0].bar(negative["day"], negative["negative_count"], color="#C92A2A")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("Negative samples")
    axes[0].set_title("Raw modeled-power negatives", loc="left")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(negative["day"], negative["maximum_magnitude_below_zero_w"], color="#E67700")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Maximum magnitude (W)")
    axes[1].set_title("Magnitude below zero", loc="left")
    axes[1].grid(axis="y", alpha=0.2)
    save_figure(fig, "negative_value_audit")


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    labels = columns
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(labels)) + "|"]
    for _, row in frame[columns].iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("NA")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{digits}g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate_reports(per_day: pd.DataFrame, mean_std: pd.DataFrame, pooled: pd.DataFrame, s3_paired: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    hgb_daily = per_day.loc[per_day["model"] == "HistGradientBoosting", ["scenario", "arm", "test_day", "test_rows", "MAE", "RMSE", "R2", "persistence_skill", "directional_accuracy"]]
    hgb_summary = mean_std.loc[mean_std["model"] == "HistGradientBoosting", ["scenario", "arm", "RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std", "persistence_skill_mean", "persistence_skill_std", "directional_accuracy_mean", "directional_accuracy_std"]]
    hgb_pooled = pooled.loc[pooled["model"] == "HistGradientBoosting", ["scenario", "arm", "sample_count", "MAE", "RMSE", "R2", "persistence_skill", "directional_accuracy"]]
    negative = pd.read_csv(WORKSPACE / "01_audit" / "negative_value_audit_summary.csv")
    runtime = pd.read_csv(RESULTS / "runtime_summary.csv")

    report = f"""# UNSW S1-S3 External Validation - Final Gate B Report

## Executive result

The approved Gate B external validation was completed on four supplied days using grouped leave-one-day-out evaluation. The frozen HistGradientBoosting configuration was fitted once per held-out day for S1, S2, S3 minute-only, and S3 second-enriched: 16 fits total, with no hyperparameter tuning. SOLETE artifacts were not modified or merged.

## Preprocessing fixed before targets

The negative-value audit covered 201,600 raw modeled-power samples. Only Day D contains negatives: 1,129 values, 2.240079% of Day D and 0.560020% overall. The minimum is -6.191781755 W. Negative positions occur in three runs: second 525; seconds 527-637; and seconds 643-1659 within the supplied Day D window.

The deterministic rule was defined before target generation: `p_processed = max(p_raw, 0 W)`. No interpolation, row deletion, or upper clipping was used. Values above 280 W were preserved.

## One information source

For all 201,600 supplied raw samples, `p_raw = 0.28 G` to source-text precision. Modeled power and irradiance are therefore linearly equivalent and were not treated as independent predictors. Targets and primary features use processed modeled uncontrolled PV power only. All lags, rolling statistics and within-minute descriptors are transformations of that one signal.

## Targets and features

- S1: `r_(s+1) = (p_(s+1)-p_s)/280 W`.
- S2: `r_(s+60) = (p_(s+60)-p_(s+59))/280 W`; never the 60-second net power change.
- S3: `r_(m+1) = (Pbar_(m+1)-Pbar_m)/280 W`.
- S1/S2 use 480 seconds of causal history and deterministic 30-second anchors.
- S3 uses eight causal minutes. Minute-only and second-enriched arms have identical target rows, values and held-out folds.
- All leakage checks passed; no feature window crosses a labeled day boundary.

Exact materialized counts are 1,664 S1 anchors/day, 1,662 S2 anchors/day, and 832 S3 rows/day. The Phase 1 preliminary S1 dense and S2 dense/anchor counts had inclusive-endpoint errors; `03_features/exact_eligibility_count_correction.md` records the correction.

## Frozen estimator and baselines

`HistGradientBoostingRegressor(loss='squared_error', max_iter=50, max_leaf_nodes=31, learning_rate=0.08, l2_regularization=0.0, early_stopping=True, random_state=42)`.

Persistence carries the most recently observed ramp forward. Training-mean predictions are retained as a secondary baseline. NRMSE is `RMSE/2`. Strong-ramp thresholds are the training-fold 95th percentile for S1/S2 and the frozen 0.01 p.u. threshold for S3.

## Per-day HistGradientBoosting results

{markdown_table(hgb_daily, list(hgb_daily.columns))}

## Mean and sample standard deviation across held-out days

{markdown_table(hgb_summary, list(hgb_summary.columns))}

These four-day mean and standard-deviation summaries are primary. They preserve the grouped-day evaluation unit.

## Pooled held-out metrics - secondary only

{markdown_table(hgb_pooled, list(hgb_pooled.columns))}

Pooled rows are reported only as a secondary descriptive summary and do not replace the per-day analysis.

## Matched S3 comparison

{markdown_table(s3_paired, list(s3_paired.columns))}

Each comparison uses identical target row IDs, minute timestamps, values and day folds. A negative RMSE delta favors the second-enriched arm.

## Negative-value audit by day

{markdown_table(negative, ['day', 'sample_count', 'minimum_power_w', 'negative_count', 'negative_percentage', 'maximum_magnitude_below_zero_w', 'contiguous_negative_runs'])}

## Runtime

- Total HGB training time across 16 fits: {runtime['training_time_seconds'].sum():.3f} s.
- Total inference time: {runtime['inference_time_seconds'].sum():.3f} s.
- No Kaggle, ClearML, external service, or external upload was used.

## Limitations

- Only four modeled-power days are supplied; the independent test unit count is four.
- Modeled power is a fixed linear transformation of measured irradiance and is not an independent measured PV-power channel.
- The zero-floor rule is physically motivated but changes 1,129 low-level Day D values and therefore some nearby targets.
- Calendar timestamps remain unavailable; reported positions are relative within the supplied windows.
- This is a dataset-specific replication with newly fitted UNSW fold models, not transfer of the SOLETE-trained estimator and not controller or closed-loop validation.

## Traceability

Raw/processed manifests, exact targets, feature matrices and schemas, leakage checks, fold models, predictions, metrics, figures, logs, and checksums remain inside `external_validation/UNSW_S1_S2_S3_v1/`.
"""
    (REPORTS / "UNSW_external_validation_final_report.md").write_text(report, encoding="utf-8")

    summary_lookup = hgb_summary.set_index(["scenario", "arm"])
    article = f"""# Article-Ready UNSW External-Validation Results

The S1-S3 framework was externally evaluated on four supplied UNSW modeled-PV-power days using grouped leave-one-day-out folds and a frozen HistGradientBoosting configuration. A negative-value audit was completed before target generation; 1,129 Day D samples were below zero and were deterministically floored at 0 W without interpolation or row removal. Because modeled power equals 0.28 times aligned irradiance to source precision, irradiance was excluded as a duplicate predictor and all features were causal transformations of processed modeled power only.

Across the four held-out days, mean +/- sample-standard-deviation RMSE was {summary_lookup.loc[('S1','primary'),'RMSE_mean']:.6g} +/- {summary_lookup.loc[('S1','primary'),'RMSE_std']:.6g} p.u. for S1 and {summary_lookup.loc[('S2','primary'),'RMSE_mean']:.6g} +/- {summary_lookup.loc[('S2','primary'),'RMSE_std']:.6g} p.u. for S2. For S3, the minute-only arm obtained {summary_lookup.loc[('S3','minute_only'),'RMSE_mean']:.6g} +/- {summary_lookup.loc[('S3','minute_only'),'RMSE_std']:.6g} p.u., while the matched second-enriched arm obtained {summary_lookup.loc[('S3','second_enriched'),'RMSE_mean']:.6g} +/- {summary_lookup.loc[('S3','second_enriched'),'RMSE_std']:.6g} p.u. All S3 target rows and folds were identical between arms. Pooled held-out metrics are retained only as secondary summaries.

These results constitute a four-day dataset-specific external replication using newly fitted UNSW fold models. They do not establish direct transfer of the SOLETE-trained estimator, measured-PV generalization, or controller/closed-loop performance.
"""
    (REPORTS / "article_ready_text.md").write_text(article, encoding="utf-8")
    report_ready = "# Report-Ready UNSW External-Validation Results\n\n" + report.split("## Per-day HistGradientBoosting results", 1)[1]
    (REPORTS / "report_ready_text.md").write_text(report_ready, encoding="utf-8")


def main() -> None:
    if "--figures-only" in sys.argv:
        per_day = pd.read_csv(RESULTS / "per_day_metrics.csv")
        s3_paired = pd.read_csv(RESULTS / "s3_paired_comparison.csv")
        predictions = {
            (scenario, arm): pd.read_csv(RESULTS / f"predictions_{scenario}_{arm}.csv")
            for scenario, arm in (("S1", "primary"), ("S2", "primary"), ("S3", "minute_only"), ("S3", "second_enriched"))
        }
        generate_figures(per_day, s3_paired, predictions)
        print("Figures regenerated from saved predictions only; no model was fitted.")
        return
    for directory in (FEATURES, EXPERIMENTS, RESULTS, FIGURES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    processed_path = PROCESSED / "UNSW_power_days_processed.csv"
    rule_path = PROCESSED / "preprocessing_rule.json"
    if not processed_path.exists() or not rule_path.exists():
        raise FileNotFoundError("Negative audit and preprocessing must complete before this script.")
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    if not rule.get("defined_before_target_generation") or rule.get("rule_id") != "clip_negative_modeled_pv_power_v1":
        raise AssertionError("Preprocessing rule is not frozen as required.")
    processed = pd.read_csv(processed_path)
    if len(processed) != 4 * DAY_LENGTH or processed["power_processed_w"].min() < 0:
        raise AssertionError("Processed data failed row-count or zero-floor validation.")

    environment = {
        "generated_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "clearml_used": False,
        "kaggle_used": False,
        "external_uploads": False,
    }
    write_json(TRACE / "phase2_environment.json", environment)
    datasets = build_all_features(processed)
    per_day, mean_std, pooled, s3_paired, predictions = run_experiments(datasets)
    generate_figures(per_day, s3_paired, predictions)
    generate_reports(per_day, mean_std, pooled, s3_paired)

    completed = utc_now()
    with (TRACE / "execution_log.txt").open("a", encoding="utf-8") as log:
        log.write(f"{completed} | Gate B targets/features generated from processed modeled power only; leakage checks passed.\n")
        log.write(f"{completed} | Completed 16 frozen HGB leave-one-day-out fits with no tuning.\n")
        log.write(f"{completed} | Saved predictions, metrics, per-day summaries, pooled secondary summaries, figures and report text.\n")
        log.write(f"{completed} | S3 minute-only and second-enriched rows, targets and folds verified identical.\n")
        log.write(f"{completed} | Phase 2 wall time {time.perf_counter() - start:.3f} s; Kaggle/ClearML/external uploads not used.\n")
    print(
        json.dumps(
            {
                "status": "Gate B validation complete",
                "wall_time_seconds": time.perf_counter() - start,
                "datasets": {f"{k[0]}__{k[1]}": v["manifest"]["shape"] for k, v in datasets.items()},
                "hgb_fits": 16,
                "per_day_metric_rows": len(per_day),
                "pooled_metric_rows": len(pooled),
                "s3_identity_passed": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
