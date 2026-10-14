from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pv_ramp_automation.one_second_extension import (
    ChronologicalSecondReader,
    SECOND_VARIABLES,
    _minute_descriptor_frame,
    backward_ramp,
    boundaries_from_minute_manifest,
    build_minute_window_memmap,
    build_prediction_frame,
    compute_second_anchor_features,
    contiguous_experiment_slices,
    ensure_minute_completeness_fields,
    minute_ramp_target,
    materialize_chronological_second_data,
    materialize_second_memmaps,
    persistence_prediction,
    s1_target,
    s2_target,
)


def test_s1_target_alignment_on_manual_sequence() -> None:
    power = np.array([0.0, 1.0, 3.0, 2.0, 5.0])
    np.testing.assert_allclose(s1_target(power), [0.2, -0.1, 0.3])


def test_s2_is_future_one_second_ramp_not_net_sixty_second_change() -> None:
    power = np.arange(0.0, 63.0)
    power[60] = 80.0
    expected_first = (power[60] - power[59]) / 10.0
    assert s2_target(power)[0] == expected_first
    assert expected_first != (power[60] - power[0]) / 10.0


def test_backward_and_minute_target_normalization() -> None:
    power = np.array([0.0, 1.0, 3.0, 2.0, 5.0])
    expected = [np.nan, 0.1, 0.2, -0.1, 0.3]
    np.testing.assert_allclose(backward_ramp(power), expected, equal_nan=True)
    np.testing.assert_allclose(minute_ramp_target(power), expected, equal_nan=True)


def test_persistence_copies_current_ramp() -> None:
    current = np.array([-0.2, 0.0, 0.4])
    predicted = persistence_prediction(current)
    np.testing.assert_array_equal(predicted, current)
    assert predicted is not current


def _synthetic_second_block() -> tuple[np.ndarray, list[str]]:
    columns = [
        "P_Solar[kW]",
        "POA Irr[kW1m2]",
        "GHI[kW1m2]",
        "Elevation[deg]",
        "Azimuth[deg]",
        "TEMPERATURE[degC]",
        "HUMIDITY[%]",
        "WIND_SPEED[m1s]",
        "WIND_DIR[deg]",
    ]
    rows = 700
    values = np.empty((rows, len(columns)), dtype=np.float64)
    t = np.arange(rows, dtype=np.float64)
    values[:, 0] = 4.0 + np.sin(t / 20.0)
    values[:, 1] = 0.5 + 0.1 * np.sin(t / 25.0)
    values[:, 2] = 0.6 + 0.1 * np.sin(t / 30.0)
    values[:, 3] = 30.0
    values[:, 4] = 180.0
    values[:, 5] = 25.0
    values[:, 6] = 0.5
    values[:, 7] = 2.0
    values[:, 8] = 90.0
    return values, columns


def test_second_features_are_causal() -> None:
    values, columns = _synthetic_second_block()
    anchor = np.array([500])
    selected = columns
    first, names = compute_second_anchor_features(
        values,
        anchor,
        columns,
        selected,
        [1, 2, 5, 10, 15, 30, 60, 120, 300, 480],
        [5, 15, 30, 60, 120, 300, 480],
        [60, 480],
        10.0,
    )
    changed = values.copy()
    changed[501:] = 1.0e6
    second, second_names = compute_second_anchor_features(
        changed,
        anchor,
        columns,
        selected,
        [1, 2, 5, 10, 15, 30, 60, 120, 300, 480],
        [5, 15, 30, 60, 120, 300, 480],
        [60, 480],
        10.0,
    )
    assert names == second_names
    np.testing.assert_allclose(first, second)


def test_minute_aggregation_uses_completed_seconds_only() -> None:
    columns = list(SECOND_VARIABLES.values())
    values = np.zeros((60, len(columns)), dtype=np.float64)
    values[:, 0] = np.arange(60, dtype=np.float64)
    values[:, 1] = 1.0
    values[:, 2] = 2.0
    timestamps = pd.date_range("2020-01-01", periods=60, freq="s").to_numpy(dtype="datetime64[ns]").astype(np.int64)
    frame = _minute_descriptor_frame(timestamps, values, {name: i for i, name in enumerate(columns)}, 10.0)
    assert frame.loc[0, "pv_mean"] == 29.5
    assert frame.loc[0, "pv_first"] == 0.0
    assert frame.loc[0, "pv_last"] == 59.0
    assert frame.loc[0, "pv_max_abs_ramp"] == 0.1
    assert frame.loc[0, "seconds_observed"] == 60
    assert frame.loc[0, "valid_seconds_observed"] == 60
    assert bool(frame.loc[0, "complete_minute"])
    assert frame.loc[0, "excluded_reason"] == ""


def test_existing_descriptor_checkpoint_can_be_upgraded_with_completeness_fields(tmp_path: Path) -> None:
    path = tmp_path / "descriptors.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=2, freq="min"),
            "pv_mean": [1.0, 2.0],
            "second_count": np.array([60, 1], dtype=np.int16),
        }
    ).to_parquet(path, index=False)
    report = ensure_minute_completeness_fields(path)
    upgraded = pd.read_parquet(path)
    assert report["complete_minute_rows"] == 1
    assert report["incomplete_minute_rows"] == 1
    assert bool(upgraded.loc[0, "complete_minute"])
    assert not bool(upgraded.loc[1, "complete_minute"])
    assert upgraded.loc[1, "excluded_reason"] == "fewer_than_60_one_second_observations"


def test_minute_windows_exclude_incomplete_target_from_both_comparison_arms(tmp_path: Path) -> None:
    row_count = 9
    feature_path = tmp_path / "features.npy"
    target_path = tmp_path / "target.npy"
    timestamp_path = tmp_path / "timestamps.npy"
    np.save(feature_path, np.arange(row_count, dtype=np.float32).reshape(-1, 1))
    np.save(target_path, np.arange(row_count, dtype=np.float64))
    timestamps = pd.date_range("2020-01-01", periods=row_count, freq="min").to_numpy(dtype="datetime64[ns]").astype(np.int64)
    np.save(timestamp_path, timestamps)
    complete_path = tmp_path / "complete.npy"
    seconds_path = tmp_path / "seconds.npy"
    valid_seconds_path = tmp_path / "valid_seconds.npy"
    complete = np.ones(row_count, dtype=bool)
    complete[-1] = False
    np.save(complete_path, complete)
    np.save(seconds_path, np.array([60] * 8 + [1], dtype=np.int16))
    np.save(valid_seconds_path, np.array([60] * 8 + [1], dtype=np.int16))
    split_summary = {
        "train": {"row_count": 3, "start_row_in_dataset": 0, "end_row_exclusive_in_dataset": 3},
        "validation": {"row_count": 3, "start_row_in_dataset": 3, "end_row_exclusive_in_dataset": 6},
        "test": {"row_count": 3, "start_row_in_dataset": 6, "end_row_exclusive_in_dataset": 9},
    }
    official = {
        "feature_path": str(feature_path),
        "target_path": str(target_path),
        "timestamp_path": str(timestamp_path),
        "feature_names": ["x"],
        "split_summary": split_summary,
    }
    completeness = {
        "complete_minute_mask_path": str(complete_path),
        "seconds_observed_path": str(seconds_path),
        "valid_seconds_observed_path": str(valid_seconds_path),
        "completeness_requirement": "exactly 60 observations",
    }
    matrix = build_minute_window_memmap(
        official,
        None,
        tmp_path / "matrix",
        lookback=2,
        horizon=1,
        label="reference",
        completeness_manifest=completeness,
    )
    assert matrix["raw_split_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert matrix["split_counts"] == {"train": 1, "validation": 1, "test": 0}
    assert matrix["excluded_samples"][0]["excluded_reason"] == "incomplete_target_minute"
    assert matrix["excluded_samples"][0]["target_seconds_observed"] == 1


def test_second_memmap_maps_singular_parquet_split_code(tmp_path: Path) -> None:
    part = tmp_path / "part_0000.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=3, freq="30s"),
            "split_code": np.array([0, 1, 2], dtype=np.int8),
            "y_s1": [0.1, 0.2, 0.3],
            "y_s2": [0.4, 0.5, 0.6],
            "current_ramp": [0.0, 0.1, 0.2],
            "solar_elevation_deg": [1.0, 2.0, 3.0],
            "irradiance_variability_60s": [0.01, 0.02, 0.03],
            "current__x": [4.0, 5.0, 6.0],
        }
    ).to_parquet(part, index=False)
    manifest = {
        "parts": [{"part": 0, "path": str(part), "rows": 3}],
        "rows": 3,
        "feature_names": ["current__x"],
    }
    matrix = materialize_second_memmaps(manifest, tmp_path / "matrix")
    split_codes = np.load(matrix["metadata_paths"]["split_codes"])
    np.testing.assert_array_equal(split_codes, [0, 1, 2])
    assert matrix["split_counts"] == {"train": 1, "validation": 1, "test": 1}


def test_split_boundaries_follow_official_minute_starts() -> None:
    manifest = {
        "split_summary": {
            "train": {"start_timestamp": "2020-01-01T00:00:00"},
            "validation": {"start_timestamp": "2020-01-02T00:00:00"},
            "test": {"start_timestamp": "2020-01-03T00:00:00"},
        }
    }
    boundaries = boundaries_from_minute_manifest(manifest, pd.Timestamp("2020-01-04T00:00:00"))
    assert boundaries[0].end_exclusive == pd.Timestamp("2020-01-02T00:00:00")
    assert boundaries[1].end_exclusive == pd.Timestamp("2020-01-03T00:00:00")
    assert boundaries[2].end_exclusive == pd.Timestamp("2020-01-04T00:00:01")


def test_split_slices_reject_noncontiguous_rows() -> None:
    split = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
    target = np.arange(6, dtype=np.float64)
    slices = contiguous_experiment_slices(split, target)
    assert slices["train"] == slice(0, 2)
    target[2] = np.nan
    slices = contiguous_experiment_slices(split, target)
    assert slices["validation"] == slice(3, 4)


def test_prediction_output_schema_and_timestamp_ordering() -> None:
    timestamps = pd.date_range("2020-01-01", periods=3, freq="min").to_numpy(dtype="datetime64[ns]").astype(np.int64)
    frame = build_prediction_frame(
        timestamps,
        "test",
        np.array([0.0, 0.02, -0.02]),
        np.array([0.0, 0.01, -0.03]),
        0.01,
        np.array([0.0, 0.01, np.inf]),
    )
    required = {
        "timestamp",
        "split",
        "y_true",
        "y_pred",
        "residual",
        "absolute_error",
        "squared_error",
        "target_magnitude_bin",
        "event_category",
    }
    assert required.issubset(frame.columns)
    assert frame["timestamp"].is_monotonic_increasing
    assert not frame["timestamp"].duplicated().any()


def test_shuffled_fixed_hdf_is_normalized_chronologically(tmp_path: Path) -> None:
    timestamps = pd.date_range("2020-01-01", periods=5, freq="s")
    order = [2, 3, 0, 1, 4]
    frame = pd.DataFrame(
        {
            "P_Solar[kW]": np.arange(5, dtype=float),
            "POA Irr[kW1m2]": np.arange(5, dtype=float),
            "GHI[kW1m2]": np.arange(5, dtype=float),
            "Elevation[deg]": np.ones(5),
            "Azimuth[deg]": np.ones(5),
            "TEMPERATURE[degC]": np.ones(5),
            "HUMIDITY[%]": np.ones(5),
            "WIND_SPEED[m1s]": np.ones(5),
            "WIND_DIR[deg]": np.ones(5),
        },
        index=timestamps,
    ).iloc[order]
    source = tmp_path / "shuffled.h5"
    frame.to_hdf(source, key="/DATA", format="fixed")
    manifest = materialize_chronological_second_data(source, tmp_path / "normalized", chunk_rows=2)
    assert manifest["timestamp_reordering_required"]
    assert manifest["duplicate_timestamp_rows"] == 0
    assert manifest["missing_chronological_rows"] == 0
    with ChronologicalSecondReader(manifest) as reader:
        ts, values = reader.read(0, 5)
        assert pd.DatetimeIndex(ts).is_monotonic_increasing
        np.testing.assert_allclose(values[:, reader.column_indices["P_Solar[kW]"]], np.arange(5))
