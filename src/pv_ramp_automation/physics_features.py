from __future__ import annotations

import json
import math
import os
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config, project_root
from .reports import write_csv, write_json
from .status import section_status, utc_now, write_status


NOTEBOOK_NAME = "03_physics_informed_features"
TIME_COLUMN = "timestamp"


def safe_name(column: str) -> str:
    return (
        column.lower()
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("%", "pct")
        .replace("1", "1")
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def move_target_last(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    columns = [column for column in df.columns if column != target_column]
    if target_column not in df.columns:
        raise ValueError(f"Target column missing: {target_column}")
    return df[columns + [target_column]]


def write_feature_csv(path: str | Path, df: pd.DataFrame) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, float_format="%.10g")
    return target


def ensure_input_frame(path: Path, target_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if TIME_COLUMN not in df.columns:
        raise ValueError(f"Missing {TIME_COLUMN} in {path}")
    if target_column not in df.columns:
        raise ValueError(f"Missing {target_column} in {path}")
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    if int(df[TIME_COLUMN].isna().sum()) > 0:
        raise ValueError(f"Invalid timestamps in {path}")
    if not df[TIME_COLUMN].is_monotonic_increasing:
        df = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    if df.columns[-1] != target_column:
        df = move_target_last(df, target_column)
    return df


def add_feature(
    df: pd.DataFrame,
    dictionary: list[dict[str, Any]],
    name: str,
    values: pd.Series | np.ndarray,
    family: str,
    source_columns: list[str],
    description: str,
    causal_rule: str,
    uses_target: bool = False,
) -> None:
    df[name] = values
    dictionary.append(
        {
            "feature_name": name,
            "family": family,
            "source_columns": source_columns,
            "description": description,
            "causal_rule": causal_rule,
            "uses_target": uses_target,
            "created_in_notebook": NOTEBOOK_NAME,
        }
    )


def maybe_numeric(df: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in df.columns:
        return None
    return pd.to_numeric(df[column], errors="coerce")


def add_time_features(df: pd.DataFrame, dictionary: list[dict[str, Any]], elevation_threshold: float) -> list[str]:
    created: list[str] = []
    ts = pd.to_datetime(df[TIME_COLUMN])
    values: dict[str, pd.Series | np.ndarray] = {
        "time_hour": ts.dt.hour.astype(float) + ts.dt.minute.astype(float) / 60.0,
        "time_dayofyear": ts.dt.dayofyear.astype(float),
        "time_month": ts.dt.month.astype(float),
        "time_hour_sin": np.sin(2 * np.pi * (ts.dt.hour.astype(float) + ts.dt.minute.astype(float) / 60.0) / 24.0),
        "time_hour_cos": np.cos(2 * np.pi * (ts.dt.hour.astype(float) + ts.dt.minute.astype(float) / 60.0) / 24.0),
        "time_dayofyear_sin": np.sin(2 * np.pi * ts.dt.dayofyear.astype(float) / 366.0),
        "time_dayofyear_cos": np.cos(2 * np.pi * ts.dt.dayofyear.astype(float) / 366.0),
    }
    elevation = maybe_numeric(df, "Elevation[deg]")
    if elevation is not None:
        values["solar_daylight_flag"] = (elevation > elevation_threshold).astype(int)
    for name, series in values.items():
        add_feature(
            df,
            dictionary,
            name,
            series,
            "solar_geometry_time",
            [TIME_COLUMN] if name.startswith("time_") else ["Elevation[deg]"],
            "Causal calendar/solar-state feature available at forecast origin.",
            "uses timestamp or current solar elevation only; no future information",
        )
        created.append(name)
    return created


def add_lag_delta_rolling_features(
    df: pd.DataFrame,
    dictionary: list[dict[str, Any]],
    source_columns: list[str],
    lags: list[int],
    windows: list[int],
) -> list[str]:
    created: list[str] = []
    for column in source_columns:
        series = maybe_numeric(df, column)
        if series is None:
            continue
        base = safe_name(column)
        for lag in lags:
            name = f"{base}_lag{lag}"
            add_feature(
                df,
                dictionary,
                name,
                series.shift(lag),
                "irradiance_pv_dynamics",
                [column],
                f"Past lag of {column} by {lag} minute step(s).",
                f"strictly past shift(+{lag}); no negative shift",
            )
            created.append(name)
        for lag in lags:
            name = f"{base}_delta{lag}"
            add_feature(
                df,
                dictionary,
                name,
                series - series.shift(lag),
                "ramp_specific_past_dynamics",
                [column],
                f"Current minus past {column} over {lag} minute step(s).",
                f"current t minus t-{lag}; no future delta",
            )
            created.append(name)
            if lag == 1:
                abs_name = f"{base}_delta{lag}_abs"
                add_feature(
                    df,
                    dictionary,
                    abs_name,
                    (series - series.shift(lag)).abs(),
                    "ramp_specific_past_dynamics",
                    [column],
                    f"Absolute current-minus-past change of {column} over {lag} minute step(s).",
                    f"current t minus t-{lag}; no future delta",
                )
                created.append(abs_name)
                sign_name = f"{base}_delta{lag}_sign"
                add_feature(
                    df,
                    dictionary,
                    sign_name,
                    np.sign(series - series.shift(lag)),
                    "ramp_specific_past_dynamics",
                    [column],
                    f"Sign of current-minus-past change of {column} over {lag} minute step(s).",
                    f"current t minus t-{lag}; no future delta",
                )
                created.append(sign_name)
        for window in windows:
            rolling = series.rolling(window=window, min_periods=window)
            feature_specs = {
                f"{base}_roll_mean{window}": rolling.mean(),
                f"{base}_roll_std{window}": rolling.std(),
            }
            if window == max(windows):
                feature_specs[f"{base}_roll_min{window}"] = rolling.min()
                feature_specs[f"{base}_roll_max{window}"] = rolling.max()
            for name, values in feature_specs.items():
                add_feature(
                    df,
                    dictionary,
                    name,
                    values,
                    "irradiance_pv_rolling_context",
                    [column],
                    f"Past/current rolling statistic over {window} one-minute rows.",
                    f"rolling window includes t-{window - 1} through t only; no centered or future window",
                )
                created.append(name)
            if window == max(windows):
                variability = rolling.std() / (rolling.mean().abs() + 1.0e-6)
                name = f"{base}_roll_var_proxy{window}"
                add_feature(
                    df,
                    dictionary,
                    name,
                    variability,
                    "irradiance_pv_rolling_context",
                    [column],
                    f"Rolling variability proxy for {column} over {window} rows.",
                    f"rolling window includes t-{window - 1} through t only; no centered or future window",
                )
                created.append(name)
    return created


def add_operating_point_features(df: pd.DataFrame, dictionary: list[dict[str, Any]], rated_power_kw: float, eps: float, saturation_fraction: float) -> list[str]:
    created: list[str] = []
    pv = maybe_numeric(df, "P_Solar[kW]")
    poa = maybe_numeric(df, "POA Irr[kW1m2]")
    ghi = maybe_numeric(df, "GHI[kW1m2]")
    if pv is not None:
        add_feature(df, dictionary, "pv_power_pu", pv / rated_power_kw, "pv_operating_point", ["P_Solar[kW]"], "PV power normalized by rated power.", "uses current measured PV power at t only")
        add_feature(df, dictionary, "pv_saturation_flag", (pv >= saturation_fraction * rated_power_kw).astype(int), "pv_operating_point", ["P_Solar[kW]"], "Simple clipping/saturation proxy near rated power.", "uses current measured PV power at t only")
        add_feature(df, dictionary, "pv_saturation_margin_pu", np.maximum(0.0, 1.0 - (pv / rated_power_kw)), "pv_operating_point", ["P_Solar[kW]"], "Remaining margin to rated power in per unit.", "uses current measured PV power at t only")
        created.extend(["pv_power_pu", "pv_saturation_flag", "pv_saturation_margin_pu"])
    if pv is not None and poa is not None:
        add_feature(df, dictionary, "pv_to_poa_ratio", pv / (poa.abs() + eps), "pv_operating_point", ["P_Solar[kW]", "POA Irr[kW1m2]"], "PV power to POA irradiance consistency ratio.", "uses current t only with safe epsilon")
        add_feature(df, dictionary, "pv_pu_to_poa_ratio", (pv / rated_power_kw) / (poa.abs() + eps), "pv_operating_point", ["P_Solar[kW]", "POA Irr[kW1m2]"], "Normalized PV power to POA irradiance ratio.", "uses current t only with safe epsilon")
        created.extend(["pv_to_poa_ratio", "pv_pu_to_poa_ratio"])
    if pv is not None and ghi is not None:
        add_feature(df, dictionary, "pv_to_ghi_ratio", pv / (ghi.abs() + eps), "pv_operating_point", ["P_Solar[kW]", "GHI[kW1m2]"], "PV power to GHI irradiance consistency ratio.", "uses current t only with safe epsilon")
        created.append("pv_to_ghi_ratio")
    if poa is not None and ghi is not None:
        add_feature(df, dictionary, "poa_to_ghi_ratio", poa / (ghi.abs() + eps), "pv_operating_point", ["POA Irr[kW1m2]", "GHI[kW1m2]"], "POA to GHI transposition/consistency proxy.", "uses current t only with safe epsilon")
        created.append("poa_to_ghi_ratio")
    if pv is not None and poa is not None and ghi is not None:
        add_feature(df, dictionary, "pv_irradiance_consistency_gap", (pv / rated_power_kw) - ((poa + ghi) / 2.0), "pv_operating_point", ["P_Solar[kW]", "POA Irr[kW1m2]", "GHI[kW1m2]"], "Simple gap between normalized PV power and mean irradiance proxy.", "uses current t only")
        created.append("pv_irradiance_consistency_gap")
    return created


def add_weather_interaction_features(df: pd.DataFrame, dictionary: list[dict[str, Any]], nominal_temp: float) -> list[str]:
    created: list[str] = []
    temp = maybe_numeric(df, "TEMPERATURE[degC]")
    wind = maybe_numeric(df, "WIND_SPEED[m1s]")
    humidity = maybe_numeric(df, "HUMIDITY[%]")
    poa = maybe_numeric(df, "POA Irr[kW1m2]")
    ghi = maybe_numeric(df, "GHI[kW1m2]")
    if temp is not None:
        add_feature(df, dictionary, "temperature_deviation_from_25c", temp - nominal_temp, "temperature_wind_effects", ["TEMPERATURE[degC]"], "Temperature deviation from nominal module condition proxy.", "uses current t only")
        created.append("temperature_deviation_from_25c")
    if temp is not None and poa is not None:
        add_feature(df, dictionary, "temperature_x_poa", (temp - nominal_temp) * poa, "temperature_wind_effects", ["TEMPERATURE[degC]", "POA Irr[kW1m2]"], "Temperature and POA interaction.", "uses current t only")
        created.append("temperature_x_poa")
    if temp is not None and ghi is not None:
        add_feature(df, dictionary, "temperature_x_ghi", (temp - nominal_temp) * ghi, "temperature_wind_effects", ["TEMPERATURE[degC]", "GHI[kW1m2]"], "Temperature and GHI interaction.", "uses current t only")
        created.append("temperature_x_ghi")
    if temp is not None and wind is not None:
        add_feature(df, dictionary, "wind_cooling_proxy", wind * (temp - nominal_temp), "temperature_wind_effects", ["WIND_SPEED[m1s]", "TEMPERATURE[degC]"], "Wind cooling interaction proxy.", "uses current t only")
        created.append("wind_cooling_proxy")
    if wind is not None and poa is not None:
        add_feature(df, dictionary, "wind_x_poa", wind * poa, "temperature_wind_effects", ["WIND_SPEED[m1s]", "POA Irr[kW1m2]"], "Wind and POA interaction.", "uses current t only")
        created.append("wind_x_poa")
    if humidity is not None and poa is not None:
        add_feature(df, dictionary, "humidity_x_poa", humidity * poa, "temperature_wind_effects", ["HUMIDITY[%]", "POA Irr[kW1m2]"], "Humidity and POA interaction.", "uses current t only")
        created.append("humidity_x_poa")
    if humidity is not None and ghi is not None:
        add_feature(df, dictionary, "humidity_x_ghi", humidity * ghi, "temperature_wind_effects", ["HUMIDITY[%]", "GHI[kW1m2]"], "Humidity and GHI interaction.", "uses current t only")
        created.append("humidity_x_ghi")
    return created


def engineer_features(df: pd.DataFrame, target_column: str, config: dict[str, Any], dataset_variant: str) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any], list[str]]:
    cfg = config.get("physics_features", {})
    rated_power_kw = float(config["project"]["rated_power_kw"])
    lags = [int(value) for value in cfg.get("lag_steps_minutes", [1, 5, 15])]
    windows = [int(value) for value in cfg.get("rolling_windows_minutes", [5, 15, 60])]
    eps = float(cfg.get("safe_division_epsilon", 1.0e-6))
    elevation_threshold = float(cfg.get("daylight_elevation_threshold_deg", 0.0))
    nominal_temp = float(cfg.get("nominal_temperature_degC", 25.0))
    saturation_fraction = float(cfg.get("saturation_power_fraction", 0.98))

    working = move_target_last(df.copy(), target_column)
    dictionary: list[dict[str, Any]] = []
    created_features: list[str] = []
    created_features.extend(add_time_features(working, dictionary, elevation_threshold))
    dynamic_sources = [column for column in ["POA Irr[kW1m2]", "GHI[kW1m2]", "P_Solar[kW]"] if column in working.columns]
    created_features.extend(add_lag_delta_rolling_features(working, dictionary, dynamic_sources, lags, windows))
    created_features.extend(add_operating_point_features(working, dictionary, rated_power_kw, eps, saturation_fraction))
    created_features.extend(add_weather_interaction_features(working, dictionary, nominal_temp))

    target_values = working[target_column].copy()
    working = working.drop(columns=[target_column])
    working[target_column] = target_values

    engineered_columns = [feature for feature in created_features if feature in working.columns]
    before_rows = int(len(working))
    nan_counts_before_drop = {column: int(working[column].isna().sum()) for column in engineered_columns if int(working[column].isna().sum()) > 0}
    inf_counts_before_cleanup = {
        column: int(np.isinf(pd.to_numeric(working[column], errors="coerce").to_numpy(dtype=float)).sum())
        for column in engineered_columns
    }
    inf_counts_before_cleanup = {column: count for column, count in inf_counts_before_cleanup.items() if count > 0}
    working[engineered_columns] = working[engineered_columns].replace([np.inf, -np.inf], np.nan)
    nan_columns_after_inf_cleanup = working[engineered_columns].isna().sum()
    drop_mask = working[engineered_columns].isna().any(axis=1)
    dropped_rows = int(drop_mask.sum())
    cleaned = working.loc[~drop_mask].copy().reset_index(drop=True)
    cleaned = move_target_last(cleaned, target_column)

    metadata = {
        "dataset_variant": dataset_variant,
        "rows_before_feature_engineering": before_rows,
        "rows_after_feature_engineering": int(len(cleaned)),
        "rows_dropped_for_lag_rolling_warmup_or_invalid_ratios": dropped_rows,
        "base_columns_before": int(df.shape[1]),
        "columns_after": int(cleaned.shape[1]),
        "engineered_feature_count": len(engineered_columns),
        "lag_steps_minutes": lags,
        "rolling_windows_minutes": windows,
        "max_lag_or_rolling_warmup_rows": int(max(max(lags, default=0), max(windows, default=1) - 1)),
        "nan_counts_before_drop": nan_counts_before_drop,
        "infinite_counts_before_cleanup": inf_counts_before_cleanup,
        "target_lag_features_created": False,
        "target_used_as_input_feature": False,
    }
    return cleaned, dictionary, metadata, engineered_columns


def leakage_checks(df: pd.DataFrame, feature_dictionary: list[dict[str, Any]], target_column: str) -> dict[str, Any]:
    feature_columns = [column for column in df.columns if column not in (TIME_COLUMN, target_column)]
    lower = [column.lower() for column in feature_columns]
    future_markers = ["future", "lead", "t_plus", "ahead"]
    centered_markers = ["centered", "rolling_center", "roll_center"]
    negative_shift_markers = ["shift-",
                              "lag-",
                              "minus_lag"]
    same_row_target_features = [item["feature_name"] for item in feature_dictionary if item.get("uses_target")]
    future_features = [column for column, low in zip(feature_columns, lower) if any(marker in low for marker in future_markers)]
    centered_features = [column for column, low in zip(feature_columns, lower) if any(marker in low for marker in centered_markers)]
    negative_shift_features = [column for column, low in zip(feature_columns, lower) if any(marker in low for marker in negative_shift_markers)]
    target_named_features = [column for column in feature_columns if target_column.lower() in column.lower()]
    checks = {
        "target_exists": target_column in df.columns,
        "target_last_column": bool(df.columns[-1] == target_column),
        "timestamps_sorted": bool(pd.to_datetime(df[TIME_COLUMN]).is_monotonic_increasing),
        "no_future_feature_names": len(future_features) == 0,
        "no_centered_rolling_feature_names": len(centered_features) == 0,
        "no_negative_shift_feature_names": len(negative_shift_features) == 0,
        "no_same_row_target_features": len(same_row_target_features) == 0,
        "no_target_named_input_features": len(target_named_features) == 0,
        "no_infinite_values": bool(np.isinf(df.select_dtypes(include="number").to_numpy(dtype=float)).sum() == 0),
        "future_feature_names": future_features,
        "centered_rolling_feature_names": centered_features,
        "negative_shift_feature_names": negative_shift_features,
        "same_row_target_features": same_row_target_features,
        "target_named_input_features": target_named_features,
    }
    checks["passed"] = all(value for key, value in checks.items() if isinstance(value, bool))
    return checks


def feature_report(dictionary: list[dict[str, Any]], selected_features: set[str]) -> pd.DataFrame:
    rows = []
    for item in dictionary:
        row = dict(item)
        row["in_selected_plus_physics_dataset"] = item["feature_name"] in selected_features
        rows.append(row)
    return pd.DataFrame(rows)


def save_diagnostic_plots(df: pd.DataFrame, output_dir: Path, engineered_columns: list[str]) -> dict[str, str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    family_candidates = [column for column in engineered_columns if column.endswith("_delta1") or column in ("pv_power_pu", "poa_to_ghi_ratio")]
    sample = df.iloc[:: max(1, len(df) // 20000)].copy()
    for column in family_candidates[:4]:
        if column not in sample.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(pd.to_datetime(sample[TIME_COLUMN]), sample[column], linewidth=0.7)
        ax.set_title(column)
        ax.set_xlabel("timestamp")
        ax.set_ylabel(column)
        fig.autofmt_xdate()
        fig.tight_layout()
        path = plot_dir / f"{column}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths[column] = str(path)
    return paths


def write_local_runner_report(report: dict[str, Any]) -> None:
    output_dir = os.environ.get("LOCAL_OUTPUT_DIR")
    if not output_dir:
        return
    payload = {
        "notebook_name": report["notebook_name"],
        "final_status": report["final_status"],
        "data_shapes": report["data_shapes"],
        "split_summary": {
            "official_split_created": False,
            "lookback_horizon_windows_created": False,
            "notebook_04_responsible_for_split_scaling_windowing": True,
        },
        "metrics": {
            "selected_engineered_feature_count": report["selected_feature_counts"]["after_physics_enrichment"],
            "full_engineered_feature_count": report["full_feature_counts"]["after_physics_enrichment"],
            "selected_rows_after": report["selected_dataset"]["rows_after_feature_engineering"],
            "full_rows_after": report["full_dataset"]["rows_after_feature_engineering"],
        },
        "report_path": report["output_files"]["notebook_03_final_report"],
    }
    write_json(Path(output_dir) / "local_clearml_report.json", payload)


def run_notebook03_physics_features(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    root = project_root()
    output_root = root / "outputs" / NOTEBOOK_NAME
    run_state = output_root / "run_state"
    output_root.mkdir(parents=True, exist_ok=True)
    run_state.mkdir(parents=True, exist_ok=True)

    target_column = str(cfg.get("target", {}).get("official_name") or cfg["data"]["official_target"])
    primary_resolution = str(cfg["data"]["primary_resolution_candidate"])
    dataset_dir = root / "outputs" / primary_resolution / "notebook_03_physics_features"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_files: dict[str, str] = {}
    warnings: list[str] = []
    errors: list[str] = []

    started = utc_now()
    try:
        nb02_report = read_json(root / "outputs" / "02_target_construction_feature_selection" / "notebook_02_final_report.json")
        if nb02_report.get("final_status") != "success":
            raise RuntimeError("Notebook 02 final report is not successful.")
        full_input = Path(nb02_report["output_files"]["original_data_all_features_clean_with_target"])
        selected_input = Path(nb02_report["output_files"]["selected_features_with_target"])
        full_df = ensure_input_frame(full_input, target_column)
        selected_df = ensure_input_frame(selected_input, target_column)
        write_status(
            run_state / "section_01_input_validation.json",
            section_status(
                "input_validation",
                started,
                "success",
                "Notebook 02 full and selected target datasets validated.",
                {
                    "full_input": str(full_input),
                    "selected_input": str(selected_input),
                    "full_rows": int(len(full_df)),
                    "selected_rows": int(len(selected_df)),
                    "target_last_full": bool(full_df.columns[-1] == target_column),
                    "target_last_selected": bool(selected_df.columns[-1] == target_column),
                },
            ),
        )
    except Exception:
        errors.append(traceback.format_exc())
        write_status(run_state / "section_01_input_validation.json", section_status("input_validation", started, "failed", "Notebook 03 input validation failed.", error_message=errors[-1]))
        raise

    started = utc_now()
    try:
        full_enriched, full_dict, full_meta, full_engineered = engineer_features(full_df, target_column, cfg, "full_plus_physics")
        selected_base_columns = [column for column in selected_df.columns if column not in (TIME_COLUMN, target_column)]
        selected_columns = [TIME_COLUMN] + selected_base_columns + full_engineered + [target_column]
        selected_enriched = full_enriched[selected_columns].copy()
        selected_meta = {
            "dataset_variant": "selected_plus_physics",
            "rows_before_feature_engineering": int(selected_df.shape[0]),
            "rows_after_feature_engineering": int(selected_enriched.shape[0]),
            "rows_dropped_for_lag_rolling_warmup_or_invalid_ratios": int(selected_df.shape[0] - selected_enriched.shape[0]),
            "base_columns_before": int(selected_df.shape[1]),
            "columns_after": int(selected_enriched.shape[1]),
            "engineered_feature_count": len(full_engineered),
            "lag_steps_minutes": full_meta["lag_steps_minutes"],
            "rolling_windows_minutes": full_meta["rolling_windows_minutes"],
            "max_lag_or_rolling_warmup_rows": full_meta["max_lag_or_rolling_warmup_rows"],
            "nan_counts_before_drop": full_meta["nan_counts_before_drop"],
            "infinite_counts_before_cleanup": full_meta["infinite_counts_before_cleanup"],
            "target_lag_features_created": False,
            "target_used_as_input_feature": False,
        }
        selected_engineered = list(full_engineered)
        full_output = dataset_dir / "original_all_features_plus_physics_informed_with_target.csv"
        selected_output = dataset_dir / "selected_features_plus_physics_informed_with_target.csv"
        write_feature_csv(full_output, full_enriched)
        write_feature_csv(selected_output, selected_enriched)
        output_files["original_all_features_plus_physics_informed_with_target"] = str(full_output)
        output_files["selected_features_plus_physics_informed_with_target"] = str(selected_output)
        write_status(
            run_state / "section_02_feature_engineering.json",
            section_status(
                "feature_engineering",
                started,
                "success",
                "Physics-informed causal features generated for full and selected datasets.",
                {
                    "full_engineered_feature_count": len(full_engineered),
                    "selected_engineered_feature_count": len(selected_engineered),
                    "full_rows_after": int(len(full_enriched)),
                    "selected_rows_after": int(len(selected_enriched)),
                },
            ),
        )
    except Exception:
        errors.append(traceback.format_exc())
        write_status(run_state / "section_02_feature_engineering.json", section_status("feature_engineering", started, "failed", "Feature engineering failed.", error_message=errors[-1]))
        raise

    started = utc_now()
    try:
        full_checks = leakage_checks(full_enriched, full_dict, target_column)
        selected_checks = leakage_checks(selected_enriched, full_dict, target_column)
        all_checks_passed = bool(full_checks["passed"] and selected_checks["passed"])
        leakage_report = {
            "full_dataset": full_checks,
            "selected_dataset": selected_checks,
            "rules": {
                "no_future_information": True,
                "no_centered_rolling": True,
                "no_negative_shift": True,
                "no_future_target_columns": True,
                "target_lags_created": False,
                "same_row_target_used_as_input": False,
                "notebook_04_must_create_supervised_alignment": True,
            },
            "passed": all_checks_passed,
        }
        leakage_path = output_root / "physics_informed_leakage_check.json"
        write_json(leakage_path, leakage_report)
        output_files["physics_informed_leakage_check"] = str(leakage_path)
        write_status(
            run_state / "section_03_leakage_checks.json",
            section_status(
                "leakage_checks",
                started,
                "success" if all_checks_passed else "failed",
                "Causal feature leakage checks completed.",
                {"passed": all_checks_passed},
                None if all_checks_passed else "One or more leakage checks failed.",
            ),
        )
    except Exception:
        errors.append(traceback.format_exc())
        write_status(run_state / "section_03_leakage_checks.json", section_status("leakage_checks", started, "failed", "Leakage checks failed.", error_message=errors[-1]))
        raise

    started = utc_now()
    try:
        selected_feature_names = set(selected_enriched.columns) - {TIME_COLUMN, target_column}
        report_df = feature_report(full_dict, selected_feature_names)
        report_path = output_root / "physics_informed_feature_report.csv"
        dictionary_path = output_root / "feature_dictionary.json"
        dropped_path = output_root / "physics_feature_rejected_report.csv"
        plots = save_diagnostic_plots(selected_enriched, output_root, selected_engineered)
        rejected = pd.DataFrame(
            [
                {
                    "feature_name": "ramp_rate_pu_lag1",
                    "family": "target_lag",
                    "reason": "Not created in Notebook 03. Past target lags are deferred unless explicitly approved because Notebook 04 windowing can expose past PV dynamics safely.",
                },
                {
                    "feature_name": "centered_rolling_features",
                    "family": "rolling",
                    "reason": "Rejected because centered rolling windows use information after time t.",
                },
                {
                    "feature_name": "future_delta_features",
                    "family": "ramp",
                    "reason": "Rejected because future deltas would leak information after forecast origin.",
                },
            ]
        )
        write_csv(report_path, report_df)
        write_json(dictionary_path, {"features": full_dict, "target_lags_created": False, "causality_policy": "features use information available up to time t only"})
        write_csv(dropped_path, rejected)
        output_files["physics_informed_feature_report"] = str(report_path)
        output_files["feature_dictionary"] = str(dictionary_path)
        output_files["physics_feature_rejected_report"] = str(dropped_path)
        write_status(
            run_state / "section_04_reports_plots.json",
            section_status(
                "reports_plots",
                started,
                "success",
                "Feature report, dictionary, rejected-feature report and diagnostic plots saved.",
                {"feature_dictionary_count": len(full_dict), "plot_count": len(plots)},
            ),
        )
    except Exception:
        errors.append(traceback.format_exc())
        write_status(run_state / "section_04_reports_plots.json", section_status("reports_plots", started, "failed", "Report writing failed.", error_message=errors[-1]))
        raise

    started = utc_now()
    try:
        final_checks = {
            "both_enriched_datasets_exist": Path(output_files["original_all_features_plus_physics_informed_with_target"]).exists() and Path(output_files["selected_features_plus_physics_informed_with_target"]).exists(),
            "target_last_full": bool(full_enriched.columns[-1] == target_column),
            "target_last_selected": bool(selected_enriched.columns[-1] == target_column),
            "no_future_information_full": bool(full_checks["passed"]),
            "no_future_information_selected": bool(selected_checks["passed"]),
            "timestamps_sorted_full": bool(pd.to_datetime(full_enriched[TIME_COLUMN]).is_monotonic_increasing),
            "timestamps_sorted_selected": bool(pd.to_datetime(selected_enriched[TIME_COLUMN]).is_monotonic_increasing),
            "selected_and_full_generated": True,
            "lookback_horizon_windows_not_created": True,
            "model_training_not_started": True,
        }
        final_status = "success" if all(final_checks.values()) and not errors else "failed"
        final_path = output_root / "notebook_03_final_report.json"
        output_files["notebook_03_final_report"] = str(final_path)
        report = {
            "notebook_name": NOTEBOOK_NAME,
            "execution_type": "local",
            "primary_resolution": primary_resolution,
            "input_files": {"full": str(full_input), "selected": str(selected_input)},
            "output_files": output_files,
            "target_column": target_column,
            "target_lag_features_created": False,
            "future_target_columns_created": False,
            "lookback_horizon_windows_created": False,
            "model_training_started": False,
            "full_dataset": full_meta,
            "selected_dataset": selected_meta,
            "full_feature_counts": {
                "before_physics_enrichment": int(full_df.shape[1] - 2),
                "engineered_features_added": len(full_engineered),
                "after_physics_enrichment": int(full_enriched.shape[1] - 2),
            },
            "selected_feature_counts": {
                "before_physics_enrichment": int(selected_df.shape[1] - 2),
                "engineered_features_added": len(selected_engineered),
                "after_physics_enrichment": int(selected_enriched.shape[1] - 2),
            },
            "data_shapes": {
                "full_input": {"rows": int(full_df.shape[0]), "columns": int(full_df.shape[1])},
                "selected_input": {"rows": int(selected_df.shape[0]), "columns": int(selected_df.shape[1])},
                "full_enriched": {"rows": int(full_enriched.shape[0]), "columns": int(full_enriched.shape[1])},
                "selected_enriched": {"rows": int(selected_enriched.shape[0]), "columns": int(selected_enriched.shape[1])},
            },
            "main_physics_feature_families": sorted(report_df["family"].unique().tolist()),
            "main_physics_features_created": full_engineered[:40],
            "features_rejected_or_deferred": rejected.to_dict(orient="records"),
            "leakage_checks": leakage_report,
            "validation_checks": final_checks,
            "plots": plots,
            "warnings": warnings,
            "errors": errors,
            "final_status": final_status,
        }
        write_json(final_path, report)
        write_local_runner_report(report)
        write_status(
            run_state / "section_05_final_validation.json",
            section_status(
                "final_validation",
                started,
                final_status,
                "Notebook 03 final validations completed.",
                final_checks,
                None if final_status == "success" else "One or more final validations failed.",
            ),
        )
        return report
    except Exception:
        errors.append(traceback.format_exc())
        write_status(run_state / "section_05_final_validation.json", section_status("final_validation", started, "failed", "Final validation failed.", error_message=errors[-1]))
        raise
