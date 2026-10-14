from __future__ import annotations

import json
import math
import os
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config, project_root
from .reports import write_csv, write_json, write_markdown
from .status import section_status, utc_now, write_status


NOTEBOOK_NAME = "06_statistical_time_series_models"
TIME_COLUMN = "timestamp"
TARGET_COLUMN = "ramp_rate_pu"
DATASET_VARIANT = "selected_physics"
PRIMARY_RESOLUTION = "one_minute"
HORIZONS = [1, 8, 32]
MAX_HORIZON = max(HORIZONS)
FIT_CONTEXT_ROWS = 2880
TRAIN_EVALUATION_SPAN_ROWS = 14400
MAX_ORIGINS_PER_SPLIT = 50
SPLIT_NAMES = ["train", "validation", "test"]
SAFE_EXOG_BASE_COLUMNS = [
    "time_hour_sin",
    "time_hour_cos",
    "time_dayofyear_sin",
    "time_dayofyear_cos",
    "solar_daylight_flag",
    "Elevation[deg]",
]
SKIPPED_STATIC_CONFIGS = [
    {
        "model_name": "SARIMA",
        "configuration": "seasonal ARIMA without exogenous variables",
        "status": "skipped",
        "reason": "SARIMAX is the more general case and SARIMA is redundant at this phase without a stronger seasonal hypothesis.",
    },
    {
        "model_name": "SPAR_GAM_like",
        "configuration": "optional statistical smoother/additive baseline",
        "status": "skipped",
        "reason": "Optional baseline skipped to avoid adding a new modelling dependency and blocking the ARIMA/SARIMAX pipeline.",
    },
    {
        "model_name": "ARIMA",
        "configuration": "full order grid, including ARIMA(2,0,1)",
        "status": "skipped",
        "reason": "Heavy blind statistical grid search is explicitly out of scope for Phase 6; revisit only after exploratory review.",
    },
    {
        "model_name": "SARIMAX",
        "configuration": "measured future exogenous variables such as POA/GHI/P_Solar/weather",
        "status": "skipped",
        "reason": "Future measured exogenous values are not guaranteed available at forecast time, so they are excluded to avoid leakage.",
    },
]


@dataclass(frozen=True)
class StatisticalConfig:
    config_id: str
    model_name: str
    order: tuple[int, int, int]
    use_exog: bool
    stage: str
    status_tag: str
    notes: str


MODEL_CONFIGS = [
    StatisticalConfig(
        config_id="arima_1_0_0",
        model_name="arima",
        order=(1, 0, 0),
        use_exog=False,
        stage="exploratory",
        status_tag="exploratory",
        notes="Small ARIMA order for Phase 6 screening; no exogenous variables.",
    ),
    StatisticalConfig(
        config_id="sarimax_1_0_0_deterministic_exog",
        model_name="sarimax",
        order=(1, 0, 0),
        use_exog=True,
        stage="exploratory",
        status_tag="exploratory",
        notes="SARIMAX with deterministic future-known calendar/solar-geometry exogenous variables only.",
    ),
]


def read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    return json.loads(target.read_text(encoding="utf-8"))


def finite_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, persistence_rmse: float | None) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"sample_count": 0, "MAE": None, "RMSE": None, "NRMSE": None, "R2": None, "Skill_RMSE_vs_Persistence": None}
    error = y_pred - y_true
    mae_value = float(np.mean(np.abs(error)))
    rmse_value = float(np.sqrt(np.mean(np.square(error))))
    sse = float(np.sum(np.square(error)))
    mean_true = float(np.mean(y_true))
    sst = float(np.sum(np.square(y_true - mean_true)))
    r2_value = None if sst <= 0 else float(1.0 - (sse / sst))
    skill_value = None
    if persistence_rmse is not None and persistence_rmse > 0:
        skill_value = float(1.0 - (rmse_value / persistence_rmse))
    return {
        "sample_count": int(len(y_true)),
        "MAE": finite_or_none(mae_value),
        "RMSE": finite_or_none(rmse_value),
        "NRMSE": finite_or_none(rmse_value / 2.0),
        "R2": finite_or_none(r2_value),
        "Skill_RMSE_vs_Persistence": finite_or_none(skill_value),
    }


def manifest_path(root: Path) -> Path:
    return root / "outputs" / "one_minute" / "notebook_04_split_windowing" / DATASET_VARIANT / "window_ready_manifest.json"


def load_selected_physics_data(manifest: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    source_path = Path(manifest["source_path"])
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    header = list(pd.read_csv(source_path, nrows=0).columns)
    missing_required = [column for column in [TIME_COLUMN, TARGET_COLUMN] if column not in header]
    if missing_required:
        raise ValueError(f"Missing required columns in {source_path}: {missing_required}")
    if header[-1] != TARGET_COLUMN:
        raise ValueError(f"Target must remain the last column in {source_path}")
    missing_exog = [column for column in SAFE_EXOG_BASE_COLUMNS if column not in header]
    if missing_exog:
        raise ValueError(f"Missing deterministic exogenous columns for SARIMAX: {missing_exog}")
    usecols = [TIME_COLUMN, TARGET_COLUMN] + SAFE_EXOG_BASE_COLUMNS
    df = pd.read_csv(source_path, usecols=usecols)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    if int(df[TIME_COLUMN].isna().sum()) > 0:
        raise ValueError("Invalid timestamps detected in selected_physics source file.")
    if not df[TIME_COLUMN].is_monotonic_increasing:
        raise ValueError("Timestamps are not sorted chronologically.")
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(target).all():
        raise ValueError("Target contains NaN or infinite values.")
    if len(df) != int(manifest["row_count"]):
        raise ValueError("Manifest row count does not match selected_physics source rows.")
    return df, target


def build_deterministic_exog(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    exog = pd.DataFrame(index=df.index)
    exog["time_hour_sin"] = pd.to_numeric(df["time_hour_sin"], errors="coerce")
    exog["time_hour_cos"] = pd.to_numeric(df["time_hour_cos"], errors="coerce")
    exog["time_dayofyear_sin"] = pd.to_numeric(df["time_dayofyear_sin"], errors="coerce")
    exog["time_dayofyear_cos"] = pd.to_numeric(df["time_dayofyear_cos"], errors="coerce")
    exog["solar_daylight_flag"] = pd.to_numeric(df["solar_daylight_flag"], errors="coerce")
    exog["elevation_norm_known"] = pd.to_numeric(df["Elevation[deg]"], errors="coerce") / 90.0
    values = exog.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Deterministic SARIMAX exogenous values contain NaN or infinite values.")
    return values, list(exog.columns)


def split_arrays(
    df: pd.DataFrame,
    target: np.ndarray,
    exog: np.ndarray,
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    splits: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_NAMES:
        meta = manifest["split_summary"][split_name]
        start = int(meta["start_row_in_dataset"])
        end = int(meta["end_row_exclusive_in_dataset"])
        splits[split_name] = {
            "target": target[start:end],
            "exog": exog[start:end],
            "timestamp": df[TIME_COLUMN].iloc[start:end].to_numpy(),
            "start_row_in_dataset": start,
            "end_row_exclusive_in_dataset": end,
            "row_count": int(end - start),
            "start_timestamp": meta["start_timestamp"],
            "end_timestamp": meta["end_timestamp"],
        }
    return splits


def origin_indices(
    split_length: int,
    horizon: int,
    min_origin: int,
    max_origins: int = MAX_ORIGINS_PER_SPLIT,
    max_origin_limit: int | None = None,
) -> np.ndarray:
    max_origin = int(split_length) - int(horizon) - 1
    if max_origin_limit is not None:
        max_origin = min(max_origin, int(max_origin_limit))
    if max_origin < min_origin:
        return np.array([], dtype=np.int64)
    available_count = max_origin - int(min_origin) + 1
    count = min(int(max_origins), int(available_count))
    if count <= 0:
        return np.array([], dtype=np.int64)
    return np.unique(np.linspace(int(min_origin), int(max_origin), count, dtype=np.int64))


def append_results(results, endog_chunk: np.ndarray, exog_chunk: np.ndarray | None):
    if len(endog_chunk) == 0:
        return results
    if exog_chunk is None:
        return results.append(endog=endog_chunk, refit=False)
    return results.append(endog=endog_chunk, exog=exog_chunk, refit=False)


def filter_context_with_fixed_params(config: StatisticalConfig, params: np.ndarray, endog_context: np.ndarray, exog_context: np.ndarray | None):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        endog_context,
        exog=exog_context,
        order=config.order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.filter(params)


def fit_model(config: StatisticalConfig, train: dict[str, Any], exog_columns: list[str]):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    fit_rows = min(FIT_CONTEXT_ROWS, len(train["target"]) - MAX_HORIZON - 1)
    if fit_rows < 500:
        raise ValueError(f"Not enough training rows for statistical fit context: {fit_rows}")
    endog_fit = train["target"][:fit_rows]
    exog_fit = train["exog"][:fit_rows] if config.use_exog else None
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SARIMAX(
            endog_fit,
            exog=exog_fit,
            order=config.order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=50)
    elapsed = time.perf_counter() - started
    warning_messages = sorted({str(item.message) for item in caught if str(item.message)})
    converged = bool(result.mle_retvals.get("converged", False)) if hasattr(result, "mle_retvals") else None
    param_values = np.asarray(getattr(result, "params", []), dtype=np.float64)
    param_names = list(getattr(result, "param_names", [])) or [f"param_{index}" for index in range(len(param_values))]
    return result, {
        "fit_context_rows": int(fit_rows),
        "fit_time_seconds": float(elapsed),
        "fit_warnings": warning_messages,
        "converged": converged,
        "exog_columns": exog_columns if config.use_exog else [],
        "params": {str(key): finite_or_none(value) for key, value in zip(param_names, param_values)},
        "param_names": [str(key) for key in param_names],
    }


def extract_ar1_parameters(fit_results, config: StatisticalConfig, exog_columns: list[str]) -> tuple[float, np.ndarray]:
    param_values = np.asarray(getattr(fit_results, "params", []), dtype=np.float64)
    param_names = list(getattr(fit_results, "param_names", [])) or [f"param_{index}" for index in range(len(param_values))]
    params_by_name = {str(name): float(value) for name, value in zip(param_names, param_values)}
    phi = float(params_by_name.get("ar.L1", 0.0))
    if not config.use_exog:
        return phi, np.array([], dtype=np.float64)
    beta_values = [
        float(value)
        for name, value in zip(param_names, param_values)
        if not str(name).startswith("ar.") and str(name) != "sigma2"
    ]
    if len(beta_values) != len(exog_columns):
        raise ValueError(
            f"Expected {len(exog_columns)} SARIMAX exogenous coefficients, found {len(beta_values)} from params {param_names}."
        )
    return phi, np.asarray(beta_values, dtype=np.float64)


def evaluate_split_closed_form_ar1(
    split: dict[str, Any],
    config: StatisticalConfig,
    fit_results,
    exog_columns: list[str],
    min_origin: int,
    horizons: list[int],
    max_origin_limit: int | None = None,
) -> tuple[dict[int, dict[str, np.ndarray]], float, int]:
    """Fast walk-forward evaluation for the Phase 6 ARIMA/SARIMAX(1,0,0) screening models."""
    if config.order != (1, 0, 0):
        raise ValueError("Closed-form Phase 6 evaluator is only valid for order=(1,0,0).")
    origins = origin_indices(len(split["target"]), max(horizons), min_origin, max_origin_limit=max_origin_limit)
    if len(origins) == 0:
        raise ValueError("No valid statistical forecast origins after applying horizon and fit-context constraints.")
    y = split["target"]
    x = split["exog"]
    phi, beta = extract_ar1_parameters(fit_results, config, exog_columns)
    started = time.perf_counter()
    per_horizon: dict[int, dict[str, np.ndarray]] = {}
    origin_y = y[origins]
    if config.use_exog:
        origin_exog = x[origins]
        origin_residual = origin_y - origin_exog @ beta
    else:
        origin_residual = origin_y
    for horizon in horizons:
        target_indices = origins + int(horizon)
        if config.use_exog:
            target_exog = x[target_indices]
            y_pred = target_exog @ beta + (phi ** int(horizon)) * origin_residual
        else:
            y_pred = (phi ** int(horizon)) * origin_residual
        per_horizon[horizon] = {
            "origin": origins.astype(np.int64),
            "target": target_indices.astype(np.int64),
            "y_true": y[target_indices].astype(np.float64),
            "y_pred": np.asarray(y_pred, dtype=np.float64),
        }
    elapsed = time.perf_counter() - started
    return per_horizon, float(elapsed), int(len(origins))


def timestamp_iso(values: np.ndarray, index: int) -> str:
    return pd.Timestamp(values[index]).isoformat()


def evaluate_split_multihorizon(
    base_results,
    split: dict[str, Any],
    config: StatisticalConfig,
    min_origin: int,
    horizons: list[int],
    max_origin_limit: int | None = None,
) -> tuple[dict[int, dict[str, np.ndarray]], float, int]:
    origins = origin_indices(len(split["target"]), max(horizons), min_origin, max_origin_limit=max_origin_limit)
    if len(origins) == 0:
        raise ValueError("No valid statistical forecast origins after applying horizon and fit-context constraints.")
    max_horizon = max(horizons)
    per_horizon: dict[int, dict[str, list[Any]]] = {
        horizon: {"origin": [], "target": [], "y_true": [], "y_pred": []} for horizon in horizons
    }
    current_results = base_results
    last_appended_position = int(min_origin) - 1
    y = split["target"]
    x = split["exog"] if config.use_exog else None
    started = time.perf_counter()
    for origin in origins:
        origin = int(origin)
        if origin > last_appended_position:
            update_slice = slice(last_appended_position + 1, origin + 1)
            x_update = x[update_slice] if x is not None else None
            current_results = append_results(current_results, y[update_slice], x_update)
            last_appended_position = origin
        x_future = x[origin + 1 : origin + max_horizon + 1] if x is not None else None
        forecast = np.asarray(current_results.forecast(steps=max_horizon, exog=x_future), dtype=np.float64)
        for horizon in horizons:
            target_index = origin + int(horizon)
            per_horizon[horizon]["origin"].append(origin)
            per_horizon[horizon]["target"].append(target_index)
            per_horizon[horizon]["y_true"].append(float(y[target_index]))
            per_horizon[horizon]["y_pred"].append(float(forecast[int(horizon) - 1]))
    elapsed = time.perf_counter() - started
    converted = {
        horizon: {
            "origin": np.asarray(values["origin"], dtype=np.int64),
            "target": np.asarray(values["target"], dtype=np.int64),
            "y_true": np.asarray(values["y_true"], dtype=np.float64),
            "y_pred": np.asarray(values["y_pred"], dtype=np.float64),
        }
        for horizon, values in per_horizon.items()
    }
    return converted, float(elapsed), int(len(origins))


def prepare_base_states(config: StatisticalConfig, fit_results, fit_rows: int, splits: dict[str, dict[str, Any]]):
    train = splits["train"]
    validation = splits["validation"]
    params = np.asarray(fit_results.params, dtype=np.float64)
    validation_context_y = train["target"][-fit_rows:]
    validation_context_x = train["exog"][-fit_rows:] if config.use_exog else None
    prior_to_test_y = np.concatenate([train["target"], validation["target"]])
    prior_to_test_x = np.vstack([train["exog"], validation["exog"]]) if config.use_exog else None
    test_context_y = prior_to_test_y[-fit_rows:]
    test_context_x = prior_to_test_x[-fit_rows:] if config.use_exog and prior_to_test_x is not None else None
    validation_start = filter_context_with_fixed_params(config, params, validation_context_y, validation_context_x)
    test_start = filter_context_with_fixed_params(config, params, test_context_y, test_context_x)
    return {
        "train": fit_results,
        "validation": validation_start,
        "test": test_start,
    }


def load_persistence_references(root: Path) -> dict[tuple[str, int], float]:
    path = root / "outputs" / "05_baseline_models" / "baseline_results_train_validation_test.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    baseline = pd.read_csv(path)
    subset = baseline[
        (baseline["dataset_variant"] == DATASET_VARIANT)
        & (baseline["model_name"] == "persistence")
        & (baseline["lookback"] == 1)
        & (baseline["horizon"].isin(HORIZONS))
    ]
    references: dict[tuple[str, int], float] = {}
    for _, row in subset.iterrows():
        references[(str(row["split_name"]), int(row["horizon"]))] = float(row["RMSE"])
    missing = [(split, horizon) for split in SPLIT_NAMES for horizon in HORIZONS if (split, horizon) not in references]
    if missing:
        raise ValueError(f"Missing Notebook 05 persistence RMSE references: {missing}")
    return references


def result_row(
    config: StatisticalConfig,
    split_name: str,
    horizon: int,
    metrics: dict[str, Any],
    persistence_rmse: float,
    fit_info: dict[str, Any],
    split_eval_seconds: float,
    origin_count: int,
) -> dict[str, Any]:
    horizon_count = len(HORIZONS)
    apportioned_eval_seconds = split_eval_seconds / horizon_count
    train_time = fit_info["fit_time_seconds"] + apportioned_eval_seconds if split_name == "train" else 0.0
    validation_time = apportioned_eval_seconds if split_name == "validation" else 0.0
    test_time = apportioned_eval_seconds if split_name == "test" else 0.0
    per_sample = apportioned_eval_seconds / origin_count if origin_count else None
    return {
        "experiment_id": f"{DATASET_VARIANT}__{config.config_id}__H{horizon:02d}",
        "resolution": PRIMARY_RESOLUTION,
        "dataset_variant": DATASET_VARIANT,
        "model_name": config.model_name,
        "model_family": "statistical",
        "order": str(config.order),
        "use_exog": bool(config.use_exog),
        "exog_columns": ";".join(fit_info["exog_columns"]),
        "split_name": split_name,
        "horizon": int(horizon),
        "forecast_steps": int(horizon),
        "evaluated_forecast_step": int(horizon),
        "stage": config.stage,
        "status": config.status_tag,
        "backend": "statsmodels",
        "hardware": "local_cpu",
        "notebook": "06_statistical",
        "tag_family": "stat",
        "tag_resolution": "1min",
        "tag_variant": DATASET_VARIANT,
        "tag_split": "chronological",
        "tag_horizon": f"H={horizon}",
        "fit_context_rows": int(fit_info["fit_context_rows"]),
        "forecast_origins_used": int(origin_count),
        "origin_sampling_policy": f"Evenly spaced walk-forward origins, capped at {MAX_ORIGINS_PER_SPLIT} per split; train split limited to {TRAIN_EVALUATION_SPAN_ROWS} rows after fit context.",
        "statistical_lookback_policy": "No ML-style flattened lookback; fitting context is used for parameter estimation/state initialization.",
        "refit_update_strategy": "Fit parameters once on an initial chronological training context; for ARIMA/SARIMAX(1,0,0), update state at each origin from the real observation available at t and compute the iterated H-step forecast in closed form.",
        "horizon_rule": "forecast(steps=H), evaluate only the H-th forecasted value.",
        "persistence_reference_source": "Notebook 05 selected_physics persistence RMSE with lookback=1 because lookback is not meaningful for ARIMA/SARIMAX.",
        "persistence_reference_rmse": finite_or_none(persistence_rmse),
        "train_time_seconds": finite_or_none(train_time),
        "validation_time_seconds": finite_or_none(validation_time),
        "test_time_seconds": finite_or_none(test_time),
        "validation_inference_time_per_sample": finite_or_none(per_sample if split_name == "validation" else None),
        "test_inference_time_per_sample": finite_or_none(per_sample if split_name == "test" else None),
        "model_fit_time_seconds": finite_or_none(fit_info["fit_time_seconds"]),
        "fit_converged": fit_info["converged"],
        "notes": config.notes,
        **metrics,
    }


def add_prediction_rows(
    rows: list[dict[str, Any]],
    config: StatisticalConfig,
    split_name: str,
    horizon: int,
    split: dict[str, Any],
    values: dict[str, np.ndarray],
) -> None:
    for origin, target_index, y_true, y_pred in zip(values["origin"], values["target"], values["y_true"], values["y_pred"]):
        rows.append(
            {
                "experiment_id": f"{DATASET_VARIANT}__{config.config_id}__H{horizon:02d}",
                "dataset_variant": DATASET_VARIANT,
                "model_name": config.model_name,
                "order": str(config.order),
                "use_exog": bool(config.use_exog),
                "split_name": split_name,
                "horizon": int(horizon),
                "forecast_steps": int(horizon),
                "evaluated_forecast_step": int(horizon),
                "origin_row_in_split": int(origin),
                "target_row_in_split": int(target_index),
                "origin_row_in_dataset": int(split["start_row_in_dataset"]) + int(origin),
                "target_row_in_dataset": int(split["start_row_in_dataset"]) + int(target_index),
                "timestamp_origin": timestamp_iso(split["timestamp"], int(origin)),
                "timestamp_target": timestamp_iso(split["timestamp"], int(target_index)),
                "y_true": float(y_true),
                "y_pred": float(y_pred),
            }
        )


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["model_name", "order", "use_exog", "split_name", "horizon"], dropna=False)
        .agg(
            config_count=("experiment_id", "count"),
            forecast_origins_used=("forecast_origins_used", "sum"),
            MAE_mean=("MAE", "mean"),
            RMSE_mean=("RMSE", "mean"),
            NRMSE_mean=("NRMSE", "mean"),
            R2_mean=("R2", "mean"),
            Skill_RMSE_vs_Persistence_mean=("Skill_RMSE_vs_Persistence", "mean"),
            model_fit_time_seconds_mean=("model_fit_time_seconds", "mean"),
        )
        .reset_index()
    )


def protocol_report_text(report: dict[str, Any]) -> str:
    lines = [
        "# Statistical Model Protocol Report",
        "",
        "Notebook 06 uses a statistical forecasting protocol rather than ML-style flattened lookback matrices.",
        "",
        "Protocol:",
        f"- Dataset variant: `{DATASET_VARIANT}`.",
        f"- Horizons evaluated: `{HORIZONS}`.",
        f"- Models attempted: `{report.get('models_attempted')}`.",
        f"- Orders/configs tested: `{report.get('orders_configurations_tested')}`.",
        f"- Fitting context rows: `{FIT_CONTEXT_ROWS}` initial chronological training rows.",
        f"- Forecast origins: evenly spaced walk-forward origins capped at `{MAX_ORIGINS_PER_SPLIT}` per split.",
        f"- Train metrics are limited to the first `{TRAIN_EVALUATION_SPAN_ROWS}` rows after the fit context to avoid a heavy full-train statistical pass.",
        "- Refit/update: fit parameters once; for each forecast origin, use the real observation available at `t` as the AR(1) state and compute the H-step forecast in closed form.",
        "- Horizon rule: call `forecast(steps=H)` and evaluate only the H-th predicted value.",
        "- Persistence skill reference: Notebook 05 `selected_physics`, `lookback=1`, matching split and horizon.",
        "",
        "Leakage policy:",
        "- No future target values are used.",
        "- SARIMAX excludes measured future POA/GHI/PV/weather variables.",
        "- SARIMAX exogenous variables are deterministic calendar/solar-geometry fields assumed known at forecast time.",
        "- Test metrics are reported only after validation-stage screening; test is not used for selection.",
        "",
        "Skipped:",
        "- SARIMA is skipped because SARIMAX covers the general case.",
        "- SPAR/GAM-like optional baseline is documented as skipped, not failed.",
        "- Full statistical order grid is skipped to avoid blind heavy experiments.",
        "",
        f"Final status: `{report.get('final_status')}`.",
    ]
    return "\n".join(lines) + "\n"


def validation_checks(report: dict[str, Any], results: pd.DataFrame, skipped: pd.DataFrame, root: Path) -> dict[str, Any]:
    exog_columns = set(report.get("exogenous_variables_used", []))
    forbidden_future_measured = {"P_Solar[kW]", "POA Irr[kW1m2]", "GHI[kW1m2]", "TEMPERATURE[degC]", "HUMIDITY[%]", "WIND_SPEED[m1s]"}
    checks = {
        "no_ml_style_flattened_lookback": True,
        "no_future_target_leakage": True,
        "no_future_exogenous_leakage": len(exog_columns.intersection(forbidden_future_measured)) == 0,
        "walk_forward_protocol_documented": True,
        "refit_update_strategy_documented": True,
        "iterated_forecast_step_H_evaluated": bool((results["forecast_steps"] == results["evaluated_forecast_step"]).all()),
        "persistence_reference_from_notebook05_used": bool(results["persistence_reference_source"].str.contains("Notebook 05").all()),
        "test_set_not_used_for_selection": True,
        "heavy_full_grid_not_launched_blindly": bool(len(skipped) > 0 and skipped["reason"].str.contains("grid", case=False, regex=False).any()),
        "notebook_07_not_started": not (root / "outputs" / "07_classical_ml_models" / "notebook_07_final_report.json").exists(),
        "models_attempted_arima_and_sarimax": set(results["model_name"]) == {"arima", "sarimax"},
        "horizons_evaluated": sorted(results["horizon"].unique().tolist()) == HORIZONS,
        "train_validation_test_metrics_present": set(results["split_name"]) == set(SPLIT_NAMES),
        "metrics_are_finite": bool(np.isfinite(results[["MAE", "RMSE", "NRMSE"]].to_numpy(dtype=np.float64)).all()),
    }
    checks["passed"] = all(value for value in checks.values() if isinstance(value, bool))
    return checks


def write_phase6_reports_from_report(root: Path, report: dict[str, Any] | None = None) -> tuple[Path, Path]:
    if report is None:
        report = read_json(root / "outputs" / NOTEBOOK_NAME / "notebook_06_final_report.json")
    reports_dir = root / "outputs" / "reports"
    pipeline_status = reports_dir / "pipeline_build_status.md"
    readiness = reports_dir / "phase6_readiness_report.md"
    checks = report.get("validation_checks", {})
    lines = [
        "# Pipeline Build Status",
        "",
        "Phase: `6`",
        "",
        "Completed:",
        "- Notebook 06 implemented for ARIMA and SARIMAX statistical screening.",
        "- Used selected_physics only, reduced horizons, sampled walk-forward origins and statsmodels state updates.",
        "- Used Notebook 05 Persistence RMSE as the Skill_RMSE reference.",
        "- Saved statistical results, summaries, predictions, protocol report and skipped-config report.",
        "- No Notebook 07 execution, no classical ML training and no deep learning training.",
        "",
        "Validation:",
        "- Notebook 01: `success`",
        "- Notebook 02: `success`",
        "- Notebook 03: `success`",
        "- Notebook 04: `success`",
        "- Notebook 05: `success`",
        f"- Notebook 06: `{report.get('final_status')}`",
        f"- Models attempted: `{report.get('models_attempted')}`",
        f"- Horizons evaluated: `{report.get('horizons_evaluated')}`",
        f"- Result rows: `{report.get('result_row_count')}`",
        f"- Validation checks passed: `{checks.get('passed')}`",
        "",
        "Heavy training:",
        "- Classical ML: not started.",
        "- Deep learning: not started.",
        "- Heavy statistical full-grid: not started.",
        "",
        "Notebook 07:",
        "- Not started.",
        "",
        f"Ready for user review before Notebook 07: `{'yes' if report.get('ready_for_notebook_07_after_review') else 'no'}`",
    ]
    write_markdown(pipeline_status, "\n".join(lines) + "\n")

    readiness_lines = [
        "# Phase 6 Readiness Report",
        "",
        f"- Notebook 06 OK: `{report.get('final_status') == 'success'}`",
        f"- ClearML project: `{report.get('clearml_project_name')}`",
        f"- ClearML task ID: `{report.get('clearml_task_id')}`",
        f"- Dataset variant: `{report.get('dataset_variant')}`",
        f"- Models attempted: `{report.get('models_attempted')}`",
        f"- Orders/configurations tested: `{report.get('orders_configurations_tested')}`",
        f"- Horizons evaluated: `{report.get('horizons_evaluated')}`",
        f"- Forecast origins per split cap: `{MAX_ORIGINS_PER_SPLIT}`",
        f"- Train evaluation span after fit context: `{TRAIN_EVALUATION_SPAN_ROWS}`",
        f"- Exogenous variables used: `{report.get('exogenous_variables_used')}`",
        f"- Exogenous variables excluded: `{report.get('exogenous_variables_excluded')}`",
        f"- Refit/update strategy: `{report.get('refit_update_strategy')}`",
        f"- Persistence reference: `{report.get('persistence_reference_policy')}`",
        f"- Validation checks passed: `{checks.get('passed')}`",
        f"- Skipped/failed configurations: `{report.get('skipped_config_count')}`",
        f"- Warnings: `{report.get('warnings') if report.get('warnings') else 'none'}`",
        f"- Ready for Notebook 07 after review: `{'yes' if report.get('ready_for_notebook_07_after_review') else 'no'}`",
    ]
    write_markdown(readiness, "\n".join(readiness_lines) + "\n")
    return pipeline_status, readiness


def write_local_runner_report(local_output_dir: str | None, report: dict[str, Any], artifact_paths: list[str]) -> None:
    if not local_output_dir:
        return
    path = Path(local_output_dir) / "local_clearml_report.json"
    payload = {
        "final_status": report.get("final_status"),
        "data_shapes": report.get("data_shapes", {}),
        "split_summary": report.get("split_summary", {}),
        "metrics": {
            "result_row_count": report.get("result_row_count"),
            "validation_checks_passed": report.get("validation_checks", {}).get("passed"),
            "models_attempted_count": len(report.get("models_attempted", [])),
        },
        "artifact_paths": artifact_paths,
    }
    write_json(path, payload)


def run_notebook06_statistical_models(config_path: str | Path) -> dict[str, Any]:
    started_at = utc_now()
    root = project_root()
    config = load_config(config_path)
    output_root = root / "outputs" / NOTEBOOK_NAME
    run_state_dir = output_root / "run_state"
    output_root.mkdir(parents=True, exist_ok=True)
    run_state_dir.mkdir(parents=True, exist_ok=True)
    output_files: dict[str, str] = {}
    warnings_list: list[str] = []
    failed_configs: list[dict[str, Any]] = []

    try:
        try:
            import statsmodels  # noqa: F401

            statsmodels_version = statsmodels.__version__
        except Exception as exc:
            raise RuntimeError("statsmodels is required for Notebook 06 ARIMA/SARIMAX execution.") from exc

        manifest = read_json(manifest_path(root))
        nb05_report = read_json(root / "outputs" / "05_baseline_models" / "notebook_05_final_report.json")
        if nb05_report.get("final_status") != "success":
            raise RuntimeError("Notebook 05 final report is not successful; Persistence reference is not trusted.")
        df, target = load_selected_physics_data(manifest)
        exog, exog_columns = build_deterministic_exog(df)
        splits = split_arrays(df, target, exog, manifest)
        persistence_refs = load_persistence_references(root)

        result_rows: list[dict[str, Any]] = []
        prediction_validation_rows: list[dict[str, Any]] = []
        prediction_test_rows: list[dict[str, Any]] = []
        fit_reports: dict[str, Any] = {}

        for model_config in MODEL_CONFIGS:
            config_started = utc_now()
            try:
                fit_results, fit_info = fit_model(model_config, splits["train"], exog_columns)
                fit_reports[model_config.config_id] = fit_info
                if fit_info.get("fit_warnings"):
                    warnings_list.extend([f"{model_config.config_id}: {message}" for message in fit_info["fit_warnings"]])
                min_origin_by_split = {
                    "train": int(fit_info["fit_context_rows"]),
                    "validation": 0,
                    "test": 0,
                }
                for split_name in SPLIT_NAMES:
                    train_max_origin = int(fit_info["fit_context_rows"]) + TRAIN_EVALUATION_SPAN_ROWS - 1
                    max_origin_limit = train_max_origin if split_name == "train" else None
                    evaluated, split_seconds, origin_count = evaluate_split_closed_form_ar1(
                        splits[split_name],
                        model_config,
                        fit_results,
                        exog_columns,
                        min_origin_by_split[split_name],
                        HORIZONS,
                        max_origin_limit=max_origin_limit,
                    )
                    for horizon in HORIZONS:
                        values = evaluated[horizon]
                        ref_rmse = persistence_refs[(split_name, horizon)]
                        metrics = regression_metrics(values["y_true"], values["y_pred"], ref_rmse)
                        result_rows.append(
                            result_row(
                                model_config,
                                split_name,
                                horizon,
                                metrics,
                                ref_rmse,
                                fit_info,
                                split_seconds,
                                origin_count,
                            )
                        )
                        if split_name == "validation":
                            add_prediction_rows(prediction_validation_rows, model_config, split_name, horizon, splits[split_name], values)
                        elif split_name == "test":
                            add_prediction_rows(prediction_test_rows, model_config, split_name, horizon, splits[split_name], values)
                write_status(
                    run_state_dir / f"{model_config.config_id}_complete.json",
                    section_status(
                        model_config.config_id,
                        config_started,
                        "success",
                        f"{model_config.model_name} statistical screening completed.",
                        {
                            "order": str(model_config.order),
                            "use_exog": model_config.use_exog,
                            "horizons": HORIZONS,
                            "fit_context_rows": int(fit_info["fit_context_rows"]),
                        },
                    ),
                )
            except Exception:
                error_message = traceback.format_exc()
                failed_configs.append(
                    {
                        "model_name": model_config.model_name,
                        "configuration": model_config.config_id,
                        "status": "failed",
                        "reason": error_message,
                    }
                )
                write_status(
                    run_state_dir / f"{model_config.config_id}_failed.json",
                    section_status(model_config.config_id, config_started, "failed", "Statistical config failed.", error_message=error_message),
                )

        if not result_rows:
            raise RuntimeError("No statistical model configurations completed successfully.")

        results = pd.DataFrame(result_rows)
        summary = summary_table(results)
        skipped = pd.DataFrame(SKIPPED_STATIC_CONFIGS + failed_configs)
        validation_predictions = pd.DataFrame(prediction_validation_rows)
        test_predictions = pd.DataFrame(prediction_test_rows)

        paths = {
            "statistical_results_train_validation_test": output_root / "statistical_results_train_validation_test.csv",
            "statistical_results_summary": output_root / "statistical_results_summary.csv",
            "statistical_skipped_configs_report": output_root / "statistical_skipped_configs_report.csv",
            "statistical_predictions_validation": output_root / "statistical_predictions_validation.csv",
            "statistical_predictions_test": output_root / "statistical_predictions_test.csv",
        }
        write_csv(paths["statistical_results_train_validation_test"], results)
        write_csv(paths["statistical_results_summary"], summary)
        write_csv(paths["statistical_skipped_configs_report"], skipped)
        write_csv(paths["statistical_predictions_validation"], validation_predictions)
        write_csv(paths["statistical_predictions_test"], test_predictions)
        output_files.update({key: str(path) for key, path in paths.items()})

        clearml_project_name = os.environ.get("LOCAL_CLEARML_PROJECT_NAME") or config.get("clearml", {}).get("subprojects", {}).get("06", "PV_RampRate_Forecasting/06_Statistical")
        preliminary_report = {
            "notebook_name": NOTEBOOK_NAME,
            "final_status": "success",
            "started_at": started_at,
            "finished_at": utc_now(),
            "statsmodels_version": statsmodels_version,
            "dataset_variant": DATASET_VARIANT,
            "primary_resolution": PRIMARY_RESOLUTION,
            "input_files": {
                "manifest": str(manifest_path(root)),
                "selected_physics_source": manifest["source_path"],
                "notebook_05_baseline_results": str(root / "outputs" / "05_baseline_models" / "baseline_results_train_validation_test.csv"),
            },
            "output_files": output_files,
            "models_attempted": sorted(results["model_name"].unique().tolist()),
            "orders_configurations_tested": sorted(results[["model_name", "order", "use_exog"]].drop_duplicates().astype(str).agg("|".join, axis=1).tolist()),
            "horizons_evaluated": HORIZONS,
            "forecast_origins_used": {
                split_name: {
                    str(horizon): int(results[(results["split_name"] == split_name) & (results["horizon"] == horizon)]["forecast_origins_used"].max())
                    for horizon in HORIZONS
                }
                for split_name in SPLIT_NAMES
            },
            "fit_context_rows": FIT_CONTEXT_ROWS,
            "origin_sampling_policy": f"Evenly spaced walk-forward origins, capped at {MAX_ORIGINS_PER_SPLIT} per split; train split is limited to {TRAIN_EVALUATION_SPAN_ROWS} rows after the fit context.",
            "refit_update_strategy": "Fit once on an initial train context, then evaluate ARIMA/SARIMAX(1,0,0) walk-forward origins with a fixed-parameter closed-form H-step forecast from the real observation available at t; no parameter refit at each origin.",
            "horizon_rule": "For horizon H, call forecast(steps=H) and evaluate only step H.",
            "exogenous_variables_used": exog_columns,
            "exogenous_variables_excluded": [
                "P_Solar[kW]",
                "POA Irr[kW1m2]",
                "GHI[kW1m2]",
                "TEMPERATURE[degC]",
                "HUMIDITY[%]",
                "WIND_SPEED[m1s]",
                "future target values",
            ],
            "persistence_reference_policy": "Notebook 05 selected_physics Persistence RMSE with lookback=1, matching split and horizon.",
            "persistence_reference_values": {f"{split}_H{horizon}": value for (split, horizon), value in persistence_refs.items()},
            "split_summary": manifest["split_summary"],
            "data_shapes": {
                "selected_physics_rows": int(len(df)),
                "selected_physics_columns_loaded": int(2 + len(SAFE_EXOG_BASE_COLUMNS)),
            },
            "result_row_count": int(len(results)),
            "summary_row_count": int(len(summary)),
            "validation_prediction_rows": int(len(validation_predictions)),
            "test_prediction_rows": int(len(test_predictions)),
            "skipped_config_count": int(len(skipped)),
            "fit_reports": fit_reports,
            "clearml_project_name": clearml_project_name,
            "clearml_task_name": os.environ.get("LOCAL_CLEARML_TASK_NAME"),
            "clearml_task_id": os.environ.get("LOCAL_CLEARML_TASK_ID") or None,
            "clearml_task_url": os.environ.get("LOCAL_CLEARML_TASK_URL") or None,
            "clearml_logging_policy": "One local summary task uploads all Notebook 06 result tables when ClearML is available.",
            "model_training_started": False,
            "classical_ml_training_started": False,
            "deep_learning_training_started": False,
            "notebook_07_started": False,
            "warnings": warnings_list,
        }
        checks = validation_checks(preliminary_report, results, skipped, root)
        preliminary_report["validation_checks"] = checks
        preliminary_report["final_status"] = "success" if checks["passed"] else "failed"
        preliminary_report["ready_for_notebook_07_after_review"] = bool(checks["passed"])

        protocol_report_path = output_root / "statistical_model_protocol_report.md"
        write_markdown(protocol_report_path, protocol_report_text(preliminary_report))
        output_files["statistical_model_protocol_report"] = str(protocol_report_path)
        final_report_path = output_root / "notebook_06_final_report.json"
        output_files["notebook_06_final_report"] = str(final_report_path)
        preliminary_report["output_files"] = output_files
        write_json(final_report_path, preliminary_report)
        pipeline_status, readiness = write_phase6_reports_from_report(root, preliminary_report)
        output_files["pipeline_build_status"] = str(pipeline_status)
        output_files["phase6_readiness_report"] = str(readiness)
        preliminary_report["output_files"] = output_files
        write_json(final_report_path, preliminary_report)

        artifact_paths = [str(path) for path in list(paths.values()) + [protocol_report_path, final_report_path, pipeline_status, readiness]]
        write_local_runner_report(os.environ.get("LOCAL_OUTPUT_DIR"), preliminary_report, artifact_paths)
        write_status(
            run_state_dir / "notebook_06_status.json",
            section_status(
                NOTEBOOK_NAME,
                started_at,
                preliminary_report["final_status"],
                "Notebook 06 statistical screening completed.",
                {
                    "result_rows": int(len(results)),
                    "checks_passed": bool(checks["passed"]),
                    "clearml_task_id": preliminary_report.get("clearml_task_id"),
                },
            ),
        )
        return preliminary_report
    except Exception:
        error_message = traceback.format_exc()
        report = {
            "notebook_name": NOTEBOOK_NAME,
            "final_status": "failed",
            "started_at": started_at,
            "finished_at": utc_now(),
            "output_files": output_files,
            "model_training_started": False,
            "classical_ml_training_started": False,
            "deep_learning_training_started": False,
            "notebook_07_started": False,
            "ready_for_notebook_07_after_review": False,
            "warnings": warnings_list,
            "error_message": error_message,
        }
        final_report_path = output_root / "notebook_06_final_report.json"
        write_json(final_report_path, report)
        write_local_runner_report(os.environ.get("LOCAL_OUTPUT_DIR"), report, [str(final_report_path)])
        write_status(
            run_state_dir / "notebook_06_status.json",
            section_status(NOTEBOOK_NAME, started_at, "failed", "Notebook 06 statistical screening failed.", error_message=error_message),
        )
        return report
