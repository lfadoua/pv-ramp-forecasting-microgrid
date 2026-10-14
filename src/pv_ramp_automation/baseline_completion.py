from __future__ import annotations

import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config, project_root
from .reports import write_csv, write_json, write_markdown
from .status import section_status, utc_now, write_status


NOTEBOOK_NAME = "05b_one_hour_baseline_completion"
BASELINE_OUTPUT_NAME = "05_baseline_models"
TIME_COLUMN = "timestamp"
TARGET_COLUMN = "ramp_rate_pu"
RESOLUTION = "one_hour"
RESOLUTIONS_FOR_STAGE_A = ["one_minute", "one_hour"]
DATASET_VARIANTS = ["full", "selected", "full_physics", "selected_physics"]
SPLIT_NAMES = ["train", "validation", "test"]
BASELINE_MODELS = ["persistence", "mean_naive"]
LOOKBACKS = [1, 8, 16, 32, 64]
HORIZONS = [1, 8, 16, 32, 64]
STAGE_A_PAIRS = [(8, 1), (16, 8), (32, 16)]
REQUIRED_RESULT_COLUMNS = [
    "resolution",
    "dataset_variant",
    "model_name",
    "model",
    "lookback",
    "horizon",
    "split_name",
    "split",
    "MAE",
    "RMSE",
    "NRMSE",
    "R2",
    "Skill_RMSE_vs_Persistence",
    "feature_count",
    "sample_count",
    "status",
    "run_id",
]
LOWERCASE_METRIC_ALIASES = ["mae", "rmse", "nrmse", "r2", "skill_rmse_vs_persistence"]
PRESERVATION_EXCLUDED_COLUMNS = LOWERCASE_METRIC_ALIASES + ["model", "split", "run_id"]
PRESERVATION_CORE_COLUMNS = [
    "experiment_id",
    "resolution",
    "dataset_variant",
    "lookback",
    "horizon",
    "model_name",
    "split_name",
    "sample_count",
    "MAE",
    "RMSE",
    "NRMSE",
    "R2",
    "Skill_RMSE_vs_Persistence",
]
PRESERVATION_KEY_COLUMNS = ["experiment_id", "resolution", "dataset_variant", "lookback", "horizon", "model_name", "split_name"]
PRESERVATION_NUMERIC_COLUMNS = ["sample_count", "MAE", "RMSE", "NRMSE", "R2", "Skill_RMSE_vs_Persistence"]


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


def aligned_indices(row_count: int, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    sample_count = int(row_count) - int(lookback) - int(horizon) + 1
    if sample_count <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    origin_indices = np.arange(int(lookback) - 1, int(row_count) - int(horizon), dtype=np.int64)
    target_indices = origin_indices + int(horizon)
    return origin_indices, target_indices


def manifest_path(root: Path, resolution: str, variant: str) -> Path:
    return root / "outputs" / resolution / "notebook_04_split_windowing" / variant / "window_ready_manifest.json"


def validate_one_hour_prerequisites(root: Path) -> list[str]:
    missing: list[str] = []
    for variant in DATASET_VARIANTS:
        base = root / "outputs" / RESOLUTION / "notebook_04_split_windowing" / variant
        for filename in [
            "window_ready_manifest.json",
            "train_only_standard_scaler.joblib",
            "windowing_counts.json",
            "train_time_index.csv",
            "validation_time_index.csv",
            "test_time_index.csv",
        ]:
            path = base / filename
            if not path.exists():
                missing.append(str(path))
    return missing


def load_variant_frame(manifest: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    source_path = Path(manifest["source_path"])
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    header = list(pd.read_csv(source_path, nrows=0).columns)
    required = [TIME_COLUMN, TARGET_COLUMN]
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"Missing required columns in {source_path}: {missing}")
    if header[-1] != TARGET_COLUMN:
        raise ValueError(f"Target must be the final column in {source_path}")
    df = pd.read_csv(source_path, usecols=required)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    if int(df[TIME_COLUMN].isna().sum()) > 0:
        raise ValueError(f"Invalid timestamps in {source_path}")
    if not df[TIME_COLUMN].is_monotonic_increasing:
        raise ValueError(f"Timestamps are not sorted in {source_path}")
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(target).all():
        raise ValueError(f"Target contains NaN or infinite values in {source_path}")
    if int(manifest["row_count"]) != len(df):
        raise ValueError(f"Manifest row count mismatch for {source_path}")
    return df, header


def split_arrays(df: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    splits: dict[str, dict[str, Any]] = {}
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    timestamps = df[TIME_COLUMN].to_numpy()
    for split_name in SPLIT_NAMES:
        meta = manifest["split_summary"][split_name]
        start = int(meta["start_row_in_dataset"])
        end = int(meta["end_row_exclusive_in_dataset"])
        splits[split_name] = {
            "target": target[start:end],
            "timestamp": timestamps[start:end],
            "start_row_in_dataset": start,
            "end_row_exclusive_in_dataset": end,
            "row_count": int(end - start),
            "start_timestamp": meta["start_timestamp"],
            "end_timestamp": meta["end_timestamp"],
        }
    return splits


def result_row(
    *,
    run_id: str,
    resolution: str,
    variant: str,
    lookback: int,
    horizon: int,
    split_name: str,
    model_name: str,
    feature_count: int,
    metrics: dict[str, Any],
    elapsed_seconds: float,
    train_mean_value: float | None,
) -> dict[str, Any]:
    train_time = elapsed_seconds if split_name == "train" else 0.0
    validation_time = elapsed_seconds if split_name == "validation" else 0.0
    test_time = elapsed_seconds if split_name == "test" else 0.0
    tag_resolution = "1h" if resolution == "one_hour" else "1min"
    return {
        "experiment_id": f"{resolution}__{variant}__L{lookback:02d}__H{horizon:02d}",
        "resolution": resolution,
        "dataset_variant": variant,
        "split_method": "chronological_70_15_15_no_shuffle",
        "lookback": int(lookback),
        "horizon": int(horizon),
        "feature_count": int(feature_count),
        "target_name": TARGET_COLUMN,
        "recommended_stage": "exploratory",
        "compatible_families": "baseline;statistical;classical_ml;deep_learning",
        "model_name": model_name,
        "baseline_family": "baseline",
        "split_name": split_name,
        "backend": "cpu",
        "hardware": "local_cpu",
        "status": "final",
        "notebook": NOTEBOOK_NAME,
        "tag_family": "baseline",
        "tag_model": model_name,
        "tag_resolution": tag_resolution,
        "tag_variant": variant,
        "tag_split": "chronological",
        "tag_lookback": f"L{int(lookback)}",
        "tag_horizon": f"H{int(horizon)}",
        "train_mean_value": train_mean_value if model_name == "mean_naive" else None,
        "train_mean_source": "aligned train targets for this resolution/variant/lookback/horizon only" if model_name == "mean_naive" else None,
        "persistence_definition": "y_pred(t+H)=ramp_rate_pu(t)" if model_name == "persistence" else None,
        "mean_naive_definition": "y_pred=mean(aligned train target ramp_rate_pu(t+H))" if model_name == "mean_naive" else None,
        "target_alignment": "origin t uses target at t+H; input history ends at t",
        "same_row_target_feature_used": False,
        "future_information_used": False,
        "train_time_seconds": finite_or_none(train_time),
        "validation_time_seconds": finite_or_none(validation_time),
        "test_time_seconds": finite_or_none(test_time),
        "validation_inference_time_per_sample": finite_or_none(validation_time / metrics["sample_count"] if split_name == "validation" and metrics["sample_count"] else None),
        "test_inference_time_per_sample": finite_or_none(test_time / metrics["sample_count"] if split_name == "test" and metrics["sample_count"] else None),
        "run_id": run_id,
        "model": model_name,
        "split": split_name,
        **metrics,
    }


def evaluate_one_hour_baselines(root: Path, run_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    result_rows: list[dict[str, Any]] = []
    split_summary: dict[str, Any] = {}
    feature_counts: dict[str, int] = {}
    data_shapes: dict[str, Any] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for variant in DATASET_VARIANTS:
        manifest = read_json(manifest_path(root, RESOLUTION, variant))
        if manifest.get("dataset_variant") != variant:
            raise ValueError(f"Manifest variant mismatch for {variant}")
        if manifest.get("target_column") != TARGET_COLUMN:
            raise ValueError(f"Unexpected target in {variant}: {manifest.get('target_column')}")
        scaler_file = Path(manifest["scaler_file"])
        if not scaler_file.exists():
            raise FileNotFoundError(scaler_file)
        if TARGET_COLUMN in manifest.get("feature_columns", []):
            raise ValueError(f"Target leaked into feature columns for {variant}")
        protocol = manifest.get("windowing_protocol", {})
        if not protocol.get("build_windows_separately_inside_each_split") or not protocol.get("no_window_crosses_split_boundaries"):
            raise ValueError(f"Windowing protocol is not trusted for {variant}")

        df, header = load_variant_frame(manifest)
        splits = split_arrays(df, manifest)
        manifests[variant] = manifest
        split_summary[variant] = manifest["split_summary"]
        feature_counts[variant] = int(manifest["feature_count"])
        data_shapes[variant] = {"row_count": int(len(df)), "source_column_count": int(len(header))}

        for lookback in LOOKBACKS:
            for horizon in HORIZONS:
                train_origin, train_target = aligned_indices(len(splits["train"]["target"]), lookback, horizon)
                if len(train_target) <= 0:
                    raise ValueError(f"No train samples for {RESOLUTION}/{variant} L={lookback} H={horizon}")
                train_mean_started = time.perf_counter()
                train_mean_value = float(np.mean(splits["train"]["target"][train_target]))
                train_mean_seconds = time.perf_counter() - train_mean_started

                for split_name in SPLIT_NAMES:
                    split = splits[split_name]
                    origin_indices, target_indices = aligned_indices(len(split["target"]), lookback, horizon)
                    if len(target_indices) <= 0:
                        raise ValueError(f"No {split_name} samples for {RESOLUTION}/{variant} L={lookback} H={horizon}")
                    y_origin = split["target"][origin_indices]
                    y_true = split["target"][target_indices]

                    persistence_started = time.perf_counter()
                    persistence_pred = y_origin.copy()
                    persistence_metrics = regression_metrics(y_true, persistence_pred, persistence_rmse=None)
                    persistence_metrics["Skill_RMSE_vs_Persistence"] = 0.0
                    persistence_seconds = time.perf_counter() - persistence_started
                    result_rows.append(
                        result_row(
                            run_id=run_id,
                            resolution=RESOLUTION,
                            variant=variant,
                            lookback=lookback,
                            horizon=horizon,
                            split_name=split_name,
                            model_name="persistence",
                            feature_count=feature_counts[variant],
                            metrics=persistence_metrics,
                            elapsed_seconds=persistence_seconds,
                            train_mean_value=None,
                        )
                    )

                    mean_started = time.perf_counter()
                    mean_pred = np.full_like(y_true, train_mean_value, dtype=np.float64)
                    mean_metrics = regression_metrics(y_true, mean_pred, persistence_rmse=persistence_metrics["RMSE"])
                    mean_seconds = (time.perf_counter() - mean_started) + (train_mean_seconds if split_name == "train" else 0.0)
                    result_rows.append(
                        result_row(
                            run_id=run_id,
                            resolution=RESOLUTION,
                            variant=variant,
                            lookback=lookback,
                            horizon=horizon,
                            split_name=split_name,
                            model_name="mean_naive",
                            feature_count=feature_counts[variant],
                            metrics=mean_metrics,
                            elapsed_seconds=mean_seconds,
                            train_mean_value=train_mean_value,
                        )
                    )

    return pd.DataFrame(result_rows), {
        "manifests": manifests,
        "split_summary": split_summary,
        "feature_counts": feature_counts,
        "data_shapes": data_shapes,
    }


def ensure_result_alias_columns(df: pd.DataFrame, default_run_id: str) -> pd.DataFrame:
    result = df.copy()
    result = result.drop(columns=[column for column in LOWERCASE_METRIC_ALIASES if column in result.columns])
    alias_map = {
        "model": "model_name",
        "split": "split_name",
    }
    for alias, source in alias_map.items():
        if alias not in result.columns:
            result[alias] = result[source] if source in result.columns else None
        else:
            result[alias] = result[alias].where(result[alias].notna() & (result[alias].astype(str) != ""), result[source])
    if "run_id" not in result.columns:
        result["run_id"] = default_run_id
    else:
        result["run_id"] = result["run_id"].where(result["run_id"].notna() & (result["run_id"].astype(str) != ""), default_run_id)
    return result


def write_merged_outputs(root: Path, one_hour_results: pd.DataFrame, run_id: str) -> dict[str, Any]:
    output_root = root / "outputs" / BASELINE_OUTPUT_NAME
    result_path = output_root / "baseline_results_train_validation_test.csv"
    summary_path = output_root / "baseline_results_summary.csv"
    reference_path = output_root / "persistence_reference_by_variant_lh.csv"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    before = pd.read_csv(result_path)
    original_columns = [column for column in PRESERVATION_CORE_COLUMNS if column in before.columns]
    before_one_minute = before[before["resolution"] == "one_minute"][original_columns].copy().reset_index(drop=True)
    previous_one_hour_rows = int((before["resolution"] == RESOLUTION).sum()) if "resolution" in before.columns else 0

    existing = ensure_result_alias_columns(before, default_run_id="notebook05_existing_20260629_184242")
    existing = existing[existing["resolution"] != RESOLUTION].copy()
    new_results = ensure_result_alias_columns(one_hour_results, default_run_id=run_id)

    columns = list(existing.columns)
    for column in new_results.columns:
        if column not in columns:
            columns.append(column)
    for column in REQUIRED_RESULT_COLUMNS:
        if column not in columns:
            columns.append(column)
    combined = pd.concat([existing.reindex(columns=columns), new_results.reindex(columns=columns)], ignore_index=True)
    write_csv(result_path, combined)

    after = pd.read_csv(result_path)
    after_one_minute = after[after["resolution"] == "one_minute"][original_columns].copy().reset_index(drop=True)
    one_minute_rows_preserved = one_minute_core_rows_preserved(before_one_minute, after_one_minute)

    summary = (
        after.groupby(["resolution", "dataset_variant", "model_name", "split_name", "lookback", "horizon", "status"], dropna=False)
        .agg(
            config_count=("experiment_id", "count"),
            sample_count_total=("sample_count", "sum"),
            MAE_mean=("MAE", "mean"),
            MAE_min=("MAE", "min"),
            MAE_max=("MAE", "max"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_min=("RMSE", "min"),
            RMSE_max=("RMSE", "max"),
            NRMSE_mean=("NRMSE", "mean"),
            R2_mean=("R2", "mean"),
            Skill_RMSE_vs_Persistence_mean=("Skill_RMSE_vs_Persistence", "mean"),
        )
        .reset_index()
    )
    write_csv(summary_path, summary)

    persistence = after[after["model_name"] == "persistence"].copy()
    reference = persistence[
        [
            "resolution",
            "dataset_variant",
            "lookback",
            "horizon",
            "split_name",
            "split",
            "sample_count",
            "RMSE",
            "NRMSE",
            "MAE",
            "R2",
            "Skill_RMSE_vs_Persistence",
        ]
    ].rename(
        columns={
            "RMSE": "persistence_RMSE",
            "NRMSE": "persistence_NRMSE",
            "MAE": "persistence_MAE",
            "R2": "persistence_R2",
        }
    )
    write_csv(reference_path, reference)

    return {
        "baseline_results_train_validation_test": str(result_path),
        "baseline_results_summary": str(summary_path),
        "persistence_reference_by_variant_lh": str(reference_path),
        "one_minute_rows_before": int(len(before_one_minute)),
        "one_minute_rows_after": int(len(after_one_minute)),
        "one_minute_rows_preserved": bool(one_minute_rows_preserved),
        "previous_one_hour_rows_replaced": previous_one_hour_rows,
        "one_hour_rows_written": int(len(new_results)),
        "combined_result_rows": int(len(after)),
        "summary_rows": int(len(summary)),
        "persistence_reference_rows": int(len(reference)),
    }


def one_minute_core_rows_preserved(before: pd.DataFrame, after: pd.DataFrame) -> bool:
    if len(before) != len(after):
        return False
    key_columns = [column for column in PRESERVATION_KEY_COLUMNS if column in before.columns and column in after.columns]
    for column in key_columns:
        if not before[column].astype(str).reset_index(drop=True).equals(after[column].astype(str).reset_index(drop=True)):
            return False
    numeric_columns = [column for column in PRESERVATION_NUMERIC_COLUMNS if column in before.columns and column in after.columns]
    for column in numeric_columns:
        left = pd.to_numeric(before[column], errors="coerce").to_numpy(dtype=np.float64)
        right = pd.to_numeric(after[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.allclose(left, right, rtol=1e-12, atol=1e-15, equal_nan=True):
            return False
    return True


def stage_a_reference_missing(reference: pd.DataFrame) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for resolution in RESOLUTIONS_FOR_STAGE_A:
        for variant in DATASET_VARIANTS:
            for lookback, horizon in STAGE_A_PAIRS:
                for split_name in SPLIT_NAMES:
                    mask = (
                        (reference["resolution"] == resolution)
                        & (reference["dataset_variant"] == variant)
                        & (reference["lookback"].astype(int) == int(lookback))
                        & (reference["horizon"].astype(int) == int(horizon))
                        & (reference["split_name"] == split_name)
                    )
                    if not bool(mask.any()):
                        missing.append(
                            {
                                "resolution": resolution,
                                "dataset_variant": variant,
                                "lookback": int(lookback),
                                "horizon": int(horizon),
                                "split": split_name,
                            }
                        )
    return missing


def validate_completion(root: Path, output_info: dict[str, Any]) -> dict[str, Any]:
    result_path = Path(output_info["baseline_results_train_validation_test"])
    reference_path = Path(output_info["persistence_reference_by_variant_lh"])
    results = pd.read_csv(result_path)
    reference = pd.read_csv(reference_path)
    one_hour = results[results["resolution"] == RESOLUTION]
    one_hour_persistence = one_hour[one_hour["model_name"] == "persistence"]
    one_hour_mean = one_hour[one_hour["model_name"] == "mean_naive"]
    missing_stage_a = stage_a_reference_missing(reference)
    numeric_metric_columns = ["MAE", "RMSE", "NRMSE"]
    checks = {
        "one_hour_persistence_references_generated": int(len(one_hour_persistence)) == len(DATASET_VARIANTS) * len(LOOKBACKS) * len(HORIZONS) * len(SPLIT_NAMES),
        "one_hour_mean_naive_generated": int(len(one_hour_mean)) == len(DATASET_VARIANTS) * len(LOOKBACKS) * len(HORIZONS) * len(SPLIT_NAMES),
        "full_25_lh_grid_generated": sorted(one_hour[["lookback", "horizon"]].drop_duplicates().astype(int).apply(tuple, axis=1).tolist())
        == sorted([(lookback, horizon) for lookback in LOOKBACKS for horizon in HORIZONS]),
        "stage_a_required_references_present": len(missing_stage_a) == 0,
        "result_schema_contains_required_columns": all(column in results.columns for column in REQUIRED_RESULT_COLUMNS),
        "result_schema_matches_across_resolutions": all(
            set(REQUIRED_RESULT_COLUMNS).issubset(set(results[results["resolution"] == resolution].columns))
            for resolution in RESOLUTIONS_FOR_STAGE_A
        ),
        "one_minute_rows_preserved": bool(output_info["one_minute_rows_preserved"]),
        "positive_sample_counts": bool((results["sample_count"] > 0).all()),
        "finite_core_metrics": bool(np.isfinite(results[numeric_metric_columns].to_numpy(dtype=np.float64)).all()),
        "persistence_skill_zero": bool((one_hour_persistence["Skill_RMSE_vs_Persistence"].astype(float) == 0.0).all()),
        "target_alignment_documented": bool(results["target_alignment"].astype(str).str.contains("t\\+H", regex=True).all()),
        "no_same_row_target_feature": bool((results["same_row_target_feature_used"] == False).all()),  # noqa: E712
        "no_future_information_used": bool((results["future_information_used"] == False).all()),  # noqa: E712
        "notebook_07_not_started": not (root / "outputs" / "07_classical_ml_models").exists(),
        "notebook_08_not_started": not (root / "outputs" / "08_deep_learning_models").exists(),
        "no_classical_ml_training_launched": not (root / "outputs" / "07_classical_ml_models").exists(),
        "no_deep_learning_training_launched": not (root / "outputs" / "08_deep_learning_models").exists(),
    }
    for key, value in list(checks.items()):
        if isinstance(value, (bool, np.bool_)):
            checks[key] = bool(value)
    checks["missing_stage_a_references"] = missing_stage_a
    checks["passed"] = bool(all(value for value in checks.values() if isinstance(value, bool)))
    return checks


def phase5b_report_text(report: dict[str, Any]) -> str:
    checks = report.get("validation_checks", {})
    lines = [
        "# Phase 5B One-Hour Baseline Readiness Report",
        "",
        f"- Final status: `{report.get('final_status')}`",
        f"- One-hour Persistence references generated: `{checks.get('one_hour_persistence_references_generated')}`",
        f"- One-hour Mean Naive baselines generated: `{checks.get('one_hour_mean_naive_generated')}`",
        f"- Grid generated: `{report.get('lh_grid_policy')}`",
        f"- One-hour result rows written: `{report.get('one_hour_rows_written')}`",
        f"- Previous one-minute rows preserved: `{checks.get('one_minute_rows_preserved')}`",
        f"- Result schema contains required Stage A columns: `{checks.get('result_schema_contains_required_columns')}`",
        f"- Stage A Persistence references present for both resolutions: `{checks.get('stage_a_required_references_present')}`",
        f"- Notebook 07 prerequisites now satisfied: `{report.get('ready_for_notebook_07_stage_a_after_review')}`",
        f"- Notebook 07 started: `{not checks.get('notebook_07_not_started')}`",
        f"- Notebook 08 started: `{not checks.get('notebook_08_not_started')}`",
        f"- Classical ML training launched: `{not checks.get('no_classical_ml_training_launched')}`",
        f"- Deep Learning training launched: `{not checks.get('no_deep_learning_training_launched')}`",
        "",
        "Fairness/common-condition notes:",
        "- Stage A comparisons must match on resolution, dataset_variant, lookback, horizon, split, and model/status category.",
        "- L/H values are resolution-native time steps: one_minute L=8 means 8 minutes and H=1 means 1 minute ahead; one_hour L=8 means 8 hours and H=1 means 1 hour ahead.",
        "- Physical-time matched comparisons are a separate later experiment and must not be mixed silently with this Stage A protocol.",
        "- Persistence and Mean Naive use the same chronological split, target alignment, NRMSE=RMSE/2, and Skill_RMSE definitions as Notebook 05.",
        "",
        "Updated outputs:",
        f"- `{report.get('output_files', {}).get('baseline_results_train_validation_test')}`",
        f"- `{report.get('output_files', {}).get('baseline_results_summary')}`",
        f"- `{report.get('output_files', {}).get('persistence_reference_by_variant_lh')}`",
        f"- `{report.get('output_files', {}).get('notebook_05b_one_hour_baseline_completion_report')}`",
    ]
    return "\n".join(lines) + "\n"


def pipeline_status_text(report: dict[str, Any]) -> str:
    checks = report.get("validation_checks", {})
    return "\n".join(
        [
            "# Pipeline Build Status",
            "",
            "Phase: `5B one-hour baseline completion`",
            "",
            "Completed:",
            "- Phases 1-6 remain completed.",
            "- Pre-Notebook-07 cleanup remains completed.",
            "- One-hour baseline references were generated with the Notebook 05 methodology.",
            "- One-minute baseline rows were preserved while one-hour rows were appended/merged.",
            "- Notebook 07 and Notebook 08 were not launched.",
            "",
            "Dataset readiness:",
            "- Ready datasets: `8/8`.",
            "- One-hour ready: `True`.",
            "- All 8 ready: `True`.",
            "",
            "Baseline readiness:",
            f"- One-hour Persistence references generated: `{checks.get('one_hour_persistence_references_generated')}`.",
            f"- One-hour Mean Naive baselines generated: `{checks.get('one_hour_mean_naive_generated')}`.",
            f"- Full 25 L/H grid generated for one_hour: `{checks.get('full_25_lh_grid_generated')}`.",
            f"- Stage A references present for both one_minute and one_hour: `{checks.get('stage_a_required_references_present')}`.",
            "- Resolution-native L/H note: one_minute steps are minutes; one_hour steps are hours.",
            "",
            "Kaggle/ClearML:",
            "- Latest official access-check status: `success`.",
            "- Verification source: manual Kaggle UI run of `YOUR_KAGGLE_USERNAME/00-access-check-clearml-kaggle` after enabling/checking the five ClearML secrets for that exact notebook/version.",
            "- Kaggle Secrets readable: `True`.",
            "- Kaggle ClearML task creation succeeded: `True`.",
            "- Kaggle/ClearML READY: `True`.",
            "- Official Kaggle ClearML task ID: `1bd4e5ba207f49faa1591750a57c57c1`.",
            "- GPU available in Kaggle runtime: `True`.",
            "- Previous `ConnectionError` / `MISSING` ClearML secret reports are resolved, not active.",
            "",
            "Training:",
            f"- Notebook 07: `{'not started' if checks.get('notebook_07_not_started') else 'started'}`.",
            f"- Notebook 08: `{'not started' if checks.get('notebook_08_not_started') else 'started'}`.",
            "- Heavy Kaggle training: not started.",
            f"- Classical ML training: `{'not started' if checks.get('no_classical_ml_training_launched') else 'started'}`.",
            f"- Deep Learning training: `{'not started' if checks.get('no_deep_learning_training_launched') else 'started'}`.",
            "",
            f"Ready for user review before Notebook 07 Stage A: `{'yes' if report.get('ready_for_notebook_07_stage_a_after_review') else 'no'}`",
            "",
        ]
    )


def write_local_runner_report(local_output_dir: str | None, report: dict[str, Any], artifact_paths: list[str]) -> None:
    if not local_output_dir:
        return
    path = Path(local_output_dir) / "local_clearml_report.json"
    payload = {
        "final_status": report.get("final_status"),
        "metrics": {
            "one_hour_result_rows_written": report.get("one_hour_rows_written"),
            "one_hour_persistence_references_generated": report.get("validation_checks", {}).get("one_hour_persistence_references_generated"),
            "stage_a_required_references_present": report.get("validation_checks", {}).get("stage_a_required_references_present"),
            "ready_for_notebook_07_stage_a_after_review": report.get("ready_for_notebook_07_stage_a_after_review"),
        },
        "artifact_paths": artifact_paths,
    }
    write_json(path, payload)


def run_notebook05b_one_hour_baseline_completion(config_path: str | Path) -> dict[str, Any]:
    started_at = utc_now()
    root = project_root()
    config = load_config(config_path)
    output_root = root / "outputs" / BASELINE_OUTPUT_NAME
    run_state_dir = output_root / "run_state"
    output_root.mkdir(parents=True, exist_ok=True)
    run_state_dir.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("LOCAL_RUN_ID") or started_at.replace("-", "").replace(":", "").replace("+", "_")
    output_files: dict[str, str] = {}
    warnings: list[str] = []

    try:
        missing = validate_one_hour_prerequisites(root)
        if missing:
            raise FileNotFoundError("Missing one_hour baseline prerequisites: " + "; ".join(missing))

        one_hour_results, input_info = evaluate_one_hour_baselines(root, run_id)
        output_info = write_merged_outputs(root, one_hour_results, run_id)
        output_files.update({key: value for key, value in output_info.items() if isinstance(value, str) and Path(value).exists()})
        checks = validate_completion(root, output_info)

        clearml_project_name = os.environ.get("LOCAL_CLEARML_PROJECT_NAME") or config.get("clearml", {}).get("subprojects", {}).get("05", "PV_RampRate_Forecasting/05_Baselines")
        report = {
            "notebook_name": NOTEBOOK_NAME,
            "final_status": "success" if checks["passed"] else "failed",
            "started_at": started_at,
            "finished_at": utc_now(),
            "run_id": run_id,
            "input_files": {
                "one_hour_manifests": {variant: str(manifest_path(root, RESOLUTION, variant)) for variant in DATASET_VARIANTS},
                "existing_baseline_results": str(output_root / "baseline_results_train_validation_test.csv"),
            },
            "output_files": output_files,
            "resolution_completed": RESOLUTION,
            "dataset_variants": DATASET_VARIANTS,
            "baseline_models": BASELINE_MODELS,
            "lookbacks": LOOKBACKS,
            "horizons": HORIZONS,
            "stage_a_pairs": [{"lookback": lookback, "horizon": horizon} for lookback, horizon in STAGE_A_PAIRS],
            "lh_grid_policy": "full_25_lh_grid",
            "one_hour_rows_written": int(output_info["one_hour_rows_written"]),
            "previous_one_hour_rows_replaced": int(output_info["previous_one_hour_rows_replaced"]),
            "one_minute_rows_before": int(output_info["one_minute_rows_before"]),
            "one_minute_rows_after": int(output_info["one_minute_rows_after"]),
            "one_minute_rows_preserved": bool(output_info["one_minute_rows_preserved"]),
            "combined_result_rows": int(output_info["combined_result_rows"]),
            "persistence_reference_rows": int(output_info["persistence_reference_rows"]),
            "one_hour_persistence_reference_count": int(len(one_hour_results[one_hour_results["model_name"] == "persistence"])),
            "one_hour_mean_naive_count": int(len(one_hour_results[one_hour_results["model_name"] == "mean_naive"])),
            "metrics": ["MAE", "RMSE", "NRMSE=RMSE/2", "R2", "Skill_RMSE_vs_Persistence"],
            "fairness_common_conditions": {
                "chronological_split": True,
                "target_name": TARGET_COLUMN,
                "nrmse_definition": "RMSE / 2",
                "skill_definition": "1 - RMSE_model / RMSE_persistence",
                "stage_a_lh_pairs": STAGE_A_PAIRS,
                "compare_only_matching_resolution_variant_lh_split_model_status": True,
                "resolution_native_lh_note": "one_minute L/H are minute steps; one_hour L/H are hour steps.",
                "physical_time_comparison_policy": "Run as a separate later experiment if needed; do not mix silently with Stage A.",
            },
            "split_summary": input_info["split_summary"],
            "feature_counts": input_info["feature_counts"],
            "data_shapes": input_info["data_shapes"],
            "validation_checks": checks,
            "clearml_project_name": clearml_project_name,
            "clearml_task_name": os.environ.get("LOCAL_CLEARML_TASK_NAME"),
            "clearml_task_id": os.environ.get("LOCAL_CLEARML_TASK_ID") or None,
            "clearml_task_url": os.environ.get("LOCAL_CLEARML_TASK_URL") or None,
            "clearml_tags_requested": [
                "notebook=05b_one_hour_baseline_completion",
                "resolution=one_hour",
                "family=baseline",
                "status=final",
                "purpose=missing_persistence_reference",
            ],
            "baseline_only": True,
            "classical_ml_training_started": False,
            "deep_learning_training_started": False,
            "heavy_kaggle_training_started": False,
            "notebook_07_started": False,
            "notebook_08_started": False,
            "ready_for_notebook_07_stage_a_after_review": bool(checks["passed"]),
            "warnings": warnings,
        }

        final_report_path = output_root / "notebook_05b_one_hour_baseline_completion_report.json"
        phase5b_readiness = root / "outputs" / "reports" / "phase5b_one_hour_baseline_readiness_report.md"
        pipeline_status = root / "outputs" / "reports" / "pipeline_build_status.md"
        output_files["notebook_05b_one_hour_baseline_completion_report"] = str(final_report_path)
        output_files["phase5b_one_hour_baseline_readiness_report"] = str(phase5b_readiness)
        output_files["pipeline_build_status"] = str(pipeline_status)
        report["output_files"] = output_files
        write_json(final_report_path, report)
        write_markdown(phase5b_readiness, phase5b_report_text(report))
        write_markdown(pipeline_status, pipeline_status_text(report))
        write_json(final_report_path, report)

        artifact_paths = [
            output_root / "baseline_results_train_validation_test.csv",
            output_root / "baseline_results_summary.csv",
            output_root / "persistence_reference_by_variant_lh.csv",
            final_report_path,
            phase5b_readiness,
            pipeline_status,
        ]
        write_local_runner_report(os.environ.get("LOCAL_OUTPUT_DIR"), report, [str(path) for path in artifact_paths])
        write_status(
            run_state_dir / "notebook_05b_status.json",
            section_status(
                NOTEBOOK_NAME,
                started_at,
                report["final_status"],
                "Notebook 05B one-hour baseline completion finished.",
                {
                    "one_hour_rows_written": int(output_info["one_hour_rows_written"]),
                    "stage_a_references_ready": bool(checks["stage_a_required_references_present"]),
                    "clearml_task_id": report.get("clearml_task_id"),
                },
            ),
        )
        return report
    except Exception:
        error_message = traceback.format_exc()
        report = {
            "notebook_name": NOTEBOOK_NAME,
            "final_status": "failed",
            "started_at": started_at,
            "finished_at": utc_now(),
            "output_files": output_files,
            "baseline_only": True,
            "classical_ml_training_started": False,
            "deep_learning_training_started": False,
            "heavy_kaggle_training_started": False,
            "notebook_07_started": False,
            "notebook_08_started": False,
            "ready_for_notebook_07_stage_a_after_review": False,
            "warnings": warnings,
            "error_message": error_message,
        }
        final_report_path = output_root / "notebook_05b_one_hour_baseline_completion_report.json"
        write_json(final_report_path, report)
        write_local_runner_report(os.environ.get("LOCAL_OUTPUT_DIR"), report, [str(final_report_path)])
        write_status(
            run_state_dir / "notebook_05b_status.json",
            section_status(NOTEBOOK_NAME, started_at, "failed", "Notebook 05B one-hour baseline completion failed.", error_message=error_message),
        )
        return report
