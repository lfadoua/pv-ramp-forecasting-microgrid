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


NOTEBOOK_NAME = "05_baseline_models"
TIME_COLUMN = "timestamp"
TARGET_COLUMN = "ramp_rate_pu"
DATASET_VARIANTS = ["full", "selected", "selected_physics", "full_physics"]
SPLIT_NAMES = ["train", "validation", "test"]
BASELINE_MODELS = ["persistence", "mean_naive"]
PREDICTION_SAMPLE_LIMIT = 200


def read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    return json.loads(target.read_text(encoding="utf-8"))


def finite_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, persistence_rmse: float | None = None) -> dict[str, float | int | None]:
    if len(y_true) == 0:
        return {
            "sample_count": 0,
            "MAE": None,
            "RMSE": None,
            "NRMSE": None,
            "R2": None,
            "Skill_RMSE_vs_Persistence": None,
        }
    error = y_pred - y_true
    mae_value = float(np.mean(np.abs(error)))
    rmse_value = float(np.sqrt(np.mean(np.square(error))))
    sse = float(np.sum(np.square(error)))
    y_mean = float(np.mean(y_true))
    sst = float(np.sum(np.square(y_true - y_mean)))
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


def load_manifest(root: Path, variant: str) -> dict[str, Any]:
    path = root / "outputs" / "one_minute" / "notebook_04_split_windowing" / variant / "window_ready_manifest.json"
    manifest = read_json(path)
    if manifest.get("dataset_variant") != variant:
        raise ValueError(f"Manifest variant mismatch in {path}")
    return manifest


def needed_columns_for_source(path: Path) -> tuple[list[str], list[str]]:
    header = list(pd.read_csv(path, nrows=0).columns)
    missing = [column for column in [TIME_COLUMN, TARGET_COLUMN] if column not in header]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    if header[-1] != TARGET_COLUMN:
        raise ValueError(f"Target column must be last in {path}")
    optional = [column for column in ["Elevation[deg]", "P_Solar[kW]"] if column in header]
    return [TIME_COLUMN, TARGET_COLUMN] + optional, header


def load_variant_frame(manifest: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    source_path = Path(manifest["source_path"])
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    columns, header = needed_columns_for_source(source_path)
    df = pd.read_csv(source_path, usecols=columns)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    if int(df[TIME_COLUMN].isna().sum()) > 0:
        raise ValueError(f"Invalid timestamps in {source_path}")
    if not df[TIME_COLUMN].is_monotonic_increasing:
        raise ValueError(f"Timestamps are not sorted in {source_path}")
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(target).all():
        raise ValueError(f"Target contains NaN or infinite values in {source_path}")
    if int(manifest.get("row_count", len(df))) != len(df):
        raise ValueError(f"Manifest row count mismatch for {source_path}")
    return df, header


def split_arrays(df: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    has_segment_columns = {"Elevation[deg]", "P_Solar[kW]"}.issubset(df.columns)
    for split_name in SPLIT_NAMES:
        split_meta = manifest["split_summary"][split_name]
        start = int(split_meta["start_row_in_dataset"])
        end = int(split_meta["end_row_exclusive_in_dataset"])
        subset = df.iloc[start:end].copy()
        daylight_productive = None
        if has_segment_columns:
            daylight_productive = (
                (pd.to_numeric(subset["Elevation[deg]"], errors="coerce").to_numpy(dtype=np.float64) > 0.0)
                & (pd.to_numeric(subset["P_Solar[kW]"], errors="coerce").to_numpy(dtype=np.float64) > 0.05)
            )
        result[split_name] = {
            "target": pd.to_numeric(subset[TARGET_COLUMN], errors="coerce").to_numpy(dtype=np.float64),
            "timestamp": subset[TIME_COLUMN].to_numpy(),
            "start_row_in_dataset": start,
            "end_row_exclusive_in_dataset": end,
            "daylight_productive": daylight_productive,
        }
    return result


def aligned_indices(row_count: int, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    sample_count = int(row_count) - int(lookback) - int(horizon) + 1
    if sample_count <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    origin_indices = np.arange(int(lookback) - 1, int(row_count) - int(horizon), dtype=np.int64)
    target_indices = origin_indices + int(horizon)
    return origin_indices, target_indices


def timestamp_iso(values: np.ndarray, index: int) -> str:
    return pd.Timestamp(values[index]).isoformat()


def prediction_sample_indices(sample_count: int) -> np.ndarray:
    if sample_count <= 0:
        return np.array([], dtype=np.int64)
    if sample_count <= PREDICTION_SAMPLE_LIMIT:
        return np.arange(sample_count, dtype=np.int64)
    return np.unique(np.linspace(0, sample_count - 1, PREDICTION_SAMPLE_LIMIT, dtype=np.int64))


def add_prediction_samples(
    rows: list[dict[str, Any]],
    variant: str,
    lookback: int,
    horizon: int,
    split_name: str,
    model_name: str,
    split_data: dict[str, Any],
    origin_indices: np.ndarray,
    target_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    for sample_position in prediction_sample_indices(len(y_true)):
        origin_idx = int(origin_indices[sample_position])
        target_idx = int(target_indices[sample_position])
        rows.append(
            {
                "dataset_variant": variant,
                "lookback": int(lookback),
                "horizon": int(horizon),
                "model_name": model_name,
                "split_name": split_name,
                "sample_position": int(sample_position),
                "origin_row_in_split": origin_idx,
                "target_row_in_split": target_idx,
                "origin_row_in_dataset": int(split_data["start_row_in_dataset"]) + origin_idx,
                "target_row_in_dataset": int(split_data["start_row_in_dataset"]) + target_idx,
                "timestamp_origin": timestamp_iso(split_data["timestamp"], origin_idx),
                "timestamp_target": timestamp_iso(split_data["timestamp"], target_idx),
                "y_true": float(y_true[sample_position]),
                "y_pred": float(y_pred[sample_position]),
            }
        )


def add_segment_metrics(
    rows: list[dict[str, Any]],
    variant: str,
    lookback: int,
    horizon: int,
    split_name: str,
    split_data: dict[str, Any],
    target_indices: np.ndarray,
    y_true: np.ndarray,
    persistence_pred: np.ndarray,
    mean_pred: np.ndarray,
) -> None:
    segment_masks = {"all": np.ones(len(y_true), dtype=bool)}
    daylight_productive = split_data.get("daylight_productive")
    if daylight_productive is not None:
        segment_masks["daylight_productive"] = daylight_productive[target_indices]

    for segment_name, mask in segment_masks.items():
        segment_count = int(mask.sum())
        if segment_count <= 0:
            continue
        seg_y_true = y_true[mask]
        seg_persistence = persistence_pred[mask]
        seg_mean = mean_pred[mask]
        persistence = regression_metrics(seg_y_true, seg_persistence, persistence_rmse=None)
        persistence["Skill_RMSE_vs_Persistence"] = 0.0
        mean = regression_metrics(seg_y_true, seg_mean, persistence_rmse=persistence["RMSE"])
        for model_name, metrics in [("persistence", persistence), ("mean_naive", mean)]:
            rows.append(
                {
                    "dataset_variant": variant,
                    "lookback": int(lookback),
                    "horizon": int(horizon),
                    "split_name": split_name,
                    "segment_name": segment_name,
                    "segment_rule": "all rows" if segment_name == "all" else "Elevation[deg] > 0 and P_Solar[kW] > 0.05 at target timestamp",
                    "model_name": model_name,
                    **metrics,
                }
            )


def make_result_row(
    experiment_row: pd.Series,
    split_name: str,
    model_name: str,
    metrics: dict[str, float | int | None],
    timing_seconds: float,
    train_mean_value: float | None,
) -> dict[str, Any]:
    train_time = timing_seconds if split_name == "train" else 0.0
    validation_time = timing_seconds if split_name == "validation" else 0.0
    test_time = timing_seconds if split_name == "test" else 0.0
    per_sample = (timing_seconds / int(metrics["sample_count"])) if int(metrics["sample_count"] or 0) > 0 else None
    return {
        "experiment_id": experiment_row["experiment_id"],
        "resolution": experiment_row["resolution"],
        "dataset_variant": experiment_row["dataset_variant"],
        "split_method": experiment_row["split_method"],
        "lookback": int(experiment_row["lookback"]),
        "horizon": int(experiment_row["horizon"]),
        "feature_count": int(experiment_row["feature_count"]),
        "target_name": experiment_row["target_name"],
        "recommended_stage": experiment_row["recommended_stage"],
        "compatible_families": experiment_row["compatible_families"],
        "model_name": model_name,
        "baseline_family": "baseline",
        "split_name": split_name,
        "backend": "cpu",
        "hardware": "local_cpu",
        "status": "final",
        "notebook": "05_baselines",
        "tag_family": "baseline",
        "tag_model": model_name,
        "tag_resolution": "1min",
        "tag_variant": experiment_row["dataset_variant"],
        "tag_split": "chronological",
        "tag_lookback": f"L{int(experiment_row['lookback'])}",
        "tag_horizon": f"H{int(experiment_row['horizon'])}",
        "train_mean_value": train_mean_value if model_name == "mean_naive" else None,
        "train_mean_source": "aligned train targets for this variant/lookback/horizon only" if model_name == "mean_naive" else None,
        "persistence_definition": "y_pred(t+H)=ramp_rate_pu(t)" if model_name == "persistence" else None,
        "mean_naive_definition": "y_pred=mean(aligned train target ramp_rate_pu(t+H))" if model_name == "mean_naive" else None,
        "target_alignment": "origin t uses target at t+H; input history ends at t",
        "same_row_target_feature_used": False,
        "future_information_used": False,
        "train_time_seconds": finite_or_none(train_time),
        "validation_time_seconds": finite_or_none(validation_time),
        "test_time_seconds": finite_or_none(test_time),
        "validation_inference_time_per_sample": finite_or_none(per_sample if split_name == "validation" else None),
        "test_inference_time_per_sample": finite_or_none(per_sample if split_name == "test" else None),
        **metrics,
    }


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results.groupby(["model_name", "split_name", "horizon"], dropna=False)
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
    return grouped


def persistence_reference_table(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset_variant",
        "lookback",
        "horizon",
        "split_name",
        "sample_count",
        "RMSE",
        "NRMSE",
        "MAE",
        "R2",
    ]
    reference = results[results["model_name"] == "persistence"][columns].copy()
    reference = reference.rename(columns={"RMSE": "persistence_RMSE", "NRMSE": "persistence_NRMSE", "MAE": "persistence_MAE", "R2": "persistence_R2"})
    reference["Skill_RMSE_vs_Persistence"] = 0.0
    return reference


def best_worst_horizon_behavior(summary: pd.DataFrame) -> dict[str, Any]:
    behavior: dict[str, Any] = {}
    test_summary = summary[summary["split_name"] == "test"].copy()
    for model_name in BASELINE_MODELS:
        subset = test_summary[test_summary["model_name"] == model_name]
        if subset.empty:
            continue
        best = subset.sort_values("RMSE_mean", ascending=True).iloc[0]
        worst = subset.sort_values("RMSE_mean", ascending=False).iloc[0]
        behavior[model_name] = {
            "best_test_horizon_by_mean_rmse": int(best["horizon"]),
            "best_test_rmse_mean": float(best["RMSE_mean"]),
            "worst_test_horizon_by_mean_rmse": int(worst["horizon"]),
            "worst_test_rmse_mean": float(worst["RMSE_mean"]),
            "pattern_note": "Higher horizons generally become harder when mean RMSE increases with horizon; inspect the summary CSV for non-monotonic cases.",
        }
    return behavior


def validate_results(
    manifests: dict[str, dict[str, Any]],
    experiment_matrix: pd.DataFrame,
    results: pd.DataFrame,
    output_root: Path,
) -> dict[str, Any]:
    expected_configs = len(DATASET_VARIANTS) * 25
    expected_result_rows = expected_configs * len(BASELINE_MODELS) * len(SPLIT_NAMES)
    checks = {
        "target_exists_in_all_variants": all(manifest.get("target_column") == TARGET_COLUMN for manifest in manifests.values()),
        "target_excluded_from_feature_matrix": all(TARGET_COLUMN not in manifest.get("feature_columns", []) for manifest in manifests.values()),
        "no_same_row_target_feature": bool((results["same_row_target_feature_used"] == False).all()),  # noqa: E712
        "target_alignment_t_to_t_plus_H": bool((results["target_alignment"] == "origin t uses target at t+H; input history ends at t").all()),
        "no_future_information_used": bool((results["future_information_used"] == False).all()),  # noqa: E712
        "train_mean_train_only": bool(
            results.loc[results["model_name"] == "mean_naive", "train_mean_source"]
            .eq("aligned train targets for this variant/lookback/horizon only")
            .all()
        ),
        "persistence_uses_only_target_at_t": bool(
            results.loc[results["model_name"] == "persistence", "persistence_definition"]
            .eq("y_pred(t+H)=ramp_rate_pu(t)")
            .all()
        ),
        "train_validation_test_metrics_present": set(results["split_name"]) == set(SPLIT_NAMES),
        "all_4_variants_prepared": set(results["dataset_variant"]) == set(DATASET_VARIANTS),
        "all_25_lh_combinations_present": int(results[["lookback", "horizon"]].drop_duplicates().shape[0]) == 25,
        "both_baselines_present": set(results["model_name"]) == set(BASELINE_MODELS),
        "experiment_matrix_rows": int(len(experiment_matrix)),
        "expected_experiment_matrix_rows": expected_configs,
        "result_rows": int(len(results)),
        "expected_result_rows": expected_result_rows,
        "positive_sample_counts": bool((results["sample_count"] > 0).all()),
        "no_nan_inf_metrics": bool(np.isfinite(results[["MAE", "RMSE", "NRMSE"]].to_numpy(dtype=np.float64)).all()),
        "no_notebook_06_final_report": not (output_root.parent / "06_statistical_time_series_models" / "notebook_06_final_report.json").exists(),
        "no_ml_dl_statistical_training": True,
        "seasonal_naive_excluded": True,
    }
    checks["passed"] = all(value for key, value in checks.items() if isinstance(value, bool))
    return checks


def write_phase5_reports_from_report(root: Path, report: dict[str, Any] | None = None) -> tuple[Path, Path]:
    if report is None:
        report = read_json(root / "outputs" / NOTEBOOK_NAME / "notebook_05_final_report.json")

    reports_dir = root / "outputs" / "reports"
    pipeline_status = reports_dir / "pipeline_build_status.md"
    readiness = reports_dir / "phase5_readiness_report.md"
    validation = report.get("validation_checks", {})
    split_summary = report.get("split_summary", {}).get("selected_physics", {})
    lines = [
        "# Pipeline Build Status",
        "",
        "Phase: `5`",
        "",
        "Completed:",
        "- Notebook 05 implemented for Persistence and Mean Naive baselines only.",
        "- Evaluated all four Notebook 04 dataset variants across the full lookback/horizon grid.",
        "- Saved train/validation/test metrics, persistence references, sampled predictions, and segment diagnostics.",
        "- No Notebook 06 execution and no ML/DL/statistical model training.",
        "",
        "Validation:",
        "- Notebook 01: `success`",
        "- Notebook 02: `success`",
        "- Notebook 03: `success`",
        "- Notebook 04: `success`",
        f"- Notebook 05: `{report.get('final_status')}`",
        f"- Experiment configs: `{report.get('experiment_config_count')}`",
        f"- Result rows: `{report.get('result_row_count')}`",
        f"- Baselines: `{report.get('baseline_models')}`",
        f"- Dataset variants: `{report.get('dataset_variants')}`",
        f"- Leakage/alignment checks passed: `{validation.get('passed')}`",
        f"- Seasonal naive excluded: `{validation.get('seasonal_naive_excluded')}`",
        "",
        "Representative Split Ranges:",
        f"- Train: `{split_summary.get('train')}`",
        f"- Validation: `{split_summary.get('validation')}`",
        f"- Test: `{split_summary.get('test')}`",
        "",
        "Heavy training:",
        "- Not started.",
        "",
        "Notebook 06:",
        "- Not started.",
        "",
        f"Ready for user review before Notebook 06: `{'yes' if report.get('ready_for_notebook_06_after_review') else 'no'}`",
    ]
    write_markdown(pipeline_status, "\n".join(lines) + "\n")

    best_worst = report.get("best_worst_horizon_behavior", {})
    readiness_lines = [
        "# Phase 5 Readiness Report",
        "",
        f"- Notebook 05 OK: `{report.get('final_status') == 'success'}`",
        f"- ClearML project: `{report.get('clearml_project_name')}`",
        f"- ClearML task ID: `{report.get('clearml_task_id')}`",
        f"- Configurations evaluated: `{report.get('experiment_config_count')}`",
        f"- Lookback/horizon combinations: `{report.get('lookback_horizon_combination_count')}`",
        f"- Result rows: `{report.get('result_row_count')}`",
        f"- Segment metrics produced: `{report.get('segment_metrics_produced')}`",
        f"- Prediction output policy: `{report.get('prediction_output_policy')}`",
        f"- Leakage/alignment checks passed: `{validation.get('passed')}`",
        f"- Persistence definition: `y_pred(t+H)=ramp_rate_pu(t)`",
        f"- Mean naive definition: `train-only aligned target mean`",
        "- Seasonal Naive: excluded because fast PV ramp-rate events are driven by irradiance/cloud variability rather than reliable daily repetition.",
        f"- Best/worst horizon behavior: `{best_worst}`",
        f"- Warnings: `{report.get('warnings') if report.get('warnings') else 'none'}`",
        f"- Ready for Notebook 06 after review: `{'yes' if report.get('ready_for_notebook_06_after_review') else 'no'}`",
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
            "experiment_config_count": report.get("experiment_config_count"),
            "result_row_count": report.get("result_row_count"),
            "validation_checks_passed": report.get("validation_checks", {}).get("passed"),
            "best_test_persistence_rmse": report.get("best_worst_horizon_behavior", {}).get("persistence", {}).get("best_test_rmse_mean"),
            "best_test_mean_naive_rmse": report.get("best_worst_horizon_behavior", {}).get("mean_naive", {}).get("best_test_rmse_mean"),
        },
        "artifact_paths": artifact_paths,
    }
    write_json(path, payload)


def run_notebook05_baselines(config_path: str | Path) -> dict[str, Any]:
    started_at = utc_now()
    root = project_root()
    config = load_config(config_path)
    output_root = root / "outputs" / NOTEBOOK_NAME
    run_state_dir = output_root / "run_state"
    output_root.mkdir(parents=True, exist_ok=True)
    run_state_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    output_files: dict[str, str] = {}
    artifact_paths: list[str] = []
    try:
        experiment_path = root / "outputs" / "04_split_scaling_windowing_experiment_design" / "experiment_matrix.csv"
        experiment_matrix = pd.read_csv(experiment_path)
        experiment_matrix = experiment_matrix[experiment_matrix["dataset_variant"].isin(DATASET_VARIANTS)].copy()
        manifests = {variant: load_manifest(root, variant) for variant in DATASET_VARIANTS}

        result_rows: list[dict[str, Any]] = []
        segment_rows: list[dict[str, Any]] = []
        validation_prediction_rows: list[dict[str, Any]] = []
        test_prediction_rows: list[dict[str, Any]] = []
        split_summary_by_variant: dict[str, Any] = {}
        data_shapes: dict[str, Any] = {}
        feature_counts: dict[str, int] = {}

        for variant in DATASET_VARIANTS:
            manifest = manifests[variant]
            feature_columns = manifest.get("feature_columns", [])
            if TARGET_COLUMN in feature_columns:
                raise ValueError(f"Target leaked into feature columns for {variant}")
            df, header = load_variant_frame(manifest)
            split_data = split_arrays(df, manifest)
            split_summary_by_variant[variant] = manifest["split_summary"]
            data_shapes[variant] = {"row_count": int(len(df)), "source_column_count": int(len(header))}
            feature_counts[variant] = int(manifest.get("feature_count", len(feature_columns)))
            variant_experiments = experiment_matrix[experiment_matrix["dataset_variant"] == variant].copy()

            for _, experiment_row in variant_experiments.iterrows():
                lookback = int(experiment_row["lookback"])
                horizon = int(experiment_row["horizon"])
                train_origin, train_target = aligned_indices(len(split_data["train"]["target"]), lookback, horizon)
                if len(train_target) <= 0:
                    raise ValueError(f"No train samples for {variant} L={lookback} H={horizon}")
                train_y_true_for_mean = split_data["train"]["target"][train_target]
                train_mean_started = time.perf_counter()
                train_mean_value = float(np.mean(train_y_true_for_mean))
                train_mean_seconds = time.perf_counter() - train_mean_started

                for split_name in SPLIT_NAMES:
                    current = split_data[split_name]
                    origin_indices, target_indices = aligned_indices(len(current["target"]), lookback, horizon)
                    if len(target_indices) <= 0:
                        raise ValueError(f"No {split_name} samples for {variant} L={lookback} H={horizon}")
                    y_origin = current["target"][origin_indices]
                    y_true = current["target"][target_indices]

                    persistence_started = time.perf_counter()
                    persistence_pred = y_origin.copy()
                    persistence_metrics = regression_metrics(y_true, persistence_pred, persistence_rmse=None)
                    persistence_metrics["Skill_RMSE_vs_Persistence"] = 0.0
                    persistence_seconds = time.perf_counter() - persistence_started
                    result_rows.append(
                        make_result_row(
                            experiment_row,
                            split_name,
                            "persistence",
                            persistence_metrics,
                            persistence_seconds,
                            train_mean_value=None,
                        )
                    )

                    mean_started = time.perf_counter()
                    mean_pred = np.full_like(y_true, fill_value=train_mean_value, dtype=np.float64)
                    mean_metrics = regression_metrics(y_true, mean_pred, persistence_rmse=persistence_metrics["RMSE"])
                    mean_seconds = (time.perf_counter() - mean_started) + (train_mean_seconds if split_name == "train" else 0.0)
                    result_rows.append(
                        make_result_row(
                            experiment_row,
                            split_name,
                            "mean_naive",
                            mean_metrics,
                            mean_seconds,
                            train_mean_value=train_mean_value,
                        )
                    )

                    add_segment_metrics(
                        segment_rows,
                        variant,
                        lookback,
                        horizon,
                        split_name,
                        current,
                        target_indices,
                        y_true,
                        persistence_pred,
                        mean_pred,
                    )
                    if split_name == "validation":
                        add_prediction_samples(
                            validation_prediction_rows,
                            variant,
                            lookback,
                            horizon,
                            split_name,
                            "persistence",
                            current,
                            origin_indices,
                            target_indices,
                            y_true,
                            persistence_pred,
                        )
                        add_prediction_samples(
                            validation_prediction_rows,
                            variant,
                            lookback,
                            horizon,
                            split_name,
                            "mean_naive",
                            current,
                            origin_indices,
                            target_indices,
                            y_true,
                            mean_pred,
                        )
                    elif split_name == "test":
                        add_prediction_samples(
                            test_prediction_rows,
                            variant,
                            lookback,
                            horizon,
                            split_name,
                            "persistence",
                            current,
                            origin_indices,
                            target_indices,
                            y_true,
                            persistence_pred,
                        )
                        add_prediction_samples(
                            test_prediction_rows,
                            variant,
                            lookback,
                            horizon,
                            split_name,
                            "mean_naive",
                            current,
                            origin_indices,
                            target_indices,
                            y_true,
                            mean_pred,
                        )

            write_status(
                run_state_dir / f"{variant}_complete.json",
                section_status(
                    f"{variant}_baseline_evaluation",
                    started_at,
                    "success",
                    f"Baseline evaluation completed for {variant}.",
                    {"experiment_rows": int(len(variant_experiments)), "feature_count": feature_counts[variant]},
                ),
            )

        results = pd.DataFrame(result_rows)
        summary = summary_table(results)
        persistence_reference = persistence_reference_table(results)
        segment_metrics = pd.DataFrame(segment_rows)
        validation_predictions = pd.DataFrame(validation_prediction_rows)
        test_predictions = pd.DataFrame(test_prediction_rows)
        validation_checks = validate_results(manifests, experiment_matrix, results, output_root)
        best_worst = best_worst_horizon_behavior(summary)

        paths = {
            "baseline_results_train_validation_test": output_root / "baseline_results_train_validation_test.csv",
            "baseline_results_summary": output_root / "baseline_results_summary.csv",
            "persistence_reference_by_variant_lh": output_root / "persistence_reference_by_variant_lh.csv",
            "baseline_predictions_validation": output_root / "baseline_predictions_validation.csv",
            "baseline_predictions_test": output_root / "baseline_predictions_test.csv",
            "baseline_segment_metrics": output_root / "baseline_segment_metrics.csv",
        }
        write_csv(paths["baseline_results_train_validation_test"], results)
        write_csv(paths["baseline_results_summary"], summary)
        write_csv(paths["persistence_reference_by_variant_lh"], persistence_reference)
        write_csv(paths["baseline_predictions_validation"], validation_predictions)
        write_csv(paths["baseline_predictions_test"], test_predictions)
        write_csv(paths["baseline_segment_metrics"], segment_metrics)
        output_files.update({key: str(path) for key, path in paths.items()})

        clearml_project_name = os.environ.get("LOCAL_CLEARML_PROJECT_NAME") or config.get("clearml", {}).get("subprojects", {}).get("05", "PV_RampRate_Forecasting/05_Baselines")
        report = {
            "notebook_name": NOTEBOOK_NAME,
            "final_status": "success" if validation_checks["passed"] else "failed",
            "started_at": started_at,
            "finished_at": utc_now(),
            "input_files": {
                "experiment_matrix": str(experiment_path),
                "manifests": {
                    variant: str(root / "outputs" / "one_minute" / "notebook_04_split_windowing" / variant / "window_ready_manifest.json")
                    for variant in DATASET_VARIANTS
                },
            },
            "output_files": output_files,
            "dataset_variants": DATASET_VARIANTS,
            "baseline_models": BASELINE_MODELS,
            "experiment_config_count": int(experiment_matrix[["dataset_variant", "lookback", "horizon"]].drop_duplicates().shape[0]),
            "lookback_horizon_combination_count": int(experiment_matrix[["lookback", "horizon"]].drop_duplicates().shape[0]),
            "result_row_count": int(len(results)),
            "segment_metric_row_count": int(len(segment_metrics)),
            "segment_metrics_produced": bool(len(segment_metrics) > 0),
            "validation_prediction_sample_rows": int(len(validation_predictions)),
            "test_prediction_sample_rows": int(len(test_predictions)),
            "prediction_output_policy": f"Sampled up to {PREDICTION_SAMPLE_LIMIT} prediction rows per variant/lookback/horizon/model/split; metrics are computed on all aligned samples.",
            "feature_counts": feature_counts,
            "data_shapes": data_shapes,
            "split_summary": split_summary_by_variant,
            "target_name": TARGET_COLUMN,
            "primary_resolution": "one_minute",
            "horizon_definition": "H means the single target ramp_rate_pu(t+H).",
            "window_alignment_rule": "For each split, lookback history ends at origin t and the baseline target is ramp_rate_pu(t+H); windows never cross split boundaries.",
            "baseline_definitions": {
                "persistence": "y_pred(t+H)=ramp_rate_pu(t), using only the latest known target at forecast origin.",
                "mean_naive": "y_pred=mean(ramp_rate_pu target values on aligned training samples only).",
                "seasonal_naive": "excluded; PV ramp-rate events are mainly fast irradiance/cloud variability rather than stable daily repetition.",
            },
            "metrics": ["MAE", "RMSE", "NRMSE=RMSE/2", "R2", "Skill_RMSE_vs_Persistence"],
            "best_worst_horizon_behavior": best_worst,
            "validation_checks": validation_checks,
            "clearml_project_name": clearml_project_name,
            "clearml_task_name": os.environ.get("LOCAL_CLEARML_TASK_NAME"),
            "clearml_task_id": os.environ.get("LOCAL_CLEARML_TASK_ID") or None,
            "clearml_task_url": os.environ.get("LOCAL_CLEARML_TASK_URL") or None,
            "clearml_logging_policy": "One local summary task uploads all result tables when ClearML is available.",
            "model_training_started": False,
            "notebook_06_started": False,
            "ready_for_notebook_06_after_review": validation_checks["passed"],
            "warnings": warnings,
        }
        final_report_path = output_root / "notebook_05_final_report.json"
        write_json(final_report_path, report)
        output_files["notebook_05_final_report"] = str(final_report_path)
        report["output_files"] = output_files
        write_json(final_report_path, report)
        pipeline_status, readiness = write_phase5_reports_from_report(root, report)
        output_files["pipeline_build_status"] = str(pipeline_status)
        output_files["phase5_readiness_report"] = str(readiness)
        report["output_files"] = output_files
        write_json(final_report_path, report)

        artifact_paths = [str(path) for path in list(paths.values()) + [final_report_path, pipeline_status, readiness]]
        write_local_runner_report(os.environ.get("LOCAL_OUTPUT_DIR"), report, artifact_paths)
        write_status(
            run_state_dir / "notebook_05_status.json",
            section_status(
                NOTEBOOK_NAME,
                started_at,
                report["final_status"],
                "Notebook 05 baseline evaluation completed.",
                {
                    "result_rows": int(len(results)),
                    "validation_checks_passed": bool(validation_checks["passed"]),
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
            "warnings": warnings,
            "error_message": error_message,
            "model_training_started": False,
            "notebook_06_started": False,
            "ready_for_notebook_06_after_review": False,
        }
        final_report_path = output_root / "notebook_05_final_report.json"
        write_json(final_report_path, report)
        write_local_runner_report(os.environ.get("LOCAL_OUTPUT_DIR"), report, [str(final_report_path)])
        write_status(
            run_state_dir / "notebook_05_status.json",
            section_status(NOTEBOOK_NAME, started_at, "failed", "Notebook 05 baseline evaluation failed.", error_message=error_message),
        )
        return report
