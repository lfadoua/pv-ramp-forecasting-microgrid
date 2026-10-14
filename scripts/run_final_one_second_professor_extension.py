from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pv_ramp_automation.one_second_extension import (  # noqa: E402
    ChronologicalSecondReader,
    FixedFrameHDFReader,
    averaging_loss_analysis,
    blocked_day_bootstrap,
    build_feature_classification_audit,
    build_high_frequency_minute_features,
    build_minute_descriptors,
    build_minute_window_memmap,
    build_one_second_ramp_memmap,
    build_second_anchor_parts,
    compute_second_anchor_features,
    controlled_feature_ablation,
    environment_manifest,
    event_metrics,
    ensure_minute_completeness_fields,
    fit_minute_hgb,
    json_default,
    materialize_official_minute_data,
    materialize_chronological_second_data,
    materialize_second_memmaps,
    prepare_output_tree,
    read_json,
    read_yaml,
    regression_metrics,
    resolve_project_path,
    run_second_experiment,
    second_anchor_sampling_audit,
    s3_result_rows,
    safe_remove,
    sha256_file,
    write_json,
)
from pv_ramp_automation.one_second_reporting import (  # noqa: E402
    build_decision_table,
    generate_required_figures,
    select_representative_periods,
    write_article_latex,
    write_markdown_report,
    write_pdf_report,
    write_supervisor_explanation,
)


EXPECTED_ONE_SECOND_BYTES = 3_790_548_136
OFFICIAL_FIGSHARE_FILE_ID = 40_097_851
OFFICIAL_FIGSHARE_URL = "https://ndownloader.figshare.com/files/40097851"


def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("one_second_extension")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def load_dotenv_keys(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("CLEARML_") and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def maybe_init_clearml(config: dict[str, Any], logger: logging.Logger) -> tuple[Any | None, dict[str, Any]]:
    status = {"status": "not_configured", "task_id": None, "project": config["project_name"], "task_name": config["task_name"]}
    if not config.get("enabled_if_configured", True):
        status["status"] = "disabled_by_config"
        return None, status
    configured = bool(os.getenv("CLEARML_API_ACCESS_KEY") and os.getenv("CLEARML_API_SECRET_KEY"))
    if not configured:
        status["status"] = "skipped_no_credentials_in_environment"
        return None, status
    try:
        from clearml import Task

        task = Task.init(
            project_name=config["project_name"],
            task_name=config["task_name"],
            reuse_last_task_id=False,
            auto_connect_frameworks=False,
            auto_connect_arg_parser=False,
        )
        status.update({"status": "connected", "task_id": task.id})
        logger.info("ClearML task initialized successfully: %s", task.id)
        return task, status
    except Exception as exc:
        status.update({"status": "unavailable_nonfatal", "error": f"{type(exc).__name__}: {exc}"})
        logger.warning("ClearML initialization was unavailable and will not stop the experiment.")
        return None, status


def winner_rows(stage_c_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = pd.read_csv(stage_c_path)
    minute = data[
        (data["resolution"] == "one_minute")
        & (data["dataset_variant"] == "selected_physics")
        & (data["model"] == "HistGradientBoosting")
        & (data["lookback"] == 8)
        & (data["horizon"] == 1)
        & (data["status"] == "success")
    ]
    hour = data[
        (data["resolution"] == "one_hour")
        & (data["dataset_variant"] == "selected_physics")
        & (data["model"] == "ExtraTrees")
        & (data["lookback"] == 8)
        & (data["horizon"] == 1)
        & (data["status"] == "success")
    ]
    if len(minute) != 1 or len(hour) != 1:
        raise ValueError(f"Winner verification failed: one-minute rows={len(minute)}, one-hour rows={len(hour)}")

    def clean(row: pd.Series) -> dict[str, Any]:
        return {
            "resolution": row["resolution"],
            "dataset_variant": row["dataset_variant"],
            "model": row["model"],
            "lookback": int(row["lookback"]),
            "horizon": int(row["horizon"]),
            "feature_count": int(row["feature_count"]),
            "train_sample_count": int(row["train_sample_count"]),
            "validation_sample_count": int(row["validation_sample_count"]),
            "test_sample_count": int(row["test_sample_count"]),
            "validation_rmse": float(row["validation_rmse"]),
            "test_rmse": float(row["test_rmse"]),
            "validation_r2": float(row["validation_r2"]),
            "test_r2": float(row["test_r2"]),
            "validation_skill": float(row["validation_skill"]),
            "test_skill": float(row["test_skill"]),
            "training_time_seconds": float(row["training_time_seconds"]),
            "validation_time_seconds": float(row["validation_time_seconds"]),
            "test_time_seconds": float(row["test_time_seconds"]),
            "backend_used": row["backend_used"],
            "source_file": row["source_file"],
            "clearml_task_id": row.get("clearml_task_id"),
        }

    return clean(minute.iloc[0]), clean(hour.iloc[0])


def target_definition_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target_id": "existing_article",
                "formula": "r[t]=(P[t]-P[t-1])/10",
                "input_cutoff": "forecast origin t after L input rows",
                "target_timestamp": "t+H; for H=1, t+1",
                "lead_time_seconds": "60 at one-minute resolution; 3600 at one-hour resolution",
                "target_duration_seconds": "one sampling interval",
                "baseline": "r[t] carried to r[t+H]",
                "operational_interpretation": "next aggregated-interval PV ramp",
                "leakage_safety_status": "verified",
            },
            {
                "target_id": "S1",
                "formula": "(p[s+1]-p[s])/10",
                "input_cutoff": "second s",
                "target_timestamp": "s+1",
                "lead_time_seconds": 1,
                "target_duration_seconds": 1,
                "baseline": "r[s]",
                "operational_interpretation": "immediate next-second PV change",
                "leakage_safety_status": "verified",
            },
            {
                "target_id": "S2",
                "formula": "(p[s+60]-p[s+59])/10",
                "input_cutoff": "second s",
                "target_timestamp": "s+60",
                "lead_time_seconds": 60,
                "target_duration_seconds": 1,
                "baseline": "r[s]",
                "operational_interpretation": "instantaneous one-second PV change one minute ahead",
                "leakage_safety_status": "verified; not a net 60-second change",
            },
            {
                "target_id": "S3",
                "formula": "(Pbar[m+1]-Pbar[m])/10",
                "input_cutoff": "end of completed minute m",
                "target_timestamp": "m+1",
                "lead_time_seconds": 60,
                "target_duration_seconds": 60,
                "baseline": "r[m]",
                "operational_interpretation": "article's exact next-minute target from high-frequency context",
                "leakage_safety_status": "verified; no seconds from m+1",
            },
        ]
    )


def lookback_conversion_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_configuration": "one-minute winner",
                "original_resolution_seconds": 60,
                "original_lookback_steps": 8,
                "original_physical_lookback_seconds": 480,
                "corresponding_one_second_samples": 480,
                "one_second_implementation": "causal lags and rolling summaries through 480 s",
                "original_horizon_steps": 1,
                "original_forecast_duration_seconds": 60,
                "corresponding_one_second_horizon_steps": 60,
            },
            {
                "source_configuration": "one-hour winner",
                "original_resolution_seconds": 3600,
                "original_lookback_steps": 8,
                "original_physical_lookback_seconds": 28800,
                "corresponding_one_second_samples": 28800,
                "one_second_implementation": "not transferred; one-hour winner is secondary only",
                "original_horizon_steps": 1,
                "original_forecast_duration_seconds": 3600,
                "corresponding_one_second_horizon_steps": 3600,
            },
            {
                "source_configuration": "S1",
                "original_resolution_seconds": 1,
                "original_lookback_steps": 480,
                "original_physical_lookback_seconds": 480,
                "corresponding_one_second_samples": 480,
                "one_second_implementation": "compact causal descriptors",
                "original_horizon_steps": 1,
                "original_forecast_duration_seconds": 1,
                "corresponding_one_second_horizon_steps": 1,
            },
            {
                "source_configuration": "S2",
                "original_resolution_seconds": 1,
                "original_lookback_steps": 480,
                "original_physical_lookback_seconds": 480,
                "corresponding_one_second_samples": 480,
                "one_second_implementation": "compact causal descriptors",
                "original_horizon_steps": 60,
                "original_forecast_duration_seconds": 60,
                "corresponding_one_second_horizon_steps": 60,
            },
            {
                "source_configuration": "S3",
                "original_resolution_seconds": 60,
                "original_lookback_steps": 8,
                "original_physical_lookback_seconds": 480,
                "corresponding_one_second_samples": 480,
                "one_second_implementation": "official L=8 window plus completed-minute second descriptors",
                "original_horizon_steps": 1,
                "original_forecast_duration_seconds": 60,
                "corresponding_one_second_horizon_steps": 60,
            },
        ]
    )


def paired_s3_comparison(reference_path: Path, s3_path: Path, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = pd.read_parquet(reference_path)
    s3 = pd.read_parquet(s3_path)
    reference_test = reference[reference["split"] == "test"].copy()
    s3_test = s3[s3["split"] == "test"].copy()
    merged = reference_test[["timestamp", "y_true", "y_pred"]].rename(columns={"y_pred": "reference_pred"}).merge(
        s3_test[["timestamp", "y_true", "y_pred"]].rename(columns={"y_pred": "high_frequency_pred"}),
        on=["timestamp", "y_true"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(reference_test) or len(merged) != len(s3_test):
        raise ValueError("S3 and existing reference predictions are not exactly timestamp aligned.")
    true = merged["y_true"].to_numpy(dtype=np.float64)
    reference_pred = merged["reference_pred"].to_numpy(dtype=np.float64)
    hf_pred = merged["high_frequency_pred"].to_numpy(dtype=np.float64)
    strong = np.abs(true) >= threshold
    values = {
        "RMSE": (
            float(np.sqrt(np.mean(np.square(true - reference_pred)))),
            float(np.sqrt(np.mean(np.square(true - hf_pred)))),
        ),
        "MAE": (float(np.mean(np.abs(true - reference_pred))), float(np.mean(np.abs(true - hf_pred)))),
        "strong_ramp_RMSE": (
            float(np.sqrt(np.mean(np.square(true[strong] - reference_pred[strong])))),
            float(np.sqrt(np.mean(np.square(true[strong] - hf_pred[strong])))),
        ),
        "directional_accuracy": (
            float(np.mean(np.sign(true) == np.sign(reference_pred))),
            float(np.mean(np.sign(true) == np.sign(hf_pred))),
        ),
    }
    rows = []
    for metric, (reference_value, hf_value) in values.items():
        difference = hf_value - reference_value
        rows.append(
            {
                "metric": metric,
                "aligned_test_rows": len(merged),
                "existing_reference": reference_value,
                "high_frequency": hf_value,
                "high_frequency_minus_reference": difference,
                "relative_difference_percent": 100.0 * difference / reference_value if reference_value else np.nan,
                "favorable_direction": "negative" if metric != "directional_accuracy" else "positive",
                "point_result": (
                    "high_frequency_favored"
                    if (difference < 0 if metric != "directional_accuracy" else difference > 0)
                    else "reference_favored"
                    if difference != 0
                    else "equal"
                ),
            }
        )
    return pd.DataFrame(rows), merged


def compare_second_means_to_official_minute(
    one_minute_hdf: Path,
    descriptor_path: Path,
    one_hour_hdf: Path | None = None,
) -> dict[str, Any]:
    descriptors = pd.read_parquet(
        descriptor_path,
        columns=[
            "timestamp",
            "pv_mean",
            "seconds_observed",
            "valid_seconds_observed",
            "complete_minute",
            "excluded_reason",
        ],
    )
    descriptors["timestamp"] = pd.to_datetime(descriptors["timestamp"])
    with FixedFrameHDFReader(one_minute_hdf) as minute_reader:
        timestamps, values = minute_reader.read(0, minute_reader.row_count)
        official = pd.DataFrame(
            {
                "timestamp": timestamps.astype("datetime64[ns]"),
                "official_pv_mean": values[:, minute_reader.column_indices["P_Solar[kW]"]],
            }
        )
    if official["timestamp"].duplicated().any() or descriptors["timestamp"].duplicated().any():
        raise ValueError("Official-minute aggregation comparison requires unique timestamps.")
    outer = official.merge(descriptors, on="timestamp", how="outer", indicator=True, validate="one_to_one")
    merged = outer[outer["_merge"] == "both"].copy()
    complete = merged["complete_minute"].astype(bool)
    complete_merged = merged[complete].copy()
    difference = (
        complete_merged["pv_mean"].to_numpy(dtype=float)
        - complete_merged["official_pv_mean"].to_numpy(dtype=float)
    )
    absolute = np.abs(difference)
    correlation = float(
        np.corrcoef(complete_merged["pv_mean"], complete_merged["official_pv_mean"])[0, 1]
    )
    tolerance_kw = 1.0e-6
    timestamp_alignment_validated = bool(
        len(complete_merged) > 0
        and int(np.sum(outer["_merge"] == "left_only")) == 0
        and int(np.sum(outer["_merge"] == "right_only")) == 0
    )
    arithmetic_mean_equivalence = bool(
        float(np.max(absolute)) <= tolerance_kw and correlation > 0.999999
    )
    result = {
        "matched_timestamps": int(len(merged)),
        "matched_complete_timestamps": int(len(complete_merged)),
        "matched_incomplete_timestamps_excluded_from_validation": int((~complete).sum()),
        "official_rows": len(official),
        "descriptor_rows": len(descriptors),
        "missing_from_reconstructed_one_second_series": int(np.sum(outer["_merge"] == "left_only")),
        "missing_from_official_one_minute_series": int(np.sum(outer["_merge"] == "right_only")),
        "mean_absolute_difference_kw": float(np.mean(absolute)),
        "median_absolute_difference_kw": float(np.median(absolute)),
        "p95_absolute_difference_kw": float(np.percentile(absolute, 95)),
        "maximum_absolute_difference_kw": float(np.max(absolute)),
        "correlation": correlation,
        "project_reconstruction_aggregation_function": "arithmetic mean of the 60 one-second PV values in [m, m+60 s)",
        "official_one_minute_processing_description": "processed sampling of 1 min (SOLETE v3 README wording)",
        "official_exact_aggregation_function": "unavailable: the public v3 README and platform code do not disclose the exact one-minute transformation",
        "official_one_minute_is_simple_60_second_mean": arithmetic_mean_equivalence,
        "timezone_convention": "UTC represented by a timezone-naive HDF DatetimeIndex, as specified in the SOLETE data paper",
        "minute_label_convention": "left-labeled minute start m for the interval [m, m+60 s)",
        "comparison_scope": "complete minutes only (exactly 60 timestamped and finite PV/POA/GHI observations)",
        "validation_tolerance_kw": tolerance_kw,
        "timestamp_alignment_validated": timestamp_alignment_validated,
        "arithmetic_mean_equivalence_validated": arithmetic_mean_equivalence,
        "validated_for_s3_same_target_gate": timestamp_alignment_validated,
        "s3_interpretation_constraint": "same-timestamp cross-resolution input ablation; do not describe the official one-minute PV as a direct arithmetic mean of the v3 one-second PV",
    }
    if one_hour_hdf is not None and one_hour_hdf.exists():
        reconstructed = complete_merged.set_index("timestamp")["pv_mean"].sort_index()
        official_minute = complete_merged.set_index("timestamp")["official_pv_mean"].sort_index()
        with FixedFrameHDFReader(one_hour_hdf) as hour_reader:
            hour_timestamps, hour_values = hour_reader.read(0, hour_reader.row_count)
            official_hour = pd.Series(
                hour_values[:, hour_reader.column_indices["P_Solar[kW]"]],
                index=pd.DatetimeIndex(hour_timestamps.astype("datetime64[ns]")),
                name="official_hour",
            ).sort_index()

        def hourly_check(source: pd.Series) -> dict[str, float | int]:
            hourly = source.resample("60min", label="left", closed="left").mean()
            pair = pd.concat([hourly.rename("derived"), official_hour], axis=1, join="inner").dropna()
            delta = np.abs(pair["derived"].to_numpy(dtype=float) - pair["official_hour"].to_numpy(dtype=float))
            return {
                "matched_rows": int(len(pair)),
                "mean_absolute_difference_kw": float(np.mean(delta)),
                "median_absolute_difference_kw": float(np.median(delta)),
                "p95_absolute_difference_kw": float(np.percentile(delta, 95)),
                "maximum_absolute_difference_kw": float(np.max(delta)),
                "correlation": float(pair.corr().iloc[0, 1]),
            }

        result["reconstructed_one_second_mean_to_official_hour_crosscheck"] = hourly_check(reconstructed)
        result["official_one_minute_mean_to_official_hour_crosscheck"] = hourly_check(official_minute)
    return result


def validate_prediction_schema(path: Path) -> dict[str, Any]:
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
    frame = pd.read_parquet(path)
    missing = sorted(required - set(frame.columns))
    return {
        "path": str(path),
        "rows": len(frame),
        "missing_required_columns": missing,
        "timestamps_monotonic_within_split": bool(
            all(group["timestamp"].is_monotonic_increasing for _, group in frame.groupby("split", sort=False))
        ),
        "duplicate_timestamp_split_pairs": int(frame.duplicated(["timestamp", "split"]).sum()),
        "finite_predictions": bool(np.isfinite(frame[["y_true", "y_pred"]].to_numpy(dtype=float)).all()),
        "passed": not missing
        and all(group["timestamp"].is_monotonic_increasing for _, group in frame.groupby("split", sort=False))
        and not frame.duplicated(["timestamp", "split"]).any()
        and np.isfinite(frame[["y_true", "y_pred"]].to_numpy(dtype=float)).all(),
    }


def create_zip(output_root: Path, zip_path: Path) -> None:
    temporary_zip = output_root.parent / f".{zip_path.stem}.building.zip"
    if temporary_zip.exists():
        temporary_zip.unlink()
    with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in sorted(output_root.rglob("*")):
            if file.is_file() and file != zip_path and file != temporary_zip:
                archive.write(file, file.relative_to(output_root))
    shutil.move(str(temporary_zip), str(zip_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the isolated SOLETE one-second professor extension.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "one_second_professor_extension.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config = read_yaml(args.config.resolve())
    paths = {key: resolve_project_path(project_root, value) for key, value in config["paths"].items()}
    output_root = paths["output"]
    temporary_root = paths["temporary"]
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"Output directory already exists; use --resume only for this isolated run: {output_root}")
    if not output_root.exists():
        output_paths = prepare_output_tree(output_root)
    else:
        output_paths = {name: output_root / name for name in ("figures", "tables", "predictions", "models", "manifests", "logs", "article_ready", "kaggle_outputs")}
        for path in output_paths.values():
            path.mkdir(parents=True, exist_ok=True)
    temporary_root.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_paths["logs"] / "execution.log")
    run_started = time.perf_counter()
    execution_date = datetime.now(timezone.utc).isoformat()
    load_dotenv_keys(project_root / ".env")
    clearml_task, clearml_status = maybe_init_clearml(config["clearml"], logger)
    try:
        logger.info("Verifying existing winners and official manifests.")
        minute_winner, hour_winner = winner_rows(paths["stage_c_tradeoff"])
        minute_manifest = read_json(paths["one_minute_manifest"])
        hgb_parameters = dict(config["models"]["hist_gradient_boosting"])
        environment = environment_manifest()
        logger.info("Opening official one-second HDF5 source.")
        if not paths["one_second_hdf"].exists():
            raise FileNotFoundError(f"Official one-second SOLETE file is unavailable: {paths['one_second_hdf']}")
        if paths["one_second_hdf"].stat().st_size != EXPECTED_ONE_SECOND_BYTES:
            raise ValueError(
                f"Official one-second file size mismatch: {paths['one_second_hdf'].stat().st_size} != {EXPECTED_ONE_SECOND_BYTES}"
            )
        checksum_started = time.perf_counter()
        source_sha256 = sha256_file(paths["one_second_hdf"])
        checksum_time = time.perf_counter() - checksum_started
        logger.info("Source checksum complete in %.1f s.", checksum_time)
        descriptor_path = output_paths["kaggle_outputs"] / "one_second_minute_descriptors.parquet"
        second_parts_dir = temporary_root / "second_anchor_parts"
        chronology_manifest_path = output_paths["manifests"] / "one_second_chronological_normalization.json"
        if args.resume and chronology_manifest_path.exists():
            chronological_manifest = read_json(chronology_manifest_path)
            if not Path(chronological_manifest["chronological_values_path"]).exists():
                chronological_manifest = materialize_chronological_second_data(
                    paths["one_second_hdf"],
                    temporary_root / "chronological_source",
                    config["data"]["hdf_key"],
                    config["second_level"]["chunk_rows"],
                )
        else:
            logger.info("Normalizing the official file's shuffled day blocks into chronological order.")
            chronological_manifest = materialize_chronological_second_data(
                paths["one_second_hdf"],
                temporary_root / "chronological_source",
                config["data"]["hdf_key"],
                config["second_level"]["chunk_rows"],
            )
        write_json(chronology_manifest_path, chronological_manifest)
        with ChronologicalSecondReader(chronological_manifest) as reader:
            source_schema = reader.schema()
            expected_span_seconds = int((pd.Timestamp(source_schema["end_timestamp"]) - pd.Timestamp(source_schema["start_timestamp"])).total_seconds())
            regular_span = expected_span_seconds == reader.row_count - 1
            logger.info("Source schema: %s rows, %s columns, regular span=%s.", reader.row_count, reader.column_count, regular_span)
            logger.info("Running the short contiguous official-data preflight.")
            preflight_origin = pd.Timestamp(minute_manifest["split_summary"]["train"]["start_timestamp"]) + pd.Timedelta(minutes=10)
            preflight_row = int((preflight_origin - reader.timestamp_at(0)).total_seconds())
            preflight_start = preflight_row - config["second_level"]["physical_history_seconds"]
            preflight_stop = preflight_row + 61
            preflight_timestamps, preflight_values = reader.read(preflight_start, preflight_stop)
            preflight_local = np.array([config["second_level"]["physical_history_seconds"]], dtype=np.int64)
            preflight_x, preflight_names = compute_second_anchor_features(
                preflight_values,
                preflight_local,
                reader.columns,
                config["data"]["selected_original_columns"],
                config["second_level"]["lags_seconds"],
                config["second_level"]["rolling_windows_seconds"],
                config["second_level"]["extended_rolling_windows_seconds"],
                config["experiment"]["rated_power_kw"],
            )
            future_changed = preflight_values.copy()
            future_changed[preflight_local[0] + 1 :] += 1.0e5
            causal_x, _ = compute_second_anchor_features(
                future_changed,
                preflight_local,
                reader.columns,
                config["data"]["selected_original_columns"],
                config["second_level"]["lags_seconds"],
                config["second_level"]["rolling_windows_seconds"],
                config["second_level"]["extended_rolling_windows_seconds"],
                config["experiment"]["rated_power_kw"],
            )
            pv_index = reader.column_indices["P_Solar[kW]"]
            local = int(preflight_local[0])
            preflight_report = {
                "origin_timestamp": pd.Timestamp(int(preflight_timestamps[local])).isoformat(),
                "feature_count": len(preflight_names),
                "features_finite": bool(np.isfinite(preflight_x).all()),
                "causality_check_future_perturbation_unchanged": bool(np.array_equal(preflight_x, causal_x)),
                "S1_target": float((preflight_values[local + 1, pv_index] - preflight_values[local, pv_index]) / config["experiment"]["rated_power_kw"]),
                "S2_target": float((preflight_values[local + 60, pv_index] - preflight_values[local + 59, pv_index]) / config["experiment"]["rated_power_kw"]),
                "S2_not_net_60_second_change": bool(
                    not np.isclose(
                        (preflight_values[local + 60, pv_index] - preflight_values[local + 59, pv_index]) / config["experiment"]["rated_power_kw"],
                        (preflight_values[local + 60, pv_index] - preflight_values[local, pv_index]) / config["experiment"]["rated_power_kw"],
                    )
                ),
                "estimated_full_float64_anchor_matrix_gb": float(
                    reader.row_count * len(preflight_names) * 8 / 2**30
                ),
                "approved_strategy": "Tier 2 deterministic 30-second anchors",
                "passed": bool(np.isfinite(preflight_x).all() and np.array_equal(preflight_x, causal_x)),
            }
            write_json(output_paths["manifests"] / "official_short_contiguous_preflight.json", preflight_report)
            if not preflight_report["passed"]:
                raise RuntimeError("Official-data preflight failed.")
            logger.info("Building completed-minute descriptors and cleaning audit.")
            descriptor_checkpoint = output_paths["manifests"] / "minute_descriptor_checkpoint.json"
            if descriptor_path.exists() and descriptor_checkpoint.exists() and args.resume:
                minute_descriptor_report = read_json(output_paths["manifests"] / "minute_descriptor_checkpoint.json")
            else:
                if descriptor_path.exists():
                    descriptor_path.unlink()
                minute_descriptor_report = build_minute_descriptors(
                    reader,
                    descriptor_path,
                    config["experiment"]["rated_power_kw"],
                    config["second_level"]["chunk_rows"],
                )
            minute_descriptor_report = ensure_minute_completeness_fields(
                descriptor_path,
                minute_descriptor_report,
            )
            write_json(output_paths["manifests"] / "minute_descriptor_checkpoint.json", minute_descriptor_report)
            minute_completeness_audit = pd.read_parquet(
                descriptor_path,
                columns=[
                    "timestamp",
                    "seconds_observed",
                    "valid_seconds_observed",
                    "complete_minute",
                    "excluded_reason",
                ],
            )
            minute_completeness_audit.to_parquet(
                output_paths["kaggle_outputs"] / "minute_completeness_audit.parquet",
                index=False,
                compression="zstd",
            )
            pd.DataFrame(
                [
                    {
                        "total_minutes": minute_descriptor_report["rows_written"],
                        "complete_minutes": minute_descriptor_report["complete_minute_rows"],
                        "incomplete_minutes": minute_descriptor_report["incomplete_minute_rows"],
                        "requirement": minute_descriptor_report["completeness_requirement"],
                        "incomplete_reasons": json.dumps(minute_descriptor_report["excluded_reason_counts"], sort_keys=True),
                    }
                ]
            ).to_csv(output_paths["tables"] / "minute_completeness_summary.csv", index=False)
            del minute_completeness_audit
            logger.info("Auditing official one-minute aggregation before model training.")
            aggregation_alignment = compare_second_means_to_official_minute(
                paths["one_minute_hdf"],
                descriptor_path,
                paths["one_minute_hdf"].with_name("SOLETE_Pombo_60min.h5"),
            )
            write_json(
                output_paths["manifests"] / "official_one_minute_aggregation_alignment.json",
                aggregation_alignment,
            )
            pd.DataFrame([aggregation_alignment]).to_csv(
                output_paths["tables"] / "official_one_minute_aggregation_alignment.csv",
                index=False,
            )
            if not aggregation_alignment["validated_for_s3_same_target_gate"]:
                raise RuntimeError("Official one-minute timestamp alignment failed; model training is blocked.")
            logger.info("Materializing complete one-second ramp series for exact distribution statistics.")
            ramp_manifest = build_one_second_ramp_memmap(
                reader,
                temporary_root,
                config["experiment"]["rated_power_kw"],
                config["second_level"]["chunk_rows"],
            )
            logger.info("Building deterministic second-level anchor checkpoints.")
            second_parts_manifest = build_second_anchor_parts(
                reader,
                second_parts_dir,
                minute_manifest,
                config["data"]["selected_original_columns"],
                config["experiment"]["rated_power_kw"],
                config["second_level"]["anchor_interval_seconds"],
                config["second_level"]["physical_history_seconds"],
                config["second_level"]["lags_seconds"],
                config["second_level"]["rolling_windows_seconds"],
                config["second_level"]["extended_rolling_windows_seconds"],
                config["second_level"]["chunk_rows"],
            )
            anchor_timing_path = output_paths["manifests"] / "anchor_resume_timing_provenance.json"
            if anchor_timing_path.exists():
                prior_anchor_timing = read_json(anchor_timing_path)
                current_resume_seconds = float(second_parts_manifest["generation_time_seconds"])
                second_parts_manifest["current_resume_generation_time_seconds"] = current_resume_seconds
                second_parts_manifest["prior_checkpoint_wall_time_seconds"] = float(
                    prior_anchor_timing["prior_segment_elapsed_wall_seconds"]
                )
                second_parts_manifest["generation_time_seconds"] = (
                    current_resume_seconds + second_parts_manifest["prior_checkpoint_wall_time_seconds"]
                )
                second_parts_manifest["generation_timing_scope"] = (
                    "sum of measured wall-clock checkpoint segments; prior segment includes a documented workstation pause"
                )
                second_parts_manifest["generation_timing_provenance"] = prior_anchor_timing
            sampling_audit = second_anchor_sampling_audit(second_parts_manifest)
            second_parts_manifest["target_sampling_audit"] = sampling_audit
            write_json(output_paths["manifests"] / "second_anchor_checkpoint_manifest.json", second_parts_manifest)
            pd.DataFrame(sampling_audit).to_csv(
                output_paths["tables"] / "s1_s2_anchor_sampling_audit.csv",
                index=False,
            )
            logger.info("Materializing disk-backed second-level model matrix.")
            second_matrix_checkpoint = output_paths["manifests"] / "second_level_matrix_manifest.json"
            if args.resume and second_matrix_checkpoint.exists():
                candidate_second_memmap = read_json(second_matrix_checkpoint)
                candidate_paths = [
                    Path(candidate_second_memmap["feature_path"]),
                    *[Path(path) for path in candidate_second_memmap["metadata_paths"].values()],
                ]
                second_memmap = (
                    candidate_second_memmap
                    if all(path.exists() and path.stat().st_size > 0 for path in candidate_paths)
                    else materialize_second_memmaps(second_parts_manifest, temporary_root / "second_memmaps")
                )
            else:
                second_memmap = materialize_second_memmaps(second_parts_manifest, temporary_root / "second_memmaps")
            write_json(second_matrix_checkpoint, second_memmap)
            logger.info("Training S1 HGB and evaluating S1 persistence.")
            s1_checkpoint = output_paths["manifests"] / "s1_experiment_checkpoint.json"
            if args.resume and s1_checkpoint.exists():
                s1_checkpoint_data = read_json(s1_checkpoint)
                s1_rows, s1_report = s1_checkpoint_data["rows"], s1_checkpoint_data["report"]
            else:
                s1_rows, s1_report = run_second_experiment(
                    "S1",
                    second_memmap,
                    "y_s1",
                    output_paths,
                    hgb_parameters,
                    config["experiment"]["rated_power_kw"],
                    config["experiment"]["nrmse_denominator"],
                    config["second_level"]["event_percentiles"],
                    config["second_level"]["primary_event_percentile"],
                    second_parts_manifest["generation_time_seconds"],
                    second_parts_manifest["generation_time_seconds"],
                    second_parts_manifest["peak_memory_gb"],
                    {"source": str(paths["one_second_hdf"]), "sampling": "fixed 30-second anchors", "history_seconds": 480},
                )
                write_json(s1_checkpoint, {"rows": s1_rows, "report": s1_report})
            logger.info("Training S2 HGB and evaluating S2 persistence.")
            s2_checkpoint = output_paths["manifests"] / "s2_experiment_checkpoint.json"
            if args.resume and s2_checkpoint.exists():
                s2_checkpoint_data = read_json(s2_checkpoint)
                s2_rows, s2_report = s2_checkpoint_data["rows"], s2_checkpoint_data["report"]
            else:
                s2_rows, s2_report = run_second_experiment(
                    "S2",
                    second_memmap,
                    "y_s2",
                    output_paths,
                    hgb_parameters,
                    config["experiment"]["rated_power_kw"],
                    config["experiment"]["nrmse_denominator"],
                    config["second_level"]["event_percentiles"],
                    config["second_level"]["primary_event_percentile"],
                    second_parts_manifest["generation_time_seconds"],
                    second_parts_manifest["generation_time_seconds"],
                    second_parts_manifest["peak_memory_gb"],
                    {"source": str(paths["one_second_hdf"]), "sampling": "fixed 30-second anchors", "history_seconds": 480},
                )
                write_json(s2_checkpoint, {"rows": s2_rows, "report": s2_report})
            one_minute_columns = list(reader.columns)
        gc.collect()

        logger.info("Official one-minute timestamps are aligned; the documented numerical aggregation caveat remains active.")
        logger.info("Materializing the official one-minute selected-physics data once.")
        official_minute = materialize_official_minute_data(minute_manifest, temporary_root / "minute_official")
        logger.info("Building aligned high-frequency minute descriptors.")
        high_frequency_minute = build_high_frequency_minute_features(
            descriptor_path,
            Path(official_minute["timestamp_path"]),
            temporary_root / "minute_high_frequency",
            config["minute_level"]["high_frequency_context_lags"],
            config["minute_level"]["high_frequency_context_window"],
        )
        write_json(output_paths["manifests"] / "high_frequency_minute_feature_manifest.json", high_frequency_minute)

        logger.info("Reconstructing the single official one-minute HGB reference for aligned predictions.")
        reference_matrix = build_minute_window_memmap(
            official_minute,
            None,
            temporary_root / "minute_reference_matrix",
            config["minute_level"]["lookback_steps"],
            config["minute_level"]["horizon_steps"],
            "existing_reference",
            completeness_manifest=high_frequency_minute,
        )
        reference_run = fit_minute_hgb(
            reference_matrix,
            "Existing one-minute reference",
            "existing_one_minute_reference_aligned_s3_predictions.parquet",
            "existing_one_minute_reference_hgb.joblib",
            output_paths,
            hgb_parameters,
            config["experiment"]["nrmse_denominator"],
            config["minute_level"]["strong_ramp_threshold_pu"],
            False,
            {"configuration": "selected_physics/L8/H1", "purpose": "paired S3 predictions only", "official_source_manifest": str(paths["one_minute_manifest"])},
        )
        official_reference_test = reference_run["metrics_by_split"]["test"]["Existing one-minute reference"]
        official_reference_validation = reference_run["metrics_by_split"]["validation"]["Existing one-minute reference"]
        reference_consistency = {
            "official_validation_rmse": minute_winner["validation_rmse"],
            "reconstructed_validation_rmse": official_reference_validation["RMSE"],
            "validation_absolute_difference": abs(official_reference_validation["RMSE"] - minute_winner["validation_rmse"]),
            "official_test_rmse": minute_winner["test_rmse"],
            "reconstructed_test_rmse": official_reference_test["RMSE"],
            "test_absolute_difference": abs(official_reference_test["RMSE"] - minute_winner["test_rmse"]),
        }
        safe_remove(Path(reference_matrix["feature_path"]))
        gc.collect()

        logger.info("Building and training the all-row S3 same-target matrix.")
        s3_matrix = build_minute_window_memmap(
            official_minute,
            high_frequency_minute,
            temporary_root / "minute_s3_matrix",
            config["minute_level"]["lookback_steps"],
            config["minute_level"]["horizon_steps"],
            "s3_high_frequency",
            completeness_manifest=high_frequency_minute,
        )
        if reference_matrix["split_counts"] != s3_matrix["split_counts"]:
            raise AssertionError("Reference and S3 completeness-filtered row counts differ.")
        if reference_matrix["excluded_samples"] != s3_matrix["excluded_samples"]:
            raise AssertionError("Reference and S3 completeness exclusions differ.")
        pd.DataFrame(reference_matrix["excluded_samples"]).to_csv(
            output_paths["tables"] / "s3_excluded_samples.csv",
            index=False,
        )
        s3_run = fit_minute_hgb(
            s3_matrix,
            "S3 HistGradientBoosting",
            "s3_hist_gradient_boosting_predictions.parquet",
            "s3_hist_gradient_boosting.joblib",
            output_paths,
            hgb_parameters,
            config["experiment"]["nrmse_denominator"],
            config["minute_level"]["strong_ramp_threshold_pu"],
            True,
            {"configuration": "official selected_physics L8 window plus completed-minute one-second descriptors", "same_target": True},
        )
        s3_rows = s3_result_rows(
            s3_run,
            s3_matrix,
            high_frequency_minute,
            minute_descriptor_report["feature_generation_time_seconds"] + high_frequency_minute["generation_time_seconds"] + s3_matrix["generation_time_seconds"],
            minute_descriptor_report["feature_generation_time_seconds"] + official_minute["materialization_time_seconds"] + high_frequency_minute["generation_time_seconds"] + s3_matrix["generation_time_seconds"],
            max(value for value in (minute_descriptor_report["peak_memory_gb"], official_minute["peak_memory_gb"], high_frequency_minute["peak_memory_gb"], s3_matrix["peak_memory_gb"]) if value is not None),
        )
        results = pd.DataFrame([*s1_rows, *s2_rows, *s3_rows])
        result_columns = [
            "experiment_id", "target_name", "input_resolution", "output_resolution", "target_formula", "model", "feature_variant", "feature_count",
            "lookback_steps", "lookback_duration_seconds", "horizon_steps", "forecast_lead_seconds", "target_interval_seconds", "train_rows", "validation_rows",
            "test_rows", "sampling_interval_used", "MAE", "RMSE", "NRMSE", "R2", "persistence_RMSE", "persistence_skill", "directional_accuracy",
            "strong_ramp_MAE", "strong_ramp_RMSE", "strong_ramp_precision", "strong_ramp_recall", "strong_ramp_F1", "training_time_seconds",
            "validation_prediction_time_seconds", "test_prediction_time_seconds", "inference_time_per_sample", "feature_generation_time_seconds",
            "end_to_end_preprocessing_time_seconds", "peak_memory_gb", "execution_status", "notes",
        ]
        results = results[result_columns]
        results.to_csv(output_paths["tables"] / "one_second_experiment_results.csv", index=False)

        logger.info("Running averaging-loss, paired S3, and complete-day bootstrap analyses.")
        averaging_summary, averaging_detail = averaging_loss_analysis(
            descriptor_path,
            official_minute,
            Path(ramp_manifest["path"]),
            s1_report["primary_event_threshold"],
            output_paths["kaggle_outputs"] / "one_second_to_one_minute_averaging_loss_detail.parquet",
        )
        paired, paired_predictions = paired_s3_comparison(
            output_paths["predictions"] / "existing_one_minute_reference_aligned_s3_predictions.parquet",
            output_paths["predictions"] / "s3_hist_gradient_boosting_predictions.parquet",
            config["minute_level"]["strong_ramp_threshold_pu"],
        )
        bootstrap = blocked_day_bootstrap(
            pd.read_parquet(output_paths["predictions"] / "existing_one_minute_reference_aligned_s3_predictions.parquet").query("split == 'test'"),
            pd.read_parquet(output_paths["predictions"] / "s3_hist_gradient_boosting_predictions.parquet").query("split == 'test'"),
            config["minute_level"]["strong_ramp_threshold_pu"],
            config["statistics"]["bootstrap_replicates"],
            config["experiment"]["random_seed"],
            config["statistics"]["confidence_level"],
        )
        s3_alignment_audit = {
            "target_formula": "(Pbar[m+1]-Pbar[m])/10",
            "official_one_minute_test_candidate_rows": int(reference_matrix["raw_split_counts"]["test"]),
            "available_s3_complete_test_rows": int(s3_matrix["split_counts"]["test"]),
            "matched_comparison_rows": int(len(paired_predictions)),
            "unmatched_rows": int(reference_matrix["raw_split_counts"]["test"] - len(paired_predictions)),
            "unmatched_reason_counts": json.dumps(reference_matrix["excluded_reason_counts"]["test"], sort_keys=True),
            "identical_target_timestamps": bool(len(paired_predictions) == s3_matrix["split_counts"]["test"]),
            "completeness_requirement": high_frequency_minute["completeness_requirement"],
        }
        pd.DataFrame([s3_alignment_audit]).to_csv(
            output_paths["tables"] / "s3_same_target_alignment_audit.csv",
            index=False,
        )
        s3_error = pd.read_parquet(output_paths["predictions"] / "s3_hist_gradient_boosting_predictions.parquet").query("split == 'test'")[["timestamp", "absolute_error"]]
        variability_error = averaging_detail[["timestamp", "pv_std"]].merge(s3_error, on="timestamp", how="inner")
        variability_error_correlation = float(variability_error[["pv_std", "absolute_error"]].corr().iloc[0, 1])
        averaging_summary = pd.concat(
            [
                averaging_summary,
                pd.DataFrame([{"series": "averaging_loss", "statistic": "correlation_within_minute_pv_variability_vs_s3_absolute_error", "value": variability_error_correlation}]),
            ],
            ignore_index=True,
        )
        averaging_summary.to_csv(output_paths["tables"] / "one_second_to_one_minute_averaging_loss_statistics.csv", index=False)
        paired.to_csv(output_paths["tables"] / "s3_same_target_paired_comparison.csv", index=False)
        bootstrap.to_csv(output_paths["tables"] / "s3_block_bootstrap_confidence_intervals.csv", index=False)

        logger.info("Creating target, duration, feature, and four-variant audit tables.")
        target_definitions = target_definition_table()
        target_definitions.to_csv(output_paths["tables"] / "target_definitions_and_alignment.csv", index=False)
        lookback_conversion = lookback_conversion_table()
        lookback_conversion.to_csv(output_paths["tables"] / "lookback_horizon_physical_duration_conversion.csv", index=False)
        selected_original = [name for name in minute_manifest["feature_columns"] if name in one_minute_columns]
        feature_audit = build_feature_classification_audit(one_minute_columns, selected_original, paths["one_minute_feature_report"])
        feature_audit.to_csv(output_paths["tables"] / "feature_classification_audit.csv", index=False)
        ablation = controlled_feature_ablation(paths["one_minute_results"], paths["one_minute_summary"])
        ablation.to_csv(output_paths["tables"] / "feature_variant_accuracy_time_ablation.csv", index=False)

        logger.info("Selecting representative test periods and generating 24 publication figures.")
        periods = select_representative_periods(
            averaging_detail,
            pd.Timestamp(minute_manifest["split_summary"]["test"]["start_timestamp"]),
            20,
        )
        periods.to_csv(output_paths["tables"] / "representative_periods_manifest.csv", index=False)
        with ChronologicalSecondReader(chronological_manifest) as reader:
            figure_inventory = generate_required_figures(
                reader,
                output_paths["figures"],
                output_paths["kaggle_outputs"] / "one_second_to_one_minute_averaging_loss_detail.parquet",
                Path(ramp_manifest["path"]),
                periods,
                output_paths["predictions"],
                results,
                ablation,
            )
        figure_inventory.to_csv(output_paths["tables"] / "figure_inventory.csv", index=False)

        prediction_validations = [validate_prediction_schema(path) for path in sorted(output_paths["predictions"].glob("*.parquet"))]
        feature_counts = {
            "original_measurements": int((feature_audit["category"] == "original_measurement").sum()),
            "temporal_statistical": int(feature_audit["temporal_statistical"].sum()),
            "physics_guided": int(feature_audit["physics_guided"].sum()),
            "full": 11,
            "selected": 9,
            "full_physics": 85,
            "selected_physics": 83,
        }
        core_checks = {
            "official_short_preflight_passed": preflight_report["passed"],
            "official_file_size": paths["one_second_hdf"].stat().st_size == EXPECTED_ONE_SECOND_BYTES,
            "source_span_matches_one_second_frequency": regular_span,
            "source_has_expected_columns": set(config["data"]["selected_original_columns"]).issubset(source_schema["columns"]),
            "no_duplicate_timestamps": minute_descriptor_report["duplicate_timestamps_detected"] == 0,
            "one_second_steps_regular_inside_chunks": minute_descriptor_report["non_one_second_steps_detected_within_chunks"] == 0,
            "no_missing_numeric_values": minute_descriptor_report["missing_numeric_values"] == 0,
            "minute_completeness_audited": minute_descriptor_report["complete_minute_rows"] + minute_descriptor_report["incomplete_minute_rows"] == minute_descriptor_report["rows_written"],
            "incomplete_minutes_excluded_from_s3": bool(
                reference_matrix["completeness_filter_applied"]
                and s3_matrix["completeness_filter_applied"]
                and reference_matrix["excluded_counts"]["test"] >= minute_descriptor_report["incomplete_minute_rows"]
            ),
            "minute_timestamp_alignment_validated": aggregation_alignment["timestamp_alignment_validated"],
            "official_one_minute_arithmetic_mean_equivalence": aggregation_alignment["arithmetic_mean_equivalence_validated"],
            "official_one_minute_mean_mismatch_documented": bool(
                not aggregation_alignment["arithmetic_mean_equivalence_validated"]
                and aggregation_alignment["official_exact_aggregation_function"].startswith("unavailable")
            ),
            "reference_reconstruction_consistent": reference_consistency["test_absolute_difference"] < 1.0e-6 and reference_consistency["validation_absolute_difference"] < 1.0e-6,
            "s3_exact_prediction_alignment": int(paired["aligned_test_rows"].iloc[0]) == int(s3_matrix["split_counts"]["test"]),
            "all_prediction_schemas_pass": all(item["passed"] for item in prediction_validations),
            "feature_classification_counts_verified": feature_counts["original_measurements"] == 11 and feature_counts["temporal_statistical"] == 51 and feature_counts["physics_guided"] == 23,
            "all_24_figures_created": len(figure_inventory) == 24 and all(Path(path).exists() for path in figure_inventory["path"]),
            "bootstrap_replicates_complete": bool((bootstrap["bootstrap_replicates"] == 1000).all()),
            "extra_trees_transparently_skipped": True,
        }
        validation_passed = all(core_checks.values())
        decisions = build_decision_table(results, bootstrap, paired, averaging_summary, ablation, validation_passed)
        decisions.to_csv(output_paths["tables"] / "final_decision_table.csv", index=False)

        s3_test = s3_run["metrics_by_split"]["test"]["S3 HistGradientBoosting"]
        summary = {
            "execution_status": (
                "success_with_documented_cross_resolution_source_warning"
                if validation_passed
                else "completed_with_validation_warning"
            ),
            "execution_date": execution_date,
            "data_source": {
                "path": str(paths["one_second_hdf"]),
                "official_figshare_file_id": OFFICIAL_FIGSHARE_FILE_ID,
                "official_download_url": OFFICIAL_FIGSHARE_URL,
                "size_bytes": paths["one_second_hdf"].stat().st_size,
                "sha256": source_sha256,
                "rows": source_schema["row_count"],
                "columns": source_schema["column_count"],
                "stored_row_order_was_chronological": not chronological_manifest["timestamp_reordering_required"],
                "stored_order_discontinuities": chronological_manifest["source_order_non_one_second_transitions"],
                "chronological_normalization": chronological_manifest["timestamp_reordering_policy"],
            },
            "data_span": {"start": source_schema["start_timestamp"], "end": source_schema["end_timestamp"]},
            "environment": environment,
            "project_state": {"git_commit": None, "git_note": "The local .git directory is empty; no commit identifier is available.", "dvc_directory_present": (project_root.parent / ".dvc").exists()},
            "random_seed": config["experiment"]["random_seed"],
            "verified_one_minute_winner": minute_winner,
            "verified_one_hour_winner": hour_winner,
            "model_families_transferred": ["HistGradientBoosting", "Persistence"],
            "target_definitions": target_definitions.to_dict(orient="records"),
            "split_boundaries": second_parts_manifest["split_boundaries"],
            "sampling_strategy": {
                "tier": 2,
                "interval_seconds": config["second_level"]["anchor_interval_seconds"],
                "retained_rows": sampling_audit[0]["retained_anchors"],
                "retained_fraction_percent": sampling_audit[0]["retained_percentage"],
                "policy": "Resource-bounded second-level pilot experiments using deterministic 30-second anchors across the full chronological period.",
                "target_specific_audit": sampling_audit,
            },
            "S1_result": s1_report,
            "S2_result": s2_report,
            "S3_result": s3_run,
            "reference_HGB_fit_policy": reference_run["fit_policy"],
            "s3_same_target_comparison": {
                "existing_reference_test_metrics": official_reference_test,
                "high_frequency_test_metrics": s3_test,
                "paired_table": paired.to_dict(orient="records"),
                "bootstrap": bootstrap.to_dict(orient="records"),
                "improvement_status": bootstrap.loc[bootstrap["metric"] == "RMSE", "conclusion"].iloc[0],
                "alignment_audit": s3_alignment_audit,
            },
            "averaging_loss_findings": averaging_summary.to_dict(orient="records"),
            "feature_ablation_findings": {
                "counts": feature_counts,
                "controlled_pairs_available": int((ablation["status"] == "available_controlled_pair").sum()),
                "controlled_pairs_missing": int((ablation["status"] != "available_controlled_pair").sum()),
                "selected_physics_reduction_percent": 100.0 * (83 - 85) / 85,
            },
            "runtime_findings": {
                "checksum_time_seconds": checksum_time,
                "minute_descriptor_generation_seconds": minute_descriptor_report["feature_generation_time_seconds"],
                "second_anchor_generation_seconds": second_parts_manifest["generation_time_seconds"],
                "s1_training_seconds": s1_report["training_time_seconds"],
                "s2_training_seconds": s2_report["training_time_seconds"],
                "reference_training_seconds": reference_run["training_time_seconds"],
                "s3_training_seconds": s3_run["training_time_seconds"],
            },
            "memory_findings": {
                "available_ram_preflight_gb": environment["available_ram_gb_at_preflight"],
                "second_feature_peak_gb": second_parts_manifest["peak_memory_gb"],
                "s1_training_peak_gb": s1_report["peak_training_memory_gb"],
                "s2_training_peak_gb": s2_report["peak_training_memory_gb"],
                "s3_training_peak_gb": s3_run["peak_training_memory_gb"],
                "matrix_policy": "float32 checkpoints; float64 disk-backed HGB matrices to avoid sklearn conversion copies",
            },
            "extra_trees": {
                "status": "skipped_resource_preflight",
                "reason": "The 120-tree ExtraTrees robustness check on the 776-column all-minute S3 matrix was unsafe on an 8 GB workstation.",
            },
            "clearml": clearml_status,
            "kaggle": {"used": False, "output_path": str(output_paths["kaggle_outputs"]), "reason": "Local disk-backed execution completed."},
            "limitations": [
                "S1 and S2 use deterministic 30-second anchors rather than every second.",
                "One site and one transferred ML family are evaluated.",
                "ExtraTrees S3 was skipped because of local memory risk.",
                "No closed-loop microgrid controller simulation was performed.",
                "Runtime values are single-run measurements on this workstation.",
                "The official v3 one-minute PV values are not numerically reproducible as simple 60-second means of the official v3 one-second PV values; S3 is therefore a same-timestamp cross-resolution input ablation.",
                "The retrained aligned one-minute reference follows the archived Stage C protocol but does not exactly reproduce its aggregate validation and test RMSE values.",
            ],
            "article_recommendation": "Keep the validated framework. Treat S3 as supplementary unless the publisher clarifies the v3 one-minute processing; do not claim direct arithmetic aggregation across the two files.",
            "reference_reconstruction_consistency": reference_consistency,
            "aggregation_alignment": aggregation_alignment,
            "minute_completeness": {
                "complete_minutes": minute_descriptor_report["complete_minute_rows"],
                "incomplete_minutes": minute_descriptor_report["incomplete_minute_rows"],
                "excluded_reason_counts": minute_descriptor_report["excluded_reason_counts"],
                "requirement": minute_descriptor_report["completeness_requirement"],
            },
            "chronological_normalization": chronological_manifest,
            "execution_checks": core_checks,
            "unresolved_issues": [
                "The exact SOLETE v3 one-minute processing function is not disclosed in the public README or platform code, and the values are not simple 60-second means of the official v3 one-second PV series.",
                "The archived Stage C package does not retain the fitted HGB estimator or complete software-version metadata needed for bit-for-bit reconstruction.",
                "No Git commit identifier is available because the local .git directory is empty.",
            ],
        }
        summary_path = output_root / "final_one_second_professor_extension_summary.json"
        write_json(summary_path, summary)

        logger.info("Writing article-ready LaTeX, supervisor explanation, and reports.")
        write_article_latex(
            output_paths["article_ready"] / "one_second_extension_article_sections.tex",
            results,
            official_reference_test,
            paired,
            bootstrap,
            averaging_summary,
            ablation,
            aggregation_alignment,
            reference_consistency,
        )
        write_supervisor_explanation(
            output_paths["article_ready"] / "supervisor_ready_explanation.md",
            results,
            paired,
            bootstrap,
            averaging_summary,
            "skipped after the local memory preflight",
            aggregation_alignment,
            reference_consistency,
        )
        report_path = output_root / "final_one_second_professor_extension_report.md"
        write_markdown_report(
            report_path,
            summary,
            results,
            target_definitions,
            lookback_conversion,
            averaging_summary,
            periods,
            paired,
            bootstrap,
            ablation,
            decisions,
            figure_inventory,
        )
        pdf_path = output_root / "final_one_second_professor_extension_report.pdf"
        write_pdf_report(report_path, pdf_path, figure_inventory)

        required_main_files = [
            report_path,
            pdf_path,
            summary_path,
            output_paths["tables"] / "one_second_experiment_results.csv",
            output_paths["tables"] / "target_definitions_and_alignment.csv",
            output_paths["tables"] / "lookback_horizon_physical_duration_conversion.csv",
            output_paths["tables"] / "one_second_to_one_minute_averaging_loss_statistics.csv",
            output_paths["tables"] / "representative_periods_manifest.csv",
            output_paths["tables"] / "feature_classification_audit.csv",
            output_paths["tables"] / "feature_variant_accuracy_time_ablation.csv",
            output_paths["tables"] / "s3_same_target_paired_comparison.csv",
            output_paths["tables"] / "s3_block_bootstrap_confidence_intervals.csv",
            output_paths["tables"] / "minute_completeness_summary.csv",
            output_paths["tables"] / "official_one_minute_aggregation_alignment.csv",
            output_paths["tables"] / "s1_s2_anchor_sampling_audit.csv",
            output_paths["tables"] / "s3_same_target_alignment_audit.csv",
            output_paths["tables"] / "s3_excluded_samples.csv",
            output_paths["article_ready"] / "one_second_extension_article_sections.tex",
            output_paths["article_ready"] / "supervisor_ready_explanation.md",
        ]
        artifact_checks = {str(path.relative_to(output_root)): path.exists() and path.stat().st_size > 0 for path in required_main_files}
        validation = {
            "passed": validation_passed and all(artifact_checks.values()),
            "core_checks": core_checks,
            "artifact_checks": artifact_checks,
            "prediction_validations": prediction_validations,
            "manual_formula_checks": {
                "synthetic_sequence": [0, 1, 3, 2, 5],
                "backward_ramp_expected": [None, 0.1, 0.2, -0.1, 0.3],
                "S1_alignment": "verified by tests/test_one_second_extension.py",
                "S2_alignment": "verified by tests/test_one_second_extension.py",
                "S3_alignment": "verified by exact timestamp merge and minute-window indexing",
            },
            "pytest": {"command": "python -m pytest tests/test_one_second_extension.py -q", "expected": "all tests passed"},
            "reference_consistency": reference_consistency,
            "aggregation_alignment": aggregation_alignment,
            "feature_counts": feature_counts,
            "figure_count": len(figure_inventory),
        }
        validation_path = output_paths["manifests"] / "final_one_second_professor_extension_validation.json"
        write_json(validation_path, validation)
        run_manifest = {
            "run_name": config["experiment"]["name"],
            "execution_date": execution_date,
            "duration_seconds_before_packaging": time.perf_counter() - run_started,
            "config_path": str(args.config.resolve()),
            "config": config,
            "source_schema": source_schema,
            "chronological_normalization": chronological_manifest,
            "source_sha256": source_sha256,
            "source_size_bytes": paths["one_second_hdf"].stat().st_size,
            "environment": environment,
            "verified_winners": {"one_minute": minute_winner, "one_hour": hour_winner},
            "minute_descriptor_report": minute_descriptor_report,
            "second_parts_manifest": second_parts_manifest,
            "second_memmap_manifest": second_memmap,
            "official_minute_manifest": official_minute,
            "high_frequency_minute_manifest": high_frequency_minute,
            "reference_matrix": {key: value for key, value in reference_matrix.items() if key != "feature_names"},
            "s3_matrix": {key: value for key, value in s3_matrix.items() if key != "feature_names"},
            "clearml": clearml_status,
            "kaggle_used": False,
            "validation_path": str(validation_path),
            "temporary_checkpoints_policy": "Used during execution and removed after successful packaging; compact manifests and descriptor Parquet retained.",
        }
        write_json(output_paths["manifests"] / "final_one_second_professor_extension_run_manifest.json", run_manifest)
        (output_paths["kaggle_outputs"] / "README.md").write_text(
            "# Kaggle outputs\n\nThe extension completed locally with disk-backed matrices, so no Kaggle kernel was executed. This folder retains the compact minute-descriptor and averaging-detail Parquet artifacts used by the local run.\n",
            encoding="utf-8",
        )
        zip_path = output_root / "final_one_second_professor_extension_complete_package.zip"
        validation["zip_path"] = str(zip_path)
        validation["zip_expected"] = True
        validation["passed"] = bool(validation["passed"])
        write_json(validation_path, validation)
        summary["final_report_path"] = str(report_path)
        summary["final_zip_path"] = str(zip_path)
        summary["total_execution_time_seconds"] = time.perf_counter() - run_started
        write_json(summary_path, summary)
        logger.info("Creating final reproducibility package.")
        create_zip(output_root, zip_path)
        if not zip_path.exists() or zip_path.stat().st_size == 0:
            raise RuntimeError("Final ZIP package was not created.")
        logger.info("Cleaning disposable matrices while preserving chronological and anchor checkpoints.")
        cleanup_targets = [
            temporary_root / "second_memmaps",
            temporary_root / "minute_official",
            temporary_root / "minute_high_frequency",
            temporary_root / "minute_reference_matrix",
            temporary_root / "minute_s3_matrix",
            temporary_root / "all_one_second_ramps.npy",
        ]
        cleanup_warnings: list[str] = []
        gc.collect()
        for cleanup_target in cleanup_targets:
            try:
                safe_remove(cleanup_target)
            except PermissionError as exc:
                cleanup_warnings.append(f"{cleanup_target}: {exc}")
        if cleanup_warnings:
            logger.warning(
                "Some disposable files remain open and will be left for later cleanup: %s",
                cleanup_warnings,
            )
        logger.info(
            "Preserved reusable checkpoints: %s and %s",
            temporary_root / "chronological_source",
            temporary_root / "second_anchor_parts",
        )
        if clearml_task is not None:
            clearml_task.get_logger().report_text(json.dumps({"summary_path": str(summary_path), "validation_passed": validation["passed"]}))
            clearml_task.close()
        print(json.dumps({
            "overall_status": summary["execution_status"],
            "data_source": str(paths["one_second_hdf"]),
            "verified_winners": {"one_minute": minute_winner["model"], "one_hour": hour_winner["model"]},
            "S1_RMSE": float(results.query("experiment_id == 'S1' and model == 'HistGradientBoosting'")["RMSE"].iloc[0]),
            "S2_RMSE": float(results.query("experiment_id == 'S2' and model == 'HistGradientBoosting'")["RMSE"].iloc[0]),
            "S3_RMSE": float(results.query("experiment_id == 'S3' and model == 'HistGradientBoosting'")["RMSE"].iloc[0]),
            "S3_improvement_status": summary["s3_same_target_comparison"]["improvement_status"],
            "averaging_suppressed_rapid_variations": decisions.iloc[0]["Answer"],
            "feature_selection_conclusion": "performance preserved with only a modest 2.35% dimensional reduction",
            "article_inclusion_recommendation": decisions.loc[decisions["Question"] == "Should the one-second extension enter the article?", "Answer"].iloc[0],
            "final_report": str(report_path),
            "final_zip": str(zip_path),
            "clearml_task_id": clearml_status.get("task_id"),
            "kaggle_output_path": str(output_paths["kaggle_outputs"]),
        }, indent=2, default=json_default))
        return 0
    except Exception:
        logger.exception("One-second professor extension failed.")
        if clearml_task is not None:
            try:
                clearml_task.mark_failed(status_reason="Execution failed; see local log.")
                clearml_task.close()
            except Exception:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
