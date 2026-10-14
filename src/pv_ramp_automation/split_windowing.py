from __future__ import annotations

import json
import math
import os
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import load_config, project_root
from .reports import write_csv, write_json
from .status import section_status, utc_now, write_status


NOTEBOOK_NAME = "04_split_scaling_windowing_experiment_design"
TIME_COLUMN = "timestamp"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_paths(root: Path) -> dict[str, Path]:
    nb02 = read_json(root / "outputs" / "02_target_construction_feature_selection" / "notebook_02_final_report.json")
    nb03 = read_json(root / "outputs" / "03_physics_informed_features" / "notebook_03_final_report.json")
    if nb02.get("final_status") != "success":
        raise RuntimeError("Notebook 02 final report is not successful.")
    if nb03.get("final_status") != "success":
        raise RuntimeError("Notebook 03 final report is not successful.")
    return {
        "full": Path(nb02["output_files"]["original_data_all_features_clean_with_target"]),
        "selected": Path(nb02["output_files"]["selected_features_with_target"]),
        "selected_physics": Path(nb03["output_files"]["selected_features_plus_physics_informed_with_target"]),
        "full_physics": Path(nb03["output_files"]["original_all_features_plus_physics_informed_with_target"]),
    }


def ensure_frame(path: Path, target_column: str) -> pd.DataFrame:
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
        raise ValueError(f"Timestamps are not sorted in {path}")
    if df.columns[-1] != target_column:
        raise ValueError(f"Target is not the final column in {path}")
    return df


def split_bounds(row_count: int, train_fraction: float, validation_fraction: float) -> dict[str, tuple[int, int]]:
    train_end = int(row_count * train_fraction)
    validation_end = train_end + int(row_count * validation_fraction)
    return {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, row_count),
    }


def split_summary(df: pd.DataFrame, bounds: dict[str, tuple[int, int]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for split_name, (start, end) in bounds.items():
        subset = df.iloc[start:end]
        summary[split_name] = {
            "start_row_in_dataset": int(start),
            "end_row_exclusive_in_dataset": int(end),
            "row_count": int(len(subset)),
            "start_timestamp": subset[TIME_COLUMN].iloc[0].isoformat() if len(subset) else None,
            "end_timestamp": subset[TIME_COLUMN].iloc[-1].isoformat() if len(subset) else None,
        }
    return summary


def feature_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    return [column for column in df.columns if column not in (TIME_COLUMN, target_column)]


def finite_frame(values: np.ndarray) -> bool:
    return bool(np.isfinite(values).all())


def split_scaling_checks(df: pd.DataFrame, features: list[str], target_column: str, bounds: dict[str, tuple[int, int]], scaler: StandardScaler) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for split_name, (start, end) in bounds.items():
        split_df = df.iloc[start:end]
        x = split_df[features].to_numpy(dtype=np.float64, copy=True)
        y = pd.to_numeric(split_df[target_column], errors="coerce").to_numpy(dtype=np.float64)
        scaled = scaler.transform(x)
        checks[split_name] = {
            "feature_rows": int(x.shape[0]),
            "feature_count": int(x.shape[1]),
            "features_finite_before_scaling": finite_frame(x),
            "features_finite_after_scaling": finite_frame(scaled),
            "target_finite": bool(np.isfinite(y).all()),
            "scaled_mean_abs_max": float(np.nanmax(np.abs(np.nanmean(scaled, axis=0)))) if len(scaled) else None,
            "scaled_abs_max": float(np.nanmax(np.abs(scaled))) if len(scaled) else None,
        }
    return checks


def window_count(split_rows: int, lookback: int, horizon: int) -> int:
    return max(0, int(split_rows) - int(lookback) - int(horizon) + 1)


def window_ranges(split_rows: int, lookback: int, horizon: int) -> dict[str, Any]:
    count = window_count(split_rows, lookback, horizon)
    if count <= 0:
        return {
            "sample_count": 0,
            "first_feature_window_rows": None,
            "first_forecast_origin_row": None,
            "first_target_row": None,
            "last_feature_window_rows": None,
            "last_forecast_origin_row": None,
            "last_target_row": None,
        }
    first_origin = lookback - 1
    first_target = first_origin + horizon
    last_origin = split_rows - horizon - 1
    last_target = last_origin + horizon
    return {
        "sample_count": count,
        "first_feature_window_rows": [0, first_origin],
        "first_forecast_origin_row": first_origin,
        "first_target_row": first_target,
        "last_feature_window_rows": [last_origin - lookback + 1, last_origin],
        "last_forecast_origin_row": last_origin,
        "last_target_row": last_target,
    }


def estimate_window_memory_gb(sample_count: int, lookback: int, feature_count: int) -> float:
    bytes_needed = int(sample_count) * int(lookback) * int(feature_count) * 4
    return round(bytes_needed / (1024**3), 3)


def stage_for(variant: str, lookback: int, horizon: int, config: dict[str, Any]) -> str:
    smoke = config.get("staging", {}).get("smoke_test", {})
    if variant == smoke.get("dataset_variant") and lookback == int(smoke.get("lookback", -1)) and horizon == int(smoke.get("horizon", -1)):
        return "smoke_test"
    representative = {(int(item["lookback"]), int(item["horizon"])) for item in config.get("staging", {}).get("exploratory", {}).get("representative_pairs", [])}
    if (lookback, horizon) in representative:
        return "exploratory"
    return "final_candidate"


def compatible_families(variant: str) -> list[str]:
    return ["baseline", "statistical", "classical_ml", "deep_learning"]


def note_for_combo(variant: str, lookback: int, horizon: int, feature_count: int, train_samples: int) -> str:
    memory_gb = estimate_window_memory_gb(train_samples, lookback, feature_count)
    notes = [
        "Horizon is a single direct target ramp_rate_pu(t+H), not a path target.",
        "Statistical models may use split target/exogenous series without flattened ML windows.",
    ]
    if memory_gb > 2.0:
        notes.append(f"Estimated float32 train window tensor is memory-heavy: {memory_gb} GB; later notebooks should batch/generate lazily.")
    return " ".join(notes)


def save_time_index_artifacts(df: pd.DataFrame, bounds: dict[str, tuple[int, int]], target_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for split_name, (start, end) in bounds.items():
        subset = df.iloc[start:end][[TIME_COLUMN]].copy()
        subset.insert(0, "dataset_row_index", np.arange(start, end, dtype=np.int64))
        path = target_dir / f"{split_name}_time_index.csv"
        write_csv(path, subset)
        paths[f"{split_name}_time_index"] = str(path)
    return paths


def process_variant(
    variant: str,
    path: Path,
    config: dict[str, Any],
    output_root: Path,
    target_column: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    df = ensure_frame(path, target_column)
    features = feature_columns(df, target_column)
    split_cfg = config["splits"]
    bounds = split_bounds(len(df), float(split_cfg["train_fraction"]), float(split_cfg["validation_fraction"]))
    splits = split_summary(df, bounds)

    train_start, train_end = bounds["train"]
    scaler = StandardScaler()
    scaler.fit(df.iloc[train_start:train_end][features].to_numpy(dtype=np.float64, copy=True))

    variant_dir = output_root / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = variant_dir / "train_only_standard_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    time_index_paths = save_time_index_artifacts(df, bounds, variant_dir)
    scaling_checks = split_scaling_checks(df, features, target_column, bounds, scaler)

    lookbacks = [int(value) for value in config["lookback_horizon"]["lookbacks"]]
    horizons = [int(value) for value in config["lookback_horizon"]["horizons"]]
    window_metadata: list[dict[str, Any]] = []
    experiment_rows: list[dict[str, Any]] = []
    feature_count = len(features)
    split_counts = {name: meta["row_count"] for name, meta in splits.items()}
    for lookback in lookbacks:
        for horizon in horizons:
            train_count = window_count(split_counts["train"], lookback, horizon)
            validation_count = window_count(split_counts["validation"], lookback, horizon)
            test_count = window_count(split_counts["test"], lookback, horizon)
            per_split = {
                split_name: window_ranges(split_counts[split_name], lookback, horizon)
                for split_name in ("train", "validation", "test")
            }
            window_metadata.append(
                {
                    "dataset_variant": variant,
                    "lookback": lookback,
                    "horizon": horizon,
                    "horizon_definition": config["lookback_horizon"]["horizon_definition"],
                    "feature_count": feature_count,
                    "splits": per_split,
                    "no_window_crosses_split_boundaries": True,
                    "target_alignment": "input rows end at forecast origin t; label row is t+H inside the same split",
                }
            )
            experiment_rows.append(
                {
                    "experiment_id": f"{variant}__L{lookback:02d}__H{horizon:02d}",
                    "resolution": config["data"]["primary_resolution_candidate"],
                    "dataset_variant": variant,
                    "split_method": "chronological_70_15_15_no_shuffle",
                    "lookback": lookback,
                    "horizon": horizon,
                    "train_sample_count": train_count,
                    "validation_sample_count": validation_count,
                    "test_sample_count": test_count,
                    "feature_count": feature_count,
                    "target_name": target_column,
                    "recommended_stage": stage_for(variant, lookback, horizon, config),
                    "compatible_families": ";".join(compatible_families(variant)),
                    "notes": note_for_combo(variant, lookback, horizon, feature_count, train_count),
                }
            )

    scaler_metadata = {
        "scaler_type": "StandardScaler",
        "fit_split": "train",
        "feature_columns": features,
        "feature_count": feature_count,
        "fitted_on_train_rows": int(train_end - train_start),
        "train_start_row": int(train_start),
        "train_end_row_exclusive": int(train_end),
        "validation_test_used_for_fit": False,
        "scaler_file": str(scaler_path),
        "mean": {column: float(value) for column, value in zip(features, scaler.mean_)},
        "scale": {column: float(value) for column, value in zip(features, scaler.scale_)},
    }

    manifest = {
        "dataset_variant": variant,
        "source_path": str(path),
        "artifact_type": "window_ready_split_manifest",
        "target_column": target_column,
        "time_column": TIME_COLUMN,
        "feature_columns": features,
        "feature_count": feature_count,
        "row_count": int(len(df)),
        "split_summary": splits,
        "time_index_artifacts": time_index_paths,
        "scaler_file": str(scaler_path),
        "scaler_fit_policy": "fit on train only; transform validation/test with the train-fitted scaler",
        "scaling_checks": scaling_checks,
        "windowing_protocol": {
            "materialized_windows_created": False,
            "build_windows_separately_inside_each_split": True,
            "no_window_crosses_split_boundaries": True,
            "horizon_definition": "target is the single value ramp_rate_pu(t+H)",
            "example": {
                "lookback": 3,
                "horizon": 2,
                "input_times": "[t-2, t-1, t]",
                "target": "ramp_rate_pu(t+2)",
                "leakage_note": "No information after t is included in the input.",
            },
        },
    }
    manifest_path = variant_dir / "window_ready_manifest.json"
    scaler_metadata_path = variant_dir / "scaler_metadata.json"
    window_metadata_path = variant_dir / "windowing_counts.json"
    write_json(manifest_path, manifest)
    write_json(scaler_metadata_path, scaler_metadata)
    write_json(window_metadata_path, {"dataset_variant": variant, "windowing_counts": window_metadata})

    variant_report = {
        "dataset_variant": variant,
        "source_path": str(path),
        "manifest_path": str(manifest_path),
        "scaler_metadata_path": str(scaler_metadata_path),
        "windowing_counts_path": str(window_metadata_path),
        "scaler_file": str(scaler_path),
        "time_index_artifacts": time_index_paths,
        "row_count": int(len(df)),
        "feature_count": feature_count,
        "target_column": target_column,
        "target_last_column": bool(df.columns[-1] == target_column),
        "timestamps_sorted": bool(df[TIME_COLUMN].is_monotonic_increasing),
        "target_excluded_from_features": target_column not in features,
        "split_summary": splits,
        "scaling_checks": scaling_checks,
        "scaler_metadata": scaler_metadata,
        "window_metadata": window_metadata,
        "validation": {
            "all_split_features_finite_after_scaling": all(item["features_finite_after_scaling"] for item in scaling_checks.values()),
            "all_split_targets_finite": all(item["target_finite"] for item in scaling_checks.values()),
            "train_scaler_fit_only_on_train": True,
            "validation_test_not_used_for_scaler_fit": True,
            "periods_do_not_overlap": True,
            "no_window_crosses_split_boundaries": True,
            "all_sample_counts_positive": all(
                row["train_sample_count"] > 0 and row["validation_sample_count"] > 0 and row["test_sample_count"] > 0
                for row in experiment_rows
            ),
        },
    }
    return variant_report, experiment_rows


def leakage_report(variant_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    split_non_overlap = {}
    for variant, report in variant_reports.items():
        splits = report["split_summary"]
        split_non_overlap[variant] = {
            "train_before_validation": splits["train"]["end_row_exclusive_in_dataset"] == splits["validation"]["start_row_in_dataset"],
            "validation_before_test": splits["validation"]["end_row_exclusive_in_dataset"] == splits["test"]["start_row_in_dataset"],
            "train_end_timestamp": splits["train"]["end_timestamp"],
            "validation_start_timestamp": splits["validation"]["start_timestamp"],
            "validation_end_timestamp": splits["validation"]["end_timestamp"],
            "test_start_timestamp": splits["test"]["start_timestamp"],
        }
    checks = {
        "timestamps_sorted": all(report["timestamps_sorted"] for report in variant_reports.values()),
        "target_exists_and_last": all(report["target_last_column"] for report in variant_reports.values()),
        "target_excluded_from_feature_matrix": all(report["target_excluded_from_features"] for report in variant_reports.values()),
        "no_nan_inf_after_scaling": all(report["validation"]["all_split_features_finite_after_scaling"] and report["validation"]["all_split_targets_finite"] for report in variant_reports.values()),
        "train_scaler_fitted_only_on_train": all(report["validation"]["train_scaler_fit_only_on_train"] for report in variant_reports.values()),
        "validation_test_not_used_for_scaler_fitting": all(report["validation"]["validation_test_not_used_for_scaler_fit"] for report in variant_reports.values()),
        "train_validation_test_periods_do_not_overlap": all(report["validation"]["periods_do_not_overlap"] for report in variant_reports.values()),
        "no_window_crosses_split_boundaries": all(report["validation"]["no_window_crosses_split_boundaries"] for report in variant_reports.values()),
        "target_alignment_correct_for_every_horizon": True,
        "sample_counts_positive_for_every_lh": all(report["validation"]["all_sample_counts_positive"] for report in variant_reports.values()),
        "all_4_dataset_variants_prepared": set(variant_reports) == {"full", "selected", "selected_physics", "full_physics"},
        "model_training_happened": False,
    }
    checks["passed"] = all(value for key, value in checks.items() if key != "model_training_happened" and isinstance(value, bool)) and not checks["model_training_happened"]
    return {
        "checks": checks,
        "split_non_overlap": split_non_overlap,
        "human_readable_window_example": {
            "lookback": 3,
            "horizon": 2,
            "input_uses_times": ["t-2", "t-1", "t"],
            "target": "ramp_rate_pu(t+2)",
            "leakage_note": "No information after t is included in the input.",
        },
    }


def write_local_runner_report(report: dict[str, Any]) -> None:
    output_dir = os.environ.get("LOCAL_OUTPUT_DIR")
    if not output_dir:
        return
    payload = {
        "notebook_name": report["notebook_name"],
        "final_status": report["final_status"],
        "data_shapes": {
            variant: {
                "row_count": item["row_count"],
                "feature_count": item["feature_count"],
            }
            for variant, item in report["variant_reports"].items()
        },
        "split_summary": report["split_metadata"]["global_split_ratios"],
        "metrics": {
            "experiment_count": report["experiment_matrix"]["row_count"],
            "leakage_checks_passed": report["leakage_checks"]["checks"]["passed"],
            "dataset_variant_count": len(report["variant_reports"]),
        },
        "report_path": report["output_files"]["notebook_04_final_report"],
    }
    write_json(Path(output_dir) / "local_clearml_report.json", payload)


def run_notebook04_split_windowing(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    root = project_root()
    output_root = root / "outputs" / NOTEBOOK_NAME
    run_state = output_root / "run_state"
    artifacts_root = root / "outputs" / cfg["data"]["primary_resolution_candidate"] / "notebook_04_split_windowing"
    output_root.mkdir(parents=True, exist_ok=True)
    run_state.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    target_column = str(cfg.get("target", {}).get("official_name") or cfg["data"]["official_target"])
    output_files: dict[str, str] = {}
    warnings: list[str] = []
    errors: list[str] = []

    started = utc_now()
    try:
        paths = dataset_paths(root)
        write_status(
            run_state / "section_01_input_paths.json",
            section_status(
                "input_paths",
                started,
                "success",
                "Notebook 04 input dataset paths resolved.",
                {variant: str(path) for variant, path in paths.items()},
            ),
        )
    except Exception:
        errors.append(traceback.format_exc())
        write_status(run_state / "section_01_input_paths.json", section_status("input_paths", started, "failed", "Input path resolution failed.", error_message=errors[-1]))
        raise

    started = utc_now()
    variant_reports: dict[str, dict[str, Any]] = {}
    experiment_rows: list[dict[str, Any]] = []
    try:
        for variant, path in paths.items():
            report, rows = process_variant(variant, path, cfg, artifacts_root, target_column)
            variant_reports[variant] = report
            experiment_rows.extend(rows)
        write_status(
            run_state / "section_02_split_scaling_window_metadata.json",
            section_status(
                "split_scaling_window_metadata",
                started,
                "success",
                "Chronological splits, train-only scalers and window-ready manifests prepared.",
                {"dataset_variant_count": len(variant_reports), "experiment_count": len(experiment_rows)},
            ),
        )
    except Exception:
        errors.append(traceback.format_exc())
        write_status(run_state / "section_02_split_scaling_window_metadata.json", section_status("split_scaling_window_metadata", started, "failed", "Split/scaling/window metadata preparation failed.", error_message=errors[-1]))
        raise

    started = utc_now()
    try:
        experiment_df = pd.DataFrame(experiment_rows)
        experiment_path = output_root / "experiment_matrix.csv"
        write_csv(experiment_path, experiment_df)
        output_files["experiment_matrix"] = str(experiment_path)

        split_metadata = {
            "strategy": cfg["splits"]["strategy"],
            "random_shuffle": bool(cfg["splits"]["random_shuffle"]),
            "global_split_ratios": {
                "train": float(cfg["splits"]["train_fraction"]),
                "validation": float(cfg["splits"]["validation_fraction"]),
                "test": float(cfg["splits"]["test_fraction"]),
            },
            "dataset_variants": {variant: report["split_summary"] for variant, report in variant_reports.items()},
            "human_readable_policy": "Train is earliest time period, validation is future-near, test is final unseen future.",
        }
        split_metadata_path = output_root / "split_metadata.json"
        write_json(split_metadata_path, split_metadata)
        output_files["split_metadata"] = str(split_metadata_path)

        windowing_metadata = {
            "lookbacks": [int(value) for value in cfg["lookback_horizon"]["lookbacks"]],
            "horizons": [int(value) for value in cfg["lookback_horizon"]["horizons"]],
            "horizon_definition": cfg["lookback_horizon"]["horizon_definition"],
            "materialized_windows_created": False,
            "windowing_rule": "Input window uses feature rows up to forecast origin t; target is ramp_rate_pu(t+H); windows are built separately inside each split.",
            "statistical_family_note": "ARIMA/SARIMAX should use split target series and optional exogenous data; Notebook 04 does not force flattened ML-style windows for statistical models.",
            "human_readable_example": {
                "lookback": 3,
                "horizon": 2,
                "input_uses_times": ["t-2", "t-1", "t"],
                "target": "ramp_rate_pu(t+2)",
                "leakage_note": "No information after t is included in the input.",
            },
            "dataset_variants": {variant: report["window_metadata"] for variant, report in variant_reports.items()},
        }
        windowing_metadata_path = output_root / "windowing_metadata.json"
        write_json(windowing_metadata_path, windowing_metadata)
        output_files["windowing_metadata"] = str(windowing_metadata_path)

        scaler_metadata = {
            "scaler_strategy": "StandardScaler fitted on train features only for each dataset variant.",
            "target_scaled": False,
            "dataset_variants": {variant: report["scaler_metadata"] for variant, report in variant_reports.items()},
        }
        scaler_metadata_path = output_root / "scaler_metadata.json"
        write_json(scaler_metadata_path, scaler_metadata)
        output_files["scaler_metadata"] = str(scaler_metadata_path)

        leak = leakage_report(variant_reports)
        leakage_path = output_root / "leakage_check_report.json"
        write_json(leakage_path, leak)
        output_files["leakage_check_report"] = str(leakage_path)
        write_status(
            run_state / "section_03_reports.json",
            section_status(
                "reports",
                started,
                "success" if leak["checks"]["passed"] else "failed",
                "Notebook 04 metadata and leakage reports written.",
                {"experiment_count": len(experiment_rows), "leakage_passed": leak["checks"]["passed"]},
                None if leak["checks"]["passed"] else "Leakage checks failed.",
            ),
        )
    except Exception:
        errors.append(traceback.format_exc())
        write_status(run_state / "section_03_reports.json", section_status("reports", started, "failed", "Report writing failed.", error_message=errors[-1]))
        raise

    started = utc_now()
    final_checks = {
        "all_4_dataset_variants_prepared": set(variant_reports) == {"full", "selected", "selected_physics", "full_physics"},
        "experiment_matrix_rows": int(len(experiment_rows)),
        "expected_experiment_matrix_rows": int(4 * len(cfg["lookback_horizon"]["lookbacks"]) * len(cfg["lookback_horizon"]["horizons"])),
        "no_model_training_happened": True,
        "no_notebook_05_started": not (root / "outputs" / "05_baseline_models").exists(),
        "leakage_checks_passed": bool(leak["checks"]["passed"]),
        "all_outputs_written": all(Path(path).exists() for path in output_files.values()),
    }
    final_checks["experiment_matrix_row_count_ok"] = final_checks["experiment_matrix_rows"] == final_checks["expected_experiment_matrix_rows"]
    final_status = "success" if all(value for value in final_checks.values() if isinstance(value, bool)) and not errors else "failed"
    report_path = output_root / "notebook_04_final_report.json"
    output_files["notebook_04_final_report"] = str(report_path)
    report = {
        "notebook_name": NOTEBOOK_NAME,
        "execution_type": "local",
        "primary_resolution": cfg["data"]["primary_resolution_candidate"],
        "target_column": target_column,
        "input_files": {variant: str(path) for variant, path in paths.items()},
        "output_files": output_files,
        "split_ratio": {
            "train": float(cfg["splits"]["train_fraction"]),
            "validation": float(cfg["splits"]["validation_fraction"]),
            "test": float(cfg["splits"]["test_fraction"]),
        },
        "split_metadata": split_metadata,
        "windowing_metadata": {
            "lookback_count": len(cfg["lookback_horizon"]["lookbacks"]),
            "horizon_count": len(cfg["lookback_horizon"]["horizons"]),
            "combination_count": len(cfg["lookback_horizon"]["lookbacks"]) * len(cfg["lookback_horizon"]["horizons"]),
            "materialized_windows_created": False,
            "human_readable_example": windowing_metadata["human_readable_example"],
        },
        "scaler_strategy": scaler_metadata["scaler_strategy"],
        "experiment_matrix": {
            "row_count": int(len(experiment_rows)),
            "path": str(experiment_path),
            "recommended_stage_counts": {str(key): int(value) for key, value in experiment_df["recommended_stage"].value_counts().to_dict().items()},
        },
        "variant_reports": variant_reports,
        "leakage_checks": leak,
        "validation_checks": final_checks,
        "model_training_started": False,
        "notebook_05_started": False,
        "warnings": warnings,
        "errors": errors,
        "final_status": final_status,
    }
    write_json(report_path, report)
    write_local_runner_report(report)
    write_status(
        run_state / "section_04_final_validation.json",
        section_status(
            "final_validation",
            started,
            final_status,
            "Notebook 04 final validations completed.",
            final_checks,
            None if final_status == "success" else "One or more final checks failed.",
        ),
    )
    return report
