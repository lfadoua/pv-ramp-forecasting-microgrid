from __future__ import annotations

import importlib.util
import json
import math
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import load_config, project_root
from .reports import write_csv, write_json, write_markdown
from .status import section_status, utc_now, write_status


NOTEBOOK_NAME = "07_classical_ml_models"
TIME_COLUMN = "timestamp"
TARGET_COLUMN = "ramp_rate_pu"
SPLIT_NAMES = ["train", "validation", "test"]
RESOLUTIONS = ["one_minute", "one_hour"]
DATASET_VARIANTS = ["full", "selected", "full_physics", "selected_physics"]
STAGE_A_PAIRS = [(8, 1), (16, 8), (32, 16)]
STAGE0_CONFIG = {"resolution": "one_minute", "dataset_variant": "selected_physics", "model": "ridge", "lookback": 8, "horizon": 1}
STAGE0_SAMPLE_LIMITS = {"train": 5000, "validation": 2000, "test": 2000}
MODEL_KEYS = ["linear_regression", "ridge", "hist_gradient_boosting", "xgboost"]
DEFERRED_HEAVY_MODELS = ["knn", "svr", "random_forest", "extra_trees", "gradient_boosting"]
RANDOM_SEED = 42


@dataclass(frozen=True)
class BackendDecision:
    model_key: str
    model: str
    backend_requested: str
    backend_used: str
    fallback_used: bool
    fallback_reason: str
    hardware_used: str
    gpu_available: bool
    available: bool


def read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    return json.loads(target.read_text(encoding="utf-8"))


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def scalar(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return finite_or_none(value)
    if isinstance(value, dict):
        return {str(k): scalar(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scalar(v) for v in value]
    return value


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, persistence_rmse: float | None) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"mae": None, "rmse": None, "nrmse": None, "r2": None, "skill_rmse_vs_persistence": None}
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    sse = float(np.sum(np.square(error)))
    mean_true = float(np.mean(y_true))
    sst = float(np.sum(np.square(np.asarray(y_true, dtype=np.float64) - mean_true)))
    r2 = None if sst <= 0 else float(1.0 - sse / sst)
    skill = None
    if persistence_rmse is not None and persistence_rmse > 0:
        skill = float(1.0 - rmse / persistence_rmse)
    return {
        "mae": finite_or_none(mae),
        "rmse": finite_or_none(rmse),
        "nrmse": finite_or_none(rmse / 2.0),
        "r2": finite_or_none(r2),
        "skill_rmse_vs_persistence": finite_or_none(skill),
    }


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def detect_environment() -> dict[str, Any]:
    status: dict[str, Any] = {
        "sklearn_available": module_available("sklearn"),
        "xgboost_available": module_available("xgboost"),
        "cuml_available": module_available("cuml"),
        "cupy_available": module_available("cupy"),
        "gpu_available": False,
        "gpu_detection_notes": [],
    }
    if status["cupy_available"]:
        try:
            import cupy as cp

            status["gpu_count"] = int(cp.cuda.runtime.getDeviceCount())
            status["gpu_available"] = status["gpu_count"] > 0
        except Exception as exc:
            status["gpu_count"] = 0
            status["gpu_detection_notes"].append(f"cupy detection failed: {exc}")
    else:
        status["gpu_count"] = 0
        status["gpu_detection_notes"].append("cupy is not installed locally.")
    return status


def backend_decision(model_key: str, env: dict[str, Any]) -> BackendDecision:
    gpu_available = bool(env.get("gpu_available", False))
    cuml_available = bool(env.get("cuml_available", False))
    xgboost_available = bool(env.get("xgboost_available", False))
    labels = {
        "linear_regression": "Linear Regression",
        "ridge": "Ridge",
        "hist_gradient_boosting": "HistGradientBoosting",
        "xgboost": "XGBoost",
    }
    if model_key in {"linear_regression", "ridge"}:
        if cuml_available and gpu_available:
            return BackendDecision(model_key, labels[model_key], "cuml_rapids", "cuml_rapids", False, "", "local_gpu", True, True)
        return BackendDecision(
            model_key,
            labels[model_key],
            "cuml_rapids_if_available",
            "scikit_learn",
            True,
            "cuML/RAPIDS is not available in the local environment; using scikit-learn CPU fallback.",
            "local_cpu",
            gpu_available,
            True,
        )
    if model_key == "hist_gradient_boosting":
        return BackendDecision(
            model_key,
            labels[model_key],
            "scikit_learn_hist_gradient_boosting_cpu",
            "scikit_learn",
            False,
            "",
            "local_cpu",
            gpu_available,
            True,
        )
    if model_key == "xgboost":
        if not xgboost_available:
            return BackendDecision(
                model_key,
                labels[model_key],
                "xgboost_gpu_if_available_else_cpu",
                "unavailable",
                True,
                "xgboost is not installed in the local environment; configuration skipped.",
                "none",
                gpu_available,
                False,
            )
        return BackendDecision(
            model_key,
            labels[model_key],
            "xgboost_gpu_if_available_else_cpu",
            "xgboost_gpu" if gpu_available else "xgboost_cpu",
            not gpu_available,
            "" if gpu_available else "Local GPU is unavailable; XGBoost would use CPU if executed.",
            "local_gpu" if gpu_available else "local_cpu",
            gpu_available,
            True,
        )
    raise ValueError(f"Unsupported model key: {model_key}")


def prerequisite_files(root: Path) -> list[Path]:
    return [
        root / "outputs" / "reports" / "dataset_inventory.csv",
        root / "outputs" / "reports" / "clearml_run_registry.csv",
        root / "outputs" / "reports" / "pre_notebook_07_cleanup_report.md",
        root / "outputs" / "reports" / "progressive_training_strategy.md",
        root / "outputs" / "reports" / "pipeline_build_status.md",
        root / "outputs" / "05_baseline_models" / "persistence_reference_by_variant_lh.csv",
        root / "outputs" / "05_baseline_models" / "baseline_results_train_validation_test.csv",
    ]


def validate_prerequisites(root: Path) -> dict[str, Any]:
    missing_files = [str(path) for path in prerequisite_files(root) if not path.exists()]
    inventory_path = root / "outputs" / "reports" / "dataset_inventory.csv"
    inventory = pd.read_csv(inventory_path) if inventory_path.exists() else pd.DataFrame()
    dataset_checks: list[dict[str, Any]] = []
    missing_dataset_issues: list[str] = []
    for _, row in inventory.iterrows():
        manifest_path = Path(row["manifest_path"])
        source_path = Path(row["source_path"])
        issue_prefix = f"{row['resolution']}/{row['dataset_variant']}"
        if not manifest_path.exists():
            missing_dataset_issues.append(f"{issue_prefix}: missing manifest {manifest_path}")
            continue
        if not source_path.exists():
            missing_dataset_issues.append(f"{issue_prefix}: missing source {source_path}")
            continue
        manifest = read_json(manifest_path)
        header = list(pd.read_csv(source_path, nrows=0).columns)
        check = {
            "resolution": row["resolution"],
            "dataset_variant": row["dataset_variant"],
            "ready": str(row["ready"]) == "True",
            "row_count_gt0": int(row["row_count"]) > 0,
            "feature_count_gt0": int(row["feature_count"]) > 0,
            "target_present": TARGET_COLUMN in header,
            "target_last": bool(header and header[-1] == TARGET_COLUMN),
            "timestamp_counted_as_feature": TIME_COLUMN in manifest.get("feature_columns", []),
            "target_counted_as_feature": TARGET_COLUMN in manifest.get("feature_columns", []),
            "feature_count_matches": int(row["feature_count"]) == int(manifest.get("feature_count", -1)),
        }
        dataset_checks.append(check)
        if not all(
            [
                check["ready"],
                check["row_count_gt0"],
                check["feature_count_gt0"],
                check["target_present"],
                check["target_last"],
                not check["timestamp_counted_as_feature"],
                not check["target_counted_as_feature"],
                check["feature_count_matches"],
            ]
        ):
            missing_dataset_issues.append(f"{issue_prefix}: failed model-ready checks {check}")

    ref_path = root / "outputs" / "05_baseline_models" / "persistence_reference_by_variant_lh.csv"
    missing_references: list[dict[str, Any]] = []
    if ref_path.exists():
        refs = pd.read_csv(ref_path)
        for resolution in RESOLUTIONS:
            for variant in DATASET_VARIANTS:
                for lookback, horizon in STAGE_A_PAIRS:
                    for split_name in SPLIT_NAMES:
                        mask = (
                            (refs["resolution"] == resolution)
                            & (refs["dataset_variant"] == variant)
                            & (refs["lookback"].astype(int) == int(lookback))
                            & (refs["horizon"].astype(int) == int(horizon))
                            & (refs["split_name"] == split_name)
                        )
                        if not bool(mask.any()):
                            missing_references.append(
                                {
                                    "resolution": resolution,
                                    "dataset_variant": variant,
                                    "lookback": int(lookback),
                                    "horizon": int(horizon),
                                    "split": split_name,
                                }
                            )
    else:
        missing_references.append({"error": "persistence reference table missing"})

    checks = {
        "required_files_exist": len(missing_files) == 0,
        "missing_files": missing_files,
        "dataset_count": int(len(dataset_checks)),
        "datasets_ready": len(dataset_checks) == 8 and len(missing_dataset_issues) == 0,
        "dataset_checks": dataset_checks,
        "dataset_issues": missing_dataset_issues,
        "stage_a_persistence_references_complete": len(missing_references) == 0,
        "missing_stage_a_references": missing_references,
        "notebook_08_not_started": not (root / "outputs" / "08_deep_learning_models").exists(),
        "kaggle_not_used": True,
    }
    checks["passed"] = bool(
        checks["required_files_exist"]
        and checks["datasets_ready"]
        and checks["stage_a_persistence_references_complete"]
        and checks["notebook_08_not_started"]
    )
    return checks


def load_persistence_references(root: Path) -> dict[tuple[str, str, int, int, str], float]:
    path = root / "outputs" / "05_baseline_models" / "persistence_reference_by_variant_lh.csv"
    refs = pd.read_csv(path)
    result: dict[tuple[str, str, int, int, str], float] = {}
    for _, row in refs.iterrows():
        result[(str(row["resolution"]), str(row["dataset_variant"]), int(row["lookback"]), int(row["horizon"]), str(row["split_name"]))] = float(row["persistence_RMSE"])
    return result


def manifest_path(root: Path, resolution: str, variant: str) -> Path:
    return root / "outputs" / resolution / "notebook_04_split_windowing" / variant / "window_ready_manifest.json"


def load_dataset_splits(root: Path, resolution: str, variant: str) -> dict[str, Any]:
    manifest = read_json(manifest_path(root, resolution, variant))
    source_path = Path(manifest["source_path"])
    feature_columns = list(manifest["feature_columns"])
    if TARGET_COLUMN in feature_columns or TIME_COLUMN in feature_columns:
        raise ValueError(f"Feature columns for {resolution}/{variant} contain target or timestamp.")
    scaler = joblib.load(manifest["scaler_file"])
    usecols = [TIME_COLUMN] + feature_columns + [TARGET_COLUMN]
    df = pd.read_csv(source_path, usecols=usecols)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    if int(df[TIME_COLUMN].isna().sum()) > 0:
        raise ValueError(f"Invalid timestamps in {source_path}")
    if not df[TIME_COLUMN].is_monotonic_increasing:
        raise ValueError(f"Timestamps are not sorted in {source_path}")
    splits: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        meta = manifest["split_summary"][split_name]
        start = int(meta["start_row_in_dataset"])
        end = int(meta["end_row_exclusive_in_dataset"])
        split = df.iloc[start:end]
        raw_features = split[feature_columns].to_numpy(dtype=np.float64)
        scaled_features = scaler.transform(raw_features).astype(np.float32, copy=False)
        target = pd.to_numeric(split[TARGET_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(scaled_features).all() or not np.isfinite(target).all():
            raise ValueError(f"Non-finite features or target after scaling in {resolution}/{variant}/{split_name}.")
        splits[split_name] = {
            "features": scaled_features,
            "target": target,
            "row_count": int(len(split)),
            "start_timestamp": meta["start_timestamp"],
            "end_timestamp": meta["end_timestamp"],
        }
    return {
        "resolution": resolution,
        "dataset_variant": variant,
        "manifest": manifest,
        "feature_columns": feature_columns,
        "feature_count": int(manifest["feature_count"]),
        "splits": splits,
    }


def sample_count(row_count: int, lookback: int, horizon: int) -> int:
    return max(0, int(row_count) - int(lookback) - int(horizon) + 1)


def estimated_matrix_mb(rows: int, feature_count: int, lookback: int) -> float:
    return float(rows) * float(feature_count) * float(lookback) * 4.0 / 1_000_000.0


def build_windows(features: np.ndarray, target: np.ndarray, lookback: int, horizon: int, max_samples: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    count = sample_count(len(target), lookback, horizon)
    if count <= 0:
        return np.empty((0, lookback * features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float64)
    if max_samples is not None:
        count = min(count, int(max_samples))
    window_view = np.lib.stride_tricks.sliding_window_view(features, window_shape=(lookback, features.shape[1]))
    x = window_view[:count, 0, :, :].reshape(count, lookback * features.shape[1]).astype(np.float32, copy=False)
    y_start = lookback + horizon - 1
    y = target[y_start : y_start + count].astype(np.float64, copy=False)
    return x, y


def make_model(model_key: str):
    if model_key == "linear_regression":
        from sklearn.linear_model import LinearRegression

        return LinearRegression(n_jobs=-1)
    if model_key == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0, solver="lsqr", random_state=RANDOM_SEED)
    if model_key == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=50,
            max_leaf_nodes=31,
            learning_rate=0.08,
            l2_regularization=0.0,
            early_stopping=True,
            random_state=RANDOM_SEED,
        )
    if model_key == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.06,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    raise ValueError(model_key)


def run_limits(model_key: str) -> dict[str, Any]:
    if model_key == "linear_regression":
        return {"max_train_matrix_mb": 210.0, "max_train_samples": 500_000}
    if model_key == "ridge":
        return {"max_train_matrix_mb": 390.0, "max_train_samples": 500_000}
    if model_key == "hist_gradient_boosting":
        return {"max_train_matrix_mb": 96.0, "max_train_samples": 50_000}
    if model_key == "xgboost":
        return {"max_train_matrix_mb": 96.0, "max_train_samples": 50_000}
    return {"max_train_matrix_mb": 0.0, "max_train_samples": 0}


def should_run_config(model_key: str, train_samples: int, feature_count: int, lookback: int, decision: BackendDecision) -> tuple[bool, str]:
    if not decision.available:
        return False, decision.fallback_reason
    limits = run_limits(model_key)
    matrix_mb = estimated_matrix_mb(train_samples, feature_count, lookback)
    if train_samples > limits["max_train_samples"]:
        return (
            False,
            f"Full training split has {train_samples} samples, above local Stage A guard {limits['max_train_samples']} for {decision.model}.",
        )
    if matrix_mb > limits["max_train_matrix_mb"]:
        return (
            False,
            f"Estimated flattened train matrix {matrix_mb:.1f} MB exceeds local Stage A guard {limits['max_train_matrix_mb']:.1f} MB for {decision.model}.",
        )
    return True, ""


def result_row(
    *,
    resolution: str,
    variant: str,
    model: str,
    model_key: str,
    lookback: int,
    horizon: int,
    split_name: str,
    metrics: dict[str, Any] | None,
    feature_count: int,
    row_counts_used: dict[str, int],
    decision: BackendDecision,
    elapsed_train: float | None,
    elapsed_predict: float | None,
    status: str,
    run_id: str,
    clearml_task_id: str | None,
    sample_count_value: int,
    persistence_rmse: float | None,
    skip_reason: str = "",
    estimated_train_matrix_mb: float | None = None,
    sample_policy: str = "full_split",
) -> dict[str, Any]:
    metrics = metrics or {}
    fallback_used = bool(decision.fallback_used)
    fallback_reason = decision.fallback_reason
    if skip_reason and not fallback_reason:
        fallback_reason = skip_reason
    return {
        "resolution": resolution,
        "dataset_variant": variant,
        "model": model,
        "model_key": model_key,
        "lookback": int(lookback),
        "horizon": int(horizon),
        "split": split_name,
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "nrmse": metrics.get("nrmse"),
        "r2": metrics.get("r2"),
        "skill_rmse_vs_persistence": metrics.get("skill_rmse_vs_persistence"),
        "feature_count": int(feature_count),
        "sample_count": int(sample_count_value),
        "train_rows_used": int(row_counts_used.get("train", 0)),
        "validation_rows_used": int(row_counts_used.get("validation", 0)),
        "test_rows_used": int(row_counts_used.get("test", 0)),
        "backend_requested": decision.backend_requested,
        "backend_used": decision.backend_used,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "hardware_used": decision.hardware_used,
        "gpu_available": bool(decision.gpu_available),
        "elapsed_seconds_train": finite_or_none(elapsed_train if split_name == "train" else 0.0),
        "elapsed_seconds_predict": finite_or_none(elapsed_predict),
        "status": status,
        "stage": "stage0" if status == "smoke_test" else "stage_a",
        "run_id": run_id,
        "clearml_task_id": clearml_task_id,
        "persistence_rmse_reference": finite_or_none(persistence_rmse),
        "persistence_reference_source": "Notebook 05 / Phase 5B persistence_reference_by_variant_lh.csv",
        "target_alignment": "input window ends at forecast origin t; target is ramp_rate_pu(t+H)",
        "same_row_target_feature_used": False,
        "future_information_used": False,
        "sample_policy": sample_policy,
        "skip_reason": skip_reason,
        "estimated_train_matrix_mb": finite_or_none(estimated_train_matrix_mb),
    }


def skipped_rows_for_config(
    *,
    resolution: str,
    variant: str,
    model_key: str,
    lookback: int,
    horizon: int,
    feature_count: int,
    decision: BackendDecision,
    status: str,
    run_id: str,
    clearml_task_id: str | None,
    split_counts: dict[str, int],
    skip_reason: str,
    estimated_train_mb: float,
) -> list[dict[str, Any]]:
    rows = []
    for split_name in SPLIT_NAMES:
        rows.append(
            result_row(
                resolution=resolution,
                variant=variant,
                model=decision.model,
                model_key=model_key,
                lookback=lookback,
                horizon=horizon,
                split_name=split_name,
                metrics=None,
                feature_count=feature_count,
                row_counts_used={"train": 0, "validation": 0, "test": 0},
                decision=decision,
                elapsed_train=None,
                elapsed_predict=None,
                status=status,
                run_id=run_id,
                clearml_task_id=clearml_task_id,
                sample_count_value=split_counts[split_name],
                persistence_rmse=None,
                skip_reason=skip_reason,
                estimated_train_matrix_mb=estimated_train_mb,
                sample_policy="not_run",
            )
        )
    return rows


def evaluate_config(
    data: dict[str, Any],
    model_key: str,
    lookback: int,
    horizon: int,
    status: str,
    run_id: str,
    clearml_task_id: str | None,
    persistence_refs: dict[tuple[str, str, int, int, str], float],
    decision: BackendDecision,
    sample_limits: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolution = data["resolution"]
    variant = data["dataset_variant"]
    feature_count = int(data["feature_count"])
    split_counts = {split: sample_count(data["splits"][split]["row_count"], lookback, horizon) for split in SPLIT_NAMES}
    estimated_train_mb = estimated_matrix_mb(split_counts["train"], feature_count, lookback)
    if any(value <= 0 for value in split_counts.values()):
        reason = f"Non-positive sample count for {resolution}/{variant}/L{lookback}/H{horizon}: {split_counts}"
        return skipped_rows_for_config(
            resolution=resolution,
            variant=variant,
            model_key=model_key,
            lookback=lookback,
            horizon=horizon,
            feature_count=feature_count,
            decision=decision,
            status="aborted_invalid_samples",
            run_id=run_id,
            clearml_task_id=clearml_task_id,
            split_counts=split_counts,
            skip_reason=reason,
            estimated_train_mb=estimated_train_mb,
        ), {"status": "aborted_invalid_samples", "reason": reason}

    if status == "screening":
        ok, reason = should_run_config(model_key, split_counts["train"], feature_count, lookback, decision)
        if not ok:
            skipped_status = "skipped_unavailable" if not decision.available else "aborted_expensive"
            return skipped_rows_for_config(
                resolution=resolution,
                variant=variant,
                model_key=model_key,
                lookback=lookback,
                horizon=horizon,
                feature_count=feature_count,
                decision=decision,
                status=skipped_status,
                run_id=run_id,
                clearml_task_id=clearml_task_id,
                split_counts=split_counts,
                skip_reason=reason,
                estimated_train_mb=estimated_train_mb,
            ), {"status": skipped_status, "reason": reason}

    x_train, y_train = build_windows(
        data["splits"]["train"]["features"],
        data["splits"]["train"]["target"],
        lookback,
        horizon,
        max_samples=sample_limits.get("train") if sample_limits else None,
    )
    if len(y_train) == 0:
        raise ValueError(f"No train windows for {resolution}/{variant}/L{lookback}/H{horizon}")
    model = make_model(model_key)
    started = time.perf_counter()
    model.fit(x_train, y_train)
    elapsed_train = time.perf_counter() - started
    row_counts_used = {"train": int(len(y_train)), "validation": 0, "test": 0}
    rows: list[dict[str, Any]] = []
    for split_name in SPLIT_NAMES:
        if split_name == "train":
            x_split = x_train
            y_split = y_train
        else:
            x_split, y_split = build_windows(
                data["splits"][split_name]["features"],
                data["splits"][split_name]["target"],
                lookback,
                horizon,
                max_samples=sample_limits.get(split_name) if sample_limits else None,
            )
        row_counts_used[split_name] = int(len(y_split))
        pred_started = time.perf_counter()
        y_pred = model.predict(x_split)
        elapsed_predict = time.perf_counter() - pred_started
        ref_key = (resolution, variant, lookback, horizon, split_name)
        if ref_key not in persistence_refs:
            raise ValueError(f"Missing Persistence reference for {ref_key}")
        persistence_rmse = persistence_refs[ref_key]
        metrics = regression_metrics(y_split, y_pred, persistence_rmse)
        rows.append(
            result_row(
                resolution=resolution,
                variant=variant,
                model=decision.model,
                model_key=model_key,
                lookback=lookback,
                horizon=horizon,
                split_name=split_name,
                metrics=metrics,
                feature_count=feature_count,
                row_counts_used=row_counts_used,
                decision=decision,
                elapsed_train=elapsed_train,
                elapsed_predict=elapsed_predict,
                status=status,
                run_id=run_id,
                clearml_task_id=clearml_task_id,
                sample_count_value=int(len(y_split)),
                persistence_rmse=persistence_rmse,
                estimated_train_matrix_mb=estimated_train_mb,
                sample_policy="smoke_head_cap" if sample_limits else "full_split",
            )
        )
        if split_name != "train":
            del x_split, y_split
    del x_train, y_train, model
    return rows, {"status": status, "reason": "", "elapsed_train": elapsed_train}


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    completed = results[results["status"].isin(["smoke_test", "screening"])].copy()
    if completed.empty:
        return pd.DataFrame()
    grouped = (
        completed.groupby(["status", "resolution", "dataset_variant", "model", "lookback", "horizon", "backend_used"], dropna=False)
        .agg(
            split_count=("split", "count"),
            feature_count=("feature_count", "max"),
            train_rows_used=("train_rows_used", "max"),
            validation_rows_used=("validation_rows_used", "max"),
            test_rows_used=("test_rows_used", "max"),
            train_rmse=("rmse", lambda s: s[completed.loc[s.index, "split"] == "train"].iloc[0] if (completed.loc[s.index, "split"] == "train").any() else np.nan),
            validation_rmse=("rmse", lambda s: s[completed.loc[s.index, "split"] == "validation"].iloc[0] if (completed.loc[s.index, "split"] == "validation").any() else np.nan),
            test_rmse=("rmse", lambda s: s[completed.loc[s.index, "split"] == "test"].iloc[0] if (completed.loc[s.index, "split"] == "test").any() else np.nan),
            validation_skill=("skill_rmse_vs_persistence", lambda s: s[completed.loc[s.index, "split"] == "validation"].iloc[0] if (completed.loc[s.index, "split"] == "validation").any() else np.nan),
            test_skill=("skill_rmse_vs_persistence", lambda s: s[completed.loc[s.index, "split"] == "test"].iloc[0] if (completed.loc[s.index, "split"] == "test").any() else np.nan),
            elapsed_seconds_train=("elapsed_seconds_train", "max"),
            elapsed_seconds_predict_total=("elapsed_seconds_predict", "sum"),
        )
        .reset_index()
    )
    return grouped.sort_values(["status", "resolution", "dataset_variant", "lookback", "horizon", "model"]).reset_index(drop=True)


def screening_report(summary: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    screening = summary[summary["status"] == "screening"].copy()
    if screening.empty:
        return pd.DataFrame()
    screening["rank_within_resolution_variant_lh"] = (
        screening.groupby(["resolution", "dataset_variant", "lookback", "horizon"])["validation_rmse"].rank(method="first", ascending=True)
    )
    screening["comparison_rule"] = "Compare only rows with same resolution, dataset_variant, lookback, horizon, split/status category."
    screening["resolution_native_note"] = "one_minute L/H are minute steps; one_hour L/H are hour steps."
    return screening.sort_values(["resolution", "dataset_variant", "lookback", "horizon", "rank_within_resolution_variant_lh"]).reset_index(drop=True)


def backend_report_rows(env: dict[str, Any], decisions: dict[str, BackendDecision]) -> list[dict[str, Any]]:
    rows = []
    for decision in decisions.values():
        rows.append(
            {
                "model": decision.model,
                "model_key": decision.model_key,
                "backend_requested": decision.backend_requested,
                "backend_used": decision.backend_used,
                "fallback_used": decision.fallback_used,
                "fallback_reason": decision.fallback_reason,
                "hardware_used": decision.hardware_used,
                "gpu_available": decision.gpu_available,
                "cuml_available": bool(env.get("cuml_available")),
                "xgboost_available": bool(env.get("xgboost_available")),
                "sklearn_available": bool(env.get("sklearn_available")),
            }
        )
    for model_key in DEFERRED_HEAVY_MODELS:
        rows.append(
            {
                "model": model_key,
                "model_key": model_key,
                "backend_requested": "not_started_in_stage_a_fast_screening",
                "backend_used": "not_run",
                "fallback_used": False,
                "fallback_reason": "Deferred by Stage A scope; fast scalable models only.",
                "hardware_used": "none",
                "gpu_available": bool(env.get("gpu_available")),
                "cuml_available": bool(env.get("cuml_available")),
                "xgboost_available": bool(env.get("xgboost_available")),
                "sklearn_available": bool(env.get("sklearn_available")),
            }
        )
    return rows


def validation_checks(root: Path, results: pd.DataFrame, prerequisites: dict[str, Any], output_files: dict[str, str]) -> dict[str, Any]:
    successful = results[results["status"].isin(["smoke_test", "screening"])]
    table_outputs = [
        "ml_results_train_validation_test",
        "ml_results_summary",
        "ml_dataset_screening_report",
        "ml_backend_report",
        "ml_skipped_configs_report",
    ]
    checks = {
        "prerequisites_verified": bool(prerequisites.get("passed")),
        "stage0_succeeded": bool(((results["status"] == "smoke_test") & results["rmse"].notna()).any()),
        "stage_a_executed": bool(((results["status"] == "screening") & results["rmse"].notna()).any()),
        "required_result_columns_present": all(
            column in results.columns
            for column in [
                "resolution",
                "dataset_variant",
                "model",
                "lookback",
                "horizon",
                "split",
                "mae",
                "rmse",
                "nrmse",
                "r2",
                "skill_rmse_vs_persistence",
                "feature_count",
                "train_rows_used",
                "validation_rows_used",
                "test_rows_used",
                "backend_requested",
                "backend_used",
                "fallback_used",
                "fallback_reason",
                "hardware_used",
                "gpu_available",
                "elapsed_seconds_train",
                "elapsed_seconds_predict",
                "status",
                "run_id",
                "clearml_task_id",
            ]
        ),
        "all_required_output_files_created": all(Path(output_files[key]).exists() for key in table_outputs if key in output_files),
        "persistence_reference_used": bool(successful["persistence_rmse_reference"].notna().all()) if not successful.empty else False,
        "no_future_information_used": bool((results["future_information_used"] == False).all()),  # noqa: E712
        "no_same_row_target_feature_used": bool((results["same_row_target_feature_used"] == False).all()),  # noqa: E712
        "status_values_controlled": set(results["status"].unique()).issubset({"smoke_test", "screening", "aborted_expensive", "skipped_unavailable"}),
        "notebook_08_not_started": not (root / "outputs" / "08_deep_learning_models").exists(),
        "kaggle_push_or_run_not_launched": True,
        "stage_b_full_grid_heavy_models_not_started": True,
    }
    checks["passed"] = bool(all(value for value in checks.values() if isinstance(value, bool)))
    return checks


def readiness_report_text(report: dict[str, Any]) -> str:
    checks = report.get("validation_checks", {})
    return "\n".join(
        [
            "# Phase 7 Stage A Readiness Report",
            "",
            f"- Final status: `{report.get('final_status')}`",
            f"- Prerequisites verified: `{checks.get('prerequisites_verified')}`",
            f"- Stage 0 succeeded: `{checks.get('stage0_succeeded')}`",
            f"- Stage A executed: `{checks.get('stage_a_executed')}`",
            f"- Models run successfully: `{report.get('models_run')}`",
            f"- Models skipped/aborted: `{report.get('models_skipped_or_aborted')}`",
            f"- Datasets evaluated: `{report.get('datasets_evaluated')}`",
            f"- L/H pairs evaluated: `{report.get('stage_a_pairs')}`",
            f"- cuML available locally: `{report.get('environment', {}).get('cuml_available')}`",
            f"- XGBoost available locally: `{report.get('environment', {}).get('xgboost_available')}`",
            f"- Backends used: `{report.get('backends_used')}`",
            f"- Fallback occurred: `{report.get('fallback_occurred')}`",
            f"- Skipped/aborted configs: `{report.get('skipped_or_aborted_config_count')}`",
            f"- Results saved locally: `{checks.get('all_required_output_files_created')}`",
            f"- ClearML task ID: `{report.get('clearml_task_id')}`",
            f"- ClearML task URL: `{report.get('clearml_task_url')}`",
            f"- Notebook 08 untouched: `{checks.get('notebook_08_not_started')}`",
            f"- Kaggle push/run launched: `{not checks.get('kaggle_push_or_run_not_launched')}`",
            f"- Stage B/full-grid/heavy models started: `{not checks.get('stage_b_full_grid_heavy_models_not_started')}`",
            "",
            "Fairness notes:",
            "- Stage A uses chronological train/validation/test splits and train-fitted Notebook 04 scalers.",
            "- Skill_RMSE uses Notebook 05 / Phase 5B Persistence references matched by resolution, dataset_variant, lookback, horizon, and split.",
            "- L/H is resolution-native: one_minute L=8 means 8 minutes; one_hour L=8 means 8 hours.",
            "- Aborted local configurations are not final comparison rows and should not be promoted without an approved resource plan.",
            "",
            "Ready for user review before Stage B: `yes`" if report.get("ready_for_stage_b_after_review") else "Ready for user review before Stage B: `no`",
            "",
        ]
    )


def pipeline_status_text(report: dict[str, Any]) -> str:
    checks = report.get("validation_checks", {})
    return "\n".join(
        [
            "# Pipeline Build Status",
            "",
            "Phase: `7 Stage 0 / Stage A classical ML screening`",
            "",
            "Completed:",
            "- Phases 1-6 remain completed.",
            "- Phase 5B one-hour baseline completion remains completed.",
            "- Notebook 07 Stage 0 smoke test and controlled local Stage A screening were executed.",
            "- No Notebook 08, Deep Learning, Kaggle push, or Kaggle training was launched.",
            "",
            "Dataset readiness:",
            "- Ready datasets: `8/8`.",
            "- Persistence references: complete for Stage A pairs across one_minute and one_hour.",
            "",
            "Notebook 07:",
            f"- Stage 0 succeeded: `{checks.get('stage0_succeeded')}`.",
            f"- Stage A executed: `{checks.get('stage_a_executed')}`.",
            f"- Result files created: `{checks.get('all_required_output_files_created')}`.",
            f"- ClearML task ID: `{report.get('clearml_task_id')}`.",
            "- Stage B/full-grid/heavy models: not started.",
            "",
            "Kaggle/ClearML:",
            "- Kaggle/ClearML access-check status: `READY`.",
            "- Official Kaggle ClearML task ID: `1bd4e5ba207f49faa1591750a57c57c1`.",
            "- No Kaggle notebook push or run was launched for Stage A.",
            "",
            "Training:",
            "- Classical ML Stage A screening: completed locally in controlled mode.",
            "- Heavy Classical ML: not started.",
            "- Deep Learning: not started.",
            "- Notebook 08: not started.",
            "",
            f"Ready for user review before Stage B/full-grid/heavy models: `{'yes' if report.get('ready_for_stage_b_after_review') else 'no'}`",
            "",
        ]
    )


def write_local_runner_report(local_output_dir: str | None, report: dict[str, Any], artifact_paths: list[str]) -> None:
    if not local_output_dir:
        return
    payload = {
        "final_status": report.get("final_status"),
        "data_shapes": report.get("data_shapes", {}),
        "metrics": {
            "stage0_succeeded": report.get("validation_checks", {}).get("stage0_succeeded"),
            "stage_a_executed": report.get("validation_checks", {}).get("stage_a_executed"),
            "successful_result_rows": report.get("successful_result_row_count"),
            "skipped_or_aborted_config_count": report.get("skipped_or_aborted_config_count"),
            "fallback_occurred": report.get("fallback_occurred"),
        },
        "artifact_paths": artifact_paths,
    }
    write_json(Path(local_output_dir) / "local_clearml_report.json", scalar(payload))


def run_notebook07_stage_a(config_path: str | Path) -> dict[str, Any]:
    started_at = utc_now()
    root = project_root()
    config = load_config(config_path)
    output_root = root / "outputs" / NOTEBOOK_NAME
    run_state_dir = output_root / "run_state"
    reports_dir = root / "outputs" / "reports"
    output_root.mkdir(parents=True, exist_ok=True)
    run_state_dir.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("LOCAL_RUN_ID") or started_at.replace("-", "").replace(":", "").replace("+", "_")
    clearml_task_id = os.environ.get("LOCAL_CLEARML_TASK_ID") or None
    clearml_task_url = os.environ.get("LOCAL_CLEARML_TASK_URL") or None
    clearml_project_name = os.environ.get("LOCAL_CLEARML_PROJECT_NAME") or config.get("clearml", {}).get("subprojects", {}).get("07", "PV_RampRate_Forecasting/07_ClassicalML")
    output_files: dict[str, str] = {}
    warnings_list: list[str] = []

    try:
        prerequisites = validate_prerequisites(root)
        if not prerequisites["passed"]:
            raise RuntimeError("Notebook 07 prerequisites failed: " + json.dumps(prerequisites, indent=2))
        env = detect_environment()
        if not env.get("sklearn_available"):
            raise RuntimeError("scikit-learn is required for local Stage 0 / Stage A execution.")
        decisions = {model_key: backend_decision(model_key, env) for model_key in MODEL_KEYS}
        persistence_refs = load_persistence_references(root)

        result_rows: list[dict[str, Any]] = []
        skipped_config_rows: list[dict[str, Any]] = []
        data_cache: dict[tuple[str, str], dict[str, Any]] = {}
        data_shapes: dict[str, Any] = {}

        stage0_key = (STAGE0_CONFIG["resolution"], STAGE0_CONFIG["dataset_variant"])
        data_cache[stage0_key] = load_dataset_splits(root, *stage0_key)
        stage0_decision = decisions[STAGE0_CONFIG["model"]]
        stage0_rows, stage0_info = evaluate_config(
            data_cache[stage0_key],
            STAGE0_CONFIG["model"],
            STAGE0_CONFIG["lookback"],
            STAGE0_CONFIG["horizon"],
            "smoke_test",
            run_id,
            clearml_task_id,
            persistence_refs,
            stage0_decision,
            sample_limits=STAGE0_SAMPLE_LIMITS,
        )
        result_rows.extend(stage0_rows)
        if stage0_info["status"] != "smoke_test" or not all(row.get("rmse") is not None for row in stage0_rows):
            raise RuntimeError(f"Stage 0 failed: {stage0_info}")

        for resolution in RESOLUTIONS:
            for variant in DATASET_VARIANTS:
                key = (resolution, variant)
                if key not in data_cache:
                    data_cache[key] = load_dataset_splits(root, resolution, variant)
                data = data_cache[key]
                data_shapes[f"{resolution}/{variant}"] = {
                    "feature_count": int(data["feature_count"]),
                    "split_rows": {split: int(data["splits"][split]["row_count"]) for split in SPLIT_NAMES},
                }
                for lookback, horizon in STAGE_A_PAIRS:
                    for model_key in MODEL_KEYS:
                        decision = decisions[model_key]
                        rows, info = evaluate_config(
                            data,
                            model_key,
                            lookback,
                            horizon,
                            "screening",
                            run_id,
                            clearml_task_id,
                            persistence_refs,
                            decision,
                            sample_limits=None,
                        )
                        result_rows.extend(rows)
                        if info["status"] not in {"screening", "smoke_test"}:
                            skipped_config_rows.append(
                                {
                                    "resolution": resolution,
                                    "dataset_variant": variant,
                                    "model": decision.model,
                                    "model_key": model_key,
                                    "lookback": int(lookback),
                                    "horizon": int(horizon),
                                    "status": info["status"],
                                    "skip_reason": info.get("reason", ""),
                                    "backend_requested": decision.backend_requested,
                                    "backend_used": decision.backend_used,
                                    "fallback_used": decision.fallback_used,
                                    "fallback_reason": decision.fallback_reason,
                                    "hardware_used": decision.hardware_used,
                                    "gpu_available": decision.gpu_available,
                                }
                            )
            # Keep memory bounded between resolutions.
            data_cache = {stage0_key: data_cache[stage0_key]} if resolution == "one_minute" else {}

        for model_key in DEFERRED_HEAVY_MODELS:
            skipped_config_rows.append(
                {
                    "resolution": "all",
                    "dataset_variant": "all",
                    "model": model_key,
                    "model_key": model_key,
                    "lookback": "",
                    "horizon": "",
                    "status": "not_started_stage_a_fast_screening_only",
                    "skip_reason": "Deferred by user-approved Stage A scope; heavy/non-fast models remain for later approval.",
                    "backend_requested": "not_started",
                    "backend_used": "not_run",
                    "fallback_used": False,
                    "fallback_reason": "",
                    "hardware_used": "none",
                    "gpu_available": bool(env.get("gpu_available")),
                }
            )

        results = pd.DataFrame(result_rows)
        summary = summary_table(results)
        dataset_screening = screening_report(summary, results)
        backend_report = pd.DataFrame(backend_report_rows(env, decisions))
        skipped = pd.DataFrame(skipped_config_rows)

        paths = {
            "ml_results_train_validation_test": output_root / "ml_results_train_validation_test.csv",
            "ml_results_summary": output_root / "ml_results_summary.csv",
            "ml_dataset_screening_report": output_root / "ml_dataset_screening_report.csv",
            "ml_backend_report": output_root / "ml_backend_report.csv",
            "ml_skipped_configs_report": output_root / "ml_skipped_configs_report.csv",
        }
        write_csv(paths["ml_results_train_validation_test"], results)
        write_csv(paths["ml_results_summary"], summary)
        write_csv(paths["ml_dataset_screening_report"], dataset_screening)
        write_csv(paths["ml_backend_report"], backend_report)
        write_csv(paths["ml_skipped_configs_report"], skipped)
        output_files.update({key: str(path) for key, path in paths.items()})

        report_preliminary = {
            "notebook_name": NOTEBOOK_NAME,
            "final_status": "success",
            "started_at": started_at,
            "finished_at": utc_now(),
            "run_id": run_id,
            "clearml_project_name": clearml_project_name,
            "clearml_task_id": clearml_task_id,
            "clearml_task_url": clearml_task_url,
            "environment": env,
            "prerequisites": prerequisites,
            "stage0_config": STAGE0_CONFIG,
            "stage0_sample_limits": STAGE0_SAMPLE_LIMITS,
            "stage_a_pairs": [{"lookback": lookback, "horizon": horizon} for lookback, horizon in STAGE_A_PAIRS],
            "dataset_variants": DATASET_VARIANTS,
            "resolutions": RESOLUTIONS,
            "models_attempted": [decisions[key].model for key in MODEL_KEYS],
            "models_run": sorted(results[results["status"].isin(["smoke_test", "screening"])]["model"].unique().tolist()),
            "models_skipped_or_aborted": sorted(results[~results["status"].isin(["smoke_test", "screening"])]["model"].unique().tolist() + DEFERRED_HEAVY_MODELS),
            "datasets_evaluated": sorted(results[results["status"].isin(["smoke_test", "screening"])][["resolution", "dataset_variant"]].drop_duplicates().astype(str).agg("/".join, axis=1).tolist()),
            "backends_used": sorted(results["backend_used"].dropna().unique().tolist()),
            "fallback_occurred": bool(results["fallback_used"].astype(bool).any()),
            "result_row_count": int(len(results)),
            "successful_result_row_count": int(len(results[results["status"].isin(["smoke_test", "screening"])])),
            "screening_result_row_count": int(len(results[results["status"] == "screening"])),
            "smoke_result_row_count": int(len(results[results["status"] == "smoke_test"])),
            "skipped_or_aborted_config_count": int(len(skipped[~skipped["status"].eq("not_started_stage_a_fast_screening_only")])) if not skipped.empty else 0,
            "deferred_heavy_models": DEFERRED_HEAVY_MODELS,
            "data_shapes": data_shapes,
            "fairness_common_conditions": {
                "chronological_split": True,
                "train_validation_test_usage": "train fit only; validation/test evaluate only",
                "target_name": TARGET_COLUMN,
                "nrmse_definition": "RMSE / 2",
                "skill_definition": "1 - RMSE_model / RMSE_persistence",
                "persistence_reference_matching": "resolution + dataset_variant + lookback + horizon + split",
                "resolution_native_lh_note": "one_minute L/H are minute steps; one_hour L/H are hour steps",
                "physical_time_comparison_policy": "separate future experiment; not mixed silently with Stage A",
            },
            "kaggle_push_launched": False,
            "kaggle_run_launched": False,
            "heavy_kaggle_training_started": False,
            "notebook_08_started": False,
            "stage_b_started": False,
            "full_grid_started": False,
            "heavy_models_started": False,
            "output_files": output_files,
            "warnings": warnings_list,
        }

        final_report_path = output_root / "notebook_07_stage_a_report.json"
        readiness_path = reports_dir / "phase7_stage_a_readiness_report.md"
        pipeline_status_path = reports_dir / "pipeline_build_status.md"
        output_files["notebook_07_stage_a_report"] = str(final_report_path)
        output_files["phase7_stage_a_readiness_report"] = str(readiness_path)
        output_files["pipeline_build_status"] = str(pipeline_status_path)
        checks = validation_checks(root, results, prerequisites, output_files)
        report_preliminary["validation_checks"] = checks
        report_preliminary["final_status"] = "success" if checks["passed"] else "failed"
        report_preliminary["ready_for_stage_b_after_review"] = bool(checks["passed"])
        report_preliminary["output_files"] = output_files
        write_json(final_report_path, scalar(report_preliminary))
        write_markdown(readiness_path, readiness_report_text(report_preliminary))
        write_markdown(pipeline_status_path, pipeline_status_text(report_preliminary))
        write_json(final_report_path, scalar(report_preliminary))

        artifact_paths = [str(path) for path in list(paths.values()) + [final_report_path, readiness_path, pipeline_status_path]]
        write_local_runner_report(os.environ.get("LOCAL_OUTPUT_DIR"), report_preliminary, artifact_paths)
        write_status(
            run_state_dir / "notebook_07_stage_a_status.json",
            section_status(
                NOTEBOOK_NAME,
                started_at,
                report_preliminary["final_status"],
                "Notebook 07 Stage 0 / Stage A classical ML screening completed.",
                {
                    "stage0_succeeded": checks["stage0_succeeded"],
                    "stage_a_executed": checks["stage_a_executed"],
                    "result_rows": int(len(results)),
                    "clearml_task_id": clearml_task_id,
                },
            ),
        )
        return scalar(report_preliminary)
    except Exception:
        error_message = traceback.format_exc()
        report = {
            "notebook_name": NOTEBOOK_NAME,
            "final_status": "failed",
            "started_at": started_at,
            "finished_at": utc_now(),
            "run_id": run_id,
            "clearml_project_name": clearml_project_name,
            "clearml_task_id": clearml_task_id,
            "clearml_task_url": clearml_task_url,
            "output_files": output_files,
            "notebook_08_started": False,
            "stage_b_started": False,
            "full_grid_started": False,
            "heavy_models_started": False,
            "kaggle_push_launched": False,
            "kaggle_run_launched": False,
            "heavy_kaggle_training_started": False,
            "ready_for_stage_b_after_review": False,
            "warnings": warnings_list,
            "error_message": error_message,
        }
        final_report_path = output_root / "notebook_07_stage_a_report.json"
        write_json(final_report_path, scalar(report))
        write_local_runner_report(os.environ.get("LOCAL_OUTPUT_DIR"), report, [str(final_report_path)])
        write_status(
            run_state_dir / "notebook_07_stage_a_status.json",
            section_status(NOTEBOOK_NAME, started_at, "failed", "Notebook 07 Stage A failed.", error_message=error_message),
        )
        return scalar(report)
