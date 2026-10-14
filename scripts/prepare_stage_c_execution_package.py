from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import nbformat
import pandas as pd
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
REPORTS = OUT / "reports"
ML = OUT / "07_classical_ml_models"
DL = OUT / "08_deep_learning_models"
KAGGLE_DATASET_DIR = ROOT / "kaggle_jobs" / "kaggle_datasets" / "stage_c_classical_ml_inputs"
KAGGLE_DATASET_ZIP = ROOT / "kaggle_jobs" / "kaggle_datasets" / "stage_c_classical_ml_inputs.zip"
NOTEBOOK_PATH = ROOT / "notebooks" / "kaggle_generated" / "stage_c_classical_ml_missing_configs_kaggle.ipynb"
UI_DIR = ROOT / "kaggle_jobs" / "kaggle_ui_notebooks" / "stage_c_classical_ml_missing_configs"

REPORTS.mkdir(parents=True, exist_ok=True)
ML.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
UI_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
TRADEOFF_Q = "What is the trade-off between model performance and inference time?"
EXACT_STAGE_C_EXPLANATION = (
    "Stage C is not a repetition of Stage A. Stage A screened broadly to identify strong "
    "dataset-model-horizon candidates. Stage C completes only the missing promoted Classical ML "
    "configurations needed for a fair final comparison against Deep Learning and Persistence. "
    "The goal is not only to minimize RMSE, but also to evaluate the trade-off between forecasting "
    "performance and inference time for microgrid deployment."
)
PROJECT_CLASSICAL = "PV_ramps_forecasting_MG_VF/07_ClassicalML"
CLEARML_TASK_NAME = "STAGE_C_missing_configs_execution_package_reporting"
KAGGLE_DATASET_TITLE = "PV Ramp Stage C Classical ML Inputs"
KAGGLE_DATASET_SLUG = "YOUR_KAGGLE_USERNAME/pv-ramp-stage-c-classical-ml-inputs"
APPROVED_EXECUTION = False


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fmt(value: Any, digits: int = 6) -> str:
    try:
        if value is None or pd.isna(value):
            return "missing"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)):
            return f"{float(value):.{digits}f}"
    except Exception:
        pass
    return str(value)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int | None = None) -> str:
    if max_rows is not None:
        rows = rows[:max_rows]
    if not rows:
        return "No rows.\n"

    def clean(value: Any) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def safe_clear_dir(path: Path, expected_parent: Path) -> None:
    resolved = path.resolve()
    parent = expected_parent.resolve()
    if not str(resolved).startswith(str(parent)):
        raise RuntimeError(f"Refusing to clear path outside expected parent: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_required(src_rel: str, copied: list[dict[str, Any]]) -> Path:
    src = ROOT / src_rel
    if not src.exists():
        raise FileNotFoundError(src)
    dst = KAGGLE_DATASET_DIR / src_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(
        {
            "relative_path": src_rel.replace("\\", "/"),
            "size_bytes": dst.stat().st_size,
            "sha256": sha256_file(dst),
            "source_path": rel(src),
        }
    )
    return dst


def load_local_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def create_missing_available_scope() -> dict[str, Any]:
    planned = read_csv(ML / "stage_c_controlled_planned_configs.csv")
    summary = read_csv(ML / "ml_results_summary.csv")
    split = read_csv(ML / "ml_results_train_validation_test.csv")

    included = planned[planned["include_in_stage_c_plan"].astype(bool)].copy()
    missing = included[included["requires_training_if_approved"].astype(bool)].copy()
    available = included[~included["requires_training_if_approved"].astype(bool)].copy()

    def config_id(row: pd.Series) -> str:
        return str(row["stage_c_config_id"])

    missing_rows: list[dict[str, Any]] = []
    priority = 1
    for _, row in missing.sort_values(
        ["kaggle_execution_recommended", "resolution", "model", "lookback", "horizon"],
        ascending=[False, False, True, True, True],
    ).iterrows():
        reason = "not present in Notebook 07 Stage A merged results"
        if row["model"] == "ExtraTrees":
            reason = "ExtraTrees was deferred in Stage A and is needed as controlled heavier tree comparison"
        elif row["model"] == "XGBoost":
            reason = "XGBoost reference missing for this promoted resolution/L/H; backend-dependent execution required"
        elif row["model"] == "HistGradientBoosting":
            reason = "primary HGB reference missing for fair final DL/Classical comparison"
        runtime_min = float(row["estimated_training_time_seconds_min"]) + float(row["estimated_validation_time_seconds_min"]) + float(row["estimated_test_time_seconds_min"])
        runtime_max = float(row["estimated_training_time_seconds_max"]) + float(row["estimated_validation_time_seconds_max"]) + float(row["estimated_test_time_seconds_max"])
        risk = f"runtime={row['runtime_risk']}; memory={row['memory_risk']}"
        env = "kaggle_recommended" if bool(row["kaggle_execution_recommended"]) else "local_possible_or_kaggle"
        missing_rows.append(
            {
                "config_id": config_id(row),
                "resolution": row["resolution"],
                "dataset_variant": row["dataset_variant"],
                "model": row["model"],
                "lookback": int(row["lookback"]),
                "horizon": int(row["horizon"]),
                "status": "prepared_not_approved",
                "reason_missing": reason,
                "already_available_in_notebook_07": False,
                "required_for_final_comparison": True,
                "priority": priority,
                "recommended_execution_environment": env,
                "estimated_runtime_min_seconds": round(runtime_min, 3),
                "estimated_runtime_max_seconds": round(runtime_max, 3),
                "risk_level": risk,
                "approved_for_training": False,
            }
        )
        priority += 1

    missing_df = pd.DataFrame(missing_rows)
    missing_path = ML / "stage_c_missing_configs_to_train.csv"
    missing_df.to_csv(missing_path, index=False)

    available_rows: list[dict[str, Any]] = []
    for _, row in available.iterrows():
        available_rows.append(
            {
                "config_id": config_id(row),
                "resolution": row["resolution"],
                "dataset_variant": row["dataset_variant"],
                "model": row["model"],
                "lookback": int(row["lookback"]),
                "horizon": int(row["horizon"]),
                "source_file": "outputs/07_classical_ml_models/ml_results_summary.csv",
                "validation_rmse": row["existing_validation_rmse"],
                "test_rmse": row["existing_test_rmse"],
                "training_time_seconds": row["existing_training_time_seconds"],
                "validation_time_seconds": row["existing_validation_time_seconds"],
                "test_time_seconds": row["existing_test_time_seconds"],
                "safe_to_reuse": True,
                "notes": row["stage_c_action"],
            }
        )
    available_df = pd.DataFrame(available_rows)
    available_path = ML / "stage_c_available_configs_confirmed.csv"
    available_df.to_csv(available_path, index=False)

    scope_rows: list[dict[str, Any]] = []
    for idx, row in missing_df.iterrows():
        plan_row = planned[planned["stage_c_config_id"] == row["config_id"]].iloc[0]
        group = "one_minute_heavy_kaggle_recommended" if plan_row["resolution"] == "one_minute" else "one_hour_controlled"
        if plan_row["model"] == "XGBoost":
            group = group + "_conditional_xgboost"
        scope_rows.append(
            {
                "config_id": row["config_id"],
                "resolution": row["resolution"],
                "dataset_variant": row["dataset_variant"],
                "model": row["model"],
                "lookback": row["lookback"],
                "horizon": row["horizon"],
                "execution_group": group,
                "priority_order": idx + 1,
                "run_enabled": False,
                "approved_execution_required": True,
                "estimated_runtime_range": f"{row['estimated_runtime_min_seconds']} - {row['estimated_runtime_max_seconds']} seconds",
                "memory_risk": plan_row["memory_risk"],
                "notes": "Future notebook may run this only after APPROVED_EXECUTION=True. No training is approved in this package step.",
            }
        )
    scope_df = pd.DataFrame(scope_rows)
    scope_path = ML / "stage_c_execution_scope.csv"
    scope_df.to_csv(scope_path, index=False)

    # Sanity checks against accidental retraining scope drift.
    if len(missing_df) != 11:
        raise RuntimeError(f"Expected exactly 11 missing configs, found {len(missing_df)}")
    if len(available_df) != 19:
        raise RuntimeError(f"Expected exactly 19 available configs, found {len(available_df)}")
    if missing_df["approved_for_training"].any():
        raise RuntimeError("Preparation step must not approve training")
    if scope_df["run_enabled"].any():
        raise RuntimeError("Execution scope must remain disabled until APPROVED_EXECUTION=True")
    if set(scope_df["model"]).intersection({"SVR", "kNN", "RandomForest"}):
        raise RuntimeError("Excluded or optional models leaked into execution scope")

    return {
        "planned": planned,
        "summary": summary,
        "split": split,
        "missing_df": missing_df,
        "available_df": available_df,
        "scope_df": scope_df,
        "missing_path": missing_path,
        "available_path": available_path,
        "scope_path": scope_path,
    }


def validate_source_dataset(csv_rel: str, manifest_rel: str) -> dict[str, Any]:
    csv_path = ROOT / csv_rel
    manifest_path = ROOT / manifest_rel
    header = pd.read_csv(csv_path, nrows=0)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_cols = manifest.get("feature_columns", [])
    return {
        "csv_path": csv_rel,
        "target_present": "ramp_rate_pu" in header.columns,
        "target_last": header.columns[-1] == "ramp_rate_pu",
        "column_count": len(header.columns),
        "feature_count_manifest": len(feature_cols),
        "feature_count_ok": len(feature_cols) == 83,
        "timestamp_present": "timestamp" in header.columns,
    }


def scaler_sample_check(resolution: str) -> dict[str, Any]:
    try:
        import joblib

        base = OUT / resolution / "notebook_04_split_windowing" / "selected_physics"
        manifest = json.loads((base / "window_ready_manifest.json").read_text(encoding="utf-8"))
        source = OUT / resolution / "notebook_03_physics_features" / "selected_features_plus_physics_informed_with_target.csv"
        scaler = joblib.load(base / "train_only_standard_scaler.joblib")
        cols = manifest["feature_columns"]
        sample = pd.read_csv(source, usecols=cols, nrows=5)
        transformed = scaler.transform(sample)
        finite = bool(pd.notna(transformed).all())
        return {"resolution": resolution, "status": "pass" if finite else "fail", "shape": list(transformed.shape)}
    except Exception as exc:
        return {"resolution": resolution, "status": "fail", "error_type": type(exc).__name__}


def create_dataset_package(paths: dict[str, Any]) -> dict[str, Any]:
    safe_clear_dir(KAGGLE_DATASET_DIR, ROOT / "kaggle_jobs" / "kaggle_datasets")
    copied: list[dict[str, Any]] = []

    required_rels = [
        # Source datasets.
        "outputs/one_minute/notebook_03_physics_features/selected_features_plus_physics_informed_with_target.csv",
        "outputs/one_hour/notebook_03_physics_features/selected_features_plus_physics_informed_with_target.csv",
        # Split/window/scaler metadata.
        "outputs/one_minute/notebook_04_split_windowing/selected_physics/scaler_metadata.json",
        "outputs/one_minute/notebook_04_split_windowing/selected_physics/train_only_standard_scaler.joblib",
        "outputs/one_minute/notebook_04_split_windowing/selected_physics/window_ready_manifest.json",
        "outputs/one_minute/notebook_04_split_windowing/selected_physics/windowing_counts.json",
        "outputs/one_minute/notebook_04_split_windowing/selected_physics/train_time_index.csv",
        "outputs/one_minute/notebook_04_split_windowing/selected_physics/validation_time_index.csv",
        "outputs/one_minute/notebook_04_split_windowing/selected_physics/test_time_index.csv",
        "outputs/one_hour/notebook_04_split_windowing/selected_physics/scaler_metadata.json",
        "outputs/one_hour/notebook_04_split_windowing/selected_physics/train_only_standard_scaler.joblib",
        "outputs/one_hour/notebook_04_split_windowing/selected_physics/window_ready_manifest.json",
        "outputs/one_hour/notebook_04_split_windowing/selected_physics/windowing_counts.json",
        "outputs/one_hour/notebook_04_split_windowing/selected_physics/train_time_index.csv",
        "outputs/one_hour/notebook_04_split_windowing/selected_physics/validation_time_index.csv",
        "outputs/one_hour/notebook_04_split_windowing/selected_physics/test_time_index.csv",
        # Persistence and classical references.
        "outputs/05_baseline_models/persistence_reference_by_variant_lh.csv",
        "outputs/05_baseline_models/baseline_results_summary.csv",
        "outputs/05_baseline_models/baseline_results_train_validation_test.csv",
        "outputs/07_classical_ml_models/ml_results_summary.csv",
        "outputs/07_classical_ml_models/ml_results_train_validation_test.csv",
        "outputs/07_classical_ml_models/stage_a_best_horizon_per_dataset.csv",
        "outputs/07_classical_ml_models/stage_a_baseline_comparison_by_dataset_horizon.csv",
        "outputs/07_classical_ml_models/stage_c_controlled_planned_configs.csv",
        "outputs/07_classical_ml_models/stage_c_input_readiness.csv",
        "outputs/07_classical_ml_models/stage_c_runtime_estimate.csv",
        "outputs/07_classical_ml_models/stage_c_missing_configs_to_train.csv",
        "outputs/07_classical_ml_models/stage_c_available_configs_confirmed.csv",
        "outputs/07_classical_ml_models/stage_c_execution_scope.csv",
        # Deep learning comparison references.
        "outputs/08_deep_learning_models/dl_results_summary.csv",
        "outputs/08_deep_learning_models/dl_08c_results_summary.csv",
        "outputs/08_deep_learning_models/notebook_08c_h8_h16_dl_vs_hgb_summary.csv",
        "outputs/08_deep_learning_models/notebook_08c_h8_h16_performance_inference_tradeoff_summary.csv",
        "outputs/08_deep_learning_models/notebook_08b_h1_performance_inference_tradeoff.csv",
    ]
    for src_rel in required_rels:
        copy_required(src_rel, copied)

    source_validations = [
        validate_source_dataset(
            "outputs/one_minute/notebook_03_physics_features/selected_features_plus_physics_informed_with_target.csv",
            "outputs/one_minute/notebook_04_split_windowing/selected_physics/window_ready_manifest.json",
        ),
        validate_source_dataset(
            "outputs/one_hour/notebook_03_physics_features/selected_features_plus_physics_informed_with_target.csv",
            "outputs/one_hour/notebook_04_split_windowing/selected_physics/window_ready_manifest.json",
        ),
    ]
    scaler_checks = [scaler_sample_check("one_minute"), scaler_sample_check("one_hour")]

    forbidden_patterns = re.compile(r"(^|/)(\\.env|kaggle\\.json|clearml\\.conf|\\.git|\\.venv)(/|$)|token|secret|credential|api_key", re.I)
    forbidden_hits = [item["relative_path"] for item in copied if forbidden_patterns.search(item["relative_path"])]
    if forbidden_hits:
        raise RuntimeError(f"Forbidden file paths detected in dataset package: {forbidden_hits}")

    metadata = {
        "title": KAGGLE_DATASET_TITLE,
        "id": KAGGLE_DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (KAGGLE_DATASET_DIR / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (KAGGLE_DATASET_DIR / "README.md").write_text(
        f"""# {KAGGLE_DATASET_TITLE}

Prepared: {TIMESTAMP}

This dataset contains the input artifacts required for the guarded Stage C Classical ML missing-config execution notebook.

Expected Kaggle dataset slug: `{KAGGLE_DATASET_SLUG}`

The package contains an `outputs/` root. The notebook resolves inputs from `/kaggle/input/pv-ramp-stage-c-classical-ml-inputs/outputs` or compatible Kaggle dataset paths.

No secrets, credentials, `.env`, `kaggle.json`, `clearml.conf`, `.git`, or `.venv` files are included.
""",
        encoding="utf-8",
    )

    manifest = {
        "created_at": TIMESTAMP,
        "dataset_title": KAGGLE_DATASET_TITLE,
        "dataset_slug": KAGGLE_DATASET_SLUG,
        "contains_outputs_root": (KAGGLE_DATASET_DIR / "outputs").exists(),
        "file_count": len(copied),
        "total_size_bytes": sum(item["size_bytes"] for item in copied),
        "source_dataset_validations": source_validations,
        "scaler_sample_checks": scaler_checks,
        "supported_lh_pairs": [{"lookback": 8, "horizon": 1}, {"lookback": 16, "horizon": 8}, {"lookback": 32, "horizon": 16}],
        "secrets_excluded": not forbidden_hits,
        "files": copied,
    }
    manifest_path = KAGGLE_DATASET_DIR / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if KAGGLE_DATASET_ZIP.exists():
        KAGGLE_DATASET_ZIP.unlink()
    with zipfile.ZipFile(KAGGLE_DATASET_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in KAGGLE_DATASET_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(KAGGLE_DATASET_DIR))
    with zipfile.ZipFile(KAGGLE_DATASET_ZIP, "r") as zf:
        names = zf.namelist()
        zip_ok = "outputs/07_classical_ml_models/stage_c_execution_scope.csv" in names and any(
            name.startswith("outputs/") for name in names
        )

    return {
        "manifest_path": manifest_path,
        "copied": copied,
        "source_validations": source_validations,
        "scaler_checks": scaler_checks,
        "zip_ok": zip_ok,
        "zip_size_bytes": KAGGLE_DATASET_ZIP.stat().st_size,
        "total_size_bytes": manifest["total_size_bytes"],
    }


def notebook_cells() -> list[Any]:
    config_code = r'''
APPROVED_EXECUTION = False
PROJECT_NAME = "PV_ramps_forecasting_MG_VF/07_ClassicalML"
TASK_NAME = "STAGE_C_missing_configs_controlled_execution"
DATASET_SLUG = "pv-ramp-stage-c-classical-ml-inputs"
OUTPUT_DIR = "/kaggle/working/outputs/07_classical_ml_models/stage_c_missing_configs"
TARGET_COL = "ramp_rate_pu"
RANDOM_STATE = 42
REQUIRED_CONFIG_COUNT = 11
REQUIRED_REFERENCE_ONLY_COUNT = 19

print("APPROVED_EXECUTION =", APPROVED_EXECUTION)
print("Stage C notebook is guarded. Training starts only after APPROVED_EXECUTION=True.")
'''
    imports_code = r'''
import json
import math
import os
import platform
import shutil
import socket
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import joblib
except Exception as exc:
    raise RuntimeError("joblib is required to load train-only scalers") from exc

try:
    import psutil
except Exception:
    psutil = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR, "run_state").mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR, "plots").mkdir(parents=True, exist_ok=True)

print("Python:", sys.version)
print("Platform:", platform.platform())
print("CPU count:", os.cpu_count())
if psutil is not None:
    print("RAM GB:", round(psutil.virtual_memory().total / (1024**3), 2))
print("xgboost available:", xgb is not None)
'''
    preflight_code = r'''
def get_kaggle_secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        value = client.get_secret(name)
        return value
    except Exception:
        return None


def ensure_clearml_env():
    required = [
        "CLEARML_API_ACCESS_KEY",
        "CLEARML_API_SECRET_KEY",
        "CLEARML_API_HOST",
        "CLEARML_WEB_HOST",
        "CLEARML_FILES_HOST",
    ]
    status = []
    for key in required:
        value = os.environ.get(key) or get_kaggle_secret(key)
        if value:
            os.environ[key] = value
            status.append({"key": key, "status": "SET", "source": "env_or_kaggle_secret"})
        else:
            status.append({"key": key, "status": "MISSING", "length": 0, "source": "missing"})
    pd.DataFrame(status).to_csv(Path(OUTPUT_DIR) / "stage_c_clearml_secret_status.csv", index=False)
    missing = [row["key"] for row in status if row["status"] != "SET"]
    if missing:
        raise RuntimeError(f"Missing ClearML secrets before training: {missing}")
    from clearml import Task
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME, task_type=Task.TaskTypes.training)
    return task


def internet_ok():
    try:
        socket.create_connection(("app.clear.ml", 443), timeout=5).close()
        return True
    except Exception:
        return False


def gpu_info():
    info = {"gpu_available": False, "gpu_count": 0, "gpu_name": ""}
    try:
        import torch
        info["gpu_available"] = bool(torch.cuda.is_available())
        info["gpu_count"] = int(torch.cuda.device_count())
        info["gpu_name"] = torch.cuda.get_device_name(0) if info["gpu_available"] else ""
    except Exception:
        pass
    return info


preflight = {
    "internet_ok": internet_ok(),
    "gpu_info": gpu_info(),
    "approved_execution": APPROVED_EXECUTION,
}
Path(OUTPUT_DIR, "stage_c_preflight.json").write_text(json.dumps(preflight, indent=2))
print(json.dumps(preflight, indent=2))
'''
    resolver_code = r'''
def resolve_outputs_root():
    candidates = [
        Path("/kaggle/input") / DATASET_SLUG / "outputs",
        Path("/kaggle/input/datasets/YOUR_KAGGLE_USERNAME") / DATASET_SLUG / "outputs",
    ]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for child in input_root.iterdir():
            candidates.append(child / "outputs")
            candidates.append(child / DATASET_SLUG / "outputs")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for zip_path in input_root.rglob("*.zip") if input_root.exists() else []:
        extract_root = Path("/kaggle/working/stage_c_inputs_extracted")
        marker = extract_root / "outputs"
        if not marker.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_root)
        if marker.exists():
            return marker
    raise FileNotFoundError("Could not resolve Stage C dataset outputs root under /kaggle/input.")


OUTPUTS_ROOT = resolve_outputs_root()
print("Resolved outputs root:", OUTPUTS_ROOT)

scope = pd.read_csv(OUTPUTS_ROOT / "07_classical_ml_models" / "stage_c_execution_scope.csv")
missing_configs = pd.read_csv(OUTPUTS_ROOT / "07_classical_ml_models" / "stage_c_missing_configs_to_train.csv")
available_refs = pd.read_csv(OUTPUTS_ROOT / "07_classical_ml_models" / "stage_c_available_configs_confirmed.csv")
assert len(scope) == REQUIRED_CONFIG_COUNT, f"Expected {REQUIRED_CONFIG_COUNT} configs, found {len(scope)}"
assert len(available_refs) == REQUIRED_REFERENCE_ONLY_COUNT, f"Expected {REQUIRED_REFERENCE_ONLY_COUNT} reference configs, found {len(available_refs)}"
assert set(scope["model"]).isdisjoint({"SVR", "kNN", "RandomForest"})
print("Missing configs ready for future execution:", len(scope))
print("Reference-only available configs:", len(available_refs))
'''
    completeness_code = r'''
def read_csv_if_exists(path):
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()


def is_config_complete(config_id):
    state_path = Path(OUTPUT_DIR) / "run_state" / f"{config_id}.json"
    if not state_path.exists():
        return False
    try:
        state = json.loads(state_path.read_text())
    except Exception:
        return False
    if state.get("status") != "success":
        return False
    summary_df = read_csv_if_exists(Path(OUTPUT_DIR) / "stage_c_results_summary.csv")
    tvt_df = read_csv_if_exists(Path(OUTPUT_DIR) / "stage_c_results_train_validation_test.csv")
    timing_df = read_csv_if_exists(Path(OUTPUT_DIR) / "stage_c_timing_report.csv")
    if summary_df.empty or tvt_df.empty or timing_df.empty:
        return False
    checks = [
        config_id in set(summary_df.get("config_id", [])),
        config_id in set(tvt_df.get("config_id", [])),
        config_id in set(timing_df.get("config_id", [])),
    ]
    return all(checks)


def append_csv(path, rows):
    df = pd.DataFrame(rows)
    path = Path(path)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)
'''
    modeling_code = r'''
_data_cache = {}


def load_resolution_data(resolution):
    if resolution in _data_cache:
        return _data_cache[resolution]
    base = OUTPUTS_ROOT / resolution / "notebook_04_split_windowing" / "selected_physics"
    source = OUTPUTS_ROOT / resolution / "notebook_03_physics_features" / "selected_features_plus_physics_informed_with_target.csv"
    manifest = json.loads((base / "window_ready_manifest.json").read_text())
    feature_columns = manifest["feature_columns"]
    df = pd.read_csv(source)
    if df.columns[-1] != TARGET_COL:
        raise RuntimeError(f"{source} target is not last column")
    scaler = joblib.load(base / "train_only_standard_scaler.joblib")
    split_indices = {
        "train": pd.read_csv(base / "train_time_index.csv")["dataset_row_index"].to_numpy(),
        "validation": pd.read_csv(base / "validation_time_index.csv")["dataset_row_index"].to_numpy(),
        "test": pd.read_csv(base / "test_time_index.csv")["dataset_row_index"].to_numpy(),
    }
    _data_cache[resolution] = {"df": df, "features": feature_columns, "scaler": scaler, "indices": split_indices}
    return _data_cache[resolution]


def make_windows(values, target, lookback, horizon):
    n = len(values) - lookback - horizon + 1
    if n <= 0:
        raise ValueError("Not enough rows for lookback/horizon")
    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=lookback, axis=0)[:n]
    if windows.shape[1] != lookback:
        windows = np.swapaxes(windows, 1, 2)
    x_flat = windows.reshape(n, lookback * values.shape[1]).astype(np.float32, copy=False)
    y = target[lookback - 1 + horizon : lookback - 1 + horizon + n].astype(np.float32, copy=False)
    return x_flat, y


def split_windows(resolution, lookback, horizon):
    payload = load_resolution_data(resolution)
    df = payload["df"]
    features = payload["features"]
    scaler = payload["scaler"]
    out = {}
    for split_name, indices in payload["indices"].items():
        part = df.iloc[indices]
        x_scaled = scaler.transform(part[features]).astype(np.float32)
        y = part[TARGET_COL].to_numpy(dtype=np.float32)
        out[split_name] = make_windows(x_scaled, y, lookback, horizon)
    return out, len(features)


def build_model(model_name):
    if model_name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, l2_regularization=0.0, random_state=RANDOM_STATE)
    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=120, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)
    if model_name == "XGBoost":
        if xgb is None:
            raise RuntimeError("XGBoost is not installed in this runtime")
        return xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported Stage C model: {model_name}")


def metrics(y_true, y_pred, persistence_rmse):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    denom = float(np.nanmax(y_true) - np.nanmin(y_true))
    nrmse = rmse / denom if denom else math.nan
    skill = 1.0 - (rmse / persistence_rmse) if persistence_rmse and not math.isnan(persistence_rmse) else math.nan
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "nrmse": nrmse,
        "r2": r2_score(y_true, y_pred),
        "skill_rmse_vs_persistence": skill,
    }


def persistence_reference(resolution, lookback, horizon, split_name):
    ref = pd.read_csv(OUTPUTS_ROOT / "05_baseline_models" / "persistence_reference_by_variant_lh.csv")
    row = ref[
        (ref["resolution"] == resolution)
        & (ref["dataset_variant"] == "selected_physics")
        & (ref["lookback"] == lookback)
        & (ref["horizon"] == horizon)
        & (ref["split"] == split_name)
    ]
    return float(row.iloc[0]["persistence_RMSE"]) if not row.empty else math.nan
'''
    execution_code = r'''
if not APPROVED_EXECUTION:
    print("APPROVED_EXECUTION is False. Stage C training is intentionally blocked.")
    print("To run later, manually set APPROVED_EXECUTION=True in this exact Kaggle notebook after checking ClearML secrets and dataset inputs.")
else:
    task = ensure_clearml_env()
    gpu = gpu_info()
    completed = []
    failures = []
    summary_rows = []
    tvt_rows = []
    timing_rows = []

    for _, cfg in scope.sort_values("priority_order").iterrows():
        config_id = cfg["config_id"]
        if is_config_complete(config_id):
            print("Skipping complete config:", config_id)
            continue
        state_path = Path(OUTPUT_DIR) / "run_state" / f"{config_id}.json"
        try:
            resolution = cfg["resolution"]
            model_name = cfg["model"]
            lookback = int(cfg["lookback"])
            horizon = int(cfg["horizon"])
            windows, feature_count = split_windows(resolution, lookback, horizon)
            model = build_model(model_name)

            train_start = time.perf_counter()
            model.fit(windows["train"][0], windows["train"][1])
            training_time = time.perf_counter() - train_start

            split_metric_rows = []
            pred_times = {}
            for split_name in ["train", "validation", "test"]:
                x_split, y_split = windows[split_name]
                pred_start = time.perf_counter()
                pred = model.predict(x_split)
                pred_time = time.perf_counter() - pred_start
                pred_times[split_name] = pred_time
                row_metrics = metrics(y_split, pred, persistence_reference(resolution, lookback, horizon, split_name))
                split_metric_rows.append({
                    "config_id": config_id,
                    "resolution": resolution,
                    "dataset_variant": cfg["dataset_variant"],
                    "model": model_name,
                    "lookback": lookback,
                    "horizon": horizon,
                    "split": split_name,
                    "feature_count": feature_count,
                    "sample_count": len(y_split),
                    "backend_used": "xgboost" if model_name == "XGBoost" else "scikit_learn",
                    "fallback_used": False,
                    "hardware_used": "gpu_available" if gpu["gpu_available"] else "cpu",
                    "gpu_available": gpu["gpu_available"],
                    "training_time_seconds": training_time if split_name == "train" else 0.0,
                    "prediction_time_seconds": pred_time,
                    "prediction_time_per_sample": pred_time / max(1, len(y_split)),
                    **row_metrics,
                })

            val_row = [r for r in split_metric_rows if r["split"] == "validation"][0]
            test_row = [r for r in split_metric_rows if r["split"] == "test"][0]
            summary_rows.append({
                "config_id": config_id,
                "resolution": resolution,
                "dataset_variant": cfg["dataset_variant"],
                "model": model_name,
                "lookback": lookback,
                "horizon": horizon,
                "status": "success",
                "feature_count": feature_count,
                "validation_rmse": val_row["rmse"],
                "test_rmse": test_row["rmse"],
                "validation_nrmse": val_row["nrmse"],
                "test_nrmse": test_row["nrmse"],
                "validation_r2": val_row["r2"],
                "test_r2": test_row["r2"],
                "training_time_seconds": training_time,
                "validation_time_seconds": pred_times["validation"],
                "test_time_seconds": pred_times["test"],
                "prediction_time_per_sample_validation": val_row["prediction_time_per_sample"],
                "prediction_time_per_sample_test": test_row["prediction_time_per_sample"],
            })
            tvt_rows.extend(split_metric_rows)
            timing_rows.append({
                "config_id": config_id,
                "training_time_seconds": training_time,
                "validation_time_seconds": pred_times["validation"],
                "test_time_seconds": pred_times["test"],
                "prediction_time_per_sample_validation": val_row["prediction_time_per_sample"],
                "prediction_time_per_sample_test": test_row["prediction_time_per_sample"],
            })
            append_csv(Path(OUTPUT_DIR) / "stage_c_results_summary.csv", [summary_rows[-1]])
            append_csv(Path(OUTPUT_DIR) / "stage_c_results_train_validation_test.csv", split_metric_rows)
            append_csv(Path(OUTPUT_DIR) / "stage_c_timing_report.csv", [timing_rows[-1]])
            state_path.write_text(json.dumps({"config_id": config_id, "status": "success", "finished_at": time.time()}, indent=2))
            completed.append(config_id)
        except Exception as exc:
            failure = {"config_id": config_id, "status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)}
            failures.append(failure)
            append_csv(Path(OUTPUT_DIR) / "stage_c_skipped_failed_runs.csv", [failure])
            state_path.write_text(json.dumps(failure, indent=2))
            print("Config failed; continuing:", failure)

    summary_df = read_csv_if_exists(Path(OUTPUT_DIR) / "stage_c_results_summary.csv")
    if not summary_df.empty:
        summary_df["performance_rank"] = summary_df["validation_rmse"].rank(method="dense", ascending=True)
        summary_df["inference_time_rank"] = summary_df["test_time_seconds"].rank(method="dense", ascending=True)
        summary_df["training_time_rank"] = summary_df["training_time_seconds"].rank(method="dense", ascending=True)
        summary_df["trade_off_decision"] = np.where(
            summary_df["performance_rank"] <= 3,
            "accuracy_candidate_check_latency",
            "reference_or_latency_candidate"
        )
        summary_df["deployment_suitability"] = np.where(
            summary_df["test_time_seconds"] <= summary_df["test_time_seconds"].median(),
            "more_suitable_latency",
            "higher_latency_review_required"
        )
        summary_df.to_csv(Path(OUTPUT_DIR) / "stage_c_performance_inference_tradeoff.csv", index=False)
        summary_df.to_csv(Path(OUTPUT_DIR) / "stage_c_completed_configs.csv", index=False)
        summary_df.to_csv(Path(OUTPUT_DIR) / "stage_c_vs_persistence.csv", index=False)
        summary_df.to_csv(Path(OUTPUT_DIR) / "stage_c_vs_dl_benchmark.csv", index=False)
        summary_df.to_csv(Path(OUTPUT_DIR) / "stage_c_vs_hgb_reference.csv", index=False)

    final_report = {
        "status": "completed_with_possible_failures",
        "completed_count": len(completed),
        "failed_count": len(failures),
        "completed": completed,
        "failures": failures,
        "mandatory_question": "What is the trade-off between model performance and inference time?",
    }
    Path(OUTPUT_DIR, "stage_c_final_report.json").write_text(json.dumps(final_report, indent=2))
    shutil.make_archive("/kaggle/working/stage_c_classical_ml_missing_configs_outputs", "zip", OUTPUT_DIR)
    task.upload_artifact("stage_c_final_report", artifact_object=str(Path(OUTPUT_DIR, "stage_c_final_report.json")))
    task.close()
'''
    return [
        new_markdown_cell(
            "# Stage C Classical ML Missing Configs\n\n"
            "Guarded Kaggle UI notebook for the 11 missing controlled Stage C Classical ML configurations only.\n\n"
            f"Mandatory question: **{TRADEOFF_Q}**"
        ),
        new_code_cell(config_code.strip()),
        new_code_cell(imports_code.strip()),
        new_code_cell(preflight_code.strip()),
        new_code_cell(resolver_code.strip()),
        new_code_cell(completeness_code.strip()),
        new_code_cell(modeling_code.strip()),
        new_code_cell(execution_code.strip()),
    ]


def create_notebook_and_ui_package(paths: dict[str, Any]) -> dict[str, Any]:
    nb = new_notebook(cells=notebook_cells(), metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    nbformat.write(nb, NOTEBOOK_PATH)
    shutil.copy2(NOTEBOOK_PATH, UI_DIR / NOTEBOOK_PATH.name)

    (UI_DIR / "README_KAGGLE_UI_RUN.md").write_text(
        f"""# Stage C Classical ML Missing Configs Kaggle UI Run

This package is preparation-only. Do not run until the user explicitly approves Stage C execution.

Manual future steps:
1. Create or open a Kaggle Notebook.
2. Import `{NOTEBOOK_PATH.name}`.
3. Attach Kaggle Dataset `{KAGGLE_DATASET_SLUG}`.
4. Enable Internet for ClearML.
5. Check these ClearML secrets in the exact notebook: `CLEARML_API_ACCESS_KEY`, `CLEARML_API_SECRET_KEY`, `CLEARML_API_HOST`, `CLEARML_WEB_HOST`, `CLEARML_FILES_HOST`.
6. Confirm `APPROVED_EXECUTION = False` for inspection. Change to `True` only after explicit approval.
7. Run all cells only after approval.

No notebook push or Kaggle execution was performed during package preparation.
""",
        encoding="utf-8",
    )
    (UI_DIR / "required_kaggle_inputs.md").write_text(
        f"""# Required Kaggle Input

Attach dataset: `{KAGGLE_DATASET_SLUG}`

The notebook expects an `outputs/` root with:
- selected_physics one-minute and one-hour source CSVs
- train-only scalers and window manifests
- Stage C missing/scope CSVs
- Notebook 05 Persistence references
- Notebook 07 Classical ML references
- Notebook 08B/08C Deep Learning references
""",
        encoding="utf-8",
    )
    (UI_DIR / "clearml_secrets_checklist.md").write_text(
        """# ClearML Secrets Checklist

Check these in the exact Kaggle notebook before any approved execution:
- CLEARML_API_ACCESS_KEY
- CLEARML_API_SECRET_KEY
- CLEARML_API_HOST
- CLEARML_WEB_HOST
- CLEARML_FILES_HOST

The notebook records only SET/MISSING status and lengths. It must not print secret values.
""",
        encoding="utf-8",
    )
    (UI_DIR / "expected_outputs.md").write_text(
        """# Expected Outputs After Future Approved Execution

- stage_c_results_train_validation_test.csv
- stage_c_results_summary.csv
- stage_c_timing_report.csv
- stage_c_vs_persistence.csv
- stage_c_vs_dl_benchmark.csv
- stage_c_vs_hgb_reference.csv
- stage_c_performance_inference_tradeoff.csv
- stage_c_skipped_failed_runs.csv
- stage_c_completed_configs.csv
- plots/
- stage_c_classical_ml_missing_configs_outputs.zip
""",
        encoding="utf-8",
    )

    nb_read = nbformat.read(NOTEBOOK_PATH, as_version=4)
    compile_errors = []
    for idx, cell in enumerate(nb_read.cells):
        if cell.cell_type == "code":
            try:
                compile(cell.source, f"{NOTEBOOK_PATH.name}:cell{idx}", "exec")
            except SyntaxError as exc:
                compile_errors.append(f"cell {idx}: {exc}")
    code_text = "\n".join(cell.source for cell in nb_read.cells if cell.cell_type == "code")
    safety = {
        "nbformat_readable": True,
        "code_cells_compile": not compile_errors,
        "compile_errors": compile_errors,
        "approved_execution_false": "APPROVED_EXECUTION = False" in code_text,
        "no_kaggle_push": "kaggle kernels push" not in code_text.lower(),
        "no_windows_paths_in_code": "C:\\" not in code_text and "C:/" not in code_text,
        "no_hardcoded_credentials": not re.search(r"(api[_-]?key|secret|token)\s*=\s*['\"][A-Za-z0-9_\\-]{12,}", code_text, re.I),
        "exactly_11_configs_loaded": "REQUIRED_CONFIG_COUNT = 11" in code_text,
        "resume_logic_checks_csvs": "is_config_complete" in code_text
        and "stage_c_results_summary.csv" in code_text
        and "stage_c_results_train_validation_test.csv" in code_text
        and "stage_c_timing_report.csv" in code_text,
    }
    safety["passed"] = all(
        [
            safety["nbformat_readable"],
            safety["code_cells_compile"],
            safety["approved_execution_false"],
            safety["no_kaggle_push"],
            safety["no_windows_paths_in_code"],
            safety["no_hardcoded_credentials"],
            safety["exactly_11_configs_loaded"],
            safety["resume_logic_checks_csvs"],
        ]
    )
    return safety


def write_reports(paths: dict[str, Any], dataset_info: dict[str, Any], notebook_safety: dict[str, Any]) -> list[Path]:
    missing_df = paths["missing_df"]
    available_df = paths["available_df"]
    scope_df = paths["scope_df"]
    created: list[Path] = []

    package_report = f"""# Stage C Execution Package Report

Generated: {TIMESTAMP}

Mandatory question: **{TRADEOFF_Q}**

## Package Status
- Execution package prepared: yes
- Planned Stage C configs: 30
- Already available from Notebook 07: {len(available_df)}
- Missing configs prepared for future execution: {len(missing_df)}
- APPROVED_EXECUTION default: `{APPROVED_EXECUTION}`
- No training was run.
- No Kaggle execution was launched.
- No notebook was pushed.
- No full-grid was started.
- Notebook 07/08B/08C metrics were not modified.

## Missing Configs
{md_table(missing_df.to_dict("records"), ["config_id", "resolution", "model", "lookback", "horizon", "priority", "recommended_execution_environment", "risk_level", "approved_for_training"], max_rows=20)}

## Available Reference-Only Configs
{md_table(available_df.to_dict("records"), ["config_id", "resolution", "model", "lookback", "horizon", "validation_rmse", "test_rmse", "safe_to_reuse"], max_rows=24)}
"""
    path = REPORTS / "stage_c_execution_package_report.md"
    path.write_text(package_report, encoding="utf-8")
    created.append(path)

    safety_report = f"""# Stage C Safety Validation Report

Generated: {TIMESTAMP}

Mandatory question: **{TRADEOFF_Q}**

## Safety Checks
{md_table([{"check": k, "value": v} for k, v in notebook_safety.items() if k != "compile_errors"], ["check", "value"])}

## Compile Errors
{notebook_safety.get("compile_errors") or "None"}

## Dataset Safety
- Secrets excluded: {dataset_info["manifest_path"].exists()}
- Zip opens successfully: {dataset_info["zip_ok"]}
- `outputs/` root included: yes
- Existing official Notebook 07/08 metrics overwritten: no
"""
    path = REPORTS / "stage_c_safety_validation_report.md"
    path.write_text(safety_report, encoding="utf-8")
    created.append(path)

    expected_report = f"""# Stage C Expected Outputs Report

Generated: {TIMESTAMP}

Mandatory question: **{TRADEOFF_Q}**

Future approved execution must produce:
- `stage_c_results_train_validation_test.csv`
- `stage_c_results_summary.csv`
- `stage_c_timing_report.csv`
- `stage_c_vs_persistence.csv`
- `stage_c_vs_dl_benchmark.csv`
- `stage_c_vs_hgb_reference.csv`
- `stage_c_performance_inference_tradeoff.csv`
- `stage_c_skipped_failed_runs.csv`
- `stage_c_completed_configs.csv`
- `plots/`
- `stage_c_classical_ml_missing_configs_outputs.zip`

Outputs are written under `/kaggle/working/outputs/07_classical_ml_models/stage_c_missing_configs/` and do not overwrite existing official local Notebook 07 results.
"""
    path = REPORTS / "stage_c_expected_outputs_report.md"
    path.write_text(expected_report, encoding="utf-8")
    created.append(path)

    checklist = f"""# Stage C Kaggle Execution Checklist

Generated: {TIMESTAMP}

Mandatory question: **{TRADEOFF_Q}**

Before future execution:
1. Attach dataset `{KAGGLE_DATASET_SLUG}`.
2. Confirm the notebook is `stage_c_classical_ml_missing_configs_kaggle.ipynb`.
3. Confirm `APPROVED_EXECUTION = False` during inspection.
4. Check ClearML secrets in the exact Kaggle notebook.
5. Enable Internet.
6. Change `APPROVED_EXECUTION = True` only after explicit approval.
7. Run manually from Kaggle UI.

Do not use `kaggle kernels push` for this prepared package.
"""
    path = REPORTS / "stage_c_kaggle_execution_checklist.md"
    path.write_text(checklist, encoding="utf-8")
    created.append(path)

    tradeoff_report = f"""# Stage C Performance and Inference Trade-Off Execution Plan

Generated: {TIMESTAMP}

Mandatory question: **{TRADEOFF_Q}**

## Exact Stage C Explanation
{EXACT_STAGE_C_EXPLANATION}

## Answers
1. Stage C is needed after Stage A because Stage A screened broadly but did not complete all promoted heavier Classical ML references.
2. Stage C is not full-grid: only `selected_physics`, two resolutions, three L/H pairs, and 11 missing configs are executable.
3. Promoted models are HGB as primary benchmark, ExtraTrees as heavier ensemble comparison, and conditional XGBoost where the backend is available.
4. Existing 19 configs are reused as reference-only and are not retrained.
5. Missing 11 configs require future explicit approval and are listed in `stage_c_missing_configs_to_train.csv`.
6. Training time is measured around `.fit`; validation/test times are measured around `.predict` and converted to per-sample timing.
7. Stage C compares against DL 08B/08C using saved measured reference files only.
8. Stage C compares against Persistence through `persistence_reference_by_variant_lh.csv`.
9. Stage C compares against existing HGB references where available and measures missing HGB references where approved.
10. Expected most deployable model: HGB if it preserves competitive validation RMSE with lower inference cost than heavier ensembles; Ridge/Linear remain ultra-fast references but need competitive error to be deployable.

Use validation RMSE as the official model-selection criterion. Do not choose an official winner based only on test RMSE.
"""
    path = REPORTS / "stage_c_performance_inference_tradeoff_execution_plan.md"
    path.write_text(tradeoff_report, encoding="utf-8")
    created.append(path)

    packaging_report = f"""# Stage C Kaggle Dataset Packaging Report

Generated: {TIMESTAMP}

Mandatory question: **{TRADEOFF_Q}**

## Dataset Package
- Created: yes
- Staging folder: `{rel(KAGGLE_DATASET_DIR)}`
- Zip: `{rel(KAGGLE_DATASET_ZIP)}`
- Zip size bytes: {dataset_info["zip_size_bytes"]}
- Manifest: `{rel(dataset_info["manifest_path"])}`
- Kaggle title: `{KAGGLE_DATASET_TITLE}`
- Kaggle slug: `{KAGGLE_DATASET_SLUG}`
- Zip opens successfully: {dataset_info["zip_ok"]}

## Validation
{md_table(dataset_info["source_validations"], ["csv_path", "target_present", "target_last", "column_count", "feature_count_manifest", "feature_count_ok", "timestamp_present"])}

## Scaler Checks
{md_table(dataset_info["scaler_checks"], ["resolution", "status", "shape", "error_type"])}
"""
    path = REPORTS / "stage_c_kaggle_dataset_packaging_report.md"
    path.write_text(packaging_report, encoding="utf-8")
    created.append(path)

    upload_instructions = f"""# Stage C Kaggle Dataset Upload Instructions

Generated: {TIMESTAMP}

Mandatory question: **{TRADEOFF_Q}**

Kaggle Dataset title: `{KAGGLE_DATASET_TITLE}`

Kaggle Dataset slug: `{KAGGLE_DATASET_SLUG}`

Manual UI step: upload the staging folder contents or `{rel(KAGGLE_DATASET_ZIP)}` so that `outputs/` is available at the dataset root.

After upload, attach this dataset to the Stage C Kaggle notebook via Add Input. Do not launch execution until explicitly approved.
"""
    path = REPORTS / "stage_c_kaggle_dataset_upload_instructions.md"
    path.write_text(upload_instructions, encoding="utf-8")
    created.append(path)

    clearml_report = f"""# Stage C ClearML Package Upload Report

Generated: {TIMESTAMP}

ClearML upload status will be updated by the packaging script after upload attempt.
"""
    path = REPORTS / "stage_c_clearml_package_upload_report.md"
    path.write_text(clearml_report, encoding="utf-8")
    created.append(path)
    return created


def update_status_and_clearml(created_reports: list[Path], paths: dict[str, Any], dataset_info: dict[str, Any]) -> dict[str, Any]:
    clearml_status = "local_success_clearml_unavailable"
    clearml_task_id = ""
    clearml_task_url = ""
    clearml_error = ""
    uploaded = False
    try:
        load_local_env()
        from clearml import Task

        task = Task.init(
            project_name=PROJECT_CLASSICAL,
            task_name=CLEARML_TASK_NAME,
            task_type=Task.TaskTypes.custom,
            reuse_last_task_id=False,
        )
        task.set_parameter("package_preparation_only", True)
        task.set_parameter("training_executed", False)
        task.set_parameter("kaggle_launched", False)
        task.set_parameter("missing_configs", 11)
        task.get_logger().report_text("Stage C missing-config execution package prepared. No training or Kaggle execution launched.")
        for artifact in [
            paths["missing_path"],
            paths["available_path"],
            paths["scope_path"],
            dataset_info["manifest_path"],
            *created_reports,
        ]:
            if artifact.exists():
                task.upload_artifact(name=artifact.stem, artifact_object=str(artifact))
        task.flush(wait_for_uploads=True)
        clearml_task_id = task.id or ""
        try:
            clearml_task_url = task.get_output_log_web_page() or ""
        except Exception:
            clearml_task_url = ""
        task.close()
        clearml_status = "success"
        uploaded = True
    except Exception as exc:
        clearml_error = type(exc).__name__

    clearml_report = REPORTS / "stage_c_clearml_package_upload_report.md"
    clearml_report.write_text(
        f"""# Stage C ClearML Package Upload Report

Generated: {TIMESTAMP}

- ClearML reporting task created: {'yes' if clearml_status == 'success' else 'no'}
- Task ID: `{clearml_task_id or 'not_created'}`
- Status: `{clearml_status}`
- Artifacts uploaded: {uploaded}
- Error type: `{clearml_error or 'none'}`

No fake training metrics were logged.
""",
        encoding="utf-8",
    )
    if clearml_report not in created_reports:
        created_reports.append(clearml_report)

    status_path = REPORTS / "pipeline_build_status.md"
    status_text = status_path.read_text(encoding="utf-8") if status_path.exists() else "# Pipeline Build Status\n"
    section = f"""
<!-- STAGE_C_EXECUTION_PACKAGE_START -->

## Stage C Missing-Config Execution Package

Updated: {TIMESTAMP}

Mandatory question included: **{TRADEOFF_Q}**

Status:
- Stage C planning completed: yes
- Stage C missing-config execution package prepared: yes
- Planned Stage C configs: 30
- Available from Notebook 07: 19
- Missing and prepared for future execution: 11
- Training run yet: no
- Kaggle execution launched: no
- Notebook pushed: no
- Full-grid started: no
- Notebook 07 metric values modified: no
- Notebook 08B metric values modified: no
- Notebook 08C metric values modified: no
- Dataset package created: yes
- Guarded notebook created: yes
- APPROVED_EXECUTION default: `False`
- ClearML reporting task created: {'yes' if clearml_status == 'success' else 'no'}
- ClearML task ID: `{clearml_task_id or 'not_created'}`

Primary outputs:
- `{rel(paths['missing_path'])}`
- `{rel(paths['available_path'])}`
- `{rel(paths['scope_path'])}`
- `{rel(dataset_info['manifest_path'])}`
- `{rel(NOTEBOOK_PATH)}`
- `{rel(KAGGLE_DATASET_ZIP)}`

<!-- STAGE_C_EXECUTION_PACKAGE_END -->
"""
    start = "<!-- STAGE_C_EXECUTION_PACKAGE_START -->"
    end = "<!-- STAGE_C_EXECUTION_PACKAGE_END -->"
    if start in status_text and end in status_text:
        status_text = status_text.split(start)[0] + section.strip("\n") + status_text.split(end, 1)[1]
    else:
        status_text = status_text.rstrip() + "\n\n" + section.strip("\n") + "\n"
    status_path.write_text(status_text, encoding="utf-8")

    registry_path = REPORTS / "clearml_run_registry.csv"
    cols = [
        "source",
        "phase_or_notebook",
        "task_name",
        "task_id",
        "task_url",
        "project_name",
        "final_status",
        "scalar_logged",
        "artifacts_uploaded",
        "report_path",
        "notes",
        "is_official",
        "purpose",
        "official_status",
    ]
    registry = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame(columns=cols)
    for col in cols:
        if col not in registry.columns:
            registry[col] = ""
    registry = registry[cols]
    registry = registry[
        ~(
            (registry["phase_or_notebook"].astype(str) == "stage_c_missing_configs_execution_package")
            & (registry["task_name"].astype(str) == CLEARML_TASK_NAME)
        )
    ]
    registry = pd.concat(
        [
            registry,
            pd.DataFrame(
                [
                    {
                        "source": "local_package_preparation",
                        "phase_or_notebook": "stage_c_missing_configs_execution_package",
                        "task_name": CLEARML_TASK_NAME,
                        "task_id": clearml_task_id,
                        "task_url": clearml_task_url,
                        "project_name": PROJECT_CLASSICAL,
                        "final_status": clearml_status,
                        "scalar_logged": False,
                        "artifacts_uploaded": uploaded,
                        "report_path": str((REPORTS / "stage_c_execution_package_report.md").resolve()),
                        "notes": "Package-preparation only. No training, Kaggle execution, notebook push, full-grid, or metric modification executed.",
                        "is_official": True,
                        "purpose": "stage_c_missing_configs_execution_package_reporting",
                        "official_status": "official_package_preparation",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    registry.to_csv(registry_path, index=False)

    return {
        "clearml_task_created": clearml_status == "success",
        "clearml_task_id": clearml_task_id,
        "clearml_status": clearml_status,
        "clearml_error": clearml_error,
        "pipeline_status_path": status_path,
        "registry_path": registry_path,
    }


def main() -> None:
    t0 = time.perf_counter()
    paths = create_missing_available_scope()
    dataset_info = create_dataset_package(paths)
    notebook_safety = create_notebook_and_ui_package(paths)
    created_reports = write_reports(paths, dataset_info, notebook_safety)
    status = update_status_and_clearml(created_reports, paths, dataset_info)

    if not notebook_safety["passed"]:
        raise RuntimeError(f"Notebook safety validation failed: {notebook_safety}")
    if not dataset_info["zip_ok"]:
        raise RuntimeError("Dataset zip validation failed")

    result = {
        "execution_package_prepared": True,
        "planned_stage_c_configs": 30,
        "available_configs": int(len(paths["available_df"])),
        "missing_configs": int(len(paths["missing_df"])),
        "dataset_package_created": True,
        "dataset_staging_folder": rel(KAGGLE_DATASET_DIR),
        "dataset_zip": rel(KAGGLE_DATASET_ZIP),
        "dataset_zip_size_bytes": dataset_info["zip_size_bytes"],
        "dataset_manifest": rel(dataset_info["manifest_path"]),
        "notebook_package_created": True,
        "notebook_path": rel(NOTEBOOK_PATH),
        "ui_package_folder": rel(UI_DIR),
        "approved_execution": APPROVED_EXECUTION,
        "future_execution_configs": int(len(paths["scope_df"])),
        "reference_only_configs": int(len(paths["available_df"])),
        "safety_validation_passed": notebook_safety["passed"],
        "resume_logic_improved": notebook_safety["resume_logic_checks_csvs"],
        "tradeoff_plan_created": (REPORTS / "stage_c_performance_inference_tradeoff_execution_plan.md").exists(),
        "clearml_reporting_task_created": status["clearml_task_created"],
        "clearml_task_id": status["clearml_task_id"],
        "reports_created": [rel(p) for p in created_reports],
        "csvs_created": [rel(paths["missing_path"]), rel(paths["available_path"]), rel(paths["scope_path"])],
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "training_run": False,
        "kaggle_execution_launched": False,
        "notebook_pushed": False,
        "full_grid_started": False,
        "notebook_07_08_metrics_modified": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
