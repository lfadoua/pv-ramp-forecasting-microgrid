from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config, project_root, resolve_path
from .data_audit import clean_dataframe, quality_summary, read_hdf_dataframe
from .physics_features import engineer_features, leakage_checks, move_target_last, write_feature_csv
from .reports import write_csv, write_json, write_markdown
from .split_windowing import leakage_report as split_leakage_report
from .split_windowing import process_variant
from .status import utc_now


TARGET_COLUMN = "ramp_rate_pu"
TIME_COLUMN = "timestamp"
RESOLUTIONS = ["one_minute", "one_hour"]
VARIANTS = ["full", "selected", "full_physics", "selected_physics"]
ONE_HOUR_NOTEBOOK_04_DIR = "notebook_04_split_windowing"


def read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    return json.loads(target.read_text(encoding="utf-8"))


def path_exists(path_like: str | Path | None) -> bool:
    return bool(path_like) and Path(path_like).exists()


def selected_feature_columns(root: Path) -> list[str]:
    path = root / "outputs" / "02_target_construction_feature_selection" / "selected_features_list.json"
    payload = read_json(path)
    return list(payload["feature_columns"])


def existing_one_minute_paths(root: Path) -> dict[str, Path]:
    nb02 = read_json(root / "outputs" / "02_target_construction_feature_selection" / "notebook_02_final_report.json")
    nb03 = read_json(root / "outputs" / "03_physics_informed_features" / "notebook_03_final_report.json")
    return {
        "full": Path(nb02["output_files"]["original_data_all_features_clean_with_target"]),
        "selected": Path(nb02["output_files"]["selected_features_with_target"]),
        "selected_physics": Path(nb03["output_files"]["selected_features_plus_physics_informed_with_target"]),
        "full_physics": Path(nb03["output_files"]["original_all_features_plus_physics_informed_with_target"]),
    }


def one_hour_paths(root: Path) -> dict[str, Path]:
    base02 = root / "outputs" / "one_hour" / "notebook_02_target_features"
    base03 = root / "outputs" / "one_hour" / "notebook_03_physics_features"
    return {
        "clean": root / "outputs" / "one_hour" / "notebook_01_cleaning" / "original_data_all_features_clean.csv",
        "full": base02 / "original_data_all_features_clean_with_target.csv",
        "selected": base02 / "selected_features_with_target.csv",
        "selected_physics": base03 / "selected_features_plus_physics_informed_with_target.csv",
        "full_physics": base03 / "original_all_features_plus_physics_informed_with_target.csv",
    }


def needs_one_hour_generation(root: Path) -> bool:
    paths = one_hour_paths(root)
    required = ["clean", "full", "selected", "selected_physics", "full_physics"]
    manifests = [
        root / "outputs" / "one_hour" / ONE_HOUR_NOTEBOOK_04_DIR / variant / "window_ready_manifest.json"
        for variant in VARIANTS
    ]
    return any(not paths[key].exists() for key in required) or any(not path.exists() for path in manifests)


def create_target(df: pd.DataFrame, rated_power_kw: float) -> pd.DataFrame:
    if TARGET_COLUMN in df.columns:
        raise ValueError(f"{TARGET_COLUMN} already exists before target creation.")
    working = df.copy()
    pv = pd.to_numeric(working["P_Solar[kW]"], errors="coerce")
    working[TARGET_COLUMN] = pv.diff() / float(rated_power_kw)
    before = len(working)
    working = working.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    if len(working) != before - 1:
        raise ValueError("Unexpected one-hour target row drop count; only the first differenced row should be dropped.")
    return move_target_last(working, TARGET_COLUMN)


def generate_one_hour_outputs(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    paths = one_hour_paths(root)
    selected_features = selected_feature_columns(root)
    raw_path = resolve_path(config["paths"]["raw_solete_files"]["one_hour"], root)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    raw = read_hdf_dataframe(raw_path, config["data"].get("hdf_key"))
    clean = clean_dataframe(raw)
    write_feature_csv(paths["clean"], clean)

    full = create_target(clean, float(config["project"]["rated_power_kw"]))
    missing_selected = [column for column in selected_features if column not in full.columns]
    if missing_selected:
        raise ValueError(f"Selected one-minute features missing in one-hour data: {missing_selected}")
    selected = full[[TIME_COLUMN] + selected_features + [TARGET_COLUMN]].copy()
    write_feature_csv(paths["full"], full)
    write_feature_csv(paths["selected"], selected)

    full_physics, full_dictionary, full_meta, full_engineered = engineer_features(full, TARGET_COLUMN, config, "full_physics")
    selected_physics, selected_dictionary, selected_meta, selected_engineered = engineer_features(selected, TARGET_COLUMN, config, "selected_physics")
    write_feature_csv(paths["full_physics"], full_physics)
    write_feature_csv(paths["selected_physics"], selected_physics)

    output_root = root / "outputs" / "one_hour" / ONE_HOUR_NOTEBOOK_04_DIR
    config_for_hour = copy.deepcopy(config)
    config_for_hour["data"]["primary_resolution_candidate"] = "one_hour"
    variant_paths = {
        "full": paths["full"],
        "selected": paths["selected"],
        "selected_physics": paths["selected_physics"],
        "full_physics": paths["full_physics"],
    }
    variant_reports: dict[str, dict[str, Any]] = {}
    experiment_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        report, rows = process_variant(variant, variant_paths[variant], config_for_hour, output_root, TARGET_COLUMN)
        variant_reports[variant] = report
        experiment_rows.extend(rows)

    experiment_matrix = pd.DataFrame(experiment_rows)
    split_metadata = {
        "resolution": "one_hour",
        "strategy": config["splits"]["strategy"],
        "random_shuffle": False,
        "global_split_ratios": {
            "train": float(config["splits"]["train_fraction"]),
            "validation": float(config["splits"]["validation_fraction"]),
            "test": float(config["splits"]["test_fraction"]),
        },
        "dataset_variants": {variant: report["split_summary"] for variant, report in variant_reports.items()},
        "human_readable_policy": "Train is earliest time period, validation is future-near, test is final unseen future.",
    }
    windowing_metadata = {
        "resolution": "one_hour",
        "dataset_variants": {
            variant: {
                "feature_count": report["feature_count"],
                "row_count": report["row_count"],
                "window_metadata": report["window_metadata"],
            }
            for variant, report in variant_reports.items()
        },
        "horizon_definition": config["lookback_horizon"]["horizon_definition"],
        "materialized_windows_created": False,
        "note": "One-hour artifacts mirror the Notebook 04 protocol without overwriting one-minute artifacts.",
    }
    scaler_metadata = {
        "resolution": "one_hour",
        "dataset_variants": {variant: report["scaler_metadata"] for variant, report in variant_reports.items()},
    }
    leakage = split_leakage_report(variant_reports)
    quality = quality_summary(clean, raw_path)
    generation_report = {
        "resolution": "one_hour",
        "generated_at": utc_now(),
        "source_path": str(raw_path),
        "cleaned_path": str(paths["clean"]),
        "quality_summary": quality,
        "target_formula": "(P_Solar[t] - P_Solar[t-1]) / P_R",
        "rated_power_kw": float(config["project"]["rated_power_kw"]),
        "selected_features_source": str(root / "outputs" / "02_target_construction_feature_selection" / "selected_features_list.json"),
        "selected_features": selected_features,
        "paths": {key: str(path) for key, path in paths.items()},
        "physics_feature_note": "Lag/rolling feature suffixes represent row steps at one-hour resolution; the one-minute outputs were not regenerated.",
        "full_physics_metadata": full_meta,
        "selected_physics_metadata": selected_meta,
        "full_physics_leakage_checks": leakage_checks(full_physics, full_dictionary, TARGET_COLUMN),
        "selected_physics_leakage_checks": leakage_checks(selected_physics, selected_dictionary, TARGET_COLUMN),
        "variant_reports": variant_reports,
        "experiment_matrix_rows": int(len(experiment_matrix)),
        "leakage_checks": leakage,
    }

    write_csv(output_root / "experiment_matrix.csv", experiment_matrix)
    write_json(output_root / "split_metadata.json", split_metadata)
    write_json(output_root / "windowing_metadata.json", windowing_metadata)
    write_json(output_root / "scaler_metadata.json", scaler_metadata)
    write_json(output_root / "leakage_check_report.json", leakage)
    write_json(output_root / "one_hour_generation_report.json", generation_report)
    return generation_report


def manifest_for(root: Path, resolution: str, variant: str) -> Path:
    if resolution == "one_minute":
        return root / "outputs" / "one_minute" / "notebook_04_split_windowing" / variant / "window_ready_manifest.json"
    return root / "outputs" / "one_hour" / ONE_HOUR_NOTEBOOK_04_DIR / variant / "window_ready_manifest.json"


def dataset_inventory(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for resolution in RESOLUTIONS:
        for variant in VARIANTS:
            manifest_path = manifest_for(root, resolution, variant)
            ready = manifest_path.exists()
            row: dict[str, Any] = {
                "resolution": resolution,
                "dataset_variant": variant,
                "ready": bool(ready),
                "manifest_path": str(manifest_path) if ready else "",
                "source_path": "",
                "row_count": None,
                "feature_count": None,
                "target_column": TARGET_COLUMN,
                "target_last_column": None,
                "train_rows": None,
                "validation_rows": None,
                "test_rows": None,
                "scaler_file": "",
                "notes": "",
            }
            if ready:
                manifest = read_json(manifest_path)
                source = Path(manifest["source_path"])
                header = list(pd.read_csv(source, nrows=0).columns) if source.exists() else []
                row.update(
                    {
                        "source_path": str(source),
                        "row_count": int(manifest["row_count"]),
                        "feature_count": int(manifest["feature_count"]),
                        "target_last_column": bool(header and header[-1] == TARGET_COLUMN),
                        "train_rows": int(manifest["split_summary"]["train"]["row_count"]),
                        "validation_rows": int(manifest["split_summary"]["validation"]["row_count"]),
                        "test_rows": int(manifest["split_summary"]["test"]["row_count"]),
                        "scaler_file": manifest.get("scaler_file", ""),
                    }
                )
            else:
                row["notes"] = "Missing manifest; dataset is not ready."
            rows.append(row)
    return pd.DataFrame(rows)


def collect_clearml_registry(root: Path, kaggle_status_override: dict[str, Any] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    forced_status_by_task: dict[str, dict[str, str]] = {}
    phase6_report = root / "outputs" / "06_statistical_time_series_models" / "notebook_06_final_report.json"
    if phase6_report.exists():
        try:
            phase6 = read_json(phase6_report)
            for item in phase6.get("previous_phase6_clearml_runs", []):
                task_id = item.get("clearml_task_id")
                if task_id:
                    forced_status_by_task[str(task_id)] = {
                        "final_status": item.get("status", "marked_failed"),
                        "notes": item.get("reason", "Status overridden from Notebook 06 final report."),
                    }
        except Exception:
            forced_status_by_task = {}
    for path in sorted((root / "outputs" / "local_runs").glob("*/*/final_report.json")):
        try:
            payload = read_json(path)
        except Exception:
            continue
        task_id = payload.get("clearml_task_id")
        forced = forced_status_by_task.get(str(task_id)) if task_id else None
        rows.append(
            {
                "source": "local_runner",
                "phase_or_notebook": payload.get("notebook_name", path.parent.parent.name),
                "task_name": payload.get("clearml_task_name"),
                "task_id": task_id,
                "task_url": payload.get("clearml_task_url"),
                "project_name": payload.get("clearml_project_name"),
                "final_status": forced["final_status"] if forced else payload.get("final_status"),
                "scalar_logged": payload.get("metrics", {}).get("clearml_scalar_logged"),
                "artifacts_uploaded": payload.get("metrics", {}).get("clearml_artifacts_uploaded"),
                "report_path": str(path),
                "notes": forced["notes"] if forced else "",
            }
        )
    for path in sorted((root / "outputs" / "kaggle_runs").glob("**/00_access_check_clearml_kaggle_final_report.json")):
        payload = read_json(path)
        rows.append(
            {
                "source": "kaggle_access_check",
                "phase_or_notebook": "00_access_check_clearml_kaggle",
                "task_name": payload.get("clearml_task_name"),
                "task_id": payload.get("clearml_task_id"),
                "task_url": payload.get("clearml_task_url"),
                "project_name": payload.get("clearml_project_name"),
                "final_status": payload.get("final_status"),
                "scalar_logged": payload.get("clearml_scalar_logged"),
                "artifacts_uploaded": payload.get("clearml_artifacts_uploaded"),
                "report_path": str(path),
                "notes": "Historical Kaggle report found locally.",
            }
        )
    if kaggle_status_override:
        rows.append(kaggle_status_override)
    return pd.DataFrame(rows)


def latest_kaggle_status(root: Path) -> dict[str, Any]:
    reports = sorted(
        (root / "outputs" / "kaggle_runs").glob("**/00_access_check_clearml_kaggle_final_report.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not reports:
        return {
            "status": "not_found",
            "kaggle_secrets_readable": False,
            "kaggle_clearml_task_creation_succeeded": False,
            "kaggle_clearml_ready": False,
            "previous_issue_resolved": False,
            "report_path": "",
            "notes": "No Kaggle access-check report found locally.",
        }
    payload = read_json(reports[0])
    ok = bool(
        payload.get("final_status") == "success"
        and payload.get("clearml_task_created")
        and payload.get("clearml_scalar_logged")
        and payload.get("clearml_artifacts_uploaded")
    )
    return {
        "status": payload.get("final_status"),
        "kaggle_secrets_readable": bool(not payload.get("missing_or_empty_secrets")),
        "kaggle_clearml_task_creation_succeeded": bool(payload.get("clearml_task_created")),
        "kaggle_clearml_ready": ok,
        "previous_issue_resolved": ok,
        "report_path": str(reports[0]),
        "notes": "Latest local Kaggle access-check report.",
    }


def progressive_training_strategy_text(config: dict[str, Any], inventory: pd.DataFrame) -> str:
    ready_count = int(inventory["ready"].sum())
    dl_models = config["deep_learning"]["mandatory_models"]
    backend_priority = config["classical_ml"]["backend_priority"]
    lines = [
        "# Progressive Training Strategy",
        "",
        "This strategy prepares Notebook 07 and Notebook 08 without launching either notebook.",
        "",
        "Dataset coverage:",
        f"- Prepared datasets expected: `8`; currently ready: `{ready_count}`.",
        "- Resolutions: `one_minute`, `one_hour`.",
        "- Variants per resolution: `full`, `selected`, `full_physics`, `selected_physics`.",
        "",
        "Notebook 07 Classical ML:",
        f"- Backend priority: `{backend_priority}`.",
        "- Each result row must include `backend_requested`, `backend_used`, `fallback_used`, `fallback_reason`, `hardware_used`, and `gpu_available`.",
        "- Do not fail the whole notebook if cuML/RAPIDS is unavailable; fall back and document why.",
        "- XGBoost should request GPU when available and stable, otherwise use CPU with a recorded fallback reason.",
        "",
        "Notebook 08 Deep Learning:",
        f"- Mandatory models: `{dl_models}`.",
        "- GRU is mandatory and must be tracked like MLP/ANN, LSTM, BiLSTM, CNN1D, and CNN-LSTM.",
        "- Heavy deep learning should run on Kaggle GPU when possible.",
        "",
        "Hardware checks before heavy training:",
        "- Kaggle GPU type, number of GPUs, CUDA availability, PyTorch/TensorFlow GPU visibility, accelerator type, and ClearML logging must be recorded.",
        "- Local machine is for preparation, reporting cleanup, inventories, and small smoke tests.",
        "",
        "No silent sample caps:",
        "- Official `status=final` results must use the full chronological train split.",
        "- Expensive configs that cannot run on the full split must be marked `status=aborted_expensive` or run only as `status=smoke_test`.",
        "",
        "Staged funnel:",
        "- Prepare all 8 datasets.",
        "- Fast-screen all 8 with unified lightweight parameters.",
        "- Promote datasets using validation results only.",
        "- Train heavy classical ML only on promoted datasets.",
        "- Train deep learning only on best promoted datasets using Kaggle GPU.",
    ]
    return "\n".join(lines) + "\n"


def cleanup_report_text(
    inventory: pd.DataFrame,
    one_hour_generated: bool,
    one_hour_report: dict[str, Any] | None,
    kaggle_status: dict[str, Any],
    updated_files: list[str],
    one_hour_generation_report_exists: bool,
) -> str:
    ready_count = int(inventory["ready"].sum())
    one_hour_ready = bool(inventory[inventory["resolution"] == "one_hour"]["ready"].all())
    all_ready = ready_count == 8
    lines = [
        "# Pre-Notebook-07 Cleanup Report",
        "",
        "Scope:",
        "- Existing one-minute outputs were inspected and kept.",
        "- No Notebook 07 or Notebook 08 training was launched.",
        "- Missing one-hour datasets were generated only when absent.",
        "",
        "Existing outputs kept:",
        "- One-minute full, selected, full_physics, and selected_physics manifests and datasets.",
        "- Notebook 01-06 reports and metrics.",
        "",
        "Fixes and additions:",
        f"- One-hour generation performed in latest invocation: `{one_hour_generated}`.",
        f"- One-hour generation report exists from this cleanup work: `{one_hour_generation_report_exists}`.",
        f"- One-hour datasets ready: `{one_hour_ready}`.",
        f"- All 8 resolution/variant datasets ready: `{all_ready}`.",
        "- Added dataset inventory and ClearML run registry.",
        "- Added staged methodology for Notebook 07/08.",
        "- Added GRU as mandatory Notebook 08 model in the central config.",
        "- Added Notebook 07 backend-priority and fallback reporting rules in the central config.",
        "",
        "Kaggle/ClearML status:",
        f"- Latest local Kaggle access-check status: `{kaggle_status['status']}`.",
        f"- Kaggle Secrets readable according to latest local report: `{kaggle_status['kaggle_secrets_readable']}`.",
        f"- Kaggle ClearML task creation succeeded according to latest local report: `{kaggle_status['kaggle_clearml_task_creation_succeeded']}`.",
        f"- Kaggle ClearML READY according to latest local report: `{kaggle_status['kaggle_clearml_ready']}`.",
        f"- Previous Kaggle/ClearML issue resolved according to latest local report: `{kaggle_status['previous_issue_resolved']}`.",
        f"- Latest Kaggle report path: `{kaggle_status['report_path']}`.",
        "",
        "One-hour generation summary:",
    ]
    if one_hour_report:
        lines.extend(
            [
                f"- Source: `{one_hour_report['source_path']}`.",
                f"- Clean rows: `{one_hour_report['quality_summary']['samples']}`.",
                f"- Target formula: `{one_hour_report['target_formula']}`.",
                f"- Experiment rows: `{one_hour_report['experiment_matrix_rows']}`.",
                f"- Split/window leakage checks passed: `{one_hour_report['leakage_checks']['checks']['passed']}`.",
            ]
        )
    else:
        lines.append("- One-hour generation was not needed in the latest invocation; existing one-hour artifacts were kept.")
    lines.extend(
        [
            "",
            "Updated files:",
            *[f"- `{path}`" for path in updated_files],
            "",
            "Stop condition:",
            "- Notebook 07 is prepared but not started.",
            "- Notebook 08 is prepared but not started.",
            "- User approval is required before launching Notebook 07.",
        ]
    )
    if not kaggle_status["kaggle_clearml_ready"]:
        lines.extend(
            [
                "",
                "Remaining risk:",
                "- The latest local Kaggle access-check report still does not prove Kaggle ClearML readiness. Rerun the access-check after confirming Kaggle Secrets in the Kaggle UI.",
            ]
        )
    return "\n".join(lines) + "\n"


def pipeline_status_text(inventory: pd.DataFrame, kaggle_status: dict[str, Any]) -> str:
    ready_count = int(inventory["ready"].sum())
    lines = [
        "# Pipeline Build Status",
        "",
        "Phase: `Pre-Notebook-07 cleanup`",
        "",
        "Completed:",
        "- Valid one-minute outputs from Phases 1-6 were kept.",
        "- Missing one-hour datasets and window-ready manifests were generated if absent.",
        "- Dataset inventory, ClearML registry, cleanup report, and progressive strategy were updated.",
        "- Notebook 07 and Notebook 08 were prepared methodologically but not launched.",
        "",
        "Dataset readiness:",
        f"- Ready datasets: `{ready_count}/8`.",
        f"- One-hour ready: `{bool(inventory[inventory['resolution'] == 'one_hour']['ready'].all())}`.",
        f"- All 8 ready: `{ready_count == 8}`.",
        "",
        "Kaggle/ClearML:",
        f"- Latest local access-check status: `{kaggle_status['status']}`.",
        f"- Kaggle ClearML READY from latest local report: `{kaggle_status['kaggle_clearml_ready']}`.",
        f"- Previous issue resolved from latest local report: `{kaggle_status['previous_issue_resolved']}`.",
        "",
        "Training:",
        "- Notebook 07: not started.",
        "- Notebook 08: not started.",
        "- Heavy training: not started.",
        "",
        "Ready for user review before Notebook 07: `yes`",
    ]
    return "\n".join(lines) + "\n"


def run_pre_notebook07_cleanup(config_path: str | Path | None = None, force_one_hour: bool = False) -> dict[str, Any]:
    root = project_root()
    config = load_config(config_path)
    updated_files: list[str] = []

    existing_minute = existing_one_minute_paths(root)
    missing_minute = [variant for variant, path in existing_minute.items() if not path.exists()]
    if missing_minute:
        raise FileNotFoundError(f"Missing valid one-minute outputs; refusing cleanup until reviewed: {missing_minute}")

    one_hour_generated = False
    one_hour_report: dict[str, Any] | None = None
    if force_one_hour or needs_one_hour_generation(root):
        one_hour_generated = True
        one_hour_report = generate_one_hour_outputs(root, config)
        updated_files.append(str(root / "outputs" / "one_hour" / ONE_HOUR_NOTEBOOK_04_DIR / "one_hour_generation_report.json"))

    inventory = dataset_inventory(root)
    inventory_path = root / "outputs" / "reports" / "dataset_inventory.csv"
    write_csv(inventory_path, inventory)
    updated_files.append(str(inventory_path))

    kaggle_status = latest_kaggle_status(root)
    registry = collect_clearml_registry(root)
    registry_path = root / "outputs" / "reports" / "clearml_run_registry.csv"
    write_csv(registry_path, registry)
    updated_files.append(str(registry_path))

    strategy_path = root / "outputs" / "reports" / "progressive_training_strategy.md"
    write_markdown(strategy_path, progressive_training_strategy_text(config, inventory))
    updated_files.append(str(strategy_path))

    pipeline_status_path = root / "outputs" / "reports" / "pipeline_build_status.md"
    updated_files.append(str(pipeline_status_path))

    cleanup_path = root / "outputs" / "reports" / "pre_notebook_07_cleanup_report.md"
    updated_files.append(str(cleanup_path))
    one_hour_generation_report_path = root / "outputs" / "one_hour" / ONE_HOUR_NOTEBOOK_04_DIR / "one_hour_generation_report.json"
    write_markdown(
        cleanup_path,
        cleanup_report_text(
            inventory,
            one_hour_generated,
            one_hour_report,
            kaggle_status,
            updated_files,
            one_hour_generation_report_path.exists(),
        ),
    )
    write_markdown(pipeline_status_path, pipeline_status_text(inventory, kaggle_status))

    result = {
        "final_status": "success",
        "generated_at": utc_now(),
        "one_minute_outputs_kept": True,
        "one_hour_generated": one_hour_generated,
        "one_hour_ready": bool(inventory[inventory["resolution"] == "one_hour"]["ready"].all()),
        "all_8_datasets_ready": bool(inventory["ready"].sum() == 8),
        "notebook_07_started": False,
        "notebook_08_started": False,
        "kaggle_status": kaggle_status,
        "updated_files": updated_files,
        "inventory_rows": int(len(inventory)),
        "clearml_registry_rows": int(len(registry)),
    }
    summary_path = root / "outputs" / "reports" / "pre_notebook_07_cleanup_summary.json"
    write_json(summary_path, result)
    result["updated_files"].append(str(summary_path))
    write_json(summary_path, result)
    return result
