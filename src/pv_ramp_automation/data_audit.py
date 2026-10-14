from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import load_config, project_root, resolve_path
from .reports import write_csv, write_json, write_markdown
from .status import section_status, utc_now, write_status


PHYSICAL_COLUMN_PATTERNS = {
    "pv_power": ["p_solar", "pv_power", "solar"],
    "ghi": ["ghi"],
    "poa": ["poa"],
    "temperature": ["temperature", "temp"],
    "humidity": ["humidity"],
    "wind_speed": ["wind_speed", "wind"],
    "wind_direction": ["wind_dir"],
    "pressure": ["pressure"],
    "solar_geometry": ["azimuth", "elevation"],
}


def _safe_name(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_").replace("\\", "_").replace("[", "").replace("]", "").replace("(", "").replace(")", "")


def read_hdf_dataframe(path: Path, key: str | None = None) -> pd.DataFrame:
    with pd.HDFStore(path, "r") as store:
        keys = store.keys()
        if not keys:
            raise ValueError(f"No HDF keys found in {path}")
        selected_key = key if key in keys else keys[0]
        df = store.select(selected_key)
    return df


def dataframe_with_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if isinstance(result.index, pd.DatetimeIndex):
        result.insert(0, "timestamp", result.index)
        result = result.reset_index(drop=True)
    elif "timestamp" in result.columns:
        result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    else:
        for column in result.columns:
            if "time" in column.lower() or "date" in column.lower():
                result.insert(0, "timestamp", pd.to_datetime(result[column], errors="coerce"))
                break
    if "timestamp" not in result.columns:
        raise ValueError("Could not identify timestamp column.")
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result = result.dropna(subset=["timestamp"]).sort_values("timestamp")
    result = result.drop_duplicates(subset=["timestamp"], keep="first")
    return result.reset_index(drop=True)


def infer_resolution(diff: pd.Timedelta | None) -> tuple[str, str]:
    if diff is None or pd.isna(diff):
        return "unknown", "unknown"
    minutes = diff.total_seconds() / 60.0
    if minutes <= 1.5:
        return "one_minute", "very-short-term / nowcasting"
    if minutes <= 5.5:
        return "five_minute", "short-term"
    if minutes <= 65:
        return "one_hour", "medium-term"
    return f"{int(round(minutes))}_minute", "long-term"


def detect_physical_variables(columns: list[str]) -> dict[str, list[str]]:
    detected: dict[str, list[str]] = {}
    lowered = {column: column.lower() for column in columns}
    for family, patterns in PHYSICAL_COLUMN_PATTERNS.items():
        matches = [column for column, low in lowered.items() if any(pattern in low for pattern in patterns)]
        detected[family] = matches
    return detected


def quality_summary(df: pd.DataFrame, source_file: Path) -> dict[str, Any]:
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    diffs = timestamps.sort_values().diff().dropna()
    median_diff = diffs.median() if not diffs.empty else None
    resolution_label, horizon_class = infer_resolution(median_diff)
    expected_count = None
    missing_timestamps_estimate = None
    if median_diff is not None and not pd.isna(median_diff) and len(timestamps) > 1:
        expected_count = int(((timestamps.max() - timestamps.min()) / median_diff) + 1)
        missing_timestamps_estimate = max(0, expected_count - timestamps.nunique())

    numeric = df.select_dtypes(include="number")
    abnormal: dict[str, Any] = {}
    for column in numeric.columns:
        low = column.lower()
        if "p_solar" in low or "power" in low or "ghi" in low or "poa" in low:
            abnormal[column] = {
                "negative_count": int((numeric[column] < 0).sum()),
                "max": float(numeric[column].max()) if numeric[column].notna().any() else None,
                "min": float(numeric[column].min()) if numeric[column].notna().any() else None,
            }

    return {
        "source_file": str(source_file),
        "source_file_size_mb": round(source_file.stat().st_size / 1024 / 1024, 3),
        "resolution_label": resolution_label,
        "resolution_class": horizon_class,
        "time_start": timestamps.min().isoformat() if timestamps.notna().any() else None,
        "time_end": timestamps.max().isoformat() if timestamps.notna().any() else None,
        "samples": int(len(df)),
        "columns": list(df.columns),
        "column_count": int(len(df.columns)),
        "physical_variables": detect_physical_variables(list(df.columns)),
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_values_by_column": {column: int(value) for column, value in df.isna().sum().items() if int(value) > 0},
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "median_frequency": str(median_diff) if median_diff is not None else None,
        "expected_regular_samples": expected_count,
        "missing_timestamps_estimate": missing_timestamps_estimate,
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1024 / 1024), 3),
        "abnormal_value_checks": abnormal,
        "contains_target_ramp_rate_pu": "ramp_rate_pu" in df.columns,
    }


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    clean = dataframe_with_timestamp(df)
    for column in clean.columns:
        if column != "timestamp":
            clean[column] = pd.to_numeric(clean[column], errors="coerce")
    if "ramp_rate_pu" in clean.columns:
        clean = clean.drop(columns=["ramp_rate_pu"])
    return clean


def suitability_reason(summary: dict[str, Any]) -> tuple[str, str]:
    label = summary["resolution_label"]
    missing = summary.get("missing_timestamps_estimate") or 0
    if label == "one_minute":
        return "primary", "Best available match for short-term PV ramp forecasting and microgrid control support."
    if label == "one_hour":
        return "optional_reference", "Useful as low-cost reference/exploratory resolution, but too coarse for fast ramp-rate control."
    if missing > 0:
        return "review", "Resolution has estimated timestamp gaps and needs review before official use."
    return "review", "Resolution suitability requires manual review."


def save_quality_plots(df: pd.DataFrame, resolution: str, output_dir: Path) -> list[str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    missing = df.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    missing.plot(kind="bar", ax=ax)
    ax.set_title(f"Missing values by column - {resolution}")
    ax.set_ylabel("count")
    fig.tight_layout()
    path = plot_dir / f"{resolution}_missing_values.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    saved.append(str(path))

    if "P_Solar[kW]" in df.columns:
        sample = df[["timestamp", "P_Solar[kW]"]].head(min(len(df), 3000))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sample["timestamp"], sample["P_Solar[kW]"], linewidth=0.8)
        ax.set_title(f"P_Solar sample - {resolution}")
        ax.set_ylabel("kW")
        fig.autofmt_xdate()
        fig.tight_layout()
        path = plot_dir / f"{resolution}_p_solar_sample.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(str(path))

    return saved


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + df.to_csv(index=False).strip() + "\n```"


def write_local_runner_report(report: dict[str, Any], quality: dict[str, Any]) -> None:
    output_dir = os.environ.get("LOCAL_OUTPUT_DIR")
    if not output_dir:
        return
    data_shapes = {
        name: {
            "samples": summary.get("samples"),
            "columns": summary.get("column_count"),
            "resolution_label": summary.get("resolution_label"),
        }
        for name, summary in quality.items()
    }
    payload = {
        "notebook_name": report["notebook_name"],
        "final_status": report["final_status"],
        "primary_resolution": report.get("primary_resolution"),
        "target_created": report.get("target_created"),
        "data_shapes": data_shapes,
        "split_summary": {"chronological_split_configured": True, "split_applied": False},
        "report_path": str(Path(report["quality_report"]).parent / "notebook_01_final_report.json"),
    }
    write_json(Path(output_dir) / "local_clearml_report.json", payload)


def selection_markdown(inventory: pd.DataFrame, quality: dict[str, Any], selected_primary: str) -> str:
    lines = [
        "# Resolution Selection Report",
        "",
        "Project label: Short-term PV ramp-rate forecasting model for microgrid control support.",
        "",
        f"Primary resolution selected: `{selected_primary}`",
        "",
        "Rationale:",
        "- The one-minute SOLETE file is the best available resolution for short-term ramp-rate forecasting and microgrid control support.",
        "- The one-hour SOLETE file is retained as an optional low-cost reference, but it is too coarse for fast PV ramp events.",
        "- Notebook 01 does not create `ramp_rate_pu`; the official target is reserved for Notebook 02.",
        "",
        "Inventory:",
        "",
        dataframe_to_markdown_table(inventory),
        "",
        "Rejected or limited resolutions:",
        "- `one_hour`: optional/reference only unless the user later approves comparative analysis.",
    ]
    return "\n".join(lines) + "\n"


def run_notebook01_audit(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    root = project_root()
    output_root = root / "outputs" / "notebook_01_data_audit"
    run_state = output_root / "run_state"
    output_root.mkdir(parents=True, exist_ok=True)
    run_state.mkdir(parents=True, exist_ok=True)

    status_files: list[str] = []
    cleaned_files: dict[str, str] = {}
    plots: dict[str, list[str]] = {}
    quality: dict[str, Any] = {}
    errors: list[str] = []

    started = utc_now()
    try:
        raw_files = {name: resolve_path(path, root) for name, path in cfg["paths"]["raw_solete_files"].items()}
        inventory_rows = []
        for configured_name, path in raw_files.items():
            inventory_rows.append({
                "configured_name": configured_name,
                "path": str(path),
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 3) if path.exists() else None,
            })
        inventory_df = pd.DataFrame(inventory_rows)
        write_csv(output_root / "resolution_inventory_raw_files.csv", inventory_df)
        write_status(run_state / "section_01_inventory.json", section_status("inventory", started, "success", "Raw SOLETE file inventory completed.", {"file_count": len(inventory_rows)}))
    except Exception:
        errors.append(traceback.format_exc())
        write_status(run_state / "section_01_inventory.json", section_status("inventory", started, "failed", "Raw SOLETE inventory failed.", error_message=errors[-1]))
        raise

    started = utc_now()
    inventory_records = []
    try:
        for configured_name, path in raw_files.items():
            if not path.exists():
                continue
            raw = read_hdf_dataframe(path, cfg["data"].get("hdf_key"))
            clean = clean_dataframe(raw)
            summary = quality_summary(clean, path)
            role, reason = suitability_reason(summary)
            summary["configured_name"] = configured_name
            summary["recommended_role"] = role
            summary["suitability_reason"] = reason
            quality[configured_name] = summary
            inventory_records.append({
                "configured_name": configured_name,
                "resolution_label": summary["resolution_label"],
                "resolution_class": summary["resolution_class"],
                "samples": summary["samples"],
                "columns": summary["column_count"],
                "time_start": summary["time_start"],
                "time_end": summary["time_end"],
                "median_frequency": summary["median_frequency"],
                "missing_values_total": summary["missing_values_total"],
                "duplicate_timestamps": summary["duplicate_timestamps"],
                "missing_timestamps_estimate": summary["missing_timestamps_estimate"],
                "memory_mb": summary["memory_mb"],
                "recommended_role": role,
                "suitability_reason": reason,
            })
            plots[configured_name] = save_quality_plots(clean, configured_name, output_root)

            if role == "primary":
                clean_dir = root / "outputs" / summary["resolution_label"] / "notebook_01_cleaning"
                clean_dir.mkdir(parents=True, exist_ok=True)
                clean_path = clean_dir / "original_data_all_features_clean.csv"
                clean.to_csv(clean_path, index=False)
                cleaned_files[configured_name] = str(clean_path)

        resolution_inventory = pd.DataFrame(inventory_records)
        write_csv(output_root / "resolution_inventory.csv", resolution_inventory)
        write_json(output_root / "resolution_quality_report.json", quality)
        selected_primary = "one_minute" if "one_minute" in quality else (next(iter(quality.keys())) if quality else "none")
        write_markdown(output_root / "resolution_selection_report.md", selection_markdown(resolution_inventory, quality, selected_primary))
        write_status(
            run_state / "section_02_resolution_quality.json",
            section_status(
                "resolution_quality",
                started,
                "success",
                "SOLETE H5 files audited and primary cleaned dataset saved.",
                {"audited_resolution_count": len(quality), "cleaned_file_count": len(cleaned_files), "selected_primary": selected_primary},
            ),
        )
    except Exception:
        errors.append(traceback.format_exc())
        write_status(run_state / "section_02_resolution_quality.json", section_status("resolution_quality", started, "failed", "Resolution quality audit failed.", error_message=errors[-1]))
        raise

    report = {
        "notebook_name": "01_data_audit_resolution_cleaning",
        "execution_type": "local",
        "project_label": cfg["project"]["label"],
        "target_created": False,
        "official_target_reserved_for": cfg["data"]["target_created_in_notebook"],
        "primary_resolution": "one_minute" if "one_minute" in quality else None,
        "optional_secondary_resolutions": ["one_hour"] if "one_hour" in quality else [],
        "quality_report": str(output_root / "resolution_quality_report.json"),
        "resolution_inventory": str(output_root / "resolution_inventory.csv"),
        "resolution_selection_report": str(output_root / "resolution_selection_report.md"),
        "cleaned_files": cleaned_files,
        "plots": plots,
        "warnings": [],
        "errors": errors,
        "final_status": "success" if not errors and bool(cleaned_files) else "failed",
    }
    write_json(output_root / "notebook_01_final_report.json", report)
    write_local_runner_report(report, quality)
    return report
