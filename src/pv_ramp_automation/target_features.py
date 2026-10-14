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


NOTEBOOK_NAME = "02_target_construction_feature_selection"
TIME_COLUMN = "timestamp"
TARGET_ALIGNMENT_NOTE = (
    "Notebook 02 performs candidate feature selection only. Final forecasting examples are created in Notebook 04. "
    "For a horizon H, the model input window must use only information available up to forecast origin t, and the "
    "target must be ramp_rate_pu(t+H). Models must never predict ramp_rate_pu(t) using same-row P_Solar(t) or "
    "same-row irradiance as features. Notebook 04 must enforce this target/input alignment during split, scaling "
    "and window construction."
)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _fraction(count: int, total: int) -> float:
    return round(float(count / total), 6) if total else 0.0


def read_notebook01_report(root: Path) -> dict[str, Any]:
    path = root / "outputs" / "notebook_01_data_audit" / "notebook_01_final_report.json"
    if not path.exists():
        raise FileNotFoundError(f"Notebook 01 final report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("final_status") != "success":
        raise RuntimeError("Notebook 01 final report is not successful.")
    return report


def input_file_from_notebook01(report: dict[str, Any], primary_resolution: str) -> Path:
    cleaned_files = report.get("cleaned_files", {})
    input_file = cleaned_files.get(primary_resolution)
    if not input_file:
        raise FileNotFoundError(f"Notebook 01 did not report a cleaned file for {primary_resolution}.")
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Cleaned primary-resolution file not found: {path}")
    return path


def ensure_timestamp(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if TIME_COLUMN not in df.columns:
        raise ValueError(f"Required time column is missing: {TIME_COLUMN}")
    result = df.copy()
    result[TIME_COLUMN] = pd.to_datetime(result[TIME_COLUMN], errors="coerce")
    invalid = int(result[TIME_COLUMN].isna().sum())
    if invalid:
        raise ValueError(f"Invalid timestamps found: {invalid}")
    if not result[TIME_COLUMN].is_monotonic_increasing:
        result = result.sort_values(TIME_COLUMN).reset_index(drop=True)
        warnings.append("Input timestamps were not sorted; rows were sorted chronologically before target creation.")
    duplicate_count = int(result[TIME_COLUMN].duplicated().sum())
    if duplicate_count:
        warnings.append(f"Duplicate timestamps detected and retained for reporting: {duplicate_count}")
    return result, warnings


def construct_target(df: pd.DataFrame, pv_power_column: str, target_column: str, rated_power_kw: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    if target_column in df.columns:
        raise ValueError(f"{target_column} already exists in the input data. Refusing to overwrite an existing target.")
    if pv_power_column not in df.columns:
        raise ValueError(f"PV power column is missing: {pv_power_column}")
    if rated_power_kw <= 0:
        raise ValueError("rated_power_kw must be positive.")

    source = df.copy()
    pv_power = pd.to_numeric(source[pv_power_column], errors="coerce")
    target = pv_power.diff() / rated_power_kw
    invalid_mask = target.isna() | ~np.isfinite(target)
    invalid_count = int(invalid_mask.sum())
    invalid_indices = [int(i) for i in np.flatnonzero(invalid_mask.to_numpy())[:10]]

    with_target = source.loc[~invalid_mask].copy().reset_index(drop=True)
    with_target[target_column] = target.loc[~invalid_mask].to_numpy()
    ordered_columns = [column for column in with_target.columns if column != target_column] + [target_column]
    with_target = with_target[ordered_columns]

    summary = {
        "target_column": target_column,
        "pv_power_column": pv_power_column,
        "rated_power_kw": float(rated_power_kw),
        "formula": "ramp_rate_pu(t) = [P_Solar(t) - P_Solar(t-1)] / P_R",
        "rows_before_target_creation": int(len(df)),
        "invalid_target_rows_dropped": invalid_count,
        "invalid_target_row_indices_sample": invalid_indices,
        "rows_after_target_creation": int(len(with_target)),
        "first_row_policy": "drop_after_documenting",
        "target_appended_as_final_column": bool(with_target.columns[-1] == target_column),
    }
    return with_target, summary


def target_statistics(df: pd.DataFrame, target_column: str, stable_threshold: float) -> dict[str, Any]:
    target = pd.to_numeric(df[target_column], errors="coerce")
    ramp_up = target > stable_threshold
    ramp_down = target < -stable_threshold
    stable = ~(ramp_up | ramp_down)
    total = int(len(target))
    category_counts = {
        "ramp_down": int(ramp_down.sum()),
        "stable": int(stable.sum()),
        "ramp_up": int(ramp_up.sum()),
    }
    category_fraction = {name: _fraction(count, total) for name, count in category_counts.items()}
    return {
        "min": _float_or_none(target.min()),
        "max": _float_or_none(target.max()),
        "mean": _float_or_none(target.mean()),
        "std": _float_or_none(target.std()),
        "median": _float_or_none(target.median()),
        "p01": _float_or_none(target.quantile(0.01)),
        "p99": _float_or_none(target.quantile(0.99)),
        "missing_count": int(target.isna().sum()),
        "infinite_count": int(np.isinf(target.to_numpy(dtype=float)).sum()),
        "stable_threshold_pu_per_step": float(stable_threshold),
        "category_counts": category_counts,
        "category_fraction": category_fraction,
    }


def target_segment_rows(df: pd.DataFrame, target_column: str, config: dict[str, Any]) -> pd.DataFrame:
    target_cfg = config.get("target", {})
    segment_cfg = target_cfg.get("segment_diagnostics", {})
    elevation_threshold = float(segment_cfg.get("daylight_elevation_threshold_deg", 0.0))
    power_threshold = float(segment_cfg.get("productive_power_threshold_kw", 0.05))

    masks: dict[str, pd.Series] = {"all_data": pd.Series(True, index=df.index)}
    if "Elevation[deg]" in df.columns:
        masks["daylight_elevation_only"] = pd.to_numeric(df["Elevation[deg]"], errors="coerce") > elevation_threshold
    if "P_Solar[kW]" in df.columns:
        masks["productive_power_only"] = pd.to_numeric(df["P_Solar[kW]"], errors="coerce") > power_threshold
    if "daylight_elevation_only" in masks and "productive_power_only" in masks:
        masks["daylight_productive"] = masks["daylight_elevation_only"] & masks["productive_power_only"]

    rows: list[dict[str, Any]] = []
    for segment_name, mask in masks.items():
        subset = df.loc[mask.fillna(False)].copy()
        stats = target_statistics(subset, target_column, float(target_cfg.get("stable_threshold_pu_per_step", 0.01)))
        total = int(stats["category_counts"]["ramp_down"] + stats["category_counts"]["stable"] + stats["category_counts"]["ramp_up"])
        rows.append(
            {
                "segment": segment_name,
                "rule": segment_rule(segment_name, elevation_threshold, power_threshold),
                "row_count": int(len(subset)),
                "row_fraction_of_all": _fraction(int(len(subset)), int(len(df))),
                "target_min": stats["min"],
                "target_max": stats["max"],
                "target_mean": stats["mean"],
                "target_std": stats["std"],
                "target_missing_count": stats["missing_count"],
                "target_infinite_count": stats["infinite_count"],
                "ramp_down_count": stats["category_counts"]["ramp_down"],
                "stable_count": stats["category_counts"]["stable"],
                "ramp_up_count": stats["category_counts"]["ramp_up"],
                "ramp_down_pct": round(100.0 * _fraction(stats["category_counts"]["ramp_down"], total), 4),
                "stable_pct": round(100.0 * _fraction(stats["category_counts"]["stable"], total), 4),
                "ramp_up_pct": round(100.0 * _fraction(stats["category_counts"]["ramp_up"], total), 4),
            }
        )
    return pd.DataFrame(rows)


def segment_rule(segment_name: str, elevation_threshold: float, power_threshold: float) -> str:
    if segment_name == "all_data":
        return "all rows after target construction"
    if segment_name == "daylight_elevation_only":
        return f"Elevation[deg] > {elevation_threshold:g}"
    if segment_name == "productive_power_only":
        return f"P_Solar[kW] > {power_threshold:g}"
    if segment_name == "daylight_productive":
        return f"Elevation[deg] > {elevation_threshold:g} and P_Solar[kW] > {power_threshold:g}"
    return "custom segment"


def save_segment_plots(df: pd.DataFrame, target_column: str, segment_df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    primary = segment_df.loc[segment_df["segment"] == "daylight_productive"]
    if primary.empty:
        return paths
    rule = str(primary.iloc[0]["rule"])
    mask = pd.Series(True, index=df.index)
    if "Elevation[deg]" in df.columns and "Elevation[deg] >" in rule:
        threshold = float(rule.split("Elevation[deg] > ", 1)[1].split(" and", 1)[0])
        mask &= pd.to_numeric(df["Elevation[deg]"], errors="coerce") > threshold
    if "P_Solar[kW]" in df.columns and "P_Solar[kW] >" in rule:
        threshold = float(rule.split("P_Solar[kW] > ", 1)[1])
        mask &= pd.to_numeric(df["P_Solar[kW]"], errors="coerce") > threshold

    daylight = pd.to_numeric(df.loc[mask, target_column], errors="coerce").dropna()
    all_target = pd.to_numeric(df[target_column], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_target, bins=120, alpha=0.45, label="all_data")
    ax.hist(daylight, bins=120, alpha=0.45, label="daylight_productive")
    ax.set_title("ramp_rate_pu histogram by segment")
    ax.set_xlabel("ramp_rate_pu")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    path = plot_dir / "target_histogram_all_vs_daylight_productive.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths["target_histogram_all_vs_daylight_productive"] = str(path)
    return paths


def physical_relevance_score(column: str) -> float:
    low = column.lower()
    if "p_solar" in low or "pv" in low:
        return 1.0
    if "poa" in low:
        return 0.95
    if "ghi" in low:
        return 0.9
    if "elevation" in low:
        return 0.78
    if "azimuth" in low:
        return 0.68
    if "temperature" in low or "temp" in low:
        return 0.62
    if "humidity" in low:
        return 0.45
    if "wind" in low:
        return 0.38
    if "pressure" in low:
        return 0.28
    if "p_gaia" in low:
        return 0.2
    return 0.1


def deployment_feasibility_score(column: str) -> float:
    low = column.lower()
    if "future" in low or "lead" in low or "target" in low:
        return 0.0
    if "azimuth" in low or "elevation" in low:
        return 1.0
    if "p_solar" in low or "ghi" in low or "poa" in low:
        return 0.95
    return 0.85


def leakage_reason(column: str, target_column: str, patterns: list[str]) -> str | None:
    low = column.lower()
    if column == target_column:
        return "official target column"
    for pattern in patterns:
        pattern_low = str(pattern).lower()
        if pattern_low and pattern_low in low:
            return f"matches leakage-risk pattern `{pattern}`"
    return None


def pearson_corr(x: pd.Series, y: pd.Series) -> float | None:
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 3:
        return None
    if aligned.iloc[:, 0].nunique(dropna=True) <= 1 or aligned.iloc[:, 1].nunique(dropna=True) <= 1:
        return None
    return _float_or_none(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def feature_selection(
    df: pd.DataFrame,
    target_column: str,
    config: dict[str, Any],
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fs_cfg = config.get("feature_selection", {})
    train_fraction = float(fs_cfg.get("provisional_train_fraction", config.get("splits", {}).get("train_fraction", 0.7)))
    train_rows = max(1, int(len(df) * train_fraction))
    train_df = df.iloc[:train_rows].copy()
    target = pd.to_numeric(train_df[target_column], errors="coerce")
    patterns = list(fs_cfg.get("leakage_risk_patterns", []))
    redundancy_threshold = float(fs_cfg.get("redundancy_correlation_threshold", 0.98))
    min_combined_score = float(fs_cfg.get("min_combined_score", 0.18))
    min_selected = int(fs_cfg.get("min_selected_features", 5))
    max_selected = int(fs_cfg.get("max_selected_features", 8))
    force_keep = set(str(column) for column in fs_cfg.get("physics_source_force_keep", []))
    force_keep_reason = str(fs_cfg.get("physics_source_force_keep_reason", "retained as physics-source variable"))

    rows: list[dict[str, Any]] = []
    for column in df.columns:
        role = "candidate_feature"
        reason = leakage_reason(column, target_column, patterns)
        if column == TIME_COLUMN:
            role = "time_index"
            reason = "timestamp retained for chronological ordering, not used as model feature"
        is_numeric = bool(pd.api.types.is_numeric_dtype(df[column]))
        if not is_numeric and column not in (TIME_COLUMN, target_column):
            reason = reason or "non-numeric column"
        corr = None
        if is_numeric and not reason:
            corr = pearson_corr(pd.to_numeric(train_df[column], errors="coerce"), target)
        abs_corr = abs(corr) if corr is not None else 0.0
        physical = physical_relevance_score(column)
        feasible = deployment_feasibility_score(column)
        combined = (0.65 * abs_corr) + (0.25 * physical) + (0.10 * feasible)
        rows.append(
            {
                "column": column,
                "role": role,
                "is_numeric": is_numeric,
                "train_only_corr_with_target": corr,
                "abs_train_only_corr_with_target": round(float(abs_corr), 8),
                "physical_relevance_score": round(float(physical), 4),
                "deployment_feasibility_score": round(float(feasible), 4),
                "combined_score": round(float(combined), 8),
                "leakage_risk_reason": reason,
                "redundant_with": None,
                "redundancy_correlation": None,
                "physics_source_force_keep": column in force_keep,
                "selection_rationale": None,
                "selected": False,
                "drop_reason": None,
            }
        )

    report_df = pd.DataFrame(rows)
    candidate_df = report_df[
        (report_df["role"] == "candidate_feature")
        & (report_df["is_numeric"])
        & (report_df["leakage_risk_reason"].isna())
    ].copy()
    candidate_df = candidate_df.sort_values(["combined_score", "physical_relevance_score"], ascending=False)

    selected: list[str] = []
    selected_set: set[str] = set()
    for _, row in candidate_df.iterrows():
        column = str(row["column"])
        if len(selected) >= max_selected:
            break
        if float(row["combined_score"]) < min_combined_score and len(selected) >= min_selected:
            continue
        redundant_with = None
        redundant_corr = None
        for kept in selected:
            corr = pearson_corr(
                pd.to_numeric(train_df[column], errors="coerce"),
                pd.to_numeric(train_df[kept], errors="coerce"),
            )
            if corr is not None and abs(corr) >= redundancy_threshold:
                redundant_with = kept
                redundant_corr = corr
                break
        if redundant_with:
            mask = report_df["column"] == column
            report_df.loc[mask, "redundant_with"] = redundant_with
            report_df.loc[mask, "redundancy_correlation"] = round(float(redundant_corr), 8)
            if column not in force_keep:
                report_df.loc[mask, "drop_reason"] = f"redundant with {redundant_with}"
                continue
            report_df.loc[mask, "selection_rationale"] = force_keep_reason
        selected.append(column)
        selected_set.add(column)
        report_df.loc[report_df["column"] == column, "selected"] = True
        if column not in force_keep:
            report_df.loc[report_df["column"] == column, "selection_rationale"] = "selected by train-only score, physical relevance, feasibility and redundancy checks"

    for index, row in report_df.iterrows():
        if bool(row["selected"]):
            continue
        if pd.notna(row["drop_reason"]) and row["drop_reason"]:
            continue
        if row["role"] == "time_index":
            report_df.at[index, "drop_reason"] = "retained only as chronological index column"
        elif row["column"] == target_column:
            report_df.at[index, "drop_reason"] = "official target, not a feature"
        elif pd.notna(row["leakage_risk_reason"]) and row["leakage_risk_reason"]:
            report_df.at[index, "drop_reason"] = row["leakage_risk_reason"]
        elif not row["is_numeric"]:
            report_df.at[index, "drop_reason"] = "non-numeric"
        elif row["column"] not in selected_set:
            report_df.at[index, "drop_reason"] = "lower rank after train-only scoring, redundancy and max-feature limit"

    dropped_df = report_df[~report_df["selected"]].copy()
    info = {
        "provisional_train_fraction": train_fraction,
        "provisional_train_rows": int(train_rows),
        "provisional_train_purpose": fs_cfg.get("provisional_split_purpose", "train-only feature ranking before official split"),
        "redundancy_correlation_threshold": redundancy_threshold,
        "min_combined_score": min_combined_score,
        "max_selected_features": max_selected,
        "physics_source_force_keep": sorted(force_keep),
        "physics_source_force_keep_reason": force_keep_reason,
        "selected_feature_count": len(selected),
    }
    return selected, report_df, dropped_df, info


def save_target_plots(df: pd.DataFrame, target_column: str, output_dir: Path) -> dict[str, str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    target = pd.to_numeric(df[target_column], errors="coerce")
    paths: dict[str, str] = {}

    max_points = 20000
    step = max(1, len(df) // max_points)
    sample = df.iloc[::step].copy()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pd.to_datetime(sample[TIME_COLUMN]), sample[target_column], linewidth=0.7)
    ax.set_title("ramp_rate_pu time series")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("ramp_rate_pu")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = plot_dir / "target_time_series.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths["target_time_series"] = str(path)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(target.dropna(), bins=120)
    ax.set_title("ramp_rate_pu histogram")
    ax.set_xlabel("ramp_rate_pu")
    ax.set_ylabel("count")
    fig.tight_layout()
    path = plot_dir / "target_histogram.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths["target_histogram"] = str(path)

    relation_columns = [
        column
        for column in df.columns
        if column != target_column and any(token in column.lower() for token in ["p_solar", "ghi", "poa"])
    ][:3]
    if relation_columns:
        sample = df.iloc[:: max(1, len(df) // 10000)].copy()
        fig, axes = plt.subplots(1, len(relation_columns), figsize=(5 * len(relation_columns), 4), squeeze=False)
        for ax, column in zip(axes[0], relation_columns):
            ax.scatter(sample[column], sample[target_column], s=4, alpha=0.25)
            ax.set_title(f"{column} vs ramp_rate_pu")
            ax.set_xlabel(column)
            ax.set_ylabel("ramp_rate_pu")
        fig.tight_layout()
        path = plot_dir / "target_relation_pv_irradiance.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths["target_relation_pv_irradiance"] = str(path)

    return paths


def write_alignment_note(output_root: Path) -> Path:
    text = "\n".join(
        [
            "# Target Alignment and Anti-Leakage Note",
            "",
            TARGET_ALIGNMENT_NOTE,
            "",
            "Implication for Notebook 04:",
            "- Candidate same-row variables in Notebook 02 are not direct supervised-learning rows.",
            "- Notebook 04 must build input windows ending at forecast origin `t`.",
            "- For horizon `H`, labels must be shifted to `ramp_rate_pu(t+H)`.",
            "- Same-row `P_Solar(t)` must never be used to predict same-row `ramp_rate_pu(t)`.",
        ]
    )
    path = output_root / "target_alignment_anti_leakage_note.md"
    path.write_text(text + "\n", encoding="utf-8")
    return path


def validation_checks(full_df: pd.DataFrame, selected_df: pd.DataFrame, selected_features: list[str], target_column: str, patterns: list[str]) -> dict[str, bool]:
    leakage_selected = [feature for feature in selected_features if leakage_reason(feature, target_column, patterns)]
    checks = {
        "target_exists_in_full_dataset": target_column in full_df.columns,
        "target_exists_in_selected_dataset": target_column in selected_df.columns,
        "target_last_column_full_dataset": bool(full_df.columns[-1] == target_column),
        "target_last_column_selected_dataset": bool(selected_df.columns[-1] == target_column),
        "target_not_in_feature_list": target_column not in selected_features,
        "no_leakage_risk_feature_selected": len(leakage_selected) == 0,
        "selected_dataset_has_fewer_columns_than_full_dataset": selected_df.shape[1] < full_df.shape[1],
        "timestamps_sorted_full_dataset": bool(pd.to_datetime(full_df[TIME_COLUMN]).is_monotonic_increasing),
        "timestamps_sorted_selected_dataset": bool(pd.to_datetime(selected_df[TIME_COLUMN]).is_monotonic_increasing),
        "outputs_have_same_row_count": int(len(full_df)) == int(len(selected_df)),
    }
    return checks


def write_local_runner_report(report: dict[str, Any]) -> None:
    output_dir = os.environ.get("LOCAL_OUTPUT_DIR")
    if not output_dir:
        return
    payload = {
        "notebook_name": report["notebook_name"],
        "final_status": report["final_status"],
        "data_shapes": {
            "full_with_target": report["data_shapes"]["full_with_target"],
            "selected_with_target": report["data_shapes"]["selected_with_target"],
        },
        "split_summary": {
            "official_split_created": False,
            "provisional_train_rows_for_feature_selection": report["feature_selection"]["provisional_train_rows"],
            "provisional_train_fraction_for_feature_selection": report["feature_selection"]["provisional_train_fraction"],
        },
        "metrics": {
            "selected_feature_count": report["feature_selection"]["selected_feature_count"],
            "rows_after_target_creation": report["target_creation"]["rows_after_target_creation"],
            "target_mean": report["target_statistics"]["mean"],
            "target_std": report["target_statistics"]["std"],
        },
        "report_path": report["output_files"]["notebook_02_final_report"],
    }
    write_json(Path(output_dir) / "local_clearml_report.json", payload)


def run_notebook02_target_features(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    root = project_root()
    output_root = root / "outputs" / NOTEBOOK_NAME
    run_state = output_root / "run_state"
    output_root.mkdir(parents=True, exist_ok=True)
    run_state.mkdir(parents=True, exist_ok=True)

    target_column = str(cfg.get("target", {}).get("official_name") or cfg["data"]["official_target"])
    pv_power_column = str(cfg["data"]["pv_power_column"])
    rated_power_kw = float(cfg["project"]["rated_power_kw"])
    primary_resolution = str(cfg["data"]["primary_resolution_candidate"])
    stable_threshold = float(cfg.get("target", {}).get("stable_threshold_pu_per_step", 0.01))
    leakage_patterns = list(cfg.get("feature_selection", {}).get("leakage_risk_patterns", []))

    warnings: list[str] = []
    errors: list[str] = []
    output_files: dict[str, str] = {}
    plots: dict[str, str] = {}

    started = utc_now()
    try:
        notebook01_report = read_notebook01_report(root)
        input_file = input_file_from_notebook01(notebook01_report, primary_resolution)
        source = pd.read_csv(input_file)
        source, time_warnings = ensure_timestamp(source)
        warnings.extend(time_warnings)
        target_preexisting = target_column in source.columns
        if target_preexisting:
            raise ValueError(f"{target_column} already exists before Notebook 02 target creation.")
        write_status(
            run_state / "section_01_input_validation.json",
            section_status(
                "input_validation",
                started,
                "success",
                "Notebook 01 clean primary-resolution input validated.",
                {
                    "input_file": str(input_file),
                    "rows": int(len(source)),
                    "columns": int(source.shape[1]),
                    "target_preexisting": target_preexisting,
                },
            ),
        )
    except Exception:
        errors.append(traceback.format_exc())
        write_status(run_state / "section_01_input_validation.json", section_status("input_validation", started, "failed", "Input validation failed.", error_message=errors[-1]))
        raise

    started = utc_now()
    try:
        full_with_target, target_creation = construct_target(source, pv_power_column, target_column, rated_power_kw)
        target_dir = root / "outputs" / primary_resolution / "notebook_02_target_features"
        target_dir.mkdir(parents=True, exist_ok=True)
        full_path = target_dir / "original_data_all_features_clean_with_target.csv"
        write_csv(full_path, full_with_target)
        output_files["original_data_all_features_clean_with_target"] = str(full_path)
        write_status(
            run_state / "section_02_target_construction.json",
            section_status(
                "target_construction",
                started,
                "success",
                "Official ramp_rate_pu target created and appended as final column.",
                target_creation,
            ),
        )
    except Exception:
        errors.append(traceback.format_exc())
        write_status(run_state / "section_02_target_construction.json", section_status("target_construction", started, "failed", "Target construction failed.", error_message=errors[-1]))
        raise

    started = utc_now()
    try:
        stats = target_statistics(full_with_target, target_column, stable_threshold)
        plots = save_target_plots(full_with_target, target_column, output_root)
        segment_df = target_segment_rows(full_with_target, target_column, cfg)
        segment_path = output_root / "target_analysis_by_segment.csv"
        write_csv(segment_path, segment_df)
        segment_plots = save_segment_plots(full_with_target, target_column, segment_df, output_root)
        plots.update(segment_plots)
        relation_columns = [
            column
            for column in full_with_target.columns
            if column not in (TIME_COLUMN, target_column) and any(token in column.lower() for token in ["p_solar", "ghi", "poa"])
        ]
        relation_summary = {
            column: pearson_corr(pd.to_numeric(full_with_target[column], errors="coerce"), pd.to_numeric(full_with_target[target_column], errors="coerce"))
            for column in relation_columns
        }
        target_analysis = {
            "target_column": target_column,
            "rated_power_kw": rated_power_kw,
            "target_statistics": stats,
            "ramp_category_distribution": stats["category_counts"],
            "ramp_category_fraction": stats["category_fraction"],
            "relation_with_pv_irradiance_variables": relation_summary,
            "target_quality_checks": {
                "missing_count": stats["missing_count"],
                "infinite_count": stats["infinite_count"],
                "abs_value_gt_1_count": int((full_with_target[target_column].abs() > 1.0).sum()),
                "timestamp_sorted": bool(pd.to_datetime(full_with_target[TIME_COLUMN]).is_monotonic_increasing),
                "duplicate_timestamps": int(pd.to_datetime(full_with_target[TIME_COLUMN]).duplicated().sum()),
            },
            "segment_diagnostics": {
                "csv_path": str(segment_path),
                "primary_daylight_productive_rule": cfg.get("target", {}).get("segment_diagnostics", {}).get("primary_daylight_productive_rule"),
                "segments": segment_df.to_dict(orient="records"),
            },
            "plots": plots,
        }
        analysis_path = output_root / "target_analysis_report.json"
        write_json(analysis_path, target_analysis)
        output_files["target_analysis_report"] = str(analysis_path)
        output_files["target_analysis_by_segment"] = str(segment_path)
        write_status(
            run_state / "section_03_target_analysis.json",
            section_status(
                "target_analysis",
                started,
                "success",
                "Target distribution, time behavior and quality checks completed.",
                {"target_min": stats["min"], "target_max": stats["max"], "target_std": stats["std"]},
            ),
        )
    except Exception:
        errors.append(traceback.format_exc())
        write_status(run_state / "section_03_target_analysis.json", section_status("target_analysis", started, "failed", "Target analysis failed.", error_message=errors[-1]))
        raise

    started = utc_now()
    try:
        selected_features, selection_report, dropped_report, selection_info = feature_selection(full_with_target, target_column, cfg)
        selected_columns = [TIME_COLUMN] + selected_features + [target_column]
        selected_df = full_with_target[selected_columns].copy()
        selected_path = target_dir / "selected_features_with_target.csv"
        write_csv(selected_path, selected_df)
        feature_report_path = output_root / "feature_selection_report.csv"
        dropped_path = output_root / "dropped_features_report.csv"
        selected_list_path = output_root / "selected_features_list.json"
        write_csv(feature_report_path, selection_report)
        write_csv(dropped_path, dropped_report)
        selected_list = {
            "feature_columns": selected_features,
            "target_column": target_column,
            "time_index_column": TIME_COLUMN,
            "selected_dataset_columns": selected_columns,
            "anti_leakage_note": "Timestamp is retained only for chronological alignment. Feature ranking used a provisional chronological train segment only; the official split is deferred to Notebook 04.",
            "target_alignment_anti_leakage_note": TARGET_ALIGNMENT_NOTE,
            **selection_info,
        }
        write_json(selected_list_path, selected_list)
        alignment_note_path = write_alignment_note(output_root)
        output_files["selected_features_with_target"] = str(selected_path)
        output_files["feature_selection_report"] = str(feature_report_path)
        output_files["dropped_features_report"] = str(dropped_path)
        output_files["selected_features_list"] = str(selected_list_path)
        output_files["target_alignment_anti_leakage_note"] = str(alignment_note_path)
        write_status(
            run_state / "section_04_feature_selection.json",
            section_status(
                "feature_selection",
                started,
                "success",
                "Train-only provisional feature ranking and leakage checks completed.",
                {"selected_feature_count": len(selected_features), "provisional_train_rows": selection_info["provisional_train_rows"]},
            ),
        )
    except Exception:
        errors.append(traceback.format_exc())
        write_status(run_state / "section_04_feature_selection.json", section_status("feature_selection", started, "failed", "Feature selection failed.", error_message=errors[-1]))
        raise

    started = utc_now()
    checks = validation_checks(full_with_target, selected_df, selected_features, target_column, leakage_patterns)
    final_status = "success" if all(checks.values()) and not errors else "failed"
    try:
        report_path = output_root / "notebook_02_final_report.json"
        output_files["notebook_02_final_report"] = str(report_path)
        report = {
            "notebook_name": NOTEBOOK_NAME,
            "execution_type": "local",
            "input_file": str(input_file),
            "primary_resolution": primary_resolution,
            "target_column": target_column,
            "pv_power_column": pv_power_column,
            "target_creation": target_creation,
            "target_statistics": stats,
            "target_segment_diagnostics": segment_df.to_dict(orient="records"),
            "ramp_category_distribution": stats["category_counts"],
            "feature_selection": selection_info,
            "target_alignment_anti_leakage_note": TARGET_ALIGNMENT_NOTE,
            "selected_features": selected_features,
            "selected_feature_count": len(selected_features),
            "original_feature_count": int(source.shape[1] - 1),
            "data_shapes": {
                "source_clean": {"rows": int(source.shape[0]), "columns": int(source.shape[1])},
                "full_with_target": {"rows": int(full_with_target.shape[0]), "columns": int(full_with_target.shape[1])},
                "selected_with_target": {"rows": int(selected_df.shape[0]), "columns": int(selected_df.shape[1])},
            },
            "validation_checks": checks,
            "leakage_checks_passed": bool(checks["target_not_in_feature_list"] and checks["no_leakage_risk_feature_selected"]),
            "official_split_created": False,
            "future_targets_created": False,
            "lookback_horizon_windows_created": False,
            "heavy_training_started": False,
            "output_files": output_files,
            "plots": plots,
            "warnings": warnings,
            "errors": errors,
            "final_status": final_status,
        }
        write_json(report_path, report)
        write_local_runner_report(report)
        write_status(
            run_state / "section_05_validation_outputs.json",
            section_status(
                "validation_outputs",
                started,
                final_status,
                "Notebook 02 output validations completed.",
                checks,
                None if final_status == "success" else "One or more validation checks failed.",
            ),
        )
        return report
    except Exception:
        errors.append(traceback.format_exc())
        write_status(run_state / "section_05_validation_outputs.json", section_status("validation_outputs", started, "failed", "Final report writing failed.", error_message=errors[-1]))
        raise
