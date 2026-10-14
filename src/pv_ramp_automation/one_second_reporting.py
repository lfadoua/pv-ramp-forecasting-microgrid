from __future__ import annotations

import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from .one_second_extension import FixedFrameHDFReader, TIME_COLUMN, load_npy_memmap, write_json


COLORS = {
    "truth": "#202124",
    "hgb": "#0072B2",
    "persistence": "#D55E00",
    "reference": "#009E73",
    "s3": "#CC79A7",
    "one_second": "#0072B2",
    "one_minute": "#E69F00",
    "neutral": "#6B7280",
    "positive": "#009E73",
    "negative": "#D55E00",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def select_representative_periods(
    averaging_detail: pd.DataFrame,
    test_start: pd.Timestamp,
    duration_minutes: int = 20,
) -> pd.DataFrame:
    frame = averaging_detail.copy()
    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN])
    frame = frame[frame[TIME_COLUMN] >= test_start].sort_values(TIME_COLUMN).reset_index(drop=True)
    window = int(duration_minutes)
    frame["variability_score"] = (
        frame["pv_std"].rolling(window, min_periods=window).mean()
        + frame["poa_std"].rolling(window, min_periods=window).mean()
        + frame["ghi_std"].rolling(window, min_periods=window).mean()
    )
    frame["activity_score"] = frame["pv_mean"].rolling(window, min_periods=window).mean()
    candidates = frame[(frame["activity_score"] > 0.25) & frame["variability_score"].notna()].copy()
    if len(candidates) < 3:
        candidates = frame[frame["variability_score"].notna()].copy()
    if len(candidates) < 3:
        raise ValueError("Insufficient test rows for representative-period selection.")
    scores = candidates["variability_score"]
    target_scores = {
        "stable": float(scores.quantile(0.05)),
        "moderate": float(scores.quantile(0.50)),
        "rapid_ramp": float(scores.quantile(0.995)),
    }
    rows: list[dict[str, Any]] = []
    selected_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for period_type in ("stable", "moderate", "rapid_ramp"):
        ranked = candidates.assign(distance=np.abs(candidates["variability_score"] - target_scores[period_type])).sort_values("distance")
        chosen = None
        for _, candidate in ranked.iterrows():
            end = pd.Timestamp(candidate[TIME_COLUMN])
            start = end - pd.Timedelta(minutes=duration_minutes - 1)
            end_plot = end + pd.Timedelta(seconds=59)
            overlaps = any(not (end_plot < old_start or start > old_end) for old_start, old_end in selected_intervals)
            if not overlaps:
                chosen = (candidate, start, end_plot)
                break
        if chosen is None:
            candidate = ranked.iloc[0]
            end = pd.Timestamp(candidate[TIME_COLUMN])
            chosen = (candidate, end - pd.Timedelta(minutes=duration_minutes - 1), end + pd.Timedelta(seconds=59))
        candidate, start, end_plot = chosen
        selected_intervals.append((start, end_plot))
        rows.append(
            {
                "period_type": period_type,
                "start_timestamp": start,
                "end_timestamp": end_plot,
                "duration_minutes": duration_minutes,
                "variability_score": float(candidate["variability_score"]),
                "selection_rule": (
                    "5th percentile of rolling test variability"
                    if period_type == "stable"
                    else "median rolling test variability"
                    if period_type == "moderate"
                    else "99.5th percentile of rolling test variability"
                ),
                "used_for_model_selection": False,
                "notes": "Illustrative test period selected after model fitting; not used for tuning or metric selection.",
            }
        )
    return pd.DataFrame(rows)


def _raw_interval(reader: FixedFrameHDFReader, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    first = reader.timestamp_at(0)
    start_row = max(0, int((start - first).total_seconds()))
    stop_row = min(reader.row_count, int((end - first).total_seconds()) + 1)
    timestamps, values = reader.read(start_row, stop_row)
    return pd.DataFrame(
        {
            TIME_COLUMN: timestamps.astype("datetime64[ns]"),
            "P_Solar[kW]": values[:, reader.column_indices["P_Solar[kW]"]],
        }
    )


def _format_time_axis(axis: plt.Axes) -> None:
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))


def _period_power_figure(
    reader: FixedFrameHDFReader,
    detail: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    path: Path,
) -> None:
    raw = _raw_interval(reader, start, end)
    minute = detail[(detail[TIME_COLUMN] >= start.floor("min")) & (detail[TIME_COLUMN] <= end.floor("min"))]
    fig, axis = plt.subplots(figsize=(7.2, 2.7))
    axis.plot(raw[TIME_COLUMN], raw["P_Solar[kW]"], color=COLORS["one_second"], linewidth=0.75, label="1 s PV power")
    axis.step(minute[TIME_COLUMN], minute["pv_mean"], where="post", color=COLORS["one_minute"], linewidth=1.35, label="1 min mean PV power")
    axis.set_ylabel("PV power (kW)")
    axis.set_xlabel("Time")
    axis.legend(frameon=False, ncol=2)
    _format_time_axis(axis)
    save_figure(fig, path)


def _period_ramp_figure(
    reader: FixedFrameHDFReader,
    detail: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    path: Path,
) -> None:
    raw = _raw_interval(reader, start, end)
    raw["ramp"] = raw["P_Solar[kW]"].diff() / 10.0
    minute = detail[(detail[TIME_COLUMN] >= start.floor("min")) & (detail[TIME_COLUMN] <= end.floor("min"))]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.0), sharex=True)
    axes[0].plot(raw[TIME_COLUMN], raw["ramp"], color=COLORS["one_second"], linewidth=0.7)
    axes[0].axhline(0.0, color="#444444", linewidth=0.6)
    axes[0].set_ylabel("1 s ramp (p.u.)")
    axes[1].step(minute[TIME_COLUMN], minute["one_minute_ramp_pu"], where="post", color=COLORS["one_minute"], linewidth=1.1)
    axes[1].axhline(0.0, color="#444444", linewidth=0.6)
    axes[1].set_ylabel("1 min ramp (p.u.)")
    axes[1].set_xlabel("Time")
    _format_time_axis(axes[1])
    save_figure(fig, path)


def _prediction_window(frame: pd.DataFrame, preferred_start: pd.Timestamp, preferred_end: pd.Timestamp, points: int) -> pd.DataFrame:
    data = frame[frame["split"] == "test"].copy()
    data[TIME_COLUMN] = pd.to_datetime(data[TIME_COLUMN])
    preferred = data[(data[TIME_COLUMN] >= preferred_start) & (data[TIME_COLUMN] <= preferred_end)]
    if len(preferred) >= 8:
        return preferred
    rolling = data["y_true"].abs().rolling(points, min_periods=max(4, points // 2)).mean()
    end_index = int(rolling.fillna(-np.inf).to_numpy().argmax())
    start_index = max(0, end_index - points + 1)
    return data.iloc[start_index : end_index + 1]


def _true_prediction_figure(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, ylabel: str, path: Path) -> None:
    selected = _prediction_window(frame, start, end, points=50)
    fig, axis = plt.subplots(figsize=(7.2, 2.8))
    axis.plot(selected[TIME_COLUMN], selected["y_true"], color=COLORS["truth"], linewidth=1.05, label="Observed")
    axis.plot(selected[TIME_COLUMN], selected["y_pred"], color=COLORS["hgb"], linewidth=1.0, label="HGB forecast")
    axis.axhline(0.0, color="#555555", linewidth=0.6)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Target timestamp")
    axis.legend(frameon=False, ncol=2)
    _format_time_axis(axis)
    save_figure(fig, path)


def _residual_figure(frame: pd.DataFrame, path: Path) -> None:
    residual = frame.loc[frame["split"] == "test", "residual"].to_numpy(dtype=np.float64)
    limits = np.quantile(residual, [0.005, 0.995])
    clipped = residual[(residual >= limits[0]) & (residual <= limits[1])]
    fig, axis = plt.subplots(figsize=(5.5, 2.8))
    axis.hist(clipped, bins=70, density=True, color=COLORS["hgb"], alpha=0.8)
    axis.axvline(0.0, color=COLORS["truth"], linewidth=0.8)
    axis.set_xlabel("Residual, observed - forecast (p.u.)")
    axis.set_ylabel("Density")
    save_figure(fig, path)


def _magnitude_error_figure(frame: pd.DataFrame, path: Path) -> None:
    test = frame[frame["split"] == "test"]
    grouped = test.groupby("target_magnitude_bin", sort=False, observed=True)["absolute_error"].agg(["mean", "count"]).reset_index()
    fig, axis = plt.subplots(figsize=(6.2, 3.0))
    axis.bar(np.arange(len(grouped)), grouped["mean"], color=COLORS["hgb"], width=0.72)
    axis.set_xticks(np.arange(len(grouped)), grouped["target_magnitude_bin"], rotation=28, ha="right")
    axis.set_ylabel("Mean absolute error (p.u.)")
    axis.set_xlabel("Training-defined target-magnitude bin")
    save_figure(fig, path)


def generate_required_figures(
    reader: FixedFrameHDFReader,
    figures_dir: Path,
    averaging_detail_path: Path,
    one_second_ramp_path: Path,
    representative_periods: pd.DataFrame,
    predictions_dir: Path,
    experiment_results: pd.DataFrame,
    ablation: pd.DataFrame,
) -> pd.DataFrame:
    configure_matplotlib()
    figures_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.read_parquet(averaging_detail_path)
    detail[TIME_COLUMN] = pd.to_datetime(detail[TIME_COLUMN])
    periods = {row["period_type"]: row for _, row in representative_periods.iterrows()}
    inventory: list[dict[str, Any]] = []

    power_specs = [
        (1, "stable", "one_second_vs_one_minute_power_stable", "One-second PV power and the corresponding one-minute mean during the stable illustrative period."),
        (2, "moderate", "one_second_vs_one_minute_power_moderate", "One-second PV power and the corresponding one-minute mean during the moderately variable illustrative period."),
        (3, "rapid_ramp", "one_second_vs_one_minute_power_rapid", "One-second PV power and the corresponding one-minute mean during the abrupt rapid-ramp illustrative period."),
    ]
    for number, period_type, slug, caption in power_specs:
        row = periods[period_type]
        path = figures_dir / f"fig{number:02d}_{slug}.png"
        _period_power_figure(reader, detail, pd.Timestamp(row["start_timestamp"]), pd.Timestamp(row["end_timestamp"]), path)
        inventory.append({"figure": number, "path": str(path), "caption": caption, "category": "data_resolution"})

    ramp_specs = [
        (4, "stable", "one_second_vs_one_minute_ramp_stable", "One-second and one-minute PV ramps during the stable illustrative period; separate panels preserve each target scale."),
        (5, "moderate", "one_second_vs_one_minute_ramp_moderate", "One-second and one-minute PV ramps during the moderately variable illustrative period."),
        (6, "rapid_ramp", "one_second_vs_one_minute_ramp_rapid", "One-second and one-minute PV ramps during the abrupt rapid-ramp illustrative period."),
    ]
    for number, period_type, slug, caption in ramp_specs:
        row = periods[period_type]
        path = figures_dir / f"fig{number:02d}_{slug}.png"
        _period_ramp_figure(reader, detail, pd.Timestamp(row["start_timestamp"]), pd.Timestamp(row["end_timestamp"]), path)
        inventory.append({"figure": number, "path": str(path), "caption": caption, "category": "data_resolution"})

    one_second = load_npy_memmap(one_second_ramp_path)
    one_second_sample = np.asarray(one_second[::100], dtype=np.float64)
    minute_ramp = detail["one_minute_ramp_pu"].to_numpy(dtype=np.float64)
    positive = np.concatenate((np.abs(one_second_sample), np.abs(minute_ramp)))
    positive = positive[positive > 0]
    bins = np.geomspace(max(1.0e-7, np.quantile(positive, 0.001)), np.quantile(positive, 0.9995), 70)
    fig, axis = plt.subplots(figsize=(6.2, 3.0))
    axis.hist(np.abs(one_second_sample), bins=bins, density=True, histtype="step", linewidth=1.4, color=COLORS["one_second"], label="1 s ramp")
    axis.hist(np.abs(minute_ramp), bins=bins, density=True, histtype="step", linewidth=1.4, color=COLORS["one_minute"], label="1 min ramp")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Absolute ramp magnitude (p.u.)")
    axis.set_ylabel("Density")
    axis.legend(frameon=False)
    path = figures_dir / "fig07_absolute_ramp_distributions.png"
    save_figure(fig, path)
    inventory.append({"figure": 7, "path": str(path), "caption": "Distributions of absolute one-second and one-minute ramp magnitudes. The one-second curve uses a deterministic 1-in-100 plotting sample; reported statistics use all rows.", "category": "data_resolution"})

    scatter = detail.iloc[:: max(1, len(detail) // 80_000)]
    fig, axis = plt.subplots(figsize=(4.8, 3.8))
    axis.scatter(scatter["pv_max_abs_ramp"], np.abs(scatter["one_minute_ramp_pu"]), s=5, alpha=0.25, color=COLORS["one_second"], linewidths=0)
    limit = float(np.quantile(np.concatenate((scatter["pv_max_abs_ramp"], np.abs(scatter["one_minute_ramp_pu"]))), 0.995))
    axis.plot([0, limit], [0, limit], linestyle="--", linewidth=0.8, color=COLORS["neutral"])
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xlabel("Maximum within-minute |1 s ramp| (p.u.)")
    axis.set_ylabel("Absolute 1 min ramp (p.u.)")
    path = figures_dir / "fig08_within_minute_peak_vs_minute_ramp.png"
    save_figure(fig, path)
    inventory.append({"figure": 8, "path": str(path), "caption": "Maximum absolute one-second ramp within each minute versus the absolute aggregated one-minute ramp.", "category": "data_resolution"})

    attenuation = detail["attenuation_indicator"].replace([np.inf, -np.inf], np.nan).dropna()
    limits = attenuation.quantile([0.005, 0.995]).to_numpy()
    fig, axis = plt.subplots(figsize=(5.5, 2.8))
    axis.hist(attenuation.clip(limits[0], limits[1]), bins=70, color=COLORS["one_minute"], alpha=0.85)
    axis.axvline(0.0, color=COLORS["truth"], linewidth=0.8)
    axis.set_xlabel("Descriptive attenuation indicator")
    axis.set_ylabel("Minutes")
    path = figures_dir / "fig09_averaging_attenuation_distribution.png"
    save_figure(fig, path)
    inventory.append({"figure": 9, "path": str(path), "caption": "Distribution of the descriptive averaging-attenuation indicator; the display is clipped at the 0.5th and 99.5th percentiles.", "category": "data_resolution"})

    cancellation_values = [
        float(detail["pv_cancellation_flag"].mean()),
        float(detail["both_strong_directions"].mean()),
        float(detail["high_variability_small_minute_ramp"].mean()),
    ]
    fig, axis = plt.subplots(figsize=(6.2, 3.0))
    axis.bar(np.arange(3), cancellation_values, color=[COLORS["neutral"], COLORS["negative"], COLORS["one_minute"]])
    axis.set_xticks(np.arange(3), ["Both signs", "Strong both signs", "High variability,\nsmall minute ramp"])
    axis.set_ylabel("Fraction of minutes")
    axis.set_ylim(0, max(cancellation_values) * 1.18 if max(cancellation_values) else 1)
    path = figures_dir / "fig10_sign_cancellation_frequency.png"
    save_figure(fig, path)
    inventory.append({"figure": 10, "path": str(path), "caption": "Frequency of within-minute sign cancellation, strong changes in both directions, and high variability hidden by a small one-minute ramp.", "category": "data_resolution"})

    prediction_files = {
        "S1": predictions_dir / "s1_hist_gradient_boosting_predictions.parquet",
        "S2": predictions_dir / "s2_hist_gradient_boosting_predictions.parquet",
        "S3": predictions_dir / "s3_hist_gradient_boosting_predictions.parquet",
        "reference": predictions_dir / "existing_one_minute_reference_aligned_s3_predictions.parquet",
    }
    predictions = {key: pd.read_parquet(path) for key, path in prediction_files.items()}
    rapid = periods["rapid_ramp"]
    rapid_start = pd.Timestamp(rapid["start_timestamp"])
    rapid_end = pd.Timestamp(rapid["end_timestamp"])
    model_specs = [
        (11, "S1", "s1_true_vs_predicted", "Next-second ramp (p.u.)", "Observed and HGB-predicted S1 next-second ramps over a rapid-variation window."),
        (12, "S2", "s2_true_vs_predicted", "Delayed 1 s ramp (p.u.)", "Observed and HGB-predicted S2 one-second ramps at a 60-second lead."),
        (13, "S3", "s3_true_vs_predicted", "Next-minute ramp (p.u.)", "Observed and HGB-predicted S3 one-minute ramps using one-second-derived descriptors."),
    ]
    for number, key, slug, ylabel, caption in model_specs:
        path = figures_dir / f"fig{number:02d}_{slug}.png"
        _true_prediction_figure(predictions[key], rapid_start, rapid_end, ylabel, path)
        inventory.append({"figure": number, "path": str(path), "caption": caption, "category": "model"})

    s3_window = _prediction_window(predictions["S3"], rapid_start, rapid_end, points=30)
    reference_test = predictions["reference"].copy()
    reference_test[TIME_COLUMN] = pd.to_datetime(reference_test[TIME_COLUMN])
    compare = s3_window[[TIME_COLUMN, "y_true", "y_pred"]].rename(columns={"y_pred": "s3_pred"}).merge(
        reference_test[[TIME_COLUMN, "y_pred"]].rename(columns={"y_pred": "reference_pred"}),
        on=TIME_COLUMN,
        how="inner",
    )
    fig, axis = plt.subplots(figsize=(7.2, 2.9))
    axis.plot(compare[TIME_COLUMN], compare["y_true"], color=COLORS["truth"], linewidth=1.1, label="Observed")
    axis.plot(compare[TIME_COLUMN], compare["reference_pred"], color=COLORS["reference"], linewidth=1.0, label="Existing 1 min input")
    axis.plot(compare[TIME_COLUMN], compare["s3_pred"], color=COLORS["s3"], linewidth=1.0, label="S3 high-frequency input")
    axis.set_ylabel("Next-minute ramp (p.u.)")
    axis.set_xlabel("Target timestamp")
    axis.legend(frameon=False, ncol=3)
    _format_time_axis(axis)
    path = figures_dir / "fig14_aligned_existing_vs_s3_predictions.png"
    save_figure(fig, path)
    inventory.append({"figure": 14, "path": str(path), "caption": "Aligned observed ramps, existing one-minute-input HGB forecasts, and S3 high-frequency-input HGB forecasts.", "category": "model"})

    for number, key in ((15, "S1"), (16, "S2"), (17, "S3")):
        path = figures_dir / f"fig{number:02d}_{key.lower()}_residual_distribution.png"
        _residual_figure(predictions[key], path)
        inventory.append({"figure": number, "path": str(path), "caption": f"Test residual distribution for {key}; the display excludes only the outer 0.5% at each tail.", "category": "model"})

    for number, key in ((18, "S1"), (19, "S2"), (20, "S3")):
        path = figures_dir / f"fig{number:02d}_{key.lower()}_error_by_target_magnitude.png"
        _magnitude_error_figure(predictions[key], path)
        inventory.append({"figure": number, "path": str(path), "caption": f"Mean absolute test error for {key} across target-magnitude bins defined without test-set tuning.", "category": "model"})

    hgb_rows = experiment_results[experiment_results["model"] == "HistGradientBoosting"].set_index("experiment_id")
    metrics = ["strong_ramp_precision", "strong_ramp_recall", "strong_ramp_F1"]
    fig, axis = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(3)
    width = 0.23
    for index, metric in enumerate(metrics):
        values = [float(hgb_rows.loc[experiment, metric]) for experiment in ("S1", "S2", "S3")]
        axis.bar(x + (index - 1) * width, values, width=width, label=metric.replace("strong_ramp_", "").replace("_", " ").title())
    axis.set_xticks(x, ["S1", "S2", "S3"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Event score")
    axis.legend(frameon=False, ncol=3)
    path = figures_dir / "fig21_strong_ramp_detection_summary.png"
    save_figure(fig, path)
    inventory.append({"figure": 21, "path": str(path), "caption": "Strong-ramp precision, recall, and F1 for the transferred HGB in S1, S2, and S3.", "category": "model"})

    plot_rows = experiment_results[experiment_results["model"].isin(["HistGradientBoosting", "Persistence"])].copy()
    fig, axis = plt.subplots(figsize=(5.8, 3.8))
    for model, group in plot_rows.groupby("model"):
        color = COLORS["hgb"] if model == "HistGradientBoosting" else COLORS["persistence"]
        x_values = np.maximum(group["inference_time_per_sample"].fillna(0).to_numpy(dtype=float), 1.0e-9)
        axis.scatter(x_values, group["RMSE"], s=46, color=color, label=model)
        for _, row in group.iterrows():
            axis.annotate(row["experiment_id"], (max(float(row["inference_time_per_sample"] or 0), 1.0e-9), float(row["RMSE"])), xytext=(4, 3), textcoords="offset points", fontsize=7)
    axis.set_xscale("log")
    axis.set_xlabel("Inference time per sample (s, log scale)")
    axis.set_ylabel("Test RMSE (p.u.)")
    axis.legend(frameon=False)
    path = figures_dir / "fig22_accuracy_vs_inference_time.png"
    save_figure(fig, path)
    inventory.append({"figure": 22, "path": str(path), "caption": "Test RMSE versus measured inference time per sample; targets differ across S1, S2, and S3 and should not be ranked by RMSE alone.", "category": "cost"})

    fig, axis = plt.subplots(figsize=(5.8, 3.8))
    hgb_only = experiment_results[experiment_results["model"] == "HistGradientBoosting"]
    axis.scatter(hgb_only["feature_count"], hgb_only["RMSE"], s=52, color=COLORS["hgb"])
    for _, row in hgb_only.iterrows():
        axis.annotate(row["experiment_id"], (row["feature_count"], row["RMSE"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Model input dimension after window expansion")
    axis.set_ylabel("Test RMSE (p.u.)")
    path = figures_dir / "fig23_feature_count_vs_rmse.png"
    save_figure(fig, path)
    inventory.append({"figure": 23, "path": str(path), "caption": "Expanded model input dimension versus test RMSE for the three transferred HGB experiments; target scales differ.", "category": "cost"})

    cost_columns = ["feature_generation_time_seconds", "training_time_seconds", "test_prediction_time_seconds"]
    fig, axis = plt.subplots(figsize=(6.8, 3.4))
    x = np.arange(len(hgb_only))
    width = 0.23
    for index, column in enumerate(cost_columns):
        values = np.maximum(hgb_only[column].to_numpy(dtype=float), 1.0e-5)
        axis.bar(x + (index - 1) * width, values, width=width, label=column.replace("_seconds", "").replace("_", " ").title())
    axis.set_xticks(x, hgb_only["experiment_id"])
    axis.set_yscale("log")
    axis.set_ylabel("Elapsed time (s, log scale)")
    axis.legend(frameon=False, ncol=3)
    path = figures_dir / "fig24_feature_training_inference_costs.png"
    save_figure(fig, path)
    inventory.append({"figure": 24, "path": str(path), "caption": "Measured high-frequency feature-generation, model-training, and test-prediction times. Timings are separated rather than merged into one runtime value.", "category": "cost"})

    return pd.DataFrame(inventory)


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "not available"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    selected = frame[list(columns)].copy() if columns is not None else frame.copy()
    headers = [str(column) for column in selected.columns]
    rows = [[_fmt(value) for value in row] for row in selected.itertuples(index=False, name=None)]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    header = "| " + " | ".join(value.ljust(width) for value, width in zip(headers, widths)) + " |"
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def build_decision_table(
    results: pd.DataFrame,
    bootstrap: pd.DataFrame,
    paired: pd.DataFrame,
    averaging_summary: pd.DataFrame,
    ablation: pd.DataFrame,
    validation_passed: bool,
) -> pd.DataFrame:
    hgb = results[results["model"] == "HistGradientBoosting"].set_index("experiment_id")
    averaging = averaging_summary[averaging_summary["series"] == "averaging_loss"].set_index("statistic")["value"]
    s3_rmse_boot = bootstrap[bootstrap["metric"] == "RMSE"].iloc[0]
    s3_strong_boot = bootstrap[bootstrap["metric"] == "strong_ramp_RMSE"].iloc[0]
    s3_rmse_diff = float(paired.loc[paired["metric"] == "RMSE", "high_frequency_minus_reference"].iloc[0])
    s3_strong_diff = float(paired.loc[paired["metric"] == "strong_ramp_RMSE", "high_frequency_minus_reference"].iloc[0])
    selected_pairs = ablation[
        (ablation["status"] == "available_controlled_pair")
        & (ablation["comparison"] == "Full Physics-Informed versus Selected Physics-Informed")
    ]
    selected_inference_faster = bool(
        len(selected_pairs)
        and (selected_pairs["percent_delta_prediction_time_total"].dropna() < 0).all()
    )
    physics_pairs = ablation[
        (ablation["status"] == "available_controlled_pair")
        & (ablation["comparison"].isin(["Full versus Full Physics-Informed", "Selected versus Selected Physics-Informed"]))
    ]
    physics_test_improvements = int((physics_pairs["percent_delta_test_rmse"].dropna() < 0).sum())
    physics_available = int(physics_pairs["percent_delta_test_rmse"].notna().sum())
    statistical = str(s3_rmse_boot["conclusion"]) == "high_frequency"
    rows = [
        {
            "Question": "Does averaging suppress rapid variations?",
            "Answer": "Yes" if float(averaging.get("attenuation_positive_fraction", 0)) > 0.5 else "Partially",
            "Evidence": f"Positive attenuation in {_fmt(100*float(averaging.get('attenuation_positive_fraction', np.nan)), 2)}% of aligned minutes; cancellation in {_fmt(100*float(averaging.get('cancellation_fraction', np.nan)), 2)}%.",
            "Confidence": "High",
            "Article action": "Show the rapid-period plot and report attenuation/cancellation statistics.",
        },
        {
            "Question": "Does S1 beat persistence?",
            "Answer": "Yes" if float(hgb.loc["S1", "persistence_skill"]) > 0 else "No",
            "Evidence": f"HGB RMSE {_fmt(hgb.loc['S1','RMSE'])}; persistence RMSE {_fmt(hgb.loc['S1','persistence_RMSE'])}; skill {_fmt(hgb.loc['S1','persistence_skill'])}.",
            "Confidence": "Medium",
            "Article action": "Report as a resource-bounded next-second pilot.",
        },
        {
            "Question": "Does S2 beat persistence?",
            "Answer": "Yes" if float(hgb.loc["S2", "persistence_skill"]) > 0 else "No",
            "Evidence": f"HGB RMSE {_fmt(hgb.loc['S2','RMSE'])}; persistence RMSE {_fmt(hgb.loc['S2','persistence_RMSE'])}; skill {_fmt(hgb.loc['S2','persistence_skill'])}.",
            "Confidence": "Medium",
            "Article action": "State that S2 forecasts a one-second change 60 s ahead, not a net 60 s change.",
        },
        {
            "Question": "Does S3 improve the same one-minute target?",
            "Answer": "Yes" if s3_rmse_diff < 0 else "No",
            "Evidence": f"Paired high-frequency minus reference RMSE: {_fmt(s3_rmse_diff)} on identical timestamps.",
            "Confidence": "High" if statistical else "Medium",
            "Article action": "Treat S3 as the controlled diagnostic comparison and retain its provenance warning.",
        },
        {
            "Question": "Are strong ramps predicted better?",
            "Answer": "Yes" if s3_strong_diff < 0 else "No",
            "Evidence": f"Paired high-frequency minus reference strong-ramp RMSE: {_fmt(s3_strong_diff)}.",
            "Confidence": "High" if str(s3_strong_boot["conclusion"]) == "high_frequency" else "Medium",
            "Article action": "Include event metrics separately from average RMSE.",
        },
        {
            "Question": "Is the improvement statistically meaningful?",
            "Answer": "Yes" if statistical else "Inconclusive",
            "Evidence": f"Day-block bootstrap RMSE-difference 95% CI [{_fmt(s3_rmse_boot['ci_lower'])}, {_fmt(s3_rmse_boot['ci_upper'])}].",
            "Confidence": "High",
            "Article action": "Quote the complete-day bootstrap interval, not only the point difference.",
        },
        {
            "Question": "Is the additional computational cost acceptable?",
            "Answer": "Partially",
            "Evidence": f"S3 feature generation {_fmt(hgb.loc['S3','feature_generation_time_seconds'],2)} s and training {_fmt(hgb.loc['S3','training_time_seconds'],2)} s; value depends on the measured accuracy/event gain.",
            "Confidence": "Medium",
            "Article action": "Separate offline feature cost, training cost, and online inference cost.",
        },
        {
            "Question": "Did physics-guided engineering improve accuracy?",
            "Answer": "Yes" if physics_available and physics_test_improvements == physics_available else "Partially" if physics_test_improvements else "No",
            "Evidence": f"{physics_test_improvements} of {physics_available} available controlled test-RMSE comparisons favor the physics-guided variant.",
            "Confidence": "Medium",
            "Article action": "Use model- and resolution-specific wording; do not call all engineered features physical.",
        },
        {
            "Question": "Did selected physics reduce inference time?",
            "Answer": "Yes" if selected_inference_faster else "Inconclusive",
            "Evidence": "Validated one-minute and one-hour HGB single-run totals are lower for Selected Physics-Informed than Full Physics-Informed." if selected_inference_faster else "Controlled timing evidence is mixed or unavailable.",
            "Confidence": "Medium",
            "Article action": "Report mixed single-run timing evidence; do not claim a general reduction." if not selected_inference_faster else "Describe the timing reduction as modest and based on single executions.",
        },
        {
            "Question": "Is Selected Physics-Informed genuinely compact?",
            "Answer": "No",
            "Evidence": "The reduction is 85 to 83 features (2.35%).",
            "Confidence": "High",
            "Article action": "Use 'modest dimensional reduction', not 'highly compact'.",
        },
        {
            "Question": "Should the one-second extension enter the article?",
            "Answer": "Yes" if validation_passed else "No",
            "Evidence": "Alignment, leakage, prediction, figure, bootstrap, and package checks passed." if validation_passed else "Timestamp and row alignment passed, but the official v3 files are not exact arithmetic aggregates and the aligned reference did not exactly reproduce the archived Stage C metrics.",
            "Confidence": "High" if validation_passed else "High",
            "Article action": "Keep S3 supplementary until source processing and archived-model reproducibility are clarified.",
        },
        {
            "Question": "Does the existing project need to be restarted?",
            "Answer": "No",
            "Evidence": "The validated one-minute and one-hour winners were retained; only the transferred winner was retrained for changed inputs.",
            "Confidence": "High",
            "Article action": "Keep the existing framework and add the isolated extension.",
        },
    ]
    allowed = {"Yes", "No", "Partially", "Inconclusive"}
    if any(row["Answer"] not in allowed for row in rows):
        raise AssertionError("Decision table contains a disallowed answer.")
    return pd.DataFrame(rows)


def write_supervisor_explanation(
    path: Path,
    results: pd.DataFrame,
    paired: pd.DataFrame,
    bootstrap: pd.DataFrame,
    averaging_summary: pd.DataFrame,
    extra_trees_status: str,
    aggregation_alignment: dict[str, Any],
    reference_consistency: dict[str, Any],
) -> None:
    hgb = results[results["model"] == "HistGradientBoosting"].set_index("experiment_id")
    s3_diff = float(paired.loc[paired["metric"] == "RMSE", "high_frequency_minus_reference"].iloc[0])
    ci = bootstrap[bootstrap["metric"] == "RMSE"].iloc[0]
    averaging = averaging_summary[averaging_summary["series"] == "averaging_loss"].set_index("statistic")["value"]
    improvement_word = "improved" if s3_diff < 0 else "did not improve"
    text = f"""# Supervisor-ready explanation

## Detailed explanation

### 1. Why the whole framework was not repeated

The original one-minute and one-hour pipelines had already completed data cleaning, target construction, chronological splitting, feature selection, model comparison, and final validation. Repeating every model would answer no new scientific question and would create avoidable opportunities for inconsistency. This extension therefore keeps those results fixed and tests only the professor's new question: whether one-second information changes what can be observed or predicted.

### 2. Why only the best model was transferred

HistGradientBoosting was the validated one-minute classical winner, so it is the mandatory transferred model for S1, S2, and S3. Its estimator configuration was reused without tuning. The fitted estimator itself could not be reused because the new feature matrices have different columns and, for S1 and S2, a different target distribution. It was retrained on each new training matrix. ExtraTrees was {extra_trees_status}; the 120-tree S3 matrix was judged unsafe on the available 8 GB machine, so skipping it was a memory decision rather than an accuracy judgment.

### 3. What the targets mean

- **S1:** information through second *s* predicts the change from *s* to *s+1*: `(p[s+1]-p[s])/10`.
- **S2:** information through *s* predicts the one-second change occurring 60 seconds later: `(p[s+60]-p[s+59])/10`.
- **S3:** one-second-derived descriptors available through completed minute *m* predict the article's existing next-minute target: `(Pbar[m+1]-Pbar[m])/10`.

S2 is not the net change `(p[s+60]-p[s])/10`. Its lead is 60 seconds, but its target interval is one second.

### 4. Why S3 is the fairest test

S1, S2, and the article result have different target distributions, so their RMSE values cannot be ranked directly. S3 removes that ambiguity: the reference and high-frequency models use identical target values, timestamps, chronological splits, normalization, HGB parameters, and metrics. Only the input information changes.

### 4.1 Cross-resolution source warning

All official one-minute timestamps match the reconstructed one-second minute labels, but the PV values do not reproduce as simple 60-second means. Across complete minutes, the mean absolute difference is {_fmt(aggregation_alignment['mean_absolute_difference_kw'],8)} kW, the 95th percentile is {_fmt(aggregation_alignment['p95_absolute_difference_kw'],8)} kW, the maximum is {_fmt(aggregation_alignment['maximum_absolute_difference_kw'],8)} kW, and correlation is {_fmt(aggregation_alignment['correlation'],8)}. Both files match the official Figshare v3 checksums. The v3 README calls the one-minute file a processed sampling but does not disclose its exact transformation. Consequently, S3 is a same-timestamp cross-resolution input ablation, not evidence that the official one-minute PV is the direct arithmetic mean of the v3 one-second PV.

### 4.2 Reference-model reproduction warning

The aligned one-minute reference was retrained under the same current scikit-learn policy as S3. It is not claimed to be a bit-for-bit reconstruction of the archived Stage C estimator. The archived and retrained validation RMSE values are `{reference_consistency['official_validation_rmse']:.8f}` and `{reference_consistency['reconstructed_validation_rmse']:.8f}`; the corresponding test values are `{reference_consistency['official_test_rmse']:.8f}` and `{reference_consistency['reconstructed_test_rmse']:.8f}`. The differences are small but nonzero, and the archived package does not contain the fitted estimator or enough environment metadata to resolve them. The paired S3 comparison remains internally controlled because both comparison models were retrained with the same current fit policy and evaluated on identical rows.

### 5. What averaging removes

One-minute averaging visibly suppresses within-minute peaks and can cancel upward and downward changes. Positive attenuation occurred in {_fmt(100*float(averaging.get('attenuation_positive_fraction', np.nan)),2)}% of aligned minutes, and both ramp signs occurred in {_fmt(100*float(averaging.get('cancellation_fraction', np.nan)),2)}%. This demonstrates loss of visible rapid variation. It does not, by itself, prove that forecast accuracy is harmed.

### 6. What the forecasting experiment found

S1 HGB achieved RMSE {_fmt(hgb.loc['S1','RMSE'])} versus persistence {_fmt(hgb.loc['S1','persistence_RMSE'])}. S2 achieved {_fmt(hgb.loc['S2','RMSE'])} versus persistence {_fmt(hgb.loc['S2','persistence_RMSE'])}. These are resource-bounded pilots based on systematic 30-second anchors over the complete chronological span.

For the controlled S3 comparison, the one-second-derived input {improvement_word} the same-target RMSE by `{abs(s3_diff):.8f}` p.u. in point terms (high-frequency minus reference `{s3_diff:.8f}`). The complete-day bootstrap 95% interval was `[{float(ci['ci_lower']):.8f}, {float(ci['ci_upper']):.8f}]`. This interval determines whether the change should be called statistically supported or inconclusive.

### 7. Computational cost and feature selection

High-frequency preprocessing is an offline cost and is reported separately from training and per-sample inference. The existing feature-selection audit also shows that Selected Physics-Informed is not a highly compact matrix: it removes only two of 85 features, or 2.35%. The controlled timing evidence is mixed or unavailable across the required pairs, so no general inference-time reduction is claimed. Generic lags, rolling statistics, and deltas are temporal-statistical features; only 23 engineered variables are physics/domain guided.

### 8. Does the current article remain valid?

Yes. The one-minute and one-hour results remain valid because this extension neither changes nor overwrites them. However, the S3 extension should remain supplementary until the official v3 one-minute processing is clarified. The article can safely clarify the target alignment, distinguish temporal-statistical from physics-guided features, and explain the four data variants without claiming that the two public resolutions are exact arithmetic aggregates.

## One-page summary

The existing framework was retained because its one-minute and one-hour results were already validated. HistGradientBoosting, the one-minute winner, was transferred without tuning and retrained because new inputs and targets cannot be passed to an old fitted estimator. Three tasks were separated: S1 predicts the next one-second change, S2 predicts a one-second change 60 seconds ahead, and S3 predicts the exact next-minute target used in the article from one-second-derived context. S3 is the decisive comparison because reference and extension share timestamps, target values, split boundaries, model family, and metrics.

The raw-versus-aggregated analysis confirms that minute averaging suppresses visible rapid changes and often cancels opposite signs within a minute. That descriptive finding is separate from prediction. The prediction question is answered only by S3. Its paired RMSE change was `{s3_diff:.8f}` p.u., with a complete-day bootstrap 95% interval of `[{float(ci['ci_lower']):.8f}, {float(ci['ci_upper']):.8f}]`. S1 and S2 remain resource-bounded pilots because the 39-million-row second-level source could not safely be fitted in full on an 8 GB workstation.

The source audit adds an important limitation: although timestamps align exactly, the official one-minute PV is not the simple 60-second mean of the official v3 one-second PV, and the public transformation is undocumented. S3 must therefore be described as a diagnostic cross-resolution ablation and should not enter the main article as direct aggregation evidence without clarification from the dataset publisher.

The feature audit corrects an important terminology point. The 74 engineered inputs comprise 51 temporal-statistical and 23 physics/domain-guided features. Selected Physics-Informed has 83 inputs versus 85 for Full Physics-Informed, so it represents only a 2.35% dimensional reduction. The article should describe this as a modest dimensional reduction with mixed single-run timing evidence, not as a highly compact or consistently faster design.

## Two-minute oral version

We did not restart the project because the original minute and hourly pipelines were already validated. We kept their winners and transferred only HistGradientBoosting, with exactly the same configuration and no new tuning. We had to retrain it because one-second descriptors change both the input matrix and, in S1 and S2, the target.

We tested three different questions. S1 predicts the next one-second change. S2 predicts a one-second change that occurs one minute later; it is not the total change over that minute. S3 is the fair comparison: it predicts exactly the article's next-minute ramp, on exactly the same timestamps, but adds causal information extracted from the seconds inside each completed minute.

There is one source warning we must state clearly. The official v3 one-minute timestamps align with the seconds, but its PV values are not reproducible by taking their simple 60-second mean, and the public documentation does not provide the exact one-minute transformation. We can discuss S3 as a diagnostic input comparison, but not as proof based on two perfectly nested versions of the same power series.

The plots and statistics show clearly that minute averaging hides rapid peaks and cancels opposite changes. But that alone does not prove a forecast benefit. The S3 paired result is what answers that question. Its high-frequency-minus-reference RMSE difference is `{s3_diff:.8f}` p.u., and the complete-day bootstrap interval is `[{float(ci['ci_lower']):.8f}, {float(ci['ci_upper']):.8f}]`. We should interpret the sign and interval, not just a tiny decimal difference.

Finally, the feature terminology needs care. Of the 74 engineered variables, 51 are temporal-statistical and 23 are physics guided. The selected physics matrix removes only two of 85 inputs. It is not highly compact, and the controlled timing evidence does not support a general claim that it is faster. The current article remains valid; S3 should stay supplementary until the source provenance is clarified, and no full rerun is required.
"""
    path.write_text(text, encoding="utf-8")


def write_article_latex(
    path: Path,
    results: pd.DataFrame,
    reference_metrics: dict[str, Any],
    paired: pd.DataFrame,
    bootstrap: pd.DataFrame,
    averaging_summary: pd.DataFrame,
    ablation: pd.DataFrame,
    aggregation_alignment: dict[str, Any],
    reference_consistency: dict[str, Any],
) -> None:
    hgb = results[results["model"] == "HistGradientBoosting"].set_index("experiment_id")
    s3_diff = float(paired.loc[paired["metric"] == "RMSE", "high_frequency_minus_reference"].iloc[0])
    ci = bootstrap[bootstrap["metric"] == "RMSE"].iloc[0]
    averaging = averaging_summary[averaging_summary["series"] == "averaging_loss"].set_index("statistic")["value"]
    tex = rf"""% Article-ready material generated from the validated one-second extension.
% Review and integrate selectively; this file does not modify the conference manuscript.

\subsection{{One-Second Multi-Resolution Extension}}
The SOLETE measurements were originally recorded at 1~Hz and distributed at several
aggregated resolutions \cite{{pombo2022solete}}. We therefore examined whether the
one-second stream reveals ramp dynamics that are suppressed by one-minute averaging,
and whether those dynamics improve forecasting of the article's existing target. The
extension preserves the validated one-minute and one-hour pipelines and transfers only
their relevant winning model; it does not repeat model selection.

\subsubsection{{Target definitions.}}
Let $p_s$ denote one-second PV power, $\bar P_m$ the official one-minute PV value,
and $P_R=10$~kW. Three tasks are kept distinct:
\begin{{align}}
y_s^{{\mathrm{{S1}}}} &= (p_{{s+1}}-p_s)/P_R,\\
y_s^{{\mathrm{{S2}}}} &= (p_{{s+60}}-p_{{s+59}})/P_R,\\
y_m^{{\mathrm{{S3}}}} &= (\bar P_{{m+1}}-\bar P_m)/P_R.
\end{{align}}
S2 predicts a one-second change occurring 60~s after the forecast origin; it is not
the net change $(p_{{s+60}}-p_s)/P_R$. S3 is the controlled same-target experiment:
the reference and high-frequency models share target timestamps, chronological splits,
normalization, model family, and evaluation metrics.

\subsubsection{{Cross-resolution provenance check.}}
All official one-minute timestamps matched the reconstructed minute labels, but the
one-minute PV values were not equal to arithmetic means of the 60 one-second PV values.
Across complete minutes, the mean absolute difference was
{float(aggregation_alignment['mean_absolute_difference_kw']):.8f}~kW, the 95th percentile
was {float(aggregation_alignment['p95_absolute_difference_kw']):.8f}~kW, the maximum was
{float(aggregation_alignment['maximum_absolute_difference_kw']):.8f}~kW, and correlation
was {float(aggregation_alignment['correlation']):.8f}. The SOLETE v3 documentation calls
the one-minute file a processed sampling but does not disclose the exact transformation.
S3 is therefore interpreted as a same-timestamp cross-resolution input ablation, not as
evidence obtained from two exactly nested arithmetic aggregates.

The one-minute reference was retrained under the same current fit policy as S3, but it
did not reproduce the archived Stage~C estimator exactly. Archived and retrained test
RMSE were {float(reference_consistency['official_test_rmse']):.8f} and
{float(reference_consistency['reconstructed_test_rmse']):.8f}, respectively. Because the
archived estimator and its complete software environment were not retained, this small
difference is reported rather than treated as numerical identity. The paired S3 comparison
remains internally controlled because both arms use the same current fitting policy and rows.

\subsubsection{{Protocol and causal high-frequency features.}}
The validated one-minute HistGradientBoosting configuration (50 boosting iterations,
31 leaf nodes, learning rate 0.08, no $\ell_2$ penalty, early stopping, and seed 42)
was retrained without tuning. S1 and S2 use deterministic 30-s anchors over the complete
chronological span and causal lags and rolling descriptors covering 480~s, the physical
duration represented by the winning $L=8$ one-minute history. S3 retains every aligned
complete minute and appends within-minute second-level descriptors to the official selected-physics
$L=8$ input. No second from minute $m+1$ is used to predict $y_m^{{\mathrm{{S3}}}}$.
The official one-second HDF5 rows are stored as shuffled day blocks; they were first
placed on the unique chronological 1~Hz grid by timestamp. Chunked HDF5 reads,
float32 checkpoints, Parquet partitions, and disk-backed model arrays then bounded
memory use on an 8~GB workstation.

Feature engineering combines generic temporal-statistical descriptors with
physics-guided variables derived from PV operating principles, consistent with the
feature-level use of domain knowledge in prior SOLETE studies
\cite{{pombo2022sensors,pombo2022benchmarking}}. It is not a physics-informed neural
network and does not embed governing PV equations in the training loss.

\subsubsection{{Variability removed by one-minute averaging.}}
Across aligned minutes, the descriptive attenuation indicator was positive in
{100*float(averaging.get('attenuation_positive_fraction', np.nan)):.2f}\% of cases,
while positive and negative second-level ramps co-occurred in
{100*float(averaging.get('cancellation_fraction', np.nan)):.2f}\%. These statistics
show that aggregation suppresses observable rapid variation; they do not by themselves
establish a forecasting penalty.

\subsubsection{{Forecasting results.}}
S1 obtained RMSE {float(hgb.loc['S1','RMSE']):.6f} and persistence skill
{float(hgb.loc['S1','persistence_skill']):.4f}. S2 obtained RMSE
{float(hgb.loc['S2','RMSE']):.6f} and persistence skill
{float(hgb.loc['S2','persistence_skill']):.4f}. Because both tasks use systematic
second-level anchors, they are reported as resource-bounded pilots rather than full-row
benchmarks.

For S3, the high-frequency model obtained test RMSE
{float(hgb.loc['S3','RMSE']):.6f}, compared with
{float(reference_metrics['RMSE']):.6f} for the reconstructed official one-minute-input
HGB on the same timestamps. The paired difference (high-frequency minus reference) was
{s3_diff:.8f}~p.u.; a 1,000-replicate complete-day bootstrap gave a 95\% interval of
[{float(ci['ci_lower']):.8f}, {float(ci['ci_upper']):.8f}]. This interval, rather than
the point estimate alone, determines whether the change is statistically supported.

\subsubsection{{Computational cost and feature-variant audit.}}
High-frequency feature construction required {float(hgb.loc['S3','feature_generation_time_seconds']):.2f}~s,
S3 training required {float(hgb.loc['S3','training_time_seconds']):.2f}~s, and test
inference required {float(hgb.loc['S3','test_prediction_time_seconds']):.4f}~s. These
quantities are reported separately because offline feature generation, model fitting,
and online inference represent different deployment costs.

The existing engineered matrices contain 51 temporal-statistical and 23
physics/domain-guided variables. Full Physics-Informed has 85 inputs, whereas Selected
Physics-Informed has 83. The 2.35\% reduction is modest; the defensible conclusion is
that the 83-input matrix is only slightly narrower. Controlled single-run timing evidence
is mixed or unavailable, so a general inference-time reduction is not claimed.

\subsubsection{{Discussion, limitations, and control relevance.}}
S1 and S2 quantify the difficulty of instantaneous ramps, whereas S3 tests whether
second-level context helps the more structured next-minute target. Their RMSE values
must not be directly ranked because the target intervals and variances differ. The
second-level forecasts may provide an anticipatory signal for a future battery,
inverter, or flexible-load control layer, but no closed-loop microgrid benefit is
claimed without controller simulation. Limitations include deterministic anchor
sampling for S1/S2, a single site, one transferred model family, and the absence of a
safe local ExtraTrees S3 run.

\subsubsection{{Conclusion.}}
The one-second extension should be presented as an isolated multi-resolution audit.
The raw stream establishes how averaging hides rapid changes, while S3 supplies a
same-target diagnostic input comparison. Because the public one-minute transformation
is not reproducible from the v3 one-second file, S3 should remain supplementary unless
the dataset publisher clarifies this provenance.
The validated one-minute and one-hour framework remains the principal benchmark.

\begin{{table}}[t]
\centering
\caption{{Targets used in the one-second multi-resolution extension.}}
\label{{tab:one_second_targets}}
\small
\begin{{tabular}}{{llll}}
\hline
Task & Target & Lead & Interpretation\\
\hline
S1 & $(p_{{s+1}}-p_s)/P_R$ & 1 s & Next one-second change\\
S2 & $(p_{{s+60}}-p_{{s+59}})/P_R$ & 60 s & Future one-second change\\
S3 & $(\bar P_{{m+1}}-\bar P_m)/P_R$ & 60 s & Existing next-minute target\\
\hline
\end{{tabular}}
\end{{table}}

\begin{{table}}[t]
\centering
\caption{{Transferred HGB results; S1 and S2 are systematic-anchor pilots.}}
\label{{tab:one_second_results}}
\small
\begin{{tabular}}{{lrrrrr}}
\hline
Task & RMSE & MAE & $R^2$ & Skill & Direction\\
\hline
S1 & {float(hgb.loc['S1','RMSE']):.5f} & {float(hgb.loc['S1','MAE']):.5f} & {float(hgb.loc['S1','R2']):.3f} & {float(hgb.loc['S1','persistence_skill']):.3f} & {float(hgb.loc['S1','directional_accuracy']):.3f}\\
S2 & {float(hgb.loc['S2','RMSE']):.5f} & {float(hgb.loc['S2','MAE']):.5f} & {float(hgb.loc['S2','R2']):.3f} & {float(hgb.loc['S2','persistence_skill']):.3f} & {float(hgb.loc['S2','directional_accuracy']):.3f}\\
S3 & {float(hgb.loc['S3','RMSE']):.5f} & {float(hgb.loc['S3','MAE']):.5f} & {float(hgb.loc['S3','R2']):.3f} & {float(hgb.loc['S3','persistence_skill']):.3f} & {float(hgb.loc['S3','directional_accuracy']):.3f}\\
\hline
\end{{tabular}}
\end{{table}}

\begin{{table}}[t]
\centering
\caption{{Paired S3 same-target comparison on identical test timestamps.}}
\label{{tab:s3_same_target}}
\small
\begin{{tabular}}{{lrrr}}
\hline
Metric & Existing input & High-frequency input & Difference\\
\hline
RMSE & {float(reference_metrics['RMSE']):.6f} & {float(hgb.loc['S3','RMSE']):.6f} & {s3_diff:.8f}\\
MAE & {float(reference_metrics['MAE']):.6f} & {float(hgb.loc['S3','MAE']):.6f} & {float(paired.loc[paired['metric']=='MAE','high_frequency_minus_reference'].iloc[0]):.8f}\\
Directional accuracy & {float(reference_metrics['directional_accuracy']):.4f} & {float(hgb.loc['S3','directional_accuracy']):.4f} & {float(paired.loc[paired['metric']=='directional_accuracy','high_frequency_minus_reference'].iloc[0]):.8f}\\
\hline
\end{{tabular}}
\end{{table}}
"""
    path.write_text(tex, encoding="utf-8")


def write_markdown_report(
    path: Path,
    summary: dict[str, Any],
    results: pd.DataFrame,
    target_definitions: pd.DataFrame,
    lookback_conversion: pd.DataFrame,
    averaging_summary: pd.DataFrame,
    periods: pd.DataFrame,
    paired: pd.DataFrame,
    bootstrap: pd.DataFrame,
    ablation: pd.DataFrame,
    decisions: pd.DataFrame,
    figure_inventory: pd.DataFrame,
) -> None:
    hgb = results[results["model"] == "HistGradientBoosting"].set_index("experiment_id")
    averaging = averaging_summary[averaging_summary["series"] == "averaging_loss"].set_index("statistic")["value"]
    reference = summary["s3_same_target_comparison"]["existing_reference_test_metrics"]
    reference_consistency = summary["reference_reconstruction_consistency"]
    completeness = summary["minute_completeness"]
    aggregation = summary["aggregation_alignment"]
    s3_alignment = summary["s3_same_target_comparison"]["alignment_audit"]
    sampling_audit = pd.DataFrame(summary["sampling_strategy"]["target_specific_audit"])
    fit_policies = [
        ("S1", summary["S1_result"]["fit_policy"]),
        ("S2", summary["S2_result"]["fit_policy"]),
        ("Existing reference", summary["reference_HGB_fit_policy"]),
        ("S3", summary["S3_result"]["fit_policy"]),
    ]
    stopping_audit = pd.DataFrame(
        [
            {
                "target": target,
                "scikit-learn": policy["scikit_learn_version"],
                "training rows": policy["original_training_rows"],
                "fit rows": policy["internal_fit_rows"],
                "stopping rows": policy["internal_early_stopping_rows"],
                "internal seed": policy["internal_split_seed"],
                "official validation rows": policy["official_validation_rows_used_for_early_stopping"],
                "test rows": policy["test_rows_used_for_fitting_or_early_stopping"],
            }
            for target, policy in fit_policies
        ]
    )
    rmse_boot = bootstrap[bootstrap["metric"] == "RMSE"].iloc[0]
    available_ablation = ablation[ablation["status"] == "available_controlled_pair"]
    report = f"""# Final One-Second Multi-Resolution Professor Extension

**Execution status:** `{summary['execution_status']}`
**Execution date:** `{summary['execution_date']}`
**Official source:** `{summary['data_source']['path']}`
**SHA-256:** `{summary['data_source']['sha256']}`

## Executive summary

This extension answers two questions that must remain separate. First, one-minute averaging does suppress visible second-level PV changes: the attenuation indicator is positive in {_fmt(100*float(averaging.get('attenuation_positive_fraction', np.nan)),2)}% of aligned minutes, and upward and downward second-level changes coexist in {_fmt(100*float(averaging.get('cancellation_fraction', np.nan)),2)}%. Second, whether that hidden variation improves forecasting is tested only by S3, where both models predict the same one-minute target on the same timestamps.

The validated one-minute HistGradientBoosting configuration was transferred without tuning. S1 and S2 are resource-bounded second-level pilot experiments using deterministic 30-second anchors across the full chronological period. S3 retains every eligible complete-minute row and excludes the final partial minute from both comparison arms. Its high-frequency-minus-reference RMSE difference is {_fmt(float(paired.loc[paired['metric']=='RMSE','high_frequency_minus_reference'].iloc[0]),8)} p.u., with a 1,000-replicate complete-day 95% bootstrap interval of [{_fmt(rmse_boot['ci_lower'],8)}, {_fmt(rmse_boot['ci_upper'],8)}].

The existing framework does not need to be restarted. The one-minute and one-hour winners remain the official benchmarks; this directory is an isolated, traceable extension.

## Professor questions

1. **Was the ramp target already present?** No. The project creates the normalized backward difference `(P[t]-P[t-1])/10` and aligns the forecast target after the input window.
2. **Would one-second data provide more information?** It provides visibly richer within-minute dynamics. S3 determines whether that information improves the same prediction task.
3. **Why not reuse the fitted model?** The estimator expects the old feature matrix. A changed input schema or target requires retraining, although the model family and hyperparameters can be retained.
4. **Why can the one-hour result look stronger?** The targets have different variance, temporal structure, and persistence difficulty. Cross-resolution RMSE or $R^2$ is not a direct ranking of operational quality.
5. **Is Selected Physics-Informed much smaller?** No. It removes two of 85 features, a 2.35% reduction.

## Verified existing winners

The final local Stage C outputs identify HistGradientBoosting on Selected Physics-Informed, `L=8`, `H=1`, as the one-minute classical winner (validation RMSE `{summary['verified_one_minute_winner']['validation_rmse']:.6f}`, test RMSE `{summary['verified_one_minute_winner']['test_rmse']:.6f}`). The one-hour final winner is ExtraTrees on the same feature variant and `L=8`, `H=1` (validation RMSE `{summary['verified_one_hour_winner']['validation_rmse']:.6f}`, test RMSE `{summary['verified_one_hour_winner']['test_rmse']:.6f}`). The one-minute HGB, not the hourly ExtraTrees, is the principal transferred model because the extension concerns second-to-minute forecasting.

## Target definitions and alignment

{markdown_table(target_definitions)}

The stored article target is `r[t]=(P[t]-P[t-1])/10`. For `H=1`, a window ending at forecast origin `t` predicts `r[t+1]=(P[t+1]-P[t])/10`. Each split is windowed independently; no input history or target crosses a chronological boundary. The reconstructed reference and S3 predictions share exactly the same target timestamps.

## Lookback and horizon

{markdown_table(lookback_conversion)}

An `L=8` history means eight minutes in the article but only eight seconds in a one-second matrix. The second-level design therefore covers 480 seconds with causal lags and rolling summaries instead of flattening `480 x d` raw observations.

## Data source and preprocessing

The official 1 Hz SOLETE HDF5 file contains `{summary['data_source']['rows']:,}` rows and `{summary['data_source']['columns']}` measured columns from `{summary['data_span']['start']}` to `{summary['data_span']['end']}`. Its stored rows consist of shuffled day blocks, with `{summary['data_source']['stored_order_discontinuities']}` non-1-Hz transitions in file order. The pipeline therefore scatters each row to its exact integer-second timestamp before any target, split, rolling statistic, or model matrix is constructed. This correction is deterministic and the normalization audit requires zero duplicate and zero missing timestamps. The complete HDF5 file is never loaded into memory. Compact checkpoints use `float32`; the final HGB matrices are disk-backed `float64` because scikit-learn otherwise creates a full conversion copy in RAM. Minute descriptors and sampled anchor parts are checkpointed as Parquet.

Cleaning counts are recorded in the run manifest. The original article intentionally retains nighttime rows; the productive condition (`Elevation > 0` and `P_Solar > 0.05 kW`) is reported as a diagnostic rather than introduced as a new filter.

### Complete-minute and official-aggregation audit

The reconstructed stream contains `{completeness['complete_minutes']:,}` complete minutes and `{completeness['incomplete_minutes']:,}` incomplete minute. Completeness requires exactly 60 timestamped rows and 60 finite PV/POA/GHI observations. The incomplete row remains in the cleaning audit but is excluded from the reference and S3 model windows.

The official one-minute PV series and the reconstructed arithmetic means have `{aggregation['matched_complete_timestamps']:,}` matched complete timestamps, with `{aggregation['missing_from_reconstructed_one_second_series']}` timestamp missing from the reconstruction and `{aggregation['missing_from_official_one_minute_series']}` missing from the official file. The mean, median, 95th-percentile, and maximum absolute power differences are `{_fmt(aggregation['mean_absolute_difference_kw'],10)}`, `{_fmt(aggregation['median_absolute_difference_kw'],10)}`, `{_fmt(aggregation['p95_absolute_difference_kw'],10)}`, and `{_fmt(aggregation['maximum_absolute_difference_kw'],10)}` kW; correlation is `{_fmt(aggregation['correlation'],10)}`. Thus, the official v3 one-minute PV is **not** a simple arithmetic mean of the current v3 one-second PV. The project reconstruction uses the arithmetic mean of the 60 samples in `[m,m+60 s)`, while the official v3 README describes its file only as a “processed sampling of 1 min” and the public code does not disclose the exact transformation. Labels are aligned at the minute start. The HDF datetimes are timezone-naive representations of UTC, as specified by the SOLETE data paper.

This distinction constrains interpretation. S3 still uses the exact official project target for both comparison arms and has complete timestamp alignment, but it is a same-timestamp cross-resolution input ablation rather than proof about a directly averaged copy of the same PV series.

## OOM-aware execution

The preflight found `{summary['environment']['total_ram_gb']:.2f}` GB of RAM. A full S1/S2 matrix would contain tens of millions of rows and was rejected before allocation. The same deterministic 30-second rule is used in training, validation, and test. These are sampled pilot results, not exhaustive full-second evaluations.

{markdown_table(sampling_audit, ['target_id','theoretically_available_anchors','retained_anchors','retained_percentage','train_anchors','validation_anchors','test_anchors','first_timestamp','last_timestamp','sampling_interval_seconds','same_sampling_policy_train_validation_test'])}

Every eligible complete-minute row is retained for S3. ExtraTrees was `{summary['extra_trees']['status']}` because its estimated tree ensemble and the {int(hgb.loc['S3','feature_count'])}-column S3 design exceeded the local safety threshold.

## HistGradientBoosting early stopping

The official HGB settings are squared-error loss, 50 maximum boosting iterations, 31 leaf nodes, learning rate 0.08, zero L2 regularization, early stopping enabled, and random state 42. Early stopping uses a seeded random 90/10 subdivision of each official training subset only. The split was materialized on disk and supplied through `X_val`/`y_val` under scikit-learn 1.9.0, matching that version's seeded split mechanics while avoiding a full in-memory copy. No official validation or test observation enters model fitting or early stopping.

{markdown_table(stopping_audit)}

This is not claimed to be a bit-for-bit reconstruction of the archived Stage C estimator. Archived versus retrained RMSE was `{reference_consistency['official_validation_rmse']:.8f}` versus `{reference_consistency['reconstructed_validation_rmse']:.8f}` on validation and `{reference_consistency['official_test_rmse']:.8f}` versus `{reference_consistency['reconstructed_test_rmse']:.8f}` on test. The archived package does not include the fitted estimator or complete environment metadata. The paired S3 result remains internally controlled because its reference and high-frequency models were retrained with the same current policy and aligned rows.

## S1: next one-second ramp

S1 predicts `(p[s+1]-p[s])/10`. HGB test RMSE is {_fmt(hgb.loc['S1','RMSE'])}, compared with persistence RMSE {_fmt(hgb.loc['S1','persistence_RMSE'])}; persistence skill is {_fmt(hgb.loc['S1','persistence_skill'])}. Directional accuracy is {_fmt(hgb.loc['S1','directional_accuracy'])}, and strong-ramp F1 is {_fmt(hgb.loc['S1','strong_ramp_F1'])}. The strong threshold was derived from the 95th percentile of the absolute training target, never from test data.

## S2: one-second ramp 60 seconds ahead

S2 predicts `(p[s+60]-p[s+59])/10`. It is a future instantaneous one-second change, not `(p[s+60]-p[s])/10`. HGB test RMSE is {_fmt(hgb.loc['S2','RMSE'])}, persistence RMSE is {_fmt(hgb.loc['S2','persistence_RMSE'])}, and persistence skill is {_fmt(hgb.loc['S2','persistence_skill'])}. The task is operationally more demanding because its lead is 60 seconds while the target interval remains one second.

## S3: same next-minute target

S3 appends causal within-minute descriptors to the official selected-physics `L=8` input and predicts `(Pbar[m+1]-Pbar[m])/10`. The official test split provides `{s3_alignment['official_one_minute_test_candidate_rows']:,}` candidate windows; `{s3_alignment['available_s3_complete_test_rows']:,}` remain after completeness filtering, `{s3_alignment['matched_comparison_rows']:,}` are compared on identical timestamps, and `{s3_alignment['unmatched_rows']}` is excluded with its reason recorded in `tables/s3_excluded_samples.csv`. The reference RMSE is {_fmt(reference['RMSE'])}; S3 RMSE is {_fmt(hgb.loc['S3','RMSE'])}. The point difference is {_fmt(float(paired.loc[paired['metric']=='RMSE','high_frequency_minus_reference'].iloc[0]),8)} p.u. The bootstrap interval above determines whether this should be described as supported improvement, plausible equivalence, or degradation.

## Experiment results

{markdown_table(results, ['experiment_id','model','feature_count','train_rows','validation_rows','test_rows','MAE','RMSE','NRMSE','R2','persistence_skill','directional_accuracy','strong_ramp_F1','training_time_seconds','test_prediction_time_seconds'])}

## Averaging-loss analysis

{markdown_table(averaging_summary)}

The attenuation indicator is descriptive, not a universal physical metric. Aggregation can suppress an observable peak without necessarily making the next-minute target harder to forecast. That distinction is why the raw-versus-aggregate plots and S3 prediction comparison are reported separately.

## Representative periods

{markdown_table(periods)}

These periods were selected from the test interval only for illustration. They were not used for hyperparameter selection, threshold selection, or metric reporting.

## Rapid-ramp event analysis

S1 and S2 thresholds are training percentiles because the article's `0.01` p.u. minute-level threshold is not physically interchangeable with a one-second event threshold. S3 retains `|r| >= 0.01` p.u. Event precision, recall, F1, strong-ramp MAE/RMSE, sign accuracy, target-magnitude bins, solar-elevation context, and irradiance-variability context are stored with the prediction tables.

## Same-target statistical comparison

{markdown_table(paired)}

{markdown_table(bootstrap)}

Differences are defined as high-frequency minus existing reference. Negative RMSE/MAE differences favor S3; positive directional-accuracy differences favor S3. Complete days are resampled to preserve within-day temporal dependence.

## Four-variant accuracy-versus-time audit

{markdown_table(available_ablation, ['resolution','model','comparison','reference_feature_count','new_feature_count','percent_delta_validation_rmse','percent_delta_test_rmse','percent_delta_training_time','percent_delta_prediction_time_total','percent_delta_feature_count'])}

The one-hour HGB outputs provide all four variants at `L=8/H=1`. The one-minute HGB outputs contain only Full Physics-Informed and Selected Physics-Informed, so Ridge supplies the complete one-minute four-variant audit; absent HGB pairs are marked unavailable rather than reconstructed. Timing differences come from single runs and should be interpreted as indicative, not deterministic.

## Feature terminology

The validated 74 engineered variables comprise 51 temporal-statistical descriptors (lags, deltas, and trailing statistics) and 23 physics/domain-guided variables (solar/time context, PV operating-point ratios, and temperature/wind interactions). Physical knowledge is introduced at feature level. This is not a physics-informed neural network and no governing equation appears in the loss.

Full has 11 original predictors; Selected has nine. Full Physics-Informed has `11 + 74 = 85`; Selected Physics-Informed has `9 + 74 = 83`. The selected base removes `P_Gaia[kW]` and `Pressure[mbar]`. It is a modest reduction, not a highly compact representation.

## Computational cost

Feature-generation, model-training, validation-prediction, test-prediction, and inference-per-sample times are separate fields. S3 high-frequency feature generation took {_fmt(hgb.loc['S3','feature_generation_time_seconds'],2)} s; HGB training took {_fmt(hgb.loc['S3','training_time_seconds'],2)} s; test prediction took {_fmt(hgb.loc['S3','test_prediction_time_seconds'],4)} s. These measurements describe this machine and run; they are not hardware-independent constants.

## Microgrid-control relevance

The second-level forecasts may provide an anticipatory signal for future battery dispatch, inverter balancing, converter coordination, ramp smoothing, or flexible-load support. No closed-loop controller was simulated here, so the report does not claim improved voltage, frequency, battery lifetime, or energy cost. Such claims require a subsequent microgrid-control experiment.

## Limitations

- S1 and S2 use deterministic anchor sampling because full-row HGB training was unsafe on 8 GB RAM.
- The study covers one SOLETE site and one transferred model family.
- ExtraTrees S3 was skipped after a memory preflight, not because it was assumed inferior.
- The HGB stopping policy follows the validated configuration; no new tuning was performed.
- The attenuation indicator compares different temporal summaries and is descriptive.
- Runtime values are single-run measurements.
- The aligned reference follows the archived Stage C protocol but does not numerically reproduce its stored aggregate metrics exactly.

## Final recommendations

Keep the validated one-minute and one-hour framework. Do not rerun the full model catalogue. Use S1/S2 as supporting pilot evidence about fast-ramp difficulty. Keep S3 supplementary unless the publisher clarifies the v3 one-minute processing; if it is shown, label it as a same-timestamp cross-resolution ablation and include the source warning beside its paired uncertainty result. Correct the feature terminology and describe the 83-feature matrix as a modest reduction.

## Article-ready conclusion

One-second data reveal rapid PV variation that arithmetic minute averaging suppresses, but visible information loss and forecasting loss are not the same claim. The aligned S3 experiment supplies a diagnostic same-target comparison, while the unresolved processing difference between the official v3 files limits its article interpretation. Its paired metrics, complete-day bootstrap, and provenance warning must be read together; the original minute and hourly results remain valid benchmarks.

## Figure inventory

{markdown_table(figure_inventory, ['figure','path','caption','category'])}

## Final decision table

{markdown_table(decisions)}

## Traceability

Every number in this report is exported to CSV or JSON under `tables/` and `manifests/`. Prediction-level values are stored as compressed Parquet files; new estimators and their parameters are stored under `models/`. No original article, model, figure, or validated output was overwritten.
"""
    path.write_text(report, encoding="utf-8")


def _escape_reportlab(text: str) -> str:
    import html
    import re

    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", escaped)
    return escaped


def write_pdf_report(markdown_path: Path, pdf_path: Path, figure_inventory: pd.DataFrame) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Times-Bold", fontSize=18, leading=22, textColor=colors.HexColor("#17324D"), spaceAfter=12))
    styles.add(ParagraphStyle(name="ReportH1", parent=styles["Heading1"], fontName="Times-Bold", fontSize=14, leading=17, textColor=colors.HexColor("#17324D"), spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="ReportH2", parent=styles["Heading2"], fontName="Times-Bold", fontSize=11.5, leading=14, textColor=colors.HexColor("#235789"), spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="ReportBody", parent=styles["BodyText"], fontName="Times-Roman", fontSize=9.2, leading=12.2, spaceAfter=6))
    styles.add(ParagraphStyle(name="ReportBullet", parent=styles["BodyText"], fontName="Times-Roman", fontSize=9.0, leading=11.5, leftIndent=12, firstLineIndent=-7, bulletIndent=4, spaceAfter=3))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontName="Times-Italic", fontSize=8.2, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#374151"), spaceBefore=3, spaceAfter=7))
    styles.add(ParagraphStyle(name="SmallCell", parent=styles["BodyText"], fontName="Times-Roman", fontSize=6.8, leading=8.2))

    def footer(canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setFont("Times-Roman", 8)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawString(1.7 * cm, 1.0 * cm, "SOLETE one-second multi-resolution extension")
        canvas.drawRightString(A4[0] - 1.7 * cm, 1.0 * cm, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.55 * cm,
        leftMargin=1.55 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.5 * cm,
        title="Final One-Second Multi-Resolution Professor Extension",
        author="PV Ramp Forecasting Project",
    )
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    story: list[Any] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            text = " ".join(line.strip() for line in paragraph_lines)
            story.append(Paragraph(_escape_reportlab(text), styles["ReportBody"]))
            paragraph_lines.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("|"):
            flush_paragraph()
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            parsed = [[cell.strip() for cell in item.strip().strip("|").split("|")] for item in table_lines]
            if len(parsed) >= 2:
                parsed.pop(1)
            if parsed:
                max_columns = len(parsed[0])
                page_width = A4[0] - 3.1 * cm
                cell_width = page_width / max_columns
                data = [[Paragraph(_escape_reportlab(cell), styles["SmallCell"]) for cell in row] for row in parsed]
                table = Table(data, colWidths=[cell_width] * max_columns, repeatRows=1, hAlign="LEFT")
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEAF4")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17324D")),
                            ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9CA3AF")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
                            ("LEFTPADDING", (0, 0), (-1, -1), 3),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                            ("TOPPADDING", (0, 0), (-1, -1), 3),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ]
                    )
                )
                story.extend((table, Spacer(1, 7)))
            continue
        if line.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(_escape_reportlab(line[2:]), styles["ReportTitle"]))
        elif line.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(_escape_reportlab(line[3:]), styles["ReportH1"]))
        elif line.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(_escape_reportlab(line[4:]), styles["ReportH2"]))
        elif line.startswith("- ") or (len(line) > 3 and line[0].isdigit() and line[1:3] == ". "):
            flush_paragraph()
            item = line[2:] if line.startswith("- ") else line[3:]
            story.append(Paragraph(_escape_reportlab(item), styles["ReportBullet"], bulletText="•"))
        elif not line.strip():
            flush_paragraph()
        else:
            paragraph_lines.append(line)
        index += 1
    flush_paragraph()
    story.append(PageBreak())
    story.append(Paragraph("Figure Portfolio", styles["ReportH1"]))
    for figure_index, (_, row) in enumerate(figure_inventory.sort_values("figure").iterrows(), start=1):
        path = Path(row["path"])
        if not path.exists():
            continue
        image = Image(str(path))
        available_width = 17.4 * cm
        available_height = 8.8 * cm
        scale = min(available_width / image.imageWidth, available_height / image.imageHeight)
        image.drawWidth = image.imageWidth * scale
        image.drawHeight = image.imageHeight * scale
        caption = Paragraph(f"Figure {int(row['figure'])}. {_escape_reportlab(str(row['caption']))}", styles["Caption"])
        story.append(KeepTogether([image, caption]))
        if figure_index % 2 == 0 and figure_index < len(figure_inventory):
            story.append(PageBreak())
    document.build(story, onFirstPage=footer, onLaterPages=footer)
