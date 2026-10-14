from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "00_source" / "extracted" / "Data" / "PV Irradiance and Power Data"
AUDIT = WORKSPACE / "01_audit"
PROCESSED = WORKSPACE / "02_processed_data"
TRACE = WORKSPACE / "07_traceability"

RATED_POWER_W = 280.0
DAY_LENGTH = 50_400
GLOBAL_OFFSETS_ZERO_BASED = {"A": 104_400, "B": 277_200, "C": 450_000, "D": 622_800}
POWER_FILES = {day: SOURCE / f"PV_Power_Day_{day}.txt" for day in "ABCD"}
IRRADIANCE_FILE = SOURCE / "Solar Irradiance_46days.txt"
TIME_FILE = SOURCE / "Time Series_1day.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_vector(path: Path) -> np.ndarray:
    values = np.loadtxt(path, skiprows=1, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"Expected a finite one-dimensional vector: {path}")
    return values


def elapsed_hhmmss(second_one_based: int) -> str:
    elapsed = int(second_one_based) - 1
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate(([0], breaks + 1))
    stops = np.concatenate((breaks + 1, [len(indices)]))
    return [(int(indices[start]), int(indices[stop - 1])) for start, stop in zip(starts, stops)]


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    time_coordinate = load_vector(TIME_FILE)
    irradiance = load_vector(IRRADIANCE_FILE)
    if len(time_coordinate) != DAY_LENGTH or not np.array_equal(time_coordinate, np.arange(1, DAY_LENGTH + 1)):
        raise AssertionError("The one-day coordinate is not the expected 1..50400 sequence.")

    raw_manifest_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    processed_frames: list[pd.DataFrame] = []
    equivalence_rows: list[dict[str, object]] = []

    rule = {
        "rule_id": "clip_negative_modeled_pv_power_v1",
        "defined_before_target_generation": True,
        "formula": "p_processed_w = max(p_raw_w, 0 W)",
        "scope": "Apply independently to every supplied Day A-D modeled PV-power sample before minute aggregation, target construction, or feature construction.",
        "negative_values": "Set to exactly 0 W.",
        "zero_and_positive_values": "Preserve unchanged.",
        "upper_clipping": "None; values above the 280 W rating remain unchanged.",
        "interpolation": "None.",
        "row_removal": "None.",
        "rationale": "Negative PV generation is physically inadmissible. A deterministic zero floor preserves timestamps and avoids data-dependent imputation.",
        "primary_feature_source": "Processed modeled uncontrolled PV power only.",
        "irradiance_feature_policy": "Excluded because raw modeled power is exactly 0.28 times aligned irradiance and therefore provides no independent information source.",
        "rated_power_w": RATED_POWER_W,
    }
    write_json(PROCESSED / "preprocessing_rule.json", rule)

    for day in "ABCD":
        raw_path = POWER_FILES[day]
        raw = load_vector(raw_path)
        if len(raw) != DAY_LENGTH:
            raise AssertionError(f"Day {day} has {len(raw)} rows instead of {DAY_LENGTH}.")
        global_offset = GLOBAL_OFFSETS_ZERO_BASED[day]
        aligned_irradiance = irradiance[global_offset : global_offset + DAY_LENGTH]
        equivalence_error = raw - 0.28 * aligned_irradiance
        equivalence_rows.append(
            {
                "day": day,
                "sample_count": DAY_LENGTH,
                "global_time_start": global_offset + 1,
                "global_time_end": global_offset + DAY_LENGTH,
                "rmse_w": float(np.sqrt(np.mean(np.square(equivalence_error)))),
                "mae_w": float(np.mean(np.abs(equivalence_error))),
                "max_abs_error_w": float(np.max(np.abs(equivalence_error))),
                "source_text_precision_equivalent": bool(np.max(np.abs(equivalence_error)) < 1.5e-7),
            }
        )

        negative_indices = np.flatnonzero(raw < 0.0)
        negative_values = raw[negative_indices]
        magnitudes = -negative_values
        runs = contiguous_runs(negative_indices)
        run_lookup: dict[int, int] = {}
        for run_id, (start, end) in enumerate(runs, start=1):
            run_values = raw[start : end + 1]
            run_rows.append(
                {
                    "day": day,
                    "run_id": run_id,
                    "start_second": start + 1,
                    "end_second": end + 1,
                    "start_elapsed_hhmmss": elapsed_hhmmss(start + 1),
                    "end_elapsed_hhmmss": elapsed_hhmmss(end + 1),
                    "sample_count": end - start + 1,
                    "minimum_power_w": float(np.min(run_values)),
                    "maximum_magnitude_below_zero_w": float(-np.min(run_values)),
                }
            )
            for index in range(start, end + 1):
                run_lookup[index] = run_id

        for index in negative_indices:
            position_rows.append(
                {
                    "day": day,
                    "sample_index_zero_based": int(index),
                    "within_day_second": int(index + 1),
                    "window_elapsed_hhmmss": elapsed_hhmmss(int(index + 1)),
                    "global_time_coordinate": int(global_offset + index + 1),
                    "contiguous_run_id": int(run_lookup[int(index)]),
                    "raw_power_w": float(raw[index]),
                    "magnitude_below_zero_w": float(-raw[index]),
                    "aligned_raw_irradiance_w_m2": float(aligned_irradiance[index]),
                }
            )

        processed = np.maximum(raw, 0.0)
        clipped = raw < 0.0
        processed_frames.append(
            pd.DataFrame(
                {
                    "day": day,
                    "within_day_second": np.arange(1, DAY_LENGTH + 1, dtype=np.int32),
                    "global_time_coordinate": np.arange(global_offset + 1, global_offset + DAY_LENGTH + 1, dtype=np.int64),
                    "power_raw_w": raw,
                    "power_processed_w": processed,
                    "clipped_negative_flag": clipped,
                }
            )
        )

        negative_count = int(len(negative_indices))
        summary_rows.append(
            {
                "day": day,
                "sample_count": DAY_LENGTH,
                "minimum_power_w": float(np.min(raw)),
                "negative_count": negative_count,
                "negative_percentage": float(100.0 * negative_count / DAY_LENGTH),
                "maximum_magnitude_below_zero_w": float(np.max(magnitudes)) if negative_count else 0.0,
                "mean_magnitude_below_zero_w": float(np.mean(magnitudes)) if negative_count else 0.0,
                "median_magnitude_below_zero_w": float(np.median(magnitudes)) if negative_count else 0.0,
                "sum_magnitude_below_zero_w": float(np.sum(magnitudes)) if negative_count else 0.0,
                "first_negative_second": int(negative_indices[0] + 1) if negative_count else None,
                "last_negative_second": int(negative_indices[-1] + 1) if negative_count else None,
                "contiguous_negative_runs": int(len(runs)),
                "longest_negative_run_samples": int(max((end - start + 1 for start, end in runs), default=0)),
                "processed_minimum_w": float(np.min(processed)),
                "values_changed_by_rule": negative_count,
            }
        )
        raw_manifest_rows.append(
            {
                "day": day,
                "source_path": str(raw_path.relative_to(WORKSPACE)).replace("\\", "/"),
                "sha256": sha256(raw_path),
                "header": "Power (W)",
                "unit": "W",
                "sample_count": DAY_LENGTH,
                "global_time_start": global_offset + 1,
                "global_time_end": global_offset + DAY_LENGTH,
                "minimum_w": float(np.min(raw)),
                "maximum_w": float(np.max(raw)),
                "negative_count": negative_count,
                "immutable_source": True,
            }
        )

    summary = pd.DataFrame(summary_rows)
    positions = pd.DataFrame(position_rows)
    runs = pd.DataFrame(run_rows)
    equivalence = pd.DataFrame(equivalence_rows)
    processed_data = pd.concat(processed_frames, ignore_index=True)

    summary.to_csv(AUDIT / "negative_value_audit_summary.csv", index=False, float_format="%.12g", lineterminator="\n")
    positions.to_csv(AUDIT / "negative_value_positions.csv", index=False, float_format="%.12g", lineterminator="\n")
    runs.to_csv(AUDIT / "negative_value_contiguous_runs.csv", index=False, float_format="%.12g", lineterminator="\n")
    equivalence.to_csv(AUDIT / "linear_equivalence_by_day.csv", index=False, float_format="%.12g", lineterminator="\n")
    pd.DataFrame(raw_manifest_rows).to_csv(PROCESSED / "raw_data_manifest.csv", index=False, float_format="%.12g", lineterminator="\n")

    processed_path = PROCESSED / "UNSW_power_days_processed.csv"
    processed_data.to_csv(processed_path, index=False, float_format="%.12g", lineterminator="\n")
    processed_hash = sha256(processed_path)
    processed_manifest_rows = []
    for day in "ABCD":
        day_frame = processed_data.loc[processed_data["day"] == day]
        processed_manifest_rows.append(
            {
                "day": day,
                "processed_path": str(processed_path.relative_to(WORKSPACE)).replace("\\", "/"),
                "processed_file_sha256": processed_hash,
                "row_start_zero_based": int(day_frame.index.min()),
                "row_end_zero_based_inclusive": int(day_frame.index.max()),
                "sample_count": int(len(day_frame)),
                "preprocessing_rule_id": rule["rule_id"],
                "changed_values": int(day_frame["clipped_negative_flag"].sum()),
                "minimum_processed_w": float(day_frame["power_processed_w"].min()),
                "maximum_processed_w": float(day_frame["power_processed_w"].max()),
                "missing_values": int(day_frame.isna().sum().sum()),
            }
        )
    pd.DataFrame(processed_manifest_rows).to_csv(
        PROCESSED / "processed_data_manifest.csv", index=False, float_format="%.12g", lineterminator="\n"
    )

    total_negative = int(summary["negative_count"].sum())
    total_samples = int(summary["sample_count"].sum())
    audit_payload = {
        "generated_at_utc": generated_at,
        "completed_before_target_generation": True,
        "rule": rule,
        "days": summary_rows,
        "overall": {
            "sample_count": total_samples,
            "negative_count": total_negative,
            "negative_percentage": 100.0 * total_negative / total_samples,
            "minimum_power_w": float(processed_data["power_raw_w"].min()),
            "maximum_magnitude_below_zero_w": float(-min(0.0, processed_data["power_raw_w"].min())),
            "values_changed": total_negative,
            "rows_removed": 0,
        },
        "linear_equivalence": {
            "formula": "p_raw_w = 0.28 * G_raw_w_m2",
            "all_201600_samples_verified": bool(equivalence["source_text_precision_equivalent"].all()),
            "maximum_abs_error_w_across_days": float(equivalence["max_abs_error_w"].max()),
            "independent_predictor_sources": 1,
            "primary_predictor": "processed modeled uncontrolled PV power",
            "excluded_redundant_predictor": "raw irradiance",
        },
        "outputs": {
            "summary_csv": "01_audit/negative_value_audit_summary.csv",
            "positions_csv": "01_audit/negative_value_positions.csv",
            "runs_csv": "01_audit/negative_value_contiguous_runs.csv",
            "raw_manifest_csv": "02_processed_data/raw_data_manifest.csv",
            "processed_manifest_csv": "02_processed_data/processed_data_manifest.csv",
            "processed_power_csv": "02_processed_data/UNSW_power_days_processed.csv",
        },
    }
    write_json(AUDIT / "negative_value_audit.json", audit_payload)
    write_json(
        AUDIT / "linear_equivalence_report.json",
        {
            "generated_at_utc": generated_at,
            "formula": "p_raw_w = 0.28 * G_raw_w_m2",
            "rated_power_interpretation": "0.28 W per (W/m^2), equivalent to 280 W at 1000 W/m^2",
            "per_day": equivalence_rows,
            "all_samples_verified": bool(equivalence["source_text_precision_equivalent"].all()),
            "limitation": "Modeled power and irradiance are linearly equivalent and cannot be counted as independent information sources. Irradiance is excluded from the primary feature matrices.",
        },
    )

    audit_md = f"""# Negative-Value Audit and Frozen Preprocessing Rule

This audit was completed before target or feature generation.

## Result

- Total modeled-power samples: {total_samples:,}.
- Negative samples: {total_negative:,} ({100.0 * total_negative / total_samples:.6f}%).
- Overall minimum: {processed_data['power_raw_w'].min():.9f} W.
- Maximum magnitude below zero: {-min(0.0, processed_data['power_raw_w'].min()):.9f} W.
- Days A-C contain no negative modeled-power samples.
- Day D contains {int(summary.loc[summary['day'] == 'D', 'negative_count'].iloc[0]):,} negatives ({float(summary.loc[summary['day'] == 'D', 'negative_percentage'].iloc[0]):.6f}% of Day D).

Exact positions and magnitudes are recorded in `negative_value_positions.csv`; contiguous intervals are recorded in `negative_value_contiguous_runs.csv`.

## Deterministic preprocessing rule

`p_processed_w = max(p_raw_w, 0 W)` for every sample before aggregation, targets, or features. Negative values become exactly zero. Zero and positive values are unchanged. No row is removed, no interpolation is used, and no upper clipping is applied.

The rule is fixed in `02_processed_data/preprocessing_rule.json`. Raw source files remain immutable. The consolidated processed file retains both `power_raw_w` and `power_processed_w` plus a clipping flag.

## Linear equivalence and predictor limitation

For every one of the 201,600 supplied samples, `p_raw_w = 0.28 G_raw_w_m2` within a maximum absolute source-text error of {equivalence['max_abs_error_w'].max():.3e} W. Modeled power and irradiance therefore contain the same information and are not independent predictor sources.

The official target is constructed from processed modeled uncontrolled PV power. Primary features use that same single base signal only. Irradiance is excluded from the feature matrices; derived lags, rolling statistics, ramps, and within-minute descriptors are transformations of one signal, not additional information sources.
"""
    (AUDIT / "negative_value_audit.md").write_text(audit_md, encoding="utf-8")
    (AUDIT / "linear_equivalence_report.md").write_text(
        "# Linear Equivalence Report\n\n"
        "All four supplied modeled-power windows satisfy `p_raw_w = 0.28 G_raw_w_m2` to source-text precision. "
        f"The maximum absolute error across {total_samples:,} samples is {equivalence['max_abs_error_w'].max():.3e} W.\n\n"
        "Consequently, modeled power and irradiance are not independent predictors and must not be counted as two information sources. "
        "The primary UNSW feature matrices use processed modeled power only.\n",
        encoding="utf-8",
    )

    with (TRACE / "execution_log.txt").open("a", encoding="utf-8") as log:
        log.write(f"{generated_at} | Gate B Phase 2: negative-value audit completed before targets.\n")
        log.write(f"{generated_at} | Frozen preprocessing: max(raw modeled power, 0 W); {total_negative} values changed, zero rows removed.\n")
        log.write(f"{generated_at} | Linear equivalence verified for {total_samples} samples; irradiance excluded as a redundant predictor.\n")

    print(
        json.dumps(
            {
                "status": "negative audit and preprocessing complete",
                "total_samples": total_samples,
                "negative_count": total_negative,
                "negative_percentage": 100.0 * total_negative / total_samples,
                "minimum_power_w": float(processed_data["power_raw_w"].min()),
                "processed_sha256": processed_hash,
                "targets_generated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
