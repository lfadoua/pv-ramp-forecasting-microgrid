from __future__ import annotations

from pathlib import Path

import nbformat as nbf
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "scientific_config.yaml"


NOTEBOOK_01_SOURCE = r'''from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path.cwd()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pv_ramp_automation.data_audit import run_notebook01_audit

report = run_notebook01_audit(ROOT / "configs" / "scientific_config.yaml")
print(json.dumps({
    "notebook": report["notebook_name"],
    "final_status": report["final_status"],
    "primary_resolution": report["primary_resolution"],
    "cleaned_files": report["cleaned_files"],
}, indent=2))

if report["final_status"] != "success":
    raise SystemExit(1)
'''


NOTEBOOK_02_SOURCE = r'''from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path.cwd()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pv_ramp_automation.target_features import run_notebook02_target_features

report = run_notebook02_target_features(ROOT / "configs" / "scientific_config.yaml")
print(json.dumps({
    "notebook": report["notebook_name"],
    "final_status": report["final_status"],
    "input_file": report["input_file"],
    "rows_before_target_creation": report["target_creation"]["rows_before_target_creation"],
    "rows_after_target_creation": report["target_creation"]["rows_after_target_creation"],
    "original_feature_count": report["original_feature_count"],
    "selected_feature_count": report["selected_feature_count"],
    "leakage_checks_passed": report["leakage_checks_passed"],
}, indent=2))

if report["final_status"] != "success":
    raise SystemExit(1)
'''


NOTEBOOK_03_SOURCE = r'''from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path.cwd()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pv_ramp_automation.physics_features import run_notebook03_physics_features

report = run_notebook03_physics_features(ROOT / "configs" / "scientific_config.yaml")
print(json.dumps({
    "notebook": report["notebook_name"],
    "final_status": report["final_status"],
    "full_rows_before": report["full_dataset"]["rows_before_feature_engineering"],
    "full_rows_after": report["full_dataset"]["rows_after_feature_engineering"],
    "selected_features_before": report["selected_feature_counts"]["before_physics_enrichment"],
    "selected_features_after": report["selected_feature_counts"]["after_physics_enrichment"],
    "full_features_before": report["full_feature_counts"]["before_physics_enrichment"],
    "full_features_after": report["full_feature_counts"]["after_physics_enrichment"],
    "leakage_checks_passed": report["leakage_checks"]["passed"],
}, indent=2))

if report["final_status"] != "success":
    raise SystemExit(1)
'''


NOTEBOOK_04_SOURCE = r'''from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path.cwd()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pv_ramp_automation.split_windowing import run_notebook04_split_windowing

report = run_notebook04_split_windowing(ROOT / "configs" / "scientific_config.yaml")
print(json.dumps({
    "notebook": report["notebook_name"],
    "final_status": report["final_status"],
    "split_ratio": report["split_ratio"],
    "dataset_variants": sorted(report["variant_reports"].keys()),
    "experiment_count": report["experiment_matrix"]["row_count"],
    "leakage_checks_passed": report["leakage_checks"]["checks"]["passed"],
    "model_training_started": report["model_training_started"],
}, indent=2))

if report["final_status"] != "success":
    raise SystemExit(1)
'''


NOTEBOOK_05_SOURCE = r'''from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path.cwd()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pv_ramp_automation.baselines import run_notebook05_baselines

report = run_notebook05_baselines(ROOT / "configs" / "scientific_config.yaml")
print(json.dumps({
    "notebook": report["notebook_name"],
    "final_status": report["final_status"],
    "dataset_variants": report.get("dataset_variants"),
    "baseline_models": report.get("baseline_models"),
    "experiment_config_count": report.get("experiment_config_count"),
    "result_row_count": report.get("result_row_count"),
    "segment_metrics_produced": report.get("segment_metrics_produced"),
    "leakage_alignment_checks_passed": report.get("validation_checks", {}).get("passed"),
    "model_training_started": report.get("model_training_started"),
    "notebook_06_started": report.get("notebook_06_started"),
}, indent=2))

if report["final_status"] != "success":
    raise SystemExit(1)
'''


NOTEBOOK_06_SOURCE = r'''from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path.cwd()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pv_ramp_automation.statistical_models import run_notebook06_statistical_models

report = run_notebook06_statistical_models(ROOT / "configs" / "scientific_config.yaml")
print(json.dumps({
    "notebook": report["notebook_name"],
    "final_status": report["final_status"],
    "dataset_variant": report.get("dataset_variant"),
    "models_attempted": report.get("models_attempted"),
    "orders_configurations_tested": report.get("orders_configurations_tested"),
    "horizons_evaluated": report.get("horizons_evaluated"),
    "result_row_count": report.get("result_row_count"),
    "validation_checks_passed": report.get("validation_checks", {}).get("passed"),
    "notebook_07_started": report.get("notebook_07_started"),
    "classical_ml_training_started": report.get("classical_ml_training_started"),
    "deep_learning_training_started": report.get("deep_learning_training_started"),
}, indent=2))

if report["final_status"] != "success":
    raise SystemExit(1)
'''


SKELETON_SOURCE = r'''from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_ID = "{notebook_id}"
NOTEBOOK_SLUG = "{slug}"
TITLE = "{title}"

ROOT = Path.cwd()
OUTPUT_DIR = ROOT / "outputs" / NOTEBOOK_SLUG
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

skeleton_report = {{
    "notebook_id": NOTEBOOK_ID,
    "notebook_slug": NOTEBOOK_SLUG,
    "title": TITLE,
    "phase1_status": "skeleton_created_not_executed",
    "heavy_training_started": False,
    "chronological_split_required": True,
    "target_creation_allowed_here": NOTEBOOK_ID == "02",
    "official_nrmse": "RMSE / 2",
    "skill_score_vs_persistence_required": True,
}}

path = OUTPUT_DIR / "skeleton_report.json"
path.write_text(json.dumps(skeleton_report, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(skeleton_report, indent=2))
'''


def markdown_for_notebook(item: dict) -> str:
    notebook_id = item["id"]
    slug = item["slug"]
    title = item["title"]
    base = [
        f"# Notebook {notebook_id} - {title}",
        "",
        "Project label: Short-term PV ramp-rate forecasting model for microgrid control support.",
        "",
        "Methodology guardrails:",
        "- Chronological split only.",
        "- No random shuffle.",
        "- Validation is used for model/lookback/horizon selection.",
        "- Test is used only for final confirmation/reporting.",
        "- Official NRMSE is RMSE / 2.",
        "- Skill score vs Persistence is required where applicable.",
        "- Heavy/full training must not run before smoke and exploratory validation.",
        "",
    ]
    if notebook_id == "01":
        base.extend([
            "Purpose:",
            "- Audit SOLETE H5 files and available resolutions.",
            "- Select the primary resolution for the official pipeline.",
            "- Clean selected resolution data.",
            "- Do not create ramp_rate_pu.",
            "",
            "Expected outputs:",
            "- outputs/notebook_01_data_audit/resolution_inventory.csv",
            "- outputs/notebook_01_data_audit/resolution_quality_report.json",
            "- outputs/notebook_01_data_audit/resolution_selection_report.md",
            "- outputs/one_minute/notebook_01_cleaning/original_data_all_features_clean.csv",
            "- outputs/notebook_01_data_audit/notebook_01_final_report.json",
        ])
    elif notebook_id == "02":
        base.extend([
            "Purpose:",
            "- Create the official `ramp_rate_pu` target from one-minute cleaned SOLETE data.",
            "- Analyze target distribution, time behavior and quality.",
            "- Select a leakage-safe feature subset using only a provisional chronological training segment.",
            "- Do not create future targets, lookback/horizon windows, or model training outputs.",
            "",
            "Expected outputs:",
            "- outputs/one_minute/notebook_02_target_features/original_data_all_features_clean_with_target.csv",
            "- outputs/one_minute/notebook_02_target_features/selected_features_with_target.csv",
            "- outputs/02_target_construction_feature_selection/target_analysis_report.json",
            "- outputs/02_target_construction_feature_selection/feature_selection_report.csv",
            "- outputs/02_target_construction_feature_selection/selected_features_list.json",
            "- outputs/02_target_construction_feature_selection/dropped_features_report.csv",
            "- outputs/02_target_construction_feature_selection/notebook_02_final_report.json",
        ])
    elif notebook_id == "03":
        base.extend([
            "Purpose:",
            "- Create physics-informed and ramp-aware features from Notebook 02 datasets.",
            "- Preserve causality: lags are strictly past, rolling windows are past/current only.",
            "- Do not create future targets, supervised lookback/horizon windows, or model training outputs.",
            "- Keep `ramp_rate_pu` as the last column and do not use it as an input feature.",
            "",
            "Expected outputs:",
            "- outputs/one_minute/notebook_03_physics_features/original_all_features_plus_physics_informed_with_target.csv",
            "- outputs/one_minute/notebook_03_physics_features/selected_features_plus_physics_informed_with_target.csv",
            "- outputs/03_physics_informed_features/physics_informed_feature_report.csv",
            "- outputs/03_physics_informed_features/feature_dictionary.json",
            "- outputs/03_physics_informed_features/physics_informed_leakage_check.json",
            "- outputs/03_physics_informed_features/notebook_03_final_report.json",
        ])
    elif notebook_id == "04":
        base.extend([
            "Purpose:",
            "- Create official chronological train/validation/test split metadata.",
            "- Fit feature scalers only on train and save train-only scaler artifacts.",
            "- Prepare lookback/horizon windowing metadata without materializing model windows.",
            "- Build the experiment matrix for all dataset variants.",
            "- Do not train models and do not start Notebook 05.",
            "",
            "Expected outputs:",
            "- outputs/one_minute/notebook_04_split_windowing/<variant>/window_ready_manifest.json",
            "- outputs/04_split_scaling_windowing_experiment_design/experiment_matrix.csv",
            "- outputs/04_split_scaling_windowing_experiment_design/split_metadata.json",
            "- outputs/04_split_scaling_windowing_experiment_design/windowing_metadata.json",
            "- outputs/04_split_scaling_windowing_experiment_design/scaler_metadata.json",
            "- outputs/04_split_scaling_windowing_experiment_design/leakage_check_report.json",
            "- outputs/04_split_scaling_windowing_experiment_design/notebook_04_final_report.json",
        ])
    elif notebook_id == "05":
        base.extend([
            "Purpose:",
            "- Evaluate only Persistence and Mean Naive baselines on the Notebook 04 split/window protocol.",
            "- Use all four dataset variants and all lookback/horizon combinations.",
            "- Keep target alignment strict: Persistence predicts ramp_rate_pu(t+H) with ramp_rate_pu(t).",
            "- Fit the Mean Naive constant on aligned training targets only.",
            "- Do not train ML, deep learning, or statistical models and do not start Notebook 06.",
            "",
            "Expected outputs:",
            "- outputs/05_baseline_models/baseline_results_train_validation_test.csv",
            "- outputs/05_baseline_models/baseline_results_summary.csv",
            "- outputs/05_baseline_models/persistence_reference_by_variant_lh.csv",
            "- outputs/05_baseline_models/baseline_predictions_validation.csv",
            "- outputs/05_baseline_models/baseline_predictions_test.csv",
            "- outputs/05_baseline_models/baseline_segment_metrics.csv",
            "- outputs/05_baseline_models/notebook_05_final_report.json",
        ])
    elif notebook_id == "06":
        base.extend([
            "Purpose:",
            "- Evaluate ARIMA and SARIMAX with a statistical walk-forward protocol.",
            "- Do not force ARIMA/SARIMAX into ML-style flattened lookback windows.",
            "- Use fit-once plus state-update behavior and evaluate forecast step H from forecast(steps=H).",
            "- Use Notebook 05 Persistence RMSE as the Skill_RMSE reference.",
            "- Do not train classical ML or deep learning models and do not start Notebook 07.",
            "",
            "Expected outputs:",
            "- outputs/06_statistical_time_series_models/statistical_results_train_validation_test.csv",
            "- outputs/06_statistical_time_series_models/statistical_results_summary.csv",
            "- outputs/06_statistical_time_series_models/statistical_model_protocol_report.md",
            "- outputs/06_statistical_time_series_models/statistical_skipped_configs_report.csv",
            "- outputs/06_statistical_time_series_models/statistical_predictions_validation.csv",
            "- outputs/06_statistical_time_series_models/statistical_predictions_test.csv",
            "- outputs/06_statistical_time_series_models/notebook_06_final_report.json",
        ])
    else:
        base.extend([
            "Phase 1 status:",
            "- Skeleton only.",
            "- Implementation will be added after Notebook 01 and smoke validation are accepted.",
            "",
            "Expected behavior:",
            f"- Save outputs under outputs/{slug}/.",
            "- Use the central scientific config.",
            "- Follow the staging policy: smoke_test, exploratory, final.",
        ])
    return "\n".join(base) + "\n"


def build_notebook(item: dict) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb.cells = [nbf.v4.new_markdown_cell(markdown_for_notebook(item))]
    if item["id"] == "01":
        nb.cells.append(nbf.v4.new_code_cell(NOTEBOOK_01_SOURCE))
    elif item["id"] == "02":
        nb.cells.append(nbf.v4.new_code_cell(NOTEBOOK_02_SOURCE))
    elif item["id"] == "03":
        nb.cells.append(nbf.v4.new_code_cell(NOTEBOOK_03_SOURCE))
    elif item["id"] == "04":
        nb.cells.append(nbf.v4.new_code_cell(NOTEBOOK_04_SOURCE))
    elif item["id"] == "05":
        nb.cells.append(nbf.v4.new_code_cell(NOTEBOOK_05_SOURCE))
    elif item["id"] == "06":
        nb.cells.append(nbf.v4.new_code_cell(NOTEBOOK_06_SOURCE))
    else:
        nb.cells.append(nbf.v4.new_code_cell(SKELETON_SOURCE.format(notebook_id=item["id"], slug=item["slug"], title=item["title"])))
    return nb


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    target_dir = ROOT / "notebooks" / "local_development"
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in config["notebooks"]["sequence"]:
        if item["id"] not in {"01", "02", "03", "04", "05", "06"}:
            continue
        path = target_dir / f"{item['slug']}.ipynb"
        nbf.write(build_notebook(item), path)
        print(path)


if __name__ == "__main__":
    main()
