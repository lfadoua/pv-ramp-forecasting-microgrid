# Physics-guided PV ramp forecasting for microgrid control support

A final-year engineering research project on forecasting signed photovoltaic (PV) power increments from causal weather, power, and physics-guided features. The official benchmark uses SOLETE at one-minute and one-hour resolution. Separate S1-S3 experiments validate target semantics and input resolution on SOLETE and four modeled-power UNSW days.

## Scientific protocol

The target is `(P[t] - P[t-1]) / rated_power`, with a 10 kW SOLETE rating. A direct forecast at lead H predicts the local increment at t+H, not net change since t. Each resolution uses chronological 70/15/15 train/validation/test splits; scalers fit on training only and windows stay inside each split. Validation RMSE selects models. NRMSE is always RMSE/2. Seed 42 is used where supported.

The selected representation has nine original predictors plus 74 derived features (51 temporal/statistical and 23 physics-guided), giving 83 columns before lookback flattening. This is physics-guided feature engineering, not a physics-informed neural network.

## Pipeline architecture

```text
SOLETE HDF /DATA -> timestamp sort and audit -> normalized ramp target
 -> train-only feature selection -> causal feature engineering
 -> chronological split -> train-only scaler -> split-local L/H windows
 -> baselines / classical models / separate Kaggle neural jobs
 -> validation-first selection -> held-out evaluation and evidence tables
```

Core implementation is in `src/pv_ramp_automation/`. Notebook-generation and job-preparation scripts are in `scripts/`. The actual neural training code is preserved in `training_notebooks/`. UNSW replication code is in `external_validation/`, with compact results under `evidence/`. This repository contains pipeline code, configuration, notebooks, tests and compact experimental evidence.

## Validated official H=1 results

Both selected models use Selected Physics-Guided inputs and L=8.

| Resolution | Validation-selected model | Validation RMSE | Test RMSE | Test NRMSE |
|---|---|---:|---:|---:|
| One minute | HistGradientBoosting | 0.0352086725 | 0.0338818117 | 0.0169409059 |
| One hour | ExtraTrees | 0.0500065199 | 0.0466553157 | 0.0233276578 |

Minute BiLSTM and hourly XGBoost have lower test RMSE in their comparisons but worse validation RMSE; they do not replace the official selections. Compact archived Stage C rows and the explicit derived RMSE/2 table are included in `evidence/`. Historical incompatible NRMSE cells are retained as evidence, with corrections recorded separately.

## Supplementary validation

S1 predicts the next one-second increment. S2 predicts a one-second increment 60 seconds ahead, not a 60-second net change. SOLETE S1/S2 use deterministic 30-second anchors and 480-second histories; test RMSE is 0.00663603/0.00677814. S2 strong-event recall is only 0.01266. S3 predicts the next official minute target; a reconstructed minute-only reference has RMSE 0.03404618 and the second-enriched arm 0.02693913 on 98,695 identical targets. The 20.875% reduction is supplementary and does not replace official Stage C.

UNSW uses four grouped leave-one-day-out days, modeled uncontrolled power, a 280 W rating, and zero-floor preprocessing for 1,129 negative Day D values. Mean held-out RMSE is 0.0121089 (S1), 0.0118360 (S2), 0.0603622 (S3 minute-only), and 0.0582383 (S3 enriched). Day D worsens slightly for enriched S3. These are newly fitted dataset-specific models, not transfer of SOLETE estimators or control validation.

## Setup and feasible verification

Python 3.10+ is required; current local checks use Python 3.11. Create a fresh environment from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/verify_published_results.py
```

Optional training/report backends are listed in `requirements-optional.txt`; install only those needed for your platform. Requirements are lower bounds, not a recovered original lockfile. No credentials are needed for tests or evidence verification. `.env.example` lists empty optional Kaggle/ClearML variables; never publish their values. Personal Kaggle identifiers in copied code are placeholders to configure locally.

## Data and execution

Raw datasets are intentionally excluded. The current hourly HDF is stored out of chronological order (456 transitions); ingestion sorts timestamps before target construction. After sorting, both official series have exact cadence, unique timestamps, and no missing values. SOLETE source DOI: `10.11583/DTU.17040626`; UNSW source DOI: `10.26190/5c8ed32da8e8f`. Obtain data from the providers and respect their terms. The default relative layout expects a sibling `data/raw/solete_official/` directory containing `SOLETE_Pombo_1min.h5`, `SOLETE_Pombo_60min.h5`, and (only for the supplementary workflow) `SOLETE_Pombo_1sec.h5`. Edit the configuration paths for another layout.

`python scripts/generate_pipeline_notebooks.py` writes implemented stages 01-06 only; it does not generate unfinished notebook skeletons. Inspect those notebooks and run stages deliberately in order. Generated historical runners depend on prior-stage outputs and must be reviewed for each resolution; they are not a single command guaranteed to reproduce every published experiment. `prepare_stage_c_execution_package.py` prepares manually guarded Kaggle jobs from prerequisite artifacts. Neural notebooks are fresh-run templates with cleared outputs; retained evidence proves past runs, not these cleaned copies. `run_final_one_second_professor_extension.py` is resource-intensive and must be deliberately authorized before a fresh run. UNSW starts with `external_validation/UNSW_S1_S2_S3_v1/07_traceability/audit_and_preprocess.py`, then `run_gate_b_validation.py`. Its source layout is `00_source/extracted/Data/PV Irradiance and Power Data/` with the four original `PV_Power_Day_A.txt` through `PV_Power_Day_D.txt` files. The same source directory also requires the supplied Solar Irradiance_46days.txt and Time Series_1day.txt files. The scripts create their processed data and grouped validation outputs locally. Full inputs and fitted models remain local.

## Reproducibility and limitations

Current verification checks saved tables and lightweight target/window/causality logic without retraining. Original Stage C model binaries and full predictions were not retained; exact model replay cannot be promised. Notebook 09 was not executed. Deep results are single runs. Job hyperparameters and hardware differ between distributed groups. HGB uses random internal early-stopping holdout within training. Hourly solar-angle fields are mostly zero. Source minute power differs from arithmetic second aggregation. UNSW has only four modeled-power days and no calendar timestamps. There is no real-time, controller, storage, inverter, EMS, MPC, closed-loop, or HIL validation.

## Files included and reproducibility

Only pipeline implementation, scientific configuration, training notebooks, verification tests, dependency files and compact benchmark evidence are included. This README is the only Markdown file. Thesis, slide, manuscript and publication-figure assets are excluded.

`evidence/solete/` preserves official minute/hour tables, baseline and neural summaries, feature/split audits and supplementary S1-S3 results. `evidence/unsw/` preserves grouped held-out results. `evidence/provenance.json` records original source-relative paths and SHA-256 values alongside hashes of sanitized publication copies. Other local archives remain outside the repository.

Use `evidence/solete/stage_c_summary_canonical.csv` for RMSE/2 NRMSE. The archived Stage C exports retain incompatible NRMSE cells for traceability; nine train/validation/test cells from minute ExtraTrees must be recomputed as RMSE/2. Historical deep H8/H16 exports also contained 24 incompatible NRMSE cells; the compact neural summary retained here reports RMSE. No RMSE value or validation-selected model was changed.

For explicit Python API execution after installing the package, the processing functions are `run_notebook01_audit`, `run_notebook02_target_features`, `run_notebook03_physics_features`, `run_notebook04_split_windowing`, `run_notebook05_baselines` and `run_notebook06_statistical_models` in their corresponding modules. `pre_notebook07_cleanup.run_pre_notebook07_cleanup` prepares the hourly/variant prerequisites; `baseline_completion.run_notebook05b_one_hour_baseline_completion` completes hourly baselines; `classical_ml.run_notebook07_stage_a` performs screening. These functions write new local outputs and can train models; inspect prerequisites and resource needs before execution. Stage C and neural training remain separate deliberate jobs, not an automatically executed full experiment.

Tests and saved-table checks need no credentials and do not retrain. Current checks use Python 3.11.9; dependency lower bounds do not recreate an original frozen training environment. Original commit provenance is unavailable because the original parent Git directory was empty. No original historical Git security scan or exact binary replay is claimed.
