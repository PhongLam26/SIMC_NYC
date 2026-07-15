# SIMC NYC Urban Service-Demand Forecasting

This repository contains the paper, source code, and compact reproducibility artifacts for:

**Category-Aware Explainable Machine Learning for Urban Service Demand Forecasting in Smart Cities**.

GitHub repository: https://github.com/PhongLam26/SIMC_NYC

The project models next-week abnormal reported NYC 311 demand at the NTA-week-complaint-category level. The final prospective alerting pipeline uses only predictors available by the end of feature week `t`: shifted 311 demand history, current complaint count, calendar timing, feature-week weather, service category, and borough identifiers.

Contemporary OSM and PLUTO 2026 snapshots are retained only for retrospective context checks. They are not used in the final LightGBM model, XGBoost model, ensemble score, category-threshold selection, headline metrics, or prospective alert production.

## Repository Contents

- `paper_springer/`: Springer LaTeX source, bibliography, figures, build script, and compiled `main.pdf`.
- `paper_overleaf/`: Overleaf-ready manuscript package with the same source, figures, references, and compiled `main.pdf`.
- `code_pulldata/`: scripts for pulling public NYC 311, NOAA, NTA, OSM, and PLUTO inputs.
- `code_processdata/`: scripts for schema checks, spatial joins, weekly aggregation, feature construction, and final dataset assembly.
- `code_model/`: scripts for dataset inspection, baselines, LightGBM/XGBoost tuning, ensemble/category thresholds, leakage audit, SHAP, checks, and paper table export.
- `configs/features_prospective.json`: standalone prospective feature protocol used by the final model.
- `data/processed/model_results/prospective/`: compact prospective model summaries, paper tables, ensemble outputs, and SHAP artifacts used by the manuscript.
- `data/processed/model_results/retrospective_context/`: compact OSM/PLUTO retrospective context checks, not final alert inputs.
- `REVISION_REPORT.md`, `prospective_leakage_audit.md`, `numerical_consistency_report.md`, and `reference_audit.md`: reviewer-facing audit reports.

Large raw data, intermediate joined 311 files, full processed modeling tables, model binaries, full scored row-level prediction dumps, and SHAP sample matrices are intentionally excluded from Git. They can be regenerated from the public data sources and scripts subject to licensing, redistribution, and storage constraints.

## Environment

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For LaTeX builds, use a TeX distribution with `pdflatex` and `bibtex` available. The current PDFs were built with MiKTeX.

## Build the Paper

```powershell
powershell -ExecutionPolicy Bypass -File .\paper_springer\build_paper.ps1 -Clean
powershell -ExecutionPolicy Bypass -File .\paper_overleaf\build_paper.ps1 -Clean
```

Expected output: `paper_springer/main.pdf` and `paper_overleaf/main.pdf`, each 12 pages.

## Prospective Modeling Protocol

The manuscript uses a target-week chronological split:

- Train target weeks: 2015-2022.
- Validation target weeks: 2023.
- Test target weeks: 2024-2025.
- Final prospective feature set: `prospective_forecast_available`.
- Prospective inputs: 61.
- Encoded model columns: 73.

Training has two stages:

- Stage A, hyperparameter tuning: LightGBM and XGBoost candidates are evaluated on a stratified 300,000-row training sample for computational reproducibility; validation 2023 selects candidates and thresholds.
- Stage B, final rebuild: selected LightGBM and XGBoost configurations are rebuilt on all 954,990 training rows before validation-only threshold selection and final 2024-2025 testing.

The scripts pass LightGBM `subsample = 0.85`, but they do not pass `subsample_freq` or `bagging_freq`; LightGBM keeps `subsample_freq = 0`, so row subsampling is inactive. XGBoost `subsample = 0.9` is active directly with `tree_method = hist`.

## Reproduce the Data Pipeline

The full data pipeline downloads public inputs and can take substantial time and disk space. Run commands from the repository root.

### 1. Data Acquisition

Core prospective inputs:

```powershell
python .\code_pulldata\pull_nyc_311.py
python .\code_pulldata\pull_noaa_weather.py
python .\code_pulldata\pull_nyc_nta.py
```

Optional retrospective context inputs:

```powershell
python .\code_pulldata\pull_osm_nyc_pois.py
python .\code_pulldata\pull_nyc_pluto_landuse.py
```

The 311 download can optionally use `SOCRATA_APP_TOKEN` to reduce throttling.

### 2. Data Processing

```powershell
python .\code_processdata\check_raw_data_schema.py
python .\code_processdata\join_311_to_nta.py
python .\code_processdata\aggregate_weekly_311_fixed_v2.py --overwrite
python .\code_processdata\build_311_temporal_features.py --overwrite
python .\code_processdata\build_weather_weekly_features.py --overwrite
python .\code_processdata\build_osm_poi_nta_features.py --overwrite
python .\code_processdata\build_pluto_nta_features.py --overwrite
python .\code_processdata\build_final_dataset.py --overwrite --progress
python .\code_model\inspect_final_dataset.py --overwrite --progress
```

The final dataset may store OSM/PLUTO columns for context checks, but the final prospective feature config excludes them.

### 3. Prospective Leakage Audit

```powershell
python .\code_model\audit_prospective_features.py `
  --config data/processed/model_ready/model_config.json `
  --prospective-config configs/features_prospective.json `
  --ensemble-dir data/processed/model_results/prospective/ensemble `
  --output prospective_leakage_audit.md
```

### 4. Prospective LightGBM Tuning

```powershell
python .\code_model\tune_gbm_hyperparams.py `
  --config data/processed/model_ready/model_config.json `
  --feature-set prospective_forecast_available `
  --analysis-type prospective `
  --models lightgbm `
  --preset balanced `
  --max-train-rows 300000 `
  --output-dir data/processed/model_results/prospective/tuning_lgbm `
  --save-models --save-predictions-for-best `
  --overwrite --progress
```

### 5. Prospective XGBoost Tuning

```powershell
python .\code_model\tune_gbm_hyperparams.py `
  --config data/processed/model_ready/model_config.json `
  --feature-set prospective_forecast_available `
  --analysis-type prospective `
  --models xgboost `
  --preset balanced `
  --max-train-rows 300000 `
  --output-dir data/processed/model_results/prospective/tuning_xgb `
  --save-models --save-predictions-for-best `
  --overwrite --progress
```

### 6-7. Final Full-Training Rebuild, Ensemble, and Category Thresholds

```powershell
python .\code_model\ensemble_and_category_thresholds.py `
  --config data/processed/model_ready/model_config.json `
  --feature-set prospective_forecast_available `
  --analysis-type prospective `
  --lgbm-tuning-dir data/processed/model_results/prospective/tuning_lgbm `
  --xgb-tuning-dir data/processed/model_results/prospective/tuning_xgb `
  --output-dir data/processed/model_results/prospective/ensemble `
  --max-train-rows 0 `
  --ensemble-weights 0.5 `
  --force-rebuild-scores `
  --save-scores `
  --overwrite --progress
```

Here `--max-train-rows 0` means rebuild the selected configurations on the full 954,990 training rows.

### 8. Paper Tables

```powershell
python .\code_model\build_paper_result_tables.py `
  --ensemble-dir data/processed/model_results/prospective/ensemble `
  --lgbm-tuning-dir data/processed/model_results/prospective/tuning_lgbm `
  --xgb-tuning-dir data/processed/model_results/prospective/tuning_xgb `
  --baselines-dir data/processed/model_results/baselines_target_week_sampled `
  --output-dir data/processed/model_results/prospective/paper_tables `
  --expected-analysis-type prospective `
  --overwrite --progress
```

Baseline comparison rows are compact historical/rule baselines. They are not final prospective alert inputs.

### 9. Prospective SHAP

```powershell
python .\code_model\run_shap_explainability.py `
  --config data/processed/model_ready/model_config.json `
  --feature-set prospective_forecast_available `
  --analysis-type prospective `
  --lgbm-tuning-dir data/processed/model_results/prospective/tuning_lgbm `
  --ensemble-dir data/processed/model_results/prospective/ensemble `
  --output-dir data/processed/model_results/prospective/shap `
  --overwrite --progress
```

SHAP explains the prospective LightGBM component, not every final ensemble alert, and should be read as predictive association rather than causality.

### Optional Retrospective Context Checks

```powershell
python .\code_model\run_paper_extended_checks.py
```

These compact checks write OSM/PLUTO context outputs under `data/processed/model_results/retrospective_context/` and prospective supplemental checks under `data/processed/model_results/prospective/paper_extended_checks/`. They are not used in final alert scoring, threshold selection, or headline metrics.

## Key Result Artifacts

- Final manuscript PDF: `paper_springer/main.pdf`
- Overleaf/reviewer PDF: `paper_overleaf/main.pdf`
- Prospective feature config: `configs/features_prospective.json`
- Prospective leakage audit: `prospective_leakage_audit.md`
- LightGBM tuning summary: `data/processed/model_results/prospective/tuning_lgbm/gbm_tuning_run_summary.json`
- XGBoost tuning summary: `data/processed/model_results/prospective/tuning_xgb/gbm_tuning_run_summary.json`
- Final ensemble/category-threshold summary: `data/processed/model_results/prospective/ensemble/ensemble_threshold_run_summary.json`
- Final model comparison table: `data/processed/model_results/prospective/paper_tables/paper_table_01_final_model_comparison.csv`
- Confusion matrix: `data/processed/model_results/prospective/paper_tables/paper_table_03_final_model_confusion.csv`
- Category thresholds: `data/processed/model_results/prospective/paper_tables/paper_table_05_category_thresholds.csv`
- Lift at K: `data/processed/model_results/prospective/paper_tables/paper_table_08_lift_at_k.csv`
- Prospective SHAP group importance: `data/processed/model_results/prospective/shap/shap_group_importance.csv`
- Prospective individual SHAP importance: `data/processed/model_results/prospective/shap/shap_global_importance.csv`
- Numerical consistency report: `numerical_consistency_report.md`
- Reference audit: `reference_audit.md`

Retrospective context checks, not used in final alerts:

- `data/processed/model_results/retrospective_context/compact_checks/retrospective_context_summary.json`
- `data/processed/model_results/retrospective_context/compact_checks/feature_group_ablation_compact.csv`

## Headline Test Metrics

For the validation-selected category-aware ensemble on the held-out 2024-2025 test period:

- F1 = 0.3802
- Precision = 0.2933
- Recall = 0.5404
- PR-AUC = 0.3310
- ROC-AUC = 0.7643
- Balanced accuracy = 0.6750
- Alert rate = 0.2349
- Confusion matrix: TP 17,059; FP 41,106; FN 14,511; TN 174,914
- Positives = 31,570; alerts = 58,165

## Data Sources

The raw inputs are public data sources, subject to their own licensing and availability:

- NYC Open Data 311 historical archive `76ig-c548`: https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-2019/76ig-c548
- NYC Open Data 311 current archive `erm2-nwe9`: https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9
- 2020 NTA boundaries `9nt8-h7nd`: https://data.cityofnewyork.us/City-Government/2020-Neighborhood-Tabulation-Areas-NTAs-/9nt8-h7nd
- NOAA GHCN-Daily station `GHCND:USW00094728`: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
- OpenStreetMap Overpass API: https://wiki.openstreetmap.org/wiki/Overpass_API
- NYC Planning PLUTO/MapPLUTO `64uk-42ks`: https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-PLUTO-/64uk-42ks

## Full Data Access

Large raw files, the full processed modeling table, row-level scores, SHAP sample matrices, and model binaries are not stored in this Git repository. They can be regenerated from the public sources above using the provided scripts.

For questions or requests for the complete local data bundle, please contact:

Tran Dai Phong Lam, `phonglam2599@gmail.com`
