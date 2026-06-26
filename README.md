# SIMC NYC Urban Service-Demand Forecasting

This repository contains the paper, source code, and compact result artifacts for the SIMC NYC study:

**Semantics-Aware Explainable Machine Learning for Urban Service Demand Forecasting in Smart Cities**.

The project models next-week abnormal increases in reported NYC 311 service demand at the NTA-week-complaint-category level. It combines NYC 311 requests, NOAA weather, OpenStreetMap POI context, and NYC PLUTO/MapPLUTO land-use context, then evaluates leakage-safe target-week chronological splits with LightGBM, XGBoost, ensemble thresholding, and SHAP explanations.

## Repository Contents

- `paper_springer/`: Springer LaTeX source, figures, bibliography, build script, and compiled `main.pdf`.
- `code_pulldata/`: scripts for pulling public NYC 311, NOAA, NTA, OSM, and PLUTO inputs.
- `code_processdata/`: scripts for schema checks, spatial joins, weekly aggregation, feature construction, and final dataset assembly.
- `code_model/`: scripts for baselines, tree models, hyperparameter tuning, ensemble/category thresholds, SHAP, and paper table export.
- `data/processed/model_results/paper_tables_target_week_sampled/`: compact paper-ready result tables matching the submitted manuscript.
- `data/processed/model_results/*`: selected small CSV/JSON/PNG artifacts needed to audit reported metrics and SHAP outputs.
- `data/processed/_*_summaries/`: compact summary files documenting dataset construction and feature coverage.

Large raw data, intermediate joined 311 files, full final modeling dataset, model binaries, and scored row-level prediction dumps are intentionally excluded from Git. They can be regenerated from the public data sources and scripts.

## Environment

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For LaTeX builds, use a TeX distribution with `pdflatex` and `bibtex` available. On this machine, the paper was built with MiKTeX.

## Build the Paper

```powershell
cd paper_springer
.\build_paper.ps1 -Clean
```

Expected output: `paper_springer/main.pdf`, 12 pages.

## Reproduce the Data Pipeline

The full data pipeline downloads public inputs and can take substantial time and disk space. The scripts use relative paths from the repository root.

Typical order:

```powershell
python .\code_pulldata\pull_nyc_311.py
python .\code_pulldata\pull_noaa_weather.py
python .\code_pulldata\pull_nyc_nta.py
python .\code_pulldata\pull_osm_nyc_pois.py
python .\code_pulldata\pull_nyc_pluto_landuse.py

python .\code_processdata\check_raw_data_schema.py
python .\code_processdata\join_311_to_nta.py
python .\code_processdata\aggregate_weekly_311_fixed_v2.py --overwrite
python .\code_processdata\build_311_temporal_features.py --overwrite
python .\code_processdata\build_weather_weekly_features.py --overwrite
python .\code_processdata\build_osm_poi_nta_features.py --overwrite
python .\code_processdata\build_pluto_nta_features.py --overwrite
python .\code_processdata\build_final_dataset.py --overwrite
python .\code_model\inspect_final_dataset.py --overwrite
```

The 311 download can optionally use `SOCRATA_APP_TOKEN` to reduce throttling.

## Reproduce Model Checks

The manuscript reports a target-week chronological evaluation. The final paper uses tuned LightGBM and XGBoost candidates trained on a stratified 300,000-row sample for local computational reproducibility.

Baseline checks:

```powershell
python .\code_model\train_baselines.py `
  --output-dir data/processed/model_results/baselines_target_week_sampled `
  --overwrite --progress
```

Tuned sampled LightGBM/XGBoost checks:

```powershell
python .\code_model\tune_gbm_hyperparams.py `
  --models lightgbm --preset balanced --max-train-rows 300000 `
  --output-dir data/processed/model_results/final_lgbm_tuned_model `
  --save-models --save-predictions-for-best `
  --overwrite --progress

python .\code_model\tune_gbm_hyperparams.py `
  --models xgboost --preset balanced --max-train-rows 300000 `
  --output-dir data/processed/model_results/final_xgb_tuned_model `
  --save-models --save-predictions-for-best `
  --overwrite --progress
```

Final ensemble/category thresholds:

```powershell
python .\code_model\ensemble_and_category_thresholds.py `
  --lgbm-tuning-dir data/processed/model_results/final_lgbm_tuned_model `
  --xgb-tuning-dir data/processed/model_results/final_xgb_tuned_model `
  --output-dir data/processed/model_results/ensemble_category_thresholds_target_week_sampled `
  --max-train-rows 300000 `
  --ensemble-weights 0.5 `
  --save-scores `
  --overwrite --progress
```

Paper result tables:

```powershell
python .\code_model\build_paper_result_tables.py `
  --ensemble-dir data/processed/model_results/ensemble_category_thresholds_target_week_sampled `
  --lgbm-tuning-dir data/processed/model_results/final_lgbm_tuned_model `
  --xgb-tuning-dir data/processed/model_results/final_xgb_tuned_model `
  --baselines-dir data/processed/model_results/baselines_target_week_sampled `
  --output-dir data/processed/model_results/paper_tables_target_week_sampled `
  --overwrite --progress
```

SHAP explanation:

```powershell
python .\code_model\run_shap_explainability.py `
  --lgbm-tuning-dir data/processed/model_results/final_lgbm_tuned_model `
  --ensemble-dir data/processed/model_results/ensemble_category_thresholds_target_week_sampled `
  --output-dir data/processed/model_results/shap_explainability_lgbm `
  --overwrite --progress
```

## Key Result Artifacts

- Final manuscript: `paper_springer/main.pdf`
- Final model comparison: `data/processed/model_results/paper_tables_target_week_sampled/paper_table_01_final_model_comparison.csv`
- Confusion matrix: `data/processed/model_results/paper_tables_target_week_sampled/paper_table_03_final_model_confusion_matrix.csv`
- Lift at K: `data/processed/model_results/paper_tables_target_week_sampled/paper_table_08_lift_at_k.csv`
- SHAP group importance: `data/processed/model_results/shap_explainability_lgbm/shap_group_importance.csv`
- Local SHAP cases: `data/processed/model_results/shap_explainability_lgbm/shap_local_cases.csv`

## Data Sources

The raw inputs are public data sources, subject to their own licensing and availability:

- NYC Open Data 311 Service Requests: https://opendata.cityofnewyork.us/
- NOAA GHCN-Daily weather data: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
- OpenStreetMap contributors: https://www.openstreetmap.org/copyright
- NYC Planning PLUTO/MapPLUTO: https://www.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page

## Full Data Access

Large raw files, the full processed modeling table, row-level scores, and model binaries are not stored in this Git repository. They can be regenerated from the public sources above using the provided scripts.

For questions or requests for the complete local data bundle, please contact:

Tran Dai Phong Lam, `phonglam2599@gmail.com`
