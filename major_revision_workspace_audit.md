# Major Revision Workspace Audit

This audit maps the current SIMC NYC workspace before major methodological changes. It separates reproducible evidence already present from work still required by the reviewer checklist.

## Data Artifacts

| Area | Status | Path / note |
| --- | --- | --- |
| Raw 311 data | Present locally, ignored by git | `data/raw/nyc_311/*.csv.gz` |
| NTA metadata | Present locally | `data/raw/nyc_nta/nyc_nta_2020_metadata.csv` |
| Weather features | Present | `data/processed/feature_tables/noaa_weather_weekly_features.csv` |
| Temporal 311 features | Present locally, ignored by git | `data/processed/feature_tables/nyc_311_weekly_temporal_features.csv.gz` |
| Final dense panel | Present locally, ignored by git | `data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz` |
| Model config | Present | `data/processed/model_ready/model_config.json` |
| Final summaries | Present | `data/processed/_final_summaries/*` |
| Model inspection summaries | Present | `data/processed/_model_summaries/*` |
| OSM/PLUTO 2026 context | Present locally | Raw and feature tables exist but must be removed from the revised main paper/pipeline claims. |

## Current Dataset Shape

- Dense panel rows: 1,355,850
- Model-ready rows: 1,325,196
- Excluded rows: 30,654
- Unique NTAs: 262
- Unique weeks: 575
- Categories: 9
- Date range by feature week: 2014-12-29 to 2025-12-29
- Time split is assigned by target week, not feature week.

## Target and Feature Construction

| Item | Status | Evidence |
| --- | --- | --- |
| Binary next-week abnormal target | Present | `abnormal_increase_next_week` |
| Next-week count target | Present | `target_next_week_count` |
| 8-week target threshold artifact | Present | `abnormal_threshold_8w` |
| Target-history features | Present | `rolling_8w_mean`, `rolling_8w_std`, `ratio_to_8w_mean`, lags, rolling windows |
| Current target shortcut risk | Confirmed | Top SHAP features include `rolling_8w_std` and `ratio_to_8w_mean`. |
| No-shortcut feature config | Not yet created | Required for P1-1. |
| Sparse-aware target definitions | Not yet created | Required for P1-2. |

## Current Model Results

| Component | Status | Path / note |
| --- | --- | --- |
| Baseline rules/logistic | Present | `data/processed/model_results/baselines_target_week_sampled` |
| LGBM tuning | Present | `data/processed/model_results/prospective/tuning_lgbm` |
| XGB tuning | Present | `data/processed/model_results/prospective/tuning_xgb` |
| Final ensemble decision layer | Present | `data/processed/model_results/prospective/ensemble` |
| Row-level scored validation/test | Present locally, ignored by git | `data/processed/model_results/prospective/ensemble/scored_validation_test.csv.gz` |
| SHAP results | Present | `data/processed/model_results/prospective/shap` |
| Current seed stability | Partial | Only seeds 7, 42, 123 in `seed_stability_onehot`; requested seeds 2026, 3407, 7777 are missing. |
| Count-model baselines | Not found | Required by P1 checklist. |
| Rolling-origin folds | Not found | Required by P2/P3 checklist. |
| Calibration outputs | Not found | Required by checklist. |
| Bootstrap confidence intervals | Not found | Required by checklist. |

## Initial Sparse-Panel Audit

The script `scripts/major_revision_initial_audit.py` was added and run successfully. It reads the final dense panel, current scored validation/test file, and validation-selected category thresholds.

Outputs:

- `data/processed/model_results/major_revision/initial_audit/sparse_panel_diagnostics.csv`
- `data/processed/model_results/major_revision/initial_audit/sparse_panel_diagnostics.md`
- `data/processed/model_results/major_revision/initial_audit/performance_by_volume_decile.csv`
- `data/processed/model_results/major_revision/initial_audit/sparse_panel_category_profile.csv`
- `data/processed/model_results/major_revision/initial_audit/capacity_precision_at_k.csv`

Key current-target diagnostics:

- Test rows: 247,590
- Test positives: 31,570
- Positive test rows with `rolling_8w_mean < 1`: 5,670, or 17.96 percent
- Positive test rows with `rolling_8w_mean < 2`: 7,469, or 23.66 percent
- D1 volume decile F1: 0.1935; D10 volume decile F1: 0.4290
- Top 5 percent score precision: 0.4460

## Paper Artifacts

| Area | Status | Path / note |
| --- | --- | --- |
| Springer source | Present | `paper_springer/main.tex` |
| Overleaf source | Present | `paper_overleaf/main.tex` |
| Submission wrapper | Present | `paper_springer/main_SIMC_submission.tex` |
| Figures | Present | `paper_springer/figures`, `paper_overleaf/figures` |
| Tables | Present | `paper_overleaf/tables` and generated paper-table CSVs |
| Current submission audit | Present | `paper_springer/SUBMISSION_AUDIT.md` |
| Archived submitted PDFs | Present | `archive/submitted_2026-07-16/` |

## Reviewer-Risk Register

| Issue | Current status | Required action |
| --- | --- | --- |
| P1-1 target/input shortcut | Confirmed risk | Run no-shortcut variants and regenerate SHAP on final selected model. |
| P1-2 sparse/zero-heavy panel | Initial diagnostic complete | Evaluate sparse-aware target definitions and volume-stratified performance. |
| P1-3 title/category-aware/ensemble claim | Open | Decide title and model after validation evidence; avoid unsupported "best overall" language. |
| Uncertainty and seeds | Partial | Run requested five seeds and confidence intervals. |
| OSM/PLUTO look-ahead | Open for manuscript | Remove from main paper/pipeline; archive as retrospective-only if retained locally. |
| Count baselines | Missing | Implement Poisson/negative-binomial style baselines if feasible. |
| Rolling-origin validation | Missing | Implement expanding-window folds. |
| Calibration | Missing | Add Brier/ECE/reliability and calibration method if selected. |
| Main manuscript rewrite | Not started in this branch | Rewrite only after revised numerical evidence is complete. |

## Guardrails

- Do not claim a checklist item is PASS unless a script or artifact supports it.
- Do not use observed t+1 weather as a forecast.
- Do not add external socioeconomic, staffing, cost, forecast archive, or new data sources.
- Do not overwrite `paper_springer/main_SIMC_submission.pdf` until the revised manuscript is frozen.
