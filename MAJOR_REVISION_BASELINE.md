# Major Revision Baseline

Created on 2026-07-16 before starting the methodological rebuild requested in `Danh_sach_can_sua_SIMC_NYC.docx` and the attached pasted checklist.

## Git Checkpoint

- Baseline branch before changes: `main`
- Revision branch: `major-revision-methodological-rebuild`
- Baseline commit: `7ccf58dd55dd2c24d6f0d5a2a3adcfa8c1bf0a6a`
- Baseline tag: `pre-major-revision-submission-2026-07-16`
- Remote: `https://github.com/PhongLam26/SIMC_NYC.git`

## Protected Submission Artifacts

- Current submission PDF: `paper_springer/main_SIMC_submission.pdf`
- Archived copy: `archive/submitted_2026-07-16/main_SIMC_submission.pdf`
- Current Springer PDF: `paper_springer/main.pdf`
- Archived copy: `archive/submitted_2026-07-16/main_springer.pdf`
- Current Overleaf PDF: `paper_overleaf/main.pdf`
- Archived copy: `archive/submitted_2026-07-16/main_overleaf.pdf`

No new build of `main_SIMC_submission.pdf` was run during this checkpoint.

## Current Manuscript State

- Current title: `Category-Aware Explainable Machine Learning for Urban Service Demand Forecasting in Smart Cities`
- Primary paper sources: `paper_springer/main.tex`, `paper_overleaf/main.tex`
- Submission wrapper: `paper_springer/main_SIMC_submission.tex`
- Bibliography: `paper_springer/references.bib`, `paper_overleaf/references.bib`
- Current page count from prior PDF audit: 12 A4 pages for `main.pdf` and `main_SIMC_submission.pdf`

## Current Data and Target

- Dense panel file: `data/processed/final/final_nyc_urban_service_demand_dataset.csv.gz`
- Output rows: 1,355,850
- Model-ready rows: 1,325,196
- Excluded rows relative to dense panel: 30,654
- Target column: `abnormal_increase_next_week`
- Target count column: `target_next_week_count`
- Target split policy: split by `target_week = week_start + 7 days`
- Current split rows: train 983,286; validation 122,616; test 247,590; bad_date 2,358
- Current train rows used in final rebuilt ensemble: 954,990

## Current Feature Set

Configured final feature set:

- Config: `configs/features_prospective.json`
- Feature set: `prospective_forecast_available`
- Analysis type: `prospective`
- Feature count: 61 input features before one-hot encoding
- Included groups: 311 history, service category, borough, calendar timing, feature-week weather
- Excluded from final prospective feature set: OSM/PLUTO 2026 and COVID period indicators

Important current reviewer risk:

- The current feature set includes `rolling_8w_mean`, `rolling_8w_std`, and `ratio_to_8w_mean`.
- The target is also defined relative to an 8-week rolling baseline.
- SHAP currently ranks `rolling_8w_std` and `ratio_to_8w_mean` as the top two features, so reviewer P1-1 is a real methodological risk.

## Current Model and Decision Layer

- Current validation-selected strategy: `ensemble_lgbm_w0p500_per_category`
- Score: average of LightGBM and XGBoost scores with 0.50/0.50 weight
- Thresholding: validation-selected per-category thresholds
- LightGBM best candidate: `008_E1_main_2015_2025_lightgbm_lgbm_regularized_depth6_leaf31_lr003_pw_balanced`
- XGBoost best candidate: `010_E1_main_2015_2025_xgboost_xgb_shallow_depth5_lr003_pw_sqrt`
- LightGBM/XGBoost full final rebuild train rows: 954,990 each

Held-out 2024-2025 current test metrics:

- F1: 0.3802
- Precision: 0.2933
- Recall: 0.5404
- PR-AUC: 0.3310
- ROC-AUC: 0.7643
- Balanced accuracy: 0.6750
- Alert rate: 0.2349
- Confusion matrix: TP 17,059; FP 41,106; FN 14,511; TN 174,914

Capacity ranking under current scores:

- Top 1 percent precision: 0.5905
- Top 5 percent precision: 0.4460
- Top 10 percent precision: 0.3809
- Category-threshold alert precision: 0.2933

## Current Explainability Artifacts

- Current SHAP folder: `data/processed/model_results/prospective/shap`
- Current paper SHAP figures: `paper_springer/figures/shap_*`, `paper_overleaf/figures/shap_*`
- Current top SHAP features include `rolling_8w_std` and `ratio_to_8w_mean`, which must be treated as evidence for the target-shortcut audit rather than as final scientific explanation.

## Initial Major-Revision Audit Outputs

Created by `scripts/major_revision_initial_audit.py`:

- `data/processed/model_results/major_revision/initial_audit/sparse_panel_diagnostics.csv`
- `data/processed/model_results/major_revision/initial_audit/sparse_panel_diagnostics.md`
- `data/processed/model_results/major_revision/initial_audit/performance_by_volume_decile.csv`
- `data/processed/model_results/major_revision/initial_audit/sparse_panel_category_profile.csv`
- `data/processed/model_results/major_revision/initial_audit/capacity_precision_at_k.csv`

Initial sparse-panel finding for the current submitted target:

- Test positives with `rolling_8w_mean < 0.25`: 2,560, or 8.11 percent of test positives
- Test positives with `rolling_8w_mean < 0.5`: 4,073, or 12.90 percent
- Test positives with `rolling_8w_mean < 1`: 5,670, or 17.96 percent
- Test positives with `rolling_8w_mean < 2`: 7,469, or 23.66 percent
- Positive rows with `target_next_week_count >= 4`: 26,154, or 82.84 percent

These diagnostics do not resolve reviewer P1-1/P1-2. They establish the current baseline that the major revision must improve or replace.

## Required Next Experiments

- Target-shortcut audit with no-shortcut feature variants.
- Sparse-aware target alternatives and construct-validity selection.
- Count-model baselines.
- Full-training ablations with no OSM/PLUTO in the main pipeline.
- Five-seed stability using the requested seeds 42, 123, 2026, 3407, 7777.
- Rolling-origin evaluation.
- Calibration, bootstrap CIs, borough/volume/category diagnostics, error analysis, and rewritten manuscript.
