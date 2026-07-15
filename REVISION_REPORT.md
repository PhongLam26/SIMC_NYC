# Revision Report

Revision date: 2026-07-15

## Purpose

This revision converts the manuscript's final alert model to a prospective feature protocol. OSM and PLUTO variables from 2026 snapshots are no longer eligible for final tuning, ensemble selection, thresholds, SHAP explanations, or headline test metrics. They remain only in a clearly labelled retrospective context check.

## Files Changed

- Manuscript sources: `paper_overleaf/main.tex`, `paper_springer/main.tex`
- Table snippets: `paper_overleaf/tables/*.tex`
- New vector workflow figure: `paper_overleaf/figures/method_pipeline_overview.pdf`, `paper_springer/figures/method_pipeline_overview.pdf`
- Revised result figures: category operating points and prospective SHAP figures in both paper figure folders
- Prospective configs: `configs/features_prospective.json`, `configs/features_retrospective_*.json`
- Model config: `data/processed/model_ready/model_config.json`
- Pipeline scripts: tuning, ensemble/thresholding, table building, SHAP, extended checks, leakage audit, and figure generation scripts under `code_model/`
- Reports: `BEFORE_CONTEXT_FIX.md`, `prospective_leakage_audit.md`, `numerical_consistency_report.md`, `REVISION_REPORT.md`

## Build Command

```powershell
powershell -ExecutionPolicy Bypass -File .\paper_overleaf\build_paper.ps1 -Clean
powershell -ExecutionPolicy Bypass -File .\paper_springer\build_paper.ps1 -Clean
```

Clean build status:

- `paper_overleaf/main.pdf`: 12 pages, A4.
- `paper_springer/main.pdf`: 12 pages, A4.
- Final LaTeX logs contain no undefined citation/reference, overfull box, missing file, fatal error, or emergency stop messages.
- PDF text layer is extractable; the Overleaf PDF text layer contains 32,921 characters.
- PDF link audit: 83 annotations, 0 visible link color/border issues.

## Feature Protocol

- Final prospective feature set: `prospective_forecast_available`
- Input columns: 61
- Encoded model columns: 73
- Final model exclusions: all OSM/POI 2026 variables, all PLUTO 2026 variables, COVID-period labels, target labels, and future-window target construction columns
- Retrospective context checks:
  - `retrospective_context_osm_2026`
  - `retrospective_context_pluto_2026`
  - `retrospective_context_full_2026`

## New Prospective Model Runs

LightGBM and XGBoost were retuned on a 300,000-row stratified training sample and rebuilt on all 954,990 training rows for final ensemble scores.

- LightGBM: `lgbm_regularized_depth6_leaf31_lr003`, balanced positive weight 6.8726, selected threshold 0.5600.
- XGBoost: `xgb_shallow_depth5_lr003`, square-root positive weight 2.6216, selected threshold 0.3510.
- Final validation-selected operational strategy: ensemble LGBM(0.50)+XGB(0.50) with category-specific validation thresholds.

## Headline Metric Change

Previous full-context model with 2026 OSM/PLUTO inputs:

- Test F1 0.3839, precision 0.3071, recall 0.5117, PR-AUC 0.3365, ROC-AUC 0.7720, alert rate 0.2124.

Revised prospective model without OSM/PLUTO final inputs:

- Test F1 0.3802, precision 0.2933, recall 0.5404, PR-AUC 0.3310, ROC-AUC 0.7643, alert rate 0.2349.
- Confusion counts: TP 17,059; FP 41,106; FN 14,511; TN 174,914.

## Retrospective Context Check

Compact checks show that adding 2026 context variables changes test F1 only modestly relative to the prospective protocol:

- Prospective-only: 0.3767 F1 in the compact weather-enhanced check.
- Prospective + OSM 2026: 0.3818 F1.
- Prospective + PLUTO 2026: 0.3813 F1.
- Prospective + OSM 2026 + PLUTO 2026: 0.3819 F1.

These rows are labelled retrospective/context-only in the manuscript and are not used as headline results.

## Explainability

SHAP was regenerated using the prospective LightGBM feature set only. The audited SHAP feature list contains 73 encoded features and no OSM/PLUTO variables. Global group shares are:

- Historical/temporal: 75.15%
- Semantic category: 8.16%
- Calendar: 6.13%
- Weather: 5.87%
- Current demand: 2.74%
- Spatial context: 1.23%

## Audits

- `prospective_leakage_audit.md`: PASS, 10/10 checks.
- `numerical_consistency_report.md`: confirms confusion-matrix reconstruction, category sums, lift-at-k counts, and SHAP feature/group totals.
- `BEFORE_CONTEXT_FIX.md`: records the previous full-context baseline and why it was superseded.

## Reviewer-Facing Files

Reviewer-facing manuscript package:

- `paper_overleaf/main.tex`
- `paper_overleaf/references.bib`
- `paper_overleaf/figures/`
- `paper_overleaf/tables/`
- `paper_overleaf/main.pdf`
- `REVISION_REPORT.md`
- `prospective_leakage_audit.md`
- `numerical_consistency_report.md`
- `BEFORE_CONTEXT_FIX.md`

Internal/generated files that reviewers do not need unless reproducing experiments:

- `data/processed/model_results/prospective/`
- `data/processed/model_results/retrospective_context/`
- intermediate LaTeX logs and aux files
