# Before Context Fix Snapshot

Date recorded: 2026-07-15

This file preserves the final-model state before the prospective-context revision that removed 2026 OSM/PLUTO variables from the operational alert model.

## Previous Final-Model Scope

- Previous headline feature set: `full_without_covid_period_features`.
- Previous input count: 213 inputs, expanded to 229 model columns after encoding.
- Context variables included OSM/POI and PLUTO features extracted from 2026 snapshots.
- Previous final outputs lived under:
  - `data/processed/model_results/final_lgbm_tuned_model/`
  - `data/processed/model_results/final_xgb_tuned_model/`
  - `data/processed/model_results/ensemble_category_thresholds_target_week_fulltrain/`
  - `data/processed/model_results/paper_tables_target_week_fulltrain/`
  - `data/processed/model_results/shap_explainability_lgbm_fulltrain/`

## Previous Headline Test Metrics

The previous validation-selected per-category ensemble reported:

- Test F1: 0.3839
- Precision: 0.3071
- Recall: 0.5117
- PR-AUC: 0.3365
- ROC-AUC: 0.7720
- Balanced accuracy: 0.6715
- Alert rate: 0.2124
- Confusion counts: TP 16,154; FP 36,440; FN 15,416; TN 179,580

## Why It Was Revised

The paper describes a prospective 2024-2025 alert setting. Because OSM and PLUTO were extracted as 2026 static snapshots, using them in the final alert model creates a possible snapshot/look-ahead concern even when targets and temporal features are shifted correctly. The revised final model therefore uses only forecast-available features at week \(t\), while OSM/PLUTO are retained only in a separate retrospective context analysis.

## Revised Location

The replacement prospective outputs are under:

- `data/processed/model_results/prospective/`
- `data/processed/model_results/retrospective_context/compact_checks/`
- `prospective_leakage_audit.md`
- `numerical_consistency_report.md`

