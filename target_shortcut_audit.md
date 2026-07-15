# Target Shortcut Audit

This audit was run for the major methodological revision on branch `major-revision-methodological-rebuild`.

Generated detailed artifacts:

- `data/processed/model_results/major_revision/model_audits/target_shortcut_results.csv`
- `data/processed/model_results/major_revision/model_audits/target_shortcut_feature_importance.csv`
- `data/processed/model_results/major_revision/model_audits/shap_global_A_current_prospective.csv`
- `data/processed/model_results/major_revision/model_audits/shap_global_B_no_8w_formula_features.csv`
- `data/processed/model_results/major_revision/model_audits/shap_global_C_reduced_nonformula_history.csv`

## Configurations

- `A_current_prospective`: current prospective LightGBM feature set.
- `B_no_8w_formula_features`: removes `rolling_8w_mean`, `rolling_8w_std`, `rolling_8w_sum`, and `ratio_to_8w_mean`.
- `C_reduced_nonformula_history`: also removes several short-term derived momentum/ratio features, while retaining lags, non-8-week rolling summaries, calendar, weather, category, and borough.

All runs use the submitted target definition and full available training rows for the current 2015-2022 train, 2023 validation, 2024-2025 test split. Thresholds are selected on validation.

## Held-Out Test Results

| Configuration | Test F1 | Test PR-AUC | Precision | Recall | Precision@5% | Delta F1 vs current | Delta PR-AUC vs current |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A current prospective | 0.3800 | 0.3301 | 0.2986 | 0.5222 | 0.4451 | 0.0000 | 0.0000 |
| B no 8-week formula features | 0.3591 | 0.3097 | 0.2717 | 0.5293 | 0.4220 | -0.0209 | -0.0204 |
| C reduced non-formula history | 0.3601 | 0.3065 | 0.2664 | 0.5558 | 0.4219 | -0.0198 | -0.0236 |

## Importance Shift

In the current model, formula-aligned 8-week predictors are prominent:

- Gain rank 1: `rolling_8w_std`
- Gain rank 5: `ratio_to_8w_mean`
- Gain rank 36: `rolling_8w_mean`

After removing those predictors, the leading SHAP features shift to other temporal signals:

- SHAP rank 1: `ratio_to_12w_mean`
- SHAP rank 2: `rolling_4w_std`
- SHAP rank 3: `rolling_12w_std`
- SHAP rank 4: `lag_8w_count`
- SHAP rank 5: `lag_12w_count`

## Interpretation

The performance drop after removing formula-aligned 8-week predictors is material enough that the submitted explanation should not be defended as-is. These results support the reviewer critique that the current target and feature design creates construct-circularity risk.

This audit is not yet the final target/model-selection evidence. It still needs rolling-origin validation, target-definition sensitivity, uncertainty intervals, and final SHAP for the selected model.
