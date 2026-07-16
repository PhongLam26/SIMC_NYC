# Final-Style Explainability Report

This pass explains the actual single LightGBM score model used for the current `T2_min_count_3` no-shortcut candidate on the final-style 2025 fold. The explanation model and score model are the same fitted LightGBM; no ensemble explanation proxy is used.

## Model and Decision Context

| validation_selected_threshold | test_pr_auc | test_f1 | test_precision | test_recall | raw_feature_count | model_feature_count | fit_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5452 | 0.3165 | 0.3613 | 0.2635 | 0.5744 | 57 | 69 | 20.1046 |

Formula-aligned 8-week target-construction predictors are removed: `rolling_8w_mean`, `rolling_8w_std`, `rolling_8w_sum`, and `ratio_to_8w_mean`.

## SHAP Group Importance

| group_rank | feature_group | feature_count | mean_abs_shap_sum | mean_abs_shap_share |
| --- | --- | --- | --- | --- |
| 1 | shifted_history | 17 | 2.4619 | 0.7259 |
| 2 | service_category | 9 | 0.2474 | 0.0730 |
| 3 | current_count | 1 | 0.2443 | 0.0720 |
| 4 | feature_week_weather | 30 | 0.2136 | 0.0630 |
| 5 | calendar | 7 | 0.1878 | 0.0554 |
| 6 | borough | 5 | 0.0363 | 0.0107 |

## Top Features

| importance_rank | feature | feature_group | mean_abs_shap | mean_abs_shap_share |
| --- | --- | --- | --- | --- |
| 1 | rolling_12w_mean | shifted_history | 0.4476 | 0.1320 |
| 2 | ratio_to_12w_mean | shifted_history | 0.4237 | 0.1249 |
| 3 | rolling_4w_std | shifted_history | 0.2635 | 0.0777 |
| 4 | complaint_count | current_count | 0.2443 | 0.0720 |
| 5 | rolling_12w_std | shifted_history | 0.2284 | 0.0673 |
| 6 | lag_8w_count | shifted_history | 0.1951 | 0.0575 |
| 7 | lag_52w_count | shifted_history | 0.1833 | 0.0540 |
| 8 | lag_12w_count | shifted_history | 0.1620 | 0.0478 |
| 9 | history_weeks_available | shifted_history | 0.1523 | 0.0449 |
| 10 | week_of_year | calendar | 0.1499 | 0.0442 |
| 11 | diff_4w_count | shifted_history | 0.1165 | 0.0344 |
| 12 | lag_1w_count | shifted_history | 0.0979 | 0.0289 |
| 13 | rolling_4w_mean | shifted_history | 0.0717 | 0.0211 |
| 14 | complaint_category_noise | service_category | 0.0599 | 0.0177 |
| 15 | complaint_category_housing | service_category | 0.0461 | 0.0136 |
| 16 | complaint_category_environment | service_category | 0.0408 | 0.0120 |
| 17 | complaint_category_water_sewer | service_category | 0.0281 | 0.0083 |
| 18 | lag_2w_count | shifted_history | 0.0281 | 0.0083 |
| 19 | month | calendar | 0.0268 | 0.0079 |
| 20 | diff_1w_count | shifted_history | 0.0264 | 0.0078 |
| 21 | complaint_category_sanitation | service_category | 0.0248 | 0.0073 |
| 22 | rolling_12w_sum | shifted_history | 0.0216 | 0.0064 |
| 23 | weather_prcp_max | feature_week_weather | 0.0208 | 0.0061 |
| 24 | weather_awnd_mean | feature_week_weather | 0.0205 | 0.0060 |
| 25 | pct_change_1w | shifted_history | 0.0199 | 0.0059 |

## Local Cases

| case_id | case_type | target_week | boroname | ntaname | complaint_category | y_true | prediction | score | target_next_week_count | z_exceedance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tp | tp_high_confidence | 2025-10-20 | Bronx | Parkchester | housing | 1 | 1 | 0.9754 | 244.0000 | 11.8455 |
| fp | fp_high_confidence | 2025-03-17 | Brooklyn | Mapleton-Midwood (West) | noise | 0 | 1 | 0.9563 | 12.0000 | -0.1465 |
| fn | fn_severe_missed | 2025-06-30 | Queens | Elmhurst | other | 1 | 0 | 0.3165 | 426.0000 | 85.7384 |

## Output Figures

- `shap_beeswarm.pdf`
- `shap_local_tp.pdf`
- `shap_local_fp.pdf`
- `shap_local_fn.pdf`

## Guardrails

- SHAP values explain fitted LightGBM score contributions, not causal effects.
- This pass supports the current candidate model; final manuscript claims still need final target/model freeze.
- Calibration is monotonic and fitted on validation, so these SHAP values explain the underlying score ranking used by the calibrated decision layer.
