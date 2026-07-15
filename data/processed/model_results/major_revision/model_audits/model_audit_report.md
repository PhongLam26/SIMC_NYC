# Target Shortcut and Count Baseline Audit

This report is generated from actual model runs for the major revision. It is still an audit artifact, not the final model-selection report.

## LightGBM Target-Shortcut Configurations

| feature_set | raw_feature_count | model_feature_count | threshold | precision | recall | f1 | pr_auc | roc_auc | precision_at_5pct | delta_vs_current_f1 | delta_vs_current_pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_current_prospective | 61 | 73 | 0.5800 | 0.2986 | 0.5222 | 0.3800 | 0.3301 | 0.7639 | 0.4451 | 0.0000 | 0.0000 |
| B_no_8w_formula_features | 57 | 69 | 0.5556 | 0.2717 | 0.5293 | 0.3591 | 0.3097 | 0.7445 | 0.4220 | -0.0209 | -0.0204 |
| C_reduced_nonformula_history | 51 | 63 | 0.5421 | 0.2664 | 0.5558 | 0.3601 | 0.3065 | 0.7452 | 0.4219 | -0.0198 | -0.0236 |

Formula-aligned features removed in the no-shortcut configuration: `rolling_8w_mean`, `rolling_8w_std`, `rolling_8w_sum`, and `ratio_to_8w_mean`.

## Top Gain Importances

| feature_set | importance_rank | feature | gain_importance |
| --- | --- | --- | --- |
| A_current_prospective | 1 | rolling_8w_std | 2503 |
| A_current_prospective | 2 | history_weeks_available | 1355 |
| A_current_prospective | 3 | week_of_year | 1180 |
| A_current_prospective | 4 | lag_52w_count | 919 |
| A_current_prospective | 5 | ratio_to_8w_mean | 843 |
| A_current_prospective | 6 | rolling_12w_mean | 723 |
| A_current_prospective | 7 | weather_tmax_max | 627 |
| A_current_prospective | 8 | weather_prcp_max | 592 |
| A_current_prospective | 9 | weather_awnd_mean | 575 |
| A_current_prospective | 10 | ratio_to_12w_mean | 570 |
| A_current_prospective | 11 | lag_1w_count | 536 |
| A_current_prospective | 12 | weather_prcp_mean | 535 |
| B_no_8w_formula_features | 1 | rolling_12w_std | 1507 |
| B_no_8w_formula_features | 2 | history_weeks_available | 1444 |
| B_no_8w_formula_features | 3 | lag_8w_count | 1317 |
| B_no_8w_formula_features | 4 | rolling_4w_std | 1225 |
| B_no_8w_formula_features | 5 | week_of_year | 1174 |
| B_no_8w_formula_features | 6 | lag_12w_count | 1095 |
| B_no_8w_formula_features | 7 | lag_52w_count | 997 |
| B_no_8w_formula_features | 8 | ratio_to_12w_mean | 916 |
| B_no_8w_formula_features | 9 | lag_1w_count | 606 |
| B_no_8w_formula_features | 10 | weather_prcp_max | 594 |
| B_no_8w_formula_features | 11 | rolling_12w_mean | 585 |
| B_no_8w_formula_features | 12 | weather_awnd_mean | 581 |
| C_reduced_nonformula_history | 1 | complaint_count | 2070 |
| C_reduced_nonformula_history | 2 | rolling_12w_std | 1536 |
| C_reduced_nonformula_history | 3 | history_weeks_available | 1455 |
| C_reduced_nonformula_history | 4 | lag_8w_count | 1437 |
| C_reduced_nonformula_history | 5 | week_of_year | 1051 |
| C_reduced_nonformula_history | 6 | rolling_4w_std | 1022 |
| C_reduced_nonformula_history | 7 | lag_12w_count | 1005 |
| C_reduced_nonformula_history | 8 | lag_52w_count | 973 |
| C_reduced_nonformula_history | 9 | rolling_12w_mean | 942 |
| C_reduced_nonformula_history | 10 | rolling_4w_mean | 638 |
| C_reduced_nonformula_history | 11 | lag_4w_count | 633 |
| C_reduced_nonformula_history | 12 | weather_awnd_max | 604 |

## SHAP Global Importance Files

- `A_current_prospective`: `data\processed\model_results\major_revision\model_audits\shap_global_A_current_prospective.csv`, sample rows = existing
- `B_no_8w_formula_features`: `data\processed\model_results\major_revision\model_audits\shap_global_B_no_8w_formula_features.csv`, sample rows = existing
- `C_reduced_nonformula_history`: `data\processed\model_results\major_revision\model_audits\shap_global_C_reduced_nonformula_history.csv`, sample rows = existing

## Count Baselines

| model_name | decision_mode | feature_count | count_mae | poisson_deviance | mean_observed_count | mean_predicted_count | precision | recall | f1 | pr_auc | precision_at_5pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| poisson_regressor_no_nta | formula_threshold | 18 | 8.6680 | 6.4720 | 28.3029 | 27.6028 | 0.1156 | 0.1980 | 0.1460 | 0.1400 | 0.1444 |
| poisson_regressor_no_nta | validation_score_threshold | 18 | 8.6680 | 6.4720 | 28.3029 | 27.6028 | 0.1388 | 0.9577 | 0.2424 | 0.1400 | 0.1444 |
| poisson_regressor_nta_fe | formula_threshold | 19 | 8.6839 | 6.4590 | 28.3029 | 27.9629 | 0.1175 | 0.2044 | 0.1492 | 0.1400 | 0.1445 |
| poisson_regressor_nta_fe | validation_score_threshold | 19 | 8.6839 | 6.4590 | 28.3029 | 27.9629 | 0.1388 | 0.9576 | 0.2424 | 0.1400 | 0.1445 |

## Guardrails

- These runs do not use OSM/PLUTO features.
- The no-shortcut model removes direct 8-week target-formula predictors but still predicts the current submitted target.
- Count baselines predict `target_next_week_count`; event metrics are reported both with the original abnormal-threshold conversion and a validation-selected score threshold.
- Final target/model selection still requires rolling-origin validation, uncertainty intervals, and target-definition sensitivity.
