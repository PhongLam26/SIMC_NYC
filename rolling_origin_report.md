# Rolling-Origin Backtest Report

This report runs expanding-window backtests using the existing data only. All rows use the no-shortcut LightGBM feature set, with formula-aligned 8-week predictors removed.

Thresholds are selected on the validation year within each fold and applied unchanged to that fold's test year. No test year is used to select thresholds, target definitions, or feature sets in this script.

## Fold Design

| fold_id | train_end_year | validation_year | test_year |
| --- | --- | --- | --- |
| fold_2021 | 2019 | 2020 | 2021 |
| fold_2022 | 2020 | 2021 | 2022 |
| fold_2023 | 2021 | 2022 | 2023 |
| fold_2024 | 2022 | 2023 | 2024 |
| final_style_2025 | 2023 | 2024 | 2025 |

## Test-Year Summary Across Folds

| target_definition | folds | pr_auc_mean | pr_auc_std | precision_at_5pct_mean | precision_at_5pct_std | f1_mean | f1_std | positive_share_mean | positive_share_std | threshold_mean | threshold_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0_current_reference | 5 | 0.2959 | 0.0153 | 0.4062 | 0.0219 | 0.3520 | 0.0093 | 0.1284 | 0.0032 | 0.5625 | 0.0481 |
| T1_min_count_2 | 5 | 0.2991 | 0.0183 | 0.4060 | 0.0251 | 0.3550 | 0.0133 | 0.1170 | 0.0025 | 0.6025 | 0.0441 |
| T2_min_count_3 | 5 | 0.2997 | 0.0174 | 0.4015 | 0.0234 | 0.3583 | 0.0138 | 0.1103 | 0.0022 | 0.6105 | 0.0521 |
| T3_mu8w_ge_1_eligible | 5 | 0.3064 | 0.0185 | 0.4199 | 0.0261 | 0.3631 | 0.0119 | 0.1372 | 0.0031 | 0.5472 | 0.0619 |

## Per-Test-Year Metrics

| target_definition | fold_id | test_year | rows | positive_share | pr_auc | precision_at_1pct | precision_at_5pct | f1 | precision | recall | threshold | alert_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0_current_reference | fold_2021 | 2021 | 122616 | 0.1338 | 0.3062 | 0.5281 | 0.4301 | 0.3633 | 0.2879 | 0.4924 | 0.5613 | 0.2289 |
| T0_current_reference | fold_2022 | 2022 | 122616 | 0.1259 | 0.2789 | 0.4711 | 0.3825 | 0.3432 | 0.2557 | 0.5219 | 0.6336 | 0.2570 |
| T0_current_reference | fold_2023 | 2023 | 122616 | 0.1270 | 0.2802 | 0.4808 | 0.3846 | 0.3417 | 0.2590 | 0.5019 | 0.5700 | 0.2462 |
| T0_current_reference | fold_2024 | 2024 | 124974 | 0.1269 | 0.3023 | 0.5320 | 0.4100 | 0.3574 | 0.2680 | 0.5363 | 0.5477 | 0.2539 |
| T0_current_reference | final_style_2025 | 2025 | 122616 | 0.1281 | 0.3120 | 0.5884 | 0.4236 | 0.3544 | 0.2576 | 0.5675 | 0.5000 | 0.2823 |
| T1_min_count_2 | fold_2021 | 2021 | 122616 | 0.1213 | 0.3132 | 0.5273 | 0.4371 | 0.3677 | 0.3038 | 0.4656 | 0.6000 | 0.1858 |
| T1_min_count_2 | fold_2022 | 2022 | 122616 | 0.1150 | 0.2820 | 0.4621 | 0.3843 | 0.3497 | 0.2634 | 0.5200 | 0.6650 | 0.2270 |
| T1_min_count_2 | fold_2023 | 2023 | 122616 | 0.1155 | 0.2772 | 0.4711 | 0.3773 | 0.3343 | 0.2726 | 0.4322 | 0.6200 | 0.1831 |
| T1_min_count_2 | fold_2024 | 2024 | 124974 | 0.1160 | 0.3056 | 0.5408 | 0.4095 | 0.3617 | 0.2775 | 0.5195 | 0.5799 | 0.2171 |
| T1_min_count_2 | final_style_2025 | 2025 | 122616 | 0.1172 | 0.3173 | 0.6015 | 0.4216 | 0.3615 | 0.2676 | 0.5568 | 0.5474 | 0.2438 |
| T2_min_count_3 | fold_2021 | 2021 | 122616 | 0.1140 | 0.3121 | 0.5126 | 0.4273 | 0.3751 | 0.2976 | 0.5072 | 0.6000 | 0.1943 |
| T2_min_count_3 | fold_2022 | 2022 | 122616 | 0.1086 | 0.2814 | 0.4694 | 0.3733 | 0.3488 | 0.2610 | 0.5254 | 0.6900 | 0.2187 |
| T2_min_count_3 | fold_2023 | 2023 | 122616 | 0.1086 | 0.2805 | 0.4662 | 0.3809 | 0.3404 | 0.2679 | 0.4665 | 0.6172 | 0.1890 |
| T2_min_count_3 | fold_2024 | 2024 | 124974 | 0.1097 | 0.3082 | 0.5352 | 0.4077 | 0.3657 | 0.2823 | 0.5193 | 0.6000 | 0.2019 |
| T2_min_count_3 | final_style_2025 | 2025 | 122616 | 0.1106 | 0.3165 | 0.5697 | 0.4180 | 0.3613 | 0.2635 | 0.5744 | 0.5452 | 0.2411 |
| T3_mu8w_ge_1_eligible | fold_2021 | 2021 | 92458 | 0.1426 | 0.3172 | 0.5232 | 0.4443 | 0.3755 | 0.2951 | 0.5159 | 0.5500 | 0.2492 |
| T3_mu8w_ge_1_eligible | fold_2022 | 2022 | 94130 | 0.1348 | 0.2854 | 0.4820 | 0.3864 | 0.3524 | 0.2766 | 0.4854 | 0.6500 | 0.2364 |
| T3_mu8w_ge_1_eligible | fold_2023 | 2023 | 92803 | 0.1352 | 0.2878 | 0.4995 | 0.3986 | 0.3485 | 0.2675 | 0.4998 | 0.5267 | 0.2527 |
| T3_mu8w_ge_1_eligible | fold_2024 | 2024 | 95444 | 0.1364 | 0.3152 | 0.5288 | 0.4276 | 0.3704 | 0.2763 | 0.5618 | 0.5233 | 0.2773 |
| T3_mu8w_ge_1_eligible | final_style_2025 | 2025 | 93979 | 0.1371 | 0.3261 | 0.6213 | 0.4424 | 0.3685 | 0.2716 | 0.5729 | 0.4860 | 0.2891 |

## Validation-Year Metrics

| target_definition | fold_id | validation_year | rows | positive_share | pr_auc | precision_at_5pct | f1 | threshold | alert_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0_current_reference | fold_2021 | 2020 | 122616 | 0.1462 | 0.3690 | 0.5151 | 0.4060 | 0.5613 | 0.2300 |
| T0_current_reference | fold_2022 | 2021 | 122616 | 0.1338 | 0.3086 | 0.4319 | 0.3591 | 0.6336 | 0.2500 |
| T0_current_reference | fold_2023 | 2022 | 122616 | 0.1259 | 0.2786 | 0.3892 | 0.3358 | 0.5700 | 0.2482 |
| T0_current_reference | fold_2024 | 2023 | 122616 | 0.1270 | 0.2936 | 0.3993 | 0.3511 | 0.5477 | 0.2700 |
| T0_current_reference | final_style_2025 | 2024 | 124974 | 0.1269 | 0.2881 | 0.3977 | 0.3523 | 0.5000 | 0.2689 |
| T1_min_count_2 | fold_2021 | 2020 | 122616 | 0.1327 | 0.3835 | 0.5213 | 0.4146 | 0.6000 | 0.1875 |
| T1_min_count_2 | fold_2022 | 2021 | 122616 | 0.1213 | 0.3075 | 0.4182 | 0.3636 | 0.6650 | 0.2200 |
| T1_min_count_2 | fold_2023 | 2022 | 122616 | 0.1150 | 0.2845 | 0.3877 | 0.3388 | 0.6200 | 0.1944 |
| T1_min_count_2 | fold_2024 | 2023 | 122616 | 0.1155 | 0.2956 | 0.3963 | 0.3504 | 0.5799 | 0.2300 |
| T1_min_count_2 | final_style_2025 | 2024 | 124974 | 0.1160 | 0.2934 | 0.3948 | 0.3581 | 0.5474 | 0.2300 |
| T2_min_count_3 | fold_2021 | 2020 | 122616 | 0.1250 | 0.3929 | 0.5213 | 0.4244 | 0.6000 | 0.1954 |
| T2_min_count_3 | fold_2022 | 2021 | 122616 | 0.1140 | 0.3133 | 0.4130 | 0.3666 | 0.6900 | 0.2133 |
| T2_min_count_3 | fold_2023 | 2022 | 122616 | 0.1086 | 0.2828 | 0.3851 | 0.3401 | 0.6172 | 0.2000 |
| T2_min_count_3 | fold_2024 | 2023 | 122616 | 0.1086 | 0.3006 | 0.4003 | 0.3547 | 0.6000 | 0.2135 |
| T2_min_count_3 | final_style_2025 | 2024 | 124974 | 0.1097 | 0.2939 | 0.3906 | 0.3611 | 0.5452 | 0.2300 |
| T3_mu8w_ge_1_eligible | fold_2021 | 2020 | 89786 | 0.1613 | 0.4056 | 0.5661 | 0.4337 | 0.5500 | 0.2563 |
| T3_mu8w_ge_1_eligible | fold_2022 | 2021 | 92458 | 0.1426 | 0.3116 | 0.4285 | 0.3686 | 0.6500 | 0.2394 |
| T3_mu8w_ge_1_eligible | fold_2023 | 2022 | 94130 | 0.1348 | 0.2943 | 0.4164 | 0.3492 | 0.5267 | 0.2700 |
| T3_mu8w_ge_1_eligible | fold_2024 | 2023 | 92803 | 0.1352 | 0.3079 | 0.4204 | 0.3606 | 0.5233 | 0.3000 |
| T3_mu8w_ge_1_eligible | final_style_2025 | 2024 | 95444 | 0.1364 | 0.3012 | 0.4081 | 0.3666 | 0.4860 | 0.2700 |

## Interpretation Guardrails

- T3 is a restricted-risk-set target and is not directly comparable to T0/T1/T2 because it excludes low-baseline rows.
- This pass does not include multiple seeds, calibration, bootstrap intervals, or COVID-exclusion sensitivity.
- Final model and target selection remain open until paired uncertainty and calibration evidence are added.
