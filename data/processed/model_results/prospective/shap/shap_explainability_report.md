# SHAP explainability report — Tuned LightGBM

This report explains the Tuned LightGBM component model used as the explanation backbone. The final decision system may use an ensemble and per-category thresholds; SHAP here should be interpreted as explaining a strong tree component of that decision system, not as causal evidence.

## Model explained

- Candidate: `008_E1_main_2015_2025_lightgbm_lgbm_regularized_depth6_leaf31_lr003_pw_balanced`

- Parameter set: `lgbm_regularized_depth6_leaf31_lr003`

- Positive-class weight mode: `balanced`

- Threshold used for local case selection: `0.560000`

- Test F1 of LightGBM component at this threshold: `0.3789`

- Test precision: `0.2855`

- Test recall: `0.5634`

- Test PR-AUC: `0.3298`


## Global SHAP importance

Top features by mean absolute SHAP value:

| importance_rank | feature | feature_group | mean_abs_shap | mean_abs_shap_share |
| --- | --- | --- | --- | --- |
| 1 | rolling_8w_std | historical_temporal | 0.7957 | 0.2662 |
| 2 | ratio_to_8w_mean | historical_temporal | 0.4499 | 0.1505 |
| 3 | lag_52w_count | historical_temporal | 0.1732 | 0.0579 |
| 4 | week_of_year | calendar | 0.1527 | 0.0511 |
| 5 | rolling_12w_mean | historical_temporal | 0.1274 | 0.0426 |
| 6 | rolling_12w_std | historical_temporal | 0.1164 | 0.0389 |
| 7 | rolling_4w_mean | historical_temporal | 0.1145 | 0.0383 |
| 8 | lag_1w_count | historical_temporal | 0.1072 | 0.0359 |
| 9 | complaint_count | current_demand | 0.0819 | 0.0274 |
| 10 | lag_12w_count | historical_temporal | 0.0688 | 0.0230 |
| 11 | complaint_category=environment | semantic_category | 0.0509 | 0.0170 |
| 12 | complaint_category=housing | semantic_category | 0.0495 | 0.0166 |
| 13 | history_weeks_available | historical_temporal | 0.0486 | 0.0163 |
| 14 | lag_2w_count | historical_temporal | 0.0446 | 0.0149 |
| 15 | lag_8w_count | historical_temporal | 0.0360 | 0.0120 |
| 16 | complaint_category=public_safety | semantic_category | 0.0343 | 0.0115 |
| 17 | complaint_category=noise | semantic_category | 0.0303 | 0.0101 |
| 18 | ratio_to_12w_mean | historical_temporal | 0.0301 | 0.0101 |
| 19 | diff_4w_count | historical_temporal | 0.0299 | 0.0100 |
| 20 | month | calendar | 0.0269 | 0.0090 |

## Feature-group SHAP importance

| group_rank | feature_group | feature_count | mean_abs_shap_sum | mean_abs_shap_share |
| --- | --- | --- | --- | --- |
| 1 | historical_temporal | 21 | 2.2462 | 0.7515 |
| 2 | semantic_category | 9 | 0.2440 | 0.0816 |
| 3 | calendar | 4 | 0.1833 | 0.0613 |
| 4 | weather | 22 | 0.1754 | 0.0587 |
| 5 | current_demand | 1 | 0.0819 | 0.0274 |
| 6 | spatial_context | 5 | 0.0367 | 0.0123 |
| 7 | other | 11 | 0.0216 | 0.0072 |

## Category-specific SHAP patterns

The file `shap_importance_by_complaint_category.csv` contains the top features for each complaint category. Use this table to support the semantics-aware interpretation that different service categories rely on different signals.

| complaint_category | category_rank | feature | feature_group | mean_abs_shap | rows |
| --- | --- | --- | --- | --- | --- |
| environment | 1 | rolling_8w_std | historical_temporal | 0.8113 | 2,777 |
| environment | 2 | ratio_to_8w_mean | historical_temporal | 0.4612 | 2,777 |
| environment | 3 | lag_52w_count | historical_temporal | 0.1747 | 2,777 |
| environment | 4 | week_of_year | calendar | 0.1529 | 2,777 |
| environment | 5 | rolling_12w_mean | historical_temporal | 0.1286 | 2,777 |
| housing | 1 | rolling_8w_std | historical_temporal | 0.8095 | 2,778 |
| housing | 2 | ratio_to_8w_mean | historical_temporal | 0.4578 | 2,778 |
| housing | 3 | lag_52w_count | historical_temporal | 0.1761 | 2,778 |
| housing | 4 | week_of_year | calendar | 0.1533 | 2,778 |
| housing | 5 | rolling_12w_mean | historical_temporal | 0.1287 | 2,778 |
| infrastructure | 1 | rolling_8w_std | historical_temporal | 0.8145 | 2,778 |
| infrastructure | 2 | ratio_to_8w_mean | historical_temporal | 0.4498 | 2,778 |
| infrastructure | 3 | lag_52w_count | historical_temporal | 0.1687 | 2,778 |
| infrastructure | 4 | week_of_year | calendar | 0.1527 | 2,778 |
| infrastructure | 5 | rolling_12w_mean | historical_temporal | 0.1264 | 2,778 |
| noise | 1 | rolling_8w_std | historical_temporal | 0.7797 | 2,778 |
| noise | 2 | ratio_to_8w_mean | historical_temporal | 0.4387 | 2,778 |
| noise | 3 | lag_52w_count | historical_temporal | 0.1692 | 2,778 |
| noise | 4 | week_of_year | calendar | 0.1538 | 2,778 |
| noise | 5 | rolling_12w_mean | historical_temporal | 0.1278 | 2,778 |
| other | 1 | rolling_8w_std | historical_temporal | 0.8163 | 2,778 |
| other | 2 | ratio_to_8w_mean | historical_temporal | 0.4535 | 2,778 |
| other | 3 | lag_52w_count | historical_temporal | 0.1682 | 2,778 |
| other | 4 | week_of_year | calendar | 0.1516 | 2,778 |
| other | 5 | rolling_12w_mean | historical_temporal | 0.1256 | 2,778 |
| parking_traffic | 1 | rolling_8w_std | historical_temporal | 0.7807 | 2,778 |
| parking_traffic | 2 | ratio_to_8w_mean | historical_temporal | 0.4433 | 2,778 |
| parking_traffic | 3 | lag_52w_count | historical_temporal | 0.1710 | 2,778 |
| parking_traffic | 4 | week_of_year | calendar | 0.1496 | 2,778 |
| parking_traffic | 5 | rolling_12w_mean | historical_temporal | 0.1242 | 2,778 |
| public_safety | 1 | rolling_8w_std | historical_temporal | 0.7672 | 2,778 |
| public_safety | 2 | ratio_to_8w_mean | historical_temporal | 0.4450 | 2,778 |
| public_safety | 3 | lag_52w_count | historical_temporal | 0.1758 | 2,778 |
| public_safety | 4 | week_of_year | calendar | 0.1540 | 2,778 |
| public_safety | 5 | rolling_12w_mean | historical_temporal | 0.1288 | 2,778 |
| sanitation | 1 | rolling_8w_std | historical_temporal | 0.7884 | 2,777 |
| sanitation | 2 | ratio_to_8w_mean | historical_temporal | 0.4543 | 2,777 |
| sanitation | 3 | lag_52w_count | historical_temporal | 0.1762 | 2,777 |
| sanitation | 4 | week_of_year | calendar | 0.1519 | 2,777 |
| sanitation | 5 | rolling_12w_mean | historical_temporal | 0.1285 | 2,777 |
| water_sewer | 1 | rolling_8w_std | historical_temporal | 0.7934 | 2,778 |
| water_sewer | 2 | ratio_to_8w_mean | historical_temporal | 0.4455 | 2,778 |
| water_sewer | 3 | lag_52w_count | historical_temporal | 0.1788 | 2,778 |
| water_sewer | 4 | week_of_year | calendar | 0.1549 | 2,778 |
| water_sewer | 5 | rolling_12w_mean | historical_temporal | 0.1283 | 2,778 |

## Local explanations

Local cases include true positives, false positives, false negatives, and true negatives selected from the test period. Use true positives as examples of correctly detected abnormal service-demand increases; use false positives/false negatives to discuss limitations.

| case_id | case_type | week_start | boroname | ntaname | complaint_category | y_true | prediction | lgbm_score | top_positive_features | top_negative_features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case_001 | true_positive_high_score | 2025-06-23 | Bronx | Mount Hope | water_sewer | 1 | 1 | 0.9803 | complaint_category=housing; ratio_to_8w_mean; lag_52w_count; week_of_year; lag_1w_count | rolling_8w_std; weather_prcp_max; history_weeks_available; weather_prcp_station_max; lag_8w_count |
| case_002 | true_positive_high_score | 2025-06-23 | Bronx | Highbridge | water_sewer | 1 | 1 | 0.9801 | week_of_year; rolling_12w_mean; lag_12w_count; month; rolling_8w_std | complaint_category=infrastructure; weather_tmin_max; complaint_category=noise; diff_4w_count; lag_1w_count |
| case_003 | true_positive_high_score | 2025-06-23 | Bronx | Mount Eden-Claremont (West) | water_sewer | 1 | 1 | 0.9799 | rolling_8w_std; week_of_year; rolling_12w_std; ratio_to_12w_mean; boroname=Brooklyn | ratio_to_8w_mean; complaint_category=other; lag_12w_count; rolling_4w_mean; history_weeks_available |
| case_004 | true_positive_high_score | 2025-06-23 | Bronx | University Heights (North)-Fordham | water_sewer | 1 | 1 | 0.9785 | lag_1w_count; lag_52w_count; rolling_4w_mean; complaint_category=housing; rolling_12w_mean | rolling_8w_std; ratio_to_8w_mean; lag_8w_count; week_of_year; weather_prcp_max |
| case_005 | true_positive_high_score | 2025-06-23 | Bronx | Riverdale-Spuyten Duyvil | water_sewer | 1 | 1 | 0.9775 | rolling_8w_std; rolling_12w_mean; lag_12w_count; complaint_count; rolling_12w_std | ratio_to_8w_mean; complaint_category=environment; month; week_of_year; complaint_category=housing |
| case_006 | false_positive_high_score | 2025-06-30 | Manhattan | Lower East Side | public_safety | 0 | 1 | 0.9711 | lag_52w_count; lag_1w_count; lag_12w_count; complaint_count; rolling_12w_mean | rolling_8w_std; ratio_to_8w_mean; history_weeks_available; lag_8w_count; weather_prcp_max |
| case_007 | false_positive_high_score | 2024-04-08 | Bronx | Mott Haven-Port Morris | infrastructure | 0 | 1 | 0.9510 | ratio_to_8w_mean; rolling_8w_std; week_of_year; complaint_category=noise; rolling_12w_mean | weather_tmin_min; lag_52w_count; weather_tmax_min; ratio_to_12w_mean; complaint_category=water_sewer |
| case_008 | false_positive_high_score | 2025-06-30 | Manhattan | Washington Heights (North) | public_safety | 0 | 1 | 0.9507 | ratio_to_8w_mean; lag_1w_count; complaint_count; lag_12w_count; rolling_12w_mean | rolling_8w_std; month; week_of_year; rolling_4w_std; rolling_8w_mean |
| case_009 | false_positive_high_score | 2024-05-13 | Manhattan | Chelsea-Hudson Yards | noise | 0 | 1 | 0.9503 | week_of_year; complaint_category=water_sewer; weather_tmax_mean; weather_awnd_mean; pct_change_1w | rolling_8w_std; ratio_to_8w_mean; lag_1w_count; complaint_category=housing; rolling_4w_mean |
| case_010 | false_positive_high_score | 2025-06-30 | Manhattan | Washington Heights (South) | public_safety | 0 | 1 | 0.9468 | rolling_8w_std; week_of_year; complaint_category=infrastructure; complaint_count; weather_prcp_mean | ratio_to_8w_mean; rolling_4w_mean; history_weeks_available; lag_1w_count; complaint_category=housing |
| case_011 | false_negative_low_score | 2025-07-07 | Manhattan | Lower East Side | infrastructure | 1 | 0 | 0.5600 | lag_52w_count; rolling_12w_mean; week_of_year; complaint_count; lag_12w_count | rolling_8w_std; ratio_to_8w_mean; lag_8w_count; complaint_category=parking_traffic; rolling_12w_std |
| case_012 | false_negative_low_score | 2024-04-22 | Brooklyn | Brighton Beach | infrastructure | 1 | 0 | 0.5600 | ratio_to_8w_mean; lag_52w_count; complaint_count; diff_4w_count; lag_1w_count | rolling_8w_std; weather_tmax_min; complaint_category=public_safety; rolling_4w_std; complaint_category=water_sewer |
| case_013 | false_negative_low_score | 2024-05-06 | Queens | Woodhaven | sanitation | 1 | 0 | 0.5599 | ratio_to_8w_mean; week_of_year; lag_12w_count; lag_52w_count; rolling_12w_mean | rolling_8w_std; complaint_category=noise; month; complaint_category=public_safety; history_weeks_available |
| case_014 | false_negative_low_score | 2024-02-19 | Brooklyn | Brighton Beach | environment | 1 | 0 | 0.5599 | lag_52w_count; week_of_year; rolling_12w_mean; lag_12w_count; complaint_count | rolling_8w_std; ratio_to_8w_mean; complaint_category=public_safety; complaint_category=noise; ratio_to_12w_mean |
| case_015 | false_negative_low_score | 2025-06-02 | Brooklyn | Brooklyn Heights | infrastructure | 1 | 0 | 0.5599 | rolling_8w_std; week_of_year; rolling_12w_std; lag_12w_count; complaint_count | ratio_to_8w_mean; lag_52w_count; complaint_category=other; rolling_4w_mean; lag_1w_count |
| case_016 | true_negative_low_score | 2025-09-08 | Queens | Elmhurst | other | 0 | 0 | 0.0019 | ratio_to_8w_mean; rolling_8w_std; rolling_12w_mean; lag_12w_count; lag_52w_count | week_of_year; history_weeks_available; complaint_category=housing; diff_4w_count; complaint_category=water_sewer |
| case_017 | true_negative_low_score | 2025-08-18 | Bronx | Riverdale-Spuyten Duyvil | water_sewer | 0 | 0 | 0.0030 | ratio_to_8w_mean; rolling_12w_mean; lag_12w_count; complaint_category=public_safety; boroname=Brooklyn | complaint_category=environment; week_of_year; lag_52w_count; rolling_8w_std; rolling_4w_mean |
| case_018 | true_negative_low_score | 2024-03-11 | Manhattan | Gramercy | parking_traffic | 0 | 0 | 0.0036 | rolling_8w_std; lag_52w_count; week_of_year; rolling_12w_mean; ratio_to_8w_mean | complaint_category=noise; ratio_to_12w_mean; diff_4w_count; lag_1w_count; weather_tavg_mean |
| case_019 | true_negative_low_score | 2025-08-04 | Staten Island | Arden Heights-Rossville | public_safety | 0 | 0 | 0.0040 | lag_52w_count; rain_day_count; lag_1w_count; complaint_count; complaint_category=housing | rolling_8w_std; ratio_to_8w_mean; lag_8w_count; week_of_year; diff_4w_count |
| case_020 | true_negative_low_score | 2025-04-07 | Queens | Bayside | water_sewer | 0 | 0 | 0.0041 | rolling_8w_std; complaint_category=noise; history_weeks_available; rolling_12w_std; complaint_count | ratio_to_8w_mean; rolling_4w_mean; lag_12w_count; lag_52w_count; lag_2w_count |

## Paper wording suggestions

Suggested wording: `Global SHAP analysis indicates that the model relies primarily on historical demand dynamics, seasonal/calendar signals, and selected contextual variables when estimating next-week abnormal service-demand risk. Category-specific SHAP summaries show that the relative importance of these factors varies across municipal service domains, supporting the semantics-aware formulation of the forecasting task.`


Careful wording: `SHAP values explain how features contribute to model predictions; they should not be interpreted as causal effects of weather, POI density, or land-use conditions on actual urban problems.`


## Output files

- global_importance: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\prospective\shap\shap_global_importance.csv`
- group_importance: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\prospective\shap\shap_group_importance.csv`
- category_importance: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\prospective\shap\shap_importance_by_complaint_category.csv`
- local_cases: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\prospective\shap\shap_local_cases.csv`
- local_contributions: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\prospective\shap\shap_local_case_feature_contributions.csv`
- summary_bar: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\prospective\shap\shap_summary_bar.png`
- group_bar: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\prospective\shap\shap_group_importance.png`
- beeswarm: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\prospective\shap\shap_beeswarm.png`
- report: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\prospective\shap\shap_explainability_report.md`
