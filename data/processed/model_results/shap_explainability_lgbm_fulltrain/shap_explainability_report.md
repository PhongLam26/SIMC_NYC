# SHAP explainability report — Tuned LightGBM

This report explains the Tuned LightGBM component model used as the explanation backbone. The final decision system may use an ensemble and per-category thresholds; SHAP here should be interpreted as explaining a strong tree component of that decision system, not as causal evidence.

## Model explained

- Candidate: `006_E1_main_2015_2025_lightgbm_lgbm_regularized_depth6_leaf31_lr003_pw_sqrt`

- Parameter set: `lgbm_regularized_depth6_leaf31_lr003`

- Positive-class weight mode: `sqrt`

- Threshold used for local case selection: `0.345000`

- Test F1 of LightGBM component at this threshold: `0.3855`

- Test precision: `0.2999`

- Test recall: `0.5393`

- Test PR-AUC: `0.3346`


## Global SHAP importance

Top features by mean absolute SHAP value:

| importance_rank | feature | feature_group | mean_abs_shap | mean_abs_shap_share |
| --- | --- | --- | --- | --- |
| 1 | rolling_8w_std | historical_temporal | 0.8248 | 0.2365 |
| 2 | ratio_to_8w_mean | historical_temporal | 0.4054 | 0.1162 |
| 3 | week_of_year | calendar | 0.1356 | 0.0389 |
| 4 | lag_52w_count | historical_temporal | 0.1313 | 0.0377 |
| 5 | rolling_4w_mean | historical_temporal | 0.1093 | 0.0313 |
| 6 | lag_1w_count | historical_temporal | 0.1005 | 0.0288 |
| 7 | complaint_category=environment | semantic_category | 0.0866 | 0.0248 |
| 8 | rolling_12w_std | historical_temporal | 0.0826 | 0.0237 |
| 9 | complaint_category=other | semantic_category | 0.0735 | 0.0211 |
| 10 | complaint_category=housing | semantic_category | 0.0607 | 0.0174 |
| 11 | history_weeks_available | historical_temporal | 0.0588 | 0.0169 |
| 12 | complaint_count | current_demand | 0.0560 | 0.0161 |
| 13 | rolling_12w_mean | historical_temporal | 0.0556 | 0.0159 |
| 14 | lag_12w_count | historical_temporal | 0.0497 | 0.0143 |
| 15 | ratio_to_12w_mean | historical_temporal | 0.0453 | 0.0130 |
| 16 | poi_semantic_entropy | osm_poi_context | 0.0438 | 0.0126 |
| 17 | pluto_lot_count | pluto_built_environment | 0.0402 | 0.0115 |
| 18 | lag_2w_count | historical_temporal | 0.0388 | 0.0111 |
| 19 | complaint_category=noise | semantic_category | 0.0359 | 0.0103 |
| 20 | poi_named_count | osm_poi_context | 0.0342 | 0.0098 |

## Feature-group SHAP importance

| group_rank | feature_group | feature_count | mean_abs_shap_sum | mean_abs_shap_share |
| --- | --- | --- | --- | --- |
| 1 | historical_temporal | 21 | 2.0561 | 0.5896 |
| 2 | semantic_category | 9 | 0.3741 | 0.1073 |
| 3 | pluto_built_environment | 53 | 0.3551 | 0.1018 |
| 4 | osm_poi_context | 103 | 0.2837 | 0.0813 |
| 5 | weather | 22 | 0.1701 | 0.0488 |
| 6 | calendar | 4 | 0.1665 | 0.0477 |
| 7 | current_demand | 1 | 0.0560 | 0.0161 |
| 8 | other | 11 | 0.0184 | 0.0053 |
| 9 | spatial_context | 5 | 0.0076 | 0.0022 |

## Category-specific SHAP patterns

The file `shap_importance_by_complaint_category.csv` contains the top features for each complaint category. Use this table to support the semantics-aware interpretation that different service categories rely on different signals.

| complaint_category | category_rank | feature | feature_group | mean_abs_shap | rows |
| --- | --- | --- | --- | --- | --- |
| environment | 1 | rolling_8w_std | historical_temporal | 0.8441 | 2,777 |
| environment | 2 | ratio_to_8w_mean | historical_temporal | 0.4150 | 2,777 |
| environment | 3 | week_of_year | calendar | 0.1360 | 2,777 |
| environment | 4 | lag_52w_count | historical_temporal | 0.1317 | 2,777 |
| environment | 5 | rolling_4w_mean | historical_temporal | 0.1132 | 2,777 |
| housing | 1 | rolling_8w_std | historical_temporal | 0.8380 | 2,778 |
| housing | 2 | ratio_to_8w_mean | historical_temporal | 0.4108 | 2,778 |
| housing | 3 | week_of_year | calendar | 0.1354 | 2,778 |
| housing | 4 | lag_52w_count | historical_temporal | 0.1345 | 2,778 |
| housing | 5 | rolling_4w_mean | historical_temporal | 0.1112 | 2,778 |
| infrastructure | 1 | rolling_8w_std | historical_temporal | 0.8380 | 2,778 |
| infrastructure | 2 | ratio_to_8w_mean | historical_temporal | 0.4052 | 2,778 |
| infrastructure | 3 | week_of_year | calendar | 0.1352 | 2,778 |
| infrastructure | 4 | lag_52w_count | historical_temporal | 0.1291 | 2,778 |
| infrastructure | 5 | rolling_4w_mean | historical_temporal | 0.1066 | 2,778 |
| noise | 1 | rolling_8w_std | historical_temporal | 0.8084 | 2,778 |
| noise | 2 | ratio_to_8w_mean | historical_temporal | 0.3947 | 2,778 |
| noise | 3 | week_of_year | calendar | 0.1367 | 2,778 |
| noise | 4 | lag_52w_count | historical_temporal | 0.1277 | 2,778 |
| noise | 5 | rolling_4w_mean | historical_temporal | 0.1045 | 2,778 |
| other | 1 | rolling_8w_std | historical_temporal | 0.8432 | 2,778 |
| other | 2 | ratio_to_8w_mean | historical_temporal | 0.4092 | 2,778 |
| other | 3 | week_of_year | calendar | 0.1343 | 2,778 |
| other | 4 | lag_52w_count | historical_temporal | 0.1289 | 2,778 |
| other | 5 | rolling_4w_mean | historical_temporal | 0.1098 | 2,778 |
| parking_traffic | 1 | rolling_8w_std | historical_temporal | 0.8085 | 2,778 |
| parking_traffic | 2 | ratio_to_8w_mean | historical_temporal | 0.4003 | 2,778 |
| parking_traffic | 3 | week_of_year | calendar | 0.1336 | 2,778 |
| parking_traffic | 4 | lag_52w_count | historical_temporal | 0.1299 | 2,778 |
| parking_traffic | 5 | rolling_4w_mean | historical_temporal | 0.1121 | 2,778 |
| public_safety | 1 | rolling_8w_std | historical_temporal | 0.7965 | 2,778 |
| public_safety | 2 | ratio_to_8w_mean | historical_temporal | 0.4027 | 2,778 |
| public_safety | 3 | week_of_year | calendar | 0.1367 | 2,778 |
| public_safety | 4 | lag_52w_count | historical_temporal | 0.1327 | 2,778 |
| public_safety | 5 | rolling_4w_mean | historical_temporal | 0.1075 | 2,778 |
| sanitation | 1 | rolling_8w_std | historical_temporal | 0.8203 | 2,777 |
| sanitation | 2 | ratio_to_8w_mean | historical_temporal | 0.4090 | 2,777 |
| sanitation | 3 | week_of_year | calendar | 0.1348 | 2,777 |
| sanitation | 4 | lag_52w_count | historical_temporal | 0.1331 | 2,777 |
| sanitation | 5 | rolling_4w_mean | historical_temporal | 0.1088 | 2,777 |
| water_sewer | 1 | rolling_8w_std | historical_temporal | 0.8261 | 2,778 |
| water_sewer | 2 | ratio_to_8w_mean | historical_temporal | 0.4015 | 2,778 |
| water_sewer | 3 | week_of_year | calendar | 0.1373 | 2,778 |
| water_sewer | 4 | lag_52w_count | historical_temporal | 0.1345 | 2,778 |
| water_sewer | 5 | rolling_4w_mean | historical_temporal | 0.1103 | 2,778 |

## Local explanations

Local cases include true positives, false positives, false negatives, and true negatives selected from the test period. Use true positives as examples of correctly detected abnormal service-demand increases; use false positives/false negatives to discuss limitations.

| case_id | case_type | week_start | boroname | ntaname | complaint_category | y_true | prediction | lgbm_score | top_positive_features | top_negative_features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case_001 | true_positive_high_score | 2025-10-06 | Manhattan | Inwood | housing | 1 | 1 | 0.9541 | rolling_8w_std; week_of_year; pluto_total_residential_area; pluto_total_units; month | complaint_category=other; ratio_to_8w_mean; rolling_4w_mean; history_weeks_available; lag_1w_count |
| case_002 | true_positive_high_score | 2025-06-23 | Bronx | Riverdale-Spuyten Duyvil | water_sewer | 1 | 1 | 0.9424 | rolling_8w_std; complaint_category=other; lag_12w_count; poi_semantic_entropy; rolling_4w_std | ratio_to_8w_mean; complaint_category=environment; month; lag_52w_count; complaint_category=housing |
| case_003 | true_positive_high_score | 2025-06-23 | Bronx | Mount Eden-Claremont (West) | water_sewer | 1 | 1 | 0.9393 | rolling_8w_std; week_of_year; poi_semantic_entropy; poi_named_count; pluto_parcel_density_per_km2 | complaint_category=other; ratio_to_8w_mean; rolling_4w_mean; lag_1w_count; lag_12w_count |
| case_004 | true_positive_high_score | 2025-06-23 | Bronx | Bedford Park | water_sewer | 1 | 1 | 0.9349 | ratio_to_8w_mean; week_of_year; complaint_category=infrastructure; diff_4w_count; complaint_category=environment | rolling_8w_std; lag_2w_count; lag_1w_count; complaint_category=noise; rolling_4w_mean |
| case_005 | true_positive_high_score | 2025-06-23 | Bronx | Soundview-Bruckner-Bronx River | water_sewer | 1 | 1 | 0.9316 | complaint_category=housing; lag_1w_count; lag_52w_count; rolling_4w_mean; complaint_count | rolling_8w_std; ratio_to_8w_mean; lag_8w_count; weather_prcp_max; history_weeks_available |
| case_006 | false_positive_high_score | 2025-06-30 | Manhattan | Lower East Side | public_safety | 0 | 1 | 0.9404 | lag_52w_count; lag_1w_count; complaint_count; lag_12w_count; rolling_4w_mean | rolling_8w_std; ratio_to_8w_mean; weather_prcp_max; history_weeks_available; complaint_category=housing |
| case_007 | false_positive_high_score | 2025-06-30 | Manhattan | Washington Heights (North) | public_safety | 0 | 1 | 0.9308 | ratio_to_8w_mean; complaint_count; lag_1w_count; lag_12w_count; ratio_to_12w_mean | rolling_8w_std; month; week_of_year; weather_tmin_mean; weather_tmax_max |
| case_008 | false_positive_high_score | 2025-06-30 | Staten Island | Great Kills-Eltingville | public_safety | 0 | 1 | 0.9154 | ratio_to_8w_mean; week_of_year; lag_12w_count; rolling_8w_std; complaint_category=other | complaint_category=environment; lag_52w_count; lag_1w_count; rolling_4w_mean; weather_tmax_max |
| case_009 | false_positive_high_score | 2025-06-30 | Manhattan | Washington Heights (South) | public_safety | 0 | 1 | 0.9108 | complaint_category=infrastructure; rolling_8w_std; week_of_year; complaint_count; complaint_category=other | ratio_to_8w_mean; history_weeks_available; lag_1w_count; rolling_4w_mean; lag_2w_count |
| case_010 | false_positive_high_score | 2025-06-30 | Brooklyn | Bedford-Stuyvesant (East) | public_safety | 0 | 1 | 0.9051 | rolling_8w_std; complaint_category=parking_traffic; complaint_category=other; rolling_12w_std; complaint_category=environment | ratio_to_8w_mean; lag_52w_count; rolling_4w_mean; poi_semantic_entropy; pluto_total_units |
| case_011 | false_negative_low_score | 2025-10-20 | Brooklyn | Spring Creek-Starrett City | parking_traffic | 1 | 0 | 0.3450 | ratio_to_8w_mean; lag_52w_count; rolling_8w_std; complaint_category=water_sewer; complaint_category=environment | week_of_year; rolling_4w_mean; lag_1w_count; complaint_category=housing; lag_12w_count |
| case_012 | false_negative_low_score | 2024-05-13 | Manhattan | Morningside Heights | sanitation | 1 | 0 | 0.3450 | rolling_8w_std; week_of_year; complaint_category=other; rolling_12w_mean; month | ratio_to_8w_mean; complaint_category=public_safety; lag_1w_count; ratio_to_12w_mean; complaint_category=noise |
| case_013 | false_negative_low_score | 2024-10-07 | Queens | Forest Hills | public_safety | 1 | 0 | 0.3450 | lag_52w_count; lag_12w_count; complaint_category=environment; complaint_count; weather_prcp_max | rolling_8w_std; ratio_to_8w_mean; week_of_year; complaint_category=housing; lag_1w_count |
| case_014 | false_negative_low_score | 2025-01-06 | Queens | Queens Village | water_sewer | 1 | 0 | 0.3450 | rolling_8w_std; complaint_category=parking_traffic; week_of_year; rolling_12w_std; complaint_category=other | ratio_to_8w_mean; rolling_4w_mean; lag_1w_count; rolling_8w_mean; lag_2w_count |
| case_015 | false_negative_low_score | 2025-07-21 | Queens | Sunnyside Yards (North) | water_sewer | 1 | 0 | 0.3450 | rolling_8w_std; weather_prcp_max; week_of_year; rolling_12w_std; rolling_8w_mean | ratio_to_8w_mean; complaint_category=other; rolling_4w_mean; lag_1w_count; lag_52w_count |
| case_016 | true_negative_low_score | 2024-12-02 | Staten Island | Hoffman & Swinburne Islands | environment | 0 | 0 | 0.0007 | complaint_category=water_sewer; rolling_8w_std; complaint_category=environment; complaint_category=other; rolling_12w_mean | ratio_to_8w_mean; week_of_year; lag_1w_count; rolling_4w_mean; complaint_category=housing |
| case_017 | true_negative_low_score | 2025-12-01 | Staten Island | Hoffman & Swinburne Islands | environment | 0 | 0 | 0.0008 | rolling_8w_std; rolling_12w_mean; complaint_category=other; rolling_12w_sum; ratio_to_8w_mean | complaint_category=environment; week_of_year; rolling_4w_mean; lag_1w_count; complaint_category=housing |
| case_018 | true_negative_low_score | 2024-12-09 | Staten Island | Hoffman & Swinburne Islands | environment | 0 | 0 | 0.0008 | complaint_category=water_sewer; complaint_category=environment; cold_day_0c_count; complaint_category=other; rolling_12w_mean | ratio_to_8w_mean; week_of_year; lag_1w_count; lag_52w_count; history_weeks_available |
| case_019 | true_negative_low_score | 2025-12-15 | Staten Island | Hoffman & Swinburne Islands | environment | 0 | 0 | 0.0008 | rolling_8w_std; complaint_category=other; rolling_12w_mean; rolling_12w_sum; weather_awnd_mean | week_of_year; complaint_category=environment; rolling_4w_mean; lag_1w_count; complaint_category=housing |
| case_020 | true_negative_low_score | 2025-12-22 | Staten Island | Hoffman & Swinburne Islands | environment | 0 | 0 | 0.0008 | rolling_8w_std; rolling_12w_mean; complaint_category=other; pluto_total_units; pluto_total_storage_area | complaint_category=environment; week_of_year; ratio_to_8w_mean; rolling_4w_mean; lag_1w_count |

## Paper wording suggestions

Suggested wording: `Global SHAP analysis indicates that the model relies primarily on historical demand dynamics, seasonal/calendar signals, and selected contextual variables when estimating next-week abnormal service-demand risk. Category-specific SHAP summaries show that the relative importance of these factors varies across municipal service domains, supporting the semantics-aware formulation of the forecasting task.`


Careful wording: `SHAP values explain how features contribute to model predictions; they should not be interpreted as causal effects of weather, POI density, or land-use conditions on actual urban problems.`


## Output files

- global_importance: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\shap_explainability_lgbm_fulltrain\shap_global_importance.csv`
- group_importance: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\shap_explainability_lgbm_fulltrain\shap_group_importance.csv`
- category_importance: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\shap_explainability_lgbm_fulltrain\shap_importance_by_complaint_category.csv`
- local_cases: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\shap_explainability_lgbm_fulltrain\shap_local_cases.csv`
- local_contributions: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\shap_explainability_lgbm_fulltrain\shap_local_case_feature_contributions.csv`
- summary_bar: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\shap_explainability_lgbm_fulltrain\shap_summary_bar.png`
- group_bar: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\shap_explainability_lgbm_fulltrain\shap_group_importance.png`
- beeswarm: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\shap_explainability_lgbm_fulltrain\shap_beeswarm.png`
- report: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\shap_explainability_lgbm_fulltrain\shap_explainability_report.md`
