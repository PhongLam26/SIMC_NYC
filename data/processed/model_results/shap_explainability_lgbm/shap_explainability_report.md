# SHAP explainability report — Tuned LightGBM

This report explains the Tuned LightGBM component model used as the explanation backbone. The final decision system may use an ensemble and per-category thresholds; SHAP here should be interpreted as explaining a strong tree component of that decision system, not as causal evidence.

## Model explained

- Candidate: `013_E1_main_2015_2025_lightgbm_lgbm_deeper_depth10_leaf127_lr003_pw_none`

- Parameter set: `lgbm_deeper_depth10_leaf127_lr003`

- Positive-class weight mode: `none`

- Threshold used for local case selection: `0.182273`

- Test F1 of LightGBM component at this threshold: `0.3828`

- Test precision: `0.3049`

- Test recall: `0.5144`

- Test PR-AUC: `0.3353`


## Global SHAP importance

Top features by mean absolute SHAP value:

| importance_rank | feature | feature_group | mean_abs_shap | mean_abs_shap_share |
| --- | --- | --- | --- | --- |
| 1 | rolling_8w_std | historical_temporal | 0.7507 | 0.2148 |
| 2 | ratio_to_8w_mean | historical_temporal | 0.4335 | 0.1240 |
| 3 | complaint_category | semantic_category | 0.2101 | 0.0601 |
| 4 | lag_52w_count | historical_temporal | 0.1279 | 0.0366 |
| 5 | week_of_year | calendar | 0.1230 | 0.0352 |
| 6 | rolling_4w_mean | historical_temporal | 0.1131 | 0.0324 |
| 7 | rolling_12w_std | historical_temporal | 0.0854 | 0.0244 |
| 8 | complaint_count | current_demand | 0.0742 | 0.0212 |
| 9 | lag_1w_count | historical_temporal | 0.0735 | 0.0210 |
| 10 | history_weeks_available | historical_temporal | 0.0703 | 0.0201 |
| 11 | lag_12w_count | historical_temporal | 0.0531 | 0.0152 |
| 12 | rolling_12w_mean | historical_temporal | 0.0509 | 0.0146 |
| 13 | poi_semantic_entropy | osm_poi_context | 0.0465 | 0.0133 |
| 14 | lag_8w_count | historical_temporal | 0.0438 | 0.0125 |
| 15 | rolling_8w_mean | historical_temporal | 0.0423 | 0.0121 |
| 16 | poi_named_count | osm_poi_context | 0.0402 | 0.0115 |
| 17 | lag_2w_count | historical_temporal | 0.0385 | 0.0110 |
| 18 | pluto_lot_count | pluto_built_environment | 0.0385 | 0.0110 |
| 19 | pluto_total_buildings | pluto_built_environment | 0.0353 | 0.0101 |
| 20 | ratio_to_12w_mean | historical_temporal | 0.0347 | 0.0099 |

## Feature-group SHAP importance

| group_rank | feature_group | feature_count | mean_abs_shap_sum | mean_abs_shap_share |
| --- | --- | --- | --- | --- |
| 1 | historical_temporal | 21 | 2.0557 | 0.5882 |
| 2 | pluto_built_environment | 53 | 0.4015 | 0.1149 |
| 3 | osm_poi_context | 99 | 0.3526 | 0.1009 |
| 4 | weather | 22 | 0.2118 | 0.0606 |
| 5 | semantic_category | 1 | 0.2101 | 0.0601 |
| 6 | calendar | 4 | 0.1606 | 0.0460 |
| 7 | current_demand | 1 | 0.0742 | 0.0212 |
| 8 | other | 11 | 0.0241 | 0.0069 |
| 9 | spatial_context | 1 | 0.0042 | 0.0012 |

## Category-specific SHAP patterns

The file `shap_importance_by_complaint_category.csv` contains the top features for each complaint category. Use this table to support the semantics-aware interpretation that different service categories rely on different signals.

| complaint_category | category_rank | feature | feature_group | mean_abs_shap | rows |
| --- | --- | --- | --- | --- | --- |
| environment | 1 | rolling_8w_std | historical_temporal | 0.7639 | 2,778 |
| environment | 2 | ratio_to_8w_mean | historical_temporal | 0.4328 | 2,778 |
| environment | 3 | complaint_category | semantic_category | 0.2167 | 2,778 |
| environment | 4 | lag_52w_count | historical_temporal | 0.1325 | 2,778 |
| environment | 5 | week_of_year | calendar | 0.1221 | 2,778 |
| housing | 1 | rolling_8w_std | historical_temporal | 0.7666 | 2,778 |
| housing | 2 | ratio_to_8w_mean | historical_temporal | 0.4292 | 2,778 |
| housing | 3 | complaint_category | semantic_category | 0.2138 | 2,778 |
| housing | 4 | lag_52w_count | historical_temporal | 0.1295 | 2,778 |
| housing | 5 | week_of_year | calendar | 0.1269 | 2,778 |
| infrastructure | 1 | rolling_8w_std | historical_temporal | 0.7638 | 2,778 |
| infrastructure | 2 | ratio_to_8w_mean | historical_temporal | 0.4415 | 2,778 |
| infrastructure | 3 | complaint_category | semantic_category | 0.2093 | 2,778 |
| infrastructure | 4 | lag_52w_count | historical_temporal | 0.1314 | 2,778 |
| infrastructure | 5 | week_of_year | calendar | 0.1246 | 2,778 |
| noise | 1 | rolling_8w_std | historical_temporal | 0.7535 | 2,778 |
| noise | 2 | ratio_to_8w_mean | historical_temporal | 0.4340 | 2,778 |
| noise | 3 | complaint_category | semantic_category | 0.2146 | 2,778 |
| noise | 4 | lag_52w_count | historical_temporal | 0.1270 | 2,778 |
| noise | 5 | week_of_year | calendar | 0.1223 | 2,778 |
| other | 1 | rolling_8w_std | historical_temporal | 0.7458 | 2,777 |
| other | 2 | ratio_to_8w_mean | historical_temporal | 0.4341 | 2,777 |
| other | 3 | complaint_category | semantic_category | 0.2078 | 2,777 |
| other | 4 | lag_52w_count | historical_temporal | 0.1273 | 2,777 |
| other | 5 | week_of_year | calendar | 0.1195 | 2,777 |
| parking_traffic | 1 | rolling_8w_std | historical_temporal | 0.7299 | 2,778 |
| parking_traffic | 2 | ratio_to_8w_mean | historical_temporal | 0.4345 | 2,778 |
| parking_traffic | 3 | complaint_category | semantic_category | 0.2104 | 2,778 |
| parking_traffic | 4 | lag_52w_count | historical_temporal | 0.1286 | 2,778 |
| parking_traffic | 5 | week_of_year | calendar | 0.1228 | 2,778 |
| public_safety | 1 | rolling_8w_std | historical_temporal | 0.7426 | 2,777 |
| public_safety | 2 | ratio_to_8w_mean | historical_temporal | 0.4345 | 2,777 |
| public_safety | 3 | complaint_category | semantic_category | 0.2028 | 2,777 |
| public_safety | 4 | lag_52w_count | historical_temporal | 0.1234 | 2,777 |
| public_safety | 5 | week_of_year | calendar | 0.1233 | 2,777 |
| sanitation | 1 | rolling_8w_std | historical_temporal | 0.7505 | 2,778 |
| sanitation | 2 | ratio_to_8w_mean | historical_temporal | 0.4303 | 2,778 |
| sanitation | 3 | complaint_category | semantic_category | 0.2054 | 2,778 |
| sanitation | 4 | lag_52w_count | historical_temporal | 0.1251 | 2,778 |
| sanitation | 5 | week_of_year | calendar | 0.1236 | 2,778 |
| water_sewer | 1 | rolling_8w_std | historical_temporal | 0.7394 | 2,777 |
| water_sewer | 2 | ratio_to_8w_mean | historical_temporal | 0.4304 | 2,777 |
| water_sewer | 3 | complaint_category | semantic_category | 0.2097 | 2,777 |
| water_sewer | 4 | lag_52w_count | historical_temporal | 0.1266 | 2,777 |
| water_sewer | 5 | week_of_year | calendar | 0.1219 | 2,777 |

## Local explanations

Local cases include true positives, false positives, false negatives, and true negatives selected from the test period. Use true positives as examples of correctly detected abnormal service-demand increases; use false positives/false negatives to discuss limitations.

| case_id | case_type | week_start | boroname | ntaname | complaint_category | y_true | prediction | lgbm_score | top_positive_features | top_negative_features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case_001 | true_positive_high_score | 2025-10-13 | Manhattan | Hamilton Heights-Sugar Hill | housing | 1 | 1 | 0.9033 | history_weeks_available; lag_52w_count; pluto_lot_count_residential; pluto_lot_count; pluto_parcel_density_per_km2 | ratio_to_8w_mean; complaint_category; week_of_year; rolling_8w_std; lag_1w_count |
| case_002 | true_positive_high_score | 2025-10-13 | Bronx | Parkchester | housing | 1 | 1 | 0.8995 | rolling_8w_std; rolling_8w_mean; rolling_12w_std; week_of_year; lag_8w_count | complaint_category; ratio_to_8w_mean; rolling_4w_mean; lag_52w_count; weather_tmin_mean |
| case_003 | true_positive_high_score | 2025-10-13 | Bronx | Pelham Parkway-Van Nest | housing | 1 | 1 | 0.8992 | complaint_category; week_of_year; rolling_8w_std; month; weather_awnd_mean | ratio_to_8w_mean; lag_1w_count; rolling_4w_mean; pct_change_1w; lag_12w_count |
| case_004 | true_positive_high_score | 2025-10-06 | Manhattan | Hamilton Heights-Sugar Hill | housing | 1 | 1 | 0.8923 | pluto_parcel_density_per_km2; pluto_lot_count_residential; ratio_to_12w_mean; history_weeks_available; poi_named_count | ratio_to_8w_mean; complaint_category; week_of_year; rolling_8w_std; weather_tmax_max |
| case_005 | true_positive_high_score | 2025-10-13 | Bronx | University Heights (North)-Fordham | housing | 1 | 1 | 0.8837 | lag_12w_count; lag_52w_count; complaint_count; ratio_to_8w_mean; lag_1w_count | rolling_8w_std; history_weeks_available; complaint_category; lag_8w_count; month |
| case_006 | false_positive_high_score | 2025-10-13 | Bronx | Morrisania | housing | 0 | 1 | 0.8335 | complaint_category; ratio_to_8w_mean; lag_1w_count; rolling_4w_mean; complaint_count | rolling_8w_std; lag_52w_count; rolling_8w_mean; rolling_4w_std; weather_tmax_max |
| case_007 | false_positive_high_score | 2025-10-13 | Bronx | Belmont | housing | 0 | 1 | 0.8260 | lag_52w_count; complaint_count; lag_1w_count; lag_12w_count; rolling_4w_mean | rolling_8w_std; lag_8w_count; ratio_to_8w_mean; rolling_4w_std; rolling_8w_mean |
| case_008 | false_positive_high_score | 2025-12-01 | Manhattan | Manhattanville-West Harlem | housing | 0 | 1 | 0.8248 | lag_12w_count; complaint_count; lag_1w_count; week_of_year; lag_52w_count | rolling_8w_std; lag_8w_count; rolling_8w_mean; history_weeks_available; rolling_4w_std |
| case_009 | false_positive_high_score | 2024-04-08 | Bronx | Mott Haven-Port Morris | infrastructure | 0 | 1 | 0.8075 | ratio_to_8w_mean; complaint_category; history_weeks_available; complaint_count; weather_tmax_mean | rolling_8w_std; lag_8w_count; rolling_4w_std; lag_1w_count; rolling_4w_mean |
| case_010 | false_positive_high_score | 2025-06-30 | Manhattan | Washington Heights (North) | public_safety | 0 | 1 | 0.7942 | complaint_category; ratio_to_8w_mean; lag_52w_count; complaint_count; rolling_4w_mean | rolling_8w_std; weather_tmin_mean; diff_4w_count; weather_tmin_max; week_of_year |
| case_011 | false_negative_low_score | 2025-06-23 | Manhattan | Stuyvesant Town-Peter Cooper Village | noise | 1 | 0 | 0.1823 | rolling_8w_std; ratio_to_8w_mean; rolling_8w_mean; poi_named_count; pluto_total_residential_area | complaint_category; week_of_year; lag_52w_count; rolling_4w_mean; rolling_12w_mean |
| case_012 | false_negative_low_score | 2024-11-11 | Staten Island | Arden Heights-Rossville | public_safety | 1 | 0 | 0.1822 | complaint_count; pluto_total_residential_area; history_weeks_available; weather_tmax_min; poi_named_count | rolling_8w_std; ratio_to_8w_mean; lag_1w_count; complaint_category; rolling_4w_mean |
| case_013 | false_negative_low_score | 2025-02-24 | Queens | Jamaica Estates-Holliswood | noise | 1 | 0 | 0.1822 | complaint_category; week_of_year; lag_52w_count; complaint_count; month | rolling_8w_std; ratio_to_8w_mean; weather_tmin_mean; diff_4w_count; weather_prcp_mean |
| case_014 | false_negative_low_score | 2025-03-17 | Queens | LaGuardia Airport | public_safety | 1 | 0 | 0.1822 | rolling_8w_std; complaint_category; weather_tmin_min; weather_tmax_min; lag_8w_count | ratio_to_8w_mean; lag_52w_count; lag_1w_count; rolling_4w_mean; week_of_year |
| case_015 | false_negative_low_score | 2025-01-27 | Brooklyn | Bushwick (West) | other | 1 | 0 | 0.1822 | rolling_8w_std; history_weeks_available; pluto_total_buildings; poi_named_count; complaint_category | rolling_12w_std; ratio_to_8w_mean; rolling_4w_mean; poi_semantic_entropy; lag_52w_count |
| case_016 | true_negative_low_score | 2025-01-20 | Bronx | North & South Brother Islands | other | 0 | 0 | 0.0003 | rolling_8w_std; ratio_to_8w_mean; complaint_category; pluto_parcel_density_per_km2; rolling_12w_std | week_of_year; rolling_4w_mean; lag_2w_count; lag_52w_count; diff_4w_count |
| case_017 | true_negative_low_score | 2025-01-13 | Bronx | North & South Brother Islands | other | 0 | 0 | 0.0003 | rolling_8w_std; complaint_category; pluto_parcel_density_per_km2; weather_awnd_mean; lag_8w_count | week_of_year; lag_52w_count; ratio_to_8w_mean; pct_change_1w; lag_1w_count |
| case_018 | true_negative_low_score | 2025-02-10 | Bronx | North & South Brother Islands | other | 0 | 0 | 0.0003 | complaint_category; rolling_8w_std; history_weeks_available; pluto_parcel_density_per_km2; poi_semantic_entropy | ratio_to_8w_mean; rolling_4w_mean; week_of_year; lag_1w_count; lag_52w_count |
| case_019 | true_negative_low_score | 2025-01-27 | Bronx | North & South Brother Islands | other | 0 | 0 | 0.0003 | rolling_8w_std; complaint_category; history_weeks_available; pluto_parcel_density_per_km2; poi_named_count | ratio_to_8w_mean; week_of_year; rolling_4w_mean; weather_tmin_mean; month |
| case_020 | true_negative_low_score | 2025-01-06 | Bronx | North & South Brother Islands | other | 0 | 0 | 0.0004 | rolling_8w_std; complaint_category; pluto_parcel_density_per_km2; weather_tmin_mean; poi_semantic_entropy | ratio_to_8w_mean; week_of_year; lag_52w_count; rolling_4w_mean; month |

## Paper wording suggestions

Suggested wording: `Global SHAP analysis indicates that the model relies primarily on historical demand dynamics, seasonal/calendar signals, and selected contextual variables when estimating next-week abnormal service-demand risk. Category-specific SHAP summaries show that the relative importance of these factors varies across municipal service domains, supporting the semantics-aware formulation of the forecasting task.`


Careful wording: `SHAP values explain how features contribute to model predictions; they should not be interpreted as causal effects of weather, POI density, or land-use conditions on actual urban problems.`


## Output files

- global_importance: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\shap_explainability_lgbm\shap_global_importance.csv`
- group_importance: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\shap_explainability_lgbm\shap_group_importance.csv`
- category_importance: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\shap_explainability_lgbm\shap_importance_by_complaint_category.csv`
- local_cases: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\shap_explainability_lgbm\shap_local_cases.csv`
- local_contributions: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\shap_explainability_lgbm\shap_local_case_feature_contributions.csv`
- summary_bar: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\shap_explainability_lgbm\shap_summary_bar.png`
- group_bar: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\shap_explainability_lgbm\shap_group_importance.png`
- beeswarm: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\shap_explainability_lgbm\shap_beeswarm.png`
- report: `D:\00_Major\K4\DAP\SIMC_project\data\processed\model_results\shap_explainability_lgbm\shap_explainability_report.md`
