# Local Case Selection Report

Cases are selected by pre-specified rules from the final-style 2025 T2 no-shortcut LightGBM candidate:

- TP: highest-score true positive.
- FP: highest-score false positive.
- FN: missed positive with largest realized z-exceedance, breaking ties by next-week count.

| case_id | case_type | target_week | nta2020 | ntaname | boroname | complaint_category | y_true | prediction | score | threshold | target_next_week_count | rolling_8w_mean | z_exceedance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tp | tp_high_confidence | 2025-10-20 | BX0904 | Parkchester | Bronx | housing | 1 | 1 | 0.9754 | 0.5452 | 244.0000 | 52.3750 | 11.8455 |
| fp | fp_high_confidence | 2025-03-17 | BK1204 | Mapleton-Midwood (West) | Brooklyn | noise | 0 | 1 | 0.9563 | 0.5452 | 12.0000 | 12.7500 | -0.1465 |
| fn | fn_severe_missed | 2025-06-30 | QN0401 | Elmhurst | Queens | other | 1 | 0 | 0.3165 | 0.5452 | 426.0000 | 11.0000 | 85.7384 |

Top local contributions are saved in `shap_local_case_feature_contributions.csv`. These explanations describe model score contributions, not causal effects.
