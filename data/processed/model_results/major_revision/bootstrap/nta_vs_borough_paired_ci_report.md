# NTA vs Borough Paired Bootstrap CI

This report compares the final no-shortcut LightGBM borough variant against the otherwise matched NTA fixed-effect variant on the same held-out 2025 rows.

- Target: `T2_min_count_3` (T0 AND count_t+1 >= 3).
- Seed: 42.
- Prediction rows: 122,616.
- Bootstrap unit: NTA-category clusters; replicates: 1,000.
- Difference direction: NTA fixed effects minus borough.

## Paired Differences

| metric | borough_value | nta_fe_value | observed_difference | ci_lower | ci_upper | ci_includes_zero |
| --- | --- | --- | --- | --- | --- | --- |
| pr_auc | 0.3165 | 0.3187 | 0.0022 | 0.0005 | 0.0040 | False |
| precision_at_5pct | 0.4180 | 0.4215 | 0.0034 | -0.0031 | 0.0095 | True |
| f1 | 0.3613 | 0.3634 | 0.0021 | -0.0002 | 0.0045 | True |
| precision | 0.2635 | 0.2682 | 0.0047 | 0.0029 | 0.0066 | False |
| recall | 0.5744 | 0.5632 | -0.0112 | -0.0157 | -0.0065 | False |

## Fit Metadata

| feature_config | fit_seconds | validation_f1 | threshold | raw_feature_count | model_feature_count | contains_borough | contains_nta_fe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 07_final_no_shortcut_borough | 17.4691 | 0.3611 | 0.5452 | 57.0000 | 69.0000 | 1.0000 | 0.0000 |
| 08_final_no_shortcut_nta_fe | 19.4110 | 0.3618 | 0.5832 | 58.0000 | 331.0000 | 1.0000 | 1.0000 |

## Interpretation Guardrail

- The NTA fixed-effect variant has a very small PR-AUC gain whose paired interval excludes zero, while precision@5% and F1 intervals include zero and recall decreases.
- The manuscript should keep the NTA result as a diagnostic spatial check rather than changing the simpler final borough model claim.
