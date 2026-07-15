# Major Revision Initial Sparse-Panel Audit

This audit is a reproducible first-pass diagnostic for reviewer P1-2. It does not resolve the target-shortcut critique by itself; it quantifies how much of the current held-out positive class comes from low historical-volume rows.

## Key Sparse Diagnostics

| metric | value | note | share_of_test_positives |
| --- | --- | --- | --- |
| test_rows | 247590.0000 | Rows in held-out 2024-2025 scored set. |  |
| test_positive_rows | 31570.0000 | Observed abnormal rows. |  |
| test_positive_share | 0.1275 | Observed abnormal prevalence. |  |
| positive_rows_with_rolling_8w_mean_lt_0.25 | 2560.0000 | Reviewer P1-2 sparse-baseline diagnostic. | 0.0811 |
| positive_rows_with_rolling_8w_mean_lt_0.5 | 4073.0000 | Reviewer P1-2 sparse-baseline diagnostic. | 0.1290 |
| positive_rows_with_rolling_8w_mean_lt_1 | 5670.0000 | Reviewer P1-2 sparse-baseline diagnostic. | 0.1796 |
| positive_rows_with_rolling_8w_mean_lt_2 | 7469.0000 | Reviewer P1-2 sparse-baseline diagnostic. | 0.2366 |
| target_count_eq_1 | 2707.0000 | Composition of positive target rows. | 0.0857 |
| target_count_eq_2 | 1586.0000 | Composition of positive target rows. | 0.0502 |
| target_count_eq_3 | 1123.0000 | Composition of positive target rows. | 0.0356 |
| target_count_ge_4 | 26154.0000 | Composition of positive target rows. | 0.8284 |

## Capacity Ranking

| risk_set | rows | true_positives | precision | recall | lift_over_base_rate | base_rate |
| --- | --- | --- | --- | --- | --- | --- |
| top_1pct | 2476 | 1462 | 0.5905 | 0.0463 | 4.6308 | 0.1275 |
| top_5pct | 12380 | 5521 | 0.4460 | 0.1749 | 3.4975 | 0.1275 |
| top_10pct | 24759 | 9431 | 0.3809 | 0.2987 | 2.9873 | 0.1275 |
| category_threshold_alerts | 58165 | 17059 | 0.2933 | 0.5404 | 2.3001 | 0.1275 |

## Performance by Historical-Volume Decile

| volume_decile | rows | rolling_8w_mean_min | rolling_8w_mean_median | rolling_8w_mean_max | positive_rows | positive_share | precision | recall | f1 | pr_auc | alert_rate | precision_at_1pct | precision_at_5pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1 | 24759 | 0.0000 | 0.0000 | 0.0000 | 1166 | 0.0471 | 0.2567 | 0.1552 | 0.1935 | 0.1488 | 0.0285 | 0.3306 | 0.2044 |
| D2 | 24759 | 0.0000 | 0.2500 | 0.5000 | 3317 | 0.1340 | 0.2610 | 0.5270 | 0.3491 | 0.2842 | 0.2705 | 0.4798 | 0.3934 |
| D3 | 24759 | 0.5000 | 1.1250 | 2.2500 | 3379 | 0.1365 | 0.2662 | 0.5061 | 0.3489 | 0.2879 | 0.2595 | 0.5202 | 0.3942 |
| D4 | 24759 | 2.2500 | 3.6250 | 5.2500 | 3641 | 0.1471 | 0.2872 | 0.5905 | 0.3864 | 0.3230 | 0.3024 | 0.5444 | 0.4321 |
| D5 | 24759 | 5.2500 | 7.1250 | 9.2500 | 3648 | 0.1473 | 0.2903 | 0.5636 | 0.3832 | 0.3377 | 0.2861 | 0.5685 | 0.4596 |
| D6 | 24759 | 9.2500 | 11.7500 | 14.6250 | 3379 | 0.1365 | 0.2876 | 0.5472 | 0.3770 | 0.3297 | 0.2597 | 0.6048 | 0.4297 |
| D7 | 24759 | 14.6250 | 18.2500 | 23.0000 | 3106 | 0.1254 | 0.2947 | 0.5393 | 0.3812 | 0.3107 | 0.2295 | 0.5000 | 0.4297 |
| D8 | 24759 | 23.0000 | 29.6250 | 39.1250 | 3182 | 0.1285 | 0.3087 | 0.5563 | 0.3970 | 0.3462 | 0.2316 | 0.6089 | 0.4903 |
| D9 | 24759 | 39.1250 | 53.7500 | 78.1250 | 3397 | 0.1372 | 0.3152 | 0.5929 | 0.4116 | 0.3848 | 0.2581 | 0.6815 | 0.5170 |
| D10 | 24759 | 78.1250 | 121.7500 | 9050.5000 | 3355 | 0.1355 | 0.3446 | 0.5681 | 0.4290 | 0.4101 | 0.2234 | 0.7339 | 0.5363 |

## Category Sparse Profile

| complaint_category | rows | positive_rows | positive_share | rolling_8w_mean_zero_share | rolling_8w_mean_lt_1_share | positive_rows_rolling_8w_mean_lt_1 | positive_share_from_rolling_8w_mean_lt_1 | mean_rolling_8w_mean | mean_target_next_week_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| environment | 27510 | 2896 | 0.1053 | 0.1956 | 0.2903 | 481 | 0.1661 | 4.3191 | 4.2337 |
| housing | 27510 | 3720 | 0.1352 | 0.0984 | 0.1808 | 417 | 0.1121 | 65.4679 | 65.8904 |
| infrastructure | 27510 | 3596 | 0.1307 | 0.0597 | 0.1811 | 578 | 0.1607 | 16.1436 | 16.3088 |
| noise | 27510 | 3982 | 0.1447 | 0.0982 | 0.1814 | 427 | 0.1072 | 56.7838 | 57.4107 |
| other | 27510 | 3620 | 0.1316 | 0.1413 | 0.4578 | 1738 | 0.4801 | 5.8018 | 5.7769 |
| parking_traffic | 27510 | 3586 | 0.1304 | 0.0507 | 0.1419 | 472 | 0.1316 | 68.2706 | 67.9210 |
| public_safety | 27510 | 3536 | 0.1285 | 0.1015 | 0.2031 | 509 | 0.1439 | 14.1304 | 14.0495 |
| sanitation | 27510 | 3172 | 0.1153 | 0.1185 | 0.2324 | 531 | 0.1674 | 15.0227 | 14.9040 |
| water_sewer | 27510 | 3462 | 0.1258 | 0.1380 | 0.2457 | 517 | 0.1493 | 8.2240 | 8.2314 |

## Interpretation Guardrails

- These diagnostics describe the current submitted target and decision layer only.
- Because the current target uses an 8-week rolling baseline, rows with low `rolling_8w_mean` need target-definition sensitivity checks before claims are made.
- Follow-up scripts should test sparse-aware targets and no-shortcut feature sets before the manuscript is rewritten.
