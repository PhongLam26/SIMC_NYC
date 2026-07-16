# Full-Training Ablation Report

This report runs a full-row final-style ablation for `T2_min_count_3` with seed `42`. It replaces the old compact 300k protocol for this pass, but it is not yet a five-seed ablation table.

All configurations exclude OSM/PLUTO and remove formula-aligned 8-week shortcut features.

## Validation Metrics for Selection

| feature_config | raw_feature_count | model_feature_count | contains_weather | contains_borough | contains_nta_fe | pr_auc | precision_at_5pct | f1 | precision | recall | threshold | fit_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 08_final_no_shortcut_nta_fe | 58 | 331 | True | True | True | 0.2951 | 0.3924 | 0.3618 | 0.2711 | 0.5436 | 0.5832 | 25.3988 |
| 07_final_no_shortcut_borough | 57 | 69 | True | True | False | 0.2939 | 0.3906 | 0.3611 | 0.2667 | 0.5590 | 0.5452 | 24.7286 |
| 06_history_calendar_weather_nta_fe | 40 | 309 | True | False | True | 0.2759 | 0.3644 | 0.3480 | 0.2650 | 0.5070 | 0.5950 | 16.2291 |
| 03_history_calendar | 16 | 24 | False | False | False | 0.2754 | 0.3633 | 0.3436 | 0.2472 | 0.5633 | 0.5466 | 11.3580 |
| 04_history_calendar_weather | 39 | 47 | True | False | False | 0.2738 | 0.3639 | 0.3466 | 0.2649 | 0.5011 | 0.6000 | 14.8665 |
| 05_history_calendar_weather_borough | 40 | 52 | True | True | False | 0.2722 | 0.3650 | 0.3448 | 0.2639 | 0.4973 | 0.5900 | 15.3385 |
| 02_history_current_count | 9 | 17 | False | False | False | 0.2464 | 0.3287 | 0.3152 | 0.2217 | 0.5454 | 0.4252 | 8.8289 |
| 01_history_lags_only | 8 | 16 | False | False | False | 0.1848 | 0.2352 | 0.2616 | 0.1627 | 0.6672 | 0.3232 | 8.2950 |
| 11_calendar_weather_only | 31 | 39 | True | False | False | 0.1699 | 0.2274 | 0.2474 | 0.1755 | 0.4190 | 0.5229 | 12.1225 |
| 09_calendar_only | 8 | 16 | False | False | False | 0.1695 | 0.2284 | 0.2405 | 0.1646 | 0.4464 | 0.5400 | 7.6844 |
| 10_weather_only | 24 | 32 | True | False | False | 0.1405 | 0.1602 | 0.2218 | 0.1423 | 0.5029 | 0.4700 | 10.9696 |

## Held-Out 2025 Diagnostics

| feature_config | pr_auc | precision_at_5pct | f1 | precision | recall | threshold | alert_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 01_history_lags_only | 0.1857 | 0.2381 | 0.2672 | 0.1663 | 0.6804 | 0.3232 | 0.4526 |
| 02_history_current_count | 0.2528 | 0.3383 | 0.3173 | 0.2235 | 0.5467 | 0.4252 | 0.2705 |
| 03_history_calendar | 0.2861 | 0.3804 | 0.3446 | 0.2478 | 0.5652 | 0.5466 | 0.2523 |
| 04_history_calendar_weather | 0.2955 | 0.3887 | 0.3537 | 0.2633 | 0.5386 | 0.6000 | 0.2262 |
| 05_history_calendar_weather_borough | 0.2987 | 0.3918 | 0.3527 | 0.2643 | 0.5297 | 0.5900 | 0.2217 |
| 06_history_calendar_weather_nta_fe | 0.2991 | 0.3926 | 0.3530 | 0.2657 | 0.5257 | 0.5950 | 0.2188 |
| 07_final_no_shortcut_borough | 0.3165 | 0.4180 | 0.3613 | 0.2635 | 0.5744 | 0.5452 | 0.2411 |
| 08_final_no_shortcut_nta_fe | 0.3187 | 0.4215 | 0.3634 | 0.2682 | 0.5632 | 0.5832 | 0.2323 |
| 09_calendar_only | 0.1811 | 0.2437 | 0.2416 | 0.1658 | 0.4451 | 0.5400 | 0.2970 |
| 10_weather_only | 0.1371 | 0.1770 | 0.2043 | 0.1276 | 0.5125 | 0.4700 | 0.4444 |
| 11_calendar_weather_only | 0.1778 | 0.2376 | 0.2519 | 0.1713 | 0.4764 | 0.5229 | 0.3077 |

## Weather-Specific Validation Rows

| feature_config | pr_auc | precision_at_5pct | f1 | delta_vs_history_calendar_pr_auc | delta_vs_history_calendar_precision_at_5pct | delta_vs_history_calendar_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| 03_history_calendar | 0.2754 | 0.3633 | 0.3436 | 0.0000 | 0.0000 | 0.0000 |
| 04_history_calendar_weather | 0.2738 | 0.3639 | 0.3466 | -0.0016 | 0.0006 | 0.0029 |
| 09_calendar_only | 0.1695 | 0.2284 | 0.2405 | -0.1059 | -0.1349 | -0.1032 |
| 10_weather_only | 0.1405 | 0.1602 | 0.2218 | -0.1349 | -0.2031 | -0.1218 |
| 11_calendar_weather_only | 0.1699 | 0.2274 | 0.2474 | -0.1055 | -0.1359 | -0.0963 |

## Guardrails

- Feature selection should use validation/backtest evidence, not 2025 test metrics.
- Weather here is city-level feature-week observed Central Park exposure, not NTA-level weather and not t+1 forecast weather.
- Historical NWS forecast archives and spatial weather grids remain Future Work because they require external data not collected in this revision.
- This pass uses one seed for all rows; five-seed ablation for final/key rows remains open.
