# Disparity and Workload Audit

This audit uses `platt_score` prediction rows for `T2_min_count_3` from the final-style 2025 bootstrap artifact. It evaluates observable group workload only; no socioeconomic or demographic data are used.

## Weekly Workload Summary

| weeks | mean_alerts_per_week | median_alerts_per_week | min_alerts_per_week | max_alerts_per_week | mean_positives_per_week | mean_alert_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 52 | 568.5769 | 510.0000 | 97 | 1055 | 260.8077 | 0.2411 |

## Borough Metrics

| group | rows | positive_share | alerts | alert_rate | precision | recall | f1 | pr_auc | precision_at_5pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Brooklyn | 32292 | 0.1101 | 8370 | 0.2592 | 0.2568 | 0.6045 | 0.3604 | 0.3131 | 0.4217 |
| Bronx | 23400 | 0.1138 | 5826 | 0.2490 | 0.2714 | 0.5935 | 0.3724 | 0.3545 | 0.4641 |
| Queens | 38376 | 0.1055 | 9279 | 0.2418 | 0.2513 | 0.5758 | 0.3499 | 0.2993 | 0.3913 |
| Staten Island | 10764 | 0.0965 | 2382 | 0.2213 | 0.2452 | 0.5621 | 0.3414 | 0.2750 | 0.3581 |
| Manhattan | 17784 | 0.1267 | 3709 | 0.2086 | 0.3084 | 0.5075 | 0.3837 | 0.3416 | 0.4573 |

## Complaint Category Metrics

| group | rows | positive_share | alerts | alert_rate | precision | recall | f1 | pr_auc | precision_at_5pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noise | 13624 | 0.1320 | 4690 | 0.3442 | 0.2915 | 0.7599 | 0.4213 | 0.4127 | 0.5279 |
| infrastructure | 13624 | 0.1236 | 3769 | 0.2766 | 0.2470 | 0.5529 | 0.3415 | 0.2599 | 0.3138 |
| parking_traffic | 13624 | 0.1229 | 3647 | 0.2677 | 0.2605 | 0.5672 | 0.3570 | 0.3035 | 0.4003 |
| environment | 13624 | 0.0911 | 3259 | 0.2392 | 0.2301 | 0.6044 | 0.3333 | 0.2748 | 0.3416 |
| water_sewer | 13624 | 0.1091 | 3246 | 0.2383 | 0.2779 | 0.6070 | 0.3812 | 0.3643 | 0.4765 |
| public_safety | 13624 | 0.1095 | 3159 | 0.2319 | 0.2824 | 0.5979 | 0.3836 | 0.2996 | 0.3886 |
| housing | 13624 | 0.1216 | 2916 | 0.2140 | 0.2949 | 0.5193 | 0.3762 | 0.3816 | 0.5176 |
| sanitation | 13624 | 0.0957 | 2793 | 0.2050 | 0.2184 | 0.4678 | 0.2978 | 0.2339 | 0.3387 |
| other | 13624 | 0.0899 | 2087 | 0.1532 | 0.2530 | 0.4310 | 0.3188 | 0.2596 | 0.3578 |

## Historical-Volume Decile Metrics

| group | rows | mean_rolling_8w_mean | positive_share | alerts | alert_rate | precision | recall | f1 | pr_auc | precision_at_5pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1 | 12262 | 0.0000 | 0.0029 | 5 | 0.0004 | 0.0000 | 0.0000 | 0.0000 | 0.0129 | 0.0195 |
| D10 | 12262 | 162.3039 | 0.1386 | 3232 | 0.2636 | 0.3063 | 0.5824 | 0.4015 | 0.3998 | 0.5554 |
| D2 | 12262 | 0.2321 | 0.0235 | 197 | 0.0161 | 0.1777 | 0.1215 | 0.1443 | 0.1131 | 0.1352 |
| D3 | 12261 | 1.2407 | 0.1201 | 3855 | 0.3144 | 0.2252 | 0.5893 | 0.3258 | 0.2601 | 0.3502 |
| D4 | 12262 | 3.6394 | 0.1443 | 4303 | 0.3509 | 0.2505 | 0.6094 | 0.3551 | 0.3041 | 0.4283 |
| D5 | 12261 | 7.2821 | 0.1488 | 3907 | 0.3187 | 0.2682 | 0.5746 | 0.3657 | 0.3172 | 0.4397 |
| D6 | 12262 | 12.0600 | 0.1378 | 3668 | 0.2991 | 0.2674 | 0.5805 | 0.3662 | 0.3092 | 0.4186 |
| D7 | 12261 | 18.7340 | 0.1265 | 3376 | 0.2753 | 0.2539 | 0.5525 | 0.3479 | 0.2827 | 0.3681 |
| D8 | 12262 | 30.5756 | 0.1271 | 3262 | 0.2660 | 0.2652 | 0.5548 | 0.3588 | 0.3136 | 0.4235 |
| D9 | 12261 | 56.1344 | 0.1364 | 3761 | 0.3067 | 0.2840 | 0.6384 | 0.3931 | 0.3713 | 0.5081 |

## Guardrails

- This is not a socioeconomic fairness audit because ACS/Census demographic data are outside the no-new-external-data boundary for this revision.
- Borough and volume-decile diagnostics are still useful for detecting workload concentration and performance heterogeneity in observable groups.
- Final manuscript fairness language should explicitly defer socioeconomic fairness to Future Work rather than implying it has been completed.
