# Seed Stability Report

This report evaluates five stochastic seeds for final-style 2025 no-shortcut LightGBM candidates. Thresholds and Platt calibrators are fitted only on validation 2024 and evaluated on test 2025.

Seeds: 42, 123, 2026, 3407, 7777.

## Test Summary Across Seeds

| target_definition | calibration_method | seeds | pr_auc_mean | pr_auc_std | precision_at_5pct_mean | precision_at_5pct_std | f1_mean | f1_std | brier_mean | brier_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0_current_reference | platt | 5 | 0.3115 | 0.0021 | 0.4248 | 0.0037 | 0.3549 | 0.0005 | 0.1011 | 0.0001 |
| T0_current_reference | uncalibrated | 5 | 0.3115 | 0.0021 | 0.4248 | 0.0037 | 0.3549 | 0.0005 | 0.1790 | 0.0030 |
| T2_min_count_3 | platt | 5 | 0.3185 | 0.0023 | 0.4194 | 0.0034 | 0.3638 | 0.0015 | 0.0868 | 0.0001 |
| T2_min_count_3 | uncalibrated | 5 | 0.3185 | 0.0023 | 0.4194 | 0.0034 | 0.3638 | 0.0015 | 0.1749 | 0.0054 |

## Per-Seed Platt-Calibrated Test Metrics

| target_definition | seed | pr_auc | precision_at_1pct | precision_at_5pct | f1 | precision | recall | brier | threshold | alert_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0_current_reference | 42 | 0.3120 | 0.5884 | 0.4236 | 0.3544 | 0.2576 | 0.5675 | 0.1011 | 0.1661 | 0.2823 |
| T0_current_reference | 123 | 0.3090 | 0.5819 | 0.4213 | 0.3551 | 0.2620 | 0.5509 | 0.1012 | 0.1730 | 0.2694 |
| T0_current_reference | 2026 | 0.3134 | 0.5966 | 0.4277 | 0.3556 | 0.2599 | 0.5629 | 0.1011 | 0.1702 | 0.2775 |
| T0_current_reference | 3407 | 0.3097 | 0.5811 | 0.4218 | 0.3543 | 0.2576 | 0.5672 | 0.1012 | 0.1673 | 0.2822 |
| T0_current_reference | 7777 | 0.3136 | 0.5901 | 0.4296 | 0.3550 | 0.2607 | 0.5561 | 0.1010 | 0.1711 | 0.2733 |
| T2_min_count_3 | 42 | 0.3165 | 0.5697 | 0.4180 | 0.3613 | 0.2635 | 0.5744 | 0.0869 | 0.1668 | 0.2411 |
| T2_min_count_3 | 123 | 0.3197 | 0.5892 | 0.4239 | 0.3645 | 0.2777 | 0.5302 | 0.0868 | 0.1871 | 0.2111 |
| T2_min_count_3 | 2026 | 0.3213 | 0.5892 | 0.4200 | 0.3651 | 0.2696 | 0.5654 | 0.0867 | 0.1755 | 0.2320 |
| T2_min_count_3 | 3407 | 0.3193 | 0.5819 | 0.4206 | 0.3645 | 0.2766 | 0.5341 | 0.0867 | 0.1878 | 0.2136 |
| T2_min_count_3 | 7777 | 0.3158 | 0.5778 | 0.4146 | 0.3635 | 0.2732 | 0.5431 | 0.0869 | 0.1787 | 0.2199 |

## Guardrails

- This is a final-style 2025 seed-stability pass, not a full rolling-origin multi-seed run for every fold.
- T0 and T2 use different target labels, so small metric differences should not be interpreted as a paired model comparison.
- The manuscript should report rounded metrics and uncertainty, not overclaim 0.001-scale differences.
