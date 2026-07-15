# Target Definition Selection Report

This first-pass report evaluates sparse-aware target definitions using the existing data only. It is not the final target-selection decision because the checklist requires rolling-origin validation and uncertainty intervals before freezing the manuscript.

All model rows below use the no-shortcut LightGBM feature set with `rolling_8w_mean`, `rolling_8w_std`, `rolling_8w_sum`, and `ratio_to_8w_mean` removed.

## Candidate Definitions

| target_definition | description |
| --- | --- |
| T0_current_reference | count_t+1 > mu_8w + 1.5 sigma_8w |
| T1_min_count_2 | T0 AND count_t+1 >= 2 |
| T2_min_count_3 | T0 AND count_t+1 >= 3 |
| T3_mu8w_ge_1_eligible | Evaluate only rows with mu_8w >= 1, target remains T0 within eligible rows |

## Test-Period Target Composition

| target_definition | rows | excluded_rows | positive_rows | positive_share | positive_share_mu8w_lt_1 | positive_count_eq_1 | positive_count_eq_2 | positive_count_eq_3 | positive_count_ge_4 | positive_share_count_ge_4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0_current_reference | 247590 | 0 | 31570 | 0.1275 | 0.1796 | 2707 | 1586 | 1123 | 26154 | 0.8284 |
| T1_min_count_2 | 247590 | 0 | 28863 | 0.1166 | 0.1027 | 0 | 1586 | 1123 | 26154 | 0.9061 |
| T2_min_count_3 | 247590 | 0 | 27277 | 0.1102 | 0.0510 | 0 | 0 | 1123 | 26154 | 0.9588 |
| T3_mu8w_ge_1_eligible | 189423 | 58167 | 25900 | 0.1367 | 0.0000 | 0 | 15 | 394 | 25491 | 0.9842 |

## Validation Metrics for Selection

| target_definition | rows | positive_share | pr_auc | precision_at_1pct | precision_at_5pct | f1 | precision | recall | threshold | alert_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T3_mu8w_ge_1_eligible | 92803 | 0.1352 | 0.3079 | 0.5770 | 0.4204 | 0.3606 | 0.2616 | 0.5803 | 0.5233 | 0.3000 |
| T2_min_count_3 | 122616 | 0.1086 | 0.3006 | 0.5346 | 0.4003 | 0.3547 | 0.2675 | 0.5260 | 0.6000 | 0.2135 |
| T1_min_count_2 | 122616 | 0.1155 | 0.2956 | 0.5281 | 0.3963 | 0.3504 | 0.2632 | 0.5242 | 0.5799 | 0.2300 |
| T0_current_reference | 122616 | 0.1270 | 0.2936 | 0.5297 | 0.3993 | 0.3511 | 0.2582 | 0.5488 | 0.5477 | 0.2700 |

## Held-Out Test Diagnostics

| target_definition | rows | positive_share | pr_auc | precision_at_1pct | precision_at_5pct | f1 | precision | recall | threshold | alert_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0_current_reference | 247590 | 0.1275 | 0.3101 | 0.5650 | 0.4236 | 0.3600 | 0.2694 | 0.5423 | 0.5477 | 0.2567 |
| T1_min_count_2 | 247590 | 0.1166 | 0.3133 | 0.5626 | 0.4185 | 0.3635 | 0.2772 | 0.5279 | 0.5799 | 0.2220 |
| T2_min_count_3 | 247590 | 0.1102 | 0.3159 | 0.5622 | 0.4149 | 0.3660 | 0.2812 | 0.5243 | 0.6000 | 0.2054 |
| T3_mu8w_ge_1_eligible | 189423 | 0.1367 | 0.3236 | 0.5826 | 0.4407 | 0.3712 | 0.2748 | 0.5721 | 0.5233 | 0.2847 |

## Volume-Decile Diagnostics

Detailed validation and test decile results are saved in `target_definition_decile_results.csv`. These results are required before any target can be defended in the manuscript because sparse low-volume rows behave differently from high-volume rows.

## Interpretation

- T1 and T2 reduce one-call positives by construction; this improves construct validity for sparse cells but changes the event being forecast.
- T3 removes low-baseline rows from the risk set, so it is not directly comparable to T0/T1/T2 and may exclude low-volume neighborhoods or categories.
- T4 hurdle-style modeling is not completed in this pass and remains open.
- Final target selection must use construct validity plus rolling-origin validation/backtesting, not the 2024-2025 held-out test diagnostics.
