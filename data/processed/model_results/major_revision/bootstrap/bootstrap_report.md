# Bootstrap Confidence Interval Report

This report provides a first cluster-bootstrap uncertainty pass for final-style 2025 predictions. It uses existing data only and samples clusters with replacement rather than individual rows.

- Fold: `final_style_2025` with train through 2023, validation 2024, and test 2025.
- Cluster unit: `nta_category`.
- Bootstrap replicates: 1000.
- Targets evaluated: T0 current reference and T2 minimum-count target.
- Models: no-shortcut LightGBM with uncalibrated, Platt, and isotonic scores.

## Main 95% CIs for Platt-Calibrated Scores

| target_definition | calibration_method | metric | ci | clusters | n_bootstrap |
| --- | --- | --- | --- | --- | --- |
| T0_current_reference | platt | pr_auc | 0.3120 [0.3038, 0.3207] | 2358 | 1000 |
| T0_current_reference | platt | f1 | 0.3544 [0.3484, 0.3602] | 2358 | 1000 |
| T0_current_reference | platt | precision | 0.2576 [0.2525, 0.2625] | 2358 | 1000 |
| T0_current_reference | platt | recall | 0.5675 [0.5572, 0.5773] | 2358 | 1000 |
| T0_current_reference | platt | precision_at_1pct | 0.5884 [0.5575, 0.6178] | 2358 | 1000 |
| T0_current_reference | platt | precision_at_5pct | 0.4236 [0.4096, 0.4379] | 2358 | 1000 |
| T0_current_reference | platt | lift_at_1pct | 4.5921 [4.3589, 4.8240] | 2358 | 1000 |
| T0_current_reference | platt | brier | 0.1011 [0.0996, 0.1025] | 2358 | 1000 |
| T0_current_reference | platt | log_loss | 0.3392 [0.3349, 0.3432] | 2358 | 1000 |
| T2_min_count_3 | platt | pr_auc | 0.3165 [0.3080, 0.3252] | 2358 | 1000 |
| T2_min_count_3 | platt | f1 | 0.3613 [0.3551, 0.3676] | 2358 | 1000 |
| T2_min_count_3 | platt | precision | 0.2635 [0.2582, 0.2690] | 2358 | 1000 |
| T2_min_count_3 | platt | recall | 0.5744 [0.5645, 0.5845] | 2358 | 1000 |
| T2_min_count_3 | platt | precision_at_1pct | 0.5697 [0.5428, 0.6023] | 2358 | 1000 |
| T2_min_count_3 | platt | precision_at_5pct | 0.4180 [0.4035, 0.4324] | 2358 | 1000 |
| T2_min_count_3 | platt | lift_at_1pct | 5.1506 [4.8934, 5.4603] | 2358 | 1000 |
| T2_min_count_3 | platt | brier | 0.0869 [0.0852, 0.0888] | 2358 | 1000 |
| T2_min_count_3 | platt | log_loss | 0.2933 [0.2881, 0.2991] | 2358 | 1000 |

## Paired Calibration Differences

Differences are challenger minus baseline within the same target and cluster sample. Negative Brier/log-loss differences favor calibration; positive PR-AUC, precision@k, and F1 differences favor calibration.

| target_definition | challenger | baseline | metric | difference_ci | favorable_direction | win_proportion | ci_includes_zero |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T0_current_reference | platt | uncalibrated | pr_auc | 0.0000 [0.0000, 0.0000] | positive | 0.0000 | True |
| T0_current_reference | platt | uncalibrated | f1 | 0.0000 [0.0000, 0.0000] | positive | 0.0000 | True |
| T0_current_reference | platt | uncalibrated | precision_at_5pct | 0.0000 [0.0000, 0.0000] | positive | 0.0000 | True |
| T0_current_reference | platt | uncalibrated | brier | -0.0731 [-0.0747, -0.0715] | negative | 1.0000 | False |
| T0_current_reference | platt | uncalibrated | log_loss | -0.1824 [-0.1860, -0.1785] | negative | 1.0000 | False |
| T2_min_count_3 | platt | uncalibrated | pr_auc | 0.0000 [0.0000, 0.0000] | positive | 0.0000 | True |
| T2_min_count_3 | platt | uncalibrated | f1 | 0.0000 [0.0000, 0.0000] | positive | 0.0000 | True |
| T2_min_count_3 | platt | uncalibrated | precision_at_5pct | 0.0000 [0.0000, 0.0000] | positive | 0.0000 | True |
| T2_min_count_3 | platt | uncalibrated | brier | -0.0807 [-0.0827, -0.0785] | negative | 1.0000 | False |
| T2_min_count_3 | platt | uncalibrated | log_loss | -0.1995 [-0.2042, -0.1948] | negative | 1.0000 | False |

## Guardrails

- T0 and T2 have different labels, so this report does not present a paired T2-minus-T0 difference as a model superiority claim.
- This is a final-style 2025 uncertainty pass, not yet the full Table 4/Table 5 CI package for every final manuscript row.
- Multiple-seed uncertainty remains open.
