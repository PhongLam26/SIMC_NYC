# Calibration Report

This report evaluates fold-specific calibration for the no-shortcut LightGBM models. Platt and isotonic calibrators are fit only on each fold's validation year, then evaluated unchanged on that fold's test year.

Calibration methods:

- `uncalibrated`: raw LightGBM probabilities.
- `platt`: logistic calibration fit on validation scores.
- `isotonic`: isotonic regression fit on validation scores.

## Mean Test Metrics Across Rolling-Origin Folds

| target_definition | calibration_method | folds | brier_mean | brier_std | ece_mean | ece_std | log_loss_mean | pr_auc_mean | precision_at_5pct_mean | f1_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0_current_reference | platt | 5 | 0.1023 | 0.0022 | 0.0083 | 0.0058 | 0.3429 | 0.2959 | 0.4062 | 0.3520 |
| T0_current_reference | isotonic | 5 | 0.1023 | 0.0022 | 0.0078 | 0.0048 | 0.3428 | 0.2894 | 0.4058 | 0.3521 |
| T0_current_reference | uncalibrated | 5 | 0.1955 | 0.0236 | 0.2820 | 0.0358 | 0.5700 | 0.2959 | 0.4062 | 0.3520 |
| T1_min_count_2 | platt | 5 | 0.0930 | 0.0016 | 0.0083 | 0.0044 | 0.3137 | 0.2991 | 0.4060 | 0.3550 |
| T1_min_count_2 | isotonic | 5 | 0.0930 | 0.0016 | 0.0073 | 0.0039 | 0.3131 | 0.2926 | 0.4046 | 0.3551 |
| T1_min_count_2 | uncalibrated | 5 | 0.1891 | 0.0228 | 0.2758 | 0.0335 | 0.5473 | 0.2991 | 0.4060 | 0.3550 |
| T2_min_count_3 | isotonic | 5 | 0.0876 | 0.0014 | 0.0070 | 0.0042 | 0.2950 | 0.2931 | 0.4015 | 0.3587 |
| T2_min_count_3 | platt | 5 | 0.0877 | 0.0014 | 0.0085 | 0.0049 | 0.2961 | 0.2997 | 0.4015 | 0.3583 |
| T2_min_count_3 | uncalibrated | 5 | 0.1852 | 0.0263 | 0.2701 | 0.0362 | 0.5331 | 0.2997 | 0.4015 | 0.3583 |
| T3_mu8w_ge_1_eligible | platt | 5 | 0.1081 | 0.0022 | 0.0104 | 0.0058 | 0.3587 | 0.3064 | 0.4199 | 0.3631 |
| T3_mu8w_ge_1_eligible | isotonic | 5 | 0.1082 | 0.0023 | 0.0097 | 0.0054 | 0.3587 | 0.2996 | 0.4208 | 0.3628 |
| T3_mu8w_ge_1_eligible | uncalibrated | 5 | 0.1912 | 0.0256 | 0.2644 | 0.0399 | 0.5601 | 0.3064 | 0.4199 | 0.3631 |

## Final-Style 2025 Test Fold

| target_definition | calibration_method | brier | ece | log_loss | calibration_slope | calibration_intercept | pr_auc | precision_at_5pct | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0_current_reference | isotonic | 0.1011 | 0.0056 | 0.3386 | 1.0524 | 0.0547 | 0.3037 | 0.4236 | 0.3543 |
| T0_current_reference | platt | 0.1011 | 0.0059 | 0.3392 | 1.0502 | 0.0533 | 0.3120 | 0.4236 | 0.3544 |
| T0_current_reference | uncalibrated | 0.1742 | 0.2522 | 0.5216 | 1.0615 | -1.6291 | 0.3120 | 0.4236 | 0.3544 |
| T2_min_count_3 | isotonic | 0.0868 | 0.0044 | 0.2923 | 1.0307 | 0.0193 | 0.3073 | 0.4198 | 0.3610 |
| T2_min_count_3 | platt | 0.0869 | 0.0057 | 0.2933 | 1.0374 | 0.0303 | 0.3165 | 0.4180 | 0.3613 |
| T2_min_count_3 | uncalibrated | 0.1676 | 0.2493 | 0.4928 | 1.0474 | -1.8292 | 0.3165 | 0.4180 | 0.3613 |

## Reliability Diagram

- PDF: `data/processed/model_results/major_revision/calibration/reliability_diagram.pdf`
- Reliability bins: `data/processed/model_results/major_revision/calibration/reliability_bins.csv`

## Interpretation Guardrails

- Calibration should be judged on Brier score, log loss, ECE, and reliability curves, not PR-AUC alone.
- Platt and isotonic are monotone transformations for most score ranges, so ranking metrics can remain nearly unchanged; do not claim ranking improvement unless PR-AUC or precision@k changes.
- Isotonic can overfit in sparse score regions; final method selection still needs uncertainty intervals and final target/model freeze.
