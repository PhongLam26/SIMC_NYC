# Count Model Extension Report

This report adds overdispersed/tree and hurdle-style count baselines for `T2_min_count_3` on the final-style 2025 fold. It complements the earlier Poisson and Poisson + NTA fixed-effect audit.

## Negative Binomial Status

| model_name | status | reason | next_action |
| --- | --- | --- | --- |
| negative_binomial_glm | blocked_missing_statsmodels | statsmodels is not installed in the local environment; sklearn has no Negative Binomial GLM. | Document blocker or install statsmodels before final NB attempt. |

## Held-Out 2025 Test Metrics

| model_name | decision_mode | count_mae | mean_predicted_count | poisson_deviance | pr_auc | precision_at_5pct | f1 | precision | recall | alert_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting_poisson_count | formula_threshold | 8.0211 | 27.6592 | 5.6952 | 0.1528 | 0.1755 | 0.1710 | 0.5129 | 0.1026 | 0.0221 |
| hist_gradient_boosting_poisson_count | validation_score_threshold | 8.0211 | 27.6592 | 5.6952 | 0.1528 | 0.1755 | 0.2429 | 0.1404 | 0.9002 | 0.7091 |
| hurdle_hgb_occurrence_poisson_positive_count | formula_threshold | 7.9377 | 27.5609 | 6.3093 | 0.1517 | 0.1722 | 0.1812 | 0.4967 | 0.1108 | 0.0247 |
| hurdle_hgb_occurrence_poisson_positive_count | validation_score_threshold | 7.9377 | 27.5609 | 6.3093 | 0.1517 | 0.1722 | 0.2437 | 0.1409 | 0.9030 | 0.7091 |

## Guardrails

- These models predict `target_next_week_count` and are converted to the candidate abnormal-event target using either the original count-threshold formula or a validation-selected score threshold.
- The hurdle model is a practical two-stage baseline, not a full statistical hurdle/negative-binomial model.
- The final manuscript should compare count baselines against the frozen final classifier with paired uncertainty before making a strong dominance claim.
