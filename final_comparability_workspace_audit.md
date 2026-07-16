# Final Same-Target Comparability Workspace Audit

| model | target | split | test rows | decision rule | status |
| --- | --- | --- | ---: | --- | --- |
| Poisson GLM, no NTA FE | T2 | train <= 2023; validation 2024; test 2025 | 122,616 | formula_threshold | PASS |
| Poisson GLM + NTA FE | T2 | train <= 2023; validation 2024; test 2025 | 122,616 | formula_threshold | PASS |
| HGB Poisson count | T2 | train <= 2023; validation 2024; test 2025 | 122,616 | formula_threshold | PASS |
| Hurdle HGB count | T2 | train <= 2023; validation 2024; test 2025 | 122,616 | formula_threshold | PASS |
| No-shortcut LightGBM classifier | T2 | train <= 2023; validation 2024; test 2025 | 122,616 | validation_score_threshold | PASS |

- Poisson preprocessing and categorical encodings are fit on train rows only; its test predictions are stored in `count_model_predictions_validation_test.csv.gz`.
- Poisson convergence: no-NTA 156/500 iterations; NTA fixed effects 199/500 iterations.
- Negative Binomial: `blocked_missing_statsmodels`. statsmodels is not installed in the local environment; sklearn has no Negative Binomial GLM.
