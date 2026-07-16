# Final Same-Target Numerical Consistency Report

All Table 3 rows are reconstructed from current held-out-2025 artifacts using T2, with 122,616 rows and 13,562 positives.

| model | rows | positives | PR-AUC | P@5% | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Poisson GLM, no NTA FE | 122,616 | 13,562 | 0.141 | 0.146 | 0.376 | 0.038 | 0.068 |
| Poisson GLM + NTA FE | 122,616 | 13,562 | 0.141 | 0.146 | 0.375 | 0.038 | 0.068 |
| HGB Poisson count | 122,616 | 13,562 | 0.153 | 0.176 | 0.513 | 0.103 | 0.171 |
| Hurdle HGB count | 122,616 | 13,562 | 0.152 | 0.172 | 0.497 | 0.111 | 0.181 |
| No-shortcut LightGBM classifier | 122,616 | 13,562 | 0.317 | 0.418 | 0.263 | 0.574 | 0.361 |

- Volume-decile rows sum to the full final-style 2025 test population.
- Supplementary baseline CIs use 250 NTA-category cluster resamples; the final LightGBM main-metric CI artifact uses 1,000. Table 3 values and the supplementary CSV retain full precision.
- T0/T1/T2/T3 sensitivity is labelled as a separate two-year diagnostic and is not mixed into the T2 final comparison.
