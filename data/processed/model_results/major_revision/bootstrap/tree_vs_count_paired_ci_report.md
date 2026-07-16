# Tree versus Count Paired Bootstrap Report

This report compares the final-style Platt-calibrated no-shortcut LightGBM against row-level count-baseline scores on the same held-out 2025 rows.

- HGB Poisson fit seconds: 10.01
- Hurdle count fit seconds: 17.03
- Bootstrap unit: NTA-category clusters.

| baseline | metric | tree | count | diff | 95% CI | includes zero |
| --- | --- | --- | --- | --- | --- | --- |
| hgb_poisson_formula | pr_auc | 0.3165 | 0.1528 | 0.1637 | 0.1559 to 0.1722 | False |
| hgb_poisson_formula | precision_at_5pct | 0.4180 | 0.1755 | 0.2425 | 0.2269 to 0.2575 | False |
| hgb_poisson_formula | f1 | 0.3613 | 0.1710 | 0.1902 | 0.1812 to 0.1989 | False |
| hgb_poisson_formula | precision | 0.2635 | 0.5129 | -0.2494 | -0.2657 to -0.2318 | False |
| hgb_poisson_formula | recall | 0.5744 | 0.1026 | 0.4718 | 0.4618 to 0.4809 | False |
| hurdle_formula | pr_auc | 0.3165 | 0.1517 | 0.1648 | 0.1565 to 0.1733 | False |
| hurdle_formula | precision_at_5pct | 0.4180 | 0.1722 | 0.2458 | 0.2296 to 0.2610 | False |
| hurdle_formula | f1 | 0.3613 | 0.1812 | 0.1800 | 0.1711 to 0.1899 | False |
| hurdle_formula | precision | 0.2635 | 0.4967 | -0.2332 | -0.2517 to -0.2156 | False |
| hurdle_formula | recall | 0.5744 | 0.1108 | 0.4636 | 0.4536 to 0.4727 | False |

Guardrail: PR-AUC and precision@5% compare ranking scores. F1/precision/recall compare the validation-selected tree threshold against the count-model formula-threshold event decision.