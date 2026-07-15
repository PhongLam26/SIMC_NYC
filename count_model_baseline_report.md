# Count Model Baseline Report

This report records the first full-data count-model baseline for the major revision.

Generated detailed artifacts:

- `data/processed/model_results/major_revision/model_audits/count_model_baseline_results.csv`
- `data/processed/model_results/major_revision/model_audits/count_model_baseline_report.md`

## Model

Two PoissonRegressor baselines were fitted to predict `target_next_week_count`:

- `poisson_regressor_no_nta`: complaint category, borough, history, and calendar features.
- `poisson_regressor_nta_fe`: the same baseline plus one-hot `nta2020` fixed effects.

For numerical stability, nonnegative count/history predictors were transformed with `log1p`. Both models used `alpha = 10.0` and `max_iter = 500`. They converged before the iteration limit:

- no-NTA model: 106 iterations
- NTA fixed-effect model: 445 iterations

## Held-Out Test Results

| Model | Decision mode | Count MAE | Mean observed count | Mean predicted count | Poisson deviance | F1 | Precision | Recall | PR-AUC | Precision@5% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Poisson no NTA | Original formula threshold | 8.6680 | 28.3029 | 27.6028 | 6.4720 | 0.1460 | 0.1156 | 0.1980 | 0.1400 | 0.1444 |
| Poisson no NTA | Validation score threshold | 8.6680 | 28.3029 | 27.6028 | 6.4720 | 0.2424 | 0.1388 | 0.9577 | 0.1400 | 0.1444 |
| Poisson + NTA FE | Original formula threshold | 8.6839 | 28.3029 | 27.9629 | 6.4590 | 0.1492 | 0.1175 | 0.2044 | 0.1400 | 0.1445 |
| Poisson + NTA FE | Validation score threshold | 8.6839 | 28.3029 | 27.9629 | 6.4590 | 0.2424 | 0.1388 | 0.9576 | 0.1400 | 0.1445 |

## Interpretation

The initial Poisson count baselines are much weaker rankers than the LightGBM binary models under the current target: test PR-AUC is about 0.1400 for Poisson versus 0.3301 for the current LightGBM and 0.3097 for the no-shortcut LightGBM.

Adding NTA fixed effects gives only a small count-deviance improvement and does not materially improve event ranking in this first baseline.

This does not complete P1-5 by itself. Negative Binomial or another overdispersed/hurdle count formulation remains to be attempted or documented as infeasible, and the final comparison still needs rolling-origin validation and uncertainty intervals.
