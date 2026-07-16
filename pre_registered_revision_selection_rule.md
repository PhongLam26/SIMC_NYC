# Pre-Registered Revision Selection Rule

This file records the selection policy for the major revision before new model-selection experiments are run.

## Candidate Design Families

The revision should compare:

- Current target with current features, as the submitted baseline.
- Current target with direct 8-week formula-aligned shortcut features removed.
- Reduced-history variants that retain temporal information without directly mirroring the target formula.
- Sparse-aware target variants that reduce pathological positives in near-zero baseline rows.
- Count-model baselines that predict `target_next_week_count` and convert counts to an abnormal-event decision rule.

## Primary Selection Principle

Select the final design by validation and rolling-origin evidence, not by the final 2024-2025 test period.

The preferred final design should satisfy:

- Construct validity: the target must represent meaningful next-week abnormal reported demand, not a low-count artifact.
- Temporal validity: feature construction must use information available at or before feature week t.
- Stability: performance should not depend on a single random seed or a single chronological split.
- Operational usefulness: precision at fixed alert capacity and lift over base rate should be emphasized over raw F1 alone.
- Interpretability validity: SHAP and local explanations must be generated from the final selected model and feature set.

## Metrics

Primary metrics for selection:

- Validation and rolling-origin PR-AUC.
- Precision at top 1 percent, 5 percent, and 10 percent scored rows.
- Calibration diagnostics where probability scores are interpreted.
- Category and historical-volume stability.

Secondary metrics:

- F1, precision, recall, ROC-AUC, balanced accuracy, alert rate, and confusion matrix.

## Tie-Breaking

If two candidates have similar validation and rolling-origin performance, prefer the simpler and more defensible candidate:

1. A single model over an ensemble.
2. A feature set without direct target-formula shortcuts.
3. A target definition with lower near-zero-baseline pathology.
4. A decision rule that is easier to explain and reproduce.

## Test-Set Use

The final 2024-2025 test period is reserved for reporting the selected design. It must not be used to choose target definitions, thresholds, hyperparameters, or model families.

## Post-Experiment Draft Freeze

This section records the current manuscript decision after the major-revision evidence was generated. It documents the rationale so that later text edits do not quietly re-select by test performance.

### Frozen Task and Target

The current draft uses `T2_min_count_3`:

`count_{t+1} > mu_8w + 1.5 sigma_8w AND count_{t+1} >= 3`

Rationale:

- T2 removes one-call and two-call positives while retaining the same dense panel risk set as T0/T1.
- In the target-selection validation split, T2 improves PR-AUC over T0/T1 while reducing the share of positives from cells with `rolling_8w_mean < 1`.
- T3 has stronger raw validation PR-AUC but changes the evaluated population by excluding low-baseline rows, so it is not used as the main target.
- The final manuscript reports T2 because it gives a clearer abnormal reported-demand event without dropping low-volume neighborhoods/categories from the task.

### Frozen Score Model

The current draft uses a single no-shortcut LightGBM classifier, not an ensemble.

Rationale:

- The single model aligns the score model and the explanation model.
- The feature set excludes the direct 8-week target-construction predictors: `rolling_8w_mean`, `rolling_8w_std`, `rolling_8w_sum`, and `ratio_to_8w_mean`.
- Count baselines are retained as comparators but are weaker event rankers.
- The final narrative avoids claiming that an ensemble or category-threshold layer is the best overall strategy.

### Frozen Feature Scope

The current draft uses the `07_final_no_shortcut_borough` feature configuration for the main candidate.

Rationale:

- It keeps shifted history, current feature-week count, non-target 12-week history, calendar fields, complaint category, borough, and citywide feature-week weather.
- The NTA fixed-effect variant is reported as a diagnostic because it is higher-dimensional and only slightly different in the current single-seed ablation.
- OSM/PLUTO/POI features are removed from the main candidate and from manuscript claims.
- Weather is retained only as a contextual feature-week control and is not claimed as a standalone contribution, because validation evidence does not show a robust weather-driven gain.

### Calibration, Threshold, and Capacity

- Platt calibration is fit on validation year 2024 only and applied unchanged to 2025.
- The classification threshold is selected on validation only; it is not selected by 2025 test metrics.
- Precision@1% and precision@5% are reported as capacity-aware ranking metrics because the validation-selected threshold produces a high weekly alert workload.

### Current Held-Out Reporting Convention

The manuscript headline reports the final-style 2025 holdout:

- train through 2023;
- validation year 2024;
- held-out test year 2025.

The older 2024-2025 diagnostic split is retained only where needed for target-composition diagnostics and is labeled as such.
