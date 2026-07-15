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
