# SHAP Calibration-Scale Audit

This audit verifies the scale used by the local SHAP waterfall figures and the scale used by the final calibrated decision layer.

Findings:

- TreeSHAP local waterfall figures decompose the fitted LightGBM raw margin before Platt calibration.
- The `score` values in the previous local-case artifact are uncalibrated LightGBM probabilities, not Platt-calibrated probabilities.
- Platt calibration is fit on validation year 2024 and is monotone for the audited score range, so it preserves the alert ordering and the same threshold decisions after transforming the threshold.
- Waterfall contributions should not be interpreted as calibrated probabilities or causal effects.

Thresholds:

- Uncalibrated LightGBM probability threshold: 0.545172.
- Equivalent Platt-calibrated probability threshold: 0.166812.

Audited local cases:

| case | true label | alert | raw margin | uncalibrated probability | Platt-calibrated probability | final threshold | previous stored score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tp | 1 | 1 | 3.6793 | 0.9754 | 0.6250 | 0.1668 | 0.9754 |
| fp | 0 | 1 | 3.0851 | 0.9563 | 0.6027 | 0.1668 | 0.9563 |
| fn | 1 | 0 | -0.7701 | 0.3165 | 0.0609 | 0.1668 | 0.3165 |

Implementation notes:

- Target: `T2_min_count_3`.
- Fold: train through 2023, validation 2024, held-out test 2025.
- Seed: 42.
- Formula-aligned 8-week target-construction predictors removed: `rolling_8w_mean`, `rolling_8w_std`, `rolling_8w_sum`, `ratio_to_8w_mean`.
- Model feature count: 69.