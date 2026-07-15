# Prospective Leakage Audit

Ensemble directory: `data\processed\model_results\prospective\ensemble`
Prospective feature config: `configs\features_prospective.json`

| Check | Status | Evidence |
| --- | --- | --- |
| Prospective config excludes OSM/PLUTO/future-target columns | PASS | prospective_forecast_available: 61 input features. |
| Prospective standalone config matches model_config feature set | PASS | standalone=61, model_config=61. |
| Final ensemble is marked prospective | PASS | analysis_type='prospective'. |
| Final ensemble uses prospective feature set | PASS | feature_set='prospective_forecast_available', expected='prospective_forecast_available'. |
| Final LightGBM feature names exclude OSM/PLUTO/future-target columns | PASS | 73 model/input feature names audited. |
| Final XGBoost feature names exclude OSM/PLUTO/future-target columns | PASS | 73 model/input feature names audited. |
| Validation/test split uses target weeks 2023 and 2024-2025 | PASS | {"test": {"min": 2024.0, "max": 2025.0, "nunique": 2}, "validation": {"min": 2023.0, "max": 2023.0, "nunique": 1}} |
| Threshold artifacts are validation-selected | PASS | threshold_selection_summary.csv and category_thresholds.csv present; both store validation_* selection metrics. |
| Weather features are feature-week covariates, not target-week weather | PASS | 22 weather-related feature names audited. |
| Lag/rolling/history features do not reference future or target-week values | PASS | 21 shifted/history feature names audited. |

Overall status: PASS
