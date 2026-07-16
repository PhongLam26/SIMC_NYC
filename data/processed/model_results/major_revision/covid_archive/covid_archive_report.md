# COVID and Archive Boundary Audit

This audit addresses reviewer P2-2 using only data already present in the workspace. It checks raw 311 schema consistency, yearly/COVID target composition, category drift, and a final-style training sensitivity that excludes 2020-2021 from the training window.

## Provenance Guardrail

- `code_pulldata/pull_nyc_311.py` currently declares `DATASET_ID = erm2-nwe9`.
- The final modeling panel does not retain a row-level `source_dataset` or archive identifier.
- Therefore, this audit can compare 2015-2019 versus 2020-2025 periods and schema consistency, but it cannot prove row-level provenance from `76ig-c548` versus `erm2-nwe9` without regenerating data with a retained source-id column.
- The manuscript should describe this as an archive/vintage limitation unless source provenance is regenerated and retained.

## Raw 311 Schema Summary

| period | files | status_ok_files | files_with_required_missing | unique_ordered_column_sequences | unique_unordered_column_sets | min_year | max_year |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2015-2019 | 261 | 261 | 0 | 4 | 1 | 2015 | 2019 |
| 2020-2025 | 421 | 421 | 0 | 5 | 1 | 2020 | 2025 |

## Processed File Rows by Year

| year | period | files | rows_read | rows_after_clean | bad_date_rows | missing_nta_rows | missing_nta_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2015 | 2015-2019 | 48 | 2087662 | 2087201 | 0 | 461 | 0.0002 |
| 2016 | 2015-2019 | 48 | 2193273 | 2192872 | 0 | 401 | 0.0002 |
| 2017 | 2015-2019 | 51 | 2296661 | 2296267 | 0 | 394 | 0.0002 |
| 2018 | 2015-2019 | 58 | 2557446 | 2557036 | 0 | 410 | 0.0002 |
| 2019 | 2015-2019 | 56 | 2494732 | 2494267 | 0 | 465 | 0.0002 |
| 2020 | 2020-2025 | 64 | 2863648 | 2862967 | 0 | 681 | 0.0002 |
| 2021 | 2020-2025 | 68 | 3149190 | 3148082 | 0 | 1108 | 0.0004 |
| 2022 | 2020-2025 | 68 | 3119377 | 3118433 | 0 | 944 | 0.0003 |
| 2023 | 2020-2025 | 68 | 3176141 | 3175482 | 0 | 659 | 0.0002 |
| 2024 | 2020-2025 | 75 | 3404033 | 3403326 | 0 | 707 | 0.0002 |
| 2025 | 2020-2025 | 78 | 3604863 | 3604196 | 0 | 667 | 0.0002 |

## Yearly Target/COVID Diagnostics

| target_year | rows | covid_period_share | t0_positive_share | t2_positive_share | mean_target_next_week_count | positive_mu8w_lt_1_share_t0 | positive_mu8w_lt_1_share_t2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | 122616 | 0.0000 | 0.1222 | 0.1039 | 20.2733 | 0.1941 | 0.0529 |
| 2020 | 122616 | 0.9808 | 0.1462 | 0.1250 | 23.2512 | 0.1918 | 0.0557 |
| 2021 | 122616 | 1.0000 | 0.1338 | 0.1140 | 25.6391 | 0.1966 | 0.0574 |
| 2022 | 122616 | 0.0192 | 0.1259 | 0.1086 | 25.3808 | 0.1785 | 0.0488 |
| 2024 | 124974 | 0.0000 | 0.1269 | 0.1097 | 27.7200 | 0.1792 | 0.0513 |
| 2025 | 122616 | 0.0000 | 0.1281 | 0.1106 | 28.8971 | 0.1800 | 0.0507 |

## Category Mix Around COVID and Final Holdout

| year | complaint_category | complaint_count | category_share |
| --- | --- | --- | --- |
| 2019 | housing | 592706 | 0.2384 |
| 2019 | parking_traffic | 485958 | 0.1955 |
| 2019 | noise | 471182 | 0.1895 |
| 2019 | sanitation | 421002 | 0.1694 |
| 2019 | infrastructure | 245387 | 0.0987 |
| 2019 | water_sewer | 111329 | 0.0448 |
| 2019 | public_safety | 90660 | 0.0365 |
| 2019 | environment | 53101 | 0.0214 |
| 2019 | other | 14512 | 0.0058 |
| 2020 | noise | 797904 | 0.2799 |
| 2020 | housing | 522834 | 0.1834 |
| 2020 | parking_traffic | 427707 | 0.1500 |
| 2020 | sanitation | 414912 | 0.1455 |
| 2020 | public_safety | 292846 | 0.1027 |
| 2020 | infrastructure | 246763 | 0.0866 |
| 2020 | water_sewer | 89587 | 0.0314 |
| 2020 | environment | 44869 | 0.0157 |
| 2020 | other | 13549 | 0.0048 |
| 2021 | noise | 764027 | 0.2430 |
| 2021 | housing | 679812 | 0.2162 |
| 2021 | parking_traffic | 628050 | 0.1998 |
| 2021 | sanitation | 486204 | 0.1547 |
| 2021 | infrastructure | 208112 | 0.0662 |
| 2021 | public_safety | 167105 | 0.0532 |
| 2021 | water_sewer | 107484 | 0.0342 |
| 2021 | environment | 60124 | 0.0191 |
| 2021 | other | 42851 | 0.0136 |
| 2024 | parking_traffic | 910841 | 0.2629 |
| 2024 | housing | 905339 | 0.2613 |
| 2024 | noise | 774998 | 0.2237 |
| 2024 | infrastructure | 230906 | 0.0667 |
| 2024 | sanitation | 200307 | 0.0578 |
| 2024 | public_safety | 196498 | 0.0567 |
| 2024 | water_sewer | 108519 | 0.0313 |
| 2024 | other | 74594 | 0.0215 |
| 2024 | environment | 62278 | 0.0180 |
| 2025 | parking_traffic | 957665 | 0.2703 |
| 2025 | housing | 907307 | 0.2561 |
| 2025 | noise | 804370 | 0.2270 |
| 2025 | infrastructure | 217748 | 0.0615 |
| 2025 | sanitation | 209701 | 0.0592 |
| 2025 | public_safety | 190005 | 0.0536 |
| 2025 | water_sewer | 117927 | 0.0333 |
| 2025 | other | 84328 | 0.0238 |
| 2025 | environment | 54191 | 0.0153 |

## Complaint-Type Overlap

| presence | complaint_types | rows_2015_2019 | rows_2020_2025 |
| --- | --- | --- | --- |
| 2015-2019_only | 55 | 279151.0000 | 0.0000 |
| 2020-2025_only | 68 | 0.0000 | 913121.0000 |
| both | 197 | 11350623.0000 | 18404132.0000 |

## Excluding 2020-2021 From Final-Style Training

| config | train_rows | train_positive_rows | pr_auc | precision_at_5pct | f1 | precision | recall | alert_rate | threshold | delta_vs_reference_pr_auc | delta_vs_reference_precision_at_5pct | delta_vs_reference_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference_train_through_2023 | 1077606 | 116966 | 0.3165 | 0.4180 | 0.3613 | 0.2635 | 0.5744 | 0.2411 | 0.5452 | 0.0000 | 0.0000 | 0.0000 |
| exclude_2020_2021_from_training | 832374 | 87657 | 0.3137 | 0.4122 | 0.3629 | 0.2725 | 0.5432 | 0.2205 | 0.5920 | -0.0028 | -0.0059 | 0.0017 |

## Interpretation Guardrails

- This is a sensitivity check, not a final model-selection rule. It uses one seed and the current T2 no-shortcut LightGBM candidate.
- 2020 and 2021 are visibly different in target prevalence and request volume, so the manuscript should discuss regime shift rather than treating the training period as homogeneous.
- Exact archive-boundary provenance is not retained in the modeling rows; any final paper claim about `76ig-c548` versus `erm2-nwe9` must be phrased as source-data/vintage limitation unless the pipeline is regenerated with a source-id field.
