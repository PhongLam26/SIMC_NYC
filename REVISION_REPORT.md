# Revision Report

Revision date: 2026-07-15

Commit containing this revision: see the latest Git commit after this report update.

## Files Changed

- `paper_springer/main.tex`
- `paper_springer/references.bib`
- `paper_springer/main.bbl`
- `paper_springer/main.pdf`
- `paper_springer/figures/method_pipeline_overview.tex`
- `paper_springer/figures/category_operating_points.pdf`
- `paper_overleaf/main.tex`
- `paper_overleaf/references.bib`
- `paper_overleaf/main.bbl`
- `paper_overleaf/main.pdf`
- `paper_overleaf/figures/method_pipeline_overview.tex`
- `paper_overleaf/figures/category_operating_points.pdf`
- `code_model/create_category_operating_figure.py`
- `reference_audit.md`
- `REVISION_REPORT.md`

## Section Structure

Before revision:

- Introduction
- Related Work
- Data and Task Formulation
- Methodology
- Experimental Setup
- Results
- Explainability Analysis
- Discussion
- Conclusion

After revision:

- 1 Introduction
- 2 Related Work
- 3 Methodology
- 3.1 Data and Task Formulation
- 3.2 Predictive Models and Category-Aware Decision Layer
- 3.3 Evaluation and Explainability Protocol
- 4 Results
- 4.1 Overall Performance and Ablation Results
- 4.2 Category-Level Alert Analysis
- 4.3 Explanation Results
- 5 Discussion
- 5.1 Operational and Methodological Implications
- 5.2 Limitations and Future Work
- 6 Conclusion

## Table and Figure Sources

| Artifact | Source of values / generation path | Notes |
|---|---|---|
| Table 1 Dataset construction summary | `data/processed/_aggregate_summaries/aggregate_weekly_311_summary.csv`, `data/processed/_feature_summaries/build_311_temporal_features_summary.json`, `data/processed/_final_summaries/build_final_dataset_summary.json`, `data/processed/_final_summaries/build_final_dataset_train_ready_summary.csv` | Panel rows, model-ready rows, NTA/week/category counts, matched 311 counts, and final feature counts. |
| Table 2 Tuning protocol | `data/processed/model_results/paper_tables_target_week_fulltrain/paper_table_06_hyperparameter_best_models.csv`, `paper_result_tables_run_summary.json`, and model scripts in `code_model/` | Compact reproducibility summary of LightGBM/XGBoost candidates, selected settings, and validation-only thresholds. |
| Table 3 Final decision-layer comparison | `data/processed/model_results/paper_tables_target_week_fulltrain/paper_table_01_final_model_comparison.csv` | Compacted for readability; full output remains in CSV. |
| Table 4 Category performance | `data/processed/model_results/paper_tables_target_week_fulltrain/paper_table_04_category_performance.csv` | Positives, alerts, PR-AUC, F1, precision, and recall are copied from generated model output. |
| Workflow figure | `paper_springer/figures/method_pipeline_overview.tex`, copied to `paper_overleaf/figures/` | TikZ vector figure revised to five-stage pipeline. |
| Category operating-point figure | Script `code_model/create_category_operating_figure.py`; input `paper_table_04_category_performance.csv`; outputs in both paper figure folders | Generated from real category performance output; PDF font regenerated with TrueType embedding. |
| SHAP group figure | Existing `figures/shap_group_importance.png`; values checked against `data/processed/model_results/shap_explainability_lgbm_fulltrain/shap_group_importance_paper_7groups.csv` | Text now reports feature counts by SHAP group and notes summed-group caveat. |

## Key Content Fixes

- PASS: Data and code availability now includes `https://github.com/PhongLam26/SIMC_NYC`.
- PASS: Thu Le email is `thulvm@fpt.edu.vn`.
- PASS: Broad "leakage-safe" wording was replaced with narrower leakage-controlled temporal wording, while target/feature shifting is described precisely.
- PASS: OSM/PLUTO are described as static 2026 snapshots with snapshot/look-ahead limitations.
- PASS: Weather covariates are described as feature-week \(t\) exposure, not target-week \(t+1\) weather.
- PASS: Dataset row wording now states that 30,654 rows are excluded relative to the full dense panel, leaving 1,325,196 model-ready rows.
- PASS: Table 3 is framed as validation-selected and operationally preferred, not best overall.
- PASS: Table 4 now explains positives, alerts, threshold dependence, score distributions, precision-recall operating points, and the Noise/Public Safety example.
- PASS: Discussion merges operational and methodological implications; limitations and future work are linked in one subsection.

## Abstract and References

- Abstract word count: 176.
- Bibliography entries: 31.
- Cited bibliography entries: 31.
- Uncited bibliography entries: 0.
- Missing bibliography entries for citations: 0.
- Reference audit file: `reference_audit.md`.

## Numerical Consistency Checks

Command used:

```powershell
@'
import pandas as pd
from pathlib import Path
cat = Path('data/processed/model_results/paper_tables_target_week_fulltrain/paper_table_04_category_performance.csv')
cm = Path('data/processed/model_results/paper_tables_target_week_fulltrain/paper_table_03_final_model_confusion.csv')
catdf = pd.read_csv(cat)
cmdf = pd.read_csv(cm)
test = cmdf[cmdf['split'].str.lower().eq('test')].iloc[0]
print(catdf['positive_rows'].sum(), test['positive_rows'])
print(catdf['predicted_positive_rows'].sum(), test['predicted_positive_rows'])
print(catdf[['tp','fp','fn','tn']].sum().to_dict())
print(test[['tp','fp','fn','tn']].to_dict())
print((catdf['precision'] - catdf['tp'] / catdf['predicted_positive_rows']).abs().max())
'@ | python -
```

Results:

- PASS: Category positive rows sum to 31,570, matching test positives.
- PASS: Category alerts sum to 52,594, matching test predicted positives.
- PASS: Category TP/FP/FN/TN sums match the test confusion matrix: TP 16,154; FP 36,440; FN 15,416; TN 179,580.
- PASS: Maximum precision rounding error is 0.000039.
- PASS: Noise/Public Safety interpretation is consistent with output: Noise positives 3,982, alerts 5,729, precision 0.3880, recall 0.5583; Public Safety positives 3,536, alerts 6,482, precision 0.3166, recall 0.5803.

## Build and PDF QA

Build command for both paper folders:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_paper.ps1
```

Final build results:

- PASS: `paper_springer/main.pdf` builds successfully.
- PASS: `paper_overleaf/main.pdf` builds successfully.
- PASS: Both PDFs are 10 pages.
- PASS: Final logs contain no unresolved references, undefined citations, duplicate labels, fatal errors, missing figures, or overfull boxes.
- Remaining warnings: benign underfull box messages from normal LaTeX page breaking and bibliography URL wrapping.
- PASS: `pdffonts` reports no Type 3 fonts.
- PASS: `pdftotext` confirms text layer contains `thulvm@fpt.edu.vn`, `https://github.com/PhongLam26/SIMC_NYC`, and the data/code availability statement.
- PASS: Rendered pages inspected: title page, methodology/tables, results/tables, figure page, and references page.

## Final Checklist

- PASS: Abstract concise and under 190 words.
- PASS: Research questions written as prose.
- PASS: Data inside Methodology.
- PASS: Related Work condensed.
- PASS: Workflow redesigned.
- PASS: Contribution explicit.
- PASS: Table 4 explained.
- PASS: Discussion regrouped.
- PASS: Limitations linked to future work.
- PASS: 30+ verified references.
- PASS: No fabricated citations.
- PASS: No `This paper`, `This research`, `This study`, `RQ1`, `RQ2`, or `RQ3` in the manuscript body.
- PASS: Tables and figures match generated data outputs.
- PASS: PDF <= 12 pages.
- PASS: Clean build.
