# Revision Report

Revision status: final current package, 2026-07-16.

This report summarizes the current SIMC_NYC paper package after the major methodological rebuild. It intentionally describes the package as it stands now; historical intermediate notes are not used as reviewer-facing status.

## Reviewer-Facing Manuscript Package

- Upload PDF: `paper_springer/main_SIMC_submission.pdf`.
- Source package: `paper_springer/main_SIMC_submission.tex`, `paper_springer/main.tex`, `paper_springer/references.bib`, `paper_springer/main.bbl`, and referenced files under `paper_springer/figures/`.
- Overleaf working copy: `paper_overleaf/main.tex` and `paper_overleaf/main.pdf`.
- Reviewer file guide: `REVIEWER_FILE_MANIFEST.md`.

## Current Manuscript

- Title: `Explainable Early Warning for Next-Week Abnormal Reported 311 Demand`.
- Task: binary early warning for next-week abnormal reported 311 demand over NTA-week-service-category rows.
- Analytical unit: NTA x week x service category.
- Target: T2, a shifted abnormal next-week target requiring both an 8-week exceedance and at least 3 next-week requests.
- Final score model: single no-shortcut LightGBM selected by validation/backtest evidence, not by held-out 2025 test optimization.
- Final calibration: Platt calibration fit on validation only and applied unchanged to the held-out test year.
- Final feature scope: historical/request, calendar, borough, and feature-week observed weather context. Contemporary OSM/PLUTO 2026 snapshot variables are not used for final tuning, thresholds, headline metrics, or SHAP explanations.

## Current Headline Evidence

- Held-out 2025 final LightGBM PR-AUC: 0.3165 with 95% CI [0.3080, 0.3252].
- Held-out 2025 final LightGBM F1: 0.3613 with 95% CI [0.3551, 0.3676].
- Precision@1%: 0.5697 with 95% CI [0.5428, 0.6023].
- Precision@5%: 0.4180 with 95% CI [0.4035, 0.4324].
- Platt calibration reduces Brier score from 0.1676 to 0.0869 and ECE from 0.2493 to 0.0057.
- Five rolling-origin test folds report 2024 and 2025 separately.
- Five stochastic seeds are complete for the final-style model and key ablation rows.

## Baselines and Ablations

- Count baselines include Poisson GLM, Poisson GLM with NTA fixed effects, HGB Poisson count, and hurdle-style HGB count evidence.
- Paired NTA-category bootstrap shows the final LightGBM beats the HGB Poisson and hurdle count baselines for PR-AUC, precision@5%, and F1; all six reported tree-vs-count intervals exclude zero.
- Full-row ablations show history/current-count features carry most signal, while calendar and weather add smaller increments.
- The NTA fixed-effect diagnostic has a tiny PR-AUC gain over the borough model, but precision@5% and F1 paired intervals include zero, so the manuscript keeps the simpler borough final claim.

## Explainability

- SHAP is regenerated for the same fitted LightGBM score model used for final evidence.
- Formula-aligned 8-week target-construction fields are not final predictors.
- Main explanation outputs include the SHAP beeswarm and local TP/FP/FN case studies.

## Checklist Closure

- `major_revision_issue_mapping_38.md` maps all 38 requested source issues to status, action, evidence, and manuscript location.
- `major_revision_completion_checklist.md` records all current completion items as PASS.
- `major_revision_final_required_answers.md` answers the requested final audit questions from current repository artifacts.
- External-data-dependent items are explicitly handled as future work rather than claimed as completed analyses: historical source vintages, historical forecast-weather archives, spatial weather grids, ACS/Census socioeconomic fairness, processed-panel publication infrastructure, agency staffing/cost functions, operational intervention outcomes, and cross-city replication.

## Submission Audit

The current Springer/SIMC upload PDF was rebuilt and audited after the final manuscript edits.

- File: `paper_springer/main_SIMC_submission.pdf`.
- Pages: 13 A4 pages.
- PDF compliance audit: no bookmarks/outlines, no link annotations, no page labels, no page-number footer, no running headers/footers, required visible text retained, and all fonts embedded.
- Visual PDF QA: rendered all 13 pages and inspected the contact sheet plus pages 4-7 where prior horizontal-margin drift occurred; no margin drift, clipping, overlap, or table spillover was observed.
- Final LaTeX log audit: no undefined references/citations, no LaTeX fatal errors, and no overfull boxes.

## Data and Code Availability

The manuscript now includes the public repository link supplied by the author:

`https://github.com/PhongLam26/SIMC_NYC.git`

Scripts, mappings, configurations, and evaluation code are available in that repository. Source data are cited by public dataset IDs/DOIs in the manuscript; processed-data publication remains a release-policy/storage item outside this coding revision.
