# Revision Report

Revision status: final current package, 2026-07-16.

This report summarizes the current SIMC_NYC paper package after the major methodological rebuild. It intentionally describes the package as it stands now; historical intermediate notes are not used as submission status.

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

- Held-out 2025 final LightGBM PR-AUC: 0.317 with 95% CI [0.308, 0.325].
- Held-out 2025 final LightGBM F1: 0.361 with 95% CI [0.355, 0.368].
- Precision@1%: 0.570 with 95% CI [0.543, 0.602].
- Precision@5%: 0.418 with 95% CI [0.404, 0.432].
- Platt calibration reduces Brier score from 0.168 to 0.087 and ECE from 0.249 to 0.006.
- Five rolling-origin test folds report 2024 and 2025 separately.
- Five stochastic seeds are complete for the selected model and key ablation rows.

## Baselines and Ablations

- Count baselines include Poisson GLM, Poisson GLM with NTA fixed effects, HGB Poisson count, and hurdle-style HGB count evidence.
- Paired NTA-category bootstrap shows the final LightGBM beats the HGB Poisson and hurdle count baselines for PR-AUC, precision@5%, and F1; all six reported tree-vs-count intervals exclude zero.
- Full-row ablations show history/current-count features carry most signal, while calendar and weather add smaller increments.
- The NTA fixed-effect diagnostic has a tiny PR-AUC gain over the borough model, but precision@5% and F1 paired intervals include zero, so the manuscript keeps the simpler borough final claim.

## Explainability

- SHAP is regenerated for the same fitted LightGBM score model used for final evidence.
- TreeSHAP local explanations are on the LightGBM raw-margin scale before Platt calibration, not on the calibrated-probability scale.
- The local TP/FP/FN text reports both uncalibrated LightGBM probabilities and Platt-calibrated probabilities.
- Formula-aligned 8-week target-construction fields are not final predictors.
- Main explanation outputs include the SHAP beeswarm and local TP/FP/FN case studies.

## Final Logic, Calibration-Scale, and Presentation Pass

- Period logic: the 27,277 positives among 247,590 rows are now described as the 2024--2025 two-year diagnostic period, avoiding the earlier possible reading that they were held-out 2025-only counts.
- Main-manuscript phrasing: revision-letter language was removed from the paper text.
- Calibration scale: Platt calibration is described as validation-only and monotone for the audited range; the paper does not claim PR-AUC gains from calibration.
- Local SHAP cases: audited values are TP 0.975 uncalibrated / 0.625 calibrated, FP 0.956 / 0.603, and FN 0.316 / 0.061, with calibrated threshold 0.167.
- Table 5: the category-prevalence column now uses `Event prevalence`.
- Weather decision: weather remains only as a documented contextual covariate. It does not improve validation PR-AUC in the isolated ablation, although validation P@5% and F1 increase slightly; the manuscript does not justify it by held-out test performance or SHAP importance.
- Table 3: every baseline/comparator row now reports P@5%, precision, recall, and F1 under its stated decision rule.
- NTA wording: tree-model diagnostic wording now uses NTA indicators; GLM count baselines retain fixed-effect terminology.
- Rolling-origin evidence: 2024 and 2025 held-out diagnostics are reported separately in the main text.
- Figure 1: labels and caption distinguish the original T0 target from the final T2 minimum-count target.
- Figure presentation: reliability, beeswarm, and local SHAP figures were regenerated with tighter crops and embedded non-Type-3 fonts.
- Rounding: main-manuscript metrics are rounded to three decimals, while CSV/report artifacts retain full precision.
- Consistency report: `final_logic_numerical_consistency_report.md` records the final numerical and layout checks.

## Checklist Closure

- `major_revision_issue_mapping_38.md` maps all 38 requested source issues to status, action, evidence, and manuscript location.
- `major_revision_completion_checklist.md` records all current completion items as PASS.
- `major_revision_final_required_answers.md` answers the requested final audit questions from current repository artifacts.
- External-data-dependent items are explicitly handled as future work rather than claimed as completed analyses: historical source vintages, historical forecast-weather archives, spatial weather grids, ACS/Census socioeconomic fairness, processed-panel publication infrastructure, agency staffing/cost functions, operational intervention outcomes, and cross-city replication.

## Archived 13-Page Submission Audit

The current Springer/SIMC upload PDF was rebuilt and audited after the final manuscript edits.

- File: `paper_springer/main_SIMC_submission.pdf`.
- Pages: 13 A4 pages. This is the archived pre-page-limit checkpoint, not the current upload artifact.
- PDF compliance audit: no bookmarks/outlines, no link annotations, no page labels, no page-number footer, no running headers/footers, required visible text retained, all fonts embedded, and no Type 3 fonts.
- Visual PDF QA: rendered all 13 pages and inspected the contact sheet plus the dense methods and results pages where tables and SHAP figures are concentrated; no margin drift, clipping, overlap, or table spillover was observed.
- Final LaTeX log audit: no undefined references/citations, no LaTeX fatal errors, and no overfull boxes.

## Final Same-Target Baseline and Repository Evidence Pass

- Poisson GLM and Poisson GLM plus NTA fixed effects were rerun on the final chronological protocol: train through 2023, validation 2024, held-out test 2025. Both use the T2 event label and the same 122,616 test rows with 13,562 positives as the HGB, hurdle, and LightGBM rows.
- Poisson convergence is documented directly in `final_comparability_workspace_audit.md`: 156 of 500 iterations without NTA fixed effects and 199 of 500 with NTA fixed effects.
- Table 3 now contains only same-target T2 rows; `original-threshold count diagnostics` language was removed.
- Detailed diagnostic artifacts remain in the public repository; they are not a separate submission dependency.
- `final_same_target_numerical_consistency_report.md` reconstructs the shared population and table metrics; `final_comparability_workspace_audit.md` records target, split, decision rule, paths, and convergence.
- Reference decision: the repository contains no venue or advisor instruction requiring 30 references. The main bibliography remains the verified, directly cited set rather than being padded with uncited entries.

## Git Evidence

- Final logic and presentation pass commit: `df501d5cdf67d6d2485b8b5e17aac9e97c5042bd`.
- Clean rebuilt manuscript-PDF commit: `f8c928e5fbbe48af809aa3fbc08ced8a60652a47`.
- Branch: `major-revision-methodological-rebuild`.
- Remote: `https://github.com/PhongLam26/SIMC_NYC.git`.
- The exact final branch HEAD is reported in the delivery response because a Git commit cannot contain its own SHA.

## Data and Code Availability

The manuscript now includes the public repository link supplied by the author:

`https://github.com/PhongLam26/SIMC_NYC.git`

Scripts, mappings, configurations, and evaluation code are available in that repository. Source data are cited by public dataset IDs/DOIs in the manuscript; processed-data publication remains a release-policy/storage item outside this coding revision.

## Single-PDF 12-Page Compliance Pass

The formal regular-manuscript range is 10--12 pages. The archived baseline was 13 pages; the current self-contained submission PDF is 11 pages after a clean build. The manuscript font, A4 page size, and template margins were not reduced to meet the limit.

- Final upload: `paper_springer/main_SIMC_submission.pdf` only. There is no separate supplementary PDF or reviewer-facing supplementary dependency.
- P1 evidence retained in the main manuscript: the target-feature separation block; T2 minimum-count target and sparse-cell reduction from 17.96\% to 5.10\%; the D1--D10 historical-volume table with prevalence, PR-AUC, F1, and P@5\%; same-target count baselines; full ablation; five-seed stability; and NTA-category cluster-bootstrap intervals.
- Table numbering is preserved for review: Table 2 is the selected-model metrics table, Table 3 is the held-out-2025 same-target baseline table, and Table 4 is the full-training ablation table. The new D1--D10 evidence is Table 5.
- Explainability is compact and self-contained: the SHAP beeswarm, a TP/FP/FN local-case table, and the severe-FN waterfall remain. The redundant TP and FP waterfall figures are not part of the manuscript.
- Mapping composition and detailed diagnostics are repository CSV/code artifacts. The data-and-code statement gives the public repository URL and does not promise a future code release.
- Validation-only thresholding, Platt calibration, leakage-controlled temporal construction, target/feature separation, class-weight definition, and the five-seed/cluster-bootstrap uncertainty wording are retained in the final main text.

### Final Build and Audit

- Springer and Overleaf sources were both clean-built with `powershell -ExecutionPolicy Bypass -File .\build_paper.ps1 -Clean`; each produced an 11-page A4 `main.pdf`.
- The submission wrapper was clean-built with `powershell -ExecutionPolicy Bypass -File .\build_SIMC_submission.ps1 -Clean` and audited with `python scripts\audit_submission_pdf.py paper_springer\main_SIMC_submission.pdf --expected-pages 11`.
- The audit passed: no outlines/bookmarks, `/OpenAction`, `/Names`, `/Dests`, `/PageLabels`, links, standalone page number, running header/footer, unembedded font, or Type 3 font. Required title, authors, emails, repository URL, DOI/citation text, dataset IDs, and headline metrics are visible as text.
- Render QA covered all 11 final pages. Tables and figures are readable and uncropped; no page has a detached heading or an excessively blank final reference tail.
- Final upload PDF SHA-256: `5160CBB0664C4C2AC3983289BEEE811B7EA9393A4096D35B656903D41FB5BC1B`.
- Implementation commit: `b065420` (`Finalize self-contained 11-page SIMC manuscript`). The follow-up documentation commit records this SHA without changing the self-contained PDF.
