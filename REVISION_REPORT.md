# Revision Report

Revision date: 2026-07-15

## Purpose

This revision converts the manuscript's final alert model to a prospective feature protocol. OSM and PLUTO variables from 2026 snapshots are no longer eligible for final tuning, ensemble selection, thresholds, SHAP explanations, or headline test metrics. They remain only in a clearly labelled retrospective context check.

## Files Changed

- Manuscript sources: `paper_overleaf/main.tex`, `paper_springer/main.tex`
- Table snippets: `paper_overleaf/tables/*.tex`
- New vector workflow figure: `paper_overleaf/figures/method_pipeline_overview.pdf`, `paper_springer/figures/method_pipeline_overview.pdf`
- Revised result figures: category operating points and prospective SHAP figures in both paper figure folders
- Prospective configs: `configs/features_prospective.json`, `configs/features_retrospective_*.json`
- Model config: `data/processed/model_ready/model_config.json`
- Pipeline scripts: tuning, ensemble/thresholding, table building, SHAP, extended checks, leakage audit, and figure generation scripts under `code_model/`
- Reports: `BEFORE_CONTEXT_FIX.md`, `prospective_leakage_audit.md`, `numerical_consistency_report.md`, `REVISION_REPORT.md`

## Build Command

```powershell
powershell -ExecutionPolicy Bypass -File .\paper_overleaf\build_paper.ps1 -Clean
powershell -ExecutionPolicy Bypass -File .\paper_springer\build_paper.ps1 -Clean
```

Clean build status:

- `paper_overleaf/main.pdf`: 12 pages, A4.
- `paper_springer/main.pdf`: 12 pages, A4.
- Final LaTeX logs contain no undefined citation/reference, overfull box, missing file, fatal error, or emergency stop messages.
- PDF text layer is extractable; the Overleaf PDF text layer contains 33,183 characters.
- PDF link audit: 83 annotations, 0 visible link color/border issues.

## Feature Protocol

- Final prospective feature set: `prospective_forecast_available`
- Input columns: 61
- Encoded model columns: 73
- Final model exclusions: all OSM/POI 2026 variables, all PLUTO 2026 variables, COVID-period labels, target labels, and future-window target construction columns
- Retrospective context checks:
  - `retrospective_context_osm_2026`
  - `retrospective_context_pluto_2026`
  - `retrospective_context_full_2026`

## New Prospective Model Runs

LightGBM and XGBoost were retuned on a 300,000-row stratified training sample and rebuilt on all 954,990 training rows for final ensemble scores.

- LightGBM: `lgbm_regularized_depth6_leaf31_lr003`, balanced positive weight 6.8726, selected threshold 0.5600.
- XGBoost: `xgb_shallow_depth5_lr003`, square-root positive weight 2.6216, selected threshold 0.3510.
- Final validation-selected decision strategy: ensemble LGBM(0.50)+XGB(0.50) with category-specific validation thresholds.

## Headline Metric Change

Previous full-context model with 2026 OSM/PLUTO inputs:

- Test F1 0.3839, precision 0.3071, recall 0.5117, PR-AUC 0.3365, ROC-AUC 0.7720, alert rate 0.2124.

Revised prospective model without OSM/PLUTO final inputs:

- Test F1 0.3802, precision 0.2933, recall 0.5404, PR-AUC 0.3310, ROC-AUC 0.7643, alert rate 0.2349.
- Confusion counts: TP 17,059; FP 41,106; FN 14,511; TN 174,914.

## Retrospective Context Check

Compact checks show that adding 2026 context variables changes test F1 only modestly relative to the prospective protocol:

- Prospective-only: 0.3767 F1 in the compact weather-enhanced check.
- Prospective + OSM 2026: 0.3818 F1.
- Prospective + PLUTO 2026: 0.3813 F1.
- Prospective + OSM 2026 + PLUTO 2026: 0.3819 F1.

These rows are labelled retrospective/context-only in the manuscript and are not used as headline results.

## Explainability

SHAP was regenerated using the prospective LightGBM feature set only. The audited SHAP feature list contains 73 encoded features and no OSM/PLUTO variables. Global group shares are:

- Historical/temporal: 75.15%
- Service category: 8.16%
- Calendar: 6.13%
- Weather: 5.87%
- Current demand: 2.74%
- Spatial identifiers: 1.23%
- Other low-contribution predictors: 0.72%

## Figure and Wording Polish Pass

This final polish pass did not rerun model training, model scoring, threshold tuning, or SHAP value generation. It only regenerated manuscript figures from existing outputs and synchronized the manuscript text.

- Removed the stronger preference-style wording from the model-comparison interpretation. The manuscript now describes the selected result as the validation-selected category-specific strategy and explicitly avoids claiming aggregate metric dominance.
- Rebuilt Figure 1 with a clearer vector layout from `code_model/create_method_pipeline_overview.py`. The figure now uses four concise stages, a separate dashed OSM/PLUTO retrospective branch, and the explicit phrase "excluded from prospective alerts".
- Rebuilt Figure 3 from existing SHAP group totals using `code_model/create_shap_group_importance_figure.py`. The plot now reports percentage shares and uses manuscript-facing group labels: Historical temporal, Service category, Calendar, Weather, Current demand, Spatial identifiers, and Other.
- Updated the Figure 3 caption and explanation text to match the plotted percentage shares, including the 0.72% Other group.
- Updated `code_model/run_shap_explainability.py` so any future group-importance regeneration uses the same percentage-axis format and display labels.

Final visual QA:

- Figure 1 was rendered from `paper_overleaf/figures/method_pipeline_overview.pdf` and again inside `paper_overleaf/main.pdf`; no text overlap, clipping, or OSM/PLUTO branch ambiguity remains.
- Figure 3 was rendered inside `paper_overleaf/main.pdf`; labels, percentages, caption, and manuscript text are consistent.
- `paper_overleaf/main.pdf` and `paper_springer/main.pdf` remain 12 pages.

## Figure 1 Placement and Typography Pass

This pass only changed Figure 1 placement and Figure 1 typography. It did not rerun model training, scoring, thresholding, SHAP computation, table generation, metric generation, or reference generation.

- Previous placement: Figure 1 floated to the top of page 3 after the Related Work heading had already begun.
- Revised placement: Figure 1 is forced before `\section{Related Work}` by keeping `placeins` loaded and adding `\FloatBarrier` immediately before the Related Work section.
- Final float control: `\begin{figure}[!ht] ... \end{figure}` followed by `\FloatBarrier`. A bottom-float trial was rejected because it increased the PDF to 13 pages.
- Figure 1 remains part of the Introduction: the Introduction text ends with the Figure 1 reference, then Figure 1 and its caption appear, and only then Section 2 begins.
- Caption remains immediately below Figure 1 and retains the existing math notation `\(t\)` and `\(t+1\)`.
- Source script updated: `code_model/create_method_pipeline_overview.py`.
- Regenerated vector outputs: `paper_overleaf/figures/method_pipeline_overview.pdf`, `paper_overleaf/figures/method_pipeline_overview.svg`, `paper_springer/figures/method_pipeline_overview.pdf`, and `paper_springer/figures/method_pipeline_overview.svg`.
- Standardized Figure 1 labels:
  - `NTA x week x category` -> `NTA × Week × Category`
  - `history t -> event t+1` -> `History at t → Event at t+1`
  - `2015-22 train` -> `2015–22 Train`
  - `2023 val.` -> `2023 Val.`
  - `2024-25 test` -> `2024–25 Test`
  - `t -> t+1 design` -> `t → t+1 design`
  - alert box labels now use arrow notation and consistent capitalization.
- Visual QA confirms the multiplication sign, arrows, and en dashes render correctly in the vector figure; labels do not touch borders; the retrospective branch remains readable and visually secondary.
- Final page count remains 12 pages for both PDFs.
- Final compile logs contain only benign underfull-box warnings in body/reference text; no undefined citations/references, duplicate labels, missing figures, hyperref option clash, fatal errors, or major overfull boxes were found.
- Hyperlink audit confirms 83 clickable annotations and 0 visible colored link/border issues; existing black hyperlink configuration was not changed.

## Table Float Placement Pass

This pass only changed table float placement and section-boundary barriers. It did not rerun model training, scoring, threshold tuning, ablation, sensitivity analysis, SHAP computation, table generation, metric generation, or reference generation.

- Files edited: `paper_overleaf/main.tex` and `paper_springer/main.tex`.
- Table 1 previous rendered placement: top of page 4 before the `3 Methodology` heading.
- Table 1 revised rendered placement: page 4 after `3 Methodology` and `3.1 Data and Task Formulation`, immediately after the dense-panel/model-ready-row paragraph that references Table 1.
- Table 3 previous rendered placement: top of page 6 before the `4 Results` heading.
- Table 3 revised rendered placement: page 6 after `4 Results` and `4.1 Overall Performance and Ablation Results`, after the baseline and ablation-introduction text.
- Additional checklist correction: Table 5 now renders after `4.2 Category-Level Alert Analysis`, not before that subsection heading.
- Additional layout polish: Table 2 now renders in-line with the surrounding Section 3.2 text on page 5 instead of floating alone at the page top, reducing the awkward page 5/6 split before Results.
- Additional layout polish: Table 4 now uses `[!htbp]`, the Results discussion was tightened around the validation-selected category-aware strategy, and the manuscript uses `\flushbottom` to reduce the obvious page 6 bottom imbalance while keeping both PDFs at 12 pages.
- Float options changed:
  - Table 1: `[t]` to `[!htbp]`.
  - Table 2: `[t]` to `[!htbp]`.
  - Table 3: `[t]` to `[!htbp]`.
  - Table 4: `[t]` to `[!htbp]`.
  - Table 5: `[t]` to `[!htbp]`.
- Added or retained controlled `\FloatBarrier` guards:
  - Existing Figure 1 barrier before Related Work remains unchanged.
  - Added a barrier before `\section{Methodology}`.
  - Added a barrier before `\section{Results}`.
  - Added a barrier immediately after `\subsection{Category-Level Alert Analysis}` to keep Table 5 inside Section 4.2.
  - Existing barriers before Discussion remain unchanged.
- Build command used:
  - `powershell -ExecutionPolicy Bypass -File .\paper_overleaf\build_paper.ps1 -Clean`
  - `powershell -ExecutionPolicy Bypass -File .\paper_springer\build_paper.ps1 -Clean`
- Final page count remains 12 pages for both PDFs.
- Compile-log audit found no undefined citations/references, duplicate labels, missing figures/tables, hyperref option clash, fatal errors, or major overfull boxes. Remaining warnings are benign underfull boxes in body/reference text.
- Visual QA rendered all 12 Overleaf PDF pages. Figure 1 remains before Related Work; Table 1 is inside Methodology; Table 2 is inside Methodology and page 5 is visually more balanced; Table 3 and Table 4 are inside Results 4.1; page 6 no longer has the obvious lower-page gap or a dangling split sentence before Table 4; Table 5 and Figure 2 are inside Results 4.2; Table 6 is before the Explanation Results subsection; Figures 3 and 4 remain before Discussion; no Results float appears after the Discussion heading.
- Hyperlink audit confirms 83 clickable annotations and 0 visible colored link/border issues; existing black hyperlink configuration was not changed.
- Headline metrics remain present and unchanged in the final PDF: F1 = 0.3802, precision = 0.2933, recall = 0.5404, PR-AUC = 0.3310, and ROC-AUC = 0.7643.

## Horizontal Alignment Pass

This pass corrected the manuscript-wide left/right drift caused by the Springer class loading two-sided article geometry with a 6 mm binding offset. It did not change font size, metrics, tables, figures, references, or hyperlink configuration.

- Files edited: `paper_overleaf/main.tex` and `paper_springer/main.tex`.
- Added `\geometry{bindingoffset=0mm,hcentering}` after the class-loaded geometry package is available, so review PDFs use the same centered text block on odd and even pages.
- Before this pass, extracted PDF block positions alternated between approximately `x=100.071 pt` on odd pages and `x=124.595 pt` on even pages.
- After this pass, pages 2-12 align at approximately `x=112.333 pt` on the left and `x=482.95 pt` on the right; page 1 remains title-layout-specific.
- Visual QA rendered pages 7-8, including Figure 2, Table 6, and Section 4.3; the text block, captions, and tables now share the same horizontal axis across page boundaries.
- Final page count remains 12 pages for both PDFs. Compile-log and hyperlink audits remain clean.

## Final Repository and Reproducibility Synchronization Pass

This pass synchronized the public GitHub repository and reviewer-facing documentation with the final manuscript. It did not rerun LightGBM training, XGBoost training, ensemble scoring, threshold tuning, ablation, sensitivity experiments, or SHAP computation.

- README title before: `Semantics-Aware Explainable Machine Learning for Urban Service Demand Forecasting in Smart Cities`.
- README title after: `Category-Aware Explainable Machine Learning for Urban Service Demand Forecasting in Smart Cities`.
- README pipeline description before: final pipeline was described as combining 311, NOAA, OSM, and PLUTO context.
- README pipeline description after: the final prospective alerting pipeline uses shifted 311 demand history, current complaint count, calendar timing, feature-week weather, service category, and borough identifiers.
- Final prospective inputs remain 61; encoded model columns remain 73.
- Hyperparameter tuning sample remains 300,000 stratified training rows.
- Final selected LightGBM/XGBoost configurations remain rebuilt on all 954,990 training rows before validation-only threshold selection.
- OSM/PLUTO 2026 snapshots are documented as retrospective context only: not final LightGBM inputs, not final XGBoost inputs, not ensemble-score inputs, not category-threshold inputs, not SHAP inputs, and not headline-metric inputs.
- Reproduction commands in `README.md` now use the prospective feature set, `analysis_type=prospective`, `configs/features_prospective.json`, and current `data/processed/model_results/prospective/` output paths.
- Key artifact paths in `README.md` were updated to current prospective artifacts and checked for existence.
- Retrospective context commands and artifacts are separated under "Optional retrospective context checks" and labelled as not used in final alerts.
- LightGBM subsampling was verified from code and summaries: scripts pass `subsample=0.85`, but do not pass `subsample_freq` or `bagging_freq`; the library default `subsample_freq=0` means row bagging is inactive. No `subsample_freq=1` was added.
- XGBoost subsampling remains unchanged: selected XGBoost uses `subsample=0.9`, `colsample_bytree=0.9`, `reg_lambda=5.0`, `tree_method=hist`, `eval_metric=aucpr`, and square-root positive weight 2.6216.
- `paper_springer/reference_verification_report.md` was updated from the obsolete 30-entry source-note wording to the current 32-entry cited bibliography.
- `reference_audit.md` and both `references.bib` files were synchronized.
- Reference [10] was updated to the verified ACM EC '24 proceedings version with DOI `10.1145/3670865.3673624`.
- LightGBM now uses official NeurIPS metadata, pages 3146-3154; the checklist-suggested 3149-3157 page range was not used because it does not match the official metadata.
- SHAP now uses official NeurIPS metadata, pages 4765-4774.
- Dataset and web references use official portals/project documentation with compact `Accessed 25 June 2026` notes.
- Reviewer file guidance was updated in `REVIEWER_FILES.md` to point reviewers toward the prospective artifacts and away from superseded local planning notes.
- Generated prospective paper-table Markdown was softened from "best validation-ranked" wording to "validation-selected category-aware decision strategy".

Build and QA for this pass:

- Build commands:
  - `powershell -ExecutionPolicy Bypass -File .\paper_overleaf\build_paper.ps1 -Clean`
  - `powershell -ExecutionPolicy Bypass -File .\paper_springer\build_paper.ps1 -Clean`
- Final page count: 12 pages for both `paper_overleaf/main.pdf` and `paper_springer/main.pdf`.
- Compile-log audit: no undefined citations, undefined references, duplicate labels, missing figures/tables, overfull boxes, fatal errors, emergency stops, or major layout errors. Remaining warnings are benign underfull boxes and normal `rerunfilecheck` package messages.
- Reference audit: 32 BibTeX entries, 32 unique citation keys, 32 `.bbl` items, 0 missing citation keys, 0 unused bibliography entries, 0 duplicate DOI entries, and 0 duplicate title entries.
- Hyperlink audit: 82 clickable PDF annotations in each PDF, 0 visible colored link or border issues; existing black hyperlink configuration was not changed.
- Visual QA: rendered all 12 Overleaf PDF pages to PNG and inspected a contact sheet plus page 5 and page 12 at higher resolution. Table 2 remains readable, the LightGBM subsample wording is clear, the bibliography stays within page 12, and the centered page geometry remains consistent.
- Headline metrics were not changed: F1 = 0.3802, precision = 0.2933, recall = 0.5404, PR-AUC = 0.3310, ROC-AUC = 0.7643, balanced accuracy = 0.6750, alert rate = 0.2349, TP = 17,059, FP = 41,106, FN = 14,511, TN = 174,914, positives = 31,570, alerts = 58,165.

## SIMC Review Manuscript Compliance Pass

This pass created a separate SIMC review-manuscript PDF for the submission system requirements. It did not rerun LightGBM training, XGBoost training, ensemble scoring, threshold tuning, ablation, sensitivity analysis, SHAP computation, or any data-processing/modeling code. Scientific content, author order, affiliations, figures, tables, references, GitHub URL, metrics, and confusion-matrix values were not changed.

Baseline internal PDF audit before the submission build:

- Baseline file audited: `paper_springer/main.pdf`.
- Page count: 12 pages, A4.
- PDF catalog had `/Outlines`, `/OpenAction`, and `/Names`.
- Bookmark/outline count: 14.
- Annotation count: 82.
- `/Subtype /Link` annotation count: 82.
- Standalone page-number footers were detected in extracted page text on pages 1-12.
- This internal PDF remains useful for local navigation, but it is not the SIMC upload file.

Submission mode and files created:

- Added wrapper source: `paper_springer/main_SIMC_submission.tex`.
- Added clean-build script: `paper_springer/build_SIMC_submission.ps1`.
- Added automated PDF audit: `scripts/audit_submission_pdf.py`.
- Added upload guidance: `paper_springer/SUBMISSION_README.md`.
- Output upload file: `paper_springer/main_SIMC_submission.pdf`.

Source-level compliance changes:

- `paper_springer/main.tex` now accepts `\SIMCsubmission` before `\documentclass`.
- In submission mode, `\PassOptionsToPackage{draft,bookmarks=false,pdfpagelabels=false}{hyperref}` is passed before the Springer class loads `hyperref`.
- `hyperref` is not loaded a second time.
- Non-submission hyperlink color configuration remains unchanged for the internal PDF.
- In submission mode, Springer page styles are neutralized with `\let\ps@plain\ps@empty`, `\let\ps@headings\ps@empty`, `\let\ps@myheadings\ps@empty`, and `\let\ps@titlepage\ps@empty`.
- Around `\maketitle`, submission mode applies `\pagenumbering{gobble}`, `\pagestyle{empty}`, and `\thispagestyle{empty}`.
- The custom `\email` macro prints email addresses as ordinary black text in submission mode instead of calling `\href{mailto:...}{...}`.
- No PDF post-processing, rasterization, cropping, white-box hiding, page-size change, font-size change, or margin reduction was used.

Clean-build command:

- `powershell -ExecutionPolicy Bypass -File .\paper_springer\build_SIMC_submission.ps1 -Clean`

Final automated audit for `paper_springer/main_SIMC_submission.pdf`:

- Page count: 12 pages.
- Paper size: A4, 595.276 x 841.89 pt.
- File size: 699,192 bytes.
- PDF root keys after build: `/Pages` and `/Type` only.
- `/Outlines`: absent.
- Bookmark/outline count: 0.
- `/OpenAction`: absent.
- `/Names`: absent.
- `/Dests`: absent.
- `/PageLabels`: absent.
- Total annotation count: 0.
- `/Subtype /Link` annotation count: 0.
- Standalone page-number footer detection: none on pages 1-12.
- Known running-header/footer string detection: none at page edges.
- Embedded-font audit with `pdffonts`: PASS.

Text-retention checks:

- Title remains visible: `Category-Aware Explainable Machine Learning`.
- All four authors remain visible: Tran Dai Phong Lam, Thu Le, Nguyen Quoc Hung, and Nguyen Trung Trinh.
- Emails remain visible as ordinary text: `phonglam2599@gmail.com`, `thulvm@fpt.edu.vn`, `hungtvt2222@gmail.com`, and `trinhnguyen112355@gmail.com`.
- GitHub URL remains visible as ordinary text: `https://github.com/PhongLam26/SIMC_NYC`.
- Dataset identifiers remain visible: `76ig-c548` and `erm2-nwe9`.
- DOI text remains visible, including `10.1371/journal.pone.0186314` in the rendered bibliography.
- Citation and cross-reference text remains visible, including `[1]` and `Figure 1`.
- Headline metrics remain unchanged and visible: F1 = 0.3802, precision = 0.2933, recall = 0.5404, PR-AUC = 0.3310, ROC-AUC = 0.7643.
- Confusion-matrix values remain unchanged and visible in the audit text: TP = 17,059, FP = 41,106, FN = 14,511, TN = 174,914.

Visual and log QA:

- Rendered all 12 submission PDF pages to PNG using Poppler.
- Created and inspected a 12-page contact sheet.
- Inspected page 1, page 2, page 5, page 6, page 8, page 9, page 11, and page 12 individually.
- Visual QA confirms no page numbers, running headers, or footers on the title page, body pages, declaration/reference pages, or final page.
- Visual QA confirms figures and tables are not cut, captions remain attached, URLs remain visible, and the centered page geometry is preserved.
- Compile warnings remaining: expected `Package hyperref Warning: Draft mode on.` plus benign underfull hbox/vbox warnings in body/reference text.
- Compile-log audit found no undefined citations, undefined references, fatal errors, emergency stops, LaTeX errors, duplicate-label warnings, or overfull boxes.

## Data and Code Availability Wording Pass

This pass rewrote only the `Declarations` -> `Data and code availability` paragraph for clarity and NOAA weather-source precision. It did not rerun any model, regenerate metrics, change tables, change figures, change references, or change dataset IDs.

Old paragraph:

`Data and code availability. Source data are NYC Open Data 311 historical archive 76ig-c548, NYC Open Data 311 current archive erm2-nwe9, NOAA GHCN-Daily station GHCND:USW00094728, OpenStreetMap Overpass, NYC PLUTO/MapPLUTO 64uk-42ks, and 2020 NTA boundaries 9nt8-h7nd; the local snapshot was extracted on 2026-06-25. Scripts, mappings, configurations, and evaluation code are available at https://github.com/PhongLam26/SIMC_NYC. The repository separates prospective model artifacts from retrospective OSM/PLUTO context checks; processed data release is subject to licensing, redistribution, and storage constraints.`

New paragraph:

`Data and code availability. Source data include NYC Open Data 311 archives 76ig-c548 and erm2-nwe9, 2020 NTA boundaries 9nt8-h7nd, NOAA GHCN-Daily observations from Central Park station GHCND:USW00094728 used as city-level weekly weather exposure, OpenStreetMap via Overpass, and NYC PLUTO/MapPLUTO 64uk-42ks. OSM/PLUTO snapshots for retrospective context analysis were extracted on 25 June 2026. Scripts, mappings, configurations, and evaluation code are available at https://github.com/PhongLam26/SIMC_NYC. The repository separates prospective model artifacts from retrospective OSM/PLUTO checks; processed-data release remains subject to licensing, redistribution, and storage constraints.`

Clarifications made:

- Central Park station is explicitly named as the NOAA GHCN-Daily weather source.
- Station ID `GHCND:USW00094728` remains visible.
- Weather is described as `city-level weekly weather exposure`, not NTA-level microclimate or spatially precise neighborhood weather.
- OSM/PLUTO is described as retrospective context analysis only, while the repository separation between prospective model artifacts and retrospective checks remains explicit.
- The public code URL remains visible as text.
- Processed-data release constraints remain explicit.

Files edited:

- `paper_springer/main.tex`
- `paper_overleaf/main.tex`
- `REVISION_REPORT.md`

Build and QA:

- Build commands:
  - `powershell -ExecutionPolicy Bypass -File .\paper_springer\build_paper.ps1 -Clean`
  - `powershell -ExecutionPolicy Bypass -File .\paper_overleaf\build_paper.ps1 -Clean`
  - `powershell -ExecutionPolicy Bypass -File .\paper_springer\build_SIMC_submission.ps1 -Clean`
- Final page count: 12 pages for `paper_springer/main.pdf`, `paper_overleaf/main.pdf`, and `paper_springer/main_SIMC_submission.pdf`.
- Text audit confirms the Central Park station ID, `city-level weekly weather exposure`, all dataset IDs, GitHub URL, OSM/PLUTO retrospective wording, and processed-data constraints remain visible in all three PDFs.
- Compile-log audit found no undefined citations, undefined references, overfull boxes, fatal errors, emergency stops, duplicate-label warnings, or LaTeX errors.
- Submission PDF audit remains PASS: 0 bookmarks/outlines, 0 link annotations, 0 page numbers, 0 running headers, and 0 footers.
- Visual QA rendered the Declarations page from `paper_springer/main_SIMC_submission.pdf`; the paragraph remains readable, URL and dataset IDs do not overflow, and no header/footer/page number appears.
- Headline metrics remain unchanged: F1 = 0.3802, precision = 0.2933, recall = 0.5404, PR-AUC = 0.3310, ROC-AUC = 0.7643.
- No model, threshold, SHAP, table, figure, reference, or dataset-ID generation was run or changed.

## Audits

- `prospective_leakage_audit.md`: PASS, 10/10 checks.
- `numerical_consistency_report.md`: confirms confusion-matrix reconstruction, category sums, lift-at-k counts, and SHAP feature/group totals.
- `BEFORE_CONTEXT_FIX.md`: records the previous full-context baseline and why it was superseded.

## Reviewer-Facing Files

Reviewer-facing manuscript package:

- `paper_overleaf/main.tex`
- `paper_overleaf/references.bib`
- `paper_overleaf/figures/`
- `paper_overleaf/tables/`
- `paper_overleaf/main.pdf`
- `REVISION_REPORT.md`
- `prospective_leakage_audit.md`
- `numerical_consistency_report.md`
- `BEFORE_CONTEXT_FIX.md`

Internal/generated files that reviewers do not need unless reproducing experiments:

- `data/processed/model_results/prospective/`
- `data/processed/model_results/retrospective_context/`
- intermediate LaTeX logs and aux files

## Major Methodological Rebuild Checkpoint

This checkpoint starts the larger methodological revision requested in `Danh_sach_can_sua_SIMC_NYC.docx` and the attached pasted checklist. It does not claim that the major revision is complete.

Git and artifact protection:

- Baseline commit: `7ccf58dd55dd2c24d6f0d5a2a3adcfa8c1bf0a6a`
- Baseline tag: `pre-major-revision-submission-2026-07-16`
- Revision branch: `major-revision-methodological-rebuild`
- Archived submitted PDFs: `archive/submitted_2026-07-16/`
- `paper_springer/main_SIMC_submission.pdf` was not rebuilt during this checkpoint.

New revision-control documents:

- `MAJOR_REVISION_BASELINE.md`
- `major_revision_workspace_audit.md`
- `pre_registered_revision_selection_rule.md`
- `archive/submitted_2026-07-16/README.md`

Initial evidence generated:

- Added reproducible audit script: `scripts/major_revision_initial_audit.py`
- Generated initial sparse-panel diagnostics under `data/processed/model_results/major_revision/initial_audit/`

Initial sparse-panel findings for the current submitted target:

- Held-out test rows: 247,590
- Held-out positives: 31,570
- Positive test rows with `rolling_8w_mean < 1`: 5,670, or 17.96 percent
- Positive test rows with `rolling_8w_mean < 2`: 7,469, or 23.66 percent
- Positive rows with `target_next_week_count >= 4`: 26,154, or 82.84 percent
- Historical-volume decile performance ranges from F1 = 0.1935 in D1 to F1 = 0.4290 in D10
- Current score top-5-percent precision is 0.4460

Reviewer checklist status at this checkpoint:

- P1-1 target/input shortcut: OPEN. Risk confirmed because current final features include `rolling_8w_mean`, `rolling_8w_std`, and `ratio_to_8w_mean`, and current SHAP ranks formula-aligned 8-week features highly.
- P1-2 sparse/zero-heavy panel: PARTIAL. Initial diagnostic generated; sparse-aware target alternatives have not yet been evaluated.
- P1-3 title/category-aware/ensemble claim: OPEN. Final title and final decision model must wait for revised validation evidence.
- OSM/PLUTO look-ahead risk: OPEN for manuscript rewrite. Existing artifacts are mapped; revised main paper should remove OSM/PLUTO from the main pipeline claims.
- Count-model baselines: OPEN.
- Five requested seeds 42, 123, 2026, 3407, 7777: OPEN. Existing seed stability only covers 7, 42, and 123.
- Rolling-origin evaluation: OPEN.
- Calibration and uncertainty intervals: OPEN.
- Final manuscript rewrite and new submission PDF: NOT STARTED in this checkpoint.

Guardrail: do not mark the major methodological revision PASS until the requested experiment set is actually run and the manuscript is rewritten from the revised evidence.

## Major Methodological Rebuild - Model Audit Pass 1

This pass adds actual model evidence for the highest-priority reviewer issues P1-1 and P1-5. It does not complete the full major revision.

New script and outputs:

- Script: `scripts/major_revision_model_audits.py`
- Target-shortcut report: `target_shortcut_audit.md`
- Count baseline report: `count_model_baseline_report.md`
- Detailed outputs: `data/processed/model_results/major_revision/model_audits/`

Target-shortcut evidence:

- Current LightGBM prospective test metrics: F1 = 0.3800, PR-AUC = 0.3301, precision = 0.2986, recall = 0.5222, precision@5% = 0.4451.
- Removing `rolling_8w_mean`, `rolling_8w_std`, `rolling_8w_sum`, and `ratio_to_8w_mean` gives test F1 = 0.3591, PR-AUC = 0.3097, precision = 0.2717, recall = 0.5293, precision@5% = 0.4220.
- The no-shortcut delta is F1 -0.0209 and PR-AUC -0.0204 relative to the current feature set.
- Current gain importance confirms formula-aligned predictors are prominent: `rolling_8w_std` rank 1 and `ratio_to_8w_mean` rank 5.
- No-shortcut SHAP shifts toward `ratio_to_12w_mean`, `rolling_4w_std`, `rolling_12w_std`, `lag_8w_count`, and `lag_12w_count`.

Count-model baseline evidence:

- Stable PoissonRegressor baselines were run with log-transformed count/history predictors, `alpha = 10.0`, and `max_iter = 500`.
- The no-NTA Poisson model converged in 106 iterations; the NTA fixed-effect Poisson model converged in 445 iterations.
- Poisson no-NTA test PR-AUC = 0.1400, formula-threshold F1 = 0.1460, count MAE = 8.6680.
- Poisson + NTA fixed effects test PR-AUC = 0.1400, formula-threshold F1 = 0.1492, count MAE = 8.6839.
- Adding NTA fixed effects slightly improves Poisson deviance but does not materially improve event ranking in this first baseline.

Reviewer status updates:

- P1-1 target/input shortcut: PARTIAL. Evidence now shows material performance loss and SHAP/importance shift after removing formula-aligned 8-week predictors. Still requires rolling-origin validation and final-model SHAP.
- P1-5 count/spatial baseline: PARTIAL. Full-data Poisson and NTA fixed-effect Poisson baselines are complete. Negative Binomial or hurdle/overdispersed count baseline remains open.
- P1-3 ensemble/title: OPEN. These results strengthen the case for simplifying the story, but final title/model choice still needs rolling-origin and uncertainty evidence.
- P1-4 uncertainty: OPEN.
- P1-6 full ablation: OPEN.

## Major Methodological Rebuild - Target Definition Pass 1

This pass adds a first sparse-aware target-definition evaluation for P1-2. It does not freeze the final target, because the checklist requires rolling-origin validation and uncertainty intervals before the manuscript is rewritten around a new target.

New script and outputs:

- Script: `scripts/major_revision_target_selection.py`
- Reviewer-facing report: `target_definition_selection_report.md`
- Detailed outputs: `data/processed/model_results/major_revision/target_selection/`

Target definitions evaluated with the no-shortcut LightGBM feature set:

- T0 current reference: current stored abnormal-increase target.
- T1 minimum future count: T0 and `target_next_week_count >= 2`.
- T2 stricter minimum future count: T0 and `target_next_week_count >= 3`.
- T3 baseline-activity eligibility: evaluates only rows with `rolling_8w_mean >= 1`, target remains T0 within eligible rows.

Test-period target-composition evidence:

- T0 has 31,570 positives; 5,670 positives, or 17.96 percent, have `rolling_8w_mean < 1`.
- T1 has 28,863 positives and reduces the low-baseline-positive share to 10.27 percent.
- T2 has 27,277 positives and reduces the low-baseline-positive share to 5.10 percent.
- T3 has 25,900 positives after excluding 58,167 test rows; it removes low-baseline positives by design and is not directly comparable to T0/T1/T2.

Validation evidence for target selection:

- T0 validation PR-AUC = 0.2936, precision@5% = 0.3993, F1 = 0.3511.
- T1 validation PR-AUC = 0.2956, precision@5% = 0.3963, F1 = 0.3504.
- T2 validation PR-AUC = 0.3006, precision@5% = 0.4003, F1 = 0.3547.
- T3 validation PR-AUC = 0.3079, precision@5% = 0.4204, F1 = 0.3606, but on a restricted risk set.

Reviewer status updates:

- P1-2 sparse/zero-heavy panel: PARTIAL. Sparse-aware T1/T2/T3 targets are now evaluated on the current train/validation/test split with volume-decile diagnostics. Final target selection remains open until rolling-origin evidence, uncertainty, and the equity/coverage implication of T3 are evaluated.
- P1-1 target/input shortcut: still PARTIAL. This pass uses no-shortcut features but still needs rolling-origin and final-model SHAP.
- P2-6 precision@k/capacity: PARTIAL. Precision@1% and precision@5% are now recorded for target definitions, but workload and final headline metric are not frozen.
- T4 hurdle target: OPEN.

## Major Methodological Rebuild - Rolling-Origin Pass 1

This pass adds expanding-window backtesting for P2-1 and year-specific drift evidence for 2024 versus 2025. It does not freeze the final model or target because multiple seeds, calibration, and uncertainty intervals are still required.

New script and outputs:

- Script: `scripts/major_revision_rolling_origin.py`
- Reviewer-facing report: `rolling_origin_report.md`
- Detailed outputs: `data/processed/model_results/major_revision/backtests/`

Fold design:

- Fold 1: train through 2019, validate 2020, test 2021.
- Fold 2: train through 2020, validate 2021, test 2022.
- Fold 3: train through 2021, validate 2022, test 2023.
- Fold 4: train through 2022, validate 2023, test 2024.
- Final-style fold: train through 2023, validate 2024, test 2025.

All folds use the no-shortcut LightGBM feature set. Thresholds are selected on each validation year and applied unchanged to that fold's test year.

Mean test-year evidence across five folds:

- T0 current reference: PR-AUC = 0.2959 +/- 0.0153, precision@5% = 0.4062 +/- 0.0219, F1 = 0.3520 +/- 0.0093.
- T1 minimum count 2: PR-AUC = 0.2991 +/- 0.0183, precision@5% = 0.4060 +/- 0.0251, F1 = 0.3550 +/- 0.0133.
- T2 minimum count 3: PR-AUC = 0.2997 +/- 0.0174, precision@5% = 0.4015 +/- 0.0234, F1 = 0.3583 +/- 0.0138.
- T3 `rolling_8w_mean >= 1` eligible rows: PR-AUC = 0.3064 +/- 0.0185, precision@5% = 0.4199 +/- 0.0261, F1 = 0.3631 +/- 0.0119, but this is a restricted risk set and not directly comparable to T0/T1/T2.

2024 versus 2025 test-year evidence:

- T0 PR-AUC improves from 0.3023 in 2024 to 0.3120 in 2025; precision@5% improves from 0.4100 to 0.4236.
- T2 PR-AUC improves from 0.3082 in 2024 to 0.3165 in 2025; precision@5% improves from 0.4077 to 0.4180.
- T3 PR-AUC improves from 0.3152 in 2024 to 0.3261 in 2025 on eligible rows.

Reviewer status updates:

- P2-1 rolling-origin evaluation: PARTIAL. Five expanding-window folds are complete and 2024/2025 are reported separately. Still requires multiple seeds, calibration drift, category drift, and COVID-exclusion sensitivity.
- P1-2 target selection: still PARTIAL. T1/T2 improve construct validity by removing one-call positives; T2 gives the strongest comparable validation/backtest PR-AUC among T0/T1/T2, but final target selection remains open until uncertainty and calibration are added.
- P1-4 uncertainty: still OPEN. The `+/-` values above are fold standard deviations, not bootstrap confidence intervals.
