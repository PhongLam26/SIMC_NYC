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

## Major Methodological Rebuild - Calibration Pass 1

This pass adds fold-specific probability calibration for P2-4. It does not freeze the final probability model, because target/model selection, bootstrap intervals, and the final manuscript tables remain open.

New script and outputs:

- Script: `scripts/major_revision_calibration.py`
- Reviewer-facing report: `calibration_report.md`
- Detailed outputs: `data/processed/model_results/major_revision/calibration/`
- Reliability diagram: `data/processed/model_results/major_revision/calibration/reliability_diagram.pdf`

Calibration protocol:

- All folds use the no-shortcut LightGBM feature set.
- Platt and isotonic calibrators are fit only on each fold's validation year.
- Test-year metrics are evaluated with the fold-specific validation-fitted calibrator.
- Metrics include Brier score, log loss, ECE, MCE, calibration slope/intercept, ranking metrics, and threshold metrics.

Mean test-year calibration evidence across five rolling-origin folds:

- T0 current reference: uncalibrated Brier = 0.1955 and ECE = 0.2820; Platt Brier = 0.1023 and ECE = 0.0083; isotonic Brier = 0.1023 and ECE = 0.0078.
- T2 minimum count 3: uncalibrated Brier = 0.1852 and ECE = 0.2701; Platt Brier = 0.0877 and ECE = 0.0085; isotonic Brier = 0.0876 and ECE = 0.0070.
- T3 eligible rows: uncalibrated Brier = 0.1912 and ECE = 0.2644; Platt Brier = 0.1081 and ECE = 0.0104; isotonic Brier = 0.1082 and ECE = 0.0097.

Final-style 2025 test evidence:

- T0: uncalibrated Brier = 0.1742 and ECE = 0.2522; Platt Brier = 0.1011 and ECE = 0.0059; isotonic Brier = 0.1011 and ECE = 0.0056.
- T2: uncalibrated Brier = 0.1676 and ECE = 0.2493; Platt Brier = 0.0869 and ECE = 0.0057; isotonic Brier = 0.0868 and ECE = 0.0044.

Reviewer status updates:

- P2-4 calibration: PARTIAL. Platt and isotonic calibration are complete across rolling-origin folds, Brier/ECE/log loss are reported, and a reliability diagram is generated. Final calibration choice remains open until final target/model freeze and uncertainty intervals.
- P1-4 uncertainty: still OPEN. Calibration does not replace bootstrap CIs or multi-seed stability.
- P2-1 rolling-origin: still PARTIAL. Calibration is now folded into rolling-origin evidence, but category drift and COVID-exclusion sensitivity remain open.

## Major Methodological Rebuild - Bootstrap CI Pass 1

This pass adds the first paired cluster-bootstrap uncertainty evidence for P1-4 and the calibration comparison. It is not the full final manuscript CI package because the final target/model/table rows and multi-seed stability remain open.

New script and outputs:

- Script: `scripts/major_revision_bootstrap_ci.py`
- Reviewer-facing report: `bootstrap_report.md`
- Detailed outputs: `data/processed/model_results/major_revision/bootstrap/`
- Row-level final-style predictions: `data/processed/model_results/major_revision/bootstrap/bootstrap_prediction_rows.csv.gz`

Bootstrap protocol:

- Fold: final-style 2025 with train through 2023, validation 2024, test 2025.
- Cluster unit: NTA x complaint category.
- Number of clusters: 2,358.
- Replicates: 1,000 with fixed seed.
- Targets evaluated: T0 current reference and T2 minimum-count target.
- Models: no-shortcut LightGBM with uncalibrated, Platt, and isotonic scores.

Main 95 percent CIs for Platt-calibrated scores:

- T0 current reference: PR-AUC = 0.3120 [0.3038, 0.3207], F1 = 0.3544 [0.3484, 0.3602], precision@5% = 0.4236 [0.4096, 0.4379], Brier = 0.1011 [0.0996, 0.1025].
- T2 minimum count 3: PR-AUC = 0.3165 [0.3080, 0.3252], F1 = 0.3613 [0.3551, 0.3676], precision@5% = 0.4180 [0.4035, 0.4324], Brier = 0.0869 [0.0852, 0.0888].
- T2 precision@1% = 0.5697 [0.5428, 0.6023] and lift@1% = 5.1506 [4.8934, 5.4603].

Paired calibration difference evidence:

- T0 Platt minus uncalibrated: Brier difference = -0.0731 [-0.0747, -0.0715], log-loss difference = -0.1824 [-0.1860, -0.1785].
- T2 Platt minus uncalibrated: Brier difference = -0.0807 [-0.0827, -0.0785], log-loss difference = -0.1995 [-0.2042, -0.1948].
- Platt does not change PR-AUC, F1, or precision@5% in this setup; these paired difference CIs are exactly zero.

Reviewer status updates:

- P1-4 uncertainty: PARTIAL. Cluster-bootstrap CIs now exist for final-style 2025 T0/T2 no-shortcut LightGBM rows and paired calibration differences. Still requires final Table 4/Table 5 row coverage and at least five stochastic seeds.
- P2-4 calibration: PARTIAL but materially strengthened. Calibration has statistically clear Brier/log-loss improvement without a ranking claim.
- Paired model-difference CI: PARTIAL. Calibration paired differences are complete; final tree-vs-baseline and final-model comparisons remain open until final model/target freeze.

## Major Methodological Rebuild - Seed Stability Pass 1

This pass adds the requested five stochastic seeds for the current final-style no-shortcut LightGBM candidates. It does not complete all seed requirements because the final target/model is not frozen and rolling-origin multi-seed evidence for every fold remains open.

New script and outputs:

- Script: `scripts/major_revision_seed_stability.py`
- Reviewer-facing report: `seed_stability_report.md`
- Detailed outputs: `data/processed/model_results/major_revision/seeds/`

Protocol:

- Fold: final-style 2025 with train through 2023, validation 2024, and test 2025.
- Seeds: 42, 123, 2026, 3407, 7777.
- Targets: T0 current reference and T2 minimum-count target.
- Model: no-shortcut LightGBM.
- Thresholds and Platt calibrators are fitted only on validation 2024.

Five-seed test evidence for Platt-calibrated scores:

- T0 current reference: PR-AUC = 0.3115 +/- 0.0021, precision@5% = 0.4248 +/- 0.0037, F1 = 0.3549 +/- 0.0005, Brier = 0.1011 +/- 0.0001.
- T2 minimum count 3: PR-AUC = 0.3185 +/- 0.0023, precision@5% = 0.4194 +/- 0.0034, F1 = 0.3638 +/- 0.0015, Brier = 0.0868 +/- 0.0001.

Reviewer status updates:

- P1-4 uncertainty: PARTIAL. Five seeds are now complete for final-style T0/T2 no-shortcut LightGBM candidates, in addition to cluster-bootstrap CIs. Still requires final manuscript-row CI coverage and multi-seed evidence for the final frozen configuration.
- Multiple-seed requirement: PARTIAL. The required seed list has been run for these two candidate targets, but not yet for all ablation rows or every stochastic final candidate.
- P1-2 target selection: still PARTIAL. T2 is stable across seeds, but final target selection remains open until the full ablation and final selection rule are applied.

## Major Methodological Rebuild - Error Severity Pass 1

This pass adds the requested error-severity analysis for P2-5 using final-style 2025 no-shortcut LightGBM prediction rows. It does not complete final Results text because final target/model selection remains open.

New script and outputs:

- Script: `scripts/major_revision_error_severity.py`
- Reviewer-facing report: `error_severity_report.md`
- Detailed outputs: `data/processed/model_results/major_revision/error_analysis/`

Protocol:

- Prediction source: `data/processed/model_results/major_revision/bootstrap/bootstrap_prediction_rows.csv.gz`
- Score: Platt-calibrated no-shortcut LightGBM.
- Severity metric: `z = (target_next_week_count - rolling_8w_mean) / (rolling_8w_std + epsilon)`.
- Additional severity checks: absolute increase, ratio increase, and target-count bands.

Final-style 2025 severity evidence:

- T0 current reference: 15,712 positives, 8,917 TP, 6,795 FN, recall = 0.5675.
- T2 minimum count 3: 13,562 positives, 7,790 TP, 5,772 FN, recall = 0.5744.
- T2 recall by z severity: 0.5038 for 1.5 <= z < 2, 0.5649 for 2 <= z < 3, and 0.6443 for z >= 3.
- T2 still misses 1,716 positive rows in the z >= 3 group, so the manuscript must not imply that high-severity surges are reliably captured.
- T2 recall by target count: 0.3969 for count = 3, 0.5797 for count 4-9, and 0.5828 for count >= 10.

Reviewer status updates:

- P2-5 error severity: PARTIAL. Required severity bins and FN counts are now generated for final-style 2025. Still needs integration into manuscript tables/figures and final frozen target/model.
- P2-6 precision@k/capacity: still PARTIAL. Severity results complement capacity metrics but do not replace final workload analysis.

## Major Methodological Rebuild - Full-Training Ablation Pass 1

This pass replaces the old compact 300k ablation evidence for P1-6 with a full-row final-style ablation on the current sparse-aware target candidate. It is still not the final Table 3 package because it uses one seed and the final model/target are not frozen.

New script and outputs:

- Script: `scripts/major_revision_full_ablation.py`
- Reviewer-facing report: `full_training_ablation_report.md`
- Detailed outputs: `data/processed/model_results/major_revision/ablations/`

Protocol:

- Target: T2 minimum-count abnormal event (`T2_min_count_3`).
- Fold: final-style 2025 with train through 2023, validation 2024, and test 2025.
- Seed: 42.
- All configurations exclude OSM/PLUTO and remove formula-aligned 8-week shortcut features.
- Primary selection view: validation PR-AUC and precision@5%; 2025 test rows are diagnostic only.

Validation evidence:

- Final no-shortcut + NTA fixed effects: PR-AUC = 0.2951, precision@5% = 0.3924, F1 = 0.3618.
- Final no-shortcut + borough: PR-AUC = 0.2939, precision@5% = 0.3906, F1 = 0.3611.
- History + calendar: PR-AUC = 0.2754, precision@5% = 0.3633, F1 = 0.3436.
- History + calendar + weather: PR-AUC = 0.2738, precision@5% = 0.3639, F1 = 0.3466.
- History lags only: PR-AUC = 0.1848, precision@5% = 0.2352, F1 = 0.2616.
- Weather only: PR-AUC = 0.1405, precision@5% = 0.1602, F1 = 0.2218.

Held-out 2025 diagnostics:

- Final no-shortcut + NTA fixed effects: PR-AUC = 0.3187, precision@5% = 0.4215, F1 = 0.3634.
- Final no-shortcut + borough: PR-AUC = 0.3165, precision@5% = 0.4180, F1 = 0.3613.
- History + calendar + weather: PR-AUC = 0.2955, precision@5% = 0.3887, F1 = 0.3537.
- History + calendar: PR-AUC = 0.2861, precision@5% = 0.3804, F1 = 0.3446.

Weather interpretation:

- Weather does not improve validation PR-AUC over history + calendar in this pass: 0.2738 vs 0.2754, a delta of -0.0016.
- Weather gives only tiny validation gains in precision@5% (+0.0006) and F1 (+0.0029).
- Weather-only is weaker than calendar-only on validation PR-AUC: 0.1405 vs 0.1695.
- The manuscript must not justify weather using 2025 test gains. Under the pre-registered rule, weather remains questionable unless rolling-origin or multi-seed validation evidence supports it.
- The available weather variables are city-level observed feature-week Central Park exposures, not NTA-level weather and not t+1 operational forecasts. Historical NWS forecast archives and spatial weather grids remain Future Work because they require external data not collected here.

NTA fixed-effect interpretation:

- Adding NTA fixed effects to the final no-shortcut configuration improves validation PR-AUC from 0.2939 to 0.2951 and 2025 diagnostic PR-AUC from 0.3165 to 0.3187.
- The gain is small and needs multi-seed and/or uncertainty evidence before the manuscript can claim a robust spatial improvement.

Reviewer status updates:

- P1-6 full-training ablation: PARTIAL. Full-row final-style T2 ablation is complete for seed 42 without OSM/PLUTO or formula-aligned 8-week shortcut features. Still requires five-seed evidence for key/final rows and final manuscript table generation.
- P2-7 weather: PARTIAL. Full-data weather ablation is complete, but validation PR-AUC does not support keeping weather as a core final feature on this pass. Forecast-weather and spatial-grid weather are deferred to Future Work due external-data limits.
- P2-8 spatial/NTA: PARTIAL. NTA fixed-effect evidence exists and slightly improves validation/test diagnostics, but the effect is not yet robustly established.
- OSM/PLUTO removal from current ablation: PASS for this artifact. The CSV audit shows no ablation row contains OSM/PLUTO features.

## Major Methodological Rebuild - COVID and Archive Boundary Pass 1

This pass addresses P2-2 with an explicit COVID/archive-boundary audit. It uses only local workspace data and does not collect new external data.

New script and outputs:

- Script: `scripts/major_revision_covid_archive_audit.py`
- Reviewer-facing report: `covid_archive_report.md`
- Detailed outputs: `data/processed/model_results/major_revision/covid_archive/`

Evidence generated:

- Raw 311 schema header scan covers all 682 local raw 311 parts: 261 files for 2015-2019 and 421 files for 2020-2025.
- No raw 311 part is missing the required columns used by the pipeline (`unique_key`, `created_date`, `closed_date`, `complaint_type`, `borough`, `latitude`, `longitude`).
- Ordered column sequences vary slightly, but unordered raw 311 column sets are consistent within both periods: one unordered set in 2015-2019 and one unordered set in 2020-2025.
- Processed rows increase over time: 2,494,267 rows after cleaning in 2019, 2,862,967 in 2020, 3,148,082 in 2021, and 3,604,196 in 2025.
- COVID-period prevalence is visibly higher: T0 positive share is 0.1462 in 2020 and 0.1338 in 2021, versus 0.1222 in 2019 and 0.1281 in 2025. T2 positive share is 0.1250 in 2020 and 0.1140 in 2021, versus 0.1039 in 2019 and 0.1106 in 2025.
- Category mix shifts strongly in 2020: noise rises from 18.95% of 2019 observed requests to 27.99% in 2020, while housing falls from 23.84% to 18.34%.
- Raw complaint-type scan finds 197 complaint types present in both 2015-2019 and 2020-2025, 55 types observed only in 2015-2019, and 68 types observed only in 2020-2025.

Final-style 2025 sensitivity:

- Reference T2 no-shortcut LightGBM trained through 2023: PR-AUC = 0.3165, precision@5% = 0.4180, F1 = 0.3613, precision = 0.2635, recall = 0.5744.
- Excluding 2020-2021 from training reduces training rows from 1,077,606 to 832,374. Test PR-AUC = 0.3137, precision@5% = 0.4122, F1 = 0.3629, precision = 0.2725, recall = 0.5432.
- Relative to the reference, excluding 2020-2021 changes PR-AUC by -0.0028, precision@5% by -0.0059, and F1 by +0.0017. The higher F1 comes with lower recall and a lower alert rate, so this should be described as a sensitivity result rather than a clear improvement.

Provenance guardrail:

- `code_pulldata/pull_nyc_311.py` currently declares `DATASET_ID = erm2-nwe9`.
- The final modeling panel does not retain a row-level `source_dataset` or archive identifier.
- Therefore, the audit can compare 2015-2019 versus 2020-2025 periods and raw schema consistency, but it cannot prove row-level provenance from `76ig-c548` versus `erm2-nwe9` unless the data pull/pipeline is regenerated with source-id retention.

Reviewer status updates:

- P2-2 COVID/archive boundary: PARTIAL but materially strengthened. Schema, category drift, complaint-type overlap, yearly target composition, and exclude-2020-2021 sensitivity are now generated. Remaining work is manuscript integration and deciding whether to regenerate source provenance or state the archive-vintage limitation.
- P2-3 data vintage/revision risk: still PARTIAL. This pass documents source-provenance limits, but it does not solve open-data revision/vintage lag because historical data vintages are not available in the workspace and should be discussed as a limitation.

## Major Methodological Rebuild - Count Baseline Extension Pass 1

This pass extends P1-5 beyond the earlier Poisson and Poisson + NTA fixed-effect baselines. It adds practical count-forecasting baselines on the final-style 2025 fold and documents the current Negative Binomial blocker.

New script and outputs:

- Script: `scripts/major_revision_count_extensions.py`
- Reviewer-facing report: `count_model_extension_report.md`
- Detailed outputs: `data/processed/model_results/major_revision/count_extensions/`

Protocol:

- Target: T2 minimum-count abnormal event (`T2_min_count_3`).
- Fold: final-style 2025 with train through 2023, validation 2024, and test 2025.
- Features: count/history/calendar plus complaint category and borough; formula-aligned 8-week shortcut features are not used as model predictors.
- Count predictions are converted to abnormal-event decisions using the original count-threshold formula and a validation-selected count-score threshold.

Negative Binomial status:

- `statsmodels` is not installed in the local environment, and scikit-learn does not provide a Negative Binomial GLM.
- The NB GLM row is therefore documented as `blocked_missing_statsmodels`, not silently omitted.

Held-out 2025 count-extension evidence:

- HistGradientBoosting Poisson count model, formula-threshold decision: count MAE = 8.0211, Poisson deviance = 5.6952, PR-AUC = 0.1528, precision@5% = 0.1755, F1 = 0.1710, precision = 0.5129, recall = 0.1026.
- HistGradientBoosting Poisson count model, validation-score threshold: PR-AUC = 0.1528, precision@5% = 0.1755, F1 = 0.2429, precision = 0.1404, recall = 0.9002, alert rate = 0.7091.
- Hurdle HGB occurrence + positive-count model, formula-threshold decision: count MAE = 7.9377, Poisson deviance = 6.3093, PR-AUC = 0.1517, precision@5% = 0.1722, F1 = 0.1812, precision = 0.4967, recall = 0.1108.
- Hurdle HGB occurrence + positive-count model, validation-score threshold: PR-AUC = 0.1517, precision@5% = 0.1722, F1 = 0.2437, precision = 0.1409, recall = 0.9030, alert rate = 0.7091.

Interpretation:

- The extra count baselines improve count MAE relative to the earlier linear Poisson baselines, but their event ranking remains weak.
- These count-extension PR-AUC and precision@5% values are far below the current final-style no-shortcut LightGBM T2 diagnostic row (PR-AUC = 0.3165, precision@5% = 0.4180).
- The manuscript can say count baselines are included and currently not competitive as event rankers, but final paired uncertainty against the frozen classifier is still required before making a strong dominance claim.

Reviewer status updates:

- P1-5 count/spatial baseline: PARTIAL but stronger. Full-data Poisson, Poisson + NTA FE, HGB Poisson count, and hurdle-style count baselines now exist. NB GLM is documented as blocked by missing `statsmodels`; a final attempt or explicit limitation is still needed before manuscript freeze.
- T4 hurdle-style modeling: PARTIAL. A practical hurdle-style count baseline exists, but it is not a full target-definition/hurdle-event formulation for final selection.

## Major Methodological Rebuild - Complaint Mapping Appendix Pass 1

This pass addresses P2-10 by generating reviewer-facing complaint-type mapping artifacts and documenting the composition of the `other` category.

New script and outputs:

- Script: `scripts/major_revision_complaint_mapping_appendix.py`
- Reviewer-facing report: `complaint_mapping_appendix_report.md`
- Detailed outputs: `data/processed/model_results/major_revision/complaint_mapping/`

Evidence generated:

- The current deterministic mapping contains 320 original complaint types mapped into nine analysis categories.
- Category request shares: housing 26.06%, noise 21.70%, parking/traffic 21.08%, sanitation 9.99%, infrastructure 8.64%, public safety 5.16%, water/sewer 3.92%, environment 2.03%, and other 1.43%.
- `other` contains 86 complaint types and 441,648 of 30,940,129 mapped requests.
- The largest `other` complaint types are Encampment (191,408; 43.34% of `other`), Drug Activity (93,873; 21.26%), Animal in a Park (29,914; 6.77%), Vending (27,575; 6.24%), and Electronics Waste (26,487; 6.00%).

Reviewer status updates:

- P2-10 complaint mapping appendix: PARTIAL. The full mapping and `other` composition artifacts now exist and can be added to the appendix/supplement. Alternate grouping sensitivity remains open.

## Major Methodological Rebuild - Disparity and Workload Pass 1

This pass adds observable group performance and workload diagnostics for P2-6 and the feasible portion of P2-9. It does not use ACS/Census socioeconomic data because the current revision boundary forbids collecting new external demographic data.

New script and outputs:

- Script: `scripts/major_revision_disparity_workload_audit.py`
- Reviewer-facing report: `disparity_workload_report.md`
- Detailed outputs: `data/processed/model_results/major_revision/disparity_workload/`

Protocol:

- Prediction source: final-style 2025 bootstrap prediction rows for T2.
- Score/decision layer: Platt-calibrated score and validation-fitted Platt threshold.
- Joined context: borough and `rolling_8w_mean` from the final modeling panel.
- Groups audited: borough, complaint category, and historical-volume decile.

Workload evidence:

- The T2 Platt threshold produces 52 test weeks with mean 568.6 alerts/week, median 510.0, minimum 97, maximum 1,055, and mean alert rate 0.2411.
- Mean positives per week are 260.8.

Borough evidence:

- Alert rates range from 0.2086 in Manhattan to 0.2592 in Brooklyn.
- Precision ranges from 0.2452 in Staten Island to 0.3084 in Manhattan.
- Recall ranges from 0.5075 in Manhattan to 0.6045 in Brooklyn.
- Precision@5% ranges from 0.3581 in Staten Island to 0.4641 in the Bronx.

Complaint-category evidence:

- Noise has the highest alert rate (0.3442) and strongest category F1 (0.4213), with recall 0.7599.
- Other has the lowest alert rate (0.1532), precision 0.2530, recall 0.4310, and F1 0.3188.
- Sanitation remains weak: precision 0.2184, recall 0.4678, F1 0.2978.

Historical-volume evidence:

- D1 has near-zero baseline volume and almost no alerts: alert rate 0.0004 and precision/recall/F1 all 0.0000 under the selected threshold.
- D10 has mean `rolling_8w_mean` 162.3, alert rate 0.2636, precision 0.3063, recall 0.5824, F1 0.4015, and precision@5% 0.5554.
- These results reinforce that low-volume cells behave very differently and need explicit discussion.

Reviewer status updates:

- P2-6 precision@k/capacity/workload: PARTIAL. Weekly workload and group workload diagnostics are now generated, but final capacity policy still needs to be frozen in the manuscript.
- P2-9 fairness: DEFERRED/PARTIAL. Borough, category, and historical-volume diagnostics are complete from available data; socioeconomic fairness by income/demographics is deferred to Future Work because ACS/Census data are external and not collected in this revision.

## Major Methodological Rebuild - Final-Style Explainability Pass 1

This pass addresses the reviewer concern that the earlier SHAP narrative did not explain the actual deployed decision model and did not include local case studies.

New script and outputs:

- Script: `scripts/major_revision_explainability.py`
- Reviewer-facing report: `final_model_explainability_report.md`
- Detailed outputs: `data/processed/model_results/major_revision/explainability/`
- Figures: `shap_beeswarm.pdf`, `shap_local_tp.pdf`, `shap_local_fp.pdf`, and `shap_local_fn.pdf`

Protocol:

- Target: T2 minimum-count abnormal event (`T2_min_count_3`).
- Fold: final-style 2025 with train through 2023, validation 2024, and test 2025.
- Model explained: the same fitted LightGBM score model used for the current no-shortcut candidate. No ensemble proxy is used.
- Formula-aligned 8-week target-construction predictors are excluded from the score model: `rolling_8w_mean`, `rolling_8w_std`, `rolling_8w_sum`, and `ratio_to_8w_mean`.
- SHAP sample: 8,000 stratified held-out rows; local cases are selected from all held-out rows by pre-specified TP/FP/FN rules.

Held-out 2025 score-model evidence:

- Validation-selected threshold = 0.5452.
- Test PR-AUC = 0.3165, F1 = 0.3613, precision = 0.2635, recall = 0.5744.
- Raw feature count = 57; one-hot encoded model feature count = 69.
- Fit time for this pass = 20.10 seconds.

Global SHAP evidence:

- Shifted history is the dominant feature group after removing the formula-aligned 8-week shortcut features: mean |SHAP| share = 72.59%.
- Service category contributes 7.30%, current count 7.20%, feature-week weather 6.30%, calendar 5.54%, and borough 1.07%.
- Top features are `rolling_12w_mean`, `ratio_to_12w_mean`, `rolling_4w_std`, `complaint_count`, and `rolling_12w_std`. None of the removed formula-aligned 8-week construction fields appear in the explained score model.

Local case evidence:

- TP case: 2025-10-20, Parkchester/Bronx, housing, score 0.9754, next-week count 244, z-exceedance 11.8455.
- FP case: 2025-03-17, Mapleton-Midwood (West)/Brooklyn, noise, score 0.9563, next-week count 12, z-exceedance -0.1465.
- FN case: 2025-06-30, Elmhurst/Queens, other, score 0.3165, next-week count 426, z-exceedance 85.7384.
- Local contribution tables are saved in `shap_local_case_feature_contributions.csv`; waterfall PDFs are generated for each case.

Reviewer status updates:

- P1-1 target shortcut/SHAP circularity: PARTIAL but materially strengthened. The final-style explanation run removes formula-aligned target-construction predictors and documents the resulting SHAP structure.
- P2-11 local SHAP case studies: PARTIAL. TP/FP/FN cases and waterfall plots now exist; manuscript integration and interpretation text remain.
- P2-12 final decision explanation: PARTIAL/REDIRECTED. This pass explains the actual single LightGBM candidate and avoids an ensemble explanation proxy. The remaining manuscript decision is to freeze the single-model decision layer or implement a true ensemble explanation if the ensemble is retained.

## Major Methodological Rebuild - Manuscript Integration Draft 1

This pass rewrites the Overleaf manuscript around the major-revision evidence generated so far. It is a draft integration pass, not the final declaration that all 38 reviewer items are closed.

Files changed:

- `paper_overleaf/main.tex`
- `paper_overleaf/main.bbl`
- `paper_overleaf/main.pdf`
- `paper_overleaf/figures/reliability_diagram.pdf`
- `paper_overleaf/figures/shap_beeswarm_final.pdf`
- `paper_overleaf/figures/shap_local_tp.pdf`
- `paper_overleaf/figures/shap_local_fp.pdf`
- `paper_overleaf/figures/shap_local_fn.pdf`

Manuscript-level changes:

- Title changed to `Explainable Early Warning for Next-Week Abnormal Reported 311 Demand`.
- Main narrative now uses the T2 minimum-count target and no-shortcut LightGBM candidate rather than the previous category-aware ensemble framing.
- OSM/PLUTO are removed from the main manuscript text and final-model claims.
- The abstract now reports both source scale and model-panel scale, and emphasizes PR-AUC, F1, precision, recall, precision@1%, precision@5%, Brier score, bootstrap intervals, count baselines, calibration, and SHAP local explanations.
- Data and code availability now includes the GitHub repository link directly and removes the earlier "will be released" style language.
- The author email audit remains clean for Thu Le: `thulvm@fpt.edu.vn`.
- SHAP beeswarm and three local TP/FP/FN waterfall figures are included in the manuscript.

Build and QA:

- Command: `paper_overleaf/build_paper.ps1 -Clean`
- Output: `paper_overleaf/main.pdf`
- Page count: 13 pages.
- LaTeX log check: no undefined citations, no undefined references, no LaTeX errors, no fatal errors, and no overfull boxes in the final log search.
- Text audit: no `Category-Aware`, no OSM/PLUTO manuscript text, no `leakage-safe`, no `will be released`, no `thulvn`, GitHub URL present.
- Visual render QA: rendered all 13 pages with `pdftoppm`; title page, tables, reliability figure, beeswarm, and TP/FP/FN local SHAP figures are readable with no horizontal margin drift, clipping, or overlapping text. Figure-only pages have some Springer-class white space but stay within the 15-page limit.

Reviewer status updates:

- P1-3 title/category-aware/ensemble claim: PARTIAL. The manuscript title and main story now avoid the unsupported category-aware ensemble claim.
- P2-8 OSM/PLUTO main-paper removal: PARTIAL/PASS for current `paper_overleaf/main.tex`. Main-text OSM/PLUTO references are removed; supplementary/repo archive wording still needs final reviewer-facing packaging.
- P2-11 local SHAP case studies: PARTIAL but integrated into the draft manuscript.
- P2-12 final decision explanation: PARTIAL but integrated through the single LightGBM candidate. This depends on freezing the final decision layer as the single calibrated LightGBM.
- P3-3/P3-5 figure cleanup: PARTIAL. The old low-information workflow and old SHAP bar figures are no longer used in the current manuscript draft; final figure polish remains possible.

## Major Methodological Rebuild - Manuscript Reviewer-Gap Patch 1

This pass closes additional manuscript-level gaps that were still visible after Draft 1, using artifacts already generated in the major-revision pipeline.

Files changed or added:

- `paper_overleaf/main.tex`
- `paper_overleaf/main.pdf`
- `paper_overleaf/supplementary_complaint_mapping.tex`
- `paper_overleaf/supplementary_complaint_mapping.pdf`
- `paper_overleaf/tables/complaint_type_category_mapping_appendix.csv`
- `paper_overleaf/tables/other_category_composition.csv`

Manuscript additions:

- Added the complaint-type mapping size and category request shares to the Methods text: 320 original complaint types mapped to nine analysis categories.
- Added the full complaint mapping and `other` composition as supplementary CSV files, plus a one-page supplementary mapping PDF.
- Added the local final-style LightGBM fit time used for the explainability pass: 20.1 seconds.
- Clarified weather evidence: citywide feature-week weather is a contextual control, not neighborhood microclimate and not a next-week weather forecast; validation PR-AUC does not support weather as a core standalone contribution.
- Added deployment feedback-loop language: alerts can change agency attention, reporting behavior, and future labels.
- Added COVID/archive-boundary sensitivity text: excluding 2020-2021 from training changes 2025 PR-AUC from 0.3165 to 0.3137 and precision@5% from 0.4180 to 0.4122.
- Strengthened data-vintage limitation language: historical source vintages are unavailable, row-level archive-source identifiers are not retained, and late entry/reclassification/deduplication/revision cannot be fully measured from the current panel.
- Data and code availability now explicitly names complaint-type mappings as available in the repository.

Build and QA:

- Main build command: `paper_overleaf/build_paper.ps1 -Clean`, followed by one extra `pdflatex` pass.
- Supplementary build command: `pdflatex -interaction=nonstopmode -halt-on-error supplementary_complaint_mapping.tex`.
- Main PDF: 13 pages.
- Supplementary mapping PDF: 1 page.
- Final log search found no undefined citations, no undefined references, no LaTeX errors, no fatal errors, and no overfull boxes.
- PDF text audit confirms: no `Category-Aware`, no OSM/PLUTO manuscript text, no `will be released`, no `thulvn`, and GitHub URL present.

Reviewer status updates:

- P2-10 complaint mapping appendix: PARTIAL/PASS for current artifact package. The mapping and `other` composition are now in Overleaf supplementary files; alternate grouping sensitivity remains open.
- P2-2 COVID/archive boundary: PARTIAL but now integrated into Discussion.
- P2-3 data vintage/revision risk: PARTIAL but now explicitly integrated into Limitations.
- P2-7 weather: PARTIAL but now explicitly caveated in Results/Explainability.
- P3-2 deployment feedback loop: PARTIAL/PASS for current manuscript text.
- P3-13 compute reporting: PARTIAL. Fit time is now reported for the current final-style LightGBM pass; full production inference-latency benchmarking remains outside the current artifact set.
