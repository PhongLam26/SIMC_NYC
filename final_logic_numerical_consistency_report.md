# Final Logic and Numerical Consistency Report

Status: PASS, 2026-07-16.

This report records the final logic, calibration-scale, formatting, and numerical checks for the current SIMC_NYC manuscript package.

## Manuscript and Build Scope

- Working branch: `major-revision-methodological-rebuild`.
- Upload PDF: `paper_springer/main_SIMC_submission.pdf`.
- Working PDFs: `paper_overleaf/main.pdf` and `paper_springer/main.pdf`.
- Page count: 12 A4 pages for all three PDFs.
- Submission audit: PASS with no bookmarks, no link annotations, no page labels, no running headers/footers, no page-number footer, all fonts embedded, and no Type 3 fonts.
- Visual audit: all 12 submission pages rendered to PNG and inspected; pages containing Tables 3--5 and Figures 1--5 show no horizontal margin drift, clipping, overlap, or figure/table spillover.

## Target and Period Logic

- The 27,277 positives among 247,590 rows are described as the 2024--2025 two-year diagnostic period, not as the 2025 held-out year alone.
- The held-out 2025 test counts remain separate in the evaluation text.
- The analytical unit remains NTA x week x complaint category.
- Table 1 states model-ready rows relative to the dense panel, avoiding the earlier possible confusion about subtracting excluded rows from valid-label rows.

## Metric Rounding and Tables

- Main-manuscript metrics are rounded to three decimals in prose and tables.
- Table 2 headline values match the rounded final LightGBM evidence: PR-AUC 0.317, F1 0.361, precision 0.263, recall 0.574, P@1% 0.570, P@5% 0.418, Brier 0.087, and log loss 0.293.
- Table 3 now reports one decision rule per row and includes P@5%, precision, recall, and F1 for every baseline/comparator row.
- Count-model rows in Table 3 are explicitly framed as formula-threshold count diagnostics; HGB and hurdle rows use the final T2 target.
- Table 4 uses `Final + NTA indicators` for the tree diagnostic, while the GLM count baseline keeps fixed-effect wording where appropriate.
- Table 5 header is now `Event prevalence`; the earlier positive-share wording was removed.

## Calibration and SHAP Scale

- Platt calibration is fit on 2024 validation data only and applied unchanged to held-out 2025.
- The manuscript makes no claim that Platt improves PR-AUC; it is described as a monotone calibration layer used for probability scale and thresholding.
- TreeSHAP values and local waterfall figures are described as LightGBM raw-margin explanations before Platt calibration.
- Local case text reports both uncalibrated LightGBM probabilities and Platt-calibrated probabilities.
- Audit thresholds: uncalibrated LightGBM threshold 0.545172; equivalent Platt-calibrated threshold 0.166812.
- Audited local cases:
  - TP: raw margin 3.6793, uncalibrated probability 0.9754, calibrated probability 0.6250, alert 1, true label 1.
  - FP: raw margin 3.0851, uncalibrated probability 0.9563, calibrated probability 0.6027, alert 1, true label 0.
  - FN: raw margin -0.7701, uncalibrated probability 0.3165, calibrated probability 0.0609, alert 0, true label 1.

## Logic and Presentation Checks

- Revision-letter phrasing was removed from the main manuscript.
- The model is described as a validation-selected / operationally preferred strategy, not as universally best on every held-out metric.
- Weather is retained only as a documented contextual covariate: it does not improve validation PR-AUC in the isolated ablation, although validation P@5% and F1 increase slightly. It is not treated as a core ranking-performance source and is not justified by held-out test results or SHAP.
- OSM/PLUTO 2026 variables are described only as retrospective diagnostic checks and are not used for final tuning, thresholds, headline metrics, or SHAP.
- Figure 1 labels and caption now distinguish the original T0 target from the final T2 minimum-count target.
- Figure PDFs were regenerated with embedded TrueType/Type 1 fonts and no Type 3 fonts.

## Final Text Audit

- Source scan found no remaining main-manuscript instances of the legacy positive-share header, final-candidate phrasing, revision-letter phrasing, old local-score wording, or the old 2025-only diagnostic-period wording.
- LaTeX log audit found no undefined references/citations, fatal LaTeX errors, or overfull boxes.
