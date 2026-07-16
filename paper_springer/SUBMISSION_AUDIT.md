# Submission Audit

Artifact folder: `paper_springer/`

## Verified Complete

- Springer LaTeX template is used through `sn-jnl.cls` with `sn-mathphys-num` style.
- Main manuscript source is `main.tex`; submission wrapper is `main_SIMC_submission.tex`.
- Compiled reviewer PDF is `main_SIMC_submission.pdf`.
- Build command verified:
  - `powershell -ExecutionPolicy Bypass -File .\build_SIMC_submission.ps1 -Clean`
- Latest verified script output:
  - `paper_springer/main_SIMC_submission.pdf`, 11 A4 pages, SHA-256 `5160CBB0664C4C2AC3983289BEEE811B7EA9393A4096D35B656903D41FB5BC1B`.
- Automated audit PASS:
  - no PDF outlines/bookmarks
  - no `/OpenAction`, `/Names`, `/Dests`, or `/PageLabels`
  - no clickable link annotations
  - 11 A4 pages after the final build
  - no standalone page-number footer
  - no known running headers/footers
  - all fonts embedded and no Type 3 fonts
  - title, authors, emails, GitHub URL, DOI, dataset IDs, citations, and current metrics remain visible as ordinary text
- Visual render QA was completed for all 11 pages, including the methods, results tables, SHAP figures, declarations, and reference tail. No clipping, overlap, orphan heading, or excessive blank final page was observed.

## Current Manuscript Claims Verified

- Title: `Explainable Early Warning for Next-Week Abnormal Reported 311 Demand`.
- Corresponding email: `phonglam2599@gmail.com`; Thu Le email: `thulvm@fpt.edu.vn`.
- Data/code availability includes `https://github.com/PhongLam26/SIMC_NYC`.
- Final model is a single no-shortcut LightGBM with Platt calibration.
- Table 3 uses only held-out-2025 T2 rows (122,616 rows; 13,562 positives) for Poisson, HGB/hurdle, and LightGBM comparisons.
- Final held-out 2025 metrics remain visible:
  - PR-AUC 0.317
  - F1 0.361
  - precision 0.263
  - recall 0.574
  - precision@1% 0.570
  - precision@5% 0.418
  - Brier score 0.087
- Submission-source stale-text audit found no superseded manuscript title, no old 0.3802/0.3310 headline metrics, and no old OSM/PLUTO bibliography entries in the rebuilt `main.bbl` files.

## Submission Files

- Upload/use `main_SIMC_submission.pdf` for SIMC review if the portal requires no links, page numbers, headers, or footers.
- Use `main.pdf` only as the normal Springer-rendered version.
- Use `references.bib`, `main.bbl`, and `figures/` as the source bundle paired with `main.tex`.
- Do not upload a separate supplementary PDF. Detailed diagnostic artifacts remain available in the public repository.
- Do not use earlier archived submissions or old generated tables as submission evidence.
