# Submission Audit

Artifact folder: `paper_springer/`

## Verified Complete

- Springer LaTeX template is used through `sn-jnl.cls` with `sn-mathphys-num` style.
- Main manuscript source is `main.tex`; submission wrapper is `main_SIMC_submission.tex`.
- Compiled reviewer PDF is `main_SIMC_submission.pdf`.
- Build command verified:
  - `powershell -ExecutionPolicy Bypass -File .\build_SIMC_submission.ps1 -Clean`
- Latest verified script output:
  - `Output written on main_SIMC_submission.pdf (12 pages, 557666 bytes).`
- Automated audit PASS:
  - no PDF outlines/bookmarks
  - no `/OpenAction`, `/Names`, `/Dests`, or `/PageLabels`
  - no clickable link annotations
  - 12 A4 pages
  - no standalone page-number footer
  - no known running headers/footers
  - all fonts embedded and no Type 3 fonts
  - title, authors, emails, GitHub URL, DOI, dataset IDs, citations, and current metrics remain visible as ordinary text
- Visual render QA PASS on `tmp/pdfs/final_logic_contact_sheet.png` and zoomed pages 5--9: no horizontal margin drift, clipping, overlap, or table spillover observed.

## Current Manuscript Claims Verified

- Title: `Explainable Early Warning for Next-Week Abnormal Reported 311 Demand`.
- Corresponding email: `phonglam2599@gmail.com`; Thu Le email: `thulvm@fpt.edu.vn`.
- Data/code availability includes `https://github.com/PhongLam26/SIMC_NYC`.
- Final model is a single no-shortcut LightGBM with Platt calibration.
- Final held-out 2025 metrics remain visible:
  - PR-AUC 0.317
  - F1 0.361
  - precision 0.263
  - recall 0.574
  - precision@1% 0.570
  - precision@5% 0.418
  - Brier score 0.087
- Submission-source stale-text audit found no old Category-Aware title, no old 0.3802/0.3310 headline metrics, and no old OSM/PLUTO bibliography entries in the rebuilt `main.bbl` files.

## Submission Files

- Upload/use `main_SIMC_submission.pdf` for SIMC review if the portal requires no links, page numbers, headers, or footers.
- Use `main.pdf` only as the normal Springer-rendered version.
- Use `references.bib`, `main.bbl`, and `figures/` as the source bundle paired with `main.tex`.
- Do not use earlier archived submissions or old generated tables as submission evidence.
