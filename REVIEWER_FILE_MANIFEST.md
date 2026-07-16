# Reviewer File Manifest

Use this list for the current major-revision submission package.

## Upload / Reviewer-Facing

- `paper_springer/main_SIMC_submission.pdf` - SIMC-safe review PDF with no links, bookmarks, page numbers, running headers, or footers.
- `paper_springer/main.tex` - Springer manuscript source matching the review PDF.
- `paper_springer/main_SIMC_submission.tex` - wrapper used to build the no-link/no-header submission PDF.
- `paper_springer/references.bib` - cleaned bibliography for the current manuscript.
- `paper_springer/main.bbl` - generated bibliography paired with `main.tex`.
- `paper_springer/figures/reliability_diagram.pdf`
- `paper_springer/figures/shap_beeswarm_final.pdf`
- `paper_springer/figures/shap_local_tp.pdf`
- `paper_springer/figures/shap_local_fp.pdf`
- `paper_springer/figures/shap_local_fn.pdf`
- `paper_overleaf/supplementary_complaint_mapping.pdf` - optional supplementary appendix if the venue allows supplement upload.
- `paper_overleaf/tables/complaint_type_category_mapping_appendix.csv` - optional supplementary mapping table.
- `paper_overleaf/tables/other_category_composition.csv` - optional supplementary `other` composition table.

## Keep Internal / Do Not Upload As Main Evidence

- `archive/` - old submitted or intermediate manuscript states.
- `REVISION_REPORT.md` - internal audit trail; useful for us, not a manuscript file.
- `major_revision_*report*.md`, `major_revision_*checklist*.md`, and `tree_vs_count_paired_ci_report.md` - internal evidence and QA notes.
- `data/processed/model_results/` - generated model artifacts; cite derived tables/figures instead of uploading raw result folders unless specifically requested.
- `tmp/` - render QA scratch files.
- Old PDFs or source bundles outside `paper_springer/` and `paper_overleaf/` unless a reviewer explicitly asks for them.

## Current Cleanliness Checks

- Current title: `Explainable Early Warning for Next-Week Abnormal Reported 311 Demand`.
- Repository URL present: `https://github.com/PhongLam26/SIMC_NYC`.
- Thu Le email checked as `thulvm@fpt.edu.vn`.
- Final model framing: single no-shortcut LightGBM with Platt calibration.
- Old `Category-Aware` title, old ensemble headline metrics, and stale OSM/PLUTO final-model claims are not part of the reviewer-facing manuscript package.
