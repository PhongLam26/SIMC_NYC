# Revision Report

Revision date: 2026-07-15

## Manuscript Structure

- PASS: Reorganized manuscript into the requested structure:
  - 1 Introduction
  - 2 Related Work
  - 3 Methodology
  - 3.1 Data and Task Formulation
  - 3.2 Predictive Models and Category-Aware Decision Layer
  - 3.3 Evaluation and Explainability Protocol
  - 4 Results
  - 4.1 Predictive Performance
  - 4.2 Category Operating Points
  - 4.3 Explainability
  - 5 Discussion
  - 5.1 Operational Implications
  - 5.2 Limitations and Future Work
  - 6 Conclusion
- PASS: Data description moved into Methodology.
- PASS: Discussion, limitations, and future work consolidated.
- PASS: RQ labels removed and research questions rewritten as prose.

## Reviewer Checklist Fixes

- PASS: Data and code availability now includes the GitHub repository:
  `https://github.com/PhongLam26/SIMC_NYC`
- PASS: Thu Le email checked and kept as `thulvm@fpt.edu.vn`.
- PASS: Removed broad "leakage-safe" wording; manuscript now uses narrower leakage-controlled wording and explicitly notes OSM/PLUTO snapshot/look-ahead limitations.
- PASS: Dataset row wording clarified:
  relative to the full dense panel, 30,654 rows are excluded due to warm-up/history requirements and unavailable next-week targets, leaving 1,325,196 model-ready rows.
- PASS: Table 3 discussion avoids claiming the selected category-aware strategy is best overall; it is framed as validation-selected and operationally preferred.
- PASS: Table 4 explanation now defines positives and alerts, and states consistency checks against total positives and alerts.

## Figures and Tables

- PASS: Workflow figure revised to five clear stages: data integration, panel and target, temporal split, modeling, decision layer.
- PASS: Added a new category operating-point figure generated from real model output:
  - Script: `code_model/create_category_operating_figure.py`
  - Source CSV: `data/processed/model_results/paper_tables_target_week_fulltrain/paper_table_04_category_performance.csv`
  - Outputs:
    - `paper_springer/figures/category_operating_points.pdf`
    - `paper_overleaf/figures/category_operating_points.pdf`
- PASS: Table 3 was compacted for readability after visual PDF inspection.

## References

- PASS: Bibliography expanded to 31 entries.
- PASS: All 31 bibliography entries are cited.
- PASS: No cited key is missing from the `.bib` files.
- PASS: `reference_audit.md` created with verification links for all entries.

## Build and PDF QA

- PASS: `paper_springer/main.pdf` builds successfully.
- PASS: `paper_overleaf/main.pdf` builds successfully.
- PASS: Both PDFs are 10 pages.
- PASS: Build logs contain no unresolved references, no undefined citations, no fatal errors, and no overfull boxes in the final logs.
- PASS: PDF text layer extraction confirms:
  - `thulvm@fpt.edu.vn`
  - `https://github.com/PhongLam26/SIMC_NYC`
  - Data and code availability statement
- PASS: `pdffonts` confirms no Type 3 fonts after regenerating the matplotlib figure with TrueType embedding.
- PASS: Visual render check completed for title page, method/table page, results/table page, figure page, and references page.

## Deliverable Folders

- PASS: `paper_springer/` contains the reviewer-ready Springer build.
- PASS: `paper_overleaf/` contains the synchronized Overleaf-ready source, bibliography, figures, and compiled PDF.
