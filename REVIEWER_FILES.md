# Reviewer File Guide

This repository keeps reviewer-facing materials and compact reproducibility outputs for the current paper, **Category-Aware Explainable Machine Learning for Urban Service Demand Forecasting in Smart Cities**, while excluding local drafting notes, private handoff files, large redistributable raw data, and temporary build artifacts.

## Include for Reviewers

- `README.md` and `requirements.txt`: environment and project entry points.
- `code_pulldata/`, `code_processdata/`, and `code_model/`: data download, feature construction, modeling, thresholding, SHAP, and paper-table scripts.
- `configs/features_prospective.json`: final prospective feature protocol.
- `data/processed/model_results/prospective/`: selected small CSV/JSON/MD/PNG outputs needed to verify the reported prospective metrics, threshold trade-offs, paper tables, and explainability figures.
- `data/processed/model_results/retrospective_context/`: labelled OSM/PLUTO 2026 context checks; these are not used in final prospective alerts.
- `paper_springer/`: manuscript source, bibliography, compiled PDF, and submission-oriented LaTeX assets.
- `paper_overleaf/`: clean Overleaf-ready manuscript package with `main.tex`, `references.bib`, `figures/`, `tables/`, Springer class/style files, build script, and compiled `main.pdf`.
- `REVISION_REPORT.md`, `prospective_leakage_audit.md`, `numerical_consistency_report.md`, and `reference_audit.md`: reviewer-facing synchronization and audit reports.

## Excluded from Reviewer Git History

- Raw or redistributable source data under `data/raw/` and large processed feature tables. Reviewers should use the public data sources and scripts instead.
- Model binaries, preprocessors, full scored prediction dumps, and SHAP sample metadata because they are large generated artifacts.
- Internal drafting checklists, Word notes, dashboard planning notes, chat handoff files, local ZIP packages, and temporary folders.
- Older non-tracked planning notes in the local workspace are superseded by `README.md` and `REVISION_REPORT.md` and should not be used as reviewer-facing evidence.
- LaTeX build byproducts such as `.aux`, `.log`, `.out`, `.blg`, and SyncTeX files.
