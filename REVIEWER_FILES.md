# Reviewer File Guide

This repository keeps reviewer-facing materials and reproducibility outputs, while excluding local drafting notes, private handoff files, large redistributable raw data, and temporary build artifacts.

## Include for Reviewers

- `README.md` and `requirements.txt`: environment and project entry points.
- `code_pulldata/`, `code_processdata/`, and `code_model/`: data download, feature construction, modeling, thresholding, SHAP, and paper-table scripts.
- `data/processed/model_results/`: selected small CSV/JSON/MD/PNG outputs needed to verify the reported metrics, threshold trade-offs, seed stability checks, and explainability figures.
- `paper_springer/`: manuscript source, bibliography, compiled PDF, and submission-oriented LaTeX assets.
- `paper_overleaf/`: clean Overleaf-ready manuscript package with `main.tex`, `references.bib`, `figures/`, `tables/`, Springer class/style files, build script, and compiled `main.pdf`.

## Excluded from Reviewer Git History

- Raw or redistributable source data under `data/raw/` and large processed feature tables. Reviewers should use the public data sources and scripts instead.
- Model binaries, preprocessors, scored prediction dumps, and SHAP sample metadata because they are large generated artifacts.
- Internal drafting checklists, Word notes, dashboard planning notes, chat handoff files, local ZIP packages, and temporary folders.
- LaTeX build byproducts such as `.aux`, `.log`, `.out`, `.blg`, and SyncTeX files.
