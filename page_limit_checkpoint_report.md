# Page-Limit Revision Checkpoint

Created before the single-PDF page-limit pass.

- Branch: `major-revision-methodological-rebuild`.
- Baseline commit: `74074014e4063bcdac424a7e6a50a9fab2025928`.
- Archived PDF: `archive/page_limit_checkpoint_20260716_142824/main_SIMC_submission_13_pages.pdf`.
- Baseline PDF: 13 A4 pages, SHA-256 `5e638496e9c25b2077b4253d5655b39388a3e04406871923d3902005b9eef726`.
- Headline held-out 2025 metrics: PR-AUC 0.317, F1 0.361, precision 0.263, recall 0.574, precision@1% 0.570, precision@5% 0.418, Brier score 0.087.
- Baseline evidence layout: four results tables (metrics, same-target count baselines, ablation, category metrics), reliability figure, SHAP beeswarm, and three local SHAP waterfalls (TP, FP, FN).
- Baseline submission audit: PASS for A4, embedded fonts, no Type 3 fonts, no links/bookmarks/headers/footers, and visible metadata; page count is non-compliant for a regular manuscript.

The following pass must produce one self-contained submission PDF of 10--12 pages without changing fitted-model metrics or removing required P1 evidence.
