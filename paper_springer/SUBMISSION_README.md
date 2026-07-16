# SIMC Review Manuscript Submission PDF

Upload file: `main_SIMC_submission.pdf`

- Pages: 12
- Paper size: A4
- PDF bookmarks/outlines: 0
- Clickable PDF links: 0
- `/Subtype /Link` annotations: 0
- Page numbers: none
- Running headers: none
- Footers: none
- Authors: all four authors retained
- URLs, DOI strings, email addresses, citations, and cross-references: visible as ordinary non-clickable text
- Current title: `Explainable Early Warning for Next-Week Abnormal Reported 311 Demand`
- Repository link visible in Data and code availability: `https://github.com/PhongLam26/SIMC_NYC`

Build command:

```powershell
powershell -ExecutionPolicy Bypass -File .\paper_springer\build_SIMC_submission.ps1 -Clean
```

Output:

```text
paper_springer/main_SIMC_submission.pdf
```

Submission source files are `main.tex`, `main_SIMC_submission.tex`, `references.bib`, `main.bbl`, and the files under `figures/` that are referenced by `main.tex`.
