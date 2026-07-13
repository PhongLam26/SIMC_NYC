# SIMC Overleaf Paper Package

Upload the contents of this `paper_overleaf/` folder to Overleaf.

Recommended Overleaf settings:

- Compiler: `pdfLaTeX`
- Main file: `main.tex`
- Bibliography file: `references.bib`
- Template files: `sn-jnl.cls`, `sn-mathphys-num.bst`

Folder layout:

- `main.tex`: main manuscript.
- `references.bib`: BibTeX references.
- `figures/`: pipeline and SHAP figures used by the paper.
- `tables/`: LaTeX table files included from `main.tex`.
- `build_paper.ps1`: local Windows build script for checking the PDF before upload.

Local build:

```powershell
cd paper_overleaf
.\build_paper.ps1 -Clean
```
