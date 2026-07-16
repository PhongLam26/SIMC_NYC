param([switch]$Clean)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if ($Clean) { Remove-Item -LiteralPath "supplementary_SIMC.aux","supplementary_SIMC.log","supplementary_SIMC.pdf" -Force -ErrorAction SilentlyContinue }
python ..\scripts\final_same_target_supplement.py
& pdflatex -interaction=nonstopmode -halt-on-error supplementary_SIMC.tex
if ($LASTEXITCODE -ne 0) { throw "supplementary pdflatex pass 1 failed" }
& pdflatex -interaction=nonstopmode -halt-on-error supplementary_SIMC.tex
if ($LASTEXITCODE -ne 0) { throw "supplementary pdflatex pass 2 failed" }
if (-not (Test-Path supplementary_SIMC.pdf)) { throw "supplementary_SIMC.pdf was not created" }
