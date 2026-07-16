param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$job = "main_SIMC_submission"
$auxFiles = @(
    "$job.aux",
    "$job.bbl",
    "$job.blg",
    "$job.log",
    "$job.out",
    "$job.pdf",
    "$job.toc",
    "$job.lof",
    "$job.lot",
    "$job.fdb_latexmk",
    "$job.fls",
    "$job.synctex.gz"
)

if ($Clean) {
    foreach ($file in $auxFiles) {
        Remove-Item -LiteralPath $file -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Command
    )
    Write-Host "==> $Name"
    & $Command[0] @($Command[1..($Command.Length - 1)])
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Invoke-Step "pdflatex submission pass 1" @("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "$job.tex")
Invoke-Step "bibtex submission" @("bibtex", $job)
Invoke-Step "pdflatex submission pass 2" @("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "$job.tex")
Invoke-Step "pdflatex submission pass 3" @("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "$job.tex")

if (-not (Test-Path -LiteralPath "$job.pdf")) {
    throw "$job.pdf was not created"
}

$log = Get-Content -LiteralPath "$job.log" -Raw
$badLogPatterns = @(
    "undefined references",
    "undefined citations",
    "Fatal error",
    "! LaTeX Error"
)

foreach ($pattern in $badLogPatterns) {
    if ($log -match [regex]::Escape($pattern)) {
        throw "LaTeX log contains: $pattern"
    }
}

$blg = Get-Content -LiteralPath "$job.blg" -Raw
if ($blg -notmatch 'warning\$ -- 0') {
    throw 'BibTeX did not report warning$ -- 0'
}

Invoke-Step "automated SIMC PDF audit" @("python", "..\scripts\audit_submission_pdf.py", "$job.pdf", "--expected-pages", "13")

$pageLine = Select-String -Path "$job.log" -Pattern "Output written on $job.pdf" | Select-Object -Last 1
Write-Host $pageLine.Line
Get-Item -LiteralPath "$job.pdf" | Select-Object FullName, Length, LastWriteTime
