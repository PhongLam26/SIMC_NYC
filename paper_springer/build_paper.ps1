param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$auxFiles = @(
    "main.aux",
    "main.bbl",
    "main.blg",
    "main.log",
    "main.out",
    "main.pdf",
    "main.fdb_latexmk",
    "main.fls"
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

Invoke-Step "pdflatex pass 1" @("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex")
Invoke-Step "bibtex" @("bibtex", "main")
Invoke-Step "pdflatex pass 2" @("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex")
Invoke-Step "pdflatex pass 3" @("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex")

if (-not (Test-Path -LiteralPath "main.pdf")) {
    throw "main.pdf was not created"
}

$log = Get-Content -LiteralPath "main.log" -Raw
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

$blg = Get-Content -LiteralPath "main.blg" -Raw
if ($blg -notmatch 'warning\$ -- 0') {
    throw 'BibTeX did not report warning$ -- 0'
}

$pageLine = Select-String -Path "main.log" -Pattern "Output written on main.pdf" | Select-Object -Last 1
Write-Host $pageLine.Line
Get-Item -LiteralPath "main.pdf" | Select-Object FullName, Length, LastWriteTime
