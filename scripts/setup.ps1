#Requires -Version 5.1
<#
.SYNOPSIS
    Bootstrap the NLP-Projects workspace virtual environment.

.DESCRIPTION
    Idempotent setup script.  Safe to re-run at any time.
    1. Creates .venv if it does not exist  (Python 3.13+)
    2. Upgrades pip / setuptools / wheel
    3. Installs GPU PyTorch via cu130 index
    4. Installs requirements.txt
    5. Registers ipykernel for Jupyter notebooks

.NOTES
    Run from the workspace root:
        powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ROOT = Split-Path -Parent $PSScriptRoot          # workspace root
$VENV = Join-Path $ROOT '.venv'
$REQ  = Join-Path $ROOT 'requirements.txt'
$TORCH_INDEX = 'https://download.pytorch.org/whl/cu130'

Write-Host ''
Write-Host '==== NLP-Projects Workspace Setup ====' -ForegroundColor Cyan
Write-Host ''

# ---------- 1. Create venv --------------------------------------------------
if (Test-Path (Join-Path $VENV 'Scripts\python.exe')) {
    Write-Host '  OK   .venv already exists - reusing.' -ForegroundColor Green
} else {
    Write-Host '  ..   Creating virtual environment in .venv ...' -ForegroundColor Yellow
    python -m venv $VENV
    Write-Host '  OK   .venv created.' -ForegroundColor Green
}

$PIP    = Join-Path $VENV 'Scripts\pip.exe'
$PYTHON = Join-Path $VENV 'Scripts\python.exe'

# ---------- 2. Upgrade pip / setuptools / wheel ------------------------------
Write-Host '  ..   Upgrading pip, setuptools, wheel ...' -ForegroundColor Yellow
& $PIP install --upgrade pip setuptools wheel | Out-Null
Write-Host '  OK   pip / setuptools / wheel up-to-date.' -ForegroundColor Green

# ---------- 3. GPU PyTorch (cu130) -------------------------------------------
Write-Host '  ..   Installing PyTorch (cu130) ...' -ForegroundColor Yellow
& $PIP install torch torchvision --index-url $TORCH_INDEX
if ($LASTEXITCODE -ne 0) {
    Write-Host ('  FAIL PyTorch install failed (exit {0}).' -f $LASTEXITCODE) -ForegroundColor Red
} else {
    Write-Host '  OK   PyTorch installed.' -ForegroundColor Green
}

# ---------- 4. Install requirements.txt --------------------------------------
if (Test-Path $REQ) {
    Write-Host '  ..   Installing requirements.txt ...' -ForegroundColor Yellow
    & $PIP install -r $REQ
    if ($LASTEXITCODE -ne 0) {
        Write-Host ('  WARN Some packages may have failed (exit {0}).' -f $LASTEXITCODE) -ForegroundColor Red
    } else {
        Write-Host '  OK   requirements.txt installed.' -ForegroundColor Green
    }
} else {
    Write-Host ('  SKIP requirements.txt not found at {0}.' -f $REQ) -ForegroundColor Yellow
}

# ---------- 5. Register ipykernel --------------------------------------------
Write-Host '  ..   Registering ipykernel (nlp-projects) ...' -ForegroundColor Yellow
& $PYTHON -m ipykernel install --user --name nlp-projects --display-name 'NLP Projects (py313)'
if ($LASTEXITCODE -ne 0) {
    Write-Host '  WARN ipykernel registration may have failed.' -ForegroundColor Red
} else {
    Write-Host '  OK   ipykernel registered as nlp-projects.' -ForegroundColor Green
}

Write-Host ''
Write-Host '==== Setup complete ====' -ForegroundColor Cyan
Write-Host ''
