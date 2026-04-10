# ============================================================================
# Computer Vision Projects — Environment Setup (Windows PowerShell)
# ============================================================================
# Usage:
#   .\scripts\setup_env.ps1
#   .\scripts\setup_env.ps1 -SkipTorch   # skip torch install (if already done)
# ============================================================================

param(
    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $RepoRoot

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  CV Projects — Environment Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# --- 1. Install requirements.txt ---
Write-Host "[1/4] Installing requirements.txt ..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed" }

# --- 1b. Verify OpenCV ---
try {
    python -c "import cv2; print(f'  OpenCV {cv2.__version__} OK')"
} catch {
    Write-Host @"
[WARN] opencv-python failed to install.
  If you are on Python 3.13 free-threaded (cp313t), binary wheels may not
  exist yet.  Options:
    1. Use standard CPython 3.13 (non-free-threaded)
    2. conda install -c conda-forge opencv
    3. Build from source: pip install --no-binary opencv-python opencv-python
"@ -ForegroundColor Red
}

# --- 2. Install PyTorch with CUDA 13.0 ---
if (-not $SkipTorch) {
    Write-Host "`n[2/4] Installing PyTorch + torchvision (CUDA 13.0) ..." -ForegroundColor Yellow
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }
} else {
    Write-Host "`n[2/4] Skipping PyTorch install (--SkipTorch)" -ForegroundColor DarkGray
}

# --- 3. Verify critical imports ---
Write-Host "`n[3/4] Verifying critical imports ..." -ForegroundColor Yellow
python -c @"
import torch, torchvision, cv2, ultralytics
print(f'  torch       {torch.__version__}  CUDA={torch.cuda.is_available()}')
print(f'  torchvision {torchvision.__version__}')
print(f'  OpenCV      {cv2.__version__}')
print(f'  Ultralytics {ultralytics.__version__}')
"@
if ($LASTEXITCODE -ne 0) { throw "Import verification failed" }

# --- 4. Run smoke tests ---
Write-Host "`n[4/4] Running smoke tests ..." -ForegroundColor Yellow
python scripts/smoke_test.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[WARN] Some smoke tests failed — check output above." -ForegroundColor Red
} else {
    Write-Host "`n[OK] All smoke tests passed!" -ForegroundColor Green
}

# --- 5. Configure git hooks ---
Write-Host "`n[+] Configuring git hooks path ..." -ForegroundColor Yellow
git config core.hooksPath .githooks 2>$null
Write-Host "  Git hooks path set to .githooks" -ForegroundColor DarkGray

Pop-Location
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green
