# scripts/run_phase2.ps1
# Phase 2 runner: patch notebooks + smoke test + summary
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_phase2.ps1
# ---------------------------------------------------------------

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Phase 2 Runner" -ForegroundColor Cyan
Write-Host " Workspace: $root" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Activate venv
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] .venv not found at $venvPython" -ForegroundColor Red
    exit 1
}
Write-Host "`n[1/4] Using Python: $venvPython" -ForegroundColor Yellow

# 2. Ensure key packages
Write-Host "`n[2/4] Checking packages..." -ForegroundColor Yellow
& $venvPython -c @"
import transformers, peft, sentence_transformers, datasets, rouge_score, sacrebleu, torch
print(f'  transformers      {transformers.__version__}')
print(f'  peft              {peft.__version__}')
print(f'  sentence-trans.   {sentence_transformers.__version__}')
print(f'  datasets          {datasets.__version__}')
print(f'  torch             {torch.__version__}')
print(f'  CUDA available    {torch.cuda.is_available()}')
try:
    import hdbscan; print(f'  hdbscan           {hdbscan.__version__}')
except: print('  hdbscan           NOT INSTALLED (optional)')
try:
    import bertopic; print(f'  bertopic          {bertopic.__version__}')
except: print('  bertopic          NOT INSTALLED (optional)')
"@
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Package check failed" -ForegroundColor Red
    exit 1
}

# 3. Run Phase 2 patching
Write-Host "`n[3/4] Patching notebooks with Phase 2 cells..." -ForegroundColor Yellow
& $venvPython scripts/phase2_patch_notebooks.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] phase2_patch_notebooks.py failed" -ForegroundColor Red
    exit 1
}

# 4. Smoke test: import all Phase 2 modules
Write-Host "`n[4/4] Smoke testing Phase 2 module imports..." -ForegroundColor Yellow
& $venvPython -c @"
import sys, importlib
modules = [
    'utils.training_common',
    'utils.train_text_classifier',
    'utils.train_summarizer',
    'utils.train_translator',
    'utils.embeddings_and_topics',
    'utils.captioning',
]
ok = 0
for m in modules:
    try:
        importlib.import_module(m)
        print(f'  OK   {m}')
        ok += 1
    except Exception as e:
        print(f'  FAIL {m}: {e}')
print(f'\nImport smoke test: {ok}/{len(modules)} passed')
if ok < len(modules):
    sys.exit(1)
"@
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] Some imports failed -- check output above" -ForegroundColor Yellow
} else {
    Write-Host "  All imports OK" -ForegroundColor Green
}

# Summary
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " Phase 2 Setup Complete" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Reports:"
Write-Host "   reports/phase2_summary.md"
Write-Host "   reports/phase2_failures.md"
Write-Host ""
Write-Host " To train a specific project, open its code.ipynb and run all cells."
Write-Host " Models + metrics will be saved to outputs/<slug>/"
Write-Host "============================================================" -ForegroundColor Cyan
