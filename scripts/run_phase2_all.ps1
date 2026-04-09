# scripts/run_phase2_all.ps1
# Phase 2.1: Execute ALL project training pipelines headlessly
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_phase2_all.ps1
# ---------------------------------------------------------------

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "[ERROR] .venv not found at $python" -ForegroundColor Red
    exit 1
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Phase 2.1 -- Run ALL project training pipelines" -ForegroundColor Cyan
Write-Host " Workspace : $root" -ForegroundColor Cyan
Write-Host " Python    : $python" -ForegroundColor Cyan
Write-Host " Time      : $(Get-Date -Format 'yyyy-MM-dd HH:mm')" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Quick sanity check
Write-Host "`n[pre-flight] Checking GPU + key packages..." -ForegroundColor Yellow
& $python -c @"
import torch
print(f'  PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}', end='')
if torch.cuda.is_available():
    print(f'  GPU={torch.cuda.get_device_name(0)}  Mem={torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')
else:
    print()
import transformers, peft, sentence_transformers
print(f'  transformers {transformers.__version__}  peft {peft.__version__}  sent-trans {sentence_transformers.__version__}')
"@
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Pre-flight check failed" -ForegroundColor Red
    exit 1
}

# Run orchestrator
Write-Host "`n[run] Starting orchestrator..." -ForegroundColor Yellow
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $python scripts/run_all_phase2.py @args
$exitCode = $LASTEXITCODE
$sw.Stop()

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " Finished in $([math]::Round($sw.Elapsed.TotalMinutes, 1)) minutes (exit=$exitCode)" -ForegroundColor $(if ($exitCode -eq 0) { "Green" } else { "Yellow" })
Write-Host " Reports:" -ForegroundColor Cyan
Write-Host "   reports/phase2_summary.md"
Write-Host "   reports/phase2_leaderboard.csv"
Write-Host "   reports/phase2_failures.md"
Write-Host "============================================================" -ForegroundColor Cyan
