<# 
.SYNOPSIS
    Phase 3 runner — one-command project (re-)training.

.PARAMETER Project
    Single project slug to run.

.PARAMETER BadOnly
    Re-run only projects that failed quality thresholds in Phase 2.1.

.PARAMETER All
    Re-run all 21 projects.

.PARAMETER Force
    Force re-run even if cached metrics exist.

.EXAMPLE
    .\scripts\run_project.ps1 -Project "toxic-comment-classification" -Force
    .\scripts\run_project.ps1 -BadOnly -Force
#>
param(
    [string]$Project,
    [switch]$BadOnly,
    [switch]$All,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path "$root\scripts\run_phase3.py")) {
    $root = Split-Path -Parent $PSScriptRoot
}
if (-not (Test-Path "$root\scripts\run_phase3.py")) {
    $root = $PSScriptRoot | Split-Path
}

$venv = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venv)) { $venv = "python" }

$args_list = @()
if ($Project)  { $args_list += "--project", $Project }
elseif ($BadOnly) { $args_list += "--bad-only" }
elseif ($All)  { $args_list += "--all" }
else { Write-Error "Specify -Project <slug>, -BadOnly, or -All"; exit 1 }

if ($Force) { $args_list += "--force" }

Write-Host "Running: $venv -m scripts.run_phase3 $($args_list -join ' ')" -ForegroundColor Cyan
Push-Location $root
& $venv -m scripts.run_phase3 @args_list
Pop-Location
