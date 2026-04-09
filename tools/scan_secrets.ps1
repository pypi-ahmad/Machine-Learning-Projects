<#
.SYNOPSIS
    Scan the repo for common hardcoded secret patterns.

.DESCRIPTION
    Searches all tracked text files (*.py, *.txt, *.ini, *.yaml, *.yml, *.env*,
    *.json, *.cfg, *.toml) for patterns that commonly indicate leaked secrets:
    API keys, passwords, tokens, AWS credentials, private keys, etc.

    This script is READ-ONLY — it does not modify or delete anything.

.EXAMPLE
    .\tools\scan_secrets.ps1
    .\tools\scan_secrets.ps1 -Path "Cli_todo"
    .\tools\scan_secrets.ps1 -Verbose

.PARAMETER Path
    Optional. Restrict scanning to a subdirectory (relative to repo root).
    Default: scan the entire repo.
#>

[CmdletBinding()]
param(
    [string]$Path = "."
)

$ErrorActionPreference = "Stop"

# ── Configuration ─────────────────────────────────────────────────────

$FilePatterns = @(
    "*.py", "*.txt", "*.ini", "*.yaml", "*.yml", "*.json",
    "*.cfg", "*.toml", "*.env", "*.env.*", "*.sh", "*.ps1", "*.md"
)

# Patterns that strongly suggest hardcoded secrets.
# Each entry: [regex pattern, human-readable label]
$SecretPatterns = @(
    @("(?i)(api[_-]?key|apikey)\s*[:=]\s*['""]?.{8,}", "API key assignment"),
    @("(?i)(secret[_-]?key|client[_-]?secret)\s*[:=]\s*['""]?.{8,}", "Secret key assignment"),
    @("(?i)(password|passwd|pwd)\s*[:=]\s*['""]?.{4,}", "Password assignment"),
    @("(?i)(token|auth[_-]?token|access[_-]?token|bearer)\s*[:=]\s*['""]?.{8,}", "Token assignment"),
    @("(?i)AWS_ACCESS_KEY_ID\s*[:=]\s*['""]?AKI", "AWS access key"),
    @("(?i)AWS_SECRET_ACCESS_KEY\s*[:=]\s*['""]?.{20,}", "AWS secret key"),
    @("-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "Private key block"),
    @("(?i)(smtp_pass|email_pass|mail_password)\s*[:=]\s*['""]?.{4,}", "Email password"),
    @("(?i)(mongo|mysql|postgres|redis)://[^/\s]+:[^/\s]+@", "Database connection string with credentials"),
    @("ghp_[A-Za-z0-9]{36}", "GitHub personal access token"),
    @("xox[bpoas]-[A-Za-z0-9\-]+", "Slack token"),
    @("sk-[A-Za-z0-9]{32,}", "OpenAI / Stripe secret key")
)

# Files / folders to skip (already known safe or irrelevant)
$ExcludeDirs = @(".git", "node_modules", "__pycache__", ".ruff_cache", ".pytest_cache", "venv", ".venv", "env")

# ── Helpers ───────────────────────────────────────────────────────────

function Get-RepoRoot {
    $root = git rev-parse --show-toplevel 2>$null
    if (-not $root) {
        Write-Error "Not inside a git repository."
        exit 1
    }
    return $root
}

# ── Main ──────────────────────────────────────────────────────────────

$RepoRoot = Get-RepoRoot
$ScanRoot = Join-Path $RepoRoot $Path | Resolve-Path

Write-Host ""
Write-Host "=== Secret Pattern Scanner ===" -ForegroundColor Cyan
Write-Host "Scanning: $ScanRoot"
Write-Host "Patterns: $($SecretPatterns.Count) rules"
Write-Host ""

$TotalMatches = 0
$MatchResults = @()

foreach ($pattern in $FilePatterns) {
    $files = Get-ChildItem -Path $ScanRoot -Recurse -Filter $pattern -File -ErrorAction SilentlyContinue |
        Where-Object {
            $skip = $false
            foreach ($dir in $ExcludeDirs) {
                if ($_.FullName -match [regex]::Escape("\$dir\")) {
                    $skip = $true
                    break
                }
            }
            -not $skip
        }

    foreach ($file in $files) {
        $relativePath = $file.FullName.Replace("$RepoRoot\", "").Replace("\", "/")
        $lineNum = 0

        foreach ($line in (Get-Content $file.FullName -ErrorAction SilentlyContinue)) {
            $lineNum++

            foreach ($sp in $SecretPatterns) {
                $regex = $sp[0]
                $label = $sp[1]

                if ($line -match $regex) {
                    $TotalMatches++
                    # Truncate match for display (don't leak the actual secret)
                    $preview = if ($line.Length -gt 120) { $line.Substring(0, 120) + "..." } else { $line }
                    $MatchResults += [PSCustomObject]@{
                        File    = $relativePath
                        Line    = $lineNum
                        Rule    = $label
                        Preview = $preview.Trim()
                    }
                    break  # one match per line is enough
                }
            }
        }
    }
}

# ── Output ────────────────────────────────────────────────────────────

if ($TotalMatches -eq 0) {
    Write-Host "No secret patterns found." -ForegroundColor Green
} else {
    Write-Host "Found $TotalMatches potential secret(s):" -ForegroundColor Yellow
    Write-Host ""

    $MatchResults | Format-Table -Property File, Line, Rule -AutoSize -Wrap

    Write-Host ""
    Write-Host "Review each match above.  Not all may be real secrets (false positives" -ForegroundColor Yellow
    Write-Host "are expected).  Replace real secrets with .env files + python-dotenv." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done.  Scanned $(($MatchResults | Select-Object File -Unique).Count) file(s) with matches out of the repo." -ForegroundColor Cyan
