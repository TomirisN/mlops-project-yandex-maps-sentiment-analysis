# Push code fixes to GitHub (excludes mlruns/ and train logs).
# Usage:
#   .\scripts\push_changes.ps1
#   .\scripts\push_changes.ps1 -DryRun

param(
    [switch]$DryRun,
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArgs)
    if ($DryRun) {
        Write-Host "[dry-run] git $($GitArgs -join ' ')" -ForegroundColor DarkGray
    } else {
        & git @GitArgs
        if ($LASTEXITCODE -ne 0) { throw "git $($GitArgs -join ' ') failed (exit $LASTEXITCODE)" }
    }
}

$pathsToAdd = @(
    ".gitignore",
    "app/core/model_manager.py",
    "app/core/text_utils.py",
    "docker/Dockerfile",
    "docker/Dockerfile.mlflow",
    "docker/docker-compose.yml",
    "docker/docker-compose.override.yml",
    "metrics.json",
    "scripts/deploy_minikube.ps1",
    "scripts/check_mlflow_registry.py",
    "scripts/reset_model_registry.py",
    "scripts/push_changes.ps1",
    "src/core/mlflow_utils.py",
    "src/core/train_model_versions.py",
    "src/core/train_v4_high_accuracy.py",
    "data/reference/reference_sample.csv",
    "reports/drift/.gitkeep"
)

Write-Host "Repository: $Root"
Write-Host "Branch: $Branch"
Write-Host "Mode: $(if ($DryRun) { 'DRY RUN (no changes)' } else { 'COMMIT + PUSH' })"
Write-Host ""

$existing = @()
$missing = @()
foreach ($p in $pathsToAdd) {
    if (Test-Path (Join-Path $Root $p)) { $existing += $p } else { $missing += $p }
}

$trackedMlruns = @(& git ls-files mlruns 2>$null)

Write-Host "Will commit ($($existing.Count) files):"
$existing | ForEach-Object { Write-Host "  + $_" }

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "Skip (not found):"
    $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
}

if ($trackedMlruns.Count -gt 0) {
    Write-Host ""
    Write-Host "Untrack mlruns/ from git (files stay on disk, $($trackedMlruns.Count) paths):"
    Invoke-Git rm -r --cached --ignore-unmatch mlruns/
}

Write-Host ""
Write-Host "Will NOT commit: mlruns/, models/*.pkl, train*.log"
Write-Host ""

foreach ($p in $existing) {
    Invoke-Git add -- $p
}

if (-not $DryRun) {
    $staged = @(git diff --cached --name-only)
    if ($staged.Count -eq 0) {
        Write-Host "Nothing to commit." -ForegroundColor Yellow
        exit 0
    }
    Write-Host "Staged ($($staged.Count) paths):"
    $staged | ForEach-Object { Write-Host "  $_" }
    Write-Host ""
}

$commitTitle = "fix: Docker MLflow, text preprocessing, drift reference data"
$commitBody = @(
    "MLflow file store in Docker (/mlflow/mlruns), mlflow_utils",
    "clean_text in API for Russian v4 inference",
    "reference_sample.csv and reports/drift for monitoring",
    "deploy_minikube: auto-detect minikube on Windows",
    "metrics.json v4; registry check/reset scripts",
    "stop tracking mlruns/ in git"
) -join "`n"

Write-Host "Commit:"
Write-Host "  $commitTitle"
Write-Host $commitBody
Write-Host ""

if ($DryRun) {
    Write-Host "[dry-run] git commit -m `"$commitTitle`""
    Write-Host "[dry-run] git push -u origin $Branch" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "OK to push. Run without -DryRun:" -ForegroundColor Green
    Write-Host "  .\scripts\push_changes.ps1"
    exit 0
}

Invoke-Git commit -m $commitTitle -m $commitBody

Write-Host "Push to origin/$Branch ..."
Invoke-Git push -u origin $Branch

Write-Host ""
Write-Host "Done: origin/$Branch" -ForegroundColor Green
git log -1 --oneline
