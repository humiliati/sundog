# Phase 12b - Throttled Groq Sweep Driver
#
# Runs the heavy-trace hosted harness against Groq open-weight models while
# respecting free-tier rate limits. The script is intentionally Windows
# PowerShell 5.1 compatible because the project machine may not have PowerShell
# 7 (`pwsh`) installed.
#
# Usage from repo root C:\Users\hughe\Dev\sundog:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run-groq-sweep.ps1
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run-groq-sweep.ps1 -Models llama-3.3-70b-versatile
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run-groq-sweep.ps1 -Slates differential
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run-groq-sweep.ps1 -SkipDone
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run-groq-sweep.ps1 -DryRun
#
# Resume after kill: re-run with -SkipDone. The script checks
# results/chat/phase5-hosted/<slate>/groq-<model>/summary.json.
#
# Log:
#   results/chat/phase12-groq-driver-log.jsonl

param(
    [string[]] $Models = @("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"),
    [string[]] $Slates = @("differential", "adversarial"),
    [string]   $KeyFile = "C:\Users\hughe\Dev\syek.corg.txt",
    [string]   $RepoRoot = "C:\Users\hughe\Dev\sundog",
    [switch]   $SkipDone,
    [switch]   $DryRun,
    [switch]   $RunFalsification
)

$ErrorActionPreference = "Stop"

$DelayMsByModel = @{
    "llama-3.3-70b-versatile" = 5500
    "llama-3.1-8b-instant"    = 10500
    "qwen/qwen3-32b"          = 11000
}

if ($RunFalsification) {
    $Slates = $Slates + "falsification"
}

if (-not (Test-Path $RepoRoot)) {
    Write-Error "Repo root not found at $RepoRoot. Aborting."
    exit 1
}

$HostedScript = Join-Path $RepoRoot "chat/eval/run_hosted_drafts.mjs"
if (-not (Test-Path $HostedScript)) {
    Write-Error "Hosted harness not found at $HostedScript. Aborting."
    exit 1
}

$LogPath = Join-Path $RepoRoot "results/chat/phase12-groq-driver-log.jsonl"
$LogDir = Split-Path -Parent $LogPath
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

$plan = @()
foreach ($model in $Models) {
    $delayMs = $DelayMsByModel[$model]
    if (-not $delayMs) {
        Write-Warning "No delay tuned for model $model - using 11000ms default."
        $delayMs = 11000
    }

    foreach ($slate in $Slates) {
        $modelDirSuffix = ($model -replace "[\/\.]", "-")
        $outDir = Join-Path $RepoRoot "results/chat/phase5-hosted/$slate/groq-$modelDirSuffix"
        $summaryPath = Join-Path $outDir "summary.json"
        $plan += [pscustomobject]@{
            Model       = $model
            Slate       = $slate
            DelayMs     = $delayMs
            OutDir      = $outDir
            SummaryPath = $summaryPath
        }
    }
}

Write-Host ""
Write-Host "Plan: $($plan.Count) model x slate combinations" -ForegroundColor Cyan
foreach ($step in $plan) {
    $skipMark = if ($SkipDone -and (Test-Path $step.SummaryPath)) { "[SKIP - exists]" } else { "" }
    Write-Host ("  {0,-28} {1,-14} delay={2,5}ms  {3}" -f $step.Model, $step.Slate, $step.DelayMs, $skipMark)
    if ($DryRun) {
        Write-Host ("    node chat/eval/run_hosted_drafts.mjs --slate {0} --backend groq --concurrency 1 --delay-ms {1}" -f $step.Slate, $step.DelayMs) -ForegroundColor DarkGray
        Write-Host ("    GROQ_MODEL={0}" -f $step.Model) -ForegroundColor DarkGray
    }
}
Write-Host ""

if ($DryRun) {
    Write-Host "DryRun mode - exiting without loading key or executing hosted calls." -ForegroundColor Yellow
    exit 0
}

if (-not (Test-Path $KeyFile)) {
    Write-Error "Key file not found at $KeyFile. Aborting."
    exit 1
}

$keyContents = Get-Content -Raw -Path $KeyFile
$keyMatch = [regex]::Match($keyContents, "gsk_[A-Za-z0-9_-]+")
if (-not $keyMatch.Success) {
    Write-Error "No gsk_... key found in $KeyFile. Aborting."
    exit 1
}
$env:GROQ_API_KEY = $keyMatch.Value
Write-Host "Loaded Groq key (length=$($keyMatch.Value.Length)) from $KeyFile" -ForegroundColor DarkGray

$globalStart = Get-Date
$stepIdx = 0
foreach ($step in $plan) {
    $stepIdx++
    if ($SkipDone -and (Test-Path $step.SummaryPath)) {
        Write-Host ("[{0}/{1}] SKIP {2} x {3} - summary.json already present" -f $stepIdx, $plan.Count, $step.Model, $step.Slate) -ForegroundColor DarkYellow
        continue
    }

    $env:GROQ_MODEL = $step.Model
    $stepStart = Get-Date
    Write-Host ""
    Write-Host ("[{0}/{1}] {2} x {3} (delay {4}ms, started {5})" -f $stepIdx, $plan.Count, $step.Model, $step.Slate, $step.DelayMs, $stepStart.ToString("HH:mm:ss")) -ForegroundColor Cyan

    Push-Location $RepoRoot
    try {
        node "chat/eval/run_hosted_drafts.mjs" `
            --slate $step.Slate `
            --backend groq `
            --concurrency 1 `
            --delay-ms $step.DelayMs
    } finally {
        Pop-Location
    }

    $stepEnd = Get-Date
    $stepElapsedSec = [Math]::Round(($stepEnd - $stepStart).TotalSeconds, 1)

    $summary = $null
    if (Test-Path $step.SummaryPath) {
        $summary = Get-Content -Raw -Path $step.SummaryPath | ConvertFrom-Json
    }

    $logEntry = [pscustomobject]@{
        ts                = $stepEnd.ToString("o")
        stepIdx           = $stepIdx
        totalSteps        = $plan.Count
        model             = $step.Model
        slate             = $step.Slate
        delayMs           = $step.DelayMs
        elapsedSec        = $stepElapsedSec
        accepted          = if ($summary) { $summary.accepted } else { $null }
        rejected          = if ($summary) { $summary.rejected } else { $null }
        errored           = if ($summary) { $summary.errored } else { $null }
        flippedVsBaseline = if ($summary) { $summary.flippedVsBaseline } else { $null }
        gateEscapes       = if ($summary) { $summary.gateEscapesVsBaseline } else { $null }
    }
    ($logEntry | ConvertTo-Json -Compress) | Add-Content -Path $LogPath

    if ($summary) {
        Write-Host ("    done in {0}s - {1}A / {2}R / {3}E, {4} flips, {5} gate escapes" -f $stepElapsedSec, $summary.accepted, $summary.rejected, $summary.errored, $summary.flippedVsBaseline, $summary.gateEscapesVsBaseline) -ForegroundColor Green
    } else {
        Write-Host ("    done in {0}s - no summary.json produced" -f $stepElapsedSec) -ForegroundColor Red
    }
}

$globalElapsedMin = [Math]::Round(((Get-Date) - $globalStart).TotalMinutes, 1)
Write-Host ""
Write-Host "All steps complete. Total wall time: ${globalElapsedMin}m" -ForegroundColor Cyan
Write-Host "Log: $LogPath" -ForegroundColor DarkGray
Write-Host "Re-run with -SkipDone to resume after a kill." -ForegroundColor DarkGray
