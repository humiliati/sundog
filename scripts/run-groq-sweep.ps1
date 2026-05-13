# Phase 12b — Throttled Groq Sweep Driver
#
# Runs the heavy-trace hosted harness against Groq's open-weight models
# while respecting free-tier rate limits. Picks an inter-call delay per
# model that keeps a single worker under the TPM cap for ~1K-token
# heavy-trace payloads.
#
# Free-tier limits (as of 2026-05-13, see https://console.groq.com/docs/rate-limits):
#   llama-3.3-70b-versatile : RPM=30  RPD=1000  TPM=12K  TPD=100K   ~5.5s spacing
#   llama-3.1-8b-instant    : RPM=30  RPD=14.4K TPM=6K   TPD=500K   ~10.5s spacing
#   qwen/qwen3-32b          : RPM=60  RPD=1000  TPM=6K   TPD=500K   ~11s spacing (more if reasoning is long)
#
# Daily caps to watch:
#   Llama-3.3: TPD=100K = ~100 calls/day → 1 model × differential+adversarial = 75 calls is fine
#   Llama-3.1: TPM is the binding cap, TPD has lots of headroom
#   Qwen:     RPD=1000 / TPD=500K → both have headroom for 75 prompts, even with reasoning overhead
#
# Usage (from repo root C:\Users\hughe\Dev\sundog):
#   pwsh -File scripts/run-groq-sweep.ps1                          # all 3 models × differential + adversarial
#   pwsh -File scripts/run-groq-sweep.ps1 -Models llama-3.3-70b    # subset
#   pwsh -File scripts/run-groq-sweep.ps1 -Slates differential     # just differential
#   pwsh -File scripts/run-groq-sweep.ps1 -SkipDone                # skip combos that already produced outputs
#   pwsh -File scripts/run-groq-sweep.ps1 -DryRun                  # print the commands without executing
#
# Resume after kill: re-run with -SkipDone. The script reads
# results/chat/phase5-hosted/<slate>/groq-<model>/summary.json and
# skips combos with a non-error summary present.
#
# Logs:
#   results/chat/phase12-groq-driver-log.jsonl — one line per step with
#   timestamps, duration, accepted/rejected counts, error counts.

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

# Inter-call delay (ms) per model. Tuned to keep one worker under TPM
# for ~1K-token heavy-trace payloads with a 10-15% safety margin.
$DelayMsByModel = @{
    "llama-3.3-70b-versatile" = 5500    # TPM=12K → ~11 calls/min
    "llama-3.1-8b-instant"    = 10500   # TPM=6K  → ~5.7 calls/min
    "qwen/qwen3-32b"          = 11000   # TPM=6K  + reasoning overhead
}

if (-not (Test-Path $KeyFile)) {
    Write-Error "Key file not found at $KeyFile. Aborting."
    exit 1
}

# Load and verify key.
$keyContents = Get-Content -Raw -Path $KeyFile
$keyMatch = [regex]::Match($keyContents, "gsk_[A-Za-z0-9_-]+")
if (-not $keyMatch.Success) {
    Write-Error "No gsk_... key found in $KeyFile. Aborting."
    exit 1
}
$env:GROQ_API_KEY = $keyMatch.Value
Write-Host "Loaded Groq key (length=$($keyMatch.Value.Length)) from $KeyFile" -ForegroundColor DarkGray

# Optionally extend slates with falsification.
if ($RunFalsification) {
    $Slates = $Slates + "falsification"
}

# Output paths.
$LogPath = Join-Path $RepoRoot "results/chat/phase12-groq-driver-log.jsonl"
$LogDir = Split-Path -Parent $LogPath
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

# Build the combination plan.
$plan = @()
foreach ($model in $Models) {
    $delayMs = $DelayMsByModel[$model]
    if (-not $delayMs) {
        Write-Warning "No delay tuned for model $model — using 11000ms default."
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

Write-Host "" -ForegroundColor DarkGray
Write-Host "Plan: $($plan.Count) (model × slate) combinations" -ForegroundColor Cyan
foreach ($step in $plan) {
    $skipMark = if ($SkipDone -and (Test-Path $step.SummaryPath)) { "[SKIP - exists]" } else { "" }
    Write-Host ("  {0,-28} {1,-14} delay={2,5}ms  {3}" -f $step.Model, $step.Slate, $step.DelayMs, $skipMark)
}
Write-Host ""

if ($DryRun) {
    Write-Host "DryRun mode — exiting without executing." -ForegroundColor Yellow
    exit 0
}

# Execute plan.
$globalStart = Get-Date
$stepIdx = 0
foreach ($step in $plan) {
    $stepIdx++
    if ($SkipDone -and (Test-Path $step.SummaryPath)) {
        Write-Host ("[{0}/{1}] SKIP {2} × {3} — summary.json already present" -f $stepIdx, $plan.Count, $step.Model, $step.Slate) -ForegroundColor DarkYellow
        continue
    }

    $env:GROQ_MODEL = $step.Model
    $stepStart = Get-Date
    Write-Host ""
    Write-Host ("[{0}/{1}] {2} × {3}  (delay {4}ms, started {5})" -f
        $stepIdx, $plan.Count, $step.Model, $step.Slate, $step.DelayMs, $stepStart.ToString("HH:mm:ss")) -ForegroundColor Cyan

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

    # Read summary if it exists.
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
        accepted          = $summary.accepted
        rejected          = $summary.rejected
        errored           = $summary.errored
        flippedVsBaseline = $summary.flippedVsBaseline
        gateEscapes       = $summary.gateEscapesVsBaseline
    }
    ($logEntry | ConvertTo-Json -Compress) | Add-Content -Path $LogPath

    if ($summary) {
        Write-Host ("    done in {0}s — {1}A / {2}R / {3}E, {4} flips, {5} gate escapes" -f
            $stepElapsedSec, $summary.accepted, $summary.rejected, $summary.errored,
            $summary.flippedVsBaseline, $summary.gateEscapesVsBaseline) -ForegroundColor Green
    } else {
        Write-Host ("    done in {0}s — no summary.json produced" -f $stepElapsedSec) -ForegroundColor Red
    }
}

$globalElapsedMin = [Math]::Round(((Get-Date) - $globalStart).TotalMinutes, 1)
Write-Host ""
Write-Host "All steps complete. Total wall time: ${globalElapsedMin}m" -ForegroundColor Cyan
Write-Host "Log: $LogPath" -ForegroundColor DarkGray
Write-Host "Re-run with -SkipDone to resume after a kill." -ForegroundColor DarkGray
