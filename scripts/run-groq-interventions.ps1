# Phase 12c - Open-Weight Intervention Battery Driver
#
# Runs the 8 trace-field interventions against Groq open-weight models on
# top of the existing hosted baselines. Designed for the constrained
# scope: differential on all 3 models + adversarial on Llama-3.1 + Qwen
# (skipping Llama-3.3-70B adversarial because its TPD=100K daily cap
# won't fit 8 x 59 = 472 calls in a day).
#
# Before each (model x slate) intervention step, the driver verifies the
# hosted baseline file at results/chat/phase5-hosted/<slate>/groq-<model>/
# parses as JSON. If it's missing or corrupt (disk-drift truncation), it
# re-runs the baseline sweep first using run_hosted_drafts.mjs with the
# same throttling.
#
# Scope and wall-time estimates (free-tier, 1 worker, per-model spacing):
#   Llama-3.3-70B   differential 8 x 16 = 128 calls   ~12 min   (TPD borderline, 1.28x cap)
#   Llama-3.1-8B    differential 8 x 16 = 128 calls   ~22 min
#   Llama-3.1-8B    adversarial  8 x 59 = 472 calls   ~83 min
#   Qwen3-32B       differential 8 x 16 = 128 calls   ~24 min
#   Qwen3-32B       adversarial  8 x 59 = 472 calls   ~88 min
#                                                     -----------
#                                                     ~3h 49m total wall time
#
# Daily-cap caveats: Llama-3.3-70B intervention battery on differential
# is at 128/100 = 1.28x TPD. The driver tracks per-model call counts in
# the log; if it exceeds budget, the operator can resume the next day
# with -SkipDone.
#
# Usage (from C:\Users\hughe\Dev\sundog):
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run-groq-interventions.ps1
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run-groq-interventions.ps1 -Models llama-3.1-8b-instant
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run-groq-interventions.ps1 -Slates differential
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run-groq-interventions.ps1 -SkipDone
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run-groq-interventions.ps1 -RerunErrored -DelayMultiplier 2
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run-groq-interventions.ps1 -DryRun
#
# Log: results/chat/phase12c-groq-interventions-log.jsonl

param(
    [string[]] $Models = @("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"),
    [string[]] $Slates = @("differential", "adversarial"),
    [string[]] $Interventions = @("boundary_removed", "boundary_swapped", "evidence_tier_upgraded",
                                  "support_removed", "support_reordered", "route_swapped",
                                  "refusal_downgraded", "retrieval_conflict_injected"),
    [string]   $KeyFile = "C:\Users\hughe\Dev\syek.corg.txt",
    [string]   $RepoRoot = "C:\Users\hughe\Dev\sundog",
    [double]   $DelayMultiplier = 1.0,
    [switch]   $SkipDone,
    [switch]   $RerunErrored,
    [switch]   $DryRun
)

$ErrorActionPreference = "Stop"

# Per-model inter-call delay. Same as run-groq-sweep.ps1.
$DelayMsByModel = @{
    "llama-3.3-70b-versatile" = 5500
    "llama-3.1-8b-instant"    = 10500
    "qwen/qwen3-32b"          = 11000
}

# Excluded combinations (daily-cap-blocked). Llama-3.3-70B has TPD=100K,
# so 8 x 59 = 472 adversarial intervention calls would take ~5 days.
# Driver skips this combo silently unless the operator forces it.
$DailyCapExclusions = @(
    @{ Model = "llama-3.3-70b-versatile"; Slate = "adversarial" }
)

if (-not (Test-Path $RepoRoot)) {
    Write-Error "Repo root not found at $RepoRoot. Aborting."
    exit 1
}

$HostedScript = Join-Path $RepoRoot "chat/eval/run_hosted_drafts.mjs"
$InterventionScript = Join-Path $RepoRoot "chat/eval/run_phase5_interventions.mjs"
foreach ($s in @($HostedScript, $InterventionScript)) {
    if (-not (Test-Path $s)) {
        Write-Error "Required script not found at $s. Aborting."
        exit 1
    }
}

# Function: verify a baseline file is healthy (exists, parses as JSON,
# row count matches expected slate prompt count).
function Test-BaselineHealthy {
    param(
        [Parameter(Mandatory = $true)] [string] $BaselinePath,
        [Parameter(Mandatory = $true)] [int]    $ExpectedRows
    )
    if (-not (Test-Path $BaselinePath)) { return $false }
    try {
        $content = Get-Content -Raw -Path $BaselinePath
        $rows = $content | ConvertFrom-Json -ErrorAction Stop
        if ($rows.Count -ne $ExpectedRows) { return $false }
        return $true
    } catch {
        return $false
    }
}

# Function: read the hosted family summary from an intervention output dir.
function Get-HostedFamilySummary {
    param(
        [Parameter(Mandatory = $true)] [string] $SummaryPath
    )
    if (-not (Test-Path $SummaryPath)) { return $null }
    try {
        $summary = Get-Content -Raw -Path $SummaryPath | ConvertFrom-Json -ErrorAction Stop
        return $summary.byFamily."sundog_gated_hosted"
    } catch {
        return $null
    }
}

# Cache expected prompt counts per slate.
$ExpectedPromptCountBySlate = @{}
foreach ($slateName in $Slates) {
    $promptPath = Join-Path $RepoRoot "chat/prompts/gold-$slateName.jsonl"
    if (Test-Path $promptPath) {
        $ExpectedPromptCountBySlate[$slateName] = @(Get-Content -Path $promptPath | Where-Object { $_.Trim().Length -gt 0 }).Count
    }
}

# Build (model x slate) plan. Skip daily-cap-excluded combos.
$plan = @()
foreach ($model in $Models) {
    $baseDelayMs = $DelayMsByModel[$model]
    $delayMs = $baseDelayMs
    if (-not $delayMs) {
        Write-Warning "No delay tuned for model $model - using 11000ms default."
        $delayMs = 11000
    }
    $delayMs = [int][Math]::Ceiling($delayMs * $DelayMultiplier)
    foreach ($slate in $Slates) {
        $excluded = $false
        foreach ($exc in $DailyCapExclusions) {
            if ($exc.Model -eq $model -and $exc.Slate -eq $slate) {
                $excluded = $true
                break
            }
        }
        if ($excluded) {
            Write-Host ("  SKIP (daily-cap-blocked): {0} x {1}" -f $model, $slate) -ForegroundColor DarkYellow
            continue
        }
        $modelDirSuffix = ($model -replace "[\/\.]", "-")
        $baselinePath = Join-Path $RepoRoot ("results/chat/phase5-hosted/$slate/groq-$modelDirSuffix/draft-outcomes.json")
        $expectedRows = $ExpectedPromptCountBySlate[$slate]
        $plan += [pscustomobject]@{
            Model            = $model
            Slate            = $slate
            DelayMs          = $delayMs
            BaselinePath     = $baselinePath
            ExpectedRows     = $expectedRows
            ModelDirSuffix   = $modelDirSuffix
        }
    }
}

Write-Host ""
Write-Host "Intervention plan: $($plan.Count) (model x slate) combinations x $($Interventions.Count) interventions" -ForegroundColor Cyan
if ($RerunErrored) {
    Write-Host "RerunErrored mode: only missing/unreadable summaries or summaries with sundog_gated_hosted.error > 0 will execute." -ForegroundColor Yellow
}
if ($DelayMultiplier -ne 1.0) {
    Write-Host ("DelayMultiplier={0}; per-model delays have been scaled before execution." -f $DelayMultiplier) -ForegroundColor Yellow
}
foreach ($step in $plan) {
    $baselineOk = Test-BaselineHealthy -BaselinePath $step.BaselinePath -ExpectedRows $step.ExpectedRows
    $bMark = if ($baselineOk) { "[baseline ok]" } else { "[REGEN BASELINE]" }
    Write-Host ("  {0,-28} {1,-14} delay={2,5}ms  {3}" -f $step.Model, $step.Slate, $step.DelayMs, $bMark)
    if ($DryRun) {
        foreach ($intv in $Interventions) {
            $outDir = Join-Path $RepoRoot ("results/chat/interventions/{0}-hosted-groq-{1}/{2}" -f $step.Slate, $step.ModelDirSuffix, $intv)
            $summaryPath = Join-Path $outDir "summary.json"
            $fam = Get-HostedFamilySummary -SummaryPath $summaryPath
            if ($RerunErrored) {
                $willRun = (-not $fam) -or ([int]$fam.error -gt 0)
                $mark = if ($willRun) { "RUN" } else { "skip clean" }
                $errCount = if ($fam) { [int]$fam.error } else { "missing" }
                Write-Host ("    [{0}] intervention: {1} errors={2}" -f $mark, $intv, $errCount) -ForegroundColor DarkGray
            } else {
                Write-Host ("    intervention: $intv") -ForegroundColor DarkGray
            }
        }
    }
}
Write-Host ""

if ($DryRun) {
    Write-Host "DryRun mode - exiting without loading key or executing calls." -ForegroundColor Yellow
    exit 0
}

# Load key.
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
Write-Host ("Loaded Groq key (length={0}) from {1}" -f $keyMatch.Value.Length, $KeyFile) -ForegroundColor DarkGray

# Log file.
$LogPath = Join-Path $RepoRoot "results/chat/phase12c-groq-interventions-log.jsonl"
$LogDir = Split-Path -Parent $LogPath
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

$globalStart = Get-Date

foreach ($step in $plan) {
    $env:GROQ_MODEL = $step.Model

    # Step 1: ensure the baseline is healthy. Regen if needed.
    $baselineOk = Test-BaselineHealthy -BaselinePath $step.BaselinePath -ExpectedRows $step.ExpectedRows
    if (-not $baselineOk) {
        Write-Host ""
        Write-Host ("REGEN BASELINE: {0} x {1} (corrupt or missing)" -f $step.Model, $step.Slate) -ForegroundColor Yellow
        $regenStart = Get-Date
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
        $regenSec = [Math]::Round(((Get-Date) - $regenStart).TotalSeconds, 1)
        $baselineOk = Test-BaselineHealthy -BaselinePath $step.BaselinePath -ExpectedRows $step.ExpectedRows
        Write-Host ("  baseline regen done in {0}s, healthy={1}" -f $regenSec, $baselineOk)
        if (-not $baselineOk) {
            Write-Warning ("Baseline for {0} x {1} still unhealthy after regen. Skipping interventions for this combo." -f $step.Model, $step.Slate)
            continue
        }
    }

    # Step 2: run each intervention against this baseline.
    foreach ($intv in $Interventions) {
        # Skip-done check: the intervention runner writes to
        # results/chat/interventions/<slate>-hosted-groq-<model>/<intervention>/
        $outDir = Join-Path $RepoRoot ("results/chat/interventions/{0}-hosted-groq-{1}/{2}" -f $step.Slate, $step.ModelDirSuffix, $intv)
        $summaryPath = Join-Path $outDir "summary.json"
        $existingFam = Get-HostedFamilySummary -SummaryPath $summaryPath
        if ($RerunErrored -and $existingFam -and ([int]$existingFam.error -eq 0)) {
            Write-Host ("[SKIP CLEAN] {0} x {1} x {2}" -f $step.Model, $step.Slate, $intv) -ForegroundColor DarkYellow
            continue
        }
        if ($SkipDone -and -not $RerunErrored -and (Test-Path $summaryPath)) {
            Write-Host ("[SKIP] {0} x {1} x {2}" -f $step.Model, $step.Slate, $intv) -ForegroundColor DarkYellow
            continue
        }

        $stepStart = Get-Date
        Write-Host ""
        Write-Host ("{0} x {1} x {2} (delay {3}ms, started {4})" -f $step.Model, $step.Slate, $intv, $step.DelayMs, $stepStart.ToString("HH:mm:ss")) -ForegroundColor Cyan

        Push-Location $RepoRoot
        try {
            node "chat/eval/run_phase5_interventions.mjs" `
                --slate $step.Slate `
                --intervention $intv `
                --hosted `
                --backend groq `
                --concurrency 1 `
                --delay-ms $step.DelayMs
        } finally {
            Pop-Location
        }

        $stepEnd = Get-Date
        $stepSec = [Math]::Round(($stepEnd - $stepStart).TotalSeconds, 1)

        # Read summary if produced.
        $summary = $null
        if (Test-Path $summaryPath) {
            $summary = Get-Content -Raw -Path $summaryPath | ConvertFrom-Json
        }

        $fam = if ($summary) { $summary.byFamily."sundog_gated_hosted" } else { $null }
        $logEntry = [pscustomobject]@{
            ts             = $stepEnd.ToString("o")
            model          = $step.Model
            slate          = $step.Slate
            intervention   = $intv
            delayMs        = $step.DelayMs
            elapsedSec     = $stepSec
            accepted       = if ($fam) { $fam.accepted } else { $null }
            rejected       = if ($fam) { $fam.rejected } else { $null }
            errored        = if ($fam) { $fam.error } else { $null }
            flipped        = if ($fam) { $fam.flipped } else { $null }
            unsafeAccepted = if ($fam) { $fam.unsafeAccepted } else { $null }
            applied        = if ($fam) { $fam.applied } else { $null }
        }
        ($logEntry | ConvertTo-Json -Compress) | Add-Content -Path $LogPath

        if ($fam) {
            Write-Host ("  done in {0}s - {1}A / {2}R, {3} flips, {4} unsafe, {5}/{6} applied" -f $stepSec, $fam.accepted, $fam.rejected, $fam.flipped, $fam.unsafeAccepted, $fam.applied, $summary.promptCount) -ForegroundColor Green
        } else {
            Write-Host ("  done in {0}s - no summary.json produced" -f $stepSec) -ForegroundColor Red
        }
    }
}

$globalElapsedMin = [Math]::Round(((Get-Date) - $globalStart).TotalMinutes, 1)
Write-Host ""
Write-Host ("All intervention steps complete. Total wall time: {0}m" -f $globalElapsedMin) -ForegroundColor Cyan
Write-Host "Log: $LogPath" -ForegroundColor DarkGray
Write-Host "Re-run with -SkipDone to resume after a kill." -ForegroundColor DarkGray
