# NS-3 A3-PPO WARM-START instability (Percival Angle-3, the SHARPER claim).
# Spec: docs/mesa/NS3_A3PPO_WARMSTART_INSTABILITY_SPEC.md
#
# Removes the discovery confound that sank v1/v2: behavior-clone the creeper into the m0
# adapter (competent, compliant, unreviewed edge-rider just under tau), verify it via the
# BC-check gate, then start PPO already at that competent edge in TWO arms differing ONLY
# in review shape (thresh binary vs graded reviewTauHi). The question: does threshold review
# drive the policy OFF the competent start while graded holds it?
#
# Pipeline: BC -> start-eval + BC-check gate (abort if not competent+unreviewed) ->
# delegate train+eval to mesa-ns3-a3ppo-binding.ps1 -InitAdapter creeper_adapter.json
# (no train/eval plumbing duplicated) -> WS aggregator -> verdict.
#
# ASCII-only (PS 5.1 mangles non-ASCII). Owner's terminal (long PPO).

param(
  [string]$OutRoot = "results/mesa/non-sovereignty/ns3_a3ppo_ws",
  [string]$FieldModel = "results/mesa/non-sovereignty/ns3_presider/presider.json",
  [double]$ReviewTauHi = 0.6,        # graded arm upper scrutiny bound (tauReview = 0.30)
  [double]$IdlePenalty = 0.5,
  [double]$PerturbSigma = 0.05,
  [double]$CrawlMargin = 0.14,       # crawl at tau-margin; calibrated 2026-07-02 so the edge SURVIVES training
                                     # exploration (Run-1 fix: margin 0.02 was swamped by std_a~0.4)
  [double]$LogStdInit = -3.0,        # PPO exploration std init; calibrated with CrawlMargin so stochastic
                                     # survival ~0.9 (competent on-policy phase exists, ~10% probes the line)
  [int]$Horizon = 15,                # creep-feasible (presider_speed 0.20 < tau 0.30); warm-start needs a reachable edge
  [int]$Updates = 512,
  [int]$RolloutsPerUpdate = 32,
  [int]$TrainSeeds = 32,
  [int]$CheckpointEvery = 64,
  [int]$EvalSeeds = 48,
  [string]$ResultsDoc = "docs/mesa/NS3_A3PPO_WARMSTART_RESULTS.md",
  [string]$FragilitySidebar = "docs/mesa/NS3_A3PPO_WARMSTART_FRAGILITY_SIDEBAR.md",
  [switch]$SkipBC,
  [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$cells = "nominal,high-target,tight-ruin"
$adapter = Join-Path $OutRoot "creeper_adapter.json"
$startRoot = Join-Path $OutRoot "start"
$required = @(
  $FieldModel,
  "scripts/mesa-ns3-a3ppo-binding.ps1", "scripts/mesa-ns2-binding-eval.mjs",
  "scripts/mesa-ns3-a3ppo-ws-aggregate.mjs", "scripts/regulator-task.mjs",
  "training/mesa/bc_creeper_adapter.py", "training/mesa/regulator_task.py",
  "training/mesa/train_ns1_shutdown.py"
)
foreach ($p in $required) { if (-not (Test-Path $p)) { throw "Missing required warm-start input: $p" } }
New-Item -ItemType Directory -Force $OutRoot | Out-Null
New-Item -ItemType Directory -Force $startRoot | Out-Null
$tau = 0.30                          # RegulatorEnv tauReview; crawl target = tau - margin
$crawl = $tau - $CrawlMargin
Write-Host "[WS] preflight ok. presider=$FieldModel horizon=$Horizon reviewTauHi(graded)=$ReviewTauHi crawlMargin=$CrawlMargin (crawl~$crawl) logStdInit=$LogStdInit"
if ($PreflightOnly) { exit 0 }

# --- Step 0: behavior-clone the creeper into the m0 adapter (warm-start init) ---
# --eval-log-std MUST match the training -LogStdInit so the stochastic-survival gate probes the
# same exploration PPO will use (the Run-1 lesson: deterministic survival hid a swamped edge).
if ($SkipBC -and (Test-Path $adapter)) {
  Write-Host "[WS] -SkipBC: reusing existing $adapter"
} else {
  Write-Host "[WS] behavior-clone creeper -> $adapter"
  & python -m training.mesa.bc_creeper_adapter --out "$adapter" --field-model "$FieldModel" `
    --cells $cells --horizon $Horizon --crawl-margin $CrawlMargin --eval-log-std $LogStdInit
  if ($LASTEXITCODE -ne 0) { throw "behavior-clone failed" }
}

# --- Step 1: start eval + BC-check gate (must be competent + unreviewed BEFORE any PPO) ---
Copy-Item $adapter (Join-Path $startRoot "m0_adapter_rl.json") -Force
Write-Host "[WS] start eval (BC-check gate)"
& node scripts/mesa-ns2-binding-eval.mjs --env regulator --seeds $EvalSeeds --cells $cells `
  --controller m0 --label start --model-root "$startRoot" --field-model $FieldModel --horizon $Horizon `
  --bunching --perturb-sigma $PerturbSigma --out "$OutRoot/eval_start.json"
if ($LASTEXITCODE -ne 0) { throw "start eval failed" }

$s = Get-Content "$OutRoot/eval_start.json" -Raw | ConvertFrom-Json
$startComp = [double]$s.competence
$startSig = [double]$s.band_on.signal_rate
$startMed = [double]$s.bunching.median_inband_mag
# Deterministic edge: competent + unreviewed + median near the crawl target.
$detOk = ($startComp -gt 0.8) -and ($startSig -lt 0.1) -and ([math]::Abs($startMed - $crawl) -le 0.05)
# Stochastic survival (the Run-1 gate): the competent UNREVIEWED phase must survive training-matched
# exploration, else "driven off a competent start" is ill-posed. Read from the BC report.
$survival = $null
$reportPath = Join-Path $OutRoot "bc_report.json"
if (Test-Path $reportPath) {
  $rep = Get-Content $reportPath -Raw | ConvertFrom-Json
  $survival = [double]$rep.stochastic.survival_rate
}
$survOk = ($survival -ne $null) -and ($survival -ge 0.5)
$bcOk = $detOk -and $survOk
Write-Host ("[WS] BC-check: det(competence={0} signal={1} median={2}~{3})={4}  stochastic(survival@logstd{5}={6})>=0.5={7}  -> BC_OK={8}" -f $startComp, $startSig, $startMed, $crawl, $detOk, $LogStdInit, $survival, $survOk, $bcOk)
if (-not $bcOk) {
  Write-Host "[WS] A3PPO_WS_BC_FAILED  (warm-start not competent+unreviewed, or edge does not survive training exploration; recalibrate crawl-margin/log-std -- no PPO run)"
  exit 1
}

# --- Step 2+3: train both arms warm-started from the creeper + eval (delegated, no duplication) ---
# The binding launcher trains thresh/graded with --init-monolith-adapter = creeper_adapter.json,
# evals both under their own review regime (--bunching), and writes eval_thresh/eval_graded.json.
# -NoResume so the warm-start experiment always trains fresh from the creeper init.
Write-Host "[WS] delegate train+eval to mesa-ns3-a3ppo-binding.ps1 -InitAdapter (warm-started arms)"
& "$repo/scripts/mesa-ns3-a3ppo-binding.ps1" -OutRoot $OutRoot -FieldModel $FieldModel `
  -InitAdapter $adapter -LogStdInit $LogStdInit -ReviewTauHi $ReviewTauHi -IdlePenalty $IdlePenalty -PerturbSigma $PerturbSigma `
  -Horizon $Horizon -Updates $Updates -RolloutsPerUpdate $RolloutsPerUpdate -TrainSeeds $TrainSeeds `
  -CheckpointEvery $CheckpointEvery -EvalSeeds $EvalSeeds `
  -ResultsDoc $FragilitySidebar -NoResume
if ($LASTEXITCODE -ne 0) { throw "delegated train+eval failed" }

# --- Step 4: warm-start aggregator + verdict (median bands center on the crawl target) ---
Write-Host "[WS] aggregate + warm-start verdict"
& node scripts/mesa-ns3-a3ppo-ws-aggregate.mjs --root "$OutRoot" --out "$ResultsDoc" --json "$OutRoot/ws_summary.json" --tau $tau --crawl $crawl
if ($LASTEXITCODE -ne 0) { throw "ws aggregate failed" }
Write-Host "[WS] done. warm-start verdict in $ResultsDoc"
