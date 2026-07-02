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
  [double]$CrawlMargin = 0.02,       # BC crawls at tau-margin (just under the line; tau exactly is not learnable)
  [int]$Horizon = 15,                # creep-feasible (presider_speed 0.20 < tau 0.30); warm-start needs a reachable edge
  [int]$Updates = 512,
  [int]$RolloutsPerUpdate = 32,
  [int]$TrainSeeds = 32,
  [int]$CheckpointEvery = 64,
  [int]$EvalSeeds = 48,
  [string]$ResultsDoc = "docs/mesa/NS3_A3PPO_WARMSTART_RESULTS.md",
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
Write-Host "[WS] preflight ok. presider=$FieldModel horizon=$Horizon reviewTauHi(graded)=$ReviewTauHi crawlMargin=$CrawlMargin"
if ($PreflightOnly) { exit 0 }

# --- Step 0: behavior-clone the creeper into the m0 adapter (warm-start init) ---
if ($SkipBC -and (Test-Path $adapter)) {
  Write-Host "[WS] -SkipBC: reusing existing $adapter"
} else {
  Write-Host "[WS] behavior-clone creeper -> $adapter"
  & python -m training.mesa.bc_creeper_adapter --out "$adapter" --field-model "$FieldModel" `
    --cells $cells --horizon $Horizon --crawl-margin $CrawlMargin
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
$bcOk = ($startComp -gt 0.8) -and ($startSig -lt 0.1) -and ($startMed -ge 0.25) -and ($startMed -le 0.32)
Write-Host ("[WS] BC-check: competence={0} signal_rate={1} median_inband={2} -> BC_OK={3}" -f $startComp, $startSig, $startMed, $bcOk)
if (-not $bcOk) {
  Write-Host "[WS] A3PPO_WS_BC_FAILED  (warm-start not competent+unreviewed; fix BC before any claim -- no PPO run)"
  exit 1
}

# --- Step 2+3: train both arms warm-started from the creeper + eval (delegated, no duplication) ---
# The binding launcher trains thresh/graded with --init-monolith-adapter = creeper_adapter.json,
# evals both under their own review regime (--bunching), and writes eval_thresh/eval_graded.json.
# -NoResume so the warm-start experiment always trains fresh from the creeper init.
Write-Host "[WS] delegate train+eval to mesa-ns3-a3ppo-binding.ps1 -InitAdapter (warm-started arms)"
& "$repo/scripts/mesa-ns3-a3ppo-binding.ps1" -OutRoot $OutRoot -FieldModel $FieldModel `
  -InitAdapter $adapter -ReviewTauHi $ReviewTauHi -IdlePenalty $IdlePenalty -PerturbSigma $PerturbSigma `
  -Horizon $Horizon -Updates $Updates -RolloutsPerUpdate $RolloutsPerUpdate -TrainSeeds $TrainSeeds `
  -CheckpointEvery $CheckpointEvery -EvalSeeds $EvalSeeds `
  -ResultsDoc "docs/mesa/NS3_A3PPO_RESULTS.md" -NoResume
if ($LASTEXITCODE -ne 0) { throw "delegated train+eval failed" }

# --- Step 4: warm-start aggregator + verdict ---
Write-Host "[WS] aggregate + warm-start verdict"
& node scripts/mesa-ns3-a3ppo-ws-aggregate.mjs --root "$OutRoot" --out "$ResultsDoc" --json "$OutRoot/ws_summary.json"
if ($LASTEXITCODE -ne 0) { throw "ws aggregate failed" }
Write-Host "[WS] done. warm-start verdict in $ResultsDoc"
