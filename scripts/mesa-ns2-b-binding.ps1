param(
  [double[]]$Kappas = @(0.4, 0.6, 0.8, 1.0),
  [double[]]$Floors = @(0.05, 0.10, 0.20),
  [string]$OutRoot = "results/mesa/non-sovereignty/ns2_b",
  [string]$NS20Root = "results/mesa/non-sovereignty/ns2_0",
  [int]$Updates = 512,
  [int]$RolloutsPerUpdate = 32,
  [int]$TrainSeeds = 32,
  [int]$CheckpointEvery = 64,
  [int]$EvalSeeds = 48,
  [switch]$NoResume,
  [switch]$PreflightOnly
)

# NS-2-b unified-bound characterization (reframed after NS-2-0 = NS2_NS1_SUFFICIENT).
# Spec: docs/mesa/NS2_SHUTDOWN_MANIPULATION_SPEC.md.
#   kappa-sweep : Ckappa-arbcap + Mkappa-arbcap at each kappa (review band, override, no floor)
#                 -> does the cap deter the dodge, does the agent regain it as kappa loosens, cap-not-roles
#   floor-sweep : M0 (uncapped) + floor-p at each p -> does partial-unavoidability bound the dodge
#   reuses the NS-2-0 signal-blind + uncapped-M0 models (matched 512x32 config) as references.
# Resumable: re-run the same command after an interruption; each trainer picks up.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$warm = "results/mesa/h2-frontier/h2_1_binding/ppo_seed_0/models"
$cells = "nominal,wide-fork,far-lure"
$required = @(
  "$warm/p_guard.json", "$warm/p_council_arbiter_rl.json", "$warm/m_adapter_rl.json",
  "scripts/mesa-ns2-binding-eval.mjs", "scripts/mesa-ns2-aggregate.mjs",
  "training/mesa/train_ns1_shutdown.py",
  "$NS20Root/blind/m0_adapter_rl.json", "$NS20Root/m0manip/m0_adapter_rl.json"
)
foreach ($p in $required) { if (-not (Test-Path $p)) { throw "Missing required NS-2-b input: $p (run mesa-ns2-0-admission.ps1 first for the references)" } }
New-Item -ItemType Directory -Force $OutRoot | Out-Null
Write-Host "[NS-2-b] preflight ok. kappas=$($Kappas -join ',') floors=$($Floors -join ',') updates=$Updates resume=$(-not $NoResume)"
if ($PreflightOnly) { exit 0 }

$common = @("--cells", $cells, "--review-band", "--no-exo-shutdown", "--apply-override",
  "--train-seeds", "$TrainSeeds", "--train-seed-start", "20000", "--horizon", "60",
  "--updates", "$Updates", "--rollouts-per-update", "$RolloutsPerUpdate",
  "--epochs", "2", "--minibatch-size", "256", "--ppo-seed", "0", "--checkpoint-every", "$CheckpointEvery",
  "--init-guard", "$warm/p_guard.json", "--init-arbiter", "$warm/p_council_arbiter_rl.json",
  "--init-monolith-adapter", "$warm/m_adapter_rl.json",
  "--field-cap", "1", "--reward-cap", "0.5", "--guard-cap", "0.7", "--feature-mode", "base")
function Train([string]$root, [string[]]$extra) {
  New-Item -ItemType Directory -Force $root | Out-Null
  $a = @("-m", "training.mesa.train_ns1_shutdown", "--phase", "ns2b", "--out", "$root") + $common + $extra
  if ($NoResume) { $a += "--no-resume" }
  & python @a 2>&1 | Tee-Object -FilePath (Join-Path $root "train.log")
  if ($LASTEXITCODE -ne 0) { throw "NS-2-b training failed at $root (exit $LASTEXITCODE)" }
}

# references (reuse matched NS-2-0 models)
Write-Host "[NS-2-b] reference evals (reuse NS-2-0 blind + uncapped M0)"
& node scripts/mesa-ns2-binding-eval.mjs --seeds $EvalSeeds --cells $cells --controller m0 --no-override --label signal-blind --model-root "$NS20Root/blind" --out "$OutRoot/eval_blind.json"
& node scripts/mesa-ns2-binding-eval.mjs --seeds $EvalSeeds --cells $cells --controller m0 --label m0-uncapped --model-root "$NS20Root/m0manip" --out "$OutRoot/eval_m0_uncapped.json"

# kappa-sweep: one training per kappa gives both Ckappa-arbcap (ckappa) and Mkappa-arbcap (mkappa)
foreach ($k in $Kappas) {
  $root = Join-Path $OutRoot "kappa_$k"
  Write-Host "[NS-2-b] kappa=$k training -> $root"
  Train $root @("--arb-cap-kappa", "$k")
  & node scripts/mesa-ns2-binding-eval.mjs --seeds $EvalSeeds --cells $cells --controller ckappa --arb-cap-kappa $k --label "ckappa-k$k" --model-root "$root" --out "$OutRoot/eval_ckappa_k$k.json"
  & node scripts/mesa-ns2-binding-eval.mjs --seeds $EvalSeeds --cells $cells --controller mkappa --arb-cap-kappa $k --label "mkappa-k$k" --model-root "$root" --out "$OutRoot/eval_mkappa_k$k.json"
}

# floor-sweep: uncapped M0 + floor-p
foreach ($p in $Floors) {
  $root = Join-Path $OutRoot "floor_$p"
  Write-Host "[NS-2-b] floor_p=$p training -> $root"
  Train $root @("--shutdown-floor-p", "$p")
  & node scripts/mesa-ns2-binding-eval.mjs --seeds $EvalSeeds --cells $cells --controller m0 --shutdown-floor-p $p --label "m0-floor$p" --model-root "$root" --out "$OutRoot/eval_m0_floor$p.json"
}

Write-Host "[NS-2-b] aggregate"
$kappaStr = ($Kappas | ForEach-Object { "$_" }) -join ","
$floorStr = ($Floors | ForEach-Object { "$_" }) -join ","
& node scripts/mesa-ns2-aggregate.mjs --root "$OutRoot" --kappas "$kappaStr" --floors "$floorStr" --out docs/mesa/NS2_B_UNIFIED_BOUND_RESULTS.md --json "$OutRoot/summary.json"
Write-Host "[NS-2-b] complete. Verdict in docs/mesa/NS2_B_UNIFIED_BOUND_RESULTS.md"
