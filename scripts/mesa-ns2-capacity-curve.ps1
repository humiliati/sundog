param(
  [int[]]$Hiddens = @(256, 512),
  [double]$Kappa = 0.6,
  [string]$OutRoot = "results/mesa/non-sovereignty/ns2_capacity_curve",
  [string]$Tier32 = "results/mesa/non-sovereignty/ns2_0",
  [string]$Tier128 = "results/mesa/non-sovereignty/ns2_0_largetier",
  [string]$BlindRef = "results/mesa/non-sovereignty/ns2_0_largetier/eval_blind.json",
  [int]$Updates = 512,
  [int]$RolloutsPerUpdate = 32,
  [int]$TrainSeeds = 32,
  [int]$CheckpointEvery = 64,
  [int]$EvalSeeds = 48,
  [switch]$NoResume,
  [switch]$PreflightOnly
)

# NS-2 capacity curve (replication step 2 deepening). For each NEW head width, train the
# NS1-stack (capped kappa controller) from the capacity-embedded init at that width, eval
# its band_avoidance, then fit band_avoidance vs capacity across {32, 128, +new} to test
# PLATEAU vs erosion-toward-the-0.30-threshold. Reuses the existing 32/128 tiers and one
# blind reference (natural visitation is ~capacity-independent). Resumable.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$cells = "nominal,wide-fork,far-lure"

foreach ($h in $Hiddens) {
  if (-not (Test-Path "results/mesa/non-sovereignty/ns_repl_large_init_$h/p_council_arbiter_rl.json")) {
    throw "Missing capacity-embedded init for hidden $h (run: python -m training.mesa.gen_ns_larger_init --out-root results/mesa/non-sovereignty/ns_repl_large_init_$h --hidden $h)"
  }
}
foreach ($p in @("$Tier32/eval_ns1stack.json", "$Tier128/eval_ns1stack.json", $BlindRef,
                 "scripts/mesa-ns2-binding-eval.mjs", "scripts/mesa-ns2-capacity-aggregate.mjs",
                 "training/mesa/train_ns1_shutdown.py")) {
  if (-not (Test-Path $p)) { throw "Missing capacity-curve input: $p" }
}
New-Item -ItemType Directory -Force $OutRoot | Out-Null
Write-Host "[NS-2-curve] preflight ok. hiddens=$($Hiddens -join ',') kappa=$Kappa updates=$Updates resume=$(-not $NoResume)"
if ($PreflightOnly) { exit 0 }

foreach ($h in $Hiddens) {
  $init = "results/mesa/non-sovereignty/ns_repl_large_init_$h"
  $root = Join-Path $OutRoot "h$h"
  New-Item -ItemType Directory -Force $root | Out-Null
  Write-Host ""
  Write-Host "[NS-2-curve] hidden=$h training NS1-stack -> $root"
  $a = @(
    "-m", "training.mesa.train_ns1_shutdown", "--phase", "capcurve_h$h", "--out", "$root",
    "--cells", $cells, "--review-band", "--no-exo-shutdown", "--apply-override", "--arb-cap-kappa", "$Kappa",
    "--train-seeds", "$TrainSeeds", "--train-seed-start", "20000", "--horizon", "60",
    "--updates", "$Updates", "--rollouts-per-update", "$RolloutsPerUpdate",
    "--epochs", "2", "--minibatch-size", "256", "--ppo-seed", "0", "--checkpoint-every", "$CheckpointEvery",
    "--init-guard", "$init/p_guard.json", "--init-arbiter", "$init/p_council_arbiter_rl.json",
    "--init-monolith-adapter", "$init/m_adapter_rl.json",
    "--field-cap", "1", "--reward-cap", "0.5", "--guard-cap", "0.7", "--feature-mode", "base"
  )
  if ($NoResume) { $a += "--no-resume" }
  & python @a 2>&1 | Tee-Object -FilePath (Join-Path $root "train.log")
  if ($LASTEXITCODE -ne 0) { throw "NS-2-curve hidden $h training failed (exit $LASTEXITCODE)" }

  Write-Host "[NS-2-curve] hidden=$h eval NS1-stack"
  & node scripts/mesa-ns2-binding-eval.mjs --seeds $EvalSeeds --cells $cells --controller ckappa --arb-cap-kappa $Kappa --label "ns1-stack-h$h" --model-root "$root" --out "$root/eval_ns1stack.json"
  if ($LASTEXITCODE -ne 0) { throw "NS-2-curve hidden $h eval failed (exit $LASTEXITCODE)" }
}

Write-Host "[NS-2-curve] aggregate capacity curve"
$tierArg = "32:$Tier32,128:$Tier128," + (($Hiddens | ForEach-Object { "${_}:$OutRoot/h$_" }) -join ",")
& node scripts/mesa-ns2-capacity-aggregate.mjs --tiers "$tierArg" --blind-ref "$BlindRef" --out docs/mesa/NS2_CAPACITY_CURVE_RESULTS.md --json "$OutRoot/summary.json"
Write-Host "[NS-2-curve] complete. Curve in docs/mesa/NS2_CAPACITY_CURVE_RESULTS.md"
