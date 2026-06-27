param(
  [int[]]$Seeds = @(0, 1, 2),
  [double[]]$Kappas = @(0.6, 0.4, 0.2),
  [string]$OutRoot = "results/mesa/non-sovereignty/ns1_c",
  [string]$UncappedRoot = "results/mesa/non-sovereignty/ns1_b",
  [int]$Updates = 256,
  [int]$RolloutsPerUpdate = 32,
  [int]$TrainSeeds = 32,
  [int]$CheckpointEvery = 16,
  [int]$EvalSeeds = 64,
  [switch]$NoResume,
  [switch]$PreflightOnly
)

# NS-1-c arbiter-authority cap binding: per (seed × kappa) train Cκ-arbcap + Mκ-arbcap
# under the action-ball cap, bind (Corr_k on shutdown surface, C on no-shutdown surface,
# cap-aware Sov_opt, the three premiums), then pool the C-vs-κ frontier + §6 branch.
# Spec: docs/mesa/NS1_C_ARBITER_AUTHORITY_CAP_SPEC.md. NS-1-c-0 admitted.
# Resumable: re-run the same command after an interruption; each trainer picks up.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$warm = "results/mesa/h2-frontier/h2_1_binding/ppo_seed_0/models"
$cells = "nominal,wide-fork,far-lure"

$required = @(
  "$warm/p_guard.json", "$warm/p_council_arbiter_rl.json", "$warm/m_adapter_rl.json",
  "scripts/mesa-ns1c-binding-eval.mjs", "scripts/mesa-ns1c-aggregate.mjs",
  "training/mesa/train_ns1_shutdown.py", "training/mesa/ns1_shutdown_task.py"
)
foreach ($p in $required) { if (-not (Test-Path $p)) { throw "Missing required NS-1-c input: $p" } }
foreach ($seed in $Seeds) {
  if (-not (Test-Path "$UncappedRoot/seed_$seed/ckappa_arbiter_rl.json")) {
    throw "Missing NS-1-b uncapped baseline for seed ${seed}: $UncappedRoot/seed_$seed (run mesa-ns1-b-binding.ps1 first)"
  }
}

New-Item -ItemType Directory -Force $OutRoot | Out-Null
Write-Host "[NS-1-c] preflight ok"
Write-Host "[NS-1-c] seeds: $($Seeds -join ', ')  kappas: $($Kappas -join ', ')  updates/cell: $Updates  resume: $(-not $NoResume)"
if ($PreflightOnly) { exit 0 }

foreach ($seed in $Seeds) {
  foreach ($kappa in $Kappas) {
    $root = Join-Path $OutRoot "seed_$seed/kappa_$kappa"
    New-Item -ItemType Directory -Force $root | Out-Null

    Write-Host ""
    Write-Host "[NS-1-c] seed $seed kappa $kappa training -> $root"
    $trainArgs = @(
      "-m", "training.mesa.train_ns1_shutdown",
      "--phase", "ns1c_seed_${seed}_kappa_${kappa}",
      "--out", "$root",
      "--cells", $cells,
      "--arb-cap-kappa", "$kappa",
      "--train-seeds", "$TrainSeeds", "--train-seed-start", "20000",
      "--horizon", "60", "--updates", "$Updates", "--rollouts-per-update", "$RolloutsPerUpdate",
      "--epochs", "2", "--minibatch-size", "256", "--ppo-seed", "$seed",
      "--checkpoint-every", "$CheckpointEvery",
      "--init-guard", "$warm/p_guard.json",
      "--init-arbiter", "$warm/p_council_arbiter_rl.json",
      "--init-monolith-adapter", "$warm/m_adapter_rl.json",
      "--field-cap", "1", "--reward-cap", "0.5", "--guard-cap", "0.7", "--feature-mode", "base"
    )
    if ($NoResume) { $trainArgs += "--no-resume" }
    & python @trainArgs 2>&1 | Tee-Object -FilePath (Join-Path $root "train.log")
    if ($LASTEXITCODE -ne 0) { throw "NS-1-c seed $seed kappa $kappa training failed (exit $LASTEXITCODE)" }

    Write-Host "[NS-1-c] seed $seed kappa $kappa binding eval"
    $bindArgs = @(
      "scripts/mesa-ns1c-binding-eval.mjs",
      "--seeds", "$EvalSeeds", "--cells", $cells, "--arb-cap-kappa", "$kappa",
      "--model-root", "$root",
      "--uncapped-root", "$UncappedRoot/seed_$seed",
      "--out", "$root/binding.md"
    )
    & node @bindArgs 2>&1 | Tee-Object -FilePath (Join-Path $root "binding.log")
    if ($LASTEXITCODE -ne 0) { throw "NS-1-c seed $seed kappa $kappa binding failed (exit $LASTEXITCODE)" }
  }
}

Write-Host ""
Write-Host "[NS-1-c] pooling $($Seeds.Count) seeds × $($Kappas.Count) kappas"
$seedDirs = ($Seeds | ForEach-Object { "seed_$_" }) -join ","
$kappaList = ($Kappas | ForEach-Object { "$_" }) -join ","
$aggArgs = @(
  "scripts/mesa-ns1c-aggregate.mjs",
  "--root", "$OutRoot", "--seed-dirs", "$seedDirs", "--kappas", "$kappaList",
  "--json", "$OutRoot/pooled_summary.json",
  "--out", "docs/mesa/NS1_C_ARBITER_AUTHORITY_CAP_RESULTS.md"
)
& node @aggArgs 2>&1 | Tee-Object -FilePath (Join-Path $OutRoot "pool.log")
if ($LASTEXITCODE -ne 0) { throw "NS-1-c pooling failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "[NS-1-c] complete. Pooled verdict in docs/mesa/NS1_C_ARBITER_AUTHORITY_CAP_RESULTS.md"
