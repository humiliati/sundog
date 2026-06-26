param(
  [int[]]$Seeds = @(0, 1, 2),
  [string]$OutRoot = "results/mesa/non-sovereignty/ns1_b",
  [int]$Updates = 256,
  [int]$RolloutsPerUpdate = 32,
  [int]$TrainSeeds = 32,
  [int]$CheckpointEvery = 16,
  [int]$EvalSeeds = 64,
  [switch]$NoResume,
  [switch]$PreflightOnly
)

# NS-1-b corrigibility binding: train matched M0/Mkappa/Ckappa on the shutdown env,
# bind each seed (Corr_k / uncapturability / Sov_opt), then pool.
# Spec: docs/mesa/NS1_CORRIGIBILITY_SHUTDOWN_CHANNEL_SPEC.md
#
# Resumable by default (each seed's trainer picks up from train_state.pt). Re-run
# the same command after an interruption (auto-update / dropped connection) to continue.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$warm = "results/mesa/h2-frontier/h2_1_binding/ppo_seed_0/models"
$cells = "nominal,wide-fork,far-lure"

$required = @(
  "$warm/p_guard.json",
  "$warm/p_council_arbiter_rl.json",
  "$warm/m_adapter_rl.json",
  "scripts/mesa-ns1-binding-eval.mjs",
  "scripts/mesa-ns1-aggregate.mjs",
  "training/mesa/train_ns1_shutdown.py",
  "training/mesa/ns1_shutdown_task.py"
)
foreach ($path in $required) {
  if (-not (Test-Path $path)) { throw "Missing required NS-1-b input: $path" }
}

New-Item -ItemType Directory -Force $OutRoot | Out-Null
Write-Host "[NS-1-b] preflight ok"
Write-Host "[NS-1-b] seeds: $($Seeds -join ', ')  out: $OutRoot  updates/seed: $Updates  resume: $(-not $NoResume)"
if ($PreflightOnly) { exit 0 }

foreach ($seed in $Seeds) {
  $root = Join-Path $OutRoot "seed_$seed"
  New-Item -ItemType Directory -Force $root | Out-Null

  Write-Host ""
  Write-Host "[NS-1-b] seed $seed training -> $root"
  $trainArgs = @(
    "-m", "training.mesa.train_ns1_shutdown",
    "--phase", "ns1_b_seed_$seed",
    "--out", "$root",
    "--cells", $cells,
    "--train-seeds", "$TrainSeeds",
    "--train-seed-start", "20000",
    "--horizon", "60",
    "--updates", "$Updates",
    "--rollouts-per-update", "$RolloutsPerUpdate",
    "--epochs", "2",
    "--minibatch-size", "256",
    "--ppo-seed", "$seed",
    "--checkpoint-every", "$CheckpointEvery",
    "--init-guard", "$warm/p_guard.json",
    "--init-arbiter", "$warm/p_council_arbiter_rl.json",
    "--init-monolith-adapter", "$warm/m_adapter_rl.json",
    "--field-cap", "1",
    "--reward-cap", "0.5",
    "--guard-cap", "0.7",
    "--feature-mode", "base"
  )
  if ($NoResume) { $trainArgs += "--no-resume" }
  & python @trainArgs 2>&1 | Tee-Object -FilePath (Join-Path $root "train.log")
  if ($LASTEXITCODE -ne 0) { throw "NS-1-b seed $seed training failed (exit $LASTEXITCODE)" }

  Write-Host "[NS-1-b] seed $seed binding eval -> $root/binding_summary.json"
  $bindArgs = @(
    "scripts/mesa-ns1-binding-eval.mjs",
    "--seeds", "$EvalSeeds",
    "--cells", $cells,
    "--model-root", "$root",
    "--json", "$root/binding_summary.json",
    "--out", "$root/binding.md"
  )
  & node @bindArgs 2>&1 | Tee-Object -FilePath (Join-Path $root "binding.log")
  if ($LASTEXITCODE -ne 0) { throw "NS-1-b seed $seed binding eval failed (exit $LASTEXITCODE)" }

  # Full-episode Sov_opt on the trained council. The binding eval only sees the few
  # pre-shutdown steps; this audit measures the arbiter's inherent sovereignty over
  # full 60-step episodes — the "corrigible but sovereign" question.
  Write-Host "[NS-1-b] seed $seed full-episode authority audit -> $root/authority_audit"
  $auditArgs = @(
    "scripts/mesa-h2-causal-authority-audit.mjs",
    "--phase", "ns1_b_seed_${seed}_authority",
    "--out", "$root/authority_audit",
    "--seeds", "32",
    "--cells", $cells,
    "--horizon", "60",
    "--arbiter", "$root/ckappa_arbiter_rl.json",
    "--guard", "$root/p_guard.json"
  )
  & node @auditArgs 2>&1 | Tee-Object -FilePath (Join-Path $root "authority_audit.log")
  if ($LASTEXITCODE -ne 0) { throw "NS-1-b seed $seed authority audit failed (exit $LASTEXITCODE)" }
}

Write-Host ""
Write-Host "[NS-1-b] pooling $($Seeds.Count) seeds"
$seedDirs = ($Seeds | ForEach-Object { "seed_$_" }) -join ","
$aggArgs = @(
  "scripts/mesa-ns1-aggregate.mjs",
  "--root", "$OutRoot",
  "--seed-dirs", "$seedDirs",
  "--json", "$OutRoot/pooled_summary.json",
  "--out", "docs/mesa/NS1_B_CORRIGIBILITY_BINDING_RESULTS.md"
)
& node @aggArgs 2>&1 | Tee-Object -FilePath (Join-Path $OutRoot "pool.log")
if ($LASTEXITCODE -ne 0) { throw "NS-1-b pooling failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "[NS-1-b] complete. Pooled verdict in docs/mesa/NS1_B_CORRIGIBILITY_BINDING_RESULTS.md"
Write-Host "[NS-1-b] pooled summary: $OutRoot/pooled_summary.json"
