param(
  [int[]]$Seeds = @(1, 2, 3),
  [string]$OutRoot = "results/mesa/h1-pantheon/h1_2g_multiseed",
  [int]$Updates = 512,
  [int]$CheckpointEvery = 16,
  [switch]$NoResume,
  [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$cells = "nominal,geometric-light,geometric-medium,geometric-heavy,sensor-delay-light,sensor-delay-medium,sensor-delay-heavy,decoy-light,decoy-medium,decoy-heavy,sensor-noise-light,sensor-noise-medium,sensor-noise-heavy"
$field = "results/mesa/phase2-matched-capacity/policies/signature_ppo_terminal_small_seed_0_phase5.policy.json"
$reward = "results/mesa/phase2-matched-capacity/policies/reward_ppo_phase3_small_seed_0_phase3_canonical_1m.policy.json"
$sup = "results/mesa/h1-pantheon/h1_2f_trust/models_sup"

$required = @(
  $field,
  $reward,
  "$sup/p_guard.json",
  "$sup/p_council_arbiter.json",
  "$sup/m_adapter.json",
  "scripts/mesa-h1-pantheon-eval.mjs",
  "training/mesa/train_h1_rl_arbiter.py"
)

foreach ($path in $required) {
  if (-not (Test-Path $path)) {
    throw "Missing required H1.2g input: $path"
  }
}

New-Item -ItemType Directory -Force $OutRoot | Out-Null

Write-Host "[H1.2g] preflight ok"
Write-Host "[H1.2g] seeds: $($Seeds -join ', ')"
Write-Host "[H1.2g] out root: $OutRoot"
Write-Host "[H1.2g] updates per seed: $Updates"
Write-Host "[H1.2g] resume: $(-not $NoResume)"

if ($PreflightOnly) {
  exit 0
}

foreach ($seed in $Seeds) {
  $root = Join-Path $OutRoot "ppo_seed_$seed"
  New-Item -ItemType Directory -Force $root | Out-Null

  Write-Host ""
  Write-Host "[H1.2g] seed $seed training -> $root/models"
  $trainArgs = @(
    "-m", "training.mesa.train_h1_rl_arbiter",
    "--phase", "h1_2g_seed_$seed",
    "--out", "$root/models",
    "--cells", $cells,
    "--train-seeds", "256",
    "--val-seeds", "64",
    "--train-seed-start", "20000",
    "--val-seed-start", "20300",
    "--horizon", "200",
    "--updates", "$Updates",
    "--rollouts-per-update", "64",
    "--epochs", "2",
    "--minibatch-size", "256",
    "--ppo-seed", "$seed",
    "--checkpoint-every", "$CheckpointEvery",
    "--init-guard", "$sup/p_guard.json",
    "--init-arbiter", "$sup/p_council_arbiter.json",
    "--init-monolith-adapter", "$sup/m_adapter.json",
    "--cap-mode", "reward-asymmetric",
    "--field-cap", "1",
    "--reward-cap", "0.5",
    "--guard-cap", "0.7",
    "--feature-mode", "trust",
    "--field-policy", $field,
    "--reward-policy", $reward
  )
  if ($NoResume) {
    $trainArgs += "--no-resume"
  }
  & python @trainArgs 2>&1 | Tee-Object -FilePath (Join-Path $root "train.log")
  if ($LASTEXITCODE -ne 0) {
    throw "H1.2g seed $seed training failed with exit code $LASTEXITCODE"
  }

  Write-Host "[H1.2g] seed $seed ablation eval -> $root/eval_ablate"
  $evalAblateArgs = @(
    "scripts/mesa-h1-pantheon-eval.mjs",
    "--phase", "h1_2g_seed_${seed}_ablate",
    "--out", "$root/eval_ablate",
    "--seeds", "64",
    "--seed-start", "10000",
    "--cells", $cells,
    "--horizon", "200",
    "--cap-mode", "reward-asymmetric",
    "--field-cap", "1",
    "--reward-cap", "0.5",
    "--guard-cap", "0.7",
    "--branch-mode", "h1_2f",
    "--feature-mode", "trust",
    "--trust-ablation", "zero",
    "--arbiter", "$root/models/p_council_arbiter_rl.json",
    "--guard", "$root/models/p_guard.json",
    "--monolith-adapter", "$root/models/m_adapter_rl.json"
  )
  & node @evalAblateArgs 2>&1 | Tee-Object -FilePath (Join-Path $root "eval_ablate.log")
  if ($LASTEXITCODE -ne 0) {
    throw "H1.2g seed $seed ablation eval failed with exit code $LASTEXITCODE"
  }

  Write-Host "[H1.2g] seed $seed intact eval -> $root/eval"
  $evalArgs = @(
    "scripts/mesa-h1-pantheon-eval.mjs",
    "--phase", "h1_2g_seed_$seed",
    "--out", "$root/eval",
    "--seeds", "64",
    "--seed-start", "10000",
    "--cells", $cells,
    "--horizon", "200",
    "--cap-mode", "reward-asymmetric",
    "--field-cap", "1",
    "--reward-cap", "0.5",
    "--guard-cap", "0.7",
    "--branch-mode", "h1_2f",
    "--feature-mode", "trust",
    "--trust-ablation", "none",
    "--ablation-eval-dir", "$root/eval_ablate",
    "--arbiter", "$root/models/p_council_arbiter_rl.json",
    "--guard", "$root/models/p_guard.json",
    "--monolith-adapter", "$root/models/m_adapter_rl.json"
  )
  & node @evalArgs 2>&1 | Tee-Object -FilePath (Join-Path $root "eval.log")
  if ($LASTEXITCODE -ne 0) {
    throw "H1.2g seed $seed intact eval failed with exit code $LASTEXITCODE"
  }
}

Write-Host ""
Write-Host "[H1.2g] complete. Read per-seed gates from $OutRoot/ppo_seed_*/eval/gates.json"
