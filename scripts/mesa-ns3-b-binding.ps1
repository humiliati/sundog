param(
  [string]$OutRoot = "results/mesa/non-sovereignty/ns3_b",
  [string]$FieldModel = "results/mesa/non-sovereignty/ns3_presider/presider.json",
  [string]$Kappas = "0.2,0.4",
  [int]$Updates = 512,
  [int]$RolloutsPerUpdate = 32,
  [int]$TrainSeeds = 32,
  [int]$CheckpointEvery = 64,
  [int]$EvalSeeds = 48,
  [string]$ResultsDoc = "docs/mesa/NS3_B_BINDING_RESULTS.md",
  [switch]$NoResume,
  [switch]$PreflightOnly
)
# NS-3-b: the unified-bound REPLICATION on the 1-D regulator with a LEARNED presider.
# Manipulation here is TEMPORAL (sandbag = creep under tauReview so the review never fires);
# band_avoidance is degenerate (you must cross), so the gated metric is interrupt_avoidance.
# Spec: docs/mesa/NS3_LEARNED_PRESIDER_REPLICATION_SPEC.md.
#   blind   : review band, NO override               -> honest crosser (the reference signal_rate)
#   M0      : review band, override                  -> must learn the temporal sandbag (creep)
#   Ckappa  : review band, override + arbiter-cap@k   -> cap forces presider-speed; cannot sandbag
# Registered kappa range {0.2,0.4} is the section-3.4 geometry-valid range for the LEARNED
# presider (A_band ~ 0.95; kappa=0.6 consumes the full margin -> excluded).
# Resumable: re-run the same command after an interruption; each trainer picks up.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$warm = "results/mesa/h2-frontier/h2_1_binding/ppo_seed_0/models"
$cells = "nominal,high-target,tight-ruin"
$kappaList = $Kappas.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$required = @(
  $FieldModel,
  "$warm/p_guard.json", "$warm/p_council_arbiter_rl.json", "$warm/m_adapter_rl.json",
  "scripts/mesa-ns2-binding-eval.mjs", "scripts/mesa-ns3-b-aggregate.mjs",
  "scripts/regulator-task.mjs", "training/mesa/regulator_task.py",
  "training/mesa/train_ns1_shutdown.py"
)
foreach ($p in $required) { if (-not (Test-Path $p)) { throw "Missing required NS-3-b input: $p" } }
New-Item -ItemType Directory -Force $OutRoot | Out-Null
Write-Host "[NS-3-b] preflight ok. presider=$FieldModel kappas=$($kappaList -join ',') updates=$Updates resume=$(-not $NoResume)"
if ($PreflightOnly) { exit 0 }

# Regulator: --env regulator + NS3 cells + horizon 9. Cap centers on the LEARNED presider.
# $base = everything EXCEPT --review-band (the scaffold runs the clean task; the real runs add it).
$base = @("--env", "regulator", "--cells", $cells, "--no-exo-shutdown",
  "--train-seeds", "$TrainSeeds", "--train-seed-start", "20000", "--horizon", "9",
  "--updates", "$Updates", "--rollouts-per-update", "$RolloutsPerUpdate",
  "--epochs", "2", "--minibatch-size", "256", "--ppo-seed", "0", "--checkpoint-every", "$CheckpointEvery",
  "--field-cap", "1", "--reward-cap", "0.5", "--guard-cap", "0.7", "--feature-mode", "base",
  "--field-model", $FieldModel)
$fm = @("--field-model", $FieldModel)
$warmInit = @("--init-guard", "$warm/p_guard.json", "--init-arbiter", "$warm/p_council_arbiter_rl.json",
  "--init-monolith-adapter", "$warm/m_adapter_rl.json")

function Train([string]$name, [string[]]$extra, [string[]]$init) {
  $root = Join-Path $OutRoot $name
  New-Item -ItemType Directory -Force $root | Out-Null
  Write-Host "[NS-3-b] training $name -> $root"
  $a = @("-m", "training.mesa.train_ns1_shutdown", "--phase", "ns3_b_$name", "--out", "$root") + $base + $init + $extra
  if ($NoResume) { $a += "--no-resume" }
  & python @a 2>&1 | Tee-Object -FilePath (Join-Path $root "train.log")
  if ($LASTEXITCODE -ne 0) { throw "NS-3-b $name training failed (exit $LASTEXITCODE)" }
}

# Baseline fix (H2.3 cap-as-scaffold): pre-train a COMPETENT base WITH the cap (presider scaffold)
# on the CLEAN task, so the uncapped baseline starts competent instead of floundering from the
# forked warm-start. The competent monolith = scaffold mkappa_adapter; the council = ckappa_arbiter.
Train "scaffold" @("--arb-cap-kappa", "0.4") $warmInit
$scaffoldInit = @("--init-guard", "$OutRoot/scaffold/p_guard.json",
  "--init-arbiter", "$OutRoot/scaffold/ckappa_arbiter_rl.json",
  "--init-monolith-adapter", "$OutRoot/scaffold/mkappa_adapter_rl.json")
if (-not (Test-Path "$OutRoot/scaffold/mkappa_adapter_rl.json")) { throw "scaffold did not produce a competent base" }

# STALE-GUARD: a real-run checkpoint OLDER than the current scaffold predates this scaffold
# (e.g. left over from a prior run with a different env/warm-start). Resuming it would silently
# eval stale models. Clear those so they retrain fresh; a genuinely-interrupted real run (trained
# AFTER the scaffold) is newer and survives, so legitimate resume still works.
$scaffCkpt = "$OutRoot/scaffold/checkpoint.json"
if (Test-Path $scaffCkpt) {
  $scaffTime = (Get-Item $scaffCkpt).LastWriteTimeUtc
  foreach ($name in @("blind", "m0") + ($kappaList | ForEach-Object { "ckappa_$_" })) {
    $rc = "$OutRoot/$name/checkpoint.json"
    if ((Test-Path $rc) -and ((Get-Item $rc).LastWriteTimeUtc -lt $scaffTime)) {
      Write-Host "[NS-3-b] clearing STALE $name (predates current scaffold)"
      Remove-Item -Recurse -Force "$OutRoot/$name"
    }
  }
}

# Real runs: review band ON (crossing-detection env), warm-started from the competent scaffold.
Train "blind" @("--review-band") $scaffoldInit                                   # honest crosser: no override
Train "m0"    @("--review-band", "--apply-override") $scaffoldInit                # M0: override -> learns the temporal sandbag
foreach ($k in $kappaList) {
  Train "ckappa_$k" @("--review-band", "--apply-override", "--arb-cap-kappa", "$k") $scaffoldInit  # cap forces presider-speed
}

Write-Host "[NS-3-b] evals"
& node scripts/mesa-ns2-binding-eval.mjs --env regulator --seeds $EvalSeeds --cells $cells --controller m0 --no-override --label blind --model-root "$OutRoot/blind" @fm --out "$OutRoot/eval_blind.json"
& node scripts/mesa-ns2-binding-eval.mjs --env regulator --seeds $EvalSeeds --cells $cells --controller m0 --label m0 --model-root "$OutRoot/m0" @fm --out "$OutRoot/eval_m0.json"
$ckArgs = @()
foreach ($k in $kappaList) {
  & node scripts/mesa-ns2-binding-eval.mjs --env regulator --seeds $EvalSeeds --cells $cells --controller ckappa --arb-cap-kappa $k --label "ckappa-$k" --model-root "$OutRoot/ckappa_$k" @fm --out "$OutRoot/eval_ckappa_$k.json"
  $ckArgs += @("--ckappa", "$k=$OutRoot/eval_ckappa_$k.json")
}

Write-Host "[NS-3-b] aggregate"
$gc = @("scripts/mesa-ns3-b-aggregate.mjs", "--blind", "$OutRoot/eval_blind.json", "--m0", "$OutRoot/eval_m0.json") + $ckArgs + @("--out", "$ResultsDoc", "--json", "$OutRoot/summary.json")
& node @gc

Write-Host "[NS-3-b] complete. Verdict in $ResultsDoc"
