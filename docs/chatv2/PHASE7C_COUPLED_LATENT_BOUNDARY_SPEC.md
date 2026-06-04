# Chat-v2 Phase 7c — Coupled Closure Boundary-Locate Cell (pre-registration)

> 2026-06-04, DRAFT for sign-off — **not run.** A **single-knob change** off the
> frozen Phase 7 / 7b cells: set `p_noise = 0.20`, everything else identical. Phase 7
> (`p=0.25`) **washed**; Phase 7b (`p=0.10`) **confirmed**. Phase 7c is the
> **maximal-information bisection** that locates the noise-survival boundary between
> them. Phase 7 and 7b stay frozen at their verdicts — this is a *new* cell, not a
> retune. Tighten-not-loosen binds. **No verdict run until sign-off.**

Reserved: same harness (`chatv2_phase0_bodyresist.py --latent coupled --p-noise 0.20`);
probe `chatv2_phase7_probe.py --mode coupled --dir phase7c-coupled-p20`.
Out: `results/chatv2/phase7c-coupled-p20/`.

## 1. Why p_noise = 0.20 (the bisection point)

Interpolating the measured **trained-body** `func(u) @ k=3` from the two anchors:

| p_noise | data `func(u)@k3` | trained-body `func(u)@k3` | verdict |
| --- | --- | --- | --- |
| 0.10 (7b) | 0.905 | ~0.82 | `closure_confirmed` |
| 0.25 (7)  | 0.757 | ~0.64 | `closure_washed_by_readout` |

The body line crosses the 0.70 bar at **`p ≈ 0.20`** (slope ≈ −1.2/unit-p; `0.82 −
1.2·(0.20−0.10) = 0.70`). So `p=0.20` is the point of **maximal outcome uncertainty** —
its verdict pins the boundary either way. Data-level bracket at `p=0.20` (from the
2026-06-03 sweep) is still clean: `func(u)` 0.602/0.706/0.800 (k=1/2/3), `k_func(data)=2`,
states cap 0.689 → `k_state(data)=none`. A valid cell sitting on the predicted edge.

## 2. The cell (inherits 7b/7; ONLY p_noise changes)

| parameter | Phase 7b | **Phase 7c** |
| --- | --- | --- |
| `p_noise` | 0.10 | **0.20** |
| `m`, graph `A`, `H`, `d_model`, bpc, steps, seeds | — | **all same** |
| encoding / de-confound / probe / targets / thresholds / branches / `u_null` | — | **all same** |

No code change (generator takes `--p-noise`, probe takes `--dir`).

## 3. Pre-registered prediction — uncertain *by design*

Body `func(u)@k3` predicted ≈ 0.70 — right at the bar. **All three outcomes are
informative**, and each updates the boundary:

| outcome | reading | boundary update |
| --- | --- | --- |
| all seeds `closure_confirmed` | survives at 0.20 | sampled-cell boundary interval **[0.20, 0.25)** |
| all seeds `closure_washed_by_readout` | washes at 0.20 | sampled-cell boundary interval **(0.10, 0.20]** |
| **mixed across seeds** (some confirm, some wash) | 0.20 *is* the transition zone | boundary **≈ 0.20** |

This is a bisection, not a bet on a sign — the value is wherever it lands.

## 4. Headline branches + controls

Per-seed branches are identical to Phase 7 §4 (`closure_confirmed` /
`closure_washed_by_readout` / `closure_absent` / `closure_void`). The receipt also
reports an **aggregate boundary classification**:

- `boundary_high`: all learned seeds confirm at 0.20;
- `boundary_low`: all learned seeds wash at 0.20;
- `boundary_transition_zone`: learned seeds split confirm/wash at 0.20;
- `boundary_void`: any seed is `closure_void` or the control contract fails.

The `u_null` control is **already banked** (noise-independent, `k_func=none`); it
carries over, no re-run required.

## 5. Run order (same discipline + timing rule)

1. Sign-off (freeze).
2. Short smoke at `p=0.20` (not verdict-bearing): confirm input-probe ≤ 0.60,
   `u` saves, and the UNLEARNED guard still fires on a 300-step undertrain. This is
   expected to stay under the 10-minute inline bar:

```powershell
$PY   = "C:/Users/hughe/.venvs/sundog-gpu/Scripts/python.exe"
$ROOT = "C:/Users/hughe/Dev/sundog"
$S    = "$ROOT/scripts/chatv2_phase0_bodyresist.py"
$SMOKE = "$ROOT/results/chatv2/phase7c-smoke"
$RESULTROOT = "$ROOT/results/chatv2"
$resolvedResultRoot = [System.IO.Path]::GetFullPath($RESULTROOT)
$resolvedSmoke = [System.IO.Path]::GetFullPath($SMOKE)
if (-not $resolvedSmoke.StartsWith($resolvedResultRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
  throw "Refusing to remove unexpected path: $resolvedSmoke"
}
if (Test-Path -LiteralPath $resolvedSmoke) {
  Remove-Item -LiteralPath $resolvedSmoke -Recurse -Force
}
& $PY $S --mode full --stage all --latent coupled --fair-readout --h-sweep 8 `
  --d-model 192 --bits-per-channel 24 --delta 0.45 --arity 2 --p-noise 0.20 `
  --max-steps 300 --min-steps 0 --patience 99 --seed 0 --out $resolvedSmoke
```

3. Training is the same measured class as Phase 7/7b: about **1.5–3 h for three
   seeds** on the GTX-1080, therefore **do not run inline**. Exact operator /
   long-budget-runner command:

```powershell
$PY   = "C:/Users/hughe/.venvs/sundog-gpu/Scripts/python.exe"
$ROOT = "C:/Users/hughe/Dev/sundog"
$S    = "$ROOT/scripts/chatv2_phase0_bodyresist.py"
$OUTROOT = "$ROOT/results/chatv2/phase7c-coupled-p20"
$resolvedRoot = [System.IO.Path]::GetFullPath($OUTROOT)

foreach ($seed in 0,1,2) {
  $out = Join-Path $OUTROOT "seed$seed"
  $resolvedOut = [System.IO.Path]::GetFullPath($out)
  if (-not $resolvedOut.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to remove unexpected path: $resolvedOut"
  }
  if (Test-Path -LiteralPath $resolvedOut) {
    Remove-Item -LiteralPath $resolvedOut -Recurse -Force
  }
  & $PY $S --mode full --stage all --latent coupled --fair-readout --h-sweep 8 `
    --d-model 192 --bits-per-channel 24 --delta 0.45 --arity 2 --p-noise 0.20 `
    --max-steps 6000 --min-steps 3000 --patience 10 --seed $seed --out $resolvedOut
}
```

Read-back paths:
`results/chatv2/phase7c-coupled-p20/seed{0,1,2}/bodies/H8_gen.npz`,
`H8_twin.npz`, `manifest.json`. Each `H8_gen.npz` must include `u`; each manifest
must show learned gen (`eval_loss < 0.673`) before the probe is meaningful.

Resume safety: each seed is independent. For a resume, omit the cleanup block and
rerun only missing/failed seed directories; do not delete completed learned seeds.

4. Headline probe, staged for the operator because Phase-7 timing is ~16–22 min:

```powershell
$PY = "C:/Users/hughe/.venvs/sundog-gpu/Scripts/python.exe"
Set-Location "C:/Users/hughe/Dev/sundog"
& $PY scripts/chatv2_phase7_probe.py --mode coupled --dir phase7c-coupled-p20
```

5. Receipt `PHASE7C_COUPLED_LATENT_BOUNDARY_RECEIPT.md` + adjudication; **update
   the noise-survival boundary table** with all three p-points (0.10, 0.20, 0.25)
   and the aggregate boundary classification above.

## 6. Tier

Pure boundary-location — no new mechanism. Maps the substrate-noise threshold at which
the closure crosses from washed to recoverable through a trained-body readout. Still a
**toy** (designed coupling, `H=8`, `d_dec≈7.6`); **no real-LLM / NSE / scaling claim;
R1 not met; unpromoted.** The deliverable is the located boundary.

---

*Sundog Research Lab — chatv2 Phase 7c spec, pre-registration draft. Single-knob
bisection off the frozen 7 / 7b cells. Internal; gated on `SUNDOG_V_ALLELOPATHY.md`.
No verdict run until sign-off.*
