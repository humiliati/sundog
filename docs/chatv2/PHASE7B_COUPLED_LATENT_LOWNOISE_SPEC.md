# Chat-v2 Phase 7b — Low-Noise Coupled Closure Spec (pre-registration)

> 2026-06-03, DRAFT for sign-off — **not run.** A **single-knob change** off the
> frozen Phase 7 (`PHASE7_COUPLED_LATENT_SPEC.md`): lower the coupling noise
> `p_noise = 0.25 → 0.10`, everything else identical. Phase 7 landed
> `closure_washed_by_readout` (data `func(u)=0.757` → trained-body `~0.63`, below the
> 0.70 bar). Phase 7b asks the next, sharper question: **does the closure survive the
> trained-body readout at low enough coupling noise?** Together with Phase 7 it
> **maps the noise-survival boundary** of the closure bracket. Phase 7 stays frozen at
> its verdict — this is a *new* cell, not a retune. Tighten-not-loosen binds.

Reserved: same harness (`scripts/chatv2_phase0_bodyresist.py --latent coupled
--p-noise 0.10`); probe `scripts/chatv2_phase7_probe.py --dir ...`.
Out: `results/chatv2/phase7b-coupled-lownoise/`.

## 1. Rationale — why p_noise = 0.10 (data-validated 2026-06-03)

Phase 7's readout dropped `func(u)` by ~0.13 (data 0.757 → body 0.63). To clear the
frozen 0.70 bar in the body, the **data-level** `func(u)` needs ≳ 0.83 at a small `k`.
A `p_noise` sweep (clean-`z` data check, `m=3` graph + an `m=2` redundancy variant)
identified `p_noise = 0.10` as the safest survival cell:

```
m3 p=0.10:  func_u: k1 0.629  k2 0.766  k3 0.905  k7 0.911
            state_z: k1 0.541  ...  k5 0.663  k6 0.705  k7 0.827
```

- `func(u) = 0.905 @ k=3` → after a ~0.13 readout wash, ~0.78 — **comfortably over
  0.70** (Phase 7's wash would have to nearly double to fail).
- `state(z)` crosses 0.70 only at `k=6` → a clean **`k_func=3 ≪ k_state=6` bracket**
  (gap 3) with margin to survive.
- Rejected alternatives: `p=0.15` (func 0.849 @ k=3 — margin too thin, ~0.72 after
  wash); `m=2` redundancy (u and states crack at the same small `k`, no usable gap).

## 2. The cell (inherits Phase 7; ONLY p_noise changes)

| parameter | Phase 7 | **Phase 7b** |
| --- | --- | --- |
| `p_noise` | 0.25 | **0.10** |
| `m`, coupling graph `A` | 3, `[001,010,100,011,101,110,111,001]` | **same** |
| `H`, `d_model`, bpc, steps, seeds | 8, 192, 24, 6000, {0,1,2} | **same** |
| encoding / de-confound | parity-channel, input-probe ≤ 0.60 | **same** |
| probe / targets / thresholds | `k_func`(u)≥0.70 sel-corrected; `k_state`(z) label-acc≥0.70 | **same** |
| branches, `u_null` control | Phase 7 §4/§5 | **same** |

The generator already takes `--p-noise`; the probe now takes `--dir`, so **no
further code change** is required for this cell.

## 3. Pre-registered prediction + bracket

Data-level bracket at `p=0.10`: `k_func(≥0.70) = 2`, `k_func(≥0.83 margin) = 3`,
`k_state(≥0.70) = 6`. **Prediction:** the closure **survives** the readout —
trained-body `k_func ∈ {2,3} < k_state ∈ {6,7}` → **`closure_confirmed`**, the lane's
first clean positive, *conditional on* the ~0.13 wash holding. If the wash is larger
than expected, the honest fallback is again `closure_washed_by_readout` — and the
boundary is then below 0.10.

## 4. Headline branches (unchanged from Phase 7 §4)

`closure_confirmed` / `closure_washed_by_readout` / `closure_absent` /
`closure_void`, exact same conditions. The headline question is solely whether the
data-validated bracket clears the 0.70 bar in the trained body at this noise level.

## 5. Negative control

The **same `u_null` control** applies and is **already banked** (Phase 7 §3:
`k_func = none` on the independent target, `control_probe.json`) — it is
noise-independent (it probes Phase-2 uncoupled bodies), so it carries over. A
`closure_confirmed` here is credible against that standing clean control. Re-running
it is optional and not required.

## 6. Boundary-mapping framing

Phase 7 (`p=0.25`, washed) + Phase 7b (`p=0.10`) **bracket the noise-survival
boundary**: if 7b confirms, the closure survives the readout somewhere in
`(0.10, 0.25)`. A finer sweep (e.g. `p ∈ {0.15, 0.20}`) is a **separate future cell**,
not part of this draft — each `p` is a full ~1.4 h train, so the budget is one cell
now. The scientific object is the boundary, not any single `p`.

## 7. Run order (same discipline + timing rule)

1. Sign-off (freeze this spec).
2. Short smoke at `p=0.10` (not verdict-bearing): confirm input-probe ≤ 0.60,
   `u` saves, and the UNLEARNED guard still fires on a 300-step undertrain. This is
   expected to stay under the 10-minute inline bar:

```powershell
$PY   = "C:/Users/hughe/.venvs/sundog-gpu/Scripts/python.exe"
$ROOT = "C:/Users/hughe/Dev/sundog"
$S    = "$ROOT/scripts/chatv2_phase0_bodyresist.py"
$SMOKE = "$ROOT/results/chatv2/phase7b-smoke"
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
  --d-model 192 --bits-per-channel 24 --delta 0.45 --arity 2 --p-noise 0.10 `
  --max-steps 300 --min-steps 0 --patience 99 --seed 0 --out $resolvedSmoke
```

3. Training is the same measured class as Phase 7: about **1.5–3 h for three
   seeds** on the GTX-1080, therefore **do not run inline**. Exact operator /
   long-budget-runner command:

```powershell
$PY   = "C:/Users/hughe/.venvs/sundog-gpu/Scripts/python.exe"
$ROOT = "C:/Users/hughe/Dev/sundog"
$S    = "$ROOT/scripts/chatv2_phase0_bodyresist.py"
$OUTROOT = "$ROOT/results/chatv2/phase7b-coupled-lownoise"
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
    --d-model 192 --bits-per-channel 24 --delta 0.45 --arity 2 --p-noise 0.10 `
    --max-steps 6000 --min-steps 3000 --patience 10 --seed $seed --out $resolvedOut
}
```

Read-back paths:
`results/chatv2/phase7b-coupled-lownoise/seed{0,1,2}/bodies/H8_gen.npz`,
`H8_twin.npz`, `manifest.json`. Each `H8_gen.npz` must include `u`; each manifest
must show learned gen (`eval_loss < 0.673`) before the probe is meaningful.

Resume safety: each seed is independent. For a resume, omit the cleanup block and
rerun only missing/failed seed directories; do not delete completed learned seeds.

4. The probe now accepts `--dir` (default `phase7-coupled-latent`). The headline
   probe is also over the 10-minute inline bar by Phase-7 timing (~16–22 min), so
   stage it for the operator:

```powershell
$PY = "C:/Users/hughe/.venvs/sundog-gpu/Scripts/python.exe"
Set-Location "C:/Users/hughe/Dev/sundog"
& $PY scripts/chatv2_phase7_probe.py --mode coupled --dir phase7b-coupled-lownoise
```

5. Receipt `PHASE7B_COUPLED_LATENT_LOWNOISE_RECEIPT.md` + branch adjudication;
   update the boundary framing with both `p` points:
   - `closure_confirmed`: trained-body `k_func ∈ {2,3}` and `k_func < k_state`
     (or `k_state` none), with the standing `u_null` control still clean.
   - `closure_washed_by_readout`: data bracket holds but trained-body `func(u)`
     never clears the selection-corrected 0.70 read.
   - `closure_absent` / `closure_void`: as Phase 7.

## 8. Honest tier

Even a `closure_confirmed` here is a **toy** closure on a *designed* coupled substrate
at *deliberately low* noise — it demonstrates the instrument **can** measure a
surviving `k_func ≪ k_state` bracket and **locates the noise where it survives**, not
that real models have one. Promote-gate **R1 not met**; no real-LLM or NSE claim. The
value is the mapped boundary (survives at 0.10 vs washes at 0.25), reported honestly
either way.

---

*Sundog Research Lab — chatv2 Phase 7b spec, pre-registration draft. Single-knob
change off the frozen Phase 7. Internal; gated on `SUNDOG_V_ALLELOPATHY.md`. No
verdict run until sign-off.*
