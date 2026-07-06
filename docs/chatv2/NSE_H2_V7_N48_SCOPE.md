# NSE-H2 — v7 N-Refinement Scope (`lock_v7_g{200,300}_n48`)

> 2026-07-06. Scope for H2 of `NSE_POST_AT_HYPOTHESES_SLATE.md` §4 (C1 resolution
> stability). **Scope only — no compute is spent by this document**; the frozen spec
> lifts from here after the one owner decision in §7 (a 3-site additive harness
> edit). Non-promotional. Binding inheritance from H1's close
> (`NSE_H1_FIBER_RECEIPT.md`, `NSE-H1-PROXY-ONLY`): **H2 lifts the v7 portable
> selector** (slate §4 item 4) — there is no H1 selector to inherit, and H2 does not
> touch proxy faithfulness.

## 1. The object being lifted (receipts-true regression targets)

The strongest current package = the two-regime v7 portable witness
(`PDE_C1_REGIME_GENERALITY_v1.md` §12, `PDE-C1-RG-POS`):

| banked run | receipt | key numbers (the comparability targets) |
| --- | --- | --- |
| `lock_v7_g200` kNN-sweep | `c1-rg-v1-g200-knn-sweep/` | portability 0.3003 (calib 0.300); `STRICTNESS_WITNESS_POSITIVE`; a_mm −0.00079; slope 0.736 |
| `lock_v7_g300` kNN-sweep | `c1-rg-v1-g300-knn-sweep/` | portability 0.2688; `STRICTNESS_WITNESS_POSITIVE`; a_mm +0.00058; slope 0.564 |
| `lock_v7_g300` twin-state | `c1-rg-v1-g300-twin-state/` | `TWIN_STATE_CERTIFIED`; 942,834 unique pairs; ε_K 0.0664; δ_H 0.0111 |
| `lock_v5` twin-state (G=200) | `c1-paired-fiber-g200/` | `TWIN_STATE_CERTIFIED`; 693,795 unique pairs; ε_K 0.0606 |

The N-refinement precedent (`PDE_C1_ROBUSTNESS_WAVE.md`): `lock_v5_n48`
(grid 32→48, n_modes 16→24, v5 objective) → `c1-n48-twin/` `TWIN_STATE_CERTIFIED`
with **25,979** unique witness pairs — the ~27× pair-count drop vs grid-32 was
accepted as certified (protocol gates, not pair-count parity, are the criterion).

**What does not exist at grid 48 (the actual H2 gap):** no kNN
control-sufficiency read under *any* objective; nothing at all under the v7
portable objective; nothing at G=300. H2 is precisely those missing halves —
it does **not** rerun `lock_v5_n48`.

## 2. The registered perturbation (exact; nothing else moves)

`grid_size 32 → 48`, `n_modes 16 → 24`. Everything else bit-identical to
`lock_v7_g{200,300}`: k_f=2, K=3 (the same 9 signature modes, d=18), dt=0.01
(precedented stable at grid 48 by `lock_v5_n48`), seed inherited from the lock
family default, burn-in 100k, portable-quantile objective q=0.70, calibration
50k / gap 5k / adjudication 50k at stride 50, τ=500, kNN sweep
k∈{10,15,20,25,30,40,50} with a_mm thresholds (≤0.005 POSITIVE / ≥0.015 NEG-A),
twin k=50 with gates (witness fraction ≥0.01, unique pairs ≥100). **Derived
quantities re-derive by their registered rules, never copied:** ε_K =
0.05·√(2·E_max) with the n48 E_max; δ_H = max(1e-6, 0.05·median‖Q_K‖) with the
n48 high-mode set (dim grows with n_modes 24). Both reported against the grid-32
values. No dt/objective/K/q retune after any read (slate admission gate,
verbatim).

## 3. Program (pinned order; ≤4 owner-run locks; hard stops)

```text
R1  lock_v7_g200_n48 --adjudicator knn-sweep    (the regression cell)
R2  lock_v7_g200_n48 --adjudicator twin-state   (compose at the n48 eps_K)
    [hard stop: R1+R2 must both pass before any G=300 run is interpreted]
R3  lock_v7_g300_n48 --adjudicator knn-sweep
R4  lock_v7_g300_n48 --adjudicator twin-state
```

G=200 first is mandatory (the v1 doc's own logic: re-establish before
generalizing). A failed R1/R2 pauses the program at a typed branch — G=300 money
is never spent reconciling a G=200 surprise.

## 4. Pass criteria (to be frozen in the spec; verdict-label form)

Per cell, "survives refinement" =

1. **Portability gate first** (H4 checklist imported): held-out damp ∈
   [0.20, 0.40], calibration-vs-adjudication reported blockwise.
2. **kNN half:** `STRICTNESS_WITNESS_POSITIVE` under the identical sweep and
   thresholds.
3. **Twin half:** `TWIN_STATE_CERTIFIED` under the identical protocol gates, at
   the same-rule ε_K as the kNN read (composition requirement).

Same verdict labels under identical protocol gates — **not** numeric equality.
The comparability report (reported, never gated): Δa_mm, slope, damp, ε_K, δ_H,
witness-pair counts vs the §1 targets (pair-count drops of the `lock_v5_n48`
magnitude are precedented and acceptable).

## 5. Branch table (slate §4 decision gate, sharpened)

| outcome | branch |
| --- | --- |
| R1+R2 pass, then R3+R4 pass | `NSE-H2-TWO-REGIME-N48-STABLE` |
| R1+R2 pass; G=300 fails with powered objective + passing diagnostics | `NSE-H2-RES-SENSITIVE(g300)` — but the G=200 half still banks as `NSE-H2-V7-N48-STABLE` |
| R1 or R2 fails with powered objective + passing diagnostics | `NSE-H2-RES-SENSITIVE(g200)` — program pauses; critical finding (the witness would be resolution-local) |
| portability fails at n48 (damp outside [0.20,0.40]) with clean diagnostics | `NSE-H2-RES-SENSITIVE(objective)` — the objective itself is resolution-sensitive; informative, no retune |
| solver diagnostics or runtime prevent a fair read (G=300 intermittency wall is the known risk) | `NSE-H2-NUMERIC-WALL` — typed, not a rescue prompt |

## 6. Solver diagnostics gate (pre-read, per run)

Dealias cutoff 16 at grid 48 (mask ≤ m/3); no NaN/Inf in the energy series;
burn-in envelope bounded; step rate within ~2× of the projected cost (a gross
slowdown flags memory pressure → NUMERIC-WALL territory). Diagnostics must pass
before any verdict field is read.

## 7. The one owner decision: harness sign-off (3-site additive edit)

Two new **verdict-bearing** presets. Exact sites in
`scripts/pde_c1_kolmogorov_cell.py` (all additive; no existing preset touched):

1. `VERDICT_BEARING_PRESETS` set (~line 57): add `"lock_v7_g200_n48"`,
   `"lock_v7_g300_n48"`.
2. `parse_args` `--preset` choices (~line 130): add both names (the AT-2 gotcha:
   this list is separate from the set and from build_config).
3. `build_config` (~line 389, after the `lock_v7_*` block):

```python
    elif args.preset in ("lock_v7_g200_n48", "lock_v7_g300_n48"):
        # H2 N-refinement (NSE_H2_V7_N48_SCOPE.md): lock_v7 portable-quantile
        # cell with the lock_v5_n48 resolution lift (grid 32 -> 48, n_modes
        # 16 -> 24; dealias cutoff ~16). Same K=3 signature, dt, seed, split.
        burnin_steps = 100_000
        sample_count = 50_000
        kf = 2
        grashof = 200.0 if args.preset == "lock_v7_g200_n48" else 300.0
        k_signature = 3
        objective = "portable-quantile"
        objective_quantile = 0.70
        calibration_sample_count = 50_000
        calibration_gap_steps = 5_000
        grid_size = 48
        n_modes = 24
```

On sign-off, the agent applies the edit, runs the harness self-test plus one
non-verdict smoke, and confirms an existing-preset regression check (config
echo of `lock_v7_g300` unchanged) before the spec freezes. The lock runs stay
owner-run.

## 8. Cost (measured-basis estimate)

Grid 48 ≈ 2.3–2.6× the per-step FFT cost of grid 32. Against the v1 doc's
measured ~40 min/kNN-sweep and ~35 min/twin at grid 32: expect **~1.5–2 h per
kNN-sweep and ~1.5 h per twin at n48**; full program ceiling ≈ 6–7 h across ≤4
runs (≈3.5 h if the program stops at G=200). Owner-run per slate rule 6 (agent
background jobs die at teardown on multi-hour runs). The R1 receipt's measured
wall-clock recalibrates the R3/R4 estimates before they fire.

## 9. Does not claim

Still finite-Galerkin, sampled-support, numerical, proxy-relative (H1's
`NSE-H1-PROXY-ONLY` typing stands and is untouched by H2). One refinement rung
(32→48) on one axis — not grid convergence, not continuum limit, not promotion.
A full pass upgrades the claim exactly as the slate words it: "two-regime,
current-selector, N-refinement-stable witness." `docs/chatv2/` stays no-publish.

Cross-refs: `NSE_POST_AT_HYPOTHESES_SLATE.md` §4,
`PDE_C1_REGIME_GENERALITY_v1.md` (§3–§6 inherited verbatim; §12 targets),
`PDE_C1_ROBUSTNESS_WAVE.md` + `results/proof/c1-n48-twin/` (the N-refinement
precedent), `NSE_H1_FIBER_RECEIPT.md` (selector inheritance),
`NSE_STATIONARITY_GATE_CHECKLIST.md` (H4, imported at the portability gate),
`results/proof/c1-rg-v1-*/` (the banked targets).
