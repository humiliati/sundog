# PDE C1 — Robustness Wave (pre-registration)

> The `OPEN` tag of the C1 proposition
> ([`PDE_C1_PROPOSITION.md`](PDE_C1_PROPOSITION.md) §6): refinement-invariance.
> This wave tests whether the regime-2 separation survives modest changes in
> discretization and proxy construction — the difference between "survives
> perturbation" and "structural." Sweeps, in priority order: **N-refinement
> (first)**, then K-window, enstrophy objective, alternate projection. Each is
> verdict-bearing and pre-registered before it runs. Finite-Galerkin
> throughout — refinement *approaches* but never reaches the PDE.

## Sweep 1 — N-refinement (running)

**Cell `lock_v5_n48`.** Identical to the established G=200 witness cell
`lock_v5` (`k_f=2`, `G=200`, `K=3`, overshoot-burnin objective + last-25%
`E_max`, same `dt=0.01`, same sampling, same seed) **except** the Galerkin
resolution is refined:

| | `lock_v5` (baseline) | `lock_v5_n48` (refined) |
| --- | --- | --- |
| grid | 32 | 48 |
| dealias cutoff `|k|` | ~10 | ~16 |
| `n_modes` | 16 | 24 |
| high-mode DOF (`Q_K`) | 422 | **1070** |
| `ν` (so `G`) | 0.07071 | 0.07071 (unchanged) |
| `K`=3 signature (`Φ_K`) | 9 modes / d=18 | same 9 modes / d=18 |

**Only the resolved-scale count changes.** `ν`, `G`, `k_f`, the objective,
`dt`, the sampling schedule, and the K=3 observation map are all held fixed.
**Stability confirmed** before launch: grid 48 at `dt=0.01` is stable over
30k steps, `E_low` plateaus to ~0.73 (matching grid 32 — large scales
converged, refinement adds small-scale content), no blow-up (the C2-style
CFL check).

**Pre-registered pass/fail.** The separation is **refinement-invariant at
this step** iff all three clauses persist at N=24:

- **(i)** `twin-state` → `TWIN_STATE_CERTIFIED` (non-injectivity persists —
  expected *more* easily, the `Q_K` null space grew 2.5×).
- **(ii)** paired fiber-constancy → `PAIRED_FIBER_CONSTANCY_POSITIVE`
  (`D_witness ≤ delta_action = 0.10`).
- **(iii)** `mz-budget` → `COUPLING_SIGNATURE_SLAVED` (`R²(R|Φ_K) ≥ 0.70`,
  controls `R²(g) > 0.90`, `R²(perm) < 0.10`).

All three → **`REFINEMENT_INVARIANT`** (flips the proposition's `OPEN`
refinement tag to `DEMONSTRATED` for the N axis). Any clause failing → the
result is resolution-dependent at this step; **file honestly, do not rescue**
(no post-hoc dt / objective / K retune — that would be the C1-NEG-B
boundary).

**Runs.** `twin-state` (clauses i + paired-ii) and `mz-budget` (clause iii)
at `lock_v5_n48`, `results/proof/c1-n48-twin/` and `…/c1-n48-mz/`. ~55 min
each (grid 48 ≈ 2.5× the per-step cost of grid 32).

### Sweep 1 result (2026-05-29) — REFINEMENT_INVARIANT at N=24

Both clause-runs returned the regime-2 verdicts, with the mz-budget
calibration controls clean at the finer resolution. Despite `Q_K` growing
**422 → 1070 DOF** (2.5×), the numbers are **nearly identical** to N=16:

| clause | N=16 (`lock_v5`) | N=24 (`lock_v5_n48`) |
| --- | --- | --- |
| (i) twin-state | `CERTIFIED`, 693,795 witness pairs | `CERTIFIED`, **689,263** |
| (ii) paired `D_witness` / `D_candidate` | 0.0367 / 0.0319 | **0.0377 / 0.0319** |
| (iii) `R²(R\|Φ_K)` | 0.998 | **0.998** |
| (iii) controls `R²(g)` / `R²(perm)` | 0.999 / −0.001 | **0.9993 / −0.0012** |
| `T_LLL` check `rms(R)/rms(T_low)`; `corr(g,R)` | 1.0; −0.85 | 1.0; −0.85 |
| `damp_fraction` | 0.298 | 0.304 |

Results: `results/proof/c1-n48-twin/`, `…/c1-n48-mz/`.

**Verdict: `REFINEMENT_INVARIANT`** — all three clauses persist, and the
quantities barely move (`D_candidate` unchanged at 0.0319; slaving 0.998 →
0.998) even though the unresolved state space grew 2.5×. The separation does
**not** degrade with resolution: strong evidence it is *structural, not a
truncation artifact*. This flips the proposition's refinement tag from `OPEN`
to `DEMONSTRATED` on the **N axis** (one step; an N=32 trend is the registered
next increment). Sweeps over K, objective, and projection remain open.

## Sweep 2 — K-window (new axis: observation choice) (running)

**Cells `lock_v5_k2`, `lock_v5_k4`** = `lock_v5` with only the signature band
`K` changed (`K=2`, d=8; `K=4`, d=32). Tests whether regime-2 is robust to
the observation choice, and brackets it.

- **`K=2` (lower / coarser):** clauses (i) twin-state + paired-(ii) +
  (iii) mz-budget. *Pass* = all three persist at a coarser observation
  (window includes K=2). A clause failing here = the lower bracket `K_lo`
  (control needs more than 4 modes) — filed honestly.
- **`K=4` (upper probe):** twin-state. **Expected coverage-limited**: at
  K=4 (d=32) signature-near pairs become sparse (the curse-of-dimensionality
  finding from v2/v4 that motivated K=3), so twin-state may return
  `TWIN_STATE_DEFERRED_COVERAGE` — a *method boundary*, **not** a regime-2
  failure or an injectivity-return. The clean `m_det` upper bracket needs a
  coverage-free state-reconstruction measure (`R²(Q_K|Φ_K)` across K) — a
  registered follow-on, not this sweep.

So the K-window demonstrates regime-2 across `K∈{2,3}` (both clean); the
upper end is honestly flagged method-limited.

## Sweep 3 — Enstrophy objective (new axis: objective construction) (running)

**Cell `lock_v5_enstrophy`** = `lock_v5` except the safety trigger watches
**low-band enstrophy** `Z_low = Σ_low|ω̂/scale|²` (energy without the
`1/|k|²` weighting) instead of energy; same quantile/`E_max` construction,
**same signature `Φ_K`**. Run twin-state (the non-injectivity clause (i) is
objective-free and identical; the new content is the **paired fiber-constancy
(ii)** under the enstrophy-triggered action). *Pass* =
`PAIRED_FIBER_CONSTANCY_POSITIVE` under enstrophy → clause (ii) is robust to
the physical observable, not an artifact of the energy proxy.

## Sweeps 4–5 (registered, not yet run)

- **N=32** (grid 64) — a second N point for a convergence *trend*. ~1.5 h.
- **Alternate projection** — a different 9-mode low subset: "not a lucky basis."
- **`m_det` upper bracket** — `R²(Q_K|Φ_K)` across K (coverage-free), to
  measure where the high modes become signature-slaved (state-determination).

## Cross-references

- [`PDE_C1_PROPOSITION.md`](PDE_C1_PROPOSITION.md) — the `OPEN` tag this closes.
- [`PDE_C1_TWIN_STATE_CERTIFICATE.md`](PDE_C1_TWIN_STATE_CERTIFICATE.md) / [`PDE_C1_PAIRED_FIBER_CONSTANCY.md`](PDE_C1_PAIRED_FIBER_CONSTANCY.md) / [`PDE_C1_MZ_ENERGY_BUDGET.md`](PDE_C1_MZ_ENERGY_BUDGET.md) — the clause adjudicators reused unchanged.
