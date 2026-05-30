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

## Sweeps 2–4 (registered, not yet run)

- **N=32** (grid 64) — a second refinement step for a *trend* (convergence),
  run only if N=24 is invariant. ~1.5 h/run.
- **K-window** — `K∈{2,4,5}` at G=200: maps the regime-2 window `[K_lo,K*)`
  and measures the internal `m_det` upper bracket `K*` (where twin-states
  vanish / injectivity returns).
- **Enstrophy objective** — `Z_low = Σ_low|ω̂|²` trigger vs energy: clause-(ii)
  objective-robustness beyond the energy proxy (needs a `Z_low` observable).
- **Alternate projection** — a different 9-mode low subset: "not a lucky
  basis."

## Cross-references

- [`PDE_C1_PROPOSITION.md`](PDE_C1_PROPOSITION.md) — the `OPEN` tag this closes.
- [`PDE_C1_TWIN_STATE_CERTIFICATE.md`](PDE_C1_TWIN_STATE_CERTIFICATE.md) / [`PDE_C1_PAIRED_FIBER_CONSTANCY.md`](PDE_C1_PAIRED_FIBER_CONSTANCY.md) / [`PDE_C1_MZ_ENERGY_BUDGET.md`](PDE_C1_MZ_ENERGY_BUDGET.md) — the clause adjudicators reused unchanged.
