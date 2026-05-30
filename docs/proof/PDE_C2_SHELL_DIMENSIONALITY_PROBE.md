# PDE C2 — Sabra Shell-State Dimensionality Probe (pre-registration)

> 2026-05-29. A **cheap de-risking probe** before committing to the adaptive
> integrator (Path A) vs a high-dim RL/LLM substrate (Path B) for a sharp
> *body-resistance* control-regime-2 (per `threebody/CROSS_SUBSTRATE_NOTES.md`
> body-resistance note + the Mesa FVE falsification). Question: **does the
> Sabra shell model have a genuinely high-dimensional body that a low-shell
> shadow fails to reconstruct?** No integrator needed — runs inside the
> fixed-dt stable window.

## Why the shell model is the right test

The shell-model control objective is **burst detection — dissipation-range
events in the *high* shells (small scales)**. That is the *opposite* of 2D
NSE, whose objective (low-band energy) lived in the *determined large scales*
and forced marginality. So the sharp question: **are the high
(burst-relevant) shells under-determined by the low shells?** If yes, the
substrate resists in exactly the sector its objective reads.

## Cell + sampling

C2 v1 stable regime (Sabra N=22 → 44 real DOF; fixed-amplitude `|u_1|`
forcing; dt=1e-4; ν=1e-7). Burn-in ~200k, then sample the **full shell state
`u`** at stride 50 for ~20k samples (~1.2M steps — well inside the ~3.5M
stable window), with a **finiteness guard** (abort + report if any
`|u_n|→nan` or `>1e6`).

## Measures (the C1 norm-lessons baked in)

1. **Effective rank** (participation ratio) of the 44-DOF state, **energy-
   weighted** (raw — expect low, dominated by low shells; the C1 trap, for
   contrast) and **per-shell-normalized** (z-score each component → the
   *intrinsic* dimensionality).
2. **Per-shell determination `R²(shell_n | low-K shells)`** across n (the
   decisive plot), via the ported held-out HGB R² estimator, for K∈{2,3,4,6}.
3. **Per-shell spectrum** `<|u_n|²>` and ε-contribution `ν k_n²|u_n|²` — to
   identify the burst-relevant (dissipation-carrying) shells and to **screen
   out numerical noise** (the C1 dissipation-range lesson: an under-determined
   shell only counts if it carries real ε, not numerical noise).

## Pre-registered pass/fail

The shell model is a **body-resistance candidate** (→ Path A justified) iff:
- normalized effective rank is high (≫ shadow dim; say `> ~8` of 44), **AND**
- the **ε-carrying (burst-relevant) shells are genuinely under-determined**
  by the low shells (`R²(shell_n | low-K)` low for those n) — *and those
  shells carry non-negligible ε (not numerical noise).*

If the normalized state is low-dim, **or** the under-determination is only in
ε-negligible (noise) shells while the ε-carrying shells are well-determined →
the shell model is **also marginal** → skip Path A, go Path B (high-dim
RL/LLM).

## Cost / build

~8-10 min run + the HGB regressions. Self-contained probe script
(`scripts/pde_c2_shell_dimensionality.py`) imports `ShellModel` from the C2
harness; no change to the verdict-bearing C2 cell.

## Result (2026-05-29) — directionally MARGINAL (third instance); favor Path B

**v1 inconclusive (artifacts), v2 fixed + ran.** v1's metrics were
contaminated (effective rank dominated by the pinned forcing shell + the
numerical-underflow tail; per-shell R² wildly negative from a block-split
artifact, with no perm control). v2 restricted to the dynamically-real
inertial shells (1–15, `|u|²>1e-11`), z-scored only those, and added the
perm-control arbiter.

**v2 numbers** (`results/proof/c2-shell-dim-v2/`, window ~200 time units ≈ 0.7
burst recurrence):
- **effective rank of the real-shell body = 1.7 of 30** — the inertial cascade
  is effectively ~2-dimensional (smooth, slaved power-law; no high-dim body).
- burst-relevant shells: **`R²_real = −2.5` ≫ `R²_perm = −197`** — the
  perm-arbiter shows the real low-shell shadow genuinely carries predictive
  info about the shells (they are **slaved**, not independent). Both negative
  confirms the block-split non-stationarity artifact, so no clean fraction is
  quoted, but the *gap* is the signal.

**Disposition: directionally MARGINAL.** Low-rank cascade + shells slaved to
the low shadow = marginal, the C1/Mesa pattern a **third time**. **Honest
limit:** the window is ~0.7 burst times (the numerical wall caps the stable
window at ~1 burst), so the slow/intermittent modes are under-sampled — a
*directional* read, not definitive; a definitive measure would need the
adaptive integrator. But eff-rank 1.7 is too low to plausibly flip.

**Consequence for the Path A vs B fork.** Every measurable dynamical-system /
control substrate — NSE (C1), Mesa, Sabra shell — is marginal on
body-resistance. So **Path A (integrator → turbulence) is not justified by the
sharp-regime-2 goal** (the best PDE candidate is directionally marginal); it
retains independent value only as C2's control experiment. **Path B (high-dim
RL/LLM, body-resisting by construction) is the route** to a sharp control
regime-2. A second-order finding also lands: the cheap probe is itself
**window-limited by the same numerical wall** — you cannot fully assess this
substrate's dimensionality without the integrator.

## Cross-references

- [`../threebody/CROSS_SUBSTRATE_NOTES.md`](../threebody/CROSS_SUBSTRATE_NOTES.md) — the body-resistance axis + the Mesa falsification this de-risks.
- [`PDE_C2_CELLSET_SABRA_v1.md`](PDE_C2_CELLSET_SABRA_v1.md) — the C2 numerical wall; the adaptive integrator is gated on this probe.
