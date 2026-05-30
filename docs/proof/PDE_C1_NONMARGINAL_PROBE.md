# PDE C1 — Non-Marginal Regime Probe

> 2026-05-29. The `m_det` probe (robustness wave §4) found the G=200/K=3
> separation **energy-marginal** (the attractor is an approximate inertial
> manifold). This note pursues a *non-marginal* regime-2 separation. Two
> paths were considered; the high-G path hit a numerical wall, so the chosen
> path is a **norm reframing** at G=200.

## 1. High-G path — attempted, hit the C2 numerical wall

Rationale: a higher Grashof gives a higher-dimensional attractor, so a fixed
K=3 signature sits far below the determining count → genuine (all-norm)
state-insufficiency, if control still holds.

**Result: numerical wall (the same one C2 hit).** A `lock_hidim_g1000`
probe (G=1000, grid 64, K=3, `ν=0.0316`) was stability-checked first:

| dt | 40k-step transient |
| --- | --- |
| 0.010 | **blow-up** (overflow in advection within a few thousand steps) |
| 0.004 | blow-up |
| 0.002 | blow-up |
| 0.001 | stable (`E_low → 2.2`) |

So fixed-dt high-G is feasible only at `dt = 0.001` — **10× the steps,
~7 h/run** at grid 64 (before a grid-96 resolution check) — **and fragile**:
dt=0.001 survived a short transient, but C2's blow-up occurred at ~3.5M steps
on an *intermittent* event, so a full ~27M-step high-G run has a real chance
of dying partway. **Decision: not pursued by brute force** (it repeats the
C2 fixed-dt trap). The robust high-G path is the **adaptive/stiff integrator
on a uniform-time grid** — the C2 resume design — which would unblock *both*
this lane and C2; deferred to that build.

## 2. Chosen path — norm reframing at G=200 (no numerical wall)

The marginality is **norm-dependent**. 2D NSE concentrates *energy* at large
scales (always near-determined), so the **energy norm** makes the separation
look marginal — but the small-scale degrees of freedom are genuinely
under-determined, and they carry the *enstrophy*. Measure state-determination
`FVE(Q_K|Φ_K)` in three norms on a uniform sample of high-mode components
(`state-recon`, K=3, G=200):

- **energy-weighted** (component variance) — large scales;
- **enstrophy-weighted** (`|k|²`) — emphasizes small scales;
- **equal-weight** (median per-component R²) — every DOF counted equally.

Already in hand: energy-weighted `FVE = 0.9994` (marginal); equal-weight
median `R² = 0.73` (non-marginal). The **enstrophy-weighted FVE** is the new
crisp number.

**Pre-registered reading.** The separation is **non-marginal in the
enstrophy / per-DOF norm** iff the enstrophy-weighted state residual
`1 − FVE_enstrophy` is materially larger than the energy residual (0.06%) —
i.e. the small scales are genuinely under-determined. Combined with the
**already-established enstrophy control-sufficiency** (paired
`PAIRED_FIBER_CONSTANCY_POSITIVE` under the `Z_low` trigger, robustness wave
§"sweep 3"), that is a legitimate non-marginal regime-2 separation in a
matched (enstrophy) norm. **Honest caveat:** it is norm-dependent — marginal
in energy, non-marginal in enstrophy/per-DOF; a skeptic can note the norm
choice, so both must be reported side by side.

## 3. Result (2026-05-29)

*(to be filled when `c1-recon-k3-norms` lands)* — the three-norm FVE at K=3,
and the reframe of proposition clause (i).

## 4. Cross-references

- [`PDE_C1_ROBUSTNESS_WAVE.md`](PDE_C1_ROBUSTNESS_WAVE.md) §4 — the m_det probe that found the energy-marginality.
- [`PDE_C1_PROPOSITION.md`](PDE_C1_PROPOSITION.md) — clause (i) to be reframed.
- [`PDE_C2_CELLSET_SABRA_v1.md`](PDE_C2_CELLSET_SABRA_v1.md) — the same numerical wall; the adaptive integrator is the shared resume path.
