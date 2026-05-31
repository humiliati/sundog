# Phase 2 v6 Follow-Up Amendment - Finite-T Polyakov Pilot Metric

Filed: **2026-05-31 (PT)**
Amendment id: `YM_P2_V6A_PILOT_METRIC_2026-05-31`
Parent spec: [`PHASE2_SU2_3D_finite_t_polyakov_v6.md`](PHASE2_SU2_3D_finite_t_polyakov_v6.md)
Triggering receipt:
[`../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v6_pilot_unbracketed.md`](../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v6_pilot_unbracketed.md)

Status: **binding v6 follow-up amendment**. It admits one amended pilot before
finite-temperature ensemble generation. It does **not** change the finite-T
geometry, target vocabulary, signature vocabulary, Stage-1 gates, Stage-2
controls, or public claim boundary.

## 1. Why This Amendment Exists

The first v6 pilot stopped correctly at `Z beta_peak_unbracketed`: the locked
grid `{6.0, 6.3, 6.55, 6.8, 7.1}` did not bracket the selected pilot peak before
generation.

The receipt exposed a metric ambiguity. The selected metric, `mean_chi_P`, was
the **mean per-config transverse-site variance** of the Polyakov loop. That is a
valid held-out target summary from amendment 2, but it is not the clean
finite-volume crossover selector. The finite-T pilot is supposed to bracket the
order-parameter fluctuation, which is an **ensemble-level** susceptibility of the
configuration order parameter.

The same pilot receipt already recorded the needed diagnostic without using the
signature or Stage-2 score: the across-config variance of `abs_mean_P` rose to
the high-beta boundary. Therefore the follow-up is constrained to:

1. clarify the pilot selector as the ensemble-level order-parameter
   susceptibility; and
2. extend the beta grid upward from the previous high boundary.

No old or new pilot row may be used to hand-pick a beta slate directly.

## 2. Amended Pilot Selection Metric

For each pilot measurement config `c`, compute the temporal Polyakov summary
already admitted by amendment 2:

```text
Pbar_c = mean over transverse sites of (1/2) Tr P_c(x_perp)
m_c    = abs(Pbar_c) = abs_mean_P
```

The **primary pilot selection metric** is:

```text
order_suscept_abs_mean_P
  = V_perp * Var_c(m_c)
  = (12 * 12) * ( mean_c(m_c^2) - mean_c(m_c)^2 )
```

The `V_perp` multiplier does not affect peak location, but records the usual
finite-volume order-parameter susceptibility scale.

Record-only pilot metrics:

- `order_suscept_mean_abs_P = V_perp * Var_c(mean_abs_P)`;
- `mean_chi_P` (the prior per-config spatial-variance metric);
- `mean_abs_mean_P`, `mean_mean_abs_P`, mean plaquette, and heatbath health.

The held-out vocab v4 candidate pool remains exactly
`{abs_mean_P, mean_abs_P, chi_P}`. The `chi_P` candidate remains the per-config
spatial variance defined in amendment 2; this amendment only changes the
pre-generation beta-slate selector.

## 3. Amended Pilot Grid

The amended grid is:

```text
{6.0, 6.3, 6.55, 6.8, 7.1, 7.4, 7.7, 8.0}
```

Rationale: keep the original literature-anchored grid and append three
high-beta points because the corrected order-parameter diagnostic rose to the
old upper boundary. No lower extension is admitted in this amendment because
the corrected selector did not fail at the low boundary.

Pilot scan settings:

```text
burn-in:      800 combined sweeps
measurements: 96
thinning:     4 combined sweeps
seed:         202605310600
```

The added pilot cost remains inside the repository's 10-minute rule. If the
amended pilot brackets a peak, the existing v6 runner proceeds to freeze the
neighboring three-beta slate and then performs the v6 Stage-1/Stage-2 path. If
the peak again lands on either boundary, the runner stops before generation as
`Z beta_peak_unbracketed`.

## 4. Exact Invocation

```powershell
npm run yang-mills:phase2:v6:finite-t:polyakov
```

The npm script must expand to:

```text
node scripts/yang-mills-phase2-v6-finite-t-polyakov.mjs --cell SU2_3D --lattice-size 12x12x4 --boundary periodic --action Wilson --generator su2_heatbath_overrelax_v1 --pilot-beta-grid 6.0,6.3,6.55,6.8,7.1,7.4,7.7,8.0 --out results/yang-mills/phase2/SU2_3D/2026-05-31_su2_3d_finite_t_polyakov_v6a --power-icc-gate 0.50 --power-agreement-gate 0.50 --leakage-cvr2-gate 0.25 --k-primary 5 --bootstrap 1000 --seed 202605310600
```

## 5. Branch Discipline

- `P2-A`, `YM-P2-NEG-A`, `YM-P2-UNDERPOWERED`, and the existing B/G/D/Z branches
  are inherited unchanged from the parent v6 spec after a beta slate is frozen.
- `Z beta_peak_unbracketed` remains a **pilot void**, not a negative and not an
  underpowered target result.
- A second `Z beta_peak_unbracketed` is not an automatic license for another
  grid retry. Continuing after v6a requires external scientific motivation or
  reviewer feedback; otherwise the honest disposition is PAUSE with v4/v5
  underpowered receipts plus the v6/v6a pilot-void receipts.

## 6. Anti-Scope-Creep

This amendment does not admit:

- changing `N_t = 4`, volume, target summaries, signature vocab, power gates, or
  controls;
- using the pilot to choose among target candidates;
- adding topological charge, 4D, smearing, or blocking;
- reinterpreting any symmetric-cell null or underpowered result.
