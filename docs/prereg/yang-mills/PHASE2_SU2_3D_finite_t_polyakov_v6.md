# Phase 2 v6 — SU(2) 2+1D Finite-Temperature Polyakov Target (binding, pre-stated)

Status: **binding pre-registration**, filed 2026-05-31. **Gated**: invoked only
if v5 ([`PHASE2_SU2_3D_relative_locality_v5.md`](PHASE2_SU2_3D_relative_locality_v5.md))
returns `YM-P2-UNDERPOWERED` (symmetric-cell Polyakov too weak in the confined
`⟨P⟩≈0` regime). Filed in parallel with v5 so an underpowered symmetric result
flows straight into the finite-T run with no second sign-off.

Follow-up amendment after the first pilot void:
[`PHASE2_SU2_3D_finite_t_polyakov_v6_AMENDMENT_2026-05-31_pilot_metric.md`](PHASE2_SU2_3D_finite_t_polyakov_v6_AMENDMENT_2026-05-31_pilot_metric.md).
This amendment governs the current v6a invocation: the target/gate/Stage-2
contract is unchanged, but the pre-generation pilot selector is clarified as
ensemble-level order-parameter susceptibility and the pilot grid is extended
upward.

Execution status: **v6a executed 2026-05-31 -> `YM-P2-NEG-A no_rank_local_structure`**.
Receipt:
[`../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v6a_finite_t_polyakov_neg_a.md`](../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v6a_finite_t_polyakov_neg_a.md).

P0 lock: [`P0_DOMAIN_AND_RECEIPT_LOCK.md`](P0_DOMAIN_AND_RECEIPT_LOCK.md) ·
amendment 2 (Polyakov class + finite-T slate):
[`P0_AMENDMENT_2026-05-31_polyakov.md`](P0_AMENDMENT_2026-05-31_polyakov.md).

## 1. Rationale

On the symmetric lattice the Polyakov loop is centre-symmetric (`⟨P⟩≈0`) and may
be underpowered. In a **finite-temperature** `12²×4` geometry straddling the
SU(2) 2+1D deconfinement crossover, `⟨|P|⟩` is a genuine order parameter that
fluctuates strongly config-to-config — the regime where a Polyakov target is
unambiguously **powered**. The Polyakov loop remains disjoint from the small-loop
signature by construction, so this is the cell where a powered **and** disjoint
held-out target is most likely to exist.

## 2. Cell / ensemble (NEW — requires generation)

- Geometry `12 × 12 × N_t`, `N_t = 4`, periodic all directions, Wilson action.
- β slate: 3 points straddling the `N_t=4` deconfinement crossover. **Pre-generation
  pilot (one-time, before frozen generation):** a short Polyakov scan over a coarse
  β grid; the registered slate is the 3 β that bracket the crossover (one below,
  one near, one above the susceptibility peak), recorded in the runner manifest
  and frozen thereafter (mirrors parent P0's "β revisable once before generation").
  - **Literature anchor (seeds the pilot grid):** the SU(2) 2+1D `N_t=4`
     deconfinement critical coupling is `β_c = 6.53661(13)` (Edwards–von Smekal
     2009, [arXiv:0908.4030](https://arxiv.org/abs/0908.4030); β = 4/g²a
     convention, identical to the sundog Wilson action). Center the coarse pilot
     grid there — e.g. `{6.0, 6.3, 6.55, 6.8, 7.1}` — to locate the finite-`12²`
     Polyakov-susceptibility peak (which sits slightly below the infinite-volume
     `β_c`), then freeze the 3-β slate to bracket it.
  - **2026-05-31 v6a amendment:** the first pilot voided because the selected peak
    landed on the grid boundary. The amended pilot uses grid
    `{6.0, 6.3, 6.55, 6.8, 7.1, 7.4, 7.7, 8.0}` and selects on
    `order_suscept_abs_mean_P = (12*12) * Var(abs_mean_P)` across pilot
    measurements; the prior `mean_chi_P` is record-only.
- Generator: the SU(2) 3D core (Creutz + Kennedy-Pendleton + Brown-Woch),
  **generalized to an asymmetric lattice** (see §5). Burn-in ≥ 2000; thinning ≥
  2·τ_int from a pilot; standard unitarity/health gates.
- 32 configs per β (matching the Phase-2 envelope), per-β seeds `202605310600 + β-index`.

## 3. Stage 1 — Polyakov target power audit (same gates as v5)

Candidate pool: the same Polyakov summaries (`abs_mean_P`, `mean_abs_P`, `chi_P`),
here with μ = the **temporal** direction, summarized over the `12²` transverse
sites. Split-half = transverse-site parity. **Frozen gates verbatim:** power
`ICC≥0.50 ∧ agreement≥0.50` (all three β); leakage `CV-R²(target|v1 signature) ≤
0.25`; `CTRL_GAUGE_RAND` invariance ≤ 1e-12. `γ_held` re-audited as the must-fail
self-validation. The v1 signature is computed on the **finite-T configs** (same
`{W11,W12,W13,W22}` small loops; signature vocab unchanged).

## 4. Stage 2 — relative-locality test (admitted primary only)

v0-identical methodology and **verbatim v0 promotion gates** (purity@5 ≥ 0.5;
margin ≥ 0.10 over RAND/META/RAW; across-β ≥ 0.05 over RAND_STRAT; PERM within
0.05 of 1/3; GAUGE_RAND ≤ 1e-12), scored against the admitted Polyakov target's
per-β tertiles. Branches: `P2-A` (the powered, disjoint positive — the lane's
target outcome) / `YM-P2-NEG-A` (informative: signature vacuous on a powered
disjoint target) / `YM-P2-UNDERPOWERED` (even finite-T Polyakov fails → **PAUSE**:
the lane has exhausted the small-loop-signature program across Wilson-loop and
Polyakov target classes; the honest endpoint) / B / G / D / Z per P0.

## 5. Implementation note (its own next step)

Unlike v5, v6 is **not** an aggregation pass. It requires:
1. generalizing `scripts/lib/yang-mills-su2-3d-core.mjs` to **asymmetric lattices**
   — `createSU2Lattice`, `linkBase`, `wrap`, and the sweep/plaquette routines
   currently assume a single cubic `L`; introduce `(Lx, Ly, Lt)`. The existing
   cubic path must stay bit-for-bit unchanged (assert `Lx=Ly=Lt` reproduces v0–v5);
2. a new finite-T runner entry `scripts/yang-mills-phase2-v6-finite-t-polyakov.mjs`
   that performs the pilot, generation, and v6 aggregation in one locked
   invocation;
3. the pilot β scan, then the frozen generation + audit.

Locked invocation:

```powershell
npm run yang-mills:phase2:v6:finite-t:polyakov
```

Pilot guard: if the pilot susceptibility peak lands on either boundary of the
locked pilot grid, the runner stops before ensemble generation as
`Z beta_peak_unbracketed`; no beta slate is frozen and no Stage 1/Stage 2 score
is admitted.

As of the v6a amendment, the locked pilot settings are 800 burn-in sweeps, 96
measurements, thinning 4, grid `{6.0, 6.3, 6.55, 6.8, 7.1, 7.4, 7.7, 8.0}`,
and selection metric `order_suscept_abs_mean_P`.

## 6. Anti-scope-creep

`N_t`, the post-pilot β slate, the candidate pool, the gates, and the v0-identical
Stage-2 are frozen here. A `YM-P2-UNDERPOWERED` at v6 is the **PAUSE** endpoint,
not a trigger for further target-shopping (anti-p-hunting binding). 4D and
topological-charge targets remain deferred.
