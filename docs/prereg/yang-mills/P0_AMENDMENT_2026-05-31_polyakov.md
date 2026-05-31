# P0 Amendment 2 - Polyakov Loop as Admissible Held-Out Target Class

Filed: **2026-05-31 (PT)**
Amendment id: `P0_AMD_002_POLYAKOV_2026-05-31`
Parent P0 lock: [`P0_DOMAIN_AND_RECEIPT_LOCK.md`](P0_DOMAIN_AND_RECEIPT_LOCK.md)
Prior amendment: [`P0_AMENDMENT_2026-05-29_ape_smearing.md`](P0_AMENDMENT_2026-05-29_ape_smearing.md)
Triggering receipt:
[`../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v4_underpowered.md`](../../yang-mills/receipts/2026-05-31_SU2_3D_phase2_v4_underpowered.md)
Triggering diagnostic:
[`../../yang-mills/receipts/2026-05-30_cheap_a_controls_target_validity.md`](../../yang-mills/receipts/2026-05-30_cheap_a_controls_target_validity.md)

Status: **P0 amendment, design lock**. Modifies the parent P0 lock as specified.
Admits no claim and no runner code by itself. Binding consumers are
[`PHASE2_SU2_3D_relative_locality_v5.md`](PHASE2_SU2_3D_relative_locality_v5.md)
(symmetric, on the v0 ensembles) and
[`PHASE2_SU2_3D_finite_t_polyakov_v6.md`](PHASE2_SU2_3D_finite_t_polyakov_v6.md)
(finite-temperature, new ensembles, pre-stated).

## Purpose

> Promote the **Polyakov loop** from "deferred at P0" to an **admissible
> held-out TARGET class** (held-out vocabulary v4). It is **not** a signature
> and **not** a topological-charge proxy (those stay deferred). The v1 bare-link
> signature and the v1/v2/v3 Wilson-loop held-out targets remain valid and
> unchanged.

**Why now.** Phase 2 v4 (`YM-P2-UNDERPOWERED`) established by audit that the 12³
envelope contains **no Wilson-loop re-summary that is both powered and disjoint**
— a structural power-vs-disjointness squeeze (high-area loops are noise; low-area
loops sit at the signature's own scale and leak). The Polyakov loop is the
natural escape: a global temporal-wrap is **disjoint from small Wilson loops by
construction**, so it dodges the squeeze on the disjointness side. Whether it is
**powered** is the open question this amendment's consumers test, under the same
powered-target discipline.

## Scope — parent P0 sections modified

- §"Scope" → "Out of scope": narrow the bullet `topological-charge proxies,
  Polyakov-loop proxies (deferred...)` to **`topological-charge proxies
  (deferred)`** only; the Polyakov loop is admitted **as a held-out target**
  by this amendment (still not as a signature, still not as a Clay/continuum
  claim).
- §"Lattice Slate (locked)" → add, for `SU2_3D` only, a **finite-temperature
  asymmetric slate** `12² × N_t` consumed by the v6 binding spec (below). The
  symmetric `8³ / 12³` slate is unchanged; the symmetric v5 consumer reuses the
  existing 12³ v0 ensembles.
- §"Held-Out Observable Label (locked)" → add **held-out target vocabulary v4**
  (Polyakov class) alongside the existing Wilson-loop targets; the v1/v2/v3
  values are unchanged. One-line clarification: the Polyakov target is gauge
  invariant and disjoint from the signature, satisfying the held-out-label rule.
- §"Open Decisions Closed By This Lock" → add a row noting the Polyakov target
  class is admitted here.

Every other section of the parent P0 lock is unchanged.

## Locked Polyakov Observable

Frozen at this amendment; may not be retuned after any receipt cites them.

| Field | Registered value |
| --- | --- |
| Object | Polyakov loop wrapping a periodic direction μ at transverse site `x_⊥`: `P(x_⊥; μ) = (½) Re Tr ∏_{t=0}^{N_μ−1} U_μ(x_⊥ + t·μ̂)`. For SU(2) this is the `[0]` (real-scalar) component of the ordered link-quaternion product, matching `meanPlaquette`'s `(½)Tr` convention. |
| Per-config summaries (held-out vocab v4 candidate pool) | `abs_mean_P` = `\|mean over x_⊥ of (½)Tr P\|`; `mean_abs_P` = mean over `x_⊥` of `\|(½)Tr P\|`; `chi_P` = spatial variance of `(½)Tr P(x_⊥)` (Polyakov susceptibility). |
| Symmetric cell (`SU2_3D` 12³) | wrap each of the 3 periodic directions; each summary averaged over the 3 wrap directions. |
| Finite-T cell (`12² × N_t`) | μ = the temporal direction only; summaries over the `12²` transverse sites. |
| Gauge invariance | the Polyakov loop is a closed gauge-invariant Wilson line; `CTRL_GAUGE_RAND` must leave each summary invariant to ≤ 1e-12. |
| Disjointness | the held-out target must still pass the consumer's leakage gate `CV-R²(target \| v1 signature) ≤ 0.25` — Polyakov is expected to pass by construction, but it is **gated, not assumed**. |

The three summaries are a frozen pool, not a scan; a future probe wanting a
different Polyakov summary (e.g. a correlator length) requires a new dated
amendment.

## Finite-Temperature Slate (consumed by v6)

| Field | Registered value |
| --- | --- |
| Geometry | `12 × 12 × N_t` (2 spatial + 1 short temporal), periodic all directions |
| `N_t` | `4` (standard 2+1D finite-T temporal extent for SU(2)) |
| β slate | 3 points straddling the SU(2) 2+1D deconfinement crossover at `N_t = 4`; **the exact β values are set once in the v6 runner manifest after a pre-generation pilot Polyakov scan** confirms the slate brackets the crossover (mirrors parent P0's "β revisable once before ensemble generation"); frozen thereafter. |
| Generator | unchanged Creutz + Kennedy-Pendleton + Brown-Woch (the SU(2) 3D core, generalized to an asymmetric lattice — an implementation requirement of the v6 runner, not a physics change). |
| Burn-in / thinning | inherit the parent P0 rules (≥ 2000 sweeps; thinning ≥ 2·τ_int from a pilot). |

## Why The Signature Stays Small-Loop Bare

The test is unchanged in spirit: *does the bare small-loop signature
`{W11,W12,W13,W22}` preserve the structure of a disjoint held-out observable*. The
Polyakov target is a different geometric object (global wrap), so a positive read
cannot be explained as signature-into-target leakage — exactly what the v4
squeeze showed the Wilson-loop targets could not guarantee. Smearing
(amendment 1) is admissible on the signature but **not** applied to the Polyakov
target.

## Health Gates Added By This Amendment

| Quantity | Threshold | Branch if missed |
| --- | --- | --- |
| `CTRL_GAUGE_RAND` on each Polyakov summary | invariant to ≤ 1e-12 after a random gauge transform | `YM-P1-NEG-A gauge_leakage` (existing branch, extended to the target side) |
| Polyakov summary defined per config | finite, non-degenerate ensemble (spread ≥ 1e-10) | `Z bin_degenerate` (existing) |
| Finite-T ensemble health (v6) | parent P0 τ_int / thinning / unitarity gates | parent P0 branches |

## Compute Cost

- **v5 (symmetric, on v0):** read the 96 stored `su2_links.jsonl` configs, compute
  the wrap loop over 3 directions × 1728 sites × 12 links — well under one second
  per config; the whole audit is a ~v4-scale aggregation pass, inside the
  10-minute cap.
- **v6 (finite-T):** requires generalizing the SU(2) 3D core to an asymmetric
  lattice and generating new `12²×4` ensembles — a real generation pass (its own
  next step), still inside the per-invocation cap at this small volume.

## Forbidden, Restated

- using the Polyakov loop as a **signature** component (it is a held-out target
  only at this amendment);
- admitting **topological-charge** proxies (still deferred);
- promoting any Polyakov result to a Clay / continuum / mass-gap claim, or to 4D
  (4D remains deferred);
- retuning the frozen summaries, `N_t`, or (after the pilot) the finite-T β slate
  without a fresh dated amendment;
- applying the v6 finite-T result to reinterpret the symmetric-cell nulls, or vice
  versa, beyond their own registered envelopes.

## Open-Decision Resolution

Parent P0 §"Open Decisions" gains:

> 6. Held-out target class beyond small Wilson loops? — Admitted: the **Polyakov
>    loop** as held-out vocab v4 at `P0_AMD_002_POLYAKOV_2026-05-31`, after the v4
>    Wilson-loop audit returned `YM-P2-UNDERPOWERED` (power-vs-disjointness
>    squeeze). Topological charge remains deferred.

## Current State

- 2026-05-31: amendment filed; Polyakov admitted as held-out vocab v4; observable
  + summaries frozen; finite-T `12²×4` slate registered (β pending pilot). Triggers
  the binding v5 (symmetric, on v0) and v6 (finite-T) specs. No runner code
  admitted by this amendment.
