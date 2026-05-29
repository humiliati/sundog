# Yang-Mills Phase 2 v2 Connected-Correlator Probe Spec

Filed: **2026-05-29 (PT)**
Author triggers:
- Phase 2 v0 named null
  [`YM-P2-NEG-A no_rank_local_structure`](../receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md)
- Phase 2 v1 (APE-smearing) named null
  [`YM-P2-NEG-A no_rank_local_structure`](../receipts/2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md)
- Phase 2 v1 probe spec (T1/T2/T3 design audit; T3 then chosen)
  [`2026-05-29_phase2_v1_smearing_probe.md`](2026-05-29_phase2_v1_smearing_probe.md)

P0 lock: [`../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
P0 amendment 1 (APE smearing):
[`../../prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md`](../../prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md)
v0 spec: [`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v0.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v0.md)
v1 spec: [`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v1.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v1.md)

Status: **probe spec**, not a binding pre-registration. This document
records the post-v1 diagnostic reading, the design-audit promotion of
T2 (from the v1 audit) to v2, and the filing list. It does not itself
admit any runner code. The binding artifact triggered by this probe is
[`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v2.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v2.md).
No P0 amendment is required for v2 because connected 2-point
correlators of bare links are not smearing or blocking; they are a new
gauge-invariant summary class within the P0 fixed-loop framework.

## v1 Result Read

From the v1 receipt:

| Quantity | v1 observed | v1 gate | v0 baseline | Verdict |
| --- | --- | --- | --- | --- |
| Primary within-β `mean_bin_purity_5` | `0.29375` | `>= 0.5` | `0.31042` | fail (and **slightly below v0**) |
| Primary margin over `CTRL_RAND` | `-0.00208` | `>= 0.10` | `0.01042` | fail (and below CTRL_RAND) |
| Max smearing det drift | `6.66e-16` | `<= 1e-10` | n/a | pass (machine epsilon) |
| Post-smearing-step link unitarity | `9.42e-16` | `<= 1e-10` | n/a | pass (machine epsilon) |
| CTRL_GAUGE_RAND bin-purity match | `1.44e-15` | `<= 1e-12` | `<= 1e-12` (v0) | pass (machine epsilon, after bug fix below) |
| Aggregation wall clock | `25.78 s` | `<= 600 s` | n/a | pass |

The corrected APE smearing produced numerically clean numerics on all
smearing-health gates AND landed a slightly worse primary score than
v0. Smearing did not move the rank-locality signal to a different
place; if anything it removed a tiny amount of structure. This is a
clean empirical statement about a frozen parameter choice
`(α, N_sm) = (0.5, 10)` registered in P0 amendment 1, not a
"smearing parameter unhelpful" claim for all `(α, N_sm)` — those
parameters are locked at amendment 1 and cannot be retuned without a
fresh amendment.

## Methodology Receipt: CTRL_GAUGE_RAND Caught A Real Bug

Before the corrected APE run produced the v1 numerics above, an
earlier draft of `scripts/lib/yang-mills-su2-3d-smearing.mjs` used the
**heatbath-staple orientation** for the APE-smearing staple sum. The
heatbath staple is the matrix product order optimized for Boltzmann
conditional sampling of a single link given its neighbors; the APE
staple is the gauge-invariant product order around the plaquette
required for the smearing formula to commute with gauge transforms.
Same staple, different traversal direction; the wrong order silently
breaks gauge invariance after the smearing step.

The bug surfaced as a `YM-P1-NEG-A gauge_leakage` because
`CTRL_GAUGE_RAND` recomputed the smeared signature on a gauge-rotated
ensemble and observed it had **changed** — the post-smearing signature
was no longer invariant under random Haar gauge transforms. The
pre-registration discipline rejected the smearing implementation
before any rank-locality score could be interpreted; the bug surfaced
as an implementation issue rather than as a misleading signal.

This is recorded as a methodology success: the entire CTRL_GAUGE_RAND
control category exists for exactly this kind of catch. The corrected
APE implementation re-ran and produced the v1 receipt's clean
numerics; the v1 NEG-A verdict is on the corrected algorithm, not on
the buggy first draft.

## Diagnostic After v1

The candidate-hypothesis table from the v1 probe spec is updated as
follows:

1. **Hypothesis 1 — UV-noise dominance on bare small Wilson loops.**
   **FALSIFIED** at v1. Smeared-loop mean/variance summaries with
   frozen lattice-QCD-standard `(α, N_sm) = (0.5, 10)` produced a
   primary score slightly *lower* than the bare-loop baseline. UV
   noise is not the limiting factor for the v0/v1 signature class on
   this cell.
2. **Hypothesis 2 — Genuine information disjointness between
   small-loop mean/var summaries and the held-out `γ_held` bin
   structure.** **STRONGLY FAVORED**, but the claim must be precise:
   v0 and v1 tested only **marginal mean and variance** summaries of
   each individual small loop, both on bare and smeared links. The
   space of gauge-invariant small-loop summaries is larger than that.

What v0 and v1 jointly established is that the **marginal** small-loop
moment summary, in either bare or smeared link representation, does
not preserve the `γ_held` bin label in within-β nearest-neighbor rank
space at this lattice and ensemble size. Whether **higher-order
joint structure of small loops** (spatial correlations, connected
multi-point functions) preserves it is still an open question, and is
the natural last test before pivoting to target-side redesign.

## Design Audit — Why T2 Now

The v1 probe spec's design-space audit listed three v1 candidates
(T1 per-orientation, T2 connected 2-point correlator, T3 APE
smearing). T3 was chosen for v1 on the grounds that the UV-noise
hypothesis was the most likely culprit and smearing is the textbook
remedy. T1 was deprioritized (orientation isotropy already clean).
T2 was deprioritized at the time because *"connected correlators of
bare small loops suffer the same UV-noise issue as the marginal means
do — they are noisy estimators per configuration."*

After v1, that reason is moot. UV noise has been ruled out as the
limiting factor. T2 now becomes the natural last test of the
small-loop hypothesis at a richer level: do connected 2-point
correlations of bare small loops at fixed displacements carry
`γ_held` structure that marginal means and variances do not?

If T2 also lands NEG-A, the small-loop hypothesis is fully
exhausted in its most natural extensions (marginal moments + 2-point
spatial structure, both bare and smeared) and the diagnostic mass
moves strongly onto the target side. That is the v3 question.

## v2 Design — Locked

Signature vocabulary v5 (this probe-spec freezes the convention; the
v2 binding spec consumes it):

- **Per-config dimension:** 20.
- **Per-loop-class:** four small loops `W ∈ {W11, W12, W13, W22}`
  (same as v0/v1 v1-vocab loops; bare-link `(1/2) Re Tr U_loop`).
- **Per-displacement-class:** five **frozen** cubic-symmetry
  displacement classes:

| Class id | Representative `r` | Cubic class size | Distance bucket |
| --- | --- | --- | --- |
| `r1` | `(1, 0, 0)` | 6 | nearest neighbor, axis |
| `r2` | `(1, 1, 0)` | 12 | plane diagonal |
| `r3` | `(1, 1, 1)` | 8 | body diagonal |
| `r4` | `(2, 0, 0)` | 6 | next-axis |
| `r5` | `(2, 1, 0)` | 24 | mixed axis + perpendicular |

- **Per-(loop, displacement-class) entry:** the connected 2-point
  correlator

```text
C_W(r-class) = mean over (position x, equivalent r in class)
               of [ W(x) * W(x + r) - <W>^2 ]

<W> = position-and-orientation average of W over the whole lattice
      for this configuration
```

  computed with position averaging running over all base sites in
  12³ and the standard 3-plane-orientation averaging for the
  loop trace. `W(x + r)` is the loop anchored at the wrapped position
  `(x + r) mod 12` per direction.

- **Why correlators only (no marginal means or variances retained):**
  v0 established marginal means + variances carry no `γ_held` signal
  in rank-locality. Including them in v5 would only dilute the test
  of whether **spatial structure alone** preserves the held-out
  label; carrying the v0 zero-signal channels alongside the new
  correlator channels also slightly worsens the distance metric in
  z-score-normalized space at N=32 per β. Keeping v5 = 20-dim
  correlators alone is the cleanest scientific probe.

- **Why these five displacement classes:** they span axis-aligned
  short (`r1`), plane-diagonal short (`r2`), body-diagonal short
  (`r3`), axis next-step (`r4`), and a mixed direction (`r5`) — five
  distinct cubic-symmetry classes covering the small-`r` regime on
  12³. A larger displacement slate adds dimension without obviously
  adding signal (correlators of small loops generally decay
  exponentially with `r`, so very large `r` is dominated by noise).
  Five is the locked count at this probe.

- **Why bare links (no smearing in v5):** smearing was tested in v1
  and did not help marginal moments. Compounding two design changes
  (smearing + correlators) would obscure which one matters; v2
  isolates the correlator change. A future v6 could test
  smeared-correlator vocab v6 if the v2 verdict warrants it.

## Scope Boundary — No Smearing Parameter Retune

P0 amendment 1 froze the APE smearing parameters at
`(α, N_sm) = (0.5, 10)` and explicitly forbade retuning them after
any receipt was filed using them. The v1 NEG-A receipt has been
filed; the smearing parameters are now **frozen for the lifetime of
this lane** unless a fresh dated P0 amendment is filed proposing
different values with explicit justification. v2 is not such an
amendment, by design: it does not use smearing at all. A future
v6 spec proposing smeared correlators (vocab v6 + amendment) would
inherit the same `(α, N_sm)` lock, not a retune.

## v3 Fallback (Pre-Stated)

If v2 also lands `YM-P2-NEG-A`, the diagnostic mass moves strongly
onto target-side redesign. The v3 probe spec at that point would have
to choose between:

| v3 candidate | P0 impact | Rationale |
| --- | --- | --- |
| **Different target derived from same loops** (e.g. variance of largest held-out loop, or per-orientation γ_held spread) | none | tests whether γ_held LS-slope is the wrong summary, with the same underlying loops |
| **Polyakov-loop-based held-out target** | P0 amendment 2 (Polyakov is deferred at P0) | tests whether a fundamentally different gauge-invariant observable encodes a label the small-loop signature can resolve |
| **Topological-charge-proxy held-out target** | P0 amendment 2 (topological is deferred at P0) | tests against a non-area-law class entirely |
| **Mass-gap proxy from correlator decay at fixed time** | within-P0 if defined as small-loop-correlator summary | tests a physics-motivated target derived from the signature itself |

Pre-stating these options is pre-registration discipline: it limits
the v3 design space and prevents a "shop for a target that works"
search if v2 also fails. The actual v3 selection (if needed) would
follow the same probe-spec → P0-impact → binding-spec flow as v1
and v2.

## Filing List

This probe spec triggers one binding document and one runner
implementation pair:

1. [`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v2.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v2.md)
   — binding Phase 2 v2 spec with signature vocab v5.
2. New shared module
   `scripts/lib/yang-mills-su2-3d-correlator.mjs` (computes the
   per-loop position-averaged Wilson-loop array plus the
   per-displacement-class connected correlator), and a new aggregation
   runner `scripts/yang-mills-phase2-v2-su2-3d-aggregate.mjs` paired
   with the npm script `yang-mills:phase2:v2:su2-3d:aggregate`. The
   v0 and v1 aggregation runners + the bare SU(2) 3D core remain
   bit-for-bit unchanged.

No new per-β ensembles are required. The v0 ensemble dirs at
`results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta<β>_ensemble_v0/`
are the v2 input, exactly as for v1.

## What A v2 Outcome Tells Us

| v2 verdict | Diagnostic interpretation | Next allowed step |
| --- | --- | --- |
| `P2-A bounded_positive` | Spatial 2-point structure of small loops carries `γ_held` within β; the v0/v1 NEG-As were specifically about *marginal* moments missing the spatial structure | draft Phase 3 observable-certificate manifest per P0 §8 Phase 3 |
| `YM-P2-NEG-A no_rank_local_structure` (again) | Small-loop hypothesis exhausted across marginal and 2-point spatial structure, bare and smeared. Move to v3 target redesign per the table above. | new v3 probe spec; likely a fresh P0 amendment for Polyakov or topological observables |
| `YM-P2-NEG-B metadata_only` | CTRL_META meets or beats correlator primary; reconsider bin convention or scoring | new v3 probe spec on the bin / scoring side |
| `YM-P2-NEG-D raw_dominates` | CTRL_RAW matches or beats correlator primary; means the lossy invariant shadow is still too coarse even with 2-point spatial structure | new v3 probe spec; signature class needs to expand beyond pure small-loop information |

Any of these is a publishable receipt under P0's named-null convention.

## Anti-Scope-Creep

The displacement slate `{r1 ... r5}` is frozen at this probe spec.
The vocab v5 dimension is 20 (4 × 5), locked. The per-loop set is
`{W11, W12, W13, W22}`, unchanged from v0. The held-out target is the
unchanged bare-link `γ_held`. The per-β tertile bin edges are
unchanged from v0 (asserted in the v2 binding spec by re-reading v0's
`per_beta_bin_edges.json`). The promotion thresholds are unchanged
from v0.

If v2 lands NEG-A and a future probe spec wants to test a different
displacement slate or include more loops in the correlator, that is
a new dated probe spec and a new vocab vNNN, never an in-place
revision of this probe or the v2 binding spec.
