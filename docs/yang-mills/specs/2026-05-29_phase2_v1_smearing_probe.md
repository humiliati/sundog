# Yang-Mills Phase 2 v1 Smearing Probe Spec

Filed: **2026-05-29 (PT)**
Author trigger: Phase 2 v0 named null
[`YM-P2-NEG-A no_rank_local_structure`](../receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md).
P0 lock: [`../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
v0 spec: [`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v0.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v0.md)

Status: **probe spec**, not a binding pre-registration. This document
records the diagnostic reading of the v0 null, the design-space
audit, the chosen v1 design change, and the filing list. It does not
itself admit any runner code. The binding artifacts triggered by this
probe are:

- a dated P0 amendment admitting APE smearing as a primary-signature
  class:
  [`../../prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md`](../../prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md);
- a Phase 2 v1 binding spec with signature vocabulary v4 (smeared) and
  the same held-out target, controls, bin convention, and thresholds
  as v0:
  [`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v1.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v1.md).

## v0 Result Read

From the v0 receipt:

| Quantity | v0 observed | v0 gate | Verdict |
| --- | --- | --- | --- |
| Primary within-β `mean_bin_purity_5` | `0.310416666667` | `>= 0.5` | fail |
| Primary margin over `CTRL_RAND` | `0.010416666667` | `>= 0.10` | fail |
| Chance baseline (tertile bins) | `0.333...` | — | primary sits **below** chance |
| Ensemble health (Phase-1-inherited gates) | all passed | — | not a void run |
| `CTRL_GAUGE_RAND` invariance | machine epsilon | `<= 1e-12` | confirms signature is invariant |

The primary score is inside the CTRL_RAND noise band and below chance.
This is not a marginal miss; this is a clean signal that the v1
signature vocabulary as registered carries no usable information about
the held-out `γ_held` tertile label within β at this lattice and
ensemble size. The pre-registration discipline did exactly what it was
designed to do: rejected a null result before any narrative dressing
could be applied.

## Diagnostic

Two non-exclusive hypotheses fit the v0 read:

1. **UV-noise dominance on bare small Wilson loops.** At moderate β
   (`{2.0, 2.4, 2.8}`) on a 12³ lattice, the per-configuration
   ensemble averages of bare `W11 ... W22` carry significant
   short-wavelength fluctuation noise relative to the per-configuration
   variation in the held-out area-law slope. The 32-config-per-β
   ensemble is small enough that the noise floor in the signature
   plausibly exceeds the signal corresponding to `γ_held` tertile
   structure.
2. **Genuine information disjointness.** The small-loop ensemble
   averages may simply not carry `γ_held` information at this lattice,
   regardless of noise. Larger-loop area-law slope could depend on
   physical structures (e.g. extended-loop fluctuation patterns) that
   are not reflected in `<W11>`, `<W12>`, `<W13>`, `<W22>` ensemble
   means / variances at all.

These hypotheses are not separable from one v0 run alone. The right
next move is a v1 design change that addresses (1) directly with a
well-validated technique; if v1 still lands NEG-A, that pushes the
probability mass onto (2) and motivates a deeper rethink of the
held-out target or signature class.

## Design Space Audit

Three v1 candidate design changes were considered (lit-pass-supported,
ordered by smallest-to-largest delta against P0):

### T1 — Per-orientation signature vocab v2

- Lift the v1 signature from position-and-orientation-averaged 8-dim
  to position-averaged-only 24-dim, with separate `mean` / `var` per
  plane orientation `{xy, xz, yz}`.
- No P0 amendment required: a new vocab version is admissible under
  P0's vocabulary-versioning convention.
- **Expected risk of NEG-A repeat: high.** Phase 1 SU(2) 3D confirmed
  orientation isotropy at 0.29 % spread, well under the 5 % gate. With
  isotropy this clean, per-orientation breakdowns are mostly noise
  replication, not new information.
- Implementation cost: small (signature emission schema bump in the
  aggregation runner).

### T2 — Connected 2-point correlator signature vocab v3

- Replace mean / var summaries with connected correlations
  `<W11(x) W11(x + r)> - <W11(x)>²` at a frozen displacement slate
  (e.g. `r ∈ {(1,0,0), (1,1,0), (1,1,1), (2,0,0)}`), plus analogous
  for higher loops.
- No P0 amendment required: connected correlators are gauge-invariant
  and built from the same loop set, so the change stays inside the
  P0 fixed-loop framework.
- **Expected risk of NEG-A repeat: medium.** Captures spatial
  structure missing from mean / var, but connected correlators of bare
  small loops suffer the same UV-noise issue as the marginal means do
  — they are noisy estimators per configuration.
- Implementation cost: medium (new core routine for correlator
  evaluation, displacement slate to freeze, schema bumps).

### T3 — APE smearing on signature, vocab v4

- Apply APE smearing to the link variables with frozen smearing
  parameters `(α, N_sm)`, then compute the same `{W11, W12, W13, W22}`
  mean / variance signature on the smeared links. Held-out target
  remains bare-link `{W14, W23, W33}` → `γ_held`.
- **P0 amendment required.** P0 lock §"Primary Signature Vocabulary"
  explicitly forbids smearing as a primary-signature class. The
  amendment freezes smearing parameters at this filing and admits
  vocabulary v4.
- **Expected risk of NEG-A repeat: low.** APE smearing is the
  textbook lattice-QCD response to UV-noise-dominated small Wilson
  loops; it suppresses short-wavelength gauge fluctuations while
  preserving gauge invariance and the area-law structure. Lit-pass §5
  identifies smearing as a candidate signature class explicitly
  deferred at P0.
- Implementation cost: medium (new smearing routine in the SU(2) 3D
  core lib, P0 amendment doc, Phase 2 v1 binding spec).

## Selection

**T3 chosen.** Rationale:

1. The v0 failure pattern is the textbook UV-noise-dominated case
   that smearing is built to address. T1 and T2 do not directly
   attack the noise floor; T3 does.
2. T1's expected information gain is small because Phase 1 already
   confirmed orientation isotropy is clean.
3. T2 risks falling into the same UV-noise trap as v0.
4. T3 is the lit-pass scaffold's natural next step after a fixed-loop
   null. P0 forbade smearing at v0 precisely so that any positive read
   with smearing would be a real signature-class promotion, not a
   p-hack.
5. The cost of T3 is small: one P0 amendment paragraph freezing two
   smearing parameters, one v1 spec mirroring v0 with vocab v4, and a
   single aggregation runner that consumes the v0 ensembles.

The selection of T3 over T1 / T2 is recorded here before any v1
implementation work begins, so that no design choice is retrofitted
to the v1 result.

## Scientific Comparison Discipline

The v1 design uses the **same ensembles** as v0 (same per-β master
seeds `0201` / `0202` / `0203`, same 12³ × 3536 combined sweeps, same
held-out target bare loops). The only differences in v1 are:

- the signature pipeline (apply APE smearing to bare links → compute
  `W11 ... W22` on smeared links);
- the aggregation runner (which is the only place smearing happens).

This gives a clean head-to-head comparison: bare-loop signature failed,
smeared-loop signature is being tested on the **identical** bare-loop
ensemble, against the **identical** held-out target with the
**identical** per-β tertile bin edges.

`γ_held` is unchanged in v1 because the held-out target loops are
unchanged, the held-out summary derivation is unchanged, and the
ensemble configurations are unchanged. The per-β tertile bin edges
computed in the v1 aggregation runner must therefore match the v0
edges to machine epsilon; the v1 runner asserts this match as a void
gate.

## Filing List

This probe spec triggers two binding documents and one runner
implementation:

1. [`../../prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md`](../../prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md)
   — frozen smearing parameters, admits signature vocab v4.
2. [`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v1.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v1.md)
   — binding Phase 2 v1 spec with vocab v4.
3. New runner `scripts/yang-mills-phase2-v1-su2-3d-aggregate.mjs` and
   npm script `yang-mills:phase2:v1:su2-3d:aggregate`. The v0
   aggregation runner remains bit-for-bit unchanged.

No new per-β ensembles are required. The v0 ensemble dirs at
`results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta<β>_ensemble_v0/`
are the v1 input.

## What A v1 Outcome Tells Us

| v1 verdict | Diagnostic interpretation | Next allowed step |
| --- | --- | --- |
| `P2-A bounded_positive` | UV-noise hypothesis confirmed; smeared signature carries `γ_held` structure on this ensemble | draft Phase 3 observable-certificate manifest per P0 §8 Phase 3 |
| `YM-P2-NEG-A no_rank_local_structure` (again) | Hypothesis (2) dominant: small-loop summaries (bare or smeared) do not carry `γ_held` information on this cell | new probe spec proposing a target / signature redefinition (e.g. switch to a non-area-law held-out target, or use a non-loop signature class); likely a v2 P0 amendment |
| `YM-P2-NEG-B metadata_only` | smearing recovered β-class information but not within-β structure | reconsider per-β bin convention or move to within-β stratified primary scoring with bin-edge robustness checks |
| `YM-P2-NEG-D raw_dominates` | smeared signature is matched by raw matrix-entry NN at the chosen margin | named null; reconsider whether the signature lives in the right space |

Either way, a v1 receipt is a real outcome; another named null is
also a publishable receipt and a directional signal for the v2 design.

## Anti-Scope-Creep

The smearing parameters `(α, N_sm)` are frozen in the P0 amendment
filed alongside this probe spec. They are not allowed to be retuned
after v1 lands a verdict. If v1 lands NEG-A and a future probe spec
wants to test different smearing parameters, that requires a new dated
P0 amendment and a new Phase 2 vNNN spec, never an in-place revision
of either this probe or the v1 spec.
