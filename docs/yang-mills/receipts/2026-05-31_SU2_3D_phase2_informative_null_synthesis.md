# Yang-Mills Phase 2 Informative-Null Synthesis — SU2 3D (arc-complete)

- Synthesis id: `2026-05-31_SU2_3D_phase2_informative_null_synthesis`
- Cell label: `SU2_3D`
- Phase: 2 synthesis (supersedes the 2026-05-29 four-probe bounded-null synthesis)
- Date: 2026-05-31
- Supersedes:
  [`2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md`](2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md)
  (the v0–v3 four-probe synthesis; still valid for its four probes, but no longer
  the lane's current conclusion)
- Basis receipts (full arc):
  - v0–v3 (Wilson-loop targets): the four `YM-P2-NEG-A` receipts.
  - v4 (Wilson re-summary powered-target audit):
    [`2026-05-31_SU2_3D_phase2_v4_underpowered.md`](2026-05-31_SU2_3D_phase2_v4_underpowered.md)
  - v5 (symmetric Polyakov powered-target audit):
    [`2026-05-31_SU2_3D_phase2_v5_polyakov_underpowered.md`](2026-05-31_SU2_3D_phase2_v5_polyakov_underpowered.md)
  - v6 pilot (finite-T β-slate selection, voided):
    [`2026-05-31_SU2_3D_phase2_v6_pilot_unbracketed.md`](2026-05-31_SU2_3D_phase2_v6_pilot_unbracketed.md)
  - v6a (finite-T Polyakov, powered):
    [`2026-05-31_SU2_3D_phase2_v6a_finite_t_polyakov_neg_a.md`](2026-05-31_SU2_3D_phase2_v6a_finite_t_polyakov_neg_a.md)
- P0 lock + amendments:
  [`../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md),
  amendment 1 (APE smearing), amendment 2 (Polyakov target class).

## Scope

A synthesis receipt, not a new run. It consolidates the **full escalation arc**
on the registered `SU2_3D`, `12³` (+ a finite-T `12²×4`) envelope into one
load-bearing conclusion. It admits no Phase 3/4, continuum, confinement,
mass-gap, or Clay claim.

## The arc (every step evidence-triggered)

| Stage | What it tested | Target powered? | Disjoint? | Verdict |
| --- | --- | --- | --- | --- |
| v0–v3 | small-loop signature (bare / smeared / correlator) vs Wilson-loop targets (`γ_held`, `σ²_W33`) | **no** (later shown noise/leaky) | n/a | 4× `YM-P2-NEG-A` (**uninformative** — see v4) |
| v4 | powered-target audit over Wilson re-summaries of `{W14,W23,W33}` | **no** — power-vs-disjointness squeeze: the one near-powered candidate (`mean_W14`, β2.0 ICC 0.487) leaked into the signature (β2.8 CV-R² 0.576); the disjoint candidates lacked power | — | `YM-P2-UNDERPOWERED` |
| v5 | powered-target audit over **symmetric** Polyakov summaries | **no** (confined `⟨P⟩≈0`) | **yes** (all 3 disjoint) | `YM-P2-UNDERPOWERED` |
| v6 pilot | finite-T β-slate selection | — | — | `Z beta_peak_unbracketed` (pilot void — grid didn't bracket the peak) |
| **v6a** | **finite-T Polyakov** at the deconfinement crossover (`N_t=4`, slate `{6.3,6.55,6.8}`, peak 6.55) | **YES** (ICC 0.965) | **YES** (CV-R² −0.332) | **`YM-P2-NEG-A` — INFORMATIVE** |

The v4→v6a chain was the audit-first response to the v0–v3 nulls: it asked, before
any scoring, **whether a held-out target was even powered enough to make a null
mean something.** v4 and v5 showed the Wilson and symmetric-Polyakov envelopes
could not supply one; v6a, in the finite-temperature regime where the Polyakov
loop is a genuine order parameter, finally did.

## The upgrade: from uninformative to informative

The v0–v3 nulls did **not** implicate the signature — their targets were
noise-floored (Wilson `γ_held`/`σ²_W33`, confirmed by the v4/v5 audits) or
underpowered, so "no rank-locality" said nothing about whether the signature
*could* carry structure. **v6a removes that escape.**

> **Load-bearing statement.** On a **powered** (split-half ICC 0.965) and
> **disjoint** (leakage CV-R² −0.332) held-out target — the Polyakov loop, the
> SU(2) confinement order parameter, measured on a finite-temperature `12²×4`
> ensemble at the deconfinement crossover (`β` slate `{6.3, 6.55, 6.8}`, the
> susceptibility peak `β=6.55` consistent with the literature `β_c=6.53661(13)`,
> Edwards–von Smekal 2009) — the unchanged bare small-loop gauge-invariant
> signature `{W11,W12,W13,W22}` does **not** preserve within-β rank-local
> structure beyond controls (within-β primary bin-purity@5 `0.3042` vs
> `CTRL_RAND 0.3292`; across-β primary `= CTRL_RAND_STRAT` exactly). All three
> Polyakov candidates were powered and disjoint; `γ_held` correctly failed the
> power self-validation; `CTRL_GAUGE_RAND` matched the primary to machine
> precision. **This is an informative null: a powered, disjoint, physically
> meaningful test, and the small-loop signature carried no rank-locality of the
> order parameter on this cell.**

The escalation is what makes the null credible: the lane did not merely fail to
find structure, it **engineered the conditions where structure could have
appeared** (a powered, disjoint, order-parameter target) and the signature still
showed none.

## What it does and does not say

- **Does:** the bare small-loop signature does not rank-localize the Polyakov
  order parameter on this `SU2_3D` finite-T cell, on a genuinely powered and
  disjoint test. This is a substantive bounded null about *this signature on
  this cell*, not the uninformative-target nulls of v0–v3.
- **Does not:** any statement about Yang-Mills existence, confinement, the mass
  gap, the continuum limit, or the Clay problem; any claim that a *richer*
  signature, a different cell, or a different observable could not separate it;
  any positive Sundog result. The result is a credible negative.

**Honest open question (put to external review).** "Informative" here means
*non-empty* — the target carried signal and the signature still did not track it
— not necessarily *surprising*. There may be a standard gauge-theory reason a
small **local**-loop signature cannot resolve the **non-local** (temporal-wrap)
Polyakov order parameter config-to-config (a center-symmetry / locality
argument), in which case the v6a null is the *expected* outcome: a clean
confirmation that the apparatus correctly reports "no structure," rather than a
discovery. Whether this null is surprising or textbook is exactly the question
the re-pointed external-review packet (Q2) asks a lattice gauge theorist. The
synthesis does not pre-judge it; it claims only that the null is credible because
the target was powered and disjoint.

## Disposition — PAUSE at the informative endpoint

Per the v6a receipt and the binding anti-p-hunting clause: **no more automatic
probes.** The lane has run its strongest powered test and returned a credible
informative null. Further target/signature work requires fresh external
scientific motivation or reviewer feedback. The disciplined next move is the
external-review packet (re-pointed to this informative-null story), not another
probe. A `yang-mills.html` generality card may carry this synthesis's load-bearing
statement only with an "external review pending / returned: <verdict>" banner.

## Public Language Check

- [x] says "informative bounded finite-lattice null" rather than "Yang-Mills result"
- [x] does not say "Sundog proves confinement" or "found a mass gap"
- [x] does not imply continuum-limit reasoning
- [x] frames the powered finite-T Polyakov read as a credible negative, not a positive
- [x] every reopen stays gated on external motivation, not p-hunting
