# Percival → Promo/Webdev Handoff — Surfaces to Update or Expand

*Drafted 2026-07-01 for the promo/webdev team. Self-contained: assumes no session
context. Owner review pending; nothing herein is committed or deployed.*

---

## TL;DR

The Percival lane (un-targeting vs quantilizers, Grail-romance scaffold) resolved its
main track and then completed a six-entry reopen slate in one day. It produced **three
distinct shoppable results** — one machine-checked theorem, one cross-channel law, one
policy-shaped oversight finding — plus a literary keeper post awaiting the owner's
re-voice. Every claim below is falsifier-fenced and in-vitro; the **Hard Fences** section
is load-bearing, not boilerplate. Nothing is committed; licensing for public Percival
surfaces is an open owner decision.

---

## What is banked (receipt chain, one table)

| result | verdict token | receipt |
| --- | --- | --- |
| Measure channel safe (static toy) | `A3_STATIC_DEFLATION_MEASURE_SAFE` | `PERCIVAL_A3_KILLSWITCH_RESULTS.md` |
| Court mechanism admitted (emergent cliff, 4/4 gates) | `B1_0_COURT_ADMITTED` | `PERCIVAL_B1_0_COURT_COORDINATION_RESULTS.md` |
| Conditional separation mapped (support-above clean / straddle partial) | `B1_SEPARATION_MAPPED_CONDITIONAL` | `PERCIVAL_B1_SEPARATION_RESULTS.md` |
| Clean unconditional prize deflates structurally on trusted bases | `B2_CLEAN_IS_CORNER` | `PERCIVAL_B2_GAMMA_PROVENANCE_RESULTS.md` (**PRIVATE**) |
| Un-targeting composes; stability = robustness not reward | `B3_COMPOSES_STABILITY_NEEDS_ROBUSTNESS` | `PERCIVAL_B3_COMPOSITION_STABILITY_RESULTS.md` |
| Cleanliness law + asymmetry-of-reach duality | (synthesis) | `PERCIVAL_S1_CLEANLINESS_LAW.md` |
| Target cap collapses to masking; leak ceilinged | `S2_TARGET_CAP_COLLAPSES_TO_MASK` | `PERCIVAL_S2_TARGET_CAP_RESULTS.md` |
| Shared court unreachable; caps compose adversarially | `S3_SHARED_COURT_UNREACHABLE_CAP_COMPOSES` | `PERCIVAL_S3_MULTIAGENT_RESULTS.md` |
| Counterproductivity general + machine-checked | `S4_COUNTERPRODUCTIVITY_GENERAL_CONFIRMED` | `PERCIVAL_S4_COUNTERPRODUCTIVITY_RESULTS.md` + `sundogcert Sundogcert/PercivalGeneral.lean` (audit green, 8563 jobs) |
| Class boundary = pointwise sign; graded-vs-threshold oversight | `S6_POINTWISE_BOUNDARY_TAX_SHAPE_REFINES` | `PERCIVAL_S6_CLASS_BOUNDARY_RESULTS.md` |

Lane spine: `docs/SUNDOG_V_PERCIVAL.md`. Umbrella: `docs/SUNDOG_V_CAUSAL_ACCESS.md`.
Slate with per-entry results: `PERCIVAL_REOPEN_HYPOTHESIS_SLATE.md`.

---

## The shoppable angles, ranked

### Angle 1 — The counterproductivity theorem *(fastest, strongest, non-literary)*
**Claim (exact form, condition included):** on a reward nonincreasing in proxy-pressure —
i.e., when the proxy is *pointwise misaligned* with the true reward — quantilizing is
counterproductive: the best `q`-quantilizer is the untilted base (`q=1`), every tilt
strictly loses, and iterated tilting drifts to ruin; un-targeting strictly out-collects
the whole family when the base support is past the oversight cliff.
**Status:** machine-checked in Lean 4 + Mathlib for arbitrary finite support
(`upper_tail_mul_le_base_mul`, `best_quantilizer_is_base_general`,
`clean_support_above_separation_general`; axioms exactly
`[propext, Classical.choice, Quot.sound]`, no `sorry`), fuzz-tested on 10,500 instances
including the actual court model.
**Audience:** the quantilization / mild-optimization / performative-prediction circle.
Cites to carry: Taylor 2015 (quantilizers); Perdomo et al. 2020 (performative
prediction — our threshold maps are the non-Lipschitz regime their theorems leave open);
Elster 1983 (*Sour Grapes*, states that are essentially by-products); Manheim &
Garrabrant 2018 (this is causal/adversarial Goodhart, not regressional).
**The condition is not optional** (see S6): drop "pointwise misaligned" and the claim is
false — for a pointwise-good proxy, tilting strictly helps.

### Angle 2 — The cleanliness law / asymmetry of reach *(the frame piece)*
**Claim:** a channel's safety primitive screens off values cleanly iff the overseer can
exogenously project to the channel's safe point; with the umbrella fence this completes
into a duality — *clean screening = the agent cannot reach the bound ∧ the bound can
reach the safe point.* Act/aggregate are clean (clip an outgoing arrow at the actuator);
target is partial (an incoming-dependence condition, not projectable — in-channel
enforcement collapses to a measure retreat priced at the proxy's reliability edge).
Composition rider (S3): behavior-keyed guarantees compose by construction; outcome-keyed
guarantees compose only as far as the reward decomposes.
**Status:** synthesis over banked receipts; a *characterization*, with live falsifiers
(the `Sov_opt` formal classification check is open and would void it).
**Audience:** causal-incentives / CID readers (Everitt et al. 2021 is the genus — cite,
don't reinvent); the alignment-theory blog register.
**Care level: high.** It earns "law" only while its falsifiers keep not firing; surface
it with the falsifiers visible, or not at all.

### Angle 3 — Graded oversight beats threshold oversight *(policy-shaped, most quotable)*
**Claim:** the cliff-edge temptation — the reward-optimal agent courting right up to the
review boundary — is a *threshold-oversight artifact*: it comes from the flat
zero-marginal region below the threshold (tilting there is free). A graded tax charges
every tilt at the margin, and when steep enough restores "don't tilt at all" as optimal
*even for a pointwise-good proxy*.
**Status:** S6 computed result (registered predictions P1–P3/P5 confirmed; P4 missed and
the miss is this finding), coherent with three independent probes (B3.2, S2.4, S3) and
with NS-3's empirical graded-review knob — the theory rediscovered the lane's own
instrument.
**Audience:** oversight-design / RLHF-adjacent readers; the most accessible of the three.
**Honest register:** one toy family, post-hoc width sweep labeled as such, interior
optima are knife-edges.

### The literary keeper *(separate track, owner-gated)*
`PARTIAL_GRACE_UNTARGETING_POST.md` — the honest deflation-shaped essay (un-targeting
helps but only half-wins on any trustworthy base; "un-targeting's clean edge appears only
in a fallen court"). **Awaiting the owner's re-voice; easter-egg / uncached-corner
disposition, same as the cap-not-council post. Not a promo blast.**

---

## Per-surface recommendations

| surface | action | notes |
| --- | --- | --- |
| `PARTIAL_GRACE_UNTARGETING_POST.md` | **UPDATE (light) after owner re-voice** | Written before the reopen slate. Two claims can now be strengthened: the Lean anchors are general (n-point), not just 3-point; and "safe not optimal" now carries S6's conditionality (dominated iff the proxy is pointwise-good). One footnote-grade addendum each; do not restructure the essay. |
| `CREDIT_THE_CAP_NOT_THE_COUNCIL_POST.md` | **HOLD** | Self-contained NS keeper; the Percival arc does not change it. A single cross-pointer to the cleanliness law is optional. |
| `SUNDOG_V_CAUSAL_ACCESS.md` (umbrella) | **CURRENT** | Already carries the S5 duality fold and per-channel ledger; no promo edit needed. It is the internal spine, not a public page. |
| New public page / post for Angle 1 | **EXPAND (new surface)** | The decoupled theorem note: claim + condition + Lean receipt + fuzz receipt + the four cites. No Grail framing required (works standalone); the romance can be one paragraph of provenance color or omitted. |
| New short note for Angle 3 | **EXPAND (new surface, optional)** | "Threshold oversight tempts; graded oversight protects" with the width-sweep table. Could ride inside Angle 1's page as a section instead of standing alone. |
| Angle 2 page | **DEFER** | Wait for the `Sov_opt` formal check (the live falsifier) before giving the law a public page; premature surfacing risks a public recut. |
| NS released benchmark (`released/non-sovereignty/`) | **NO CHANGE** | RELEASE_READY, Apache-2.0 scoped to its manifest. Percival artifacts are **not** in that manifest (see blockers). |

---

## HARD FENCES — the do-not-say list (load-bearing)

1. **No foundation-model or deployed-agent claims.** Everything is in-vitro toy tier:
   analytic court, binary cues, uniform bases. The results are mechanisms and
   falsifiable predictions, not transfer results.
2. **The counterproductivity claim always carries its condition** — "when the proxy is
   pointwise misaligned." Without it the claim is false (S6, P3: tilt strictly helps for
   a pointwise-good proxy). Any surface that drops the condition is wrong, not simplified.
3. **No measured-real-base assertions.** B2 characterizes *structure* (the clean prize
   requires a base with no restraint mass); it does not assert any real base's location.
   B2's spec and receipt are **PRIVATE** — do not surface, quote, or link them.
4. **The clean unconditional separation was NOT won.** It deflated structurally on
   trusted bases. What is banked on a trusted base is a *partial* separation. Surfaces
   must not read as "un-targeting beats quantilizers, full stop."
5. **Lean scope honesty.** The general theorems cover unweighted (uniform-base) finite
   lists; weighted/continuous forms are computed receipts only. The court dynamics live
   in the computed models, not the proofs. Say "machine-checked anchors," not "fully
   formalized theory."
6. **Single-knight vs multi-agent.** The separation results are single-knight; S3 shows
   the shared-regime multi-agent case is *worse* for un-targeting (coalition threshold
   ≈ 0.286), so do not extrapolate the single-agent numbers upward.
7. **"Reliability edge, not synergy"** — the S2 tax formula is specific to the
   binary-symmetric conditionally-independent model; richer joints are future work.
8. **The medievalist branch is GATED** (B5). No outreach to Grail scholars asserting the
   corpus encodes formal structure — that is manufacturing humanities validation. If
   that channel ever opens, it asks *their* question, and only with owner sign-off.
9. **Falsifiers stay visible.** The house style that makes these results credible is the
   fence-first register. A promo surface that strips the falsifiers breaks the brand.

---

## Operational blockers (must clear before any deploy)

- **Everything is uncommitted** across `sundog` and `sundogcert` (the entire reopen
  slate, the probes, `PercivalGeneral.lean`, the law doc, the umbrella edits, both
  keeper posts). Owner commit required first.
- **Licensing:** the repo root is `UNLICENSED`; the Apache-2.0 grant covers only the
  files in the NS release manifest. Percival scripts/docs/Lean are currently
  all-rights-reserved by default. Public Percival surfaces need an owner licensing
  decision (extend the manifest, a separate grant, or page-only publication).
- **Deploy conventions:** only `dist/` is published (Cloudflare Pages, `npm run
  deploy`, owner-gated). Run `npm run sundog:check` after touching any public page.
- **Owner re-voice** pending on the literary post before any deposit.

---

## Receipts index (repo-relative)

Scripts: `scripts/percival-a3-killswitch-toy.mjs`, `percival-b1-0-court-coordination.mjs`,
`percival-b1-separation-bakeoff.mjs`, `percival-b2-gamma-provenance.mjs` (private output),
`percival-b3-composition-stability.mjs`, `percival-s2-target-cap-probe.mjs`,
`percival-s3-multiagent-pivotality.mjs`, `percival-s4-counterproductivity-general.mjs`,
`percival-s6-class-boundary.mjs`.
Docs: everything under `docs/percival/`; spine `docs/SUNDOG_V_PERCIVAL.md`; umbrella
`docs/SUNDOG_V_CAUSAL_ACCESS.md`.
Lean: `sundogcert` — `Sundogcert/Percival.lean` (3-point anchors),
`Sundogcert/PercivalGeneral.lean` (general), registered in `Sundogcert/AxiomAudit.lean`;
verify with `lake build Sundogcert.AxiomAudit` (green = axiom-clean).
JSON artifacts: `results/percival/*/summary.json`.

---

*Suggested sequencing: Angle 1 note first (self-contained, strongest receipt), Angle 3
as a rider or short follow-up, Angle 2 deferred behind its falsifier, literary post on
the owner's clock. One channel at a time, per house practice.*
