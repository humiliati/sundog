# PDE C1 Finite-Galerkin Separation — External Review Packet

> Minimal packet for an external sanity check by a PDE / data-assimilation
> analyst. This is **not** a public page, **not** a Navier–Stokes
> existence/smoothness claim, and **not** a determining-modes theorem. It
> exists to make it easy for a reviewer to say "yes, this separation is
> honestly framed," "no, this language overstates X," or "you missed standard
> issue Y." Companion to the outreach text in
> [`PDE_C1_EXTERNAL_REVIEW_EMAIL_DRAFT.md`](PDE_C1_EXTERNAL_REVIEW_EMAIL_DRAFT.md).

**Date:** 2026-06-02
**Status:** draft reviewer packet
**Primary artifact:** [`PDE_C1_SEPARATION_STATEMENT.md`](PDE_C1_SEPARATION_STATEMENT.md)
**Current result:** [`PDE_C1_REGIME_GENERALITY_v1.md`](PDE_C1_REGIME_GENERALITY_v1.md) §12 — `PDE-C1-RG-POS`

## Reviewer Snapshot

Repository map:

- Primary working packet: `docs/proof/` (the `PDE_C1_*` family).
- Reviewer-facing statement: [`PDE_C1_SEPARATION_STATEMENT.md`](PDE_C1_SEPARATION_STATEMENT.md).
- Main ledger: [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md).
- Reading note this consolidates: [`PDE_DETERMINING_MODES_POSTULATE1.md`](PDE_DETERMINING_MODES_POSTULATE1.md).
- Current conclusion: [`PDE_C1_REGIME_GENERALITY_v1.md`](PDE_C1_REGIME_GENERALITY_v1.md).

Why we looked at finite-Galerkin Navier–Stokes:

> A 2D Kolmogorov-flow Galerkin truncation was used as a concrete substrate to
> test whether a compact observation can be **control-sufficient** for a
> registered safety decision **without being state-reconstructive** — i.e. a
> signature below the determining-modes count that still decides the action.
> A positive separation witness was obtained. The review question is whether
> that separation is framed honestly, or is ordinary observer / data-
> assimilation behavior that should not be made load-bearing.

Current status:

- The separation is **certified as a finite-dimensional witness** on the
  sampled support, at two Grashof values (`G = 200`, `G = 300`), `k_f = 2`,
  `K = 3`, under a regime-portable objective (`PDE-C1-RG-POS`).
- No Navier–Stokes existence/smoothness, no infinite-dimensional attractor,
  no new determining-modes bound, and no Clay-problem claim is live.
- **C1 remains PROVISIONAL / UNPROMOTED.** Proxy faithfulness and external PDE
  review are the open gates; the public surface is blocked until an external
  sanity check confirms or corrects the framing.

Core mapping:

| Sundog concept | C1 instantiation | Current read |
| --- | --- | --- |
| Hidden state | Galerkin state of `S_N` (`d_state = 440` real DOF) | only ~4% is signature-resolved |
| Admitted trace / signature `Φ_K` | low-Fourier band `K = 3`, `d = 18` real components | **certified non-injective** on `supp μ` (twin-state) |
| Decision from the trace | registered binary action `{no_op, damp_low_band}` via held-out look-ahead-max quantile objective `J_q` | **`Φ_K`-measurable a.e.** up to a measure-`δ ≈ 0.037` boundary layer (kNN POSITIVE + paired exhibit) |
| Failure boundary | the `K = 3 < K*` gap; two-point Grashof axis; `k_f` fixed | named explicitly; generality beyond is the open robustness sub-goal |

Ways to falsify or weaken this packet:

- Show that "state-insufficient on the sampled support" is too strong for a
  sampled-SRB / finite-Galerkin twin-state certificate, and should be weakened
  to a finite-sample statement.
- Show that "control-sufficient on `Φ_K`-fibers up to a measure-zero decision
  surface" is the natural reading of the kNN/disintegration evidence overstated
  — e.g. that the `mean_minority → 0` extrapolation does not support a.e.
  `Φ_K`-measurability.
- Show that the held-out look-ahead-max quantile objective `J_q` is implicitly
  a function of `Φ_K` after all (anti-tautology in §3 of the statement fails),
  collapsing the separation to a definitional artifact.
- Identify this as ordinary data-assimilation / observer / LES-closure behavior
  ("coarse predicts coarse because coarse slaves fine") that the non-injectivity
  certificate does **not** actually foreclose.
- Recommend narrowing "finite-Galerkin separation" to "two-point, single-
  objective-family numerical observation."

## One-Sentence Ask

Please sanity-check whether our separation conclusion is correctly framed: on
the sampled invariant measure of a 2D Kolmogorov-flow Galerkin model
(`k_f = 2`, `K = 3`, `G ∈ {200, 300}`), the low-band signature `Φ_K` is
**state-insufficient** (certified non-injective) yet **control-sufficient** for
a registered decision (`Φ_K`-measurable up to a measure-`δ` boundary layer),
and that this is a genuine finite-Galerkin separation rather than an artifact of
objective construction, sampling, or ordinary observer behavior.

## What We Are Not Asking

- Not asking for a review of Navier–Stokes existence/smoothness.
- Not asking whether this bears on the Clay Millennium problem.
- Not asking for a new determining-modes theorem or bound.
- Not asking for endorsement of Sundog, the broader apparatus, or any public
  presentation.
- Not asking the reviewer to debug the entire repository.

## What We Are Asking

Please check the following four calls (these are the locked reviewer questions
from the separation statement, in plainest form):

1. **State-insufficiency language.** Given the twin-state certificate is
   sampled-SRB / finite-Galerkin (not an exact attractor theorem), is
   **"`Φ_K` is non-injective / state-insufficient on `supp μ`"** reasonable, or
   should it be weakened to a finite-sample claim?
2. **Control-sufficiency language.** Is the kNN/disintegration reading of
   **"`pi*` factors through `Φ_K` up to `μ`-measure zero"** (local action-mixing
   `mean_minority → 0` as signature tolerance → 0, plus paired action-constancy
   on the certified non-injective pairs) mathematically fair, or overstated?
3. **Objective legitimacy.** Is a **held-out look-ahead-max quantile** a
   defensible way to define a regime-portable proxy action — given that the
   older burn-in-percentile trigger went vacuous at `G = 300`
   (`damp_fraction ≈ 0.004`) and was replaced, not retuned?
4. **Real separation vs renaming.** Would you call this a genuine
   *"a sub-determining mode set can be control-sufficient without being
   state-reconstructive"* separation, or ordinary functional-observability /
   data-assimilation / LES-closure behavior that should not be load-bearing?

A short reply is enough. The most useful possible answers are:

```text
The framing seems conservative / basically right.
Weaken or rename X (e.g. "state-insufficient" → finite-sample only).
The control-sufficiency / fiber language is too strong because Z.
This is standard observer / data-assimilation, cite B.
The objective is implicitly a function of Φ_K; the separation is vacuous because W.
```

## Reviewer Time Budget

Suggested review paths:

- **10 minutes:** read [`PDE_C1_SEPARATION_STATEMENT.md`](PDE_C1_SEPARATION_STATEMENT.md)
  §1 (the statement), §3 (anti-tautology), and §7 (the pivotal claim + anti-folklore guard).
- **25 minutes:** additionally inspect [`PDE_C1_REGIME_GENERALITY_v1.md`](PDE_C1_REGIME_GENERALITY_v1.md)
  §3 (portable objective), §6 (portability gate), and §12 (the result table).
- **60 minutes:** additionally skim the branch/verdict summaries in
  [`PDE_C1_KNN_CONVERGENCE_CHECK.md`](PDE_C1_KNN_CONVERGENCE_CHECK.md) and
  [`PDE_C1_TWIN_STATE_CERTIFICATE.md`](PDE_C1_TWIN_STATE_CERTIFICATE.md), and the
  prior-art delta table in the separation statement §7.

## Core Claim To Audit

From the separation statement (§7), the pivotal claim a reviewer must bless or
reject — everything else is support:

> On the sampled invariant measure of this Galerkin 2D-NSE system, the low-band
> projection `Φ_K` is non-injective (state-unobservable) yet the registered
> decision `a(x) = 1[ max_{t≤τ} E_low(flow_t x) > E_max ]` is `Φ_K`-measurable
> up to a measure-`δ` boundary layer (decision-observable), with `K` below the
> determining-modes count.

This sentence is the thing under review. If it is too strong, the packet should
be revised before any public surface exists.

**Anti-folklore guard (worth pre-empting).** The natural reviewer reflex —
"this is just LES / closure: of course coarse energy is predictable from coarse
state" — is exactly what the non-injectivity certificate is meant to foreclose:
LES / AIM / closure assume or derive **state-sufficiency** via slaving; C1
*certifies state-insufficiency* (twin states) and shows the **decision** survives
the genuinely unresolved fine state. We want this distinction specifically
stress-tested.

## Result Summary (`PDE-C1-RG-POS`, 2026-05-29)

| Run | Portability gate (adj damp ∈ [0.20, 0.40]) | Verdict |
| --- | --- | --- |
| `lock_v7_g200` kNN (positive control) | 0.3003 ✓ | `STRICTNESS_WITNESS_POSITIVE`, `a_mm = −0.00079`, slope 0.736 |
| `lock_v7_g300` kNN (generality test) | 0.2688 ✓ | `STRICTNESS_WITNESS_POSITIVE`, `a_mm = +0.00058`, slope 0.564 |
| `lock_v7_g300` twin-state (support companion) | — | `TWIN_STATE_CERTIFIED` (100% witness coverage, 942,834 unique pairs, `δ_H = 0.0111`, median ‖Q_K‖ = 0.222) |

Supporting: the resolved/unresolved gap is 18 of 440 real DOF (~4%); the
unresolved→resolved energy coupling is ~99% signature-determined
(`R²(R|Φ_K) = 0.998` at `G=200` / `0.990` at `G=300`, neg-control ~0); the
control half is robust to halving the signature dimension (`d_K = 32` vs `18`),
to sample budget (50k vs 200k), and to the objective construction (burn-in-
percentile and held-out-quantile both POSITIVE at `G=200`). Deterministic on
seed `20260528`.

## Files To Read

Primary:

- [`PDE_C1_SEPARATION_STATEMENT.md`](PDE_C1_SEPARATION_STATEMENT.md) — the
  self-contained reviewer-facing separation (statement, anti-tautology,
  anti-vacuity ledger, determining-modes comparator, prior-art delta).
- [`PDE_C1_REGIME_GENERALITY_v1.md`](PDE_C1_REGIME_GENERALITY_v1.md) — the
  two-regime portable objective and the `PDE-C1-RG-POS` result.

Secondary (the two adjudicator halves and the exhibit that composes them):

- [`PDE_C1_TWIN_STATE_CERTIFICATE.md`](PDE_C1_TWIN_STATE_CERTIFICATE.md) —
  state-insufficiency half.
- [`PDE_C1_KNN_CONVERGENCE_CHECK.md`](PDE_C1_KNN_CONVERGENCE_CHECK.md) —
  control-sufficiency adjudicator.
- [`PDE_C1_PAIRED_FIBER_CONSTANCY.md`](PDE_C1_PAIRED_FIBER_CONSTANCY.md) — the
  paired exhibit composing both on the same certified pairs.
- [`PDE_C1_MZ_ENERGY_BUDGET.md`](PDE_C1_MZ_ENERGY_BUDGET.md) — the measured
  energy-budget closure mechanism (`R ≈ R(Φ_K)`).

Context (not load-bearing on the separation):

- [`PDE_DETERMINING_MODES_POSTULATE1.md`](PDE_DETERMINING_MODES_POSTULATE1.md) —
  the Reading-2 regime taxonomy this instantiates.
- [`PDE_C1_REGIME_GENERALITY_v0.md`](PDE_C1_REGIME_GENERALITY_v0.md) — the
  `G=300` fixed-percentile probe that deferred for objective vacuity (why the
  objective was replaced).
- [`PDE_C1_MECHANISM_RECON.md`](PDE_C1_MECHANISM_RECON.md) — observer-theory
  restatement and prior-art positioning.

Runnable / result directories (under `results/proof/`, not in git; include
separately if sending a ZIP):

- `results/proof/c1-rg-v1-g200-knn-sweep/`, `results/proof/c1-rg-v1-g300-knn-sweep/`,
  `results/proof/c1-rg-v1-g300-twin-state/` — the three verdict-bearing run dirs.

## If The Reviewer Has Only One Comment

Ask them to answer this:

> Is the certified non-injectivity of `Φ_K` enough to distinguish this from
> ordinary LES / approximate-inertial-manifold / closure behavior (which assume
> or derive state-sufficiency), so that "the decision is `Φ_K`-measurable up to a
> measure-`δ` boundary layer while the state is not `Φ_K`-recoverable" is a real
> finite-Galerkin separation — or is there a standard reading under which this is
> expected observer behavior and the language should be downgraded? If there is
> such a reading, please name it.

## Output We Want From Review

Any of:

- "OK as a finite-Galerkin separation witness; keep public claims blocked or
  cautious."
- "The control-sufficiency / fiber language is fine, but 'state-insufficient'
  should be a finite-sample statement because X."
- "This is standard functional-observability / determining-functionals
  language; cite B and rename."
- "The objective is implicitly `Φ_K`-measurable; the separation is vacuous
  because W."
- "The separation is real but the prior-art delta vs Montanari–Motter /
  Cockburn–Jones–Titi is overstated; phrase it as X."

## Packet Hygiene

Do not send a polished public page first. Send this packet, the separation
statement, or a PDF rendering. A future `navierstokes.html` / generality-gallery
card may be made later only if it carries the same load-bearing separation
statement, the full scope clauses (two-point Grashof axis, `k_f` fixed, single
objective family, finite-Galerkin, sampled-support, numerical), and a clear
"external review pending" / "external review returned: <verdict>" banner.

Public-language boundary (binding on every output of this review process):

- **Allowed:** "Sundog is drafting a finite-Galerkin separation: a low-band
  signature that is state-insufficient yet control-sufficient for a registered
  decision, below the determining-modes count, on a sampled 2D-NSE Galerkin
  attractor."
- **Forbidden:** "Sundog has a Navier–Stokes result." "Sundog solved / made
  progress on the Clay problem." "Sundog proved a determining-modes bound."
  "Finite-Galerkin separation implies the infinite-dimensional NSE theorem."

A certified finite-Galerkin witness is **not** a Navier–Stokes result; it is a
provisional, unpromoted separation inside the registered envelope. The forbidden
phrasing remains forbidden even after a positive review.

The packet does **not** include code-level audit of the integrator or
adjudicators beyond what is needed to verify the pre-registered gates and the
deterministic re-run; code-level review is a separate scope.
