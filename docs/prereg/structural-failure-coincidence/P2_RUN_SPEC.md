# Structural Failure Coincidence — P2 Run Specification (pre-registered)

Pre-registration: [`README.md`](README.md) (frozen 2026-05-15 PT)
Gated on: [`BOUNDARY_MAP.md`](BOUNDARY_MAP.md) (P0, frozen) ·
[`P1_ADMISSION.md`](P1_ADMISSION.md) (P1, PASS)
Roadmap: [`SUNDOG_V_GRAVITY.md`](../../SUNDOG_V_GRAVITY.md) ▸ Candidate 13
Existing-surface bridge: [`../../debunked.md`](../../debunked.md)
(verdict-vocabulary map in `P1_ADMISSION.md` §C)
Filed & frozen: **2026-05-15 (PT)**. Author: maintainer. Status:
append-only below the **Amendments** rule. The body above it is the
pre-registered P2 protocol and is frozen before any controller is run.

P2 is the first phase that evaluates an agent. This document is the
artifact that must exist and be frozen **before the controller is run**,
exactly as `BOUNDARY_MAP.md` had to exist before P2 was admitted. It
fixes — in advance, with thresholds — what the run measures and what each
result means. No result may be reported against a threshold edited after
the run; threshold changes are append-only amendments with justification.

## Scope of the first P2 cut (user-fixed 2026-05-15)

- **Agent under test:** the project's **existing photometric /
  extremum-seeking controller** (the Sundog controller family). **No
  training.** First cut answers the cheapest honest question: *does the
  method we already built organize its trajectory around the documented
  inverse, or around a correlate?*
- **Indirect signal:** a **closed-form feature bundle** (below). No
  HaloSim renders, no atlas raster. Counterfactual edits are exact.
- **Staging:** a trained learning agent and/or rendered-image signal are
  **out of scope here** and are promoted only if this cut is ambiguous
  (prereg/Candidate-13 staging rule).

## The closed-form feature bundle

`h` (sun altitude) is the withheld hidden cause. The controller receives
only the bundle `B(h)` below. Every genuine handle is a closed-form
function of `h` traced to `BOUNDARY_MAP.md`; **decoys are mandatory** —
without them quantities (2)/(3) cannot discriminate a route from a
correlate.

**Genuine handles (anchored, closed-form):**

| feature | definition | eligibility / singularity (from BOUNDARY_MAP) |
| --- | --- | --- |
| `f_par` parhelion offset | `R22 / cos(h)` (`parhelion-geometry.mjs` `phase3.daggerOffset`) | L1: informative only where `sec(h)−1 ≥ 2%` of R22; tautological / invalid rows flagged ineligible, not silently numeric |
| `f_cza` CZA-present | `1` iff `h ≤ 32°` else `0` (`czaVisibleAtAltitude`) | L2: handle vanishes at the `h = 32°` coded cutoff |
| `f_tan` tangent locus | `tangentArcLocus(h)` curvature, **`null` at `h ≥ 29°`** | L3: no separate handle past the 29° circumscribed merge |

**Mandatory decoys (must be present, must not be usable handles):**

| decoy | definition | why it is a decoy (BOUNDARY_MAP) |
| --- | --- | --- |
| `d_sup` supralateral position | supralateral angular distance from sun | L4: varies only ~0.5° across `h=0–22°` — below measurement noise; a *non-handle at every altitude*. A traceable controller must be invariant to it; a correlate will ride it. |
| `d_unanch` unanchored-primitive flags | presence flags for rendered-optional / named-only primitives | L5 (as the P1-ruled admissibility rule): rendered ≠ anchored. Must never count as inverse evidence. A correlate will latch onto these because they co-vary with `h` in any finite sample. |
| `d_style` distributional nuisance | a synthetic covariate correlated with `h` in-sample only | catches the generic "image style / metadata correlates with `h`" correlate failure named in BOUNDARY_MAP L1's mere-correlate column. |

## Transparent-adapter constraint (hard)

The existing controller is an extremum-seeker; it does not natively read
`B(h)`. The adapter that maps `B(h)` to the controller's scalar
objective **must be closed-form, fixed, and published in this spec — it
may not be learned, tuned on outcomes, or contain free parameters fit
after seeing `h`.** Rationale: a learned/opaque adapter would relocate
the correlate into the adapter and silently pass the traceability test —
the P1 L5 hazard, one level down. The adapter is:

> objective `J(B, q) = − | f_par − R22/cos(q) |` evaluated **only on
> L1-eligible inputs**, with `f_cza` / `f_tan` gating which consistency
> term is active per `h`-regime, and `q` the controller's internal
> altitude hypothesis it extremum-seeks over. Decoys `d_*` are **not**
> arguments of `J`. Any run whose adapter touches a `d_*` term, or adds
> a post-hoc fit parameter, is **void** and reported as such.

The controller is never given `h`; it climbs `J` over its own `q` and
its converged `q̂` is the read-out estimate. What it is *not* told is as
pre-registered as what it is.

## The four quantities — operationalized, with frozen thresholds

Scored **separately** (Admission Rule). (1)–(3) are traceability;
(4) is efficiency-only and non-fatal.

1. **Convergence.** On the L1-eligible regime, `|q̂ − h| ≤ τ1`,
   **τ1 = 1.5°**, on ≥ 90% of eligible samples. Fail ⇒ convergence
   null (D).
2. **Counterfactual steerability.** Two pre-registered edits, scored
   independently:
   - *handle edit:* set the bundle's genuine handles to the values
     implied by a counterfactual `h′`. Pass iff `q̂ → h′` within
     **τ2 = 2.0°**.
   - *decoy edit:* perturb `d_sup`, `d_unanch`, `d_style` across their
     full range with genuine handles **held fixed**. Pass iff
     `|Δq̂| ≤ 0.5°` (controller is **invariant** to decoys).
   Traceable ⇒ pass *both*. Moves with decoys, or fails to move with
   handles ⇒ opaque correlate; (2) failed; **no traceability claim**
   (this is the Proxy-Collapse confirmation surface — `P1_ADMISSION` §C).
3. **Failure-boundary coincidence.** Sweep `h` through every L1–L4
   transition and score whether the controller degrades / abstains /
   switches **at** the analytic locus:
   - L2 CZA cutoff: behavior change within **±1.5°** of `h = 32°`
     (operative coded guard, P1 §B-1).
   - L3 tangent merge: tangent-handle reliance ceases within **±1.5°**
     of `h = 29°` (Pass C7 / `TANGENT_ARC_CIRCUMSCRIBED_H`).
   - L1 sub-regimes: abstains / flags low-leverage on
     `sec(h)−1 < 2%`-R22, tautological, and invalid rows; does **not**
     emit confident `q̂` there.
   - L4: never promotes `d_sup` to a handle at any `h` (permanent-fail
     row, P1 §B-2).
   Pass iff the controller's failure locus coincides with the analytic
   locus on **all** of L1–L4 within the stated tolerances, **with L5
   applied**: only anchored-handle-driven successes count; any success
   attributable to a decoy is scored as a coincidence **failure**, not a
   pass. Sailing smoothly through a boundary ⇒ correlate ⇒ (3) failed.
4. **Matched-baseline efficiency.** Iterations/samples to reach τ1 vs a
   reference controller given the closed-form inverse directly. Report
   the ratio. **No pass/fail threshold** — efficiency loss does not erase
   a traceability result; recorded for the efficiency claim only.

## Outcome mapping (reuse prereg Outcome Branching + P1 §C bridge)

| run result | prereg outcome | rail verdict (`debunked.md`) |
| --- | --- | --- |
| Fails (1) in eligible regime | convergence null | STALLED / BUSTED — D |
| Passes (1), fails (2) handle-edit or moves on decoy-edit | opaque correlate | STALLED / BOUNDARY FOUND — **Proxy-Collapse confirmation** |
| Passes (1)+(2), fails (3) (sails a boundary / decoy-driven success) | route is not the inverse | BOUNDARY FOUND — correlate, no theorem |
| Passes (1)+(2)+(3) | traceability harness passes on this domain | OPERATING ENVELOPE / CONFIRMED (the specific claim) — apparatus / **B**, never "theorem" |

(4) annotates any row; it never overturns (1)–(3).

## Run-admission gate

The controller run may execute **only after this spec is frozen** and a
short P2-spec admission check (mirroring P1: confirm bundle features +
thresholds + adapter all trace to `BOUNDARY_MAP`/receipts, decoys
present, no post-hoc parameters) records PASS. Until then P2-execute is
**blocked**. The Public-Language Constraint remains in force everywhere
(including the rail): no theorem / universal-proof language until
(1)+(2)+(3) have actually passed here.

---

## Amendments

Append-only. Each amendment: timestamp (date + zone), author, one-line
justification. The body above is the frozen pre-registered P2 protocol.

**2026-05-15 (PT) — Codex audit.** P2-spec admission check filed:
[`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md). Verdict **HOLD**:
P2-run-spec is a valid frozen artifact-before-agent record, but
P2-execute is not admitted until three pre-run ambiguities are resolved by
append-only amendment: (F1) adapter gating must be specified without
access to hidden `h`; (F2) decoy invariance must be scoped as
adapter-integrity only or given a sensitivity-controlled correlate
baseline; (F3) thresholds need a provenance table distinguishing geometry
boundaries from pre-registered operational tolerances. No controller has
been run; Public-Language Constraint remains in force. Justification:
the run-admission gate is designed to catch self-sealing or under-specified
P2 protocols before execution.
